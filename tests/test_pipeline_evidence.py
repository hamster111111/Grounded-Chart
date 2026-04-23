import unittest

from grounded_chart import (
    ArtistTrace,
    AxisRequirementSpec,
    FigureTrace,
    FigureRequirementSpec,
    GroundedChartPipeline,
    HeuristicIntentParser,
    PlotTrace,
    DataPoint,
    MeasureSpec,
    ParsedRequirementBundle,
    TableSchema,
)
from grounded_chart.requirements import ChartRequirementPlan, PanelRequirementPlan, RequirementNode
from grounded_chart.schema import AxisTrace, ChartIntentPlan


class StaticParsedBundleParser:
    def __init__(self, bundle: ParsedRequirementBundle) -> None:
        self.bundle = bundle

    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        return self.bundle.plan

    def parse_requirements(self, query: str, schema: TableSchema) -> ParsedRequirementBundle:
        return self.bundle


class PipelineEvidenceTest(unittest.TestCase):
    def test_pipeline_skips_generic_axis_labels_from_parser_native_requirements(self):
        query = 'Add a main title to the plot, such as "A colored bubble plot". Label the x-axis as "the X axis". Label the y-axis as "the Y axis".'
        requirements = (
            RequirementNode(
                requirement_id="shared.title_0",
                scope="figure",
                type="annotation",
                name="title",
                value="A colored bubble plot",
                source_span='Add a main title to the plot, such as "A colored bubble plot".',
            ),
            RequirementNode(
                requirement_id="panel_0.title_0",
                scope="panel",
                type="annotation",
                name="title",
                value="A colored bubble plot",
                source_span="A colored bubble plot",
                panel_id="panel_0",
            ),
            RequirementNode(
                requirement_id="panel_0.axis_label_0",
                scope="panel",
                type="annotation",
                name="axis_label",
                value="the X axis",
                source_span='Label the x-axis as "the X axis".',
                panel_id="panel_0",
            ),
            RequirementNode(
                requirement_id="panel_0.axis_label_1",
                scope="panel",
                type="annotation",
                name="axis_label",
                value="the Y axis",
                source_span='Label the y-axis as "the Y axis".',
                panel_id="panel_0",
            ),
        )
        bundle = ParsedRequirementBundle(
            plan=ChartIntentPlan(
                chart_type="unknown",
                dimensions=(),
                measure=MeasureSpec(column=None, agg="none"),
                raw_query=query,
                confidence=0.9,
            ),
            requirement_plan=ChartRequirementPlan(
                requirements=requirements,
                panels=(
                    PanelRequirementPlan(
                        panel_id="panel_0",
                        chart_type="unknown",
                        requirement_ids=tuple(
                            requirement.requirement_id
                            for requirement in requirements
                            if requirement.scope == "panel"
                        ),
                    ),
                ),
                figure_requirements={"title": "A colored bubble plot"},
                raw_query=query,
            ),
        )

        pipeline = GroundedChartPipeline(parser=StaticParsedBundleParser(bundle))
        result = pipeline.run(
            query=query,
            schema=TableSchema(columns={}),
            rows=(),
            actual_trace=PlotTrace(chart_type="unknown", points=(), source="actual"),
            actual_figure=FigureTrace(
                title="",
                axes=(
                    AxisTrace(
                        index=0,
                        title="A colored bubble plot",
                        xlabel="Anything",
                        ylabel="Else",
                    ),
                ),
                source="actual_figure",
            ),
            verify_data=False,
        )

        self.assertTrue(result.report.ok)
        self.assertEqual("A colored bubble plot", result.expected_figure.axes[0].title)
        self.assertIsNone(result.expected_figure.axes[0].xlabel)
        self.assertIsNone(result.expected_figure.axes[0].ylabel)

    def test_pipeline_normalizes_wrapped_quotes_in_subplot_titles(self):
        query = "Title this subplot 'semilogy'."
        requirements = (
            RequirementNode(
                requirement_id="panel_0.title_0",
                scope="panel",
                type="annotation",
                name="title",
                value="'semilogy'",
                source_span="Title this subplot 'semilogy'",
                panel_id="panel_0",
            ),
        )
        bundle = ParsedRequirementBundle(
            plan=ChartIntentPlan(
                chart_type="unknown",
                dimensions=(),
                measure=MeasureSpec(column=None, agg="none"),
                raw_query=query,
                confidence=0.9,
            ),
            requirement_plan=ChartRequirementPlan(
                requirements=requirements,
                panels=(
                    PanelRequirementPlan(
                        panel_id="panel_0",
                        chart_type="unknown",
                        requirement_ids=("panel_0.title_0",),
                    ),
                ),
                raw_query=query,
            ),
        )

        pipeline = GroundedChartPipeline(parser=StaticParsedBundleParser(bundle))
        result = pipeline.run(
            query=query,
            schema=TableSchema(columns={}),
            rows=(),
            actual_trace=PlotTrace(chart_type="unknown", points=(), source="actual"),
            actual_figure=FigureTrace(
                title="",
                axes=(AxisTrace(index=0, title="semilogy"),),
                source="actual_figure",
            ),
            verify_data=False,
        )

        self.assertTrue(result.report.ok)
        self.assertEqual("semilogy", result.expected_figure.axes[0].title)

    def test_pipeline_derives_axes_count_from_subplot_layout_phrase(self):
        query = "Create a side-by-side figure."
        requirements = (
            RequirementNode(
                requirement_id="shared.subplot_layout_0",
                scope="figure",
                type="figure_composition",
                name="subplot_layout",
                value="side-by-side",
                source_span="side-by-side",
            ),
        )
        bundle = ParsedRequirementBundle(
            plan=ChartIntentPlan(
                chart_type="unknown",
                dimensions=(),
                measure=MeasureSpec(column=None, agg="none"),
                raw_query=query,
                confidence=0.9,
            ),
            requirement_plan=ChartRequirementPlan(
                requirements=requirements,
                panels=(),
                figure_requirements={"subplot_layout": "side-by-side"},
                raw_query=query,
            ),
        )

        pipeline = GroundedChartPipeline(parser=StaticParsedBundleParser(bundle))
        result = pipeline.run(
            query=query,
            schema=TableSchema(columns={}),
            rows=(),
            actual_trace=PlotTrace(chart_type="unknown", points=(), source="actual"),
            actual_figure=FigureTrace(title="", axes=(), source="actual_figure"),
            verify_data=False,
        )

        self.assertFalse(result.report.ok)
        self.assertEqual(2, result.expected_figure.axes_count)
        self.assertIn("wrong_axes_count", result.report.error_codes)
        self.assertIn("shared.subplot_layout_0", result.evidence_graph.failed_requirement_ids)

    def test_pipeline_derives_expected_figure_from_parser_native_requirements(self):
        query = "Create a bar chart titled Revenue Dashboard with Category and Sales axes."
        requirements = (
            RequirementNode(
                requirement_id="shared.subplot_count_0",
                scope="figure",
                type="figure_composition",
                name="subplot_count",
                value=1,
                source_span="one subplot",
            ),
            RequirementNode(
                requirement_id="shared.title_0",
                scope="figure",
                type="annotation",
                name="title",
                value="Revenue Dashboard",
                source_span="Revenue Dashboard",
            ),
            RequirementNode(
                requirement_id="panel_0.title_0",
                scope="panel",
                type="annotation",
                name="title",
                value="Sales by Category",
                source_span="Sales by Category",
                panel_id="panel_0",
            ),
            RequirementNode(
                requirement_id="panel_0.axis_label_0",
                scope="panel",
                type="annotation",
                name="axis_label",
                value="Category",
                source_span="x-axis label Category",
                panel_id="panel_0",
            ),
            RequirementNode(
                requirement_id="panel_0.axis_label_1",
                scope="panel",
                type="annotation",
                name="axis_label",
                value="Sales",
                source_span="y-axis label Sales",
                panel_id="panel_0",
            ),
            RequirementNode(
                requirement_id="panel_0.artist_type_0",
                scope="panel",
                type="encoding",
                name="artist_type",
                value="bar",
                source_span="bar chart",
                panel_id="panel_0",
            ),
        )
        bundle = ParsedRequirementBundle(
            plan=ChartIntentPlan(
                chart_type="unknown",
                dimensions=(),
                measure=MeasureSpec(column=None, agg="none"),
                raw_query=query,
                confidence=0.9,
            ),
            requirement_plan=ChartRequirementPlan(
                requirements=requirements,
                panels=(
                    PanelRequirementPlan(
                        panel_id="panel_0",
                        chart_type="unknown",
                        requirement_ids=tuple(
                            requirement.requirement_id
                            for requirement in requirements
                            if requirement.scope == "panel"
                        ),
                    ),
                ),
                figure_requirements={"subplot_count": 1, "title": "Revenue Dashboard"},
                raw_query=query,
            ),
        )

        pipeline = GroundedChartPipeline(parser=StaticParsedBundleParser(bundle))
        result = pipeline.run(
            query=query,
            schema=TableSchema(columns={}),
            rows=(),
            actual_trace=PlotTrace(chart_type="unknown", points=(), source="actual"),
            actual_figure=FigureTrace(
                title="Revenue Dashboard",
                axes=(
                    AxisTrace(
                        index=0,
                        title="Sales by Category",
                        xlabel="Category",
                        ylabel="Sales",
                        artists=(ArtistTrace(artist_type="bar", count=1),),
                    ),
                ),
                source="actual_figure",
            ),
            verify_data=False,
        )

        self.assertTrue(result.report.ok)
        self.assertIsNotNone(result.expected_figure)
        self.assertEqual(1, result.expected_figure.axes_count)
        self.assertEqual("Revenue Dashboard", result.expected_figure.figure_title)
        self.assertEqual("Sales by Category", result.expected_figure.axes[0].title)
        self.assertEqual("Category", result.expected_figure.axes[0].xlabel)
        self.assertEqual("Sales", result.expected_figure.axes[0].ylabel)
        self.assertEqual(("bar",), result.expected_figure.axes[0].artist_types)
        self.assertEqual(("shared.subplot_count_0",), result.expected_figure.provenance["axes_count"])
        self.assertEqual(("panel_0.axis_label_0",), result.expected_figure.axes[0].provenance["xlabel"])

    def test_pipeline_preserves_sparse_one_based_panel_alignment(self):
        query = "Create three stacked subplots. In the second subplot show a 2D histogram. In the third subplot show another 2D histogram."
        requirements = (
            RequirementNode(
                requirement_id="shared.subplot_count_0",
                scope="figure",
                type="figure_composition",
                name="subplot_count",
                value=3,
                source_span="three stacked subplots",
            ),
            RequirementNode(
                requirement_id="panel_2.artist_type_0",
                scope="panel",
                type="encoding",
                name="artist_type",
                value="hist2d",
                source_span="2D histogram",
                panel_id="panel_2",
            ),
            RequirementNode(
                requirement_id="panel_3.artist_type_1",
                scope="panel",
                type="encoding",
                name="artist_type",
                value="hist2d",
                source_span="2D histogram",
                panel_id="panel_3",
            ),
        )
        bundle = ParsedRequirementBundle(
            plan=ChartIntentPlan(
                chart_type="unknown",
                dimensions=(),
                measure=MeasureSpec(column=None, agg="none"),
                raw_query=query,
                confidence=0.9,
            ),
            requirement_plan=ChartRequirementPlan(
                requirements=requirements,
                panels=(
                    PanelRequirementPlan(panel_id="panel_1", chart_type="unknown"),
                    PanelRequirementPlan(panel_id="panel_2", chart_type="unknown", requirement_ids=("panel_2.artist_type_0",)),
                    PanelRequirementPlan(panel_id="panel_3", chart_type="unknown", requirement_ids=("panel_3.artist_type_1",)),
                ),
                figure_requirements={"subplot_count": 3},
                raw_query=query,
            ),
        )

        pipeline = GroundedChartPipeline(parser=StaticParsedBundleParser(bundle))
        result = pipeline.run(
            query=query,
            schema=TableSchema(columns={}),
            rows=(),
            actual_trace=PlotTrace(chart_type="unknown", points=(), source="actual"),
            actual_figure=FigureTrace(
                title="",
                axes=(
                    AxisTrace(index=0, title="Top Panel", artists=(ArtistTrace(artist_type="line", count=1),)),
                    AxisTrace(index=1, title="Middle Histogram", artists=(ArtistTrace(artist_type="image", count=1),)),
                    AxisTrace(index=2, title="Bottom Histogram", artists=(ArtistTrace(artist_type="image", count=1),)),
                ),
                source="actual_figure",
            ),
            verify_data=False,
        )

        self.assertTrue(result.report.ok)
        self.assertEqual(3, result.expected_figure.axes_count)
        self.assertEqual((1, 2), tuple(axis.axis_index for axis in result.expected_figure.axes))

    def test_pipeline_evidence_maps_parser_native_figure_failures(self):
        query = "Create two subplots titled Revenue Dashboard with bar marks."
        requirements = (
            RequirementNode(
                requirement_id="shared.subplot_count_0",
                scope="figure",
                type="figure_composition",
                name="subplot_count",
                value=2,
                source_span="two subplots",
            ),
            RequirementNode(
                requirement_id="shared.title_0",
                scope="figure",
                type="annotation",
                name="title",
                value="Revenue Dashboard",
                source_span="Revenue Dashboard",
            ),
            RequirementNode(
                requirement_id="panel_0.title_0",
                scope="panel",
                type="annotation",
                name="title",
                value="Sales by Category",
                source_span="Sales by Category",
                panel_id="panel_0",
            ),
            RequirementNode(
                requirement_id="panel_0.axis_label_0",
                scope="panel",
                type="annotation",
                name="axis_label",
                value="Category",
                source_span="x-axis label Category",
                panel_id="panel_0",
            ),
            RequirementNode(
                requirement_id="panel_0.axis_label_1",
                scope="panel",
                type="annotation",
                name="axis_label",
                value="Sales",
                source_span="y-axis label Sales",
                panel_id="panel_0",
            ),
            RequirementNode(
                requirement_id="panel_0.artist_type_0",
                scope="panel",
                type="encoding",
                name="artist_type",
                value="bar",
                source_span="bar chart",
                panel_id="panel_0",
            ),
        )
        bundle = ParsedRequirementBundle(
            plan=ChartIntentPlan(
                chart_type="unknown",
                dimensions=(),
                measure=MeasureSpec(column=None, agg="none"),
                raw_query=query,
                confidence=0.9,
            ),
            requirement_plan=ChartRequirementPlan(
                requirements=requirements,
                panels=(
                    PanelRequirementPlan(
                        panel_id="panel_0",
                        chart_type="unknown",
                        requirement_ids=tuple(
                            requirement.requirement_id
                            for requirement in requirements
                            if requirement.scope == "panel"
                        ),
                    ),
                ),
                figure_requirements={"subplot_count": 2, "title": "Revenue Dashboard"},
                raw_query=query,
            ),
        )

        pipeline = GroundedChartPipeline(parser=StaticParsedBundleParser(bundle))
        result = pipeline.run(
            query=query,
            schema=TableSchema(columns={}),
            rows=(),
            actual_trace=PlotTrace(chart_type="unknown", points=(), source="actual"),
            actual_figure=FigureTrace(
                title="Wrong Dashboard",
                axes=(
                    AxisTrace(
                        index=0,
                        title="Wrong Title",
                        xlabel="Wrong X",
                        ylabel="Wrong Y",
                        artists=(ArtistTrace(artist_type="line", count=1),),
                    ),
                ),
                source="actual_figure",
            ),
            verify_data=False,
        )

        self.assertFalse(result.report.ok)
        self.assertIn("wrong_axes_count", result.report.error_codes)
        self.assertIn("wrong_figure_title", result.report.error_codes)
        self.assertIn("wrong_axis_title", result.report.error_codes)
        self.assertIn("wrong_x_label", result.report.error_codes)
        self.assertIn("wrong_y_label", result.report.error_codes)
        self.assertIn("missing_artist_type", result.report.error_codes)
        self.assertIn("shared.subplot_count_0", result.evidence_graph.failed_requirement_ids)
        self.assertIn("shared.title_0", result.evidence_graph.failed_requirement_ids)
        self.assertIn("panel_0.title_0", result.evidence_graph.failed_requirement_ids)
        self.assertIn("panel_0.axis_label_0", result.evidence_graph.failed_requirement_ids)
        self.assertIn("panel_0.axis_label_1", result.evidence_graph.failed_requirement_ids)
        self.assertIn("panel_0.artist_type_0", result.evidence_graph.failed_requirement_ids)

    def test_pipeline_builds_requirement_plan_and_evidence_graph(self):
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser())
        result = pipeline.run(
            query="Show total sales by category in a bar chart, ascending.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 4},
                {"category": "A", "sales": 6},
                {"category": "B", "sales": 5},
            ),
            actual_trace=PlotTrace(
                chart_type="bar",
                points=(DataPoint("B", 5), DataPoint("A", 10)),
                source="actual",
            ),
            expected_figure=FigureRequirementSpec(
                axes_count=1,
                axes=(AxisRequirementSpec(title="Sales by Category", xlabel="Category"),),
            ),
        )

        self.assertIsNotNone(result.requirement_plan)
        self.assertIsNotNone(result.evidence_graph)
        requirement_ids = {requirement.requirement_id for requirement in result.requirement_plan.requirements}
        self.assertIn("panel_0.chart_type", requirement_ids)
        self.assertIn("panel_0.aggregation", requirement_ids)
        self.assertIn("figure.axes_count", requirement_ids)
        self.assertIn("panel_0.axis_0.title", requirement_ids)
        requirements = {requirement.requirement_id: requirement for requirement in result.requirement_plan.requirements}
        self.assertEqual(requirements["panel_0.chart_type"].source_span, "bar chart")
        self.assertEqual(requirements["panel_0.aggregation"].source_span, "total")
        self.assertEqual(requirements["figure.axes_count"].source_span, "1")
        self.assertTrue(any(link.requirement_id == "panel_0.chart_type" for link in result.evidence_graph.links))
        self.assertIn("expected.plot_trace", {artifact.artifact_id for artifact in result.evidence_graph.expected_artifacts})
        self.assertIn("actual.plot_trace", {artifact.artifact_id for artifact in result.evidence_graph.actual_artifacts})

    def test_pipeline_evidence_marks_failed_figure_requirement(self):
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser())
        result = pipeline.run(
            query="Show total sales by category in a bar chart.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "B", "sales": 5},
            ),
            actual_trace=PlotTrace(
                chart_type="bar",
                points=(DataPoint("A", 10), DataPoint("B", 5)),
                source="actual",
            ),
            expected_figure=FigureRequirementSpec(axes_count=2),
            actual_figure=FigureTrace(axes=(), source="actual_figure"),
        )

        self.assertIn("wrong_axes_count", result.report.error_codes)
        self.assertIn("figure.axes_count", result.evidence_graph.failed_requirement_ids)

    def test_pipeline_only_enforces_order_for_explicit_sort(self):
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser())
        unsorted_result = pipeline.run(
            query="Show total sales by category in a bar chart.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "B", "sales": 5},
            ),
            actual_trace=PlotTrace(
                chart_type="bar",
                points=(DataPoint("B", 5), DataPoint("A", 10)),
                source="actual",
            ),
        )
        sorted_result = pipeline.run(
            query="Show total sales by category in a bar chart, ascending.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "B", "sales": 5},
            ),
            actual_trace=PlotTrace(
                chart_type="bar",
                points=(DataPoint("A", 10), DataPoint("B", 5)),
                source="actual",
            ),
        )

        self.assertTrue(unsorted_result.report.ok)
        self.assertFalse(sorted_result.report.ok)
        self.assertIn("wrong_order", sorted_result.report.error_codes)


if __name__ == "__main__":
    unittest.main()
