import unittest

from grounded_chart import (
    AxisRequirementSpec,
    FigureTrace,
    FigureRequirementSpec,
    GroundedChartPipeline,
    HeuristicIntentParser,
    PlotTrace,
    DataPoint,
    TableSchema,
)


class PipelineEvidenceTest(unittest.TestCase):
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
