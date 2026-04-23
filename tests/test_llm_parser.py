import unittest

from grounded_chart import LLMIntentParser, TableSchema, derive_expected_figure


class StubLLMClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_text(self, **kwargs):
        raise AssertionError("complete_text should not be called in this test")

    def complete_json(self, **kwargs):
        return dict(self.payload)


class LLMIntentParserTest(unittest.TestCase):
    def test_llm_intent_parser_maps_json_payload_to_plan(self):
        parser = LLMIntentParser(
            StubLLMClient(
                {
                    "chart_type": "line",
                    "dimensions": ["month"],
                    "measure_column": "sales",
                    "aggregation": "sum",
                    "requirements": [
                        {
                            "name": "chart_type",
                            "value": "line",
                            "source_span": "line chart",
                            "status": "explicit",
                            "confidence": 0.95,
                        },
                        {
                            "name": "measure_column",
                            "value": "sales",
                            "source_span": "sales",
                            "status": "explicit",
                            "confidence": 0.9,
                        },
                    ],
                    "sort": {"by": "measure", "direction": "desc"},
                    "limit": 5,
                    "confidence": 0.92,
                }
            )
        )

        bundle = parser.parse_requirements(
            "Plot total sales by month as a line chart, descending, top 5.",
            TableSchema(columns={"month": "str", "sales": "number"}),
        )
        plan = bundle.plan

        self.assertEqual("line", plan.chart_type)
        self.assertEqual(("month",), plan.dimensions)
        self.assertEqual("sales", plan.measure.column)
        self.assertEqual("sum", plan.measure.agg)
        self.assertEqual("measure", plan.sort.by)
        self.assertEqual("desc", plan.sort.direction)
        self.assertEqual(5, plan.limit)
        self.assertEqual(0.92, plan.confidence)
        requirements = {requirement.requirement_id: requirement for requirement in bundle.requirement_plan.requirements}
        self.assertEqual("line chart", requirements["panel_0.chart_type"].source_span)
        self.assertEqual("explicit", requirements["panel_0.chart_type"].status)
        self.assertEqual("sales", requirements["panel_0.measure_column"].source_span)

    def test_llm_requirements_are_source_of_truth_over_top_level_summary(self):
        parser = LLMIntentParser(
            StubLLMClient(
                {
                    "chart_type": "bar",
                    "dimensions": ["wrong_dimension"],
                    "measure_column": "wrong_measure",
                    "aggregation": "mean",
                    "requirements": [
                        {
                            "scope": "panel",
                            "type": "encoding",
                            "name": "chart_type",
                            "value": "line",
                            "source_span": "line chart",
                            "status": "explicit",
                            "confidence": 0.95,
                        },
                        {
                            "scope": "panel",
                            "type": "data_operation",
                            "name": "aggregation",
                            "value": "sum",
                            "source_span": "total",
                            "status": "explicit",
                            "confidence": 0.9,
                        },
                        {
                            "scope": "panel",
                            "type": "data_operation",
                            "name": "measure_column",
                            "value": "sales",
                            "source_span": "sales",
                            "status": "explicit",
                            "confidence": 0.9,
                        },
                        {
                            "scope": "panel",
                            "type": "data_operation",
                            "name": "dimensions",
                            "value": ["month"],
                            "source_span": "month",
                            "status": "explicit",
                            "confidence": 0.9,
                        },
                    ],
                    "confidence": 0.8,
                }
            )
        )

        bundle = parser.parse_requirements(
            "Plot total sales by month as a line chart.",
            TableSchema(columns={"month": "str", "sales": "number"}),
        )

        self.assertEqual("line", bundle.plan.chart_type)
        self.assertEqual("sum", bundle.plan.measure.agg)
        self.assertEqual("sales", bundle.plan.measure.column)
        self.assertEqual(("month",), bundle.plan.dimensions)

    def test_llm_requirement_normalizer_downgrades_ungrounded_or_invalid_claims(self):
        parser = LLMIntentParser(
            StubLLMClient(
                {
                    "requirements": [
                        {
                            "scope": "panel",
                            "type": "data_operation",
                            "name": "measure_column",
                            "value": "profit",
                            "source_span": "net profit",
                            "status": "explicit",
                            "confidence": 0.9,
                        },
                        {
                            "scope": "panel",
                            "type": "data_operation",
                            "name": "dimensions",
                            "value": ["month"],
                            "source_span": "time bucket",
                            "status": "explicit",
                            "confidence": 0.9,
                        },
                    ],
                    "chart_type": "bar",
                    "measure_column": "sales",
                    "dimensions": ["month"],
                    "aggregation": "sum",
                    "confidence": 0.8,
                }
            )
        )

        bundle = parser.parse_requirements(
            "Plot total sales by month.",
            TableSchema(columns={"month": "str", "sales": "number"}),
        )
        requirements = {requirement.name: requirement for requirement in bundle.requirement_plan.requirements}

        self.assertEqual("unsupported", requirements["measure_column"].status)
        self.assertEqual("", requirements["measure_column"].source_span)
        self.assertIn("not present in the schema", requirements["measure_column"].assumption)
        self.assertEqual("inferred", requirements["dimensions"].status)
        self.assertEqual("", requirements["dimensions"].source_span)
        self.assertIsNone(bundle.plan.measure.column)
        self.assertEqual(("month",), bundle.plan.dimensions)

    def test_schemaless_cleanup_drops_meta_and_uses_artist_type_instead_of_fake_core_fields(self):
        parser = LLMIntentParser(
            StubLLMClient(
                {
                    "chart_type": "multi_panel_figure",
                    "aggregation": "none",
                    "requirements": [
                        {
                            "scope": "panel_1",
                            "type": "artist",
                            "name": "plot_type",
                            "value": "semilogy",
                            "source_span": "plot the exponential decay of the data with a decay factor of 7.0 using a logarithmic scale on the y-axis",
                            "status": "explicit",
                            "confidence": 1.0,
                        },
                        {
                            "scope": "figure",
                            "type": "layout",
                            "name": "subplot_grid",
                            "value": "2x2",
                            "source_span": "4 subplots arranged in a 2x2 grid",
                            "status": "explicit",
                            "confidence": 1.0,
                        },
                        {
                            "scope": "panel",
                            "type": "display",
                            "name": "show_plot",
                            "value": True,
                            "source_span": "display the plot",
                            "status": "explicit",
                            "confidence": 1.0,
                        },
                        {
                            "scope": "panel",
                            "type": "library",
                            "name": "programming_language",
                            "value": "Python",
                            "source_span": "using Python",
                            "status": "explicit",
                            "confidence": 1.0,
                        },
                    ],
                    "confidence": 0.9,
                }
            )
        )

        bundle = parser.parse_requirements(
            "Create 4 subplots arranged in a 2x2 grid and plot the exponential decay of the data with a decay factor of 7.0 using a logarithmic scale on the y-axis, then display the plot using Python.",
            TableSchema(columns={}),
        )

        names = [requirement.name for requirement in bundle.requirement_plan.requirements]
        self.assertIn("artist_type", names)
        self.assertIn("subplot_layout", names)
        self.assertNotIn("show_plot", names)
        self.assertNotIn("programming_language", names)
        self.assertNotIn("aggregation", names)
        self.assertEqual("unknown", bundle.plan.chart_type)
        self.assertEqual("none", bundle.plan.measure.agg)
        artist = next(requirement for requirement in bundle.requirement_plan.requirements if requirement.name == "artist_type")
        self.assertEqual("line", artist.value)
        self.assertEqual("panel_1.artist_type_0", artist.requirement_id)
        self.assertEqual("core", artist.priority)
        self.assertIn("no single canonical chart_type", artist.assumption)

    def test_schemaless_cleanup_keeps_single_top_level_chart_type_when_no_artist_requirement_exists(self):
        parser = LLMIntentParser(
            StubLLMClient(
                {
                    "chart_type": "pie",
                    "aggregation": "none",
                    "requirements": [
                        {
                            "scope": "figure",
                            "type": "title",
                            "name": "main_title",
                            "value": "Market Share",
                            "source_span": "Title the chart Market Share",
                            "status": "explicit",
                            "confidence": 1.0,
                        }
                    ],
                    "confidence": 0.9,
                }
            )
        )

        bundle = parser.parse_requirements(
            "Title the chart Market Share.",
            TableSchema(columns={}),
        )

        requirements = {requirement.name: requirement for requirement in bundle.requirement_plan.requirements}
        self.assertEqual("pie", bundle.plan.chart_type)
        self.assertIn("chart_type", requirements)
        self.assertNotIn("aggregation", requirements)
        self.assertNotIn("measure_column", requirements)
        self.assertNotIn("dimensions", requirements)
        self.assertIn("title", requirements)

    def test_schemaless_cleanup_preserves_core_priority_for_artist_type(self):
        parser = LLMIntentParser(
            StubLLMClient(
                {
                    "chart_type": "multi_panel_figure",
                    "requirements": [
                        {
                            "scope": "panel",
                            "type": "encoding",
                            "name": "artist_type",
                            "value": "pie",
                            "source_span": "pie chart",
                            "status": "explicit",
                            "confidence": 1.0,
                            "priority": "core",
                        }
                    ],
                    "confidence": 0.9,
                }
            )
        )

        bundle = parser.parse_requirements(
            "Create a pie chart with custom labels.",
            TableSchema(columns={}),
        )

        requirement = bundle.requirement_plan.requirements[0]
        self.assertEqual("artist_type", requirement.name)
        self.assertEqual("core", requirement.priority)

    def test_schemaless_cleanup_downgrades_shape_only_artist_type_to_ambiguous(self):
        parser = LLMIntentParser(
            StubLLMClient(
                {
                    "chart_type": "multi_panel_figure",
                    "requirements": [
                        {
                            "scope": "panel_4",
                            "type": "encoding",
                            "name": "artist_type",
                            "value": "scatter",
                            "source_span": "draw squares",
                            "status": "explicit",
                            "confidence": 1.0,
                            "assumption": "Squares are drawn as scatter plot markers",
                        },
                        {
                            "scope": "panel_4",
                            "type": "style",
                            "name": "marker_shape",
                            "value": "square",
                            "source_span": "squares",
                            "status": "explicit",
                            "confidence": 1.0,
                        },
                    ],
                    "confidence": 0.9,
                }
            )
        )

        bundle = parser.parse_requirements(
            "In the fourth subplot, draw squares at random positions.",
            TableSchema(columns={}),
        )

        artist = next(requirement for requirement in bundle.requirement_plan.requirements if requirement.name == "artist_type")
        self.assertEqual("ambiguous", artist.status)
        self.assertIn("does not uniquely determine the rendering primitive", artist.assumption)
        self.assertIsNone(derive_expected_figure(bundle.requirement_plan))


if __name__ == "__main__":
    unittest.main()
