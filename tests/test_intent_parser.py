import unittest

from grounded_chart import HeuristicIntentParser, TableSchema


class HeuristicIntentParserTest(unittest.TestCase):
    def test_prefers_numeric_measure_over_dimension_mention(self):
        schema = TableSchema(columns={"category": "str", "sales": "number"})
        plan = HeuristicIntentParser().parse("Show total sales by category in a bar chart.", schema)
        self.assertEqual(plan.measure.column, "sales")
        self.assertEqual(plan.measure.agg, "sum")
        self.assertEqual(plan.dimensions, ("category",))

    def test_can_extract_two_dimensions_from_by_phrase(self):
        schema = TableSchema(columns={"category": "str", "segment": "str", "sales": "number"})
        plan = HeuristicIntentParser().parse(
            "Show total sales by category and segment in a bar chart.",
            schema,
        )
        self.assertEqual(plan.measure.column, "sales")
        self.assertEqual(plan.measure.agg, "sum")
        self.assertEqual(plan.dimensions, ("category", "segment"))

    def test_prefers_measure_before_by_phrase_for_scatter_style_query(self):
        schema = TableSchema(columns={"month": "number", "region": "str", "sales": "number"})
        plan = HeuristicIntentParser().parse(
            "Show sales by month and region in a scatter chart.",
            schema,
        )
        self.assertEqual(plan.measure.column, "sales")
        self.assertEqual(plan.measure.agg, "none")
        self.assertEqual(plan.dimensions, ("month", "region"))

    def test_parse_requirements_emits_parser_native_provenance(self):
        schema = TableSchema(columns={"category": "str", "sales": "number"})
        bundle = HeuristicIntentParser().parse_requirements(
            "Show top 5 total sales by category in a bar chart.",
            schema,
        )
        requirements = {requirement.requirement_id: requirement for requirement in bundle.requirement_plan.requirements}

        self.assertEqual(bundle.plan.limit, 5)
        self.assertEqual(bundle.plan.sort.direction, "desc")
        self.assertEqual(requirements["panel_0.chart_type"].source_span, "bar chart")
        self.assertEqual(requirements["panel_0.chart_type"].status, "explicit")
        self.assertEqual(requirements["panel_0.aggregation"].source_span, "total")
        self.assertEqual(requirements["panel_0.measure_column"].source_span, "sales")
        self.assertEqual(requirements["panel_0.dimensions"].source_span, "category")
        self.assertEqual(requirements["panel_0.limit"].source_span, "top 5")
        self.assertEqual(requirements["panel_0.sort"].source_span, "top 5")
        self.assertEqual(requirements["panel_0.limit"].depends_on, ("panel_0.sort",))

    def test_parse_requirements_labels_schema_defaults_as_assumptions(self):
        schema = TableSchema(columns={"category": "str", "sales": "number"})
        bundle = HeuristicIntentParser().parse_requirements("Show performance.", schema)
        requirements = {requirement.requirement_id: requirement for requirement in bundle.requirement_plan.requirements}

        self.assertEqual(requirements["panel_0.chart_type"].status, "assumed")
        self.assertEqual(requirements["panel_0.chart_type"].source_span, "")
        self.assertIsNotNone(requirements["panel_0.chart_type"].assumption)
        self.assertEqual(requirements["panel_0.measure_column"].status, "assumed")
        self.assertEqual(requirements["panel_0.measure_column"].source_span, "")
        self.assertIn("first numeric column", requirements["panel_0.measure_column"].assumption)
        self.assertEqual(requirements["panel_0.dimensions"].status, "assumed")
        self.assertIn("first categorical column", requirements["panel_0.dimensions"].assumption)

    def test_parse_requirements_extracts_explicit_filter(self):
        schema = TableSchema(columns={"category": "str", "sales": "number"})
        bundle = HeuristicIntentParser().parse_requirements(
            "Show total sales by category where sales greater than 10.",
            schema,
        )
        requirements = {requirement.requirement_id: requirement for requirement in bundle.requirement_plan.requirements}

        self.assertEqual(bundle.plan.filters[0].column, "sales")
        self.assertEqual(bundle.plan.filters[0].op, "gt")
        self.assertEqual(bundle.plan.filters[0].value, 10)
        self.assertEqual(requirements["panel_0.filter_0"].source_span, "sales greater than 10")
        self.assertEqual(requirements["panel_0.filter_0"].status, "explicit")


if __name__ == "__main__":
    unittest.main()
