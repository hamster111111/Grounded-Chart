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


if __name__ == "__main__":
    unittest.main()
