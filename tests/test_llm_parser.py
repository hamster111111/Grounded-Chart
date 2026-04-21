import unittest

from grounded_chart import LLMIntentParser, TableSchema


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
                    "sort": {"by": "measure", "direction": "desc"},
                    "limit": 5,
                    "confidence": 0.92,
                }
            )
        )

        plan = parser.parse(
            "Plot total sales by month as a line chart, descending, top 5.",
            TableSchema(columns={"month": "str", "sales": "number"}),
        )

        self.assertEqual("line", plan.chart_type)
        self.assertEqual(("month",), plan.dimensions)
        self.assertEqual("sales", plan.measure.column)
        self.assertEqual("sum", plan.measure.agg)
        self.assertEqual("measure", plan.sort.by)
        self.assertEqual("desc", plan.sort.direction)
        self.assertEqual(5, plan.limit)
        self.assertEqual(0.92, plan.confidence)


if __name__ == "__main__":
    unittest.main()
