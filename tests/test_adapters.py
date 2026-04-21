import unittest
import json
import tempfile
from pathlib import Path

from grounded_chart import GroundedChartPipeline, HeuristicIntentParser, RuleBasedRepairer, TableSchema
from grounded_chart_adapters import ChartCase, InMemoryCaseAdapter, JsonCaseAdapter


class AdapterTest(unittest.TestCase):
    def test_in_memory_adapter_runs_case_through_pipeline(self):
        case = ChartCase(
            case_id="toy-1",
            query="Show total sales by category in a bar chart, ascending.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "A", "sales": 15},
                {"category": "B", "sales": 7},
            ),
            generated_code="""
import matplotlib.pyplot as plt
categories = [row["category"] for row in rows]
sales = [row["sales"] for row in rows]
plt.bar(categories, sales)
""",
        )
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())
        result = InMemoryCaseAdapter([case]).run(pipeline)[0]
        self.assertEqual(result.case.case_id, "toy-1")
        self.assertFalse(result.pipeline_result.report.ok)
        self.assertEqual(result.pipeline_result.repair_plan.scope, "data_transformation")

    def test_json_adapter_loads_generated_code_from_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_path = root / "case_code.py"
            code_path.write_text("import matplotlib.pyplot as plt\nplt.bar(['A'], [1])\n", encoding="utf-8")
            cases_path = root / "cases.json"
            cases_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "path-case",
                            "query": "Show sales by category in a bar chart.",
                            "schema": {"columns": {"category": "str", "sales": "number"}},
                            "rows": [{"category": "A", "sales": 1}],
                            "generated_code_path": "case_code.py",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            loaded = list(JsonCaseAdapter(cases_path).iter_cases())

        self.assertEqual(1, len(loaded))
        self.assertIn("plt.bar", loaded[0].generated_code)


if __name__ == "__main__":
    unittest.main()
