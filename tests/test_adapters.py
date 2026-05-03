import unittest
import json
import tempfile
from pathlib import Path

from grounded_chart.api import DataPoint, GroundedChartPipeline, HeuristicIntentParser, PlotTrace, RuleBasedRepairer, TableSchema
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
        actual_artifacts = {artifact.artifact_id: artifact for artifact in result.pipeline_result.evidence_graph.actual_artifacts}
        self.assertIn("actual.x_values", actual_artifacts)
        self.assertIn("actual.y_values", actual_artifacts)
        self.assertIn("actual.plot_points", actual_artifacts)
        self.assertIn("actual.semantic_data_variables", actual_artifacts)
        self.assertIn("actual.sequence_candidates", actual_artifacts)
        self.assertIn("actual.data_variables", actual_artifacts)
        self.assertEqual(["A", "A", "B"], actual_artifacts["actual.x_values"].payload)
        self.assertEqual([10, 15, 7], actual_artifacts["actual.y_values"].payload)
        self.assertEqual(
            [{"x": "A", "y": 10}, {"x": "A", "y": 15}, {"x": "B", "y": 7}],
            actual_artifacts["actual.plot_points"].payload,
        )
        semantic_payload = actual_artifacts["actual.semantic_data_variables"].payload
        self.assertEqual("categorical_sequence", semantic_payload["sequence_candidates"][0]["role"])
        self.assertEqual("numeric_sequence", semantic_payload["sequence_candidates"][1]["role"])
        self.assertEqual(semantic_payload["sequence_candidates"], actual_artifacts["actual.sequence_candidates"].payload)
        variables = {item["name"]: item["preview"] for item in actual_artifacts["actual.data_variables"].payload}
        self.assertEqual(["A", "A", "B"], variables["categories"])
        self.assertEqual([10, 15, 7], variables["sales"])

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


    def test_json_adapter_loads_utf8_bom_case_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_path = root / "case_code.py"
            code_path.write_text("import matplotlib.pyplot as plt\nplt.bar(['A'], [1])\n", encoding="utf-8")
            cases_path = root / "cases.json"
            cases_path.write_text(
                "\ufeff"
                + json.dumps(
                    [
                        {
                            "case_id": "bom-case",
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
        self.assertEqual("bom-case", loaded[0].case_id)

    def test_json_adapter_loads_figure_artifact_contracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_path = root / "case_code.py"
            code_path.write_text("import matplotlib.pyplot as plt\nplt.bar(['A'], [1])\n", encoding="utf-8")
            cases_path = root / "cases.json"
            cases_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "visual-contract-case",
                            "query": "Draw a connected stacked bar chart.",
                            "schema": {"columns": {}},
                            "rows": [],
                            "generated_code_path": "case_code.py",
                            "figure_requirements": {
                                "axes_count": 1,
                                "source_spans": {"axes_count": "one subplot"},
                                "artifact_contracts": [
                                    {
                                        "artifact_type": "panel_chart_type",
                                        "expected": {"chart_type": "stacked_bar"},
                                        "locator": {"panel_id": "panel_0"},
                                        "source_requirement_id": "panel_0.chart_type",
                                        "source_span": "stacked bar chart",
                                    }
                                ],
                                "axes": [
                                    {
                                        "axis_index": 0,
                                        "title": "Summary",
                                        "source_spans": {"title": "title Summary"},
                                    }
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            loaded = list(JsonCaseAdapter(cases_path).iter_cases())

        figure = loaded[0].figure_requirements
        self.assertEqual({"axes_count": "one subplot"}, figure.source_spans)
        self.assertEqual("title Summary", figure.axes[0].source_spans["title"])
        self.assertEqual("panel_chart_type", figure.artifact_contracts[0]["artifact_type"])
        self.assertEqual("stacked_bar", figure.artifact_contracts[0]["expected"]["chart_type"])

    def test_json_adapter_loads_expected_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_path = root / "case_code.py"
            code_path.write_text("import matplotlib.pyplot as plt\nplt.bar([1], [4])\n", encoding="utf-8")
            cases_path = root / "cases.json"
            cases_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "expected-trace-case",
                            "query": "Plot the specified bar values.",
                            "schema": {"columns": {}},
                            "rows": [],
                            "generated_code_path": "case_code.py",
                            "verification_mode": "figure_and_data",
                            "expected_trace": {
                                "chart_type": "bar",
                                "source": "unit_test_expected_points",
                                "points": [{"x": 1, "y": 4}],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            loaded = list(JsonCaseAdapter(cases_path).iter_cases())

        self.assertEqual("figure_and_data", loaded[0].verification_mode)
        self.assertIsNotNone(loaded[0].expected_trace)
        self.assertEqual("bar", loaded[0].expected_trace.chart_type)
        self.assertEqual((DataPoint(1, 4),), loaded[0].expected_trace.points)
        self.assertEqual("unit_test_expected_points", loaded[0].expected_trace.source)


    def test_json_adapter_auto_extracts_expected_trace_from_instruction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_path = root / "case_code.py"
            code_path.write_text("import matplotlib.pyplot as plt\nplt.bar([1, 2], [9, 9])\n", encoding="utf-8")
            cases_path = root / "cases.json"
            cases_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "instruction-derived-expected-trace",
                            "query": "MatPlotBench-style case.",
                            "expert_instruction": "Create a bar plot. Use x-values 1, 2 and y-values 4, 3.",
                            "schema": {"columns": {}},
                            "rows": [],
                            "generated_code_path": "case_code.py",
                            "verification_mode": "figure_only",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            loaded = list(JsonCaseAdapter(cases_path).iter_cases())

        self.assertIsNotNone(loaded[0].expected_trace)
        self.assertEqual("bar", loaded[0].expected_trace.chart_type)
        self.assertEqual((DataPoint(1, 4), DataPoint(2, 3)), loaded[0].expected_trace.points)
        self.assertTrue(loaded[0].expected_trace.source.startswith("expert_instruction:"))


    def test_json_adapter_does_not_extract_expected_trace_from_failure_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_path = root / "case_code.py"
            code_path.write_text("import matplotlib.pyplot as plt\nplt.bar([1, 2], [9, 9])\n", encoding="utf-8")
            cases_path = root / "cases.json"
            cases_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "no-leakage-from-reason",
                            "query": "MatPlotBench smoke case.",
                            "schema": {"columns": {}},
                            "rows": [],
                            "generated_code_path": "case_code.py",
                            "metadata": {
                                "reason": "The instruction specifies bar x-values 1,2 with y-values 4,3."
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            loaded = list(JsonCaseAdapter(cases_path).iter_cases())

        self.assertIsNone(loaded[0].expected_trace)

    def test_expected_trace_override_enables_data_verification_for_figure_case(self):
        case = ChartCase(
            case_id="expected-trace-override",
            query="MatPlotBench-style smoke case with explicit expected plotted values.",
            schema=TableSchema(columns={}),
            rows=(),
            generated_code="""
import matplotlib.pyplot as plt
plt.bar([1, 2, 3, 4, 5], [10, 15, 7, 12, 9])
""",
            verification_mode="figure_only",
            expected_trace=PlotTrace(
                "bar",
                (DataPoint(1, 4), DataPoint(2, 3), DataPoint(3, 2), DataPoint(4, 4)),
                source="unit_test_expected_points",
            ),
        )
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=None)

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertEqual(case.expected_trace, result.expected_trace)
        self.assertIn("length_mismatch_extra_points", result.report.error_codes)
        self.assertIn("unexpected_data_point", result.report.error_codes)
        self.assertIn("wrong_aggregation_value", result.report.error_codes)
        self.assertEqual(5, len(result.actual_trace.points))

if __name__ == "__main__":
    unittest.main()
