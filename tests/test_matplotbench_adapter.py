import json
import tempfile
import unittest
from pathlib import Path

from grounded_chart_adapters import JsonCaseAdapter, MatplotBenchEvalWorkspaceExporter, MatplotBenchGenerationAdapter, MatplotBenchInstructionAdapter, MatplotBenchWorkspaceAdapter


class MatplotBenchAdapterTest(unittest.TestCase):
    def test_loads_native_matplotbench_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ground_truth").mkdir()
            (root / "data" / "1").mkdir(parents=True)
            (root / "ground_truth" / "example_1.png").write_bytes(b"fake")
            (root / "benchmark_instructions.json").write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "simple_instruction": "Draw a bar chart.",
                            "expert_instruction": "Use matplotlib bar.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = list(MatplotBenchInstructionAdapter(root).iter_records())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].case_id, "1")
            self.assertTrue(records[0].has_external_data)
            self.assertTrue(records[0].has_ground_truth_image)
            self.assertEqual(records[0].metadata["benchmark"], "MatPlotBench")

    def test_loads_json_chart_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case-1",
                            "query": "Show total sales by category in a bar chart.",
                            "schema": {"columns": {"category": "str", "sales": "number"}},
                            "rows": [{"category": "A", "sales": 1}],
                            "generated_code": "import matplotlib.pyplot as plt\nplt.bar(['A'], [1])",
                            "verification_mode": "figure_only",
                            "figure_requirements": {
                                "axes_count": 1,
                                "axes": [
                                    {
                                        "axis_index": 0,
                                        "title": "Sales",
                                        "xlabel": "Category",
                                        "ylabel": "Value",
                                    }
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            cases = list(JsonCaseAdapter(path).iter_cases())

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "case-1")
        self.assertEqual(cases[0].schema.columns["sales"], "number")
        self.assertEqual(cases[0].figure_requirements.axes_count, 1)
        self.assertEqual(cases[0].figure_requirements.axes[0].title, "Sales")
        self.assertEqual(cases[0].verification_mode, "figure_only")

    def test_loads_oracle_requirements_from_json_chart_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case-oracle",
                            "query": "Show total sales by category in a bar chart.",
                            "schema": {"columns": {"category": "str", "sales": "number"}},
                            "rows": [{"category": "A", "sales": 1}],
                            "generated_code": "import matplotlib.pyplot as plt\nplt.bar(['A'], [1])",
                            "oracle_plan": {
                                "chart_type": "bar",
                                "dimensions": ["category"],
                                "measure": {"column": "sales", "agg": "sum"},
                                "filters": [],
                                "sort": None,
                                "limit": None,
                                "raw_query": "Show total sales by category in a bar chart.",
                                "confidence": 1.0,
                            },
                            "oracle_requirement_plan": {
                                "raw_query": "Show total sales by category in a bar chart.",
                                "shared_requirement_ids": [],
                                "figure_requirements": {},
                                "requirements": [
                                    {
                                        "requirement_id": "panel_0.chart_type",
                                        "scope": "panel",
                                        "type": "encoding",
                                        "name": "chart_type",
                                        "value": "bar",
                                        "source_span": "bar chart",
                                        "status": "explicit",
                                        "confidence": 1.0,
                                        "depends_on": [],
                                        "priority": "core",
                                        "panel_id": "panel_0",
                                    },
                                    {
                                        "requirement_id": "panel_0.measure_column",
                                        "scope": "panel",
                                        "type": "data_operation",
                                        "name": "measure_column",
                                        "value": "sales",
                                        "source_span": "sales",
                                        "status": "explicit",
                                        "confidence": 1.0,
                                        "depends_on": [],
                                        "priority": "core",
                                        "panel_id": "panel_0",
                                    },
                                ],
                                "panels": [
                                    {
                                        "panel_id": "panel_0",
                                        "chart_type": "bar",
                                        "requirement_ids": ["panel_0.chart_type", "panel_0.measure_column"],
                                        "data_ops": {
                                            "dimensions": ["category"],
                                            "measure_column": "sales",
                                            "aggregation": "sum",
                                            "filters": [],
                                        },
                                        "encodings": {"chart_type": "bar"},
                                        "annotations": {},
                                        "presentation_constraints": {"sort": None, "limit": None},
                                    }
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            adapter = JsonCaseAdapter(path, parse_source_mode="oracle")
            self.assertTrue(adapter.supports_oracle_requirements())
            cases = list(adapter.iter_cases())

        self.assertEqual(1, len(cases))
        self.assertEqual("oracle", cases[0].parse_source)
        self.assertIsNotNone(cases[0].parsed_requirements)
        self.assertEqual("bar", cases[0].parsed_requirements.plan.chart_type)
        self.assertEqual(
            "panel_0.chart_type",
            cases[0].parsed_requirements.requirement_plan.requirements[0].requirement_id,
        )

    def test_loads_matplotbench_workspace_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "benchmark_data"
            workspace = Path(tmpdir) / "workspace_full"
            case_dir = workspace / "example_1"
            (root / "ground_truth").mkdir(parents=True)
            (case_dir).mkdir(parents=True)
            (root / "ground_truth" / "example_1.png").write_bytes(b"fake")
            (root / "benchmark_instructions.json").write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "simple_instruction": "Draw a bar chart.",
                            "expert_instruction": "Use matplotlib bar.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (workspace / "eval_results.json").write_text(
                json.dumps(
                    {
                        "summary": {"total": 1, "avg_score": 20},
                        "details": {"1": {"score": 20, "raw": "bad plot", "error": None}},
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "code_action_deepseek_initial_0.py").write_text("initial", encoding="utf-8")
            (case_dir / "code_action_deepseek_vis_refined_0.py").write_text("refined 0", encoding="utf-8")
            (case_dir / "code_action_deepseek_vis_refined_1.py").write_text("refined 1", encoding="utf-8")
            (case_dir / "code_action_deepseek_vis_refined_1.py.log").write_text("log", encoding="utf-8")
            (case_dir / "novice_final.png").write_bytes(b"fake image")
            manifest_path = Path(tmpdir) / "out" / "manifest.json"

            adapter = MatplotBenchWorkspaceAdapter(root, workspace)
            records = list(adapter.iter_records())
            candidates = adapter.write_candidate_manifest(manifest_path, score_lte=50)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].score, 20)
            self.assertEqual(records[0].selected_code_stage, "vis_refined")
            self.assertEqual(records[0].selected_code_iteration, 1)
            self.assertEqual(records[0].read_selected_code(), "refined 1")
            self.assertTrue(records[0].has_output_image)
            self.assertEqual(adapter.summary()["low_score_lt_50"], 1)
            self.assertEqual(len(candidates), 1)
            self.assertTrue(manifest_path.exists())


    def test_loads_matplotbench_generation_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "native_failed.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "matplotbench-ds-failed-4",
                            "native_id": 4,
                            "expert_instruction": "Create three confidence ellipses.",
                            "simple_instruction": "Draw ellipses.",
                            "schema": {"columns": {}},
                            "rows": [],
                            "score": 40,
                            "ground_truth_path": str(Path(tmpdir) / "gt.png"),
                        },
                        {
                            "case_id": "table-case",
                            "query": "Draw sales by product.",
                            "schema": {"columns": {"product": "string", "sales": "number"}},
                            "rows": [{"product": "A", "sales": 2}],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            cases = list(MatplotBenchGenerationAdapter(path).iter_cases())

        self.assertEqual(2, len(cases))
        self.assertEqual("instruction_only", cases[0].generation_mode)
        self.assertEqual("Create three confidence ellipses.", cases[0].query)
        self.assertEqual(4, cases[0].metadata["native_id"])
        self.assertEqual("table", cases[1].generation_mode)
        self.assertEqual("number", cases[1].schema.columns["sales"])
        self.assertEqual(2, cases[1].rows[0]["sales"])
    def test_exports_generation_summary_to_eval_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "generation" / "case"
            case_dir.mkdir(parents=True)
            image = case_dir / "figure.png"
            code = case_dir / "generated_final.py"
            manifest = case_dir / "generation_manifest.json"
            report = case_dir / "generation_report.json"
            image.write_bytes(b"fake image")
            code.write_text("print('plot')", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "matplotbench-ds-failed-4",
                                "image_path": str(image),
                                "final_code_path": str(code),
                                "manifest_path": str(manifest),
                                "report_path": str(report),
                                "metadata": {"native_id": 4},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            workspace = root / "eval_workspace"

            export_manifest = MatplotBenchEvalWorkspaceExporter(summary, workspace).export()

            example_dir = workspace / "example_4"
            self.assertTrue((example_dir / "novice_final.png").exists())
            self.assertTrue((example_dir / "code_action_groundedchart_initial_0.py").exists())
            self.assertTrue((example_dir / "groundedchart_generation_manifest.json").exists())
            self.assertTrue((workspace / "groundedchart_eval_export_manifest.json").exists())
            self.assertEqual([4], export_manifest["native_ids"])
            self.assertIn("--start_id 4 --end_id 4", export_manifest["eval_qwen_commands_by_id"][0])
if __name__ == "__main__":
    unittest.main()
