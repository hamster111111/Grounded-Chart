import json
import tempfile
import unittest
from pathlib import Path

from grounded_chart_adapters import JsonCaseAdapter, MatplotBenchInstructionAdapter, MatplotBenchWorkspaceAdapter


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


if __name__ == "__main__":
    unittest.main()
