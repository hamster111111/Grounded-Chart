from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.run_relaxed_fidelity_ablation import (
    apply_repair_payload,
    build_code_context,
    build_relaxed_comparisons,
    build_repair_user_payload,
    build_source_file_evidence,
    build_source_schema_constraints,
    combine_evaluator_with_cases,
    compact_payload,
    detect_source_data_violation,
    normalize_matplotbench_evaluator_detail,
    plan_initial_repair,
    parse_variants,
)


class RelaxedFidelityAblationHelpersTest(unittest.TestCase):
    def test_parse_variants_rejects_unknown_variant(self) -> None:
        with self.assertRaises(ValueError):
            parse_variants("vanilla_repair,bad_variant")

    def test_build_code_context_keeps_short_code_full(self) -> None:
        code = "import matplotlib.pyplot as plt\nplt.plot([1], [2])\n"
        context = build_code_context(code, max_chars=4000)
        self.assertEqual("full", context["mode"])
        self.assertEqual(code, context["code"])

    def test_build_code_context_compacts_long_code_to_relevant_snippets(self) -> None:
        filler = "\n".join(f"value_{index} = {index}" for index in range(300))
        code = filler + "\nfig, ax = plt.subplots()\nax.set_title('Bad')\nfig.savefig(OUTPUT_PATH)\n"
        context = build_code_context(code, max_chars=1200)
        self.assertEqual("snippets", context["mode"])
        snippets = "\n".join(snippet["text"] for snippet in context["snippets"])
        self.assertIn("ax.set_title('Bad')", snippets)
        self.assertLess(len(snippets), len(code))

    def test_apply_repair_payload_prefers_patch_ops(self) -> None:
        code = "import matplotlib.pyplot as plt\nplt.title('Bad')\nplt.savefig(OUTPUT_PATH)\n"
        payload = {
            "summary": "Fix title.",
            "repair_kind": "patch_ops",
            "patch_ops": [
                {
                    "op": "replace_text",
                    "anchor": {"kind": "text", "text": "plt.title('Bad')", "occurrence": 1},
                    "new_value": "plt.title('Good')",
                }
            ],
            "repaired_code": "this should not be used",
        }
        result = apply_repair_payload(
            code,
            payload,
            max_patch_ops=2,
            max_changed_lines=5,
            allow_full_rewrite=True,
        )
        self.assertEqual("patch_ops", result.source)
        self.assertTrue(result.patch_applied)
        self.assertIn("plt.title('Good')", result.code)
        self.assertNotIn("this should not be used", result.code)

    def test_apply_repair_payload_can_disable_full_rewrite(self) -> None:
        code = "plt.title('Bad')\n"
        payload = {"repair_kind": "full_rewrite", "repaired_code": "plt.title('Good')\n"}
        result = apply_repair_payload(
            code,
            payload,
            max_patch_ops=2,
            max_changed_lines=5,
            allow_full_rewrite=False,
        )
        self.assertEqual("no_change", result.source)
        self.assertEqual(code, result.code)

    def test_compact_payload_enforces_budget_shape(self) -> None:
        payload = {"items": [{"text": "x" * 500} for _ in range(50)]}
        compacted = compact_payload(payload, max_chars=300)
        self.assertIsInstance(compacted, dict)
        self.assertIn("truncated", compacted)

    def test_official_evaluator_score_is_primary_metric(self) -> None:
        summary = combine_evaluator_with_cases(
            [
                {"case_id": "case-a", "native_id": 1, "original_score": 20, "render_ok": True},
                {"case_id": "case-b", "native_id": 2, "original_score": 10, "render_ok": False},
            ],
            {"details": {"1": {"score": 40}}},
        )

        self.assertEqual("official_avg_delta", summary["primary_metric"])
        self.assertEqual(15.0, summary["original_avg_score"])
        self.assertEqual(20.0, summary["official_final_avg_score"])
        self.assertEqual(5.0, summary["official_avg_delta"])
        self.assertEqual(0.0, summary["details"]["2"]["final_score"])
        self.assertNotIn("pass_rate_50", summary)

    def test_evaluator_detail_requires_final_score_marker(self) -> None:
        detail = normalize_matplotbench_evaluator_detail(
            {
                "score": 3,
                "raw": "The response was truncated near x from 3^-3",
                "error": None,
            }
        )

        self.assertIsNone(detail["score"])
        self.assertEqual(3, detail["raw_score_from_evaluator"])
        self.assertFalse(detail["final_score_marker_present"])
        self.assertEqual("evaluator_parse_error_missing_final_score", detail["score_parse_status"])

    def test_evaluator_detail_uses_explicit_final_score_marker(self) -> None:
        detail = normalize_matplotbench_evaluator_detail(
            {
                "score": 3,
                "raw": "Analysis with many numbers 1 2 3.\n[FINAL SCORE]: 50",
                "error": None,
            }
        )

        self.assertEqual(50, detail["score"])
        self.assertEqual(3, detail["raw_score_from_evaluator"])
        self.assertTrue(detail["final_score_marker_present"])
        self.assertEqual("final_score_marker", detail["score_parse_status"])

    def test_official_evaluator_parse_error_is_flagged_in_summary(self) -> None:
        summary = combine_evaluator_with_cases(
            [{"case_id": "case-a", "native_id": 1, "original_score": 20, "render_ok": True}],
            {
                "details": {
                    "1": {
                        "score": None,
                        "raw_score_from_evaluator": 3,
                        "score_parse_status": "evaluator_parse_error_missing_final_score",
                        "final_score_marker_present": False,
                        "error": "evaluator_parse_error_missing_final_score",
                    }
                }
            },
        )

        self.assertEqual(0.0, summary["details"]["1"]["final_score"])
        self.assertEqual(1, summary["evaluator_parse_error_cases"])
        self.assertEqual(3, summary["details"]["1"]["raw_score_from_evaluator"])
        self.assertIn("evaluator_parse_error_missing_final_score", summary["details"]["1"]["score_source"])

    def test_comparison_uses_official_score_delta(self) -> None:
        comparisons = build_relaxed_comparisons(
            {
                "vanilla_repair": {
                    "runtime_render_ok_cases_diagnostic": 1,
                    "llm_usage": {"call_count": 1, "total_tokens": 100},
                    "evaluator_summary": {
                        "summary": {
                            "official_final_avg_score": 20,
                            "official_avg_delta": 0,
                        }
                    },
                },
                "fidelity_repair": {
                    "runtime_render_ok_cases_diagnostic": 1,
                    "llm_usage": {"call_count": 2, "total_tokens": 150},
                    "evaluator_summary": {
                        "summary": {
                            "official_final_avg_score": 30,
                            "official_avg_delta": 10,
                        }
                    },
                },
            }
        )

        self.assertEqual("official_avg_score_delta", comparisons[0]["primary_metric"])
        self.assertEqual(10.0, comparisons[0]["official_avg_score_delta"])
        self.assertNotIn("official_score_ge_50_rate_delta", comparisons[0])

    def test_source_file_evidence_detects_required_available_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Imports.csv").write_text("year,value\n2020,10\n2021,12\n", encoding="utf-8")

            evidence = build_source_file_evidence(
                source_workspace=workspace,
                instruction="Create the chart using Imports.csv.",
                max_preview_rows=5,
                max_files=12,
            )

        self.assertEqual(["Imports.csv"], evidence["mentioned_available_files"])
        self.assertEqual(["year", "value"], evidence["available_files"][0]["columns"])
        self.assertEqual(2, evidence["available_files"][0]["row_count_preview"])

    def test_source_data_violation_escalates_random_data_without_required_file_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Imports.csv").write_text("year,value\n2020,10\n", encoding="utf-8")
            evidence = build_source_file_evidence(
                source_workspace=workspace,
                instruction="Use Imports.csv for the chart.",
                max_preview_rows=5,
                max_files=12,
            )

        violation = detect_source_data_violation(
            "import numpy as np\nvalues = np.random.normal(size=20)\n",
            source_file_evidence=evidence,
            instruction="Use Imports.csv for the chart.",
        )
        plan = plan_initial_repair(source_file_evidence=evidence, source_data_violation=violation)

        self.assertEqual("critical", violation["severity"])
        self.assertEqual(["Imports.csv"], violation["missing_file_uses"])
        self.assertEqual("structural_regeneration", plan["repair_action"])
        self.assertEqual("full_chart_code", plan["allowed_edit_scope"])

    def test_source_data_violation_allows_code_that_reads_required_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Imports.csv").write_text("year,value\n2020,10\n", encoding="utf-8")
            evidence = build_source_file_evidence(
                source_workspace=workspace,
                instruction="Use Imports.csv for the chart.",
                max_preview_rows=5,
                max_files=12,
            )

        violation = detect_source_data_violation(
            "import pandas as pd\ndf = pd.read_csv('Imports.csv')\n",
            source_file_evidence=evidence,
            instruction="Use Imports.csv for the chart.",
        )
        plan = plan_initial_repair(source_file_evidence=evidence, source_data_violation=violation)

        self.assertEqual("none", violation["severity"])
        self.assertEqual(["Imports.csv"], violation["used_files"])
        self.assertEqual("local_patch", plan["repair_action"])

    def test_source_schema_constraints_warn_for_wide_table_pivot_mismatch(self) -> None:
        evidence = {
            "available_files": [
                {
                    "name": "Imports.csv",
                    "columns": ["Year", "Urban", "Rural"],
                }
            ]
        }

        constraints = build_source_schema_constraints(evidence)

        self.assertEqual("wide_year_measure_table", constraints["files"][0]["schema_type"])
        self.assertTrue(
            any("melt before long-form Category/Value logic" in item for item in constraints["files"][0]["usage_constraints"])
        )

    def test_fidelity_payload_includes_source_evidence_but_vanilla_does_not(self) -> None:
        common = {
            "case_id": "case-a",
            "instruction": "Use Imports.csv.",
            "original_score": 20,
            "code_context": {"mode": "full", "code": "print('x')"},
            "script_observation": {"ok": True},
            "source_file_evidence": {"mentioned_available_files": ["Imports.csv"]},
            "source_schema_constraints": {"files": [{"name": "Imports.csv"}]},
            "source_data_violation": {"severity": "critical"},
            "repair_plan": {"repair_action": "structural_regeneration"},
            "expected_requirements": {"payload": {"requirements": []}},
            "diagnosis": {"payload": {"mismatches": []}},
        }

        vanilla_payload = build_repair_user_payload(common, variant="vanilla_repair", allow_full_rewrite=True)
        fidelity_payload = build_repair_user_payload(common, variant="fidelity_repair", allow_full_rewrite=True)

        self.assertNotIn("source_file_evidence", vanilla_payload)
        self.assertEqual(["Imports.csv"], fidelity_payload["source_file_evidence"]["mentioned_available_files"])
        self.assertEqual("Imports.csv", fidelity_payload["source_schema_constraints"]["files"][0]["name"])
        self.assertEqual("structural_regeneration", fidelity_payload["repair_plan"]["repair_action"])


if __name__ == "__main__":
    unittest.main()
