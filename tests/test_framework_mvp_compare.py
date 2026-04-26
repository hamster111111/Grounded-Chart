import unittest

import examples.run_framework_mvp_compare as mvp_compare
from grounded_chart import AblationRunConfig, OpenAICompatibleConfig


class FrameworkMvpCompareTest(unittest.TestCase):
    def test_llm_repair_ablation_builds_vanilla_and_evidence_variants(self):
        config = AblationRunConfig(
            bench_name="small_error_bench",
            parser_backend="heuristic",
            repair_backend="llm",
            repair_policy_mode="strict",
            enable_repair_loop=True,
            max_repair_rounds=1,
            repair_provider=OpenAICompatibleConfig(
                model="test-model",
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
        )

        variants = mvp_compare.build_variants(
            config,
            repair_rounds_override=1,
            parse_sources=("predicted",),
            llm_repair_ablation=True,
        )

        names = [variant["name"] for variant in variants]
        self.assertIn("verify_only", names)
        self.assertIn("rule_repair", names)
        self.assertIn("vanilla_llm_repair", names)
        self.assertIn("evidence_guided_llm_repair", names)
        self.assertNotIn("configured_repair", names)
        notes = {variant["name"]: variant["config_note"] for variant in variants}
        self.assertEqual("verification_errors_only", notes["vanilla_llm_repair"]["repair_prompt"])
        self.assertEqual("failure_atoms", notes["evidence_guided_llm_repair"]["repair_prompt"])

    def test_default_configured_variant_is_preserved_without_llm_ablation_flag(self):
        config = AblationRunConfig(
            bench_name="small_error_bench",
            parser_backend="heuristic",
            repair_backend="llm",
            repair_provider=OpenAICompatibleConfig(
                model="test-model",
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
        )

        variants = mvp_compare.build_variants(
            config,
            repair_rounds_override=1,
            parse_sources=("predicted",),
        )

        names = [variant["name"] for variant in variants]
        self.assertIn("configured_repair", names)
        self.assertNotIn("vanilla_llm_repair", names)
        self.assertNotIn("evidence_guided_llm_repair", names)

    def test_expected_artifact_ablation_builds_verifier_variant(self):
        config = AblationRunConfig(
            bench_name="small_error_bench",
            parser_backend="heuristic",
            repair_backend="rule",
            parser_provider=OpenAICompatibleConfig(
                model="artifact-model",
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
        )

        variants = mvp_compare.build_variants(
            config,
            repair_rounds_override=1,
            parse_sources=("predicted",),
            expected_artifact_ablation=True,
        )

        names = [variant["name"] for variant in variants]
        self.assertIn("evidence_artifact_verifier", names)
        variant = next(item for item in variants if item["name"] == "evidence_artifact_verifier")
        self.assertEqual("evidence_artifact_verifier", variant["base_name"])
        self.assertIsNotNone(variant["expected_artifact_extractor"])
        self.assertEqual("artifact-model", variant["config_note"]["expected_artifact_model"])
        self.assertFalse(variant["config_note"]["enable_repair_loop"])

    def test_llm_usage_metrics_include_expected_artifact_extraction(self):
        metrics = mvp_compare.llm_usage_metrics_from_cases(
            [
                {
                    "case_id": "artifact-case",
                    "case_metadata": {
                        "expected_artifact_extraction": {
                            "llm_trace": {
                                "usage": {"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15}
                            }
                        }
                    },
                    "repair_attempts": [],
                }
            ]
        )

        self.assertEqual(1, metrics["call_count"])
        self.assertEqual(7, metrics["prompt_tokens"])
        self.assertEqual(8, metrics["completion_tokens"])
        self.assertEqual(15, metrics["total_tokens"])

    def test_llm_usage_metrics_count_attempt_traces_once(self):
        metrics = mvp_compare.llm_usage_metrics_from_cases(
            [
                {
                    "case_id": "a",
                    "repair_trace": {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
                    "repair_attempts": [
                        {"llm_trace": {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}}
                    ],
                },
                {
                    "case_id": "b",
                    "repair_trace": {"usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}},
                    "repair_attempts": [],
                },
            ]
        )

        self.assertEqual(2, metrics["call_count"])
        self.assertEqual(14, metrics["prompt_tokens"])
        self.assertEqual(25, metrics["completion_tokens"])
        self.assertEqual(39, metrics["total_tokens"])

    def test_build_comparisons_includes_llm_prompt_ablation(self):
        base_case = {
            "case_id": "case-1",
            "status": "failed",
            "parse_source": "predicted",
            "error_codes": [],
            "failed_requirement_ids": ["r1"],
            "passed_requirement_ids": [],
            "repair_attempt_count": 1,
            "resolved_requirement_ids": [],
            "repairability": "auto_repairable",
            "expected_improvement": "partial_or_full_pass",
        }
        variants = {
            "verify_only": {
                "name": "verify_only",
                "base_name": "verify_only",
                "parse_source": "predicted",
                "summary": {"total_cases": 1, "passed_cases": 0, "overall_pass_rate": 0.0},
                "requirement_metrics": {"requirement_satisfaction": 0.0, "failed_requirements": 1},
                "llm_usage_metrics": {"call_count": 0, "total_tokens": 0},
                "cases": [base_case],
            },
            "vanilla_llm_repair": {
                "name": "vanilla_llm_repair",
                "base_name": "vanilla_llm_repair",
                "parse_source": "predicted",
                "summary": {"total_cases": 1, "passed_cases": 0, "overall_pass_rate": 0.0},
                "requirement_metrics": {"requirement_satisfaction": 0.0, "failed_requirements": 1},
                "llm_usage_metrics": {"call_count": 1, "total_tokens": 100},
                "cases": [base_case],
            },
            "evidence_guided_llm_repair": {
                "name": "evidence_guided_llm_repair",
                "base_name": "evidence_guided_llm_repair",
                "parse_source": "predicted",
                "summary": {"total_cases": 1, "passed_cases": 1, "overall_pass_rate": 1.0},
                "requirement_metrics": {"requirement_satisfaction": 1.0, "failed_requirements": 0},
                "llm_usage_metrics": {"call_count": 1, "total_tokens": 160},
                "cases": [{**base_case, "status": "passed", "failed_requirement_ids": []}],
            },
        }

        comparisons = mvp_compare.build_comparisons(variants)
        ablation = next(item for item in comparisons if item["comparison_type"] == "llm_repair_prompt_ablation")
        self.assertEqual("vanilla_llm_repair", ablation["baseline"])
        self.assertEqual("evidence_guided_llm_repair", ablation["candidate"])
        self.assertEqual(1, ablation["case_pass_delta"])
        self.assertEqual(60, ablation["llm_total_token_delta"])

if __name__ == "__main__":
    unittest.main()