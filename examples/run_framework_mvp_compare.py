from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from grounded_chart import (
    AblationRunConfig,
    GroundedChartPipeline,
    HeuristicIntentParser,
    LLMIntentParser,
    LLMRepairer,
    OpenAICompatibleLLMClient,
    RuleBasedRepairer,
    TieredRepairer,
    load_ablation_run_config,
)
from grounded_chart_adapters import BatchRunner, JsonCaseAdapter, write_batch_report_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal GroundedChart MVP comparison across baseline and repair variants.")
    parser.add_argument(
        "--bench",
        default=None,
        help="Bench name without extension. Defaults to the configured bench or repair_loop_bench.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML/TOML config for an LLM-backed variant.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to outputs/mvp_compare_<bench>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_ablation_run_config(project_root / args.config) if args.config else None
    bench_name = args.bench or (config.bench_name if config is not None else "repair_loop_bench")
    bench_path = project_root / "benchmarks" / f"{bench_name}.json"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / f"mvp_compare_{bench_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variants(config)
    reports: dict[str, Any] = {}
    variant_summaries: dict[str, dict[str, Any]] = {}

    for variant in variants:
        variant_dir = output_dir / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        adapter = JsonCaseAdapter(bench_path)
        batch = BatchRunner(variant["pipeline"], continue_on_error=True).run(adapter)
        batch.report.write_json(variant_dir / "report.json")
        batch.report.write_jsonl(variant_dir / "cases.jsonl")
        write_batch_report_html(
            batch.report,
            variant_dir / "report.html",
            title=f"GroundedChart MVP Variant: {variant['label']}",
        )
        reports[variant["name"]] = batch.report
        variant_summaries[variant["name"]] = build_variant_summary(
            name=variant["name"],
            label=variant["label"],
            config_note=variant["config_note"],
            report=batch.report.to_dict(),
            report_path=variant_dir / "report.json",
        )

    compare_summary = {
        "bench": bench_name,
        "bench_path": str(bench_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "variants": variant_summaries,
        "comparisons": build_comparisons(variant_summaries),
    }
    compare_path = output_dir / "compare_summary.json"
    compare_path.write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Bench:", bench_path)
    print("Output:", output_dir)
    print("Compare summary:", compare_path)
    for variant_name, variant_summary in variant_summaries.items():
        summary = variant_summary["summary"]
        requirement = variant_summary["requirement_metrics"]
        print(
            variant_name,
            "pass_rate=",
            round(float(summary["overall_pass_rate"]), 4),
            "failed_reqs=",
            requirement["failed_requirements"],
            "repair_attempt_cases=",
            variant_summary["repair_metrics"]["cases_with_repair_attempts"],
        )
    for comparison in compare_summary["comparisons"]:
        print(
            f"{comparison['baseline']} -> {comparison['candidate']}",
            "newly_passed_cases=",
            comparison["newly_passed_cases"],
            "failed_requirement_delta=",
            comparison["failed_requirement_delta"],
            "unexpected_success_cases=",
            len(comparison["unexpected_success_cases"]),
        )


def build_variants(config: AblationRunConfig | None) -> list[dict[str, Any]]:
    variants = [
        {
            "name": "verify_only",
            "label": "Verify Only",
            "config_note": "heuristic parser, no repairer, no repair loop",
            "pipeline": GroundedChartPipeline(
                parser=HeuristicIntentParser(),
                repairer=None,
                enable_bounded_repair_loop=False,
            ),
        },
        {
            "name": "rule_repair",
            "label": "Rule Repair",
            "config_note": "heuristic parser + rule repairer + bounded repair loop",
            "pipeline": GroundedChartPipeline(
                parser=HeuristicIntentParser(),
                repairer=RuleBasedRepairer(),
                enable_bounded_repair_loop=True,
                max_repair_rounds=2,
            ),
        },
    ]
    if config is not None:
        variants.append(
            {
                "name": "configured_repair",
                "label": "Configured Repair",
                "config_note": {
                    "parser_backend": config.parser_backend,
                    "repair_backend": config.repair_backend,
                    "repair_tier_mode": config.repair_tier_mode,
                    "enable_repair_loop": config.enable_repair_loop,
                    "max_repair_rounds": config.max_repair_rounds,
                    "output_name": config.output_name,
                    "parser_model": config.parser_provider.model if config.parser_provider else None,
                    "repair_model": config.repair_provider.model if config.repair_provider else None,
                },
                "pipeline": GroundedChartPipeline(
                    parser=build_parser(config),
                    repairer=build_repairer(config),
                    enable_bounded_repair_loop=config.enable_repair_loop,
                    max_repair_rounds=config.max_repair_rounds,
                ),
            }
        )
    return variants


def build_parser(config: AblationRunConfig):
    if config.parser_backend == "llm":
        return LLMIntentParser(build_client(config.parser_provider, role="parser"))
    return HeuristicIntentParser()


def build_repairer(config: AblationRunConfig):
    if config.repair_backend == "tiered":
        llm_repairer = LLMRepairer(build_client(config.repair_provider, role="repair"))
        if config.repair_tier_mode == "llm_only":
            return TieredRepairer(
                deterministic_repairer=RuleBasedRepairer(),
                llm_repairer=llm_repairer,
                llm_scopes=(
                    "local_patch",
                    "data_transformation",
                    "structural_regeneration",
                    "backend_specific_regeneration",
                ),
            )
        if config.repair_tier_mode == "hybrid":
            return TieredRepairer(
                deterministic_repairer=RuleBasedRepairer(),
                llm_repairer=llm_repairer,
            )
        return RuleBasedRepairer()
    if config.repair_backend == "llm":
        return LLMRepairer(build_client(config.repair_provider, role="repair"))
    return RuleBasedRepairer()


def build_client(config, *, role: str):
    if config is None:
        raise ValueError(f"LLM {role} backend is enabled but no provider configuration was found.")
    if not str(config.api_key or "").strip():
        raise ValueError(
            f"LLM {role} backend is enabled but API key is empty. "
            "Fill `api_key` in the config file or set the configured environment variable."
        )
    return OpenAICompatibleLLMClient(config)


def build_variant_summary(
    *,
    name: str,
    label: str,
    config_note: Any,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    cases = list(report.get("cases", []))
    verdict_counter = Counter()
    failed_requirements = 0
    passed_requirements = 0
    abstained_requirements = 0
    unsupported_requirements = 0
    cases_with_repair_attempts = 0
    cases_with_resolved_requirements = 0
    resolved_requirement_ids: set[str] = set()

    normalized_cases = []
    for case in cases:
        links = list((case.get("evidence_graph") or {}).get("links", []))
        failed_ids = sorted(
            link["requirement_id"]
            for link in links
            if link.get("verdict") == "fail"
        )
        passed_ids = sorted(
            link["requirement_id"]
            for link in links
            if link.get("verdict") == "pass"
        )
        repair_attempts = list(case.get("repair_attempts", []))
        case_resolved_ids = sorted(
            {
                requirement_id
                for attempt in repair_attempts
                for requirement_id in attempt.get("resolved_requirement_ids", [])
            }
        )
        if repair_attempts:
            cases_with_repair_attempts += 1
        if case_resolved_ids:
            cases_with_resolved_requirements += 1
            resolved_requirement_ids.update(case_resolved_ids)

        for link in links:
            verdict = str(link.get("verdict") or "unknown")
            verdict_counter[verdict] += 1
        failed_requirements += len(failed_ids)
        passed_requirements += len(passed_ids)
        abstained_requirements += verdict_counter_delta(links, "abstain")
        unsupported_requirements += verdict_counter_delta(links, "unsupported")
        normalized_cases.append(
            {
                "case_id": case["case_id"],
                "status": case["status"],
                "error_codes": list(case.get("error_codes", [])),
                "failed_requirement_ids": failed_ids,
                "passed_requirement_ids": passed_ids,
                "repair_attempt_count": len(repair_attempts),
                "resolved_requirement_ids": case_resolved_ids,
                "repairability": (case.get("case_metadata") or {}).get("repairability"),
                "expected_improvement": (case.get("case_metadata") or {}).get("expected_improvement"),
            }
        )

    verifiable_requirements = passed_requirements + failed_requirements
    requirement_satisfaction = (passed_requirements / verifiable_requirements) if verifiable_requirements else 0.0
    return {
        "name": name,
        "label": label,
        "config_note": config_note,
        "report_path": str(report_path),
        "summary": dict(report.get("summary", {})),
        "requirement_metrics": {
            "verifiable_requirements": verifiable_requirements,
            "passed_requirements": passed_requirements,
            "failed_requirements": failed_requirements,
            "abstained_requirements": abstained_requirements,
            "unsupported_requirements": unsupported_requirements,
            "requirement_satisfaction": requirement_satisfaction,
        },
        "repair_metrics": {
            "cases_with_repair_attempts": cases_with_repair_attempts,
            "cases_with_resolved_requirements": cases_with_resolved_requirements,
            "resolved_requirement_count": len(resolved_requirement_ids),
        },
        "cases": normalized_cases,
    }


def verdict_counter_delta(links: list[dict[str, Any]], verdict: str) -> int:
    return sum(1 for link in links if link.get("verdict") == verdict)


def is_guarded_repairability(value: Any) -> bool:
    return str(value or "").strip() in {"diagnose_only", "unsupported", "route_only"}


def build_comparisons(variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = variants.get("verify_only")
    if baseline is None:
        return []
    comparisons = []
    for candidate_name, candidate in variants.items():
        if candidate_name == "verify_only":
            continue
        comparisons.append(compare_variants(baseline, candidate))
    return comparisons


def compare_variants(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_cases = {case["case_id"]: case for case in baseline["cases"]}
    candidate_cases = {case["case_id"]: case for case in candidate["cases"]}

    newly_passed_cases = 0
    regressed_cases = 0
    reduced_failure_cases = 0
    increased_failure_cases = 0
    case_deltas = []
    unexpected_success_cases = []

    for case_id in sorted(set(baseline_cases) | set(candidate_cases)):
        base_case = baseline_cases.get(case_id, {})
        cand_case = candidate_cases.get(case_id, {})
        base_failed = set(base_case.get("failed_requirement_ids", []))
        cand_failed = set(cand_case.get("failed_requirement_ids", []))
        resolved = sorted(base_failed - cand_failed)
        new_failures = sorted(cand_failed - base_failed)

        if base_case.get("status") != "passed" and cand_case.get("status") == "passed":
            newly_passed_cases += 1
        if base_case.get("status") == "passed" and cand_case.get("status") != "passed":
            regressed_cases += 1
        if len(cand_failed) < len(base_failed):
            reduced_failure_cases += 1
        if len(cand_failed) > len(base_failed):
            increased_failure_cases += 1

        repairability = cand_case.get("repairability") or base_case.get("repairability")
        if (
            base_case.get("status") != "passed"
            and cand_case.get("status") == "passed"
            and is_guarded_repairability(repairability)
        ):
            unexpected_success_cases.append(
                {
                    "case_id": case_id,
                    "repairability": repairability,
                    "expected_improvement": cand_case.get("expected_improvement") or base_case.get("expected_improvement"),
                }
            )

        case_deltas.append(
            {
                "case_id": case_id,
                "baseline_status": base_case.get("status"),
                "candidate_status": cand_case.get("status"),
                "baseline_failed_requirements": len(base_failed),
                "candidate_failed_requirements": len(cand_failed),
                "resolved_requirement_ids": resolved,
                "new_failed_requirement_ids": new_failures,
                "repairability": repairability,
                "expected_improvement": cand_case.get("expected_improvement") or base_case.get("expected_improvement"),
            }
        )

    base_summary = baseline["summary"]
    cand_summary = candidate["summary"]
    base_req = baseline["requirement_metrics"]
    cand_req = candidate["requirement_metrics"]
    guarded_success_count = len(unexpected_success_cases)
    total_cases = max(int(cand_summary.get("total_cases", 0)), 1)
    policy_clean_passed_cases = cand_summary["passed_cases"] - guarded_success_count
    policy_clean_pass_rate = policy_clean_passed_cases / total_cases
    return {
        "baseline": baseline["name"],
        "candidate": candidate["name"],
        "case_pass_delta": cand_summary["passed_cases"] - base_summary["passed_cases"],
        "overall_pass_rate_delta": cand_summary["overall_pass_rate"] - base_summary["overall_pass_rate"],
        "policy_clean_case_pass_delta": policy_clean_passed_cases - base_summary["passed_cases"],
        "policy_clean_candidate_passed_cases": policy_clean_passed_cases,
        "policy_clean_candidate_pass_rate": policy_clean_pass_rate,
        "policy_clean_overall_pass_rate_delta": policy_clean_pass_rate - base_summary["overall_pass_rate"],
        "requirement_satisfaction_delta": cand_req["requirement_satisfaction"] - base_req["requirement_satisfaction"],
        "failed_requirement_delta": cand_req["failed_requirements"] - base_req["failed_requirements"],
        "newly_passed_cases": newly_passed_cases,
        "policy_clean_newly_passed_cases": newly_passed_cases - guarded_success_count,
        "regressed_cases": regressed_cases,
        "reduced_failure_cases": reduced_failure_cases,
        "increased_failure_cases": increased_failure_cases,
        "unexpected_success_cases": unexpected_success_cases,
        "case_deltas": case_deltas,
    }


if __name__ == "__main__":
    main()
