from __future__ import annotations

import argparse
import json
from dataclasses import replace
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
    LLMExpectedArtifactExtractor,
    LLMIntentParser,
    LLMRepairer,
    OpenAICompatibleLLMClient,
    RuleBasedRepairer,
    TieredRepairer,
    load_ablation_run_config,
    merge_expected_figure_specs,
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
    parser.add_argument(
        "--repair-rounds",
        type=int,
        default=None,
        help="Optional override for max repair rounds used by repair-enabled variants.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to run after case-id filtering.",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Optional comma-separated case IDs to run, preserving benchmark order.",
    )
    parser.add_argument(
        "--parse-source",
        choices=("predicted", "oracle", "both"),
        default="predicted",
        help="Which requirement-plan source to evaluate. 'both' requires oracle plans in the benchmark JSON.",
    )
    parser.add_argument(
        "--expected-artifact-ablation",
        action="store_true",
        help="With an LLM config, add a verifier-only variant that extracts source-grounded expected artifacts and compiles verifiable figure requirements.",
    )
    parser.add_argument(
        "--llm-repair-ablation",
        action="store_true",
        help="With an LLM repair config, run vanilla and evidence-guided LLM repair variants instead of one configured variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_ablation_run_config(project_root / args.config) if args.config else None
    bench_name = args.bench or (config.bench_name if config is not None else "repair_loop_bench")
    bench_path = project_root / "benchmarks" / f"{bench_name}.json"
    case_ids = parse_case_ids(args.case_ids)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided.")
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / f"mvp_compare_{bench_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    parse_sources = resolve_parse_sources(args.parse_source)
    probe_adapter = JsonCaseAdapter(bench_path)
    if "oracle" in parse_sources and not probe_adapter.supports_oracle_requirements():
        raise ValueError(
            f"Bench {bench_path} does not contain oracle_plan/oracle_requirement_plan payloads, "
            "but --parse-source requested oracle evaluation."
        )

    variants = build_variants(
        config,
        repair_rounds_override=args.repair_rounds,
        parse_sources=parse_sources,
        llm_repair_ablation=args.llm_repair_ablation,
        expected_artifact_ablation=args.expected_artifact_ablation,
    )
    reports: dict[str, Any] = {}
    variant_summaries: dict[str, dict[str, Any]] = {}

    for variant in variants:
        variant_dir = output_dir / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        adapter = build_case_adapter(
            bench_path,
            parse_source_mode=variant["parse_source"],
            case_ids=case_ids,
            limit=args.limit,
        )
        if variant.get("expected_artifact_extractor") is not None:
            adapter = ExpectedArtifactAugmentingAdapter(
                bench_path,
                adapter,
                variant["expected_artifact_extractor"],
            )
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
            base_name=variant["base_name"],
            label=variant["label"],
            parse_source=variant["parse_source"],
            config_note=variant["config_note"],
            report=batch.report.to_dict(),
            report_path=variant_dir / "report.json",
        )

    compare_summary = {
        "bench": bench_name,
        "bench_path": str(bench_path),
        "case_filter": {
            "case_ids": list(case_ids),
            "limit": args.limit,
            "active": bool(case_ids or args.limit is not None),
        },
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
        llm_usage = variant_summary.get("llm_usage_metrics", {})
        print(
            variant_name,
            "pass_rate=",
            round(float(summary["overall_pass_rate"]), 4),
            "hard_pass_rate=",
            round(float(summary.get("overall_hard_pass_rate", summary["overall_pass_rate"])), 4),
            "failed_reqs=",
            requirement["failed_requirements"],
            "hard_failed_reqs=",
            requirement.get("hard_failed_requirements", 0),
            "warning_failed_reqs=",
            requirement.get("warning_failed_requirements", 0),
            "repair_attempt_cases=",
            variant_summary["repair_metrics"]["cases_with_repair_attempts"],
            "repair_llm_calls=",
            llm_usage.get("repair_call_count", llm_usage.get("call_count", 0)),
            "repair_tokens=",
            llm_usage.get("repair_total_tokens", llm_usage.get("total_tokens", 0)),
            "artifact_llm_calls=",
            llm_usage.get("expected_artifact_call_count", 0),
            "artifact_tokens=",
            llm_usage.get("expected_artifact_total_tokens", 0),
        )
    for comparison in compare_summary["comparisons"]:
        print(
            f"{comparison['baseline']} -> {comparison['candidate']}",
            "type=",
            comparison["comparison_type"],
            "newly_passed_cases=",
            comparison["newly_passed_cases"],
            "failed_requirement_delta=",
            comparison["failed_requirement_delta"],
            "unexpected_success_cases=",
            len(comparison["unexpected_success_cases"]),
        )


def resolve_parse_sources(value: str) -> tuple[str, ...]:
    normalized = str(value or "predicted").strip().lower()
    if normalized == "both":
        return ("predicted", "oracle")
    return (normalized,)



def parse_case_ids(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


class CaseFilteringAdapter:
    """Adapter wrapper for deterministic small-sample MVP runs."""

    def __init__(self, base_adapter: Any, *, case_ids: tuple[str, ...], limit: int | None) -> None:
        self.base_adapter = base_adapter
        self.case_ids = tuple(case_ids)
        self.case_id_set = set(case_ids)
        self.limit = limit

    def iter_cases(self):
        yielded = 0
        for case in self.base_adapter.iter_cases():
            if self.case_id_set and case.case_id not in self.case_id_set:
                continue
            yield case
            yielded += 1
            if self.limit is not None and yielded >= self.limit:
                break


def build_case_adapter(
    bench_path: Path,
    *,
    parse_source_mode: str,
    case_ids: tuple[str, ...],
    limit: int | None,
):
    adapter = JsonCaseAdapter(bench_path, parse_source_mode=parse_source_mode)
    if case_ids or limit is not None:
        return CaseFilteringAdapter(adapter, case_ids=case_ids, limit=limit)
    return adapter


class ExpectedArtifactAugmentingAdapter:
    """Adapter wrapper that injects LLM-extracted verifiable expected artifacts."""

    def __init__(self, bench_path: Path, base_adapter: Any, extractor: LLMExpectedArtifactExtractor) -> None:
        self.base_adapter = base_adapter
        self.extractor = extractor
        self.raw_by_case_id = _load_raw_cases_by_id(bench_path)

    def iter_cases(self):
        for case in self.base_adapter.iter_cases():
            raw = self.raw_by_case_id.get(case.case_id, {})
            metadata = dict(case.metadata)
            sources = _source_texts(raw)
            if not sources:
                metadata["expected_artifact_extraction"] = {"attempted": False, "reason": "no_source_text"}
                yield replace(case, metadata=metadata)
                continue
            source, text = sources[0]
            try:
                result = self.extractor.extract(text, source=source, case_id=case.case_id)
            except Exception as exc:
                metadata["expected_artifact_extraction"] = {
                    "attempted": True,
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                yield replace(case, metadata=metadata)
                continue

            merged_figure = merge_expected_figure_specs(case.figure_requirements, result.figure_requirements)
            metadata["expected_artifact_extraction"] = {
                "attempted": True,
                "source": source,
                "extractor": self.extractor.extractor_name,
                "artifact_count": len(result.artifacts),
                "artifact_types": [artifact.artifact_type for artifact in result.artifacts],
                "plot_trace_count": len(result.plot_traces),
                "figure_requirement_count": _figure_requirement_count(result.figure_requirements),
                "llm_trace": _llm_trace_summary(result.llm_trace),
            }
            yield replace(case, figure_requirements=merged_figure, metadata=metadata)


def _load_raw_cases_by_id(bench_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(bench_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    return {str(item.get("case_id")): item for item in data if isinstance(item, dict) and item.get("case_id")}


def _source_texts(raw: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    sources: list[tuple[str, str]] = []
    for key in ("query", "simple_instruction", "instruction", "raw_instruction", "prompt", "expert_instruction"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            sources.append((key, value))
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    for key in ("source_instruction", "simple_instruction", "instruction", "raw_instruction", "expert_instruction"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            sources.append((f"metadata.{key}", value))
    return tuple(sources)


def _llm_trace_summary(trace: Any) -> dict[str, Any] | None:
    if trace is None:
        return None
    usage = getattr(trace, "usage", None)
    return {
        "provider": getattr(trace, "provider", None),
        "model": getattr(trace, "model", None),
        "temperature": getattr(trace, "temperature", None),
        "max_tokens": getattr(trace, "max_tokens", None),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
        } if usage is not None else {},
    }


def _figure_requirement_count(figure: Any) -> int:
    if figure is None:
        return 0
    count = sum(1 for value in (figure.axes_count, figure.figure_title, figure.size_inches) if value is not None)
    for axis in figure.axes:
        count += sum(
            1
            for value in (axis.title, axis.xlabel, axis.ylabel, axis.zlabel, axis.projection, axis.xscale, axis.yscale, axis.zscale)
            if value is not None
        )
        count += len(axis.xtick_labels) + len(axis.ytick_labels) + len(axis.ztick_labels)
        count += len(axis.legend_labels) + len(axis.artist_types) + len(axis.text_contains)
        count += len(axis.artist_counts) + len(axis.min_artist_counts)
    return count

def build_variants(
    config: AblationRunConfig | None,
    *,
    repair_rounds_override: int | None = None,
    parse_sources: tuple[str, ...] = ("predicted",),
    llm_repair_ablation: bool = False,
    expected_artifact_ablation: bool = False,
) -> list[dict[str, Any]]:
    configured_rounds = int(repair_rounds_override) if repair_rounds_override is not None else (
        config.max_repair_rounds if config is not None else 2
    )
    variants = []
    multiple_sources = len(parse_sources) > 1
    for parse_source in parse_sources:
        suffix = f"__{parse_source}" if multiple_sources or parse_source != "predicted" else ""
        parse_label = f"{parse_source.title()} Plan"
        variants.extend(
            [
                {
                    "name": f"verify_only{suffix}",
                    "base_name": "verify_only",
                    "parse_source": parse_source,
                    "label": f"Verify Only ({parse_label})",
                    "config_note": {
                        "parse_source": parse_source,
                        "parser_backend": "heuristic",
                        "repair_backend": "none",
                        "enable_repair_loop": False,
                    },
                    "pipeline": GroundedChartPipeline(
                        parser=HeuristicIntentParser(),
                        repairer=None,
                        enable_bounded_repair_loop=False,
                    ),
                },
                {
                    "name": f"rule_repair{suffix}",
                    "base_name": "rule_repair",
                    "parse_source": parse_source,
                    "label": f"Rule Repair ({parse_label})",
                    "config_note": {
                        "parse_source": parse_source,
                        "parser_backend": "heuristic",
                        "repair_backend": "rule",
                        "repair_policy_mode": "strict",
                        "enable_repair_loop": True,
                        "max_repair_rounds": configured_rounds,
                    },
                    "pipeline": GroundedChartPipeline(
                        parser=HeuristicIntentParser(),
                        repairer=RuleBasedRepairer(),
                        repair_policy_mode="strict",
                        enable_bounded_repair_loop=True,
                        max_repair_rounds=configured_rounds,
                    ),
                },
            ]
        )
        if config is not None and expected_artifact_ablation:
            variants.append(
                {
                    "name": f"evidence_artifact_verifier{suffix}",
                    "base_name": "evidence_artifact_verifier",
                    "parse_source": parse_source,
                    "label": f"Evidence Artifact Verifier ({parse_label})",
                    "config_note": {
                        "parse_source": parse_source,
                        "parser_backend": "heuristic",
                        "repair_backend": "none",
                        "expected_artifact_extractor": "llm_expected_artifact_v1",
                        "expected_artifact_model": _expected_artifact_provider(config).model if _expected_artifact_provider(config) is not None else None,
                        "enable_repair_loop": False,
                    },
                    "expected_artifact_extractor": LLMExpectedArtifactExtractor(
                        build_client(_expected_artifact_provider(config), role="expected artifact extractor")
                    ),
                    "pipeline": GroundedChartPipeline(
                        parser=HeuristicIntentParser(),
                        repairer=None,
                        enable_bounded_repair_loop=False,
                    ),
                }
            )

        if config is not None:
            if llm_repair_ablation and config.repair_backend in {"llm", "tiered"}:
                variants.extend(
                    build_llm_repair_ablation_variants(
                        config,
                        parse_source=parse_source,
                        parse_label=parse_label,
                        suffix=suffix,
                        configured_rounds=configured_rounds,
                    )
                )
            else:
                variants.append(
                    {
                        "name": f"configured_repair{suffix}",
                        "base_name": "configured_repair",
                        "parse_source": parse_source,
                        "label": f"Configured Repair ({parse_label})",
                        "config_note": configured_config_note(
                            config,
                            parse_source=parse_source,
                            configured_rounds=configured_rounds,
                            repair_prompt="default",
                        ),
                        "pipeline": GroundedChartPipeline(
                            parser=build_parser(config),
                            repairer=build_repairer(config),
                            repair_policy_mode=config.repair_policy_mode,
                            enable_bounded_repair_loop=config.enable_repair_loop,
                            max_repair_rounds=configured_rounds,
                        ),
                    }
                )
    return variants


def build_llm_repair_ablation_variants(
    config: AblationRunConfig,
    *,
    parse_source: str,
    parse_label: str,
    suffix: str,
    configured_rounds: int,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for name, label, include_failure_atoms, repair_prompt in (
        ("vanilla_llm_repair", "Vanilla LLM Repair", False, "verification_errors_only"),
        ("evidence_guided_llm_repair", "Evidence-Guided LLM Repair", True, "failure_atoms"),
    ):
        variants.append(
            {
                "name": f"{name}{suffix}",
                "base_name": name,
                "parse_source": parse_source,
                "label": f"{label} ({parse_label})",
                "config_note": configured_config_note(
                    config,
                    parse_source=parse_source,
                    configured_rounds=configured_rounds,
                    repair_prompt=repair_prompt,
                ),
                "pipeline": GroundedChartPipeline(
                    parser=build_parser(config),
                    repairer=build_repairer(config, include_failure_atoms=include_failure_atoms),
                    repair_policy_mode=config.repair_policy_mode,
                    enable_bounded_repair_loop=config.enable_repair_loop,
                    max_repair_rounds=configured_rounds,
                ),
            }
        )
    return variants


def configured_config_note(
    config: AblationRunConfig,
    *,
    parse_source: str,
    configured_rounds: int,
    repair_prompt: str,
) -> dict[str, Any]:
    return {
        "parse_source": parse_source,
        "parser_backend": config.parser_backend,
        "repair_backend": config.repair_backend,
        "repair_tier_mode": config.repair_tier_mode,
        "repair_policy_mode": config.repair_policy_mode,
        "repair_prompt": repair_prompt,
        "enable_repair_loop": config.enable_repair_loop,
        "max_repair_rounds": configured_rounds,
        "output_name": config.output_name,
        "parser_model": config.parser_provider.model if config.parser_provider else None,
        "repair_model": config.repair_provider.model if config.repair_provider else None,
    }


def _expected_artifact_provider(config: AblationRunConfig):
    return config.parser_provider or config.repair_provider

def build_parser(config: AblationRunConfig):
    if config.parser_backend == "llm":
        return LLMIntentParser(build_client(config.parser_provider, role="parser"))
    return HeuristicIntentParser()


def build_repairer(config: AblationRunConfig, *, include_failure_atoms: bool = True):
    if config.repair_backend == "tiered":
        llm_repairer = LLMRepairer(build_client(config.repair_provider, role="repair"), include_failure_atoms=include_failure_atoms)
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
        return LLMRepairer(build_client(config.repair_provider, role="repair"), include_failure_atoms=include_failure_atoms)
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
    base_name: str,
    label: str,
    parse_source: str,
    config_note: Any,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    cases = list(report.get("cases", []))
    summary = dict(report.get("summary", {}))
    verdict_counter = Counter()
    case_verdict_counter = Counter()
    requirement_severity_counter = Counter()
    requirement_match_policy_counter = Counter()
    failed_requirement_severity_counter = Counter()
    failed_requirements = 0
    passed_requirements = 0
    hard_failed_requirements = 0
    hard_passed_requirements = 0
    warning_failed_requirements = 0
    abstained_requirements = 0
    unsupported_requirements = 0
    cases_with_repair_attempts = 0
    cases_with_resolved_requirements = 0
    resolved_requirement_ids: set[str] = set()

    def metric_ids(metrics: dict[str, Any], key: str) -> list[str]:
        return sorted(str(value) for value in (metrics.get(key) or []) if value is not None)

    def update_counter(counter: Counter, mapping: Any) -> bool:
        if not isinstance(mapping, dict):
            return False
        for key, value in mapping.items():
            try:
                counter[str(key)] += int(value or 0)
            except (TypeError, ValueError):
                counter[str(key)] += 1
        return bool(mapping)

    def metric_int(key: str, fallback: int) -> int:
        try:
            return int(summary.get(key, fallback) or 0)
        except (TypeError, ValueError):
            return fallback

    def metric_float(key: str, fallback: float) -> float:
        try:
            return float(summary.get(key, fallback) or 0.0)
        except (TypeError, ValueError):
            return fallback

    normalized_cases = []
    for case in cases:
        links = list((case.get("evidence_graph") or {}).get("links", []))
        metrics = dict(case.get("requirement_metrics") or {})
        link_meta_by_requirement: dict[str, dict[str, Any]] = {}
        for link in links:
            requirement_id = link.get("requirement_id")
            if requirement_id is not None:
                link_meta_by_requirement.setdefault(str(requirement_id), link)

        failed_ids = metric_ids(metrics, "failed_requirement_ids") or sorted(
            str(link["requirement_id"])
            for link in links
            if link.get("verdict") == "fail" and link.get("requirement_id") is not None
        )
        passed_ids = metric_ids(metrics, "passed_requirement_ids") or sorted(
            str(link["requirement_id"])
            for link in links
            if link.get("verdict") == "pass" and link.get("requirement_id") is not None
        )

        def requirement_severity(requirement_id: str) -> str:
            link = link_meta_by_requirement.get(requirement_id, {})
            return str(link.get("requirement_severity") or link.get("severity") or "error")

        hard_failed_ids = metric_ids(metrics, "hard_failed_requirement_ids")
        warning_failed_ids = metric_ids(metrics, "warning_failed_requirement_ids")
        hard_passed_ids = metric_ids(metrics, "hard_passed_requirement_ids")
        if "warning_failed_requirement_ids" not in metrics:
            warning_failed_ids = sorted(
                requirement_id for requirement_id in failed_ids if requirement_severity(requirement_id) != "error"
            )
        if "hard_failed_requirement_ids" not in metrics:
            hard_failed_ids = sorted(
                requirement_id
                for requirement_id in failed_ids
                if requirement_id not in set(warning_failed_ids) and requirement_severity(requirement_id) == "error"
            )
        if "hard_passed_requirement_ids" not in metrics:
            hard_passed_ids = sorted(
                requirement_id for requirement_id in passed_ids if requirement_severity(requirement_id) == "error"
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
        if not update_counter(requirement_severity_counter, metrics.get("requirement_severity_counts")):
            for link in links:
                if link.get("verdict") in {"pass", "fail"}:
                    requirement_severity_counter[str(link.get("requirement_severity") or link.get("severity") or "error")] += 1
        if not update_counter(requirement_match_policy_counter, metrics.get("requirement_match_policy_counts")):
            for link in links:
                if link.get("verdict") in {"pass", "fail"}:
                    requirement_match_policy_counter[str(link.get("requirement_match_policy") or link.get("match_policy") or "exact")] += 1
        if not update_counter(failed_requirement_severity_counter, metrics.get("failed_requirement_severity_counts")):
            for requirement_id in failed_ids:
                failed_requirement_severity_counter[requirement_severity(requirement_id)] += 1

        failed_requirements += len(failed_ids)
        passed_requirements += len(passed_ids)
        hard_failed_requirements += len(hard_failed_ids)
        hard_passed_requirements += len(hard_passed_ids)
        warning_failed_requirements += len(warning_failed_ids)
        abstained_requirements += int(metrics.get("abstained_requirements") or verdict_counter_delta(links, "abstain"))
        unsupported_requirements += int(metrics.get("unsupported_requirements") or verdict_counter_delta(links, "unsupported"))

        case_verdict = case.get("case_verdict")
        if not case_verdict:
            if case.get("status") == "passed":
                case_verdict = "passed"
            elif hard_failed_ids:
                case_verdict = "hard_failed"
            elif warning_failed_ids:
                case_verdict = "warning_only_failed"
            else:
                case_verdict = case.get("status") or "unknown"
        case_verdict_counter[str(case_verdict)] += 1

        normalized_cases.append(
            {
                "case_id": case["case_id"],
                "status": case["status"],
                "case_verdict": case_verdict,
                "parse_source": case.get("parse_source"),
                "error_codes": list(case.get("error_codes", [])),
                "failed_requirement_ids": failed_ids,
                "hard_failed_requirement_ids": hard_failed_ids,
                "warning_failed_requirement_ids": warning_failed_ids,
                "passed_requirement_ids": passed_ids,
                "hard_passed_requirement_ids": hard_passed_ids,
                "repair_attempt_count": len(repair_attempts),
                "resolved_requirement_ids": case_resolved_ids,
                "repairability": (case.get("case_metadata") or {}).get("repairability"),
                "expected_improvement": (case.get("case_metadata") or {}).get("expected_improvement"),
            }
        )

    verifiable_requirements = passed_requirements + failed_requirements
    hard_verifiable_requirements = hard_passed_requirements + hard_failed_requirements
    requirement_satisfaction = (passed_requirements / verifiable_requirements) if verifiable_requirements else 0.0
    hard_requirement_satisfaction = (
        hard_passed_requirements / hard_verifiable_requirements if hard_verifiable_requirements else 0.0
    )
    return {
        "name": name,
        "base_name": base_name,
        "label": label,
        "parse_source": parse_source,
        "config_note": config_note,
        "report_path": str(report_path),
        "summary": summary,
        "requirement_metrics": {
            "verifiable_requirements": metric_int("verifiable_requirements", verifiable_requirements),
            "passed_requirements": metric_int("passed_requirements", passed_requirements),
            "failed_requirements": metric_int("failed_requirements", failed_requirements),
            "hard_verifiable_requirements": metric_int("hard_verifiable_requirements", hard_verifiable_requirements),
            "hard_passed_requirements": metric_int("hard_passed_requirements", hard_passed_requirements),
            "hard_failed_requirements": metric_int("hard_failed_requirements", hard_failed_requirements),
            "warning_failed_requirements": metric_int("warning_failed_requirements", warning_failed_requirements),
            "abstained_requirements": metric_int("abstained_requirements", abstained_requirements),
            "unsupported_requirements": metric_int("unsupported_requirements", unsupported_requirements),
            "requirement_satisfaction": metric_float("requirement_satisfaction", requirement_satisfaction),
            "hard_requirement_satisfaction": metric_float("hard_requirement_satisfaction", hard_requirement_satisfaction),
            "requirement_severity_counts": dict(summary.get("requirement_severity_counts") or requirement_severity_counter),
            "requirement_match_policy_counts": dict(summary.get("requirement_match_policy_counts") or requirement_match_policy_counter),
            "failed_requirement_severity_counts": dict(summary.get("failed_requirement_severity_counts") or failed_requirement_severity_counter),
            "case_verdict_counts": dict(summary.get("case_verdict_counts") or case_verdict_counter),
        },
        "repair_metrics": {
            "cases_with_repair_attempts": cases_with_repair_attempts,
            "cases_with_resolved_requirements": cases_with_resolved_requirements,
            "resolved_requirement_count": len(resolved_requirement_ids),
        },
        "llm_usage_metrics": llm_usage_metrics_from_cases(report.get("cases", [])),
        "cases": normalized_cases,
    }

def llm_usage_metrics_from_cases(cases: list[dict[str, Any]]) -> dict[str, int]:
    total_metrics = _empty_llm_usage_metrics()
    repair_metrics = _empty_llm_usage_metrics()
    expected_artifact_metrics = _empty_llm_usage_metrics()
    for case in cases:
        repair_traces = []
        attempts = list(case.get("repair_attempts") or [])
        for attempt in attempts:
            trace = attempt.get("llm_trace")
            if trace:
                repair_traces.append(trace)
        if not repair_traces and case.get("repair_trace"):
            repair_traces.append(case["repair_trace"])
        for trace in repair_traces:
            _record_trace_usage(repair_metrics, trace)
            _record_trace_usage(total_metrics, trace)

        expected_artifact_trace = ((case.get("case_metadata") or {}).get("expected_artifact_extraction") or {}).get("llm_trace")
        if expected_artifact_trace:
            _record_trace_usage(expected_artifact_metrics, expected_artifact_trace)
            _record_trace_usage(total_metrics, expected_artifact_trace)

    return {
        **total_metrics,
        **_prefixed_llm_usage_metrics("repair", repair_metrics),
        **_prefixed_llm_usage_metrics("expected_artifact", expected_artifact_metrics),
    }


def _empty_llm_usage_metrics() -> dict[str, int]:
    return {
        "call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "missing_usage_count": 0,
    }


def _record_trace_usage(metrics: dict[str, int], trace: dict[str, Any]) -> None:
    metrics["call_count"] += 1
    usage = dict(trace.get("usage") or {})
    if not usage:
        metrics["missing_usage_count"] += 1
        return
    metrics["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    metrics["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    metrics["total_tokens"] += int(usage.get("total_tokens") or 0)


def _prefixed_llm_usage_metrics(prefix: str, metrics: dict[str, int]) -> dict[str, int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def verdict_counter_delta(links: list[dict[str, Any]], verdict: str) -> int:
    return sum(1 for link in links if link.get("verdict") == verdict)


def is_guarded_repairability(value: Any) -> bool:
    return str(value or "").strip() in {"diagnose_only", "unsupported", "route_only"}


def build_comparisons(variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    variants_by_parse_source: dict[str, dict[str, dict[str, Any]]] = {}
    variants_by_base_name: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in variants.values():
        variants_by_parse_source.setdefault(variant["parse_source"], {})[variant["base_name"]] = variant
        variants_by_base_name.setdefault(variant["base_name"], {})[variant["parse_source"]] = variant

    for parse_source, grouped in variants_by_parse_source.items():
        baseline = grouped.get("verify_only")
        if baseline is None:
            continue
        for base_name, candidate in grouped.items():
            if base_name == "verify_only":
                continue
            comparisons.append(
                compare_variants(
                    baseline,
                    candidate,
                    comparison_type="repair_within_parse_source",
                )
            )
        vanilla = grouped.get("vanilla_llm_repair")
        evidence_guided = grouped.get("evidence_guided_llm_repair")
        if vanilla is not None and evidence_guided is not None:
            comparisons.append(
                compare_variants(
                    vanilla,
                    evidence_guided,
                    comparison_type="llm_repair_prompt_ablation",
                )
            )

    for base_name, grouped in variants_by_base_name.items():
        predicted = grouped.get("predicted")
        oracle = grouped.get("oracle")
        if predicted is None or oracle is None:
            continue
        comparisons.append(
            compare_variants(
                predicted,
                oracle,
                comparison_type="parse_source_gap",
            )
        )
    return comparisons


def compare_variants(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    comparison_type: str = "variant_delta",
) -> dict[str, Any]:
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
        base_hard_failed = set(base_case.get("hard_failed_requirement_ids", base_failed))
        cand_hard_failed = set(cand_case.get("hard_failed_requirement_ids", cand_failed))
        resolved = sorted(base_failed - cand_failed)
        new_failures = sorted(cand_failed - base_failed)
        hard_resolved = sorted(base_hard_failed - cand_hard_failed)
        new_hard_failures = sorted(cand_hard_failed - base_hard_failed)

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
                "baseline_hard_failed_requirements": len(base_hard_failed),
                "candidate_hard_failed_requirements": len(cand_hard_failed),
                "resolved_requirement_ids": resolved,
                "new_failed_requirement_ids": new_failures,
                "hard_resolved_requirement_ids": hard_resolved,
                "new_hard_failed_requirement_ids": new_hard_failures,
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
        "comparison_type": comparison_type,
        "baseline": baseline["name"],
        "candidate": candidate["name"],
        "baseline_parse_source": baseline.get("parse_source"),
        "candidate_parse_source": candidate.get("parse_source"),
        "case_pass_delta": cand_summary["passed_cases"] - base_summary["passed_cases"],
        "overall_pass_rate_delta": cand_summary["overall_pass_rate"] - base_summary["overall_pass_rate"],
        "policy_clean_case_pass_delta": policy_clean_passed_cases - base_summary["passed_cases"],
        "policy_clean_candidate_passed_cases": policy_clean_passed_cases,
        "policy_clean_candidate_pass_rate": policy_clean_pass_rate,
        "policy_clean_overall_pass_rate_delta": policy_clean_pass_rate - base_summary["overall_pass_rate"],
        "requirement_satisfaction_delta": cand_req["requirement_satisfaction"] - base_req["requirement_satisfaction"],
        "hard_requirement_satisfaction_delta": cand_req.get("hard_requirement_satisfaction", 0.0) - base_req.get("hard_requirement_satisfaction", 0.0),
        "failed_requirement_delta": cand_req["failed_requirements"] - base_req["failed_requirements"],
        "hard_failed_requirement_delta": cand_req.get("hard_failed_requirements", 0) - base_req.get("hard_failed_requirements", 0),
        "warning_failed_requirement_delta": cand_req.get("warning_failed_requirements", 0) - base_req.get("warning_failed_requirements", 0),
        "llm_call_delta": candidate.get("llm_usage_metrics", {}).get("call_count", 0) - baseline.get("llm_usage_metrics", {}).get("call_count", 0),
        "llm_total_token_delta": candidate.get("llm_usage_metrics", {}).get("total_tokens", 0) - baseline.get("llm_usage_metrics", {}).get("total_tokens", 0),
        "repair_llm_call_delta": candidate.get("llm_usage_metrics", {}).get("repair_call_count", 0) - baseline.get("llm_usage_metrics", {}).get("repair_call_count", 0),
        "repair_llm_total_token_delta": candidate.get("llm_usage_metrics", {}).get("repair_total_tokens", 0) - baseline.get("llm_usage_metrics", {}).get("repair_total_tokens", 0),
        "expected_artifact_llm_call_delta": candidate.get("llm_usage_metrics", {}).get("expected_artifact_call_count", 0) - baseline.get("llm_usage_metrics", {}).get("expected_artifact_call_count", 0),
        "expected_artifact_llm_total_token_delta": candidate.get("llm_usage_metrics", {}).get("expected_artifact_total_tokens", 0) - baseline.get("llm_usage_metrics", {}).get("expected_artifact_total_tokens", 0),
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
