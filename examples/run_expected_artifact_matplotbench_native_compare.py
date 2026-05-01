from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from grounded_chart import (
    GroundedChartPipeline,
    HeuristicIntentParser,
    LLMExpectedArtifactExtractor,
    OpenAICompatibleLLMClient,
    load_ablation_run_config,
)
from grounded_chart_adapters import BatchRunner, InMemoryCaseAdapter, write_batch_report_html

from examples.run_framework_mvp_compare import (
    ExpectedArtifactAugmentingAdapter,
    _figure_requirement_count,
    llm_usage_metrics_from_cases,
)
from examples.run_matplotbench_figure_bridge_smoke import _load_native_failed_cases


FIGURE_SIGNAL_KEYWORDS = {
    "subplot": 4,
    "subplots": 4,
    "mosaic": 5,
    "layout": 3,
    "figure size": 3,
    "figsize": 3,
    "title": 3,
    "legend": 3,
    "axis": 2,
    "axes": 2,
    "label": 2,
    "tick": 2,
    "projection": 4,
    "ellipse": 3,
    "annotation": 3,
    "text": 2,
    "pie": 3,
    "connecting line": 3,
    "hatch": 2,
    "transparency": 2,
    "colorbar": 3,
    "gridline": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline verification against LLM expected-artifact figure verification on MatPlotBench native failed cases."
    )
    parser.add_argument("--config", default="configs/llm_ablation.deepseek.yaml")
    parser.add_argument("--bench", default="benchmarks/matplotbench_ds_failed_native.json")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--case-ids", nargs="*", default=())
    parser.add_argument("--output-dir", default="outputs/expected_artifact_matplotbench_native_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_ablation_run_config(project_root / args.config)
    provider = config.parser_provider or config.repair_provider
    if provider is None:
        raise ValueError("No LLM provider found in config. Fill llm.default, llm.parser, or llm.repair first.")

    bench_path = project_root / args.bench
    raw_cases = _load_raw_cases(bench_path)
    chart_cases = _load_native_failed_cases(bench_path)
    selected = _select_cases(raw_cases, chart_cases, limit=max(1, int(args.limit)), explicit_case_ids=tuple(args.case_ids))
    selected_cases = tuple(chart_cases[item["case_id"]] for item in selected)

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = GroundedChartPipeline(
        parser=HeuristicIntentParser(),
        repairer=None,
        enable_bounded_repair_loop=False,
    )

    baseline = BatchRunner(pipeline, continue_on_error=True).run(InMemoryCaseAdapter(selected_cases))
    baseline_dir = output_dir / "baseline_verify_only"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline.report.write_json(baseline_dir / "report.json")
    baseline.report.write_jsonl(baseline_dir / "cases.jsonl")
    write_batch_report_html(
        baseline.report,
        baseline_dir / "report.html",
        title="MatPlotBench Native Baseline Verify Only",
    )

    extractor = LLMExpectedArtifactExtractor(OpenAICompatibleLLMClient(provider))
    evidence_adapter = ExpectedArtifactAugmentingAdapter(
        bench_path,
        InMemoryCaseAdapter(selected_cases),
        extractor,
    )
    evidence = BatchRunner(pipeline, continue_on_error=True).run(evidence_adapter)
    evidence_dir = output_dir / "evidence_artifact_verifier"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence.report.write_json(evidence_dir / "report.json")
    evidence.report.write_jsonl(evidence_dir / "cases.jsonl")
    write_batch_report_html(
        evidence.report,
        evidence_dir / "report.html",
        title="MatPlotBench Native Evidence Artifact Verifier",
    )

    summary = _build_summary(
        bench_path=bench_path,
        selected=selected,
        baseline_report=baseline.report.to_dict(),
        evidence_report=evidence.report.to_dict(),
    )
    (output_dir / "selection.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "compare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Bench:", bench_path)
    print("Output:", output_dir)
    print("Selected:", [item["case_id"] for item in selected])
    print("Baseline:", summary["baseline"])
    print("Evidence:", summary["evidence_artifact_verifier"])
    print("Delta:", summary["delta"])
    print("LLM usage:", summary["llm_usage_metrics"])
    for case in summary["cases"]:
        print(
            case["case_id"],
            "signal=", case["selection_score"],
            "baseline=", case["baseline_status"],
            "evidence=", case["evidence_status"],
            "fig_reqs=", case["extracted_figure_requirement_count"],
            "evidence_errors=", case["evidence_error_codes"],
        )


def _load_raw_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def _select_cases(
    raw_cases: list[dict[str, Any]],
    chart_cases: dict[str, Any],
    *,
    limit: int,
    explicit_case_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    raw_by_id = {str(item.get("case_id")): item for item in raw_cases if item.get("case_id")}
    if explicit_case_ids:
        selected = []
        for rank, case_id in enumerate(explicit_case_ids, start=1):
            if case_id not in chart_cases:
                raise KeyError(f"Case not found: {case_id}")
            raw = raw_by_id.get(case_id, {})
            selected.append(_selection_item(raw, case_id=case_id, rank=rank, reason="explicit_case_id"))
        return selected

    ranked = []
    for raw in raw_cases:
        case_id = str(raw.get("case_id") or "")
        if not case_id or case_id not in chart_cases:
            continue
        ranked.append(_selection_item(raw, case_id=case_id, rank=0, reason="figure_signal_keyword_score"))
    ranked.sort(
        key=lambda item: (
            -int(item["selection_score"]),
            float(item["benchmark_score"]) if item.get("benchmark_score") is not None else float("inf"),
            int(item["native_id"]) if item.get("native_id") is not None else 10**9,
            item["case_id"],
        )
    )
    selected = ranked[:limit]
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
    return selected


def _selection_item(raw: dict[str, Any], *, case_id: str, rank: int, reason: str) -> dict[str, Any]:
    source_text = _case_source_text(raw)
    hits = _keyword_hits(source_text)
    return {
        "case_id": case_id,
        "rank": rank,
        "native_id": raw.get("native_id"),
        "benchmark_score": raw.get("score"),
        "selection_reason": reason,
        "selection_score": sum(FIGURE_SIGNAL_KEYWORDS[key] for key in hits),
        "keyword_hits": hits,
        "source_preview": source_text[:500],
    }


def _case_source_text(raw: dict[str, Any]) -> str:
    parts = []
    for key in ("query", "simple_instruction", "instruction", "expert_instruction"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def _keyword_hits(text: str) -> list[str]:
    normalized = text.lower()
    return [keyword for keyword in FIGURE_SIGNAL_KEYWORDS if keyword in normalized]


def _build_summary(
    *,
    bench_path: Path,
    selected: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    evidence_report: dict[str, Any],
) -> dict[str, Any]:
    selected_by_id = {item["case_id"]: item for item in selected}
    baseline_cases = {case["case_id"]: case for case in baseline_report.get("cases", [])}
    evidence_cases = {case["case_id"]: case for case in evidence_report.get("cases", [])}
    case_summaries = []
    newly_detected_failure_cases = 0
    cases_with_extracted_figure_requirements = 0
    total_extracted_figure_requirements = 0

    for case_id in selected_by_id:
        baseline_case = baseline_cases.get(case_id, {})
        evidence_case = evidence_cases.get(case_id, {})
        extraction = (evidence_case.get("case_metadata") or {}).get("expected_artifact_extraction") or {}
        figure_requirement_count = int(extraction.get("figure_requirement_count") or 0)
        if figure_requirement_count > 0:
            cases_with_extracted_figure_requirements += 1
            total_extracted_figure_requirements += figure_requirement_count
        baseline_status = baseline_case.get("status")
        evidence_status = evidence_case.get("status")
        baseline_verdict = baseline_case.get("case_verdict")
        evidence_verdict = evidence_case.get("case_verdict")
        if baseline_status == "passed" and evidence_status == "failed":
            newly_detected_failure_cases += 1
        case_summaries.append(
            {
                "case_id": case_id,
                "native_id": selected_by_id[case_id].get("native_id"),
                "selection_score": selected_by_id[case_id].get("selection_score"),
                "keyword_hits": selected_by_id[case_id].get("keyword_hits", []),
                "baseline_status": baseline_status,
                "evidence_status": evidence_status,
                "baseline_case_verdict": baseline_verdict,
                "evidence_case_verdict": evidence_verdict,
                "baseline_error_codes": list(baseline_case.get("error_codes", [])),
                "evidence_error_codes": list(evidence_case.get("error_codes", [])),
                "extracted_artifact_types": list(extraction.get("artifact_types", [])),
                "extracted_figure_requirement_count": figure_requirement_count,
                "extracted_plot_trace_count": int(extraction.get("plot_trace_count") or 0),
                "llm_total_tokens": ((extraction.get("llm_trace") or {}).get("usage") or {}).get("total_tokens"),
                "evidence_figure_requirements": evidence_case.get("figure_requirements"),
            }
        )

    baseline_summary = dict(baseline_report.get("summary", {}))
    evidence_summary = dict(evidence_report.get("summary", {}))
    return {
        "bench_path": str(bench_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_case_count": len(selected),
        "baseline": _compact_report_summary(baseline_summary),
        "evidence_artifact_verifier": _compact_report_summary(evidence_summary),
        "delta": {
            "passed_cases_delta": int(evidence_summary.get("passed_cases", 0)) - int(baseline_summary.get("passed_cases", 0)),
            "failed_cases_delta": int(evidence_summary.get("failed_cases", 0)) - int(baseline_summary.get("failed_cases", 0)),
            "newly_detected_failure_cases": newly_detected_failure_cases,
            "cases_with_extracted_figure_requirements": cases_with_extracted_figure_requirements,
            "total_extracted_figure_requirements": total_extracted_figure_requirements,
        },
        "llm_usage_metrics": llm_usage_metrics_from_cases(evidence_report.get("cases", [])),
        "cases": case_summaries,
    }


def _compact_report_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_cases": summary.get("total_cases"),
        "completed_cases": summary.get("completed_cases"),
        "passed_cases": summary.get("passed_cases"),
        "failed_cases": summary.get("failed_cases"),
        "errored_cases": summary.get("errored_cases"),
        "hard_failed_cases": summary.get("hard_failed_cases"),
        "warning_only_failed_cases": summary.get("warning_only_failed_cases"),
        "soft_passed_cases": summary.get("soft_passed_cases"),
        "overall_pass_rate": summary.get("overall_pass_rate"),
        "overall_hard_pass_rate": summary.get("overall_hard_pass_rate"),
        "case_verdict_counts": summary.get("case_verdict_counts"),
    }


if __name__ == "__main__":
    main()

