from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Iterable

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from grounded_chart import OpenAICompatibleLLMClient, load_ablation_run_config
from grounded_chart.expected_artifacts import ExplicitPointExpectedTraceExtractor, LLMExpectedArtifactExtractor
from grounded_chart.schema import PlotTrace


DEFAULT_MATPLOTBENCH_INSTRUCTIONS = Path(r"D:\Code\autoReaserch\MatPlotAgent\benchmark_data\benchmark_instructions.json")
DEFAULT_BENCHES = (
    project_root / "benchmarks" / "matplotbench_failed_tasks.json",
    project_root / "benchmarks" / "matplotbench_ds_failed_native.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit how often expected plotted-data artifacts can be extracted from benchmark instructions."
    )
    parser.add_argument(
        "--bench",
        action="append",
        default=[],
        help="Benchmark JSON list to audit. Can be repeated. Defaults to known MatPlotBench subsets if present.",
    )
    parser.add_argument(
        "--matplotbench-instructions",
        default=str(DEFAULT_MATPLOTBENCH_INSTRUCTIONS),
        help="Optional MatPlotBench benchmark_instructions.json to audit directly.",
    )
    parser.add_argument(
        "--no-default-benches",
        action="store_true",
        help="Only audit explicitly provided --bench paths.",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Optional YAML/TOML config. If provided, run LLM artifact extraction on a limited number of records.",
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=5,
        help="Maximum number of records to send to the LLM extractor when --llm-config is set.",
    )
    parser.add_argument(
        "--llm-on",
        choices=("uncovered", "all"),
        default="uncovered",
        help="Whether LLM extraction is attempted only on rule-uncovered records or all records.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/expected_artifact_coverage",
        help="Output directory for coverage_summary.json and coverage_cases.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _load_datasets(args)
    llm_extractor = _build_llm_extractor(args.llm_config) if args.llm_config else None
    llm_remaining = max(0, int(args.llm_limit))

    all_case_reports: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for dataset_name, dataset_kind, path, records in datasets:
        case_reports: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            report = _audit_record(dataset_name=dataset_name, dataset_kind=dataset_kind, raw=record)
            should_attempt_llm = (
                llm_extractor is not None
                and llm_remaining > 0
                and (args.llm_on == "all" or not report["covered_by_existing_or_auto"])
            )
            if should_attempt_llm:
                report = _augment_with_llm_extraction(report, record, llm_extractor)
                llm_remaining -= 1
            case_reports.append(report)
        all_case_reports.extend(case_reports)
        dataset_summaries[dataset_name] = _summarize_cases(dataset_name, dataset_kind, path, case_reports)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "extractor": ExplicitPointExpectedTraceExtractor.extractor_name,
        "llm_extractor": LLMExpectedArtifactExtractor.extractor_name if llm_extractor is not None else None,
        "llm_limit": int(args.llm_limit) if llm_extractor is not None else 0,
        "llm_on": args.llm_on if llm_extractor is not None else None,
        "datasets": dataset_summaries,
        "overall": _summarize_cases("overall", "combined", None, all_case_reports),
        "notes": [
            "Rule coverage counts only explicit x/y numeric sequence extraction from task text.",
            "Failure-analysis fields such as metadata.reason are intentionally excluded to avoid evaluation leakage.",
            "LLM extraction, when enabled, accepts only candidates with source_span grounded in the instruction text.",
        ],
    }

    _write_json(output_dir / "coverage_summary.json", summary)
    _write_jsonl(output_dir / "coverage_cases.jsonl", all_case_reports)

    print("Output:", output_dir)
    print("Rule extractor:", ExplicitPointExpectedTraceExtractor.extractor_name)
    if llm_extractor is not None:
        print("LLM extractor:", LLMExpectedArtifactExtractor.extractor_name, "limit=", args.llm_limit, "mode=", args.llm_on)
    for dataset_name, data in dataset_summaries.items():
        print(
            dataset_name,
            "total=",
            data["total_records"],
            "auto_extracted=",
            data["auto_extracted_records"],
            "auto_rate=",
            data["auto_extraction_rate"],
            "covered_existing_or_auto=",
            data["covered_by_existing_or_auto_records"],
            "llm_attempted=",
            data["llm_attempted_records"],
            "llm_artifact_records=",
            data["llm_artifact_records"],
            "llm_plot_trace_records=",
            data["llm_plot_trace_records"],
        )


def _load_datasets(args: argparse.Namespace) -> list[tuple[str, str, Path, list[dict[str, Any]]]]:
    datasets: list[tuple[str, str, Path, list[dict[str, Any]]]] = []
    bench_paths = [Path(item) for item in args.bench]
    if not args.no_default_benches:
        bench_paths.extend(path for path in DEFAULT_BENCHES if path.exists())
    for path in _dedupe_paths(bench_paths):
        records = _load_json_list(path)
        datasets.append((path.stem, "bench_json", path, records))

    instruction_path = Path(args.matplotbench_instructions) if args.matplotbench_instructions else None
    if instruction_path is not None and instruction_path.exists():
        records = _load_json_list(instruction_path)
        datasets.append(("matplotbench_instructions", "matplotbench_instructions", instruction_path, records))
    return datasets


def _build_llm_extractor(config_path: str) -> LLMExpectedArtifactExtractor:
    config = load_ablation_run_config(project_root / config_path)
    provider = config.parser_provider or config.repair_provider
    if provider is None:
        raise ValueError("--llm-config did not define llm.default, llm.parser, or llm.repair provider settings.")
    return LLMExpectedArtifactExtractor(OpenAICompatibleLLMClient(provider))


def _audit_record(*, dataset_name: str, dataset_kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    extractor = ExplicitPointExpectedTraceExtractor()
    sources = _source_texts(raw)
    extraction = None
    for source, text in sources:
        extraction = extractor.extract(text, default_chart_type=_default_chart_type(raw), source=source)
        if extraction is not None:
            break
    trace = extraction.trace if extraction is not None else None
    explicit_expected = _has_explicit_expected_trace(raw)
    return {
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "case_id": str(raw.get("case_id") or raw.get("id") or raw.get("native_id") or ""),
        "native_id": raw.get("native_id", raw.get("id")),
        "score": raw.get("score"),
        "has_explicit_expected_trace": explicit_expected,
        "auto_extracted": trace is not None,
        "covered_by_existing_or_auto": explicit_expected or trace is not None,
        "source_labels": [source for source, _ in sources],
        "extracted_source": _source_prefix(trace.source) if trace is not None else None,
        "chart_type": trace.chart_type if trace is not None else None,
        "point_count": len(trace.points) if trace is not None else 0,
        "points_preview": _points_preview(trace) if trace is not None else [],
        "matched_text": extraction.matched_text if extraction is not None else None,
        "confidence": extraction.confidence if extraction is not None else None,
        "extractor": extraction.extractor if extraction is not None else None,
        "llm_attempted": False,
        "llm_error_type": None,
        "llm_error_message": None,
        "llm_artifact_count": 0,
        "llm_plot_trace_count": 0,
        "llm_artifact_types": [],
        "llm_artifacts_preview": [],
        "llm_primary_trace_points_preview": [],
        "llm_figure_requirement_count": 0,
        "llm_figure_requirements_preview": None,
        "llm_total_tokens": None,
        "llm_model": None,
    }


def _augment_with_llm_extraction(
    report: dict[str, Any],
    raw: dict[str, Any],
    llm_extractor: LLMExpectedArtifactExtractor,
) -> dict[str, Any]:
    sources = _source_texts(raw)
    if not sources:
        return {**report, "llm_attempted": False}
    source, text = sources[0]
    try:
        result = llm_extractor.extract(text, source=source, case_id=str(raw.get("case_id") or raw.get("id") or ""))
    except Exception as exc:
        return {
            **report,
            "llm_attempted": True,
            "llm_error_type": type(exc).__name__,
            "llm_error_message": str(exc),
            "llm_artifact_count": 0,
            "llm_plot_trace_count": 0,
            "llm_artifact_types": [],
            "llm_artifacts_preview": [],
            "llm_figure_requirement_count": 0,
            "llm_figure_requirements_preview": None,
        }
    token_usage = None
    if result.llm_trace is not None and result.llm_trace.usage is not None:
        token_usage = result.llm_trace.usage.total_tokens
    return {
        **report,
        "llm_attempted": True,
        "llm_error_type": None,
        "llm_error_message": None,
        "llm_artifact_count": len(result.artifacts),
        "llm_plot_trace_count": len(result.plot_traces),
        "llm_artifact_types": [artifact.artifact_type for artifact in result.artifacts],
        "llm_artifacts_preview": _artifact_preview(result.artifacts),
        "llm_primary_trace_points_preview": _points_preview(result.primary_trace) if result.primary_trace is not None else [],
        "llm_figure_requirement_count": _figure_requirement_count(result.figure_requirements),
        "llm_figure_requirements_preview": _figure_requirements_preview(result.figure_requirements),
        "llm_total_tokens": token_usage,
        "llm_model": result.llm_trace.model if result.llm_trace is not None else None,
    }


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


def _default_chart_type(raw: dict[str, Any]) -> str:
    if raw.get("expected_chart_type"):
        return str(raw["expected_chart_type"])
    for key in ("chart_type", "plot_type"):
        if raw.get(key):
            return str(raw[key])
    return "unknown"


def _has_explicit_expected_trace(raw: dict[str, Any]) -> bool:
    return raw.get("expected_trace") is not None or raw.get("expected_points") is not None


def _summarize_cases(dataset_name: str, dataset_kind: str, path: Path | None, cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    explicit = sum(1 for case in cases if case["has_explicit_expected_trace"])
    auto = sum(1 for case in cases if case["auto_extracted"])
    covered = sum(1 for case in cases if case["covered_by_existing_or_auto"])
    source_counts = Counter(case["extracted_source"] for case in cases if case["extracted_source"])
    chart_type_counts = Counter(case["chart_type"] for case in cases if case["chart_type"])
    point_count_counts = Counter(str(case["point_count"]) for case in cases if case["point_count"])
    llm_artifact_type_counts = Counter(
        artifact_type
        for case in cases
        for artifact_type in case.get("llm_artifact_types", [])
    )
    return {
        "dataset_name": dataset_name,
        "dataset_kind": dataset_kind,
        "path": str(path) if path is not None else None,
        "total_records": total,
        "records_with_task_text": sum(1 for case in cases if case["source_labels"]),
        "explicit_expected_trace_records": explicit,
        "auto_extracted_records": auto,
        "covered_by_existing_or_auto_records": covered,
        "auto_extraction_rate": _rate(auto, total),
        "covered_by_existing_or_auto_rate": _rate(covered, total),
        "extracted_source_counts": dict(source_counts),
        "chart_type_counts": dict(chart_type_counts),
        "point_count_counts": dict(point_count_counts),
        "auto_extracted_case_ids": [case["case_id"] for case in cases if case["auto_extracted"]],
        "llm_attempted_records": sum(1 for case in cases if case.get("llm_attempted")),
        "llm_error_records": sum(1 for case in cases if case.get("llm_error_type")),
        "llm_artifact_records": sum(1 for case in cases if case.get("llm_artifact_count", 0) > 0),
        "llm_plot_trace_records": sum(1 for case in cases if case.get("llm_plot_trace_count", 0) > 0),
        "llm_figure_requirement_records": sum(1 for case in cases if case.get("llm_figure_requirement_count", 0) > 0),
        "llm_figure_requirement_total": sum(case.get("llm_figure_requirement_count") or 0 for case in cases),
        "llm_artifact_type_counts": dict(llm_artifact_type_counts),
        "llm_total_tokens": sum(case.get("llm_total_tokens") or 0 for case in cases),
    }




def _artifact_preview(artifacts, *, limit: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "artifact_type": artifact.artifact_type,
            "panel_id": artifact.panel_id,
            "chart_type": artifact.chart_type,
            "role": getattr(artifact, "role", None),
            "source_span": artifact.source_span,
            "confidence": artifact.confidence,
            "value": artifact.value,
        }
        for artifact in artifacts[:limit]
    ]


def _figure_requirement_count(figure) -> int:
    if figure is None:
        return 0
    count = 0
    for value in (figure.axes_count, figure.figure_title, figure.size_inches):
        if value is not None:
            count += 1
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


def _figure_requirements_preview(figure) -> dict[str, Any] | None:
    if figure is None:
        return None
    return {
        "axes_count": figure.axes_count,
        "figure_title": figure.figure_title,
        "size_inches": list(figure.size_inches) if figure.size_inches else None,
        "source_spans": dict(getattr(figure, "source_spans", {})),
        "axes": [
            {
                "axis_index": axis.axis_index,
                "title": axis.title,
                "xlabel": axis.xlabel,
                "ylabel": axis.ylabel,
                "zlabel": axis.zlabel,
                "projection": axis.projection,
                "xscale": axis.xscale,
                "yscale": axis.yscale,
                "zscale": axis.zscale,
                "legend_labels": list(axis.legend_labels),
                "text_contains": list(axis.text_contains),
                "artist_types": list(axis.artist_types),
                "source_spans": dict(getattr(axis, "source_spans", {})),
            }
            for axis in figure.axes
        ],
    }


def _points_preview(trace: PlotTrace, *, limit: int = 8) -> list[dict[str, Any]]:
    return [
        {"x": point.x, "y": point.y}
        for point in trace.points[:limit]
    ]


def _source_prefix(source: str) -> str:
    return source.split(":", 1)[0]


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 4)


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        resolved = _resolve_path(path)
        key = str(resolved).lower()
        if key not in seen:
            result.append(resolved)
            seen.add(key)
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
