from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from grounded_chart.api import HeuristicIntentParser, TableSchema
from grounded_chart.verification.intent_parser import _build_llm_requirement_bundle
from grounded_chart_adapters import RequirementExtractionReport, RequirementExtractionRunner


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    bench_path = project_root / args.bench
    llm_cases_path = project_root / args.llm_cases
    output_dir = project_root / "outputs" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_records = json.loads(bench_path.read_text(encoding="utf-8"))
    llm_raw_cases = json.loads(llm_cases_path.read_text(encoding="utf-8"))

    llm_report = rebuild_llm_report(llm_raw_cases)
    heuristic_runner = RequirementExtractionRunner(HeuristicIntentParser())
    heuristic_report = heuristic_runner.run(bench_records)
    compare = build_compare(llm_report, heuristic_report, bench_records)

    llm_report.write_json(output_dir / "llm_rebuilt_report.json")
    llm_report.write_jsonl(output_dir / "llm_rebuilt_cases.jsonl")
    heuristic_report.write_json(output_dir / "heuristic_report.json")
    heuristic_report.write_jsonl(output_dir / "heuristic_cases.jsonl")
    (output_dir / "compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "llm_report": str(output_dir / "llm_rebuilt_report.json"),
                "heuristic_report": str(output_dir / "heuristic_report.json"),
                "compare": str(output_dir / "compare.json"),
                "llm_total_cases": llm_report.summary.total_cases,
                "heuristic_total_cases": heuristic_report.summary.total_cases,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heuristic vs LLM requirement extraction on MatPlotBench failed native cases.")
    parser.add_argument("--bench", default="benchmarks/matplotbench_ds_failed_native.json")
    parser.add_argument("--llm-cases", default="outputs/matplotbench_requirement_extraction/full_run/cases.json")
    parser.add_argument("--output-dir", default="matplotbench_requirement_extraction/parser_compare")
    return parser.parse_args()


def rebuild_llm_report(raw_cases: list[dict[str, Any]]) -> RequirementExtractionReport:
    case_reports = []
    for case in raw_cases:
        query = str(case.get("query") or case.get("simple_instruction") or case.get("expert_instruction") or "")
        payload = case.get("raw_response") or {}
        bundle = _build_llm_requirement_bundle(query, TableSchema(columns={}), payload, default_confidence=0.7)
        requirement_plan = requirement_plan_to_dict(bundle.requirement_plan)
        summary = summarize_requirement_plan(query, requirement_plan)
        case_reports.append(
            {
                "case_id": case["case_id"],
                "query": query,
                "status": "ok",
                "plan": plan_to_dict(bundle.plan),
                "requirement_plan": requirement_plan,
                "raw_response": payload,
                "summary": summary,
                "metadata": {
                    "native_id": case.get("native_id"),
                    "score": case.get("score"),
                },
                "exception_type": None,
                "exception_message": None,
            }
        )
    return RequirementExtractionReport.from_case_reports(tuple(dict_to_case_report(case) for case in case_reports))


def build_compare(
    llm_report: RequirementExtractionReport,
    heuristic_report: RequirementExtractionReport,
    bench_records: list[dict[str, Any]],
) -> dict[str, Any]:
    llm_by_case = {case.case_id: case.to_dict() for case in llm_report.cases}
    heuristic_by_case = {case.case_id: case.to_dict() for case in heuristic_report.cases}
    native_ids = {record["case_id"]: record.get("native_id") for record in bench_records}
    scores = {record["case_id"]: record.get("score") for record in bench_records}

    case_comparisons: list[dict[str, Any]] = []
    llm_name_counts: Counter[str] = Counter()
    heuristic_name_counts: Counter[str] = Counter()
    for case_id in sorted(set(llm_by_case) | set(heuristic_by_case), key=lambda item: int(native_ids.get(item) or 0)):
        llm_case = llm_by_case.get(case_id)
        heuristic_case = heuristic_by_case.get(case_id)
        llm_requirements = (llm_case or {}).get("requirement_plan", {}).get("requirements", [])
        heuristic_requirements = (heuristic_case or {}).get("requirement_plan", {}).get("requirements", [])
        llm_names = {requirement["name"] for requirement in llm_requirements}
        heuristic_names = {requirement["name"] for requirement in heuristic_requirements}
        llm_name_counts.update(llm_names)
        heuristic_name_counts.update(heuristic_names)
        case_comparisons.append(
            {
                "case_id": case_id,
                "native_id": native_ids.get(case_id),
                "score": scores.get(case_id),
                "llm": case_snapshot(llm_case),
                "heuristic": case_snapshot(heuristic_case),
                "delta": {
                    "requirement_count": ((llm_case or {}).get("summary", {}).get("requirement_count", 0))
                    - ((heuristic_case or {}).get("summary", {}).get("requirement_count", 0)),
                    "grounded_span_count": ((llm_case or {}).get("summary", {}).get("grounded_span_count", 0))
                    - ((heuristic_case or {}).get("summary", {}).get("grounded_span_count", 0)),
                    "core_requirement_count": count_core_requirements(llm_requirements) - count_core_requirements(heuristic_requirements),
                },
                "llm_only_names": sorted(llm_names - heuristic_names),
                "heuristic_only_names": sorted(heuristic_names - llm_names),
            }
        )

    case_comparisons.sort(key=lambda item: (item["native_id"] or 0))
    largest_requirement_gains = sorted(case_comparisons, key=lambda item: item["delta"]["requirement_count"], reverse=True)[:10]
    largest_core_gains = sorted(case_comparisons, key=lambda item: item["delta"]["core_requirement_count"], reverse=True)[:10]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "llm_summary": llm_report.summary.to_dict(),
        "heuristic_summary": heuristic_report.summary.to_dict(),
        "delta": diff_summary(heuristic_report.summary.to_dict(), llm_report.summary.to_dict()),
        "name_coverage": {
            "llm_top_requirement_names": dict(llm_name_counts.most_common(30)),
            "heuristic_top_requirement_names": dict(heuristic_name_counts.most_common(30)),
        },
        "largest_requirement_gains": largest_requirement_gains,
        "largest_core_gains": largest_core_gains,
        "cases": case_comparisons,
    }


def case_snapshot(case: dict[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    requirements = (case.get("requirement_plan") or {}).get("requirements", [])
    return {
        "status": case.get("status"),
        "chart_type": (case.get("plan") or {}).get("chart_type"),
        "requirement_count": (case.get("summary") or {}).get("requirement_count", 0),
        "grounded_span_count": (case.get("summary") or {}).get("grounded_span_count", 0),
        "core_requirement_count": count_core_requirements(requirements),
        "unknown_chart_type": (case.get("plan") or {}).get("chart_type") == "unknown",
    }


def count_core_requirements(requirements: list[dict[str, Any]]) -> int:
    return sum(1 for requirement in requirements if requirement.get("priority") == "core")


def diff_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_requirements": int(after["total_requirements"]) - int(before["total_requirements"]),
        "avg_requirements_per_case": round(float(after["avg_requirements_per_case"]) - float(before["avg_requirements_per_case"]), 2),
        "cases_with_unknown_chart_type": int(after["cases_with_unknown_chart_type"]) - int(before["cases_with_unknown_chart_type"]),
        "cases_with_only_secondary_requirements": int(after["cases_with_only_secondary_requirements"]) - int(before["cases_with_only_secondary_requirements"]),
        "explicit_without_span": int(after["explicit_without_span"]) - int(before["explicit_without_span"]),
        "grounded_spans": int(after["grounded_spans"]) - int(before["grounded_spans"]),
        "status_counts": counter_delta(before.get("status_counts", {}), after.get("status_counts", {})),
        "scope_counts": counter_delta(before.get("scope_counts", {}), after.get("scope_counts", {})),
        "chart_type_counts": counter_delta(before.get("chart_type_counts", {}), after.get("chart_type_counts", {})),
        "exception_counts": counter_delta(before.get("exception_counts", {}), after.get("exception_counts", {})),
        "top_requirement_names": counter_delta(before.get("top_requirement_names", {}), after.get("top_requirement_names", {})),
    }


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(before) | set(after))
    delta: dict[str, Any] = {}
    for key in keys:
        change = int(after.get(key, 0)) - int(before.get(key, 0))
        if change != 0:
            delta[key] = change
    return delta


def summarize_requirement_plan(query: str, requirement_plan: dict[str, Any] | None) -> dict[str, Any]:
    requirements = [] if requirement_plan is None else requirement_plan.get("requirements", [])
    status_counts = Counter(str(requirement.get("status")) for requirement in requirements)
    scope_counts = Counter(str(requirement.get("scope")) for requirement in requirements)
    explicit_without_span = sum(
        1 for requirement in requirements if requirement.get("status") == "explicit" and not requirement.get("source_span")
    )
    grounded_span_count = sum(
        1
        for requirement in requirements
        if requirement.get("source_span") and str(requirement.get("source_span")).lower() in query.lower()
    )
    return {
        "requirement_count": len(requirements),
        "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts),
        "explicit_without_span": explicit_without_span,
        "grounded_span_count": grounded_span_count,
    }


def plan_to_dict(plan) -> dict[str, Any]:
    return {
        "chart_type": plan.chart_type,
        "dimensions": list(plan.dimensions),
        "measure": {"column": plan.measure.column, "agg": plan.measure.agg},
        "filters": [
            {"column": filter_spec.column, "op": filter_spec.op, "value": jsonable(filter_spec.value)}
            for filter_spec in plan.filters
        ],
        "sort": {"by": plan.sort.by, "direction": plan.sort.direction} if plan.sort is not None else None,
        "limit": plan.limit,
        "confidence": plan.confidence,
    }


def requirement_plan_to_dict(plan) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "raw_query": plan.raw_query,
        "shared_requirement_ids": list(plan.shared_requirement_ids),
        "figure_requirements": jsonable(plan.figure_requirements),
        "requirements": [
            {
                "requirement_id": requirement.requirement_id,
                "scope": requirement.scope,
                "type": requirement.type,
                "name": requirement.name,
                "value": jsonable(requirement.value),
                "source_span": requirement.source_span,
                "status": requirement.status,
                "confidence": requirement.confidence,
                "depends_on": list(requirement.depends_on),
                "priority": requirement.priority,
                "panel_id": requirement.panel_id,
                "assumption": requirement.assumption,
                "is_verifiable": requirement.is_verifiable,
            }
            for requirement in plan.requirements
        ],
        "panels": [
            {
                "panel_id": panel.panel_id,
                "chart_type": panel.chart_type,
                "requirement_ids": list(panel.requirement_ids),
                "data_ops": jsonable(panel.data_ops),
                "encodings": jsonable(panel.encodings),
                "annotations": jsonable(panel.annotations),
                "presentation_constraints": jsonable(panel.presentation_constraints),
            }
            for panel in plan.panels
        ],
    }


def dict_to_case_report(payload: dict[str, Any]):
    from grounded_chart_adapters.requirement_extraction import RequirementExtractionCaseReport

    return RequirementExtractionCaseReport(
        case_id=payload["case_id"],
        query=payload["query"],
        status=payload["status"],
        plan=payload["plan"],
        requirement_plan=payload["requirement_plan"],
        raw_response=payload["raw_response"],
        summary=payload["summary"],
        metadata=payload["metadata"],
        exception_type=payload["exception_type"],
        exception_message=payload["exception_message"],
    )


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return str(value)


if __name__ == "__main__":
    main()
