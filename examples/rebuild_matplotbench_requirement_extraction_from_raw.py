from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from grounded_chart.intent_parser import _build_llm_requirement_bundle
from grounded_chart.schema import TableSchema


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.input
    output_dir = project_root / "outputs" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads(input_path.read_text(encoding="utf-8"))
    rebuilt_cases = [rebuild_case(case) for case in cases]
    rebuilt_cases.sort(key=lambda item: int(item.get("native_id") or 0))

    cases_path = output_dir / "cases.json"
    summary_path = output_dir / "summary.json"
    compare_path = output_dir / "compare.json"

    summary = build_summary(rebuilt_cases)
    previous_summary = build_summary(cases)
    compare = {"before": previous_summary, "after": summary, "delta": diff_summary(previous_summary, summary)}

    cases_path.write_text(json.dumps(rebuilt_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "cases": str(cases_path),
                "summary": str(summary_path),
                "compare": str(compare_path),
                "total_cases": len(rebuilt_cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild MatPlotBench requirement extraction from saved raw LLM responses.")
    parser.add_argument("--input", default="outputs/matplotbench_requirement_extraction/full_run/cases.json")
    parser.add_argument("--output-dir", default="matplotbench_requirement_extraction/full_run_rebuilt")
    return parser.parse_args()


def rebuild_case(case: dict[str, Any]) -> dict[str, Any]:
    query = str(case.get("expert_instruction") or case.get("simple_instruction") or "")
    payload = case.get("raw_response") or {}
    bundle = _build_llm_requirement_bundle(query, TableSchema(columns={}), payload, default_confidence=0.7)
    return {
        "case_id": case["case_id"],
        "native_id": case.get("native_id"),
        "score": case.get("score"),
        "simple_instruction": case.get("simple_instruction"),
        "expert_instruction": case.get("expert_instruction"),
        "plan": plan_to_dict(bundle.plan),
        "requirements": [requirement_to_dict(req) for req in bundle.requirement_plan.requirements],
        "raw_response": payload,
        "summary": summarize_case(case, bundle.requirement_plan.requirements),
    }


def summarize_case(record: dict[str, Any], requirements: list[Any]) -> dict[str, Any]:
    expert_instruction = str(record.get("expert_instruction") or "")
    status_counts = Counter(requirement.status for requirement in requirements)
    scope_counts = Counter(requirement.scope for requirement in requirements)
    explicit_without_span = sum(1 for requirement in requirements if requirement.status == "explicit" and not requirement.source_span)
    grounded_span_count = sum(
        1
        for requirement in requirements
        if requirement.source_span and requirement.source_span.lower() in expert_instruction.lower()
    )
    return {
        "requirement_count": len(requirements),
        "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts),
        "explicit_without_span": explicit_without_span,
        "grounded_span_count": grounded_span_count,
    }


def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    chart_type_counts: Counter[str] = Counter()
    total_requirements = 0
    explicit_without_span = 0
    grounded_spans = 0
    cases_with_unknown_chart_type = 0
    cases_with_only_secondary_requirements = 0

    for case in cases:
        summary = case["summary"]
        total_requirements += summary["requirement_count"]
        explicit_without_span += summary["explicit_without_span"]
        grounded_spans += summary["grounded_span_count"]
        status_counts.update(summary["status_counts"])
        scope_counts.update(summary["scope_counts"])
        chart_type_counts.update([case["plan"]["chart_type"]])
        if case["plan"]["chart_type"] == "unknown":
            cases_with_unknown_chart_type += 1
        core_names = {requirement["name"] for requirement in case["requirements"] if requirement["priority"] == "core"}
        if not core_names:
            cases_with_only_secondary_requirements += 1
        for requirement in case["requirements"]:
            name_counts.update([requirement["name"]])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_cases": len(cases),
        "total_requirements": total_requirements,
        "avg_requirements_per_case": round(total_requirements / len(cases), 2) if cases else 0.0,
        "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts),
        "chart_type_counts": dict(chart_type_counts),
        "explicit_without_span": explicit_without_span,
        "grounded_spans": grounded_spans,
        "cases_with_unknown_chart_type": cases_with_unknown_chart_type,
        "cases_with_only_secondary_requirements": cases_with_only_secondary_requirements,
        "top_requirement_names": dict(name_counts.most_common(25)),
    }


def diff_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_requirements": after["total_requirements"] - before["total_requirements"],
        "avg_requirements_per_case": round(after["avg_requirements_per_case"] - before["avg_requirements_per_case"], 2),
        "cases_with_unknown_chart_type": after["cases_with_unknown_chart_type"] - before["cases_with_unknown_chart_type"],
        "cases_with_only_secondary_requirements": after["cases_with_only_secondary_requirements"] - before["cases_with_only_secondary_requirements"],
        "status_counts": counter_delta(before["status_counts"], after["status_counts"]),
        "scope_counts": counter_delta(before["scope_counts"], after["scope_counts"]),
        "chart_type_counts": counter_delta(before["chart_type_counts"], after["chart_type_counts"]),
        "top_requirement_names": counter_delta(before["top_requirement_names"], after["top_requirement_names"]),
    }


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(before) | set(after))
    delta: dict[str, Any] = {}
    for key in keys:
        change = int(after.get(key, 0)) - int(before.get(key, 0))
        if change != 0:
            delta[key] = change
    return delta


def plan_to_dict(plan) -> dict[str, Any]:
    return {
        "chart_type": plan.chart_type,
        "dimensions": list(plan.dimensions),
        "measure": {"column": plan.measure.column, "agg": plan.measure.agg},
        "filters": [
            {"column": filter_spec.column, "op": filter_spec.op, "value": filter_spec.value}
            for filter_spec in plan.filters
        ],
        "sort": {"by": plan.sort.by, "direction": plan.sort.direction} if plan.sort is not None else None,
        "limit": plan.limit,
        "confidence": plan.confidence,
    }


def requirement_to_dict(requirement) -> dict[str, Any]:
    return {
        "requirement_id": requirement.requirement_id,
        "scope": requirement.scope,
        "type": requirement.type,
        "name": requirement.name,
        "value": requirement.value,
        "source_span": requirement.source_span,
        "status": requirement.status,
        "confidence": requirement.confidence,
        "depends_on": list(requirement.depends_on),
        "priority": requirement.priority,
        "panel_id": requirement.panel_id,
        "assumption": requirement.assumption,
    }


if __name__ == "__main__":
    main()
