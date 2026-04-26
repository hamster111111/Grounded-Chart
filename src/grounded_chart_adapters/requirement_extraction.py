from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from grounded_chart import IntentParser, build_requirement_plan
from grounded_chart.schema import ParsedRequirementBundle, TableSchema

ExtractionStatus = Literal["ok", "error"]


@dataclass(frozen=True)
class RequirementExtractionCaseReport:
    case_id: str
    query: str
    status: ExtractionStatus
    plan: dict[str, Any]
    requirement_plan: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    exception_type: str | None = None
    exception_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "status": self.status,
            "plan": _jsonable(self.plan),
            "requirement_plan": _jsonable(self.requirement_plan),
            "raw_response": _jsonable(self.raw_response),
            "summary": _jsonable(self.summary),
            "metadata": _jsonable(self.metadata),
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


@dataclass(frozen=True)
class RequirementExtractionSummary:
    total_cases: int
    ok_cases: int
    errored_cases: int
    total_requirements: int
    avg_requirements_per_case: float
    status_counts: dict[str, int] = field(default_factory=dict)
    scope_counts: dict[str, int] = field(default_factory=dict)
    chart_type_counts: dict[str, int] = field(default_factory=dict)
    explicit_without_span: int = 0
    grounded_spans: int = 0
    cases_with_unknown_chart_type: int = 0
    cases_with_only_secondary_requirements: int = 0
    exception_counts: dict[str, int] = field(default_factory=dict)
    top_requirement_names: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "ok_cases": self.ok_cases,
            "errored_cases": self.errored_cases,
            "total_requirements": self.total_requirements,
            "avg_requirements_per_case": self.avg_requirements_per_case,
            "status_counts": dict(self.status_counts),
            "scope_counts": dict(self.scope_counts),
            "chart_type_counts": dict(self.chart_type_counts),
            "explicit_without_span": self.explicit_without_span,
            "grounded_spans": self.grounded_spans,
            "cases_with_unknown_chart_type": self.cases_with_unknown_chart_type,
            "cases_with_only_secondary_requirements": self.cases_with_only_secondary_requirements,
            "exception_counts": dict(self.exception_counts),
            "top_requirement_names": dict(self.top_requirement_names),
        }


@dataclass(frozen=True)
class RequirementExtractionReport:
    cases: tuple[RequirementExtractionCaseReport, ...]
    summary: RequirementExtractionSummary

    @classmethod
    def from_case_reports(
        cls,
        cases: list[RequirementExtractionCaseReport] | tuple[RequirementExtractionCaseReport, ...],
    ) -> "RequirementExtractionReport":
        normalized = tuple(cases)
        return cls(cases=normalized, summary=summarize_extraction_case_reports(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def write_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(case.to_dict(), ensure_ascii=False) for case in self.cases]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class RequirementExtractionRunner:
    """Run parser-native requirement extraction over native benchmark prompts."""

    def __init__(
        self,
        parser: IntentParser,
        *,
        default_schema: TableSchema | None = None,
        prefer_expert_instruction: bool = True,
        continue_on_error: bool = True,
    ) -> None:
        self.parser = parser
        self.default_schema = default_schema or TableSchema(columns={})
        self.prefer_expert_instruction = prefer_expert_instruction
        self.continue_on_error = continue_on_error

    def run(self, records: Iterable[Any]) -> RequirementExtractionReport:
        case_reports: list[RequirementExtractionCaseReport] = []
        for record in records:
            try:
                case_reports.append(self._run_record(record))
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                case_reports.append(self._error_report(record, exc))
        return RequirementExtractionReport.from_case_reports(case_reports)

    def _run_record(self, record: Any) -> RequirementExtractionCaseReport:
        query = _resolve_query(record, prefer_expert=self.prefer_expert_instruction)
        schema = _resolve_schema(record, default=self.default_schema)
        bundle = self._parse_requirements(query, schema)
        return RequirementExtractionCaseReport(
            case_id=_resolve_case_id(record),
            query=query,
            status="ok",
            plan=_plan_to_dict(bundle.plan),
            requirement_plan=_requirement_plan_to_dict(bundle.requirement_plan),
            raw_response=dict(bundle.raw_response),
            summary=_summarize_requirement_plan(query, bundle.requirement_plan),
            metadata=_resolve_metadata(record),
        )

    def _error_report(self, record: Any, exc: BaseException) -> RequirementExtractionCaseReport:
        query = _resolve_query(record, prefer_expert=self.prefer_expert_instruction)
        return RequirementExtractionCaseReport(
            case_id=_resolve_case_id(record),
            query=query,
            status="error",
            plan={
                "chart_type": "unknown",
                "dimensions": [],
                "measure": {"column": None, "agg": "none"},
                "filters": [],
                "sort": None,
                "limit": None,
                "confidence": None,
            },
            requirement_plan=None,
            raw_response=None,
            summary={
                "requirement_count": 0,
                "status_counts": {"error": 1},
                "scope_counts": {},
                "explicit_without_span": 0,
                "grounded_span_count": 0,
            },
            metadata=_resolve_metadata(record),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )

    def _parse_requirements(self, query: str, schema: TableSchema) -> ParsedRequirementBundle:
        parse_requirements = getattr(self.parser, "parse_requirements", None)
        if callable(parse_requirements):
            return parse_requirements(query, schema)
        plan = self.parser.parse(query, schema)
        return ParsedRequirementBundle(plan=plan, requirement_plan=build_requirement_plan(plan), raw_response={})


def summarize_extraction_case_reports(
    cases: tuple[RequirementExtractionCaseReport, ...],
) -> RequirementExtractionSummary:
    status_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    chart_type_counts: Counter[str] = Counter()
    exception_counts: Counter[str] = Counter()
    total_requirements = 0
    explicit_without_span = 0
    grounded_spans = 0
    ok_cases = 0
    cases_with_unknown_chart_type = 0
    cases_with_only_secondary_requirements = 0

    for case in cases:
        total_requirements += int(case.summary.get("requirement_count", 0))
        explicit_without_span += int(case.summary.get("explicit_without_span", 0))
        grounded_spans += int(case.summary.get("grounded_span_count", 0))
        status_counts.update(case.summary.get("status_counts", {}))
        scope_counts.update(case.summary.get("scope_counts", {}))
        chart_type = str(case.plan.get("chart_type") or "unknown")
        chart_type_counts.update([chart_type])
        if chart_type == "unknown":
            cases_with_unknown_chart_type += 1
        if case.status == "ok":
            ok_cases += 1
        if case.exception_type:
            exception_counts.update([case.exception_type])
        requirements = (case.requirement_plan or {}).get("requirements", [])
        core_names = {requirement["name"] for requirement in requirements if requirement.get("priority") == "core"}
        if case.status == "ok" and not core_names:
            cases_with_only_secondary_requirements += 1
        for requirement in requirements:
            name = requirement.get("name")
            if name:
                name_counts.update([str(name)])

    total_cases = len(cases)
    return RequirementExtractionSummary(
        total_cases=total_cases,
        ok_cases=ok_cases,
        errored_cases=total_cases - ok_cases,
        total_requirements=total_requirements,
        avg_requirements_per_case=round(total_requirements / total_cases, 2) if total_cases else 0.0,
        status_counts=dict(status_counts),
        scope_counts=dict(scope_counts),
        chart_type_counts=dict(chart_type_counts),
        explicit_without_span=explicit_without_span,
        grounded_spans=grounded_spans,
        cases_with_unknown_chart_type=cases_with_unknown_chart_type,
        cases_with_only_secondary_requirements=cases_with_only_secondary_requirements,
        exception_counts=dict(exception_counts),
        top_requirement_names=dict(name_counts.most_common(25)),
    )


def _resolve_case_id(record: Any) -> str:
    if isinstance(record, dict):
        for key in ("case_id", "id", "native_id"):
            value = record.get(key)
            if value is not None:
                return str(value)
    value = getattr(record, "case_id", None)
    if value is not None:
        return str(value)
    raise KeyError("Could not resolve case_id from requirement extraction record.")


def _resolve_query(record: Any, *, prefer_expert: bool) -> str:
    preferred_instruction = getattr(record, "preferred_instruction", None)
    if callable(preferred_instruction):
        return str(preferred_instruction(prefer_expert=prefer_expert))
    if isinstance(record, dict):
        if prefer_expert and record.get("expert_instruction"):
            return str(record["expert_instruction"])
        for key in ("simple_instruction", "query", "instruction"):
            value = record.get(key)
            if value:
                return str(value)
    for attr in ("query", "instruction", "simple_instruction", "expert_instruction"):
        value = getattr(record, attr, None)
        if value:
            return str(value)
    return ""


def _resolve_schema(record: Any, *, default: TableSchema) -> TableSchema:
    if isinstance(record, dict):
        schema = record.get("schema")
        if isinstance(schema, TableSchema):
            return schema
        if isinstance(schema, dict):
            columns = schema.get("columns")
            if isinstance(columns, dict):
                return TableSchema(columns={str(key): str(value) for key, value in columns.items()})
    schema = getattr(record, "schema", None)
    if isinstance(schema, TableSchema):
        return schema
    return default


def _resolve_metadata(record: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(record, dict):
        existing = record.get("metadata")
        if isinstance(existing, dict):
            metadata.update(existing)
        for key in ("native_id", "score", "simple_instruction", "expert_instruction"):
            if key in record and record[key] is not None:
                metadata.setdefault(key, record[key])
        return metadata
    existing = getattr(record, "metadata", None)
    if isinstance(existing, dict):
        metadata.update(existing)
    for attr in ("native_id", "score"):
        value = getattr(record, attr, None)
        if value is not None:
            metadata.setdefault(attr, value)
    return metadata


def _summarize_requirement_plan(query: str, requirement_plan) -> dict[str, Any]:
    requirements = tuple(getattr(requirement_plan, "requirements", ()))
    status_counts = Counter(requirement.status for requirement in requirements)
    scope_counts = Counter(requirement.scope for requirement in requirements)
    explicit_without_span = sum(1 for requirement in requirements if requirement.status == "explicit" and not requirement.source_span)
    grounded_span_count = sum(
        1
        for requirement in requirements
        if requirement.source_span and requirement.source_span.lower() in query.lower()
    )
    return {
        "requirement_count": len(requirements),
        "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts),
        "explicit_without_span": explicit_without_span,
        "grounded_span_count": grounded_span_count,
    }


def _plan_to_dict(plan) -> dict[str, Any]:
    return {
        "chart_type": plan.chart_type,
        "dimensions": list(plan.dimensions),
        "measure": {"column": plan.measure.column, "agg": plan.measure.agg},
        "filters": [
            {"column": filter_spec.column, "op": filter_spec.op, "value": _jsonable(filter_spec.value)}
            for filter_spec in plan.filters
        ],
        "sort": {"by": plan.sort.by, "direction": plan.sort.direction} if plan.sort is not None else None,
        "limit": plan.limit,
        "confidence": plan.confidence,
    }


def _requirement_plan_to_dict(plan) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "raw_query": plan.raw_query,
        "shared_requirement_ids": list(plan.shared_requirement_ids),
        "figure_requirements": _jsonable(plan.figure_requirements),
        "requirements": [
            {
                "requirement_id": requirement.requirement_id,
                "scope": requirement.scope,
                "type": requirement.type,
                "name": requirement.name,
                "value": _jsonable(requirement.value),
                "source_span": requirement.source_span,
                "status": requirement.status,
                "confidence": requirement.confidence,
                "depends_on": list(requirement.depends_on),
                "priority": requirement.priority,
                "panel_id": requirement.panel_id,
                "assumption": requirement.assumption,
                "severity": requirement.severity,
                "match_policy": requirement.match_policy,
                "is_verifiable": requirement.is_verifiable,
            }
            for requirement in plan.requirements
        ],
        "panels": [
            {
                "panel_id": panel.panel_id,
                "chart_type": panel.chart_type,
                "requirement_ids": list(panel.requirement_ids),
                "data_ops": _jsonable(panel.data_ops),
                "encodings": _jsonable(panel.encodings),
                "annotations": _jsonable(panel.annotations),
                "presentation_constraints": _jsonable(panel.presentation_constraints),
            }
            for panel in plan.panels
        ],
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)
