from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from grounded_chart import infer_backend_profile
from grounded_chart.schema import DataPoint, PipelineResult, SortSpec, VerificationError
from grounded_chart_adapters.base import AdapterRunResult, ChartCase

CaseStatus = Literal["passed", "failed", "error"]


@dataclass(frozen=True)
class CaseReport:
    """JSON-serializable benchmark-facing report for one chart case."""

    case_id: str
    status: CaseStatus
    ok: bool
    query: str
    expected_chart_type: str | None = None
    actual_chart_type: str | None = None
    backend_profile: dict[str, Any] | None = None
    expected_points: tuple[dict[str, Any], ...] = ()
    actual_points: tuple[dict[str, Any], ...] = ()
    figure_requirements: dict[str, Any] | None = None
    actual_figure: dict[str, Any] | None = None
    intent: dict[str, Any] = field(default_factory=dict)
    requirement_plan: dict[str, Any] | None = None
    evidence_graph: dict[str, Any] | None = None
    error_codes: tuple[str, ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    repair_level: int | None = None
    repair_scope: str | None = None
    repair_strategy: str | None = None
    repair_instruction: str | None = None
    repaired_code: str | None = None
    repair_attempts: tuple[dict[str, Any], ...] = ()
    exception_type: str | None = None
    exception_message: str | None = None
    case_metadata: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "ok": self.ok,
            "query": self.query,
            "expected_chart_type": self.expected_chart_type,
            "actual_chart_type": self.actual_chart_type,
            "backend_profile": _jsonable(self.backend_profile),
            "expected_points": list(self.expected_points),
            "actual_points": list(self.actual_points),
            "figure_requirements": _jsonable(self.figure_requirements),
            "actual_figure": _jsonable(self.actual_figure),
            "intent": _jsonable(self.intent),
            "requirement_plan": _jsonable(self.requirement_plan),
            "evidence_graph": _jsonable(self.evidence_graph),
            "error_codes": list(self.error_codes),
            "errors": list(self.errors),
            "repair_level": self.repair_level,
            "repair_scope": self.repair_scope,
            "repair_strategy": self.repair_strategy,
            "repair_instruction": self.repair_instruction,
            "repaired_code": self.repaired_code,
            "repair_attempts": list(self.repair_attempts),
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "case_metadata": _jsonable(self.case_metadata),
            "run_metadata": _jsonable(self.run_metadata),
        }


@dataclass(frozen=True)
class BatchSummary:
    """Aggregate metrics for benchmark probes.

    `pass_rate` is computed over completed cases. `overall_pass_rate` is
    computed over all requested cases, so execution errors are penalized.
    """

    total_cases: int
    completed_cases: int
    passed_cases: int
    failed_cases: int
    errored_cases: int
    pass_rate: float
    overall_pass_rate: float
    completion_rate: float
    error_counts: dict[str, int] = field(default_factory=dict)
    operator_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    repair_level_counts: dict[str, int] = field(default_factory=dict)
    repair_scope_counts: dict[str, int] = field(default_factory=dict)
    expected_chart_type_counts: dict[str, int] = field(default_factory=dict)
    actual_chart_type_counts: dict[str, int] = field(default_factory=dict)
    backend_name_counts: dict[str, int] = field(default_factory=dict)
    backend_support_tier_counts: dict[str, int] = field(default_factory=dict)
    backend_verification_mode_counts: dict[str, int] = field(default_factory=dict)
    exception_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "errored_cases": self.errored_cases,
            "pass_rate": self.pass_rate,
            "overall_pass_rate": self.overall_pass_rate,
            "completion_rate": self.completion_rate,
            "error_counts": dict(self.error_counts),
            "operator_counts": dict(self.operator_counts),
            "severity_counts": dict(self.severity_counts),
            "repair_level_counts": dict(self.repair_level_counts),
            "repair_scope_counts": dict(self.repair_scope_counts),
            "expected_chart_type_counts": dict(self.expected_chart_type_counts),
            "actual_chart_type_counts": dict(self.actual_chart_type_counts),
            "backend_name_counts": dict(self.backend_name_counts),
            "backend_support_tier_counts": dict(self.backend_support_tier_counts),
            "backend_verification_mode_counts": dict(self.backend_verification_mode_counts),
            "exception_counts": dict(self.exception_counts),
        }


@dataclass(frozen=True)
class BatchReport:
    """Case reports plus aggregate metrics for one adapter run."""

    cases: tuple[CaseReport, ...]
    summary: BatchSummary

    @classmethod
    def from_case_reports(cls, cases: list[CaseReport] | tuple[CaseReport, ...]) -> "BatchReport":
        normalized = tuple(cases)
        return cls(cases=normalized, summary=summarize_case_reports(normalized))

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


def case_report_from_result(result: AdapterRunResult) -> CaseReport:
    pipeline_result = result.pipeline_result
    verification = pipeline_result.report
    status: CaseStatus = "passed" if verification.ok else "failed"
    repair_level = 0 if verification.ok else None
    repair_scope = "none" if verification.ok else None
    if pipeline_result.repair_plan is not None:
        repair_level = int(pipeline_result.repair_plan.repair_level)
        repair_scope = pipeline_result.repair_plan.scope
    backend_profile = infer_backend_profile(
        actual_trace=pipeline_result.actual_trace,
        actual_figure=pipeline_result.actual_figure,
        generated_code=result.case.generated_code,
    )
    return CaseReport(
        case_id=result.case.case_id,
        status=status,
        ok=verification.ok,
        query=result.case.query,
        expected_chart_type=pipeline_result.expected_trace.chart_type,
        actual_chart_type=pipeline_result.actual_trace.chart_type,
        backend_profile=_backend_profile_to_dict(backend_profile),
        expected_points=tuple(_point_to_dict(point) for point in pipeline_result.expected_trace.points),
        actual_points=tuple(_point_to_dict(point) for point in pipeline_result.actual_trace.points),
        figure_requirements=_figure_requirement_to_dict(pipeline_result.expected_figure),
        actual_figure=_figure_trace_to_dict(pipeline_result.actual_figure),
        intent=_intent_to_dict(pipeline_result),
        requirement_plan=_requirement_plan_to_dict(pipeline_result.requirement_plan),
        evidence_graph=_evidence_graph_to_dict(pipeline_result.evidence_graph),
        error_codes=verification.error_codes,
        errors=tuple(_error_to_dict(error) for error in verification.errors),
        repair_level=repair_level,
        repair_scope=repair_scope,
        repair_strategy=pipeline_result.repair.strategy if pipeline_result.repair else None,
        repair_instruction=pipeline_result.repair.instruction if pipeline_result.repair else None,
        repaired_code=pipeline_result.repaired_code,
        repair_attempts=tuple(_repair_attempt_to_dict(attempt) for attempt in pipeline_result.repair_attempts),
        exception_type=pipeline_result.execution_exception_type,
        exception_message=pipeline_result.execution_exception_message,
        case_metadata=_jsonable(result.case.metadata),
        run_metadata=_jsonable(result.metadata),
    )


def case_report_from_exception(case: ChartCase, exc: BaseException) -> CaseReport:
    backend_profile = infer_backend_profile(generated_code=case.generated_code)
    return CaseReport(
        case_id=case.case_id,
        status="error",
        ok=False,
        query=case.query,
        backend_profile=_backend_profile_to_dict(backend_profile),
        error_codes=("execution_error",),
        errors=(
            {
                "code": "execution_error",
                "message": str(exc),
                "operator": "execution",
                "severity": "error",
                "expected": None,
                "actual": type(exc).__name__,
                "requirement_id": None,
            },
        ),
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        case_metadata=_jsonable(case.metadata),
    )


def summarize_case_reports(cases: tuple[CaseReport, ...]) -> BatchSummary:
    total = len(cases)
    completed = sum(1 for case in cases if case.status != "error")
    passed = sum(1 for case in cases if case.status == "passed")
    failed = sum(1 for case in cases if case.status == "failed")
    errored = sum(1 for case in cases if case.status == "error")
    error_counts = Counter(code for case in cases for code in case.error_codes)
    operator_counts = Counter(error.get("operator") or "unknown" for case in cases for error in case.errors)
    severity_counts = Counter(error.get("severity") or "unknown" for case in cases for error in case.errors)
    repair_level_counts = Counter(str(case.repair_level) for case in cases if case.repair_level is not None)
    repair_scope_counts = Counter(case.repair_scope for case in cases if case.repair_scope)
    expected_chart_type_counts = Counter(case.expected_chart_type for case in cases if case.expected_chart_type)
    actual_chart_type_counts = Counter(case.actual_chart_type for case in cases if case.actual_chart_type)
    backend_name_counts = Counter(
        case.backend_profile.get("backend_name") for case in cases if case.backend_profile and case.backend_profile.get("backend_name")
    )
    backend_support_tier_counts = Counter(
        case.backend_profile.get("support_tier") for case in cases if case.backend_profile and case.backend_profile.get("support_tier")
    )
    backend_verification_mode_counts = Counter(
        case.backend_profile.get("verification_mode")
        for case in cases
        if case.backend_profile and case.backend_profile.get("verification_mode")
    )
    exception_counts = Counter(case.exception_type for case in cases if case.exception_type)
    return BatchSummary(
        total_cases=total,
        completed_cases=completed,
        passed_cases=passed,
        failed_cases=failed,
        errored_cases=errored,
        pass_rate=passed / completed if completed else 0.0,
        overall_pass_rate=passed / total if total else 0.0,
        completion_rate=completed / total if total else 0.0,
        error_counts=dict(error_counts),
        operator_counts=dict(operator_counts),
        severity_counts=dict(severity_counts),
        repair_level_counts=dict(repair_level_counts),
        repair_scope_counts=dict(repair_scope_counts),
        expected_chart_type_counts=dict(expected_chart_type_counts),
        actual_chart_type_counts=dict(actual_chart_type_counts),
        backend_name_counts=dict(backend_name_counts),
        backend_support_tier_counts=dict(backend_support_tier_counts),
        backend_verification_mode_counts=dict(backend_verification_mode_counts),
        exception_counts=dict(exception_counts),
    )


def _intent_to_dict(result: PipelineResult) -> dict[str, Any]:
    plan = result.plan
    return {
        "chart_type": plan.chart_type,
        "dimensions": list(plan.dimensions),
        "measure": {
            "column": plan.measure.column,
            "agg": plan.measure.agg,
        },
        "filters": [
            {
                "column": filter_spec.column,
                "op": filter_spec.op,
                "value": _jsonable(filter_spec.value),
            }
            for filter_spec in plan.filters
        ],
        "sort": _sort_to_dict(plan.sort),
        "limit": plan.limit,
        "confidence": plan.confidence,
    }


def _figure_requirement_to_dict(figure: Any) -> dict[str, Any] | None:
    if figure is None:
        return None
    return {
        "axes_count": figure.axes_count,
        "figure_title": figure.figure_title,
        "size_inches": list(figure.size_inches) if figure.size_inches else None,
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
                "xtick_labels": list(axis.xtick_labels),
                "ytick_labels": list(axis.ytick_labels),
                "ztick_labels": list(axis.ztick_labels),
                "bounds": list(axis.bounds) if axis.bounds else None,
                "legend_labels": list(axis.legend_labels),
                "artist_types": list(axis.artist_types),
                "artist_counts": dict(axis.artist_counts),
                "min_artist_counts": dict(axis.min_artist_counts),
                "text_contains": list(axis.text_contains),
            }
            for axis in figure.axes
        ],
    }


def _figure_trace_to_dict(figure: Any) -> dict[str, Any] | None:
    if figure is None:
        return None
    return {
        "title": figure.title,
        "size_inches": list(figure.size_inches) if figure.size_inches else None,
        "axes_count": figure.axes_count,
        "axes": [
            {
                "index": axis.index,
                "title": axis.title,
                "xlabel": axis.xlabel,
                "ylabel": axis.ylabel,
                "zlabel": axis.zlabel,
                "projection": axis.projection,
                "xscale": axis.xscale,
                "yscale": axis.yscale,
                "zscale": axis.zscale,
                "xtick_labels": list(axis.xtick_labels),
                "ytick_labels": list(axis.ytick_labels),
                "ztick_labels": list(axis.ztick_labels),
                "bounds": list(axis.bounds) if axis.bounds else None,
                "legend_labels": list(axis.legend_labels),
                "texts": list(axis.texts),
                "artists": [
                    {
                        "artist_type": artist.artist_type,
                        "label": artist.label,
                        "color": _jsonable(artist.color),
                        "linestyle": artist.linestyle,
                        "marker": artist.marker,
                        "count": artist.count,
                    }
                    for artist in axis.artists
                ],
            }
            for axis in figure.axes
        ],
    }


def _requirement_plan_to_dict(plan: Any) -> dict[str, Any] | None:
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


def _evidence_graph_to_dict(graph: Any) -> dict[str, Any] | None:
    if graph is None:
        return None
    return {
        "failed_requirement_ids": list(graph.failed_requirement_ids),
        "passed_requirement_ids": list(graph.passed_requirement_ids),
        "expected_artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "requirement_ids": list(artifact.requirement_ids),
                "source": artifact.source,
                "program": artifact.program,
                "input_hash": artifact.input_hash,
                "artifact_hash": artifact.artifact_hash,
                "panel_id": artifact.panel_id,
            }
            for artifact in graph.expected_artifacts
        ],
        "actual_artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "requirement_ids": list(artifact.requirement_ids),
                "source": artifact.source,
                "program": artifact.program,
                "input_hash": artifact.input_hash,
                "artifact_hash": artifact.artifact_hash,
                "panel_id": artifact.panel_id,
            }
            for artifact in graph.actual_artifacts
        ],
        "links": [
            {
                "requirement_id": link.requirement_id,
                "expected_artifact_id": link.expected_artifact_id,
                "actual_artifact_id": link.actual_artifact_id,
                "verdict": link.verdict,
                "error_codes": list(link.error_codes),
                "message": link.message,
            }
            for link in graph.links
        ],
    }


def _repair_attempt_to_dict(attempt: Any) -> dict[str, Any]:
    return {
        "round_index": attempt.round_index,
        "applied": attempt.applied,
        "strategy": attempt.strategy,
        "scope": attempt.scope,
        "targeted_requirement_ids": list(attempt.targeted_requirement_ids),
        "targeted_error_codes": list(attempt.targeted_error_codes),
        "resolved_requirement_ids": list(attempt.resolved_requirement_ids),
        "unresolved_requirement_ids": list(attempt.unresolved_requirement_ids),
        "report_ok": attempt.report_ok,
        "instruction": attempt.instruction,
        "input_code": attempt.input_code,
        "output_code": attempt.output_code,
    }


def _sort_to_dict(sort: SortSpec | None) -> dict[str, Any] | None:
    if sort is None:
        return None
    return {"by": sort.by, "direction": sort.direction}


def _backend_profile_to_dict(profile: Any) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "backend_name": profile.backend_name,
        "support_tier": profile.support_tier,
        "verification_mode": profile.verification_mode,
        "notes": profile.notes,
    }


def _point_to_dict(point: DataPoint) -> dict[str, Any]:
    return {
        "x": _jsonable(point.x),
        "y": _jsonable(point.y),
        "meta": _jsonable(point.meta),
    }


def _error_to_dict(error: VerificationError) -> dict[str, Any]:
    return {
        "code": error.code,
        "message": error.message,
        "expected": _jsonable(error.expected),
        "actual": _jsonable(error.actual),
        "operator": error.operator,
        "requirement_id": error.requirement_id,
        "severity": error.severity,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
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
