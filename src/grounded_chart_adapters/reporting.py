from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from grounded_chart import infer_backend_profile, repair_action_class_for_scope
from grounded_chart.diagnostics import failure_atoms_from_evidence_graph, failure_atoms_to_dicts
from grounded_chart.schema import DataPoint, PipelineResult, SortSpec, VerificationError
from grounded_chart_adapters.base import AdapterRunResult, ChartCase

CaseStatus = Literal["passed", "failed", "error"]
CaseVerdict = Literal["passed", "hard_failed", "warning_only_failed", "error"]


@dataclass(frozen=True)
class CaseReport:
    """JSON-serializable benchmark-facing report for one chart case."""

    case_id: str
    status: CaseStatus
    ok: bool
    query: str
    case_verdict: CaseVerdict | None = None
    parse_source: str | None = None
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
    repair_action_class: str | None = None
    repair_routing_reason: str | None = None
    repair_strategy: str | None = None
    repair_instruction: str | None = None
    repair_trace: dict[str, Any] | None = None
    repair_patch_ops: tuple[dict[str, Any], ...] = ()
    repaired_code: str | None = None
    repair_attempts: tuple[dict[str, Any], ...] = ()
    repair_loop_status: str | None = None
    repair_loop_reason: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    case_metadata: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    requirement_metrics: dict[str, Any] | None = None
    failure_taxonomy: dict[str, Any] | None = None
    artifact_chain_summary: tuple[dict[str, Any], ...] = ()
    failure_atoms: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "case_verdict": _case_verdict_from_case(self),
            "ok": self.ok,
            "query": self.query,
            "parse_source": self.parse_source,
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
            "repair_action_class": self.repair_action_class,
            "repair_routing_reason": self.repair_routing_reason,
            "repair_strategy": self.repair_strategy,
            "repair_instruction": self.repair_instruction,
            "repair_trace": _jsonable(self.repair_trace),
            "repair_patch_ops": list(self.repair_patch_ops),
            "repaired_code": self.repaired_code,
            "repair_attempts": list(self.repair_attempts),
            "repair_loop_status": self.repair_loop_status,
            "repair_loop_reason": self.repair_loop_reason,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "case_metadata": _jsonable(self.case_metadata),
            "run_metadata": _jsonable(self.run_metadata),
            "requirement_metrics": _jsonable(self.requirement_metrics),
            "failure_taxonomy": _jsonable(self.failure_taxonomy),
            "artifact_chain_summary": list(self.artifact_chain_summary),
            "failure_atoms": list(self.failure_atoms),
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
    hard_failed_cases: int
    warning_only_failed_cases: int
    soft_passed_cases: int
    pass_rate: float
    overall_pass_rate: float
    hard_pass_rate: float
    overall_hard_pass_rate: float
    completion_rate: float
    parse_source_counts: dict[str, int] = field(default_factory=dict)
    total_requirements: int = 0
    verifiable_requirements: int = 0
    passed_requirements: int = 0
    failed_requirements: int = 0
    hard_verifiable_requirements: int = 0
    hard_passed_requirements: int = 0
    hard_failed_requirements: int = 0
    warning_failed_requirements: int = 0
    abstained_requirements: int = 0
    unsupported_requirements: int = 0
    ambiguous_requirements: int = 0
    requirement_coverage: float = 0.0
    requirement_satisfaction: float = 0.0
    hard_requirement_satisfaction: float = 0.0
    error_counts: dict[str, int] = field(default_factory=dict)
    operator_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    requirement_status_counts: dict[str, int] = field(default_factory=dict)
    requirement_type_counts: dict[str, int] = field(default_factory=dict)
    failed_requirement_type_counts: dict[str, int] = field(default_factory=dict)
    requirement_severity_counts: dict[str, int] = field(default_factory=dict)
    requirement_match_policy_counts: dict[str, int] = field(default_factory=dict)
    failed_requirement_severity_counts: dict[str, int] = field(default_factory=dict)
    failure_stage_counts: dict[str, int] = field(default_factory=dict)
    failure_family_counts: dict[str, int] = field(default_factory=dict)
    repair_level_counts: dict[str, int] = field(default_factory=dict)
    repair_scope_counts: dict[str, int] = field(default_factory=dict)
    repair_action_class_counts: dict[str, int] = field(default_factory=dict)
    repair_action_outcomes: dict[str, dict[str, int]] = field(default_factory=dict)
    expected_chart_type_counts: dict[str, int] = field(default_factory=dict)
    actual_chart_type_counts: dict[str, int] = field(default_factory=dict)
    backend_name_counts: dict[str, int] = field(default_factory=dict)
    backend_support_tier_counts: dict[str, int] = field(default_factory=dict)
    backend_verification_mode_counts: dict[str, int] = field(default_factory=dict)
    exception_counts: dict[str, int] = field(default_factory=dict)
    case_verdict_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "errored_cases": self.errored_cases,
            "hard_failed_cases": self.hard_failed_cases,
            "warning_only_failed_cases": self.warning_only_failed_cases,
            "soft_passed_cases": self.soft_passed_cases,
            "pass_rate": self.pass_rate,
            "overall_pass_rate": self.overall_pass_rate,
            "hard_pass_rate": self.hard_pass_rate,
            "overall_hard_pass_rate": self.overall_hard_pass_rate,
            "completion_rate": self.completion_rate,
            "parse_source_counts": dict(self.parse_source_counts),
            "total_requirements": self.total_requirements,
            "verifiable_requirements": self.verifiable_requirements,
            "passed_requirements": self.passed_requirements,
            "failed_requirements": self.failed_requirements,
            "hard_verifiable_requirements": self.hard_verifiable_requirements,
            "hard_passed_requirements": self.hard_passed_requirements,
            "hard_failed_requirements": self.hard_failed_requirements,
            "warning_failed_requirements": self.warning_failed_requirements,
            "abstained_requirements": self.abstained_requirements,
            "unsupported_requirements": self.unsupported_requirements,
            "ambiguous_requirements": self.ambiguous_requirements,
            "requirement_coverage": self.requirement_coverage,
            "requirement_satisfaction": self.requirement_satisfaction,
            "hard_requirement_satisfaction": self.hard_requirement_satisfaction,
            "error_counts": dict(self.error_counts),
            "operator_counts": dict(self.operator_counts),
            "severity_counts": dict(self.severity_counts),
            "requirement_status_counts": dict(self.requirement_status_counts),
            "requirement_type_counts": dict(self.requirement_type_counts),
            "failed_requirement_type_counts": dict(self.failed_requirement_type_counts),
            "requirement_severity_counts": dict(self.requirement_severity_counts),
            "requirement_match_policy_counts": dict(self.requirement_match_policy_counts),
            "failed_requirement_severity_counts": dict(self.failed_requirement_severity_counts),
            "failure_stage_counts": dict(self.failure_stage_counts),
            "failure_family_counts": dict(self.failure_family_counts),
            "repair_level_counts": dict(self.repair_level_counts),
            "repair_scope_counts": dict(self.repair_scope_counts),
            "repair_action_class_counts": dict(self.repair_action_class_counts),
            "repair_action_outcomes": _jsonable(self.repair_action_outcomes),
            "expected_chart_type_counts": dict(self.expected_chart_type_counts),
            "actual_chart_type_counts": dict(self.actual_chart_type_counts),
            "backend_name_counts": dict(self.backend_name_counts),
            "backend_support_tier_counts": dict(self.backend_support_tier_counts),
            "backend_verification_mode_counts": dict(self.backend_verification_mode_counts),
            "exception_counts": dict(self.exception_counts),
            "case_verdict_counts": dict(self.case_verdict_counts),
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


def _case_verdict_from_verification(verification) -> CaseVerdict:
    if verification.ok:
        return "passed"
    if any(_is_blocking_error(error) for error in verification.errors):
        return "hard_failed"
    return "warning_only_failed"


def _case_verdict_from_case(case: CaseReport) -> CaseVerdict:
    if case.case_verdict is not None:
        return case.case_verdict
    if case.status == "error":
        return "error"
    if case.status == "passed":
        return "passed"
    if any(_is_blocking_error_dict(error) for error in case.errors):
        return "hard_failed"
    return "warning_only_failed"


def _is_blocking_error(error: VerificationError) -> bool:
    return (error.severity or "error") == "error"


def _is_blocking_error_dict(error: dict[str, Any]) -> bool:
    return str(error.get("severity") or "error") == "error"

def case_report_from_result(result: AdapterRunResult) -> CaseReport:
    pipeline_result = result.pipeline_result
    verification = pipeline_result.report
    status: CaseStatus = "passed" if verification.ok else "failed"
    case_verdict = _case_verdict_from_verification(verification)
    repair_level = 0 if verification.ok else None
    repair_scope = "none" if verification.ok else None
    repair_routing_reason = "All verifiable requirements passed." if verification.ok else None
    if pipeline_result.repair_plan is not None:
        repair_level = int(pipeline_result.repair_plan.repair_level)
        repair_scope = pipeline_result.repair_plan.scope
        repair_routing_reason = pipeline_result.repair_plan.reason
    repair_action_class = repair_action_class_for_scope(repair_scope)
    requirement_metrics = _requirement_metrics_from_result(pipeline_result)
    failure_taxonomy = _failure_taxonomy_from_result(
        status=status,
        pipeline_result=pipeline_result,
        requirement_metrics=requirement_metrics,
    )
    backend_profile = infer_backend_profile(
        actual_trace=pipeline_result.actual_trace,
        actual_figure=pipeline_result.actual_figure,
        generated_code=result.case.generated_code,
    )
    return CaseReport(
        case_id=result.case.case_id,
        status=status,
        ok=verification.ok,
        case_verdict=case_verdict,
        query=result.case.query,
        parse_source=pipeline_result.parse_source,
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
        repair_action_class=repair_action_class,
        repair_routing_reason=repair_routing_reason,
        repair_strategy=pipeline_result.repair.strategy if pipeline_result.repair else None,
        repair_instruction=pipeline_result.repair.instruction if pipeline_result.repair else None,
        repair_trace=_llm_trace_to_dict(pipeline_result.repair.llm_trace) if pipeline_result.repair else None,
        repair_patch_ops=tuple(_patch_op_to_dict(operation) for operation in (pipeline_result.repair.patch_ops if pipeline_result.repair else ())),
        repaired_code=pipeline_result.repaired_code,
        repair_attempts=tuple(_repair_attempt_to_dict(attempt) for attempt in pipeline_result.repair_attempts),
        repair_loop_status=pipeline_result.repair_loop_status,
        repair_loop_reason=pipeline_result.repair_loop_reason,
        exception_type=pipeline_result.execution_exception_type,
        exception_message=pipeline_result.execution_exception_message,
        case_metadata=_jsonable(result.case.metadata),
        run_metadata=_jsonable(result.metadata),
        requirement_metrics=requirement_metrics,
        failure_taxonomy=failure_taxonomy,
        artifact_chain_summary=tuple(_artifact_chain_summary_from_result(pipeline_result)),
        failure_atoms=tuple(failure_atoms_to_dicts(failure_atoms_from_evidence_graph(pipeline_result.evidence_graph))),
    )


def case_report_from_exception(case: ChartCase, exc: BaseException) -> CaseReport:
    backend_profile = infer_backend_profile(generated_code=case.generated_code)
    return CaseReport(
        case_id=case.case_id,
        status="error",
        ok=False,
        case_verdict="error",
        query=case.query,
        parse_source=case.parse_source,
        backend_profile=_backend_profile_to_dict(backend_profile),
        error_codes=("execution_error",),
        errors=(
            {
                "code": "execution_error",
                "message": str(exc),
                "operator": "execution",
                "severity": "error",
                "match_policy": None,
                "expected": None,
                "actual": type(exc).__name__,
                "requirement_id": None,
            },
        ),
        repair_action_class="abstain",
        repair_routing_reason="Code execution failed before repair routing could produce a scoped action.",
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        case_metadata=_jsonable(case.metadata),
        failure_taxonomy={
            "primary_stage": "code_execution_failure",
            "primary_family": "runtime_compatibility",
            "reason": "Code execution failed before the pipeline could extract comparable artifacts.",
            "failed_requirement_ids": [],
            "failed_requirement_types": [],
            "abstained_requirement_ids": [],
            "unsupported_requirement_ids": [],
        },
    )


def summarize_case_reports(cases: tuple[CaseReport, ...]) -> BatchSummary:
    total = len(cases)
    completed = sum(1 for case in cases if case.status != "error")
    passed = sum(1 for case in cases if case.status == "passed")
    failed = sum(1 for case in cases if case.status == "failed")
    errored = sum(1 for case in cases if case.status == "error")
    case_verdict_counts = Counter(_case_verdict_from_case(case) for case in cases)
    hard_failed = case_verdict_counts["hard_failed"]
    warning_only_failed = case_verdict_counts["warning_only_failed"]
    soft_passed = passed + warning_only_failed
    error_counts = Counter(code for case in cases for code in case.error_codes)
    operator_counts = Counter(error.get("operator") or "unknown" for case in cases for error in case.errors)
    severity_counts = Counter(error.get("severity") or "unknown" for case in cases for error in case.errors)
    parse_source_counts = Counter(case.parse_source for case in cases if case.parse_source)
    requirement_status_counts = Counter()
    requirement_type_counts = Counter()
    failed_requirement_type_counts = Counter()
    requirement_severity_counts = Counter()
    requirement_match_policy_counts = Counter()
    failed_requirement_severity_counts = Counter()
    failure_stage_counts = Counter()
    failure_family_counts = Counter()
    repair_level_counts = Counter(str(case.repair_level) for case in cases if case.repair_level is not None)
    repair_scope_counts = Counter(case.repair_scope for case in cases if case.repair_scope)
    repair_action_class_counts = Counter(case.repair_action_class for case in cases if case.repair_action_class)
    repair_action_outcomes_counter: dict[str, Counter[str]] = {}
    for case in cases:
        action_class = case.repair_action_class
        if not action_class:
            continue
        repair_action_outcomes_counter.setdefault(action_class, Counter())
        repair_action_outcomes_counter[action_class][case.status] += 1
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
    total_requirements = 0
    verifiable_requirements = 0
    passed_requirements = 0
    failed_requirements = 0
    hard_verifiable_requirements = 0
    hard_passed_requirements = 0
    hard_failed_requirements = 0
    warning_failed_requirements = 0
    abstained_requirements = 0
    unsupported_requirements = 0
    ambiguous_requirements = 0
    for case in cases:
        requirement_metrics = dict(case.requirement_metrics or {})
        taxonomy = dict(case.failure_taxonomy or {})
        if requirement_metrics:
            total_requirements += int(requirement_metrics.get("total_requirements", 0) or 0)
            verifiable_requirements += int(requirement_metrics.get("verifiable_requirements", 0) or 0)
            passed_requirements += int(requirement_metrics.get("passed_requirements", 0) or 0)
            failed_requirements += int(requirement_metrics.get("failed_requirements", 0) or 0)
            hard_verifiable_requirements += int(requirement_metrics.get("hard_verifiable_requirements", 0) or 0)
            hard_passed_requirements += int(requirement_metrics.get("hard_passed_requirements", 0) or 0)
            hard_failed_requirements += int(requirement_metrics.get("hard_failed_requirements", 0) or 0)
            warning_failed_requirements += int(requirement_metrics.get("warning_failed_requirements", 0) or 0)
            abstained_requirements += int(requirement_metrics.get("abstained_requirements", 0) or 0)
            unsupported_requirements += int(requirement_metrics.get("unsupported_requirements", 0) or 0)
            ambiguous_requirements += int(requirement_metrics.get("ambiguous_requirements", 0) or 0)
            requirement_status_counts.update(requirement_metrics.get("requirement_status_counts", {}))
            requirement_type_counts.update(requirement_metrics.get("requirement_type_counts", {}))
            failed_requirement_type_counts.update(requirement_metrics.get("failed_requirement_type_counts", {}))
            requirement_severity_counts.update(requirement_metrics.get("requirement_severity_counts", {}))
            requirement_match_policy_counts.update(requirement_metrics.get("requirement_match_policy_counts", {}))
            failed_requirement_severity_counts.update(requirement_metrics.get("failed_requirement_severity_counts", {}))
        primary_stage = str(taxonomy.get("primary_stage") or "").strip()
        if primary_stage:
            failure_stage_counts.update([primary_stage])
        primary_family = str(taxonomy.get("primary_family") or "").strip()
        if primary_family:
            failure_family_counts.update([primary_family])
    requirement_coverage = (verifiable_requirements / total_requirements) if total_requirements else 0.0
    requirement_satisfaction = (passed_requirements / verifiable_requirements) if verifiable_requirements else 0.0
    hard_requirement_satisfaction = (
        hard_passed_requirements / hard_verifiable_requirements if hard_verifiable_requirements else 0.0
    )
    return BatchSummary(
        total_cases=total,
        completed_cases=completed,
        passed_cases=passed,
        failed_cases=failed,
        errored_cases=errored,
        hard_failed_cases=hard_failed,
        warning_only_failed_cases=warning_only_failed,
        soft_passed_cases=soft_passed,
        pass_rate=passed / completed if completed else 0.0,
        overall_pass_rate=passed / total if total else 0.0,
        hard_pass_rate=soft_passed / completed if completed else 0.0,
        overall_hard_pass_rate=soft_passed / total if total else 0.0,
        completion_rate=completed / total if total else 0.0,
        parse_source_counts=dict(parse_source_counts),
        total_requirements=total_requirements,
        verifiable_requirements=verifiable_requirements,
        passed_requirements=passed_requirements,
        failed_requirements=failed_requirements,
        hard_verifiable_requirements=hard_verifiable_requirements,
        hard_passed_requirements=hard_passed_requirements,
        hard_failed_requirements=hard_failed_requirements,
        warning_failed_requirements=warning_failed_requirements,
        abstained_requirements=abstained_requirements,
        unsupported_requirements=unsupported_requirements,
        ambiguous_requirements=ambiguous_requirements,
        requirement_coverage=requirement_coverage,
        requirement_satisfaction=requirement_satisfaction,
        hard_requirement_satisfaction=hard_requirement_satisfaction,
        error_counts=dict(error_counts),
        operator_counts=dict(operator_counts),
        severity_counts=dict(severity_counts),
        requirement_status_counts=dict(requirement_status_counts),
        requirement_type_counts=dict(requirement_type_counts),
        failed_requirement_type_counts=dict(failed_requirement_type_counts),
        requirement_severity_counts=dict(requirement_severity_counts),
        requirement_match_policy_counts=dict(requirement_match_policy_counts),
        failed_requirement_severity_counts=dict(failed_requirement_severity_counts),
        failure_stage_counts=dict(failure_stage_counts),
        failure_family_counts=dict(failure_family_counts),
        repair_level_counts=dict(repair_level_counts),
        repair_scope_counts=dict(repair_scope_counts),
        repair_action_class_counts=dict(repair_action_class_counts),
        repair_action_outcomes={key: dict(value) for key, value in repair_action_outcomes_counter.items()},
        expected_chart_type_counts=dict(expected_chart_type_counts),
        actual_chart_type_counts=dict(actual_chart_type_counts),
        backend_name_counts=dict(backend_name_counts),
        backend_support_tier_counts=dict(backend_support_tier_counts),
        backend_verification_mode_counts=dict(backend_verification_mode_counts),
        exception_counts=dict(exception_counts),
        case_verdict_counts=dict(case_verdict_counts),
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
        "source_spans": dict(getattr(figure, "source_spans", {})),
        "artifact_contracts": _jsonable(getattr(figure, "artifact_contracts", ())),
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
                "source_spans": dict(getattr(axis, "source_spans", {})),
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
                "payload_preview": _artifact_payload_preview(artifact.payload),
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
                "payload_preview": _artifact_payload_preview(artifact.payload),
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


def _artifact_payload_preview(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, list):
        return _jsonable(payload[:20])
    if isinstance(payload, tuple):
        return _jsonable(payload[:20])
    return _jsonable(payload)


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
        "decision_action": attempt.decision_action,
        "decision_reason": attempt.decision_reason,
        "decision_next_scope": attempt.decision_next_scope,
        "llm_trace": _llm_trace_to_dict(attempt.llm_trace),
        "patch_ops": [_patch_op_to_dict(operation) for operation in attempt.patch_ops],
        "input_code": attempt.input_code,
        "output_code": attempt.output_code,
    }


def _llm_trace_to_dict(trace: Any) -> dict[str, Any] | None:
    if trace is None:
        return None
    usage = getattr(trace, "usage", None)
    return {
        "provider": getattr(trace, "provider", None),
        "model": getattr(trace, "model", None),
        "base_url": getattr(trace, "base_url", None),
        "temperature": getattr(trace, "temperature", None),
        "max_tokens": getattr(trace, "max_tokens", None),
        "raw_text": getattr(trace, "raw_text", None),
        "parsed_json": _jsonable(getattr(trace, "parsed_json", None)),
        "usage": (
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "raw": _jsonable(getattr(usage, "raw", None)),
            }
            if usage is not None
            else None
        ),
        "raw_response": _jsonable(getattr(trace, "raw_response", None)),
    }


def _patch_op_to_dict(operation: Any) -> dict[str, Any]:
    anchor = getattr(operation, "anchor", None)
    return {
        "op": getattr(operation, "op", None),
        "anchor": {
            "kind": getattr(anchor, "kind", None),
            "name": getattr(anchor, "name", None),
            "text": getattr(anchor, "text", None),
            "occurrence": getattr(anchor, "occurrence", None),
        }
        if anchor is not None
        else None,
        "arg_index": getattr(operation, "arg_index", None),
        "keyword": getattr(operation, "keyword", None),
        "new_value": _jsonable(getattr(operation, "new_value", None)),
        "description": getattr(operation, "description", None),
    }


def _requirement_metrics_from_result(pipeline_result: PipelineResult) -> dict[str, Any] | None:
    requirement_plan = pipeline_result.requirement_plan
    evidence_graph = pipeline_result.evidence_graph
    if requirement_plan is None:
        return None

    requirements = tuple(requirement_plan.requirements)
    requirement_by_id = {requirement.requirement_id: requirement for requirement in requirements}
    status_counts = Counter(requirement.status for requirement in requirements)
    type_counts = Counter(requirement.type for requirement in requirements)
    severity_counts = Counter(requirement.severity for requirement in requirements)
    match_policy_counts = Counter(requirement.match_policy for requirement in requirements)

    passed_requirement_ids: list[str] = []
    failed_requirement_ids: list[str] = []
    abstained_requirement_ids: list[str] = []
    unsupported_requirement_ids: list[str] = []
    failed_requirement_type_counts = Counter()
    failed_requirement_severity_counts = Counter()
    if evidence_graph is not None:
        for link in evidence_graph.links:
            requirement_id = link.requirement_id
            if link.verdict == "pass":
                passed_requirement_ids.append(requirement_id)
            elif link.verdict == "fail":
                failed_requirement_ids.append(requirement_id)
                requirement = requirement_by_id.get(requirement_id)
                failed_requirement_type_counts.update([requirement.type if requirement is not None else "unknown"])
                failed_requirement_severity_counts.update([requirement.severity if requirement is not None else "unknown"])
            elif link.verdict == "abstain":
                abstained_requirement_ids.append(requirement_id)
            elif link.verdict == "unsupported":
                unsupported_requirement_ids.append(requirement_id)

    ambiguous_requirement_ids = [requirement.requirement_id for requirement in requirement_plan.ambiguous_requirements]
    if not unsupported_requirement_ids:
        unsupported_requirement_ids = [requirement.requirement_id for requirement in requirement_plan.unsupported_requirements]

    total_requirements = len(requirements)
    verifiable_requirements = len(requirement_plan.verifiable_requirements)
    hard_verifiable_requirement_ids = [
        requirement.requirement_id
        for requirement in requirement_plan.verifiable_requirements
        if requirement.severity == "error"
    ]
    passed_requirements = len(passed_requirement_ids)
    failed_requirements = len(failed_requirement_ids)
    hard_passed_requirement_ids = [
        requirement_id
        for requirement_id in passed_requirement_ids
        if requirement_by_id.get(requirement_id) is not None
        and requirement_by_id[requirement_id].severity == "error"
    ]
    hard_failed_requirement_ids = [
        requirement_id
        for requirement_id in failed_requirement_ids
        if requirement_by_id.get(requirement_id) is not None
        and requirement_by_id[requirement_id].severity == "error"
    ]
    warning_failed_requirement_ids = [
        requirement_id
        for requirement_id in failed_requirement_ids
        if requirement_by_id.get(requirement_id) is not None
        and requirement_by_id[requirement_id].severity != "error"
    ]
    abstained_requirements = len(abstained_requirement_ids)
    unsupported_requirements = len(unsupported_requirement_ids)
    ambiguous_requirements = len(ambiguous_requirement_ids)
    return {
        "total_requirements": total_requirements,
        "verifiable_requirements": verifiable_requirements,
        "passed_requirements": passed_requirements,
        "failed_requirements": failed_requirements,
        "hard_verifiable_requirements": len(hard_verifiable_requirement_ids),
        "hard_passed_requirements": len(hard_passed_requirement_ids),
        "hard_failed_requirements": len(hard_failed_requirement_ids),
        "warning_failed_requirements": len(warning_failed_requirement_ids),
        "abstained_requirements": abstained_requirements,
        "unsupported_requirements": unsupported_requirements,
        "ambiguous_requirements": ambiguous_requirements,
        "requirement_coverage": (verifiable_requirements / total_requirements) if total_requirements else 0.0,
        "requirement_satisfaction": (passed_requirements / verifiable_requirements) if verifiable_requirements else 0.0,
        "hard_requirement_satisfaction": (
            len(hard_passed_requirement_ids) / len(hard_verifiable_requirement_ids)
            if hard_verifiable_requirement_ids
            else 0.0
        ),
        "requirement_status_counts": dict(status_counts),
        "requirement_type_counts": dict(type_counts),
        "failed_requirement_type_counts": dict(failed_requirement_type_counts),
        "requirement_severity_counts": dict(severity_counts),
        "requirement_match_policy_counts": dict(match_policy_counts),
        "failed_requirement_severity_counts": dict(failed_requirement_severity_counts),
        "passed_requirement_ids": passed_requirement_ids,
        "hard_passed_requirement_ids": hard_passed_requirement_ids,
        "failed_requirement_ids": failed_requirement_ids,
        "hard_failed_requirement_ids": hard_failed_requirement_ids,
        "warning_failed_requirement_ids": warning_failed_requirement_ids,
        "abstained_requirement_ids": abstained_requirement_ids,
        "unsupported_requirement_ids": unsupported_requirement_ids,
        "ambiguous_requirement_ids": ambiguous_requirement_ids,
    }


def _failure_taxonomy_from_result(
    *,
    status: CaseStatus,
    pipeline_result: PipelineResult,
    requirement_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = dict(requirement_metrics or {})
    failed_requirement_ids = list(metrics.get("failed_requirement_ids", []))
    abstained_requirement_ids = list(metrics.get("abstained_requirement_ids", []))
    unsupported_requirement_ids = list(metrics.get("unsupported_requirement_ids", []))
    failed_requirement_type_counts = dict(metrics.get("failed_requirement_type_counts", {}))
    failed_requirement_types = sorted(
        requirement_type
        for requirement_type, count in failed_requirement_type_counts.items()
        if int(count) > 0
    )

    if status == "passed":
        return {
            "primary_stage": "none",
            "primary_family": "none",
            "reason": "All verifiable requirements passed.",
            "failed_requirement_ids": failed_requirement_ids,
            "failed_requirement_types": failed_requirement_types,
            "abstained_requirement_ids": abstained_requirement_ids,
            "unsupported_requirement_ids": unsupported_requirement_ids,
        }

    if _repair_policy_blocked(pipeline_result):
        return {
            "primary_stage": "repair_policy_block",
            "primary_family": _primary_failure_family(
                failed_requirement_types=failed_requirement_types,
                runtime_failure=_has_runtime_failure(pipeline_result),
                abstained_requirement_ids=abstained_requirement_ids,
            ),
            "reason": pipeline_result.repair_loop_reason
            or (pipeline_result.repair_plan.reason if pipeline_result.repair_plan is not None else "")
            or "Automatic repair was blocked by policy.",
            "failed_requirement_ids": failed_requirement_ids,
            "failed_requirement_types": failed_requirement_types,
            "abstained_requirement_ids": abstained_requirement_ids,
            "unsupported_requirement_ids": unsupported_requirement_ids,
        }

    if pipeline_result.repair_attempts:
        return {
            "primary_stage": "repair_failure",
            "primary_family": _primary_failure_family(
                failed_requirement_types=failed_requirement_types,
                runtime_failure=_has_runtime_failure(pipeline_result),
                abstained_requirement_ids=abstained_requirement_ids,
            ),
            "reason": pipeline_result.repair_loop_reason or "Repair attempts ended with unresolved failures.",
            "failed_requirement_ids": failed_requirement_ids,
            "failed_requirement_types": failed_requirement_types,
            "abstained_requirement_ids": abstained_requirement_ids,
            "unsupported_requirement_ids": unsupported_requirement_ids,
        }

    if _has_runtime_failure(pipeline_result):
        return {
            "primary_stage": "code_execution_failure",
            "primary_family": "runtime_compatibility",
            "reason": _runtime_failure_reason(pipeline_result),
            "failed_requirement_ids": failed_requirement_ids,
            "failed_requirement_types": failed_requirement_types,
            "abstained_requirement_ids": abstained_requirement_ids,
            "unsupported_requirement_ids": unsupported_requirement_ids,
        }

    verifiable_requirements = int(metrics.get("verifiable_requirements", 0) or 0)
    failed_requirements = int(metrics.get("failed_requirements", 0) or 0)
    abstained_requirements = int(metrics.get("abstained_requirements", 0) or 0)
    ambiguous_requirements = int(metrics.get("ambiguous_requirements", 0) or 0)
    unsupported_requirements = int(metrics.get("unsupported_requirements", 0) or 0)

    if verifiable_requirements == 0 and (ambiguous_requirements > 0 or unsupported_requirements > 0):
        return {
            "primary_stage": "parse_ambiguity",
            "primary_family": "ambiguous_or_unsupported",
            "reason": "Parsing produced no verifiable requirements because the request remains ambiguous or unsupported.",
            "failed_requirement_ids": failed_requirement_ids,
            "failed_requirement_types": failed_requirement_types,
            "abstained_requirement_ids": abstained_requirement_ids,
            "unsupported_requirement_ids": unsupported_requirement_ids,
        }

    if failed_requirements > 0:
        return {
            "primary_stage": "requirement_violation",
            "primary_family": _primary_failure_family(
                failed_requirement_types=failed_requirement_types,
                runtime_failure=False,
                abstained_requirement_ids=abstained_requirement_ids,
            ),
            "reason": f"{failed_requirements} verifiable requirement(s) failed.",
            "failed_requirement_ids": failed_requirement_ids,
            "failed_requirement_types": failed_requirement_types,
            "abstained_requirement_ids": abstained_requirement_ids,
            "unsupported_requirement_ids": unsupported_requirement_ids,
        }

    if abstained_requirements > 0:
        return {
            "primary_stage": "trace_extraction_failure",
            "primary_family": "trace_binding",
            "reason": "At least one requirement could not be bound to comparable expected and actual artifacts.",
            "failed_requirement_ids": failed_requirement_ids,
            "failed_requirement_types": failed_requirement_types,
            "abstained_requirement_ids": abstained_requirement_ids,
            "unsupported_requirement_ids": unsupported_requirement_ids,
        }

    if ambiguous_requirements > 0 or unsupported_requirements > 0:
        return {
            "primary_stage": "parse_ambiguity",
            "primary_family": "ambiguous_or_unsupported",
            "reason": "Some extracted requirements remain ambiguous or unsupported.",
            "failed_requirement_ids": failed_requirement_ids,
            "failed_requirement_types": failed_requirement_types,
            "abstained_requirement_ids": abstained_requirement_ids,
            "unsupported_requirement_ids": unsupported_requirement_ids,
        }

    return {
        "primary_stage": "unknown_failure",
        "primary_family": _primary_failure_family(
            failed_requirement_types=failed_requirement_types,
            runtime_failure=False,
            abstained_requirement_ids=abstained_requirement_ids,
        ),
        "reason": "The case failed, but the failure did not match a known taxonomy stage.",
        "failed_requirement_ids": failed_requirement_ids,
        "failed_requirement_types": failed_requirement_types,
        "abstained_requirement_ids": abstained_requirement_ids,
        "unsupported_requirement_ids": unsupported_requirement_ids,
    }


def _artifact_chain_summary_from_result(pipeline_result: PipelineResult) -> list[dict[str, Any]]:
    graph = pipeline_result.evidence_graph
    if graph is None:
        return []
    expected_artifacts = {artifact.artifact_id: artifact for artifact in graph.expected_artifacts}
    actual_artifacts = {artifact.artifact_id: artifact for artifact in graph.actual_artifacts}
    requirement_meta = {
        requirement.requirement_id: requirement
        for requirement in graph.requirements
    }
    summaries: list[dict[str, Any]] = []
    for link in graph.links:
        if link.verdict not in {"fail", "abstain", "unsupported"}:
            continue
        expected_artifact = expected_artifacts.get(link.expected_artifact_id or "")
        actual_artifact = actual_artifacts.get(link.actual_artifact_id or "")
        summaries.append(
            {
                "requirement_id": link.requirement_id,
                "requirement_name": getattr(requirement_meta.get(link.requirement_id), "name", None),
                "requirement_severity": getattr(requirement_meta.get(link.requirement_id), "severity", None),
                "requirement_match_policy": getattr(requirement_meta.get(link.requirement_id), "match_policy", None),
                "verdict": link.verdict,
                "error_codes": list(link.error_codes),
                "expected_artifact_id": link.expected_artifact_id,
                "actual_artifact_id": link.actual_artifact_id,
                "expected_preview": _artifact_summary_preview(expected_artifact),
                "actual_preview": _artifact_summary_preview(actual_artifact),
                "diagnosis": _artifact_chain_diagnosis(link, expected_artifact, actual_artifact),
            }
        )
    return summaries


def _artifact_chain_diagnosis(link: Any, expected_artifact: Any, actual_artifact: Any) -> str:
    requirement = link.requirement_id
    errors = ", ".join(link.error_codes) if link.error_codes else "no concrete verifier error"
    expected_label = _artifact_label(expected_artifact)
    actual_label = _artifact_label(actual_artifact)
    if expected_artifact is None and actual_artifact is None:
        return f"{requirement}: {link.verdict}; no expected or actual artifact was bound ({errors})."
    if expected_artifact is None:
        return f"{requirement}: {link.verdict}; actual evidence comes from {actual_label}, but no expected artifact was bound ({errors})."
    if actual_artifact is None:
        return f"{requirement}: {link.verdict}; expected evidence comes from {expected_label}, but no actual artifact was bound ({errors})."
    return f"{requirement}: {link.verdict}; compare {expected_label} with {actual_label} ({errors})."


def _artifact_label(artifact: Any) -> str:
    if artifact is None:
        return "none"
    if artifact.program:
        return f"{artifact.artifact_id}/{artifact.program}"
    source = str(getattr(artifact, "source", "") or "")
    stage = source.rsplit(":", 1)[-1] if ":" in source else ""
    if stage and stage != str(artifact.artifact_id):
        return f"{artifact.artifact_id}/{stage}"
    return str(artifact.artifact_id)


def _artifact_summary_preview(artifact: Any) -> Any:
    if artifact is None:
        return None
    return _artifact_payload_preview(artifact.payload)


def _repair_policy_blocked(pipeline_result: PipelineResult) -> bool:
    repair_plan = pipeline_result.repair_plan
    repair = pipeline_result.repair
    if repair_plan is not None and repair_plan.scope == "policy_blocked":
        return True
    return repair is not None and repair.strategy == "policy_gate_abstain"


def _has_runtime_failure(pipeline_result: PipelineResult) -> bool:
    if pipeline_result.execution_exception_type:
        return True
    return any(error.code == "execution_error" for error in pipeline_result.report.errors)


def _runtime_failure_reason(pipeline_result: PipelineResult) -> str:
    if pipeline_result.execution_exception_message:
        return pipeline_result.execution_exception_message
    for error in pipeline_result.report.errors:
        if error.code == "execution_error":
            return error.message
    return "Code execution failed before verifiable artifacts were produced."


def _primary_failure_family(
    *,
    failed_requirement_types: list[str],
    runtime_failure: bool,
    abstained_requirement_ids: list[str],
) -> str:
    if runtime_failure:
        return "runtime_compatibility"
    normalized_types = sorted(dict.fromkeys(failed_requirement_types))
    if len(normalized_types) == 1:
        return normalized_types[0]
    if len(normalized_types) > 1:
        return "mixed"
    if abstained_requirement_ids:
        return "trace_binding"
    return "none"


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
        "match_policy": error.match_policy,
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
