from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from grounded_chart.core.canonical_executor import CanonicalExecutor, Row
from grounded_chart.verification.diagnostics import failure_atoms_from_evidence_graph
from grounded_chart.verification.evidence import (
    attach_figure_requirements,
    bind_requirement_policy_to_verification,
    build_evidence_graph,
    build_requirement_plan,
    derive_expected_figure,
    merge_expected_figure_specs,
)
from grounded_chart.verification.intent_parser import IntentParser
from grounded_chart.repair.loop import BoundedRepairLoop
from grounded_chart.repair.policy import RuleBasedRepairPlanner, apply_auto_repair_gate, normalize_repair_policy_mode
from grounded_chart.repair.repairer import Repairer
from grounded_chart.core.schema import (
    FigureRequirementSpec,
    FigureTrace,
    ParsedRequirementBundle,
    PipelineResult,
    PlotTrace,
    RepairPatch,
    TableSchema,
    VerificationError,
    VerificationReport,
)
from grounded_chart.runtime.trace_runner import MatplotlibTraceRunner
from grounded_chart.verification.verifier import OperatorLevelVerifier


class GroundedChartPipeline:
    """End-to-end verification pipeline for an already generated chart."""

    def __init__(
        self,
        parser: IntentParser,
        executor: CanonicalExecutor | None = None,
        verifier: OperatorLevelVerifier | None = None,
        repairer: Repairer | None = None,
        trace_runner: MatplotlibTraceRunner | None = None,
        enable_bounded_repair_loop: bool = False,
        max_repair_rounds: int = 2,
        repair_policy_mode: str = "exploratory",
        repair_planner: RuleBasedRepairPlanner | None = None,
    ) -> None:
        self.parser = parser
        self.executor = executor or CanonicalExecutor()
        self.verifier = verifier or OperatorLevelVerifier()
        self.repairer = repairer
        self.trace_runner = trace_runner or MatplotlibTraceRunner()
        self.enable_bounded_repair_loop = enable_bounded_repair_loop
        self.max_repair_rounds = max_repair_rounds
        self.repair_policy_mode = normalize_repair_policy_mode(repair_policy_mode)
        self.repair_planner = repair_planner or RuleBasedRepairPlanner()

    def run(
        self,
        query: str,
        schema: TableSchema,
        rows: Iterable[Row],
        actual_trace: PlotTrace,
        generated_code: str = "",
        expected_figure: FigureRequirementSpec | None = None,
        actual_figure: FigureTrace | None = None,
        verify_data: bool = True,
        execution_dir: str | None = None,
        file_path: str | None = None,
        case_metadata: dict | None = None,
        parsed_requirements: ParsedRequirementBundle | None = None,
        parse_source: str | None = None,
        expected_trace_override: PlotTrace | None = None,
    ) -> PipelineResult:
        parsed = parsed_requirements or self._parse(query, schema)
        parse_source = self._resolve_parse_source(parse_source, parsed_requirements=parsed_requirements)
        plan = parsed.plan
        parser_expected_figure = derive_expected_figure(parsed.requirement_plan)
        merged_expected_figure = merge_expected_figure_specs(parser_expected_figure, expected_figure)
        requirement_plan = attach_figure_requirements(parsed.requirement_plan, expected_figure=expected_figure)
        expected_trace = expected_trace_override or self.executor.execute(plan, rows)
        report = self.verifier.verify(
            expected_trace,
            actual_trace,
            expected_figure=merged_expected_figure,
            actual_figure=actual_figure,
            verify_data=verify_data,
            enforce_order=plan.sort is not None,
        )
        report = bind_requirement_policy_to_verification(report, requirement_plan)
        evidence_graph = build_evidence_graph(
            requirement_plan,
            report,
            expected_trace,
            actual_trace,
            actual_figure,
        )
        repair = None
        repair_plan = None
        repair_loop_status = None
        repair_loop_reason = None
        if self.repairer is not None and not report.ok:
            repair_plan, repair = self._prepare_repair(
                generated_code=generated_code,
                plan=plan,
                report=report,
                case_metadata=case_metadata,
                evidence_graph=evidence_graph,
            )
            if repair is not None and (repair_plan is None or not repair_plan.should_repair):
                repair_loop_status = repair.loop_signal
                repair_loop_reason = repair.loop_reason or repair.instruction
        result = PipelineResult(
            plan=plan,
            expected_trace=expected_trace,
            actual_trace=actual_trace,
            report=report,
            expected_figure=merged_expected_figure,
            actual_figure=actual_figure,
            requirement_plan=requirement_plan,
            evidence_graph=evidence_graph,
            repair_plan=repair_plan,
            repair=repair,
            repair_loop_status=repair_loop_status,
            repair_loop_reason=repair_loop_reason,
            parse_source=parse_source,
            parser_raw_response=dict(getattr(parsed, "raw_response", {}) or {}),
        )
        if self.enable_bounded_repair_loop and self.repairer is not None and repair_plan is not None and repair_plan.should_repair:
            loop = BoundedRepairLoop(
                parser=self.parser,
                executor=self.executor,
                verifier=self.verifier,
                repairer=self.repairer,
                trace_runner=self.trace_runner,
                max_rounds=self.max_repair_rounds,
                repair_policy_mode=self.repair_policy_mode,
            )
            return loop.run(
                query=query,
                schema=schema,
                rows=rows,
                initial_result=result,
                generated_code=generated_code,
                expected_figure=expected_figure,
                execution_dir=execution_dir,
                file_path=file_path,
                verify_data=verify_data,
            )
        return result

    def run_with_execution_error(
        self,
        *,
        query: str,
        schema: TableSchema,
        rows: Iterable[Row],
        generated_code: str,
        execution_exception: BaseException,
        expected_figure: FigureRequirementSpec | None = None,
        verify_data: bool = True,
        execution_dir: str | None = None,
        file_path: str | None = None,
        case_metadata: dict | None = None,
        parsed_requirements: ParsedRequirementBundle | None = None,
        parse_source: str | None = None,
        expected_trace_override: PlotTrace | None = None,
    ) -> PipelineResult:
        parsed = parsed_requirements or self._parse(query, schema)
        parse_source = self._resolve_parse_source(parse_source, parsed_requirements=parsed_requirements)
        plan = parsed.plan
        parser_expected_figure = derive_expected_figure(parsed.requirement_plan)
        merged_expected_figure = merge_expected_figure_specs(parser_expected_figure, expected_figure)
        requirement_plan = attach_figure_requirements(parsed.requirement_plan, expected_figure=expected_figure)
        expected_trace = expected_trace_override or self.executor.execute(plan, rows)
        actual_trace = PlotTrace(
            chart_type="unknown",
            points=(),
            source="execution_error",
            raw={
                "trace_error": "execution_error",
                "exception_type": type(execution_exception).__name__,
                "exception_message": str(execution_exception),
            },
        )
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="execution_error",
                    message=str(execution_exception),
                    expected=None,
                    actual=type(execution_exception).__name__,
                    operator="execution",
                    requirement_id="panel_0.chart_type",
                    severity="error",
                ),
            ),
            expected_trace=expected_trace,
            actual_trace=actual_trace,
            expected_figure=merged_expected_figure,
            actual_figure=None,
        )
        report = bind_requirement_policy_to_verification(report, requirement_plan)
        evidence_graph = build_evidence_graph(
            requirement_plan,
            report,
            expected_trace,
            actual_trace,
            None,
        )
        repair = None
        repair_plan = None
        repair_loop_status = None
        repair_loop_reason = None
        if self.repairer is not None:
            repair_plan, repair = self._prepare_repair(
                generated_code=generated_code,
                plan=plan,
                report=report,
                case_metadata=case_metadata,
                evidence_graph=evidence_graph,
            )
            if repair is not None and (repair_plan is None or not repair_plan.should_repair):
                repair_loop_status = repair.loop_signal
                repair_loop_reason = repair.loop_reason or repair.instruction
        result = PipelineResult(
            plan=plan,
            expected_trace=expected_trace,
            actual_trace=actual_trace,
            report=report,
            expected_figure=merged_expected_figure,
            actual_figure=None,
            requirement_plan=requirement_plan,
            evidence_graph=evidence_graph,
            repair_plan=repair_plan,
            repair=repair,
            repair_loop_status=repair_loop_status,
            repair_loop_reason=repair_loop_reason,
            execution_exception_type=type(execution_exception).__name__,
            execution_exception_message=str(execution_exception),
            parse_source=parse_source,
            parser_raw_response=dict(getattr(parsed, "raw_response", {}) or {}),
        )
        if self.enable_bounded_repair_loop and self.repairer is not None and repair_plan is not None and repair_plan.should_repair:
            loop = BoundedRepairLoop(
                parser=self.parser,
                executor=self.executor,
                verifier=self.verifier,
                repairer=self.repairer,
                trace_runner=self.trace_runner,
                max_rounds=self.max_repair_rounds,
                repair_policy_mode=self.repair_policy_mode,
            )
            repaired = loop.run(
                query=query,
                schema=schema,
                rows=rows,
                initial_result=result,
                generated_code=generated_code,
                expected_figure=merged_expected_figure,
                execution_dir=execution_dir,
                file_path=file_path,
                verify_data=verify_data,
            )
            return replace(
                repaired,
                execution_exception_type=type(execution_exception).__name__,
                execution_exception_message=str(execution_exception),
            )
        return result

    def _parse(self, query: str, schema: TableSchema):
        parse_requirements = getattr(self.parser, "parse_requirements", None)
        if callable(parse_requirements):
            return parse_requirements(query, schema)
        plan = self.parser.parse(query, schema)
        return _ParsedFallback(plan=plan, requirement_plan=build_requirement_plan(plan))

    def _resolve_parse_source(
        self,
        parse_source: str | None,
        *,
        parsed_requirements: ParsedRequirementBundle | None,
    ) -> str:
        normalized = str(parse_source or "").strip().lower()
        if normalized in {"predicted", "oracle"}:
            return normalized
        if parsed_requirements is not None:
            return "oracle"
        return "predicted"

    def _prepare_repair(
        self,
        *,
        generated_code: str,
        plan,
        report: VerificationReport,
        case_metadata: dict | None,
        evidence_graph=None,
    ) -> tuple:
        preview_plan = self.repair_planner.plan(report, generated_code=generated_code)
        effective_case_metadata = dict(case_metadata or {})
        diagnostic_repairability = _diagnostic_repairability_from_evidence(evidence_graph)
        if diagnostic_repairability is not None:
            effective_case_metadata["repairability"] = diagnostic_repairability
        gate = apply_auto_repair_gate(
            preview_plan,
            case_metadata=effective_case_metadata,
            mode=self.repair_policy_mode,
        )
        repair_plan = gate.effective_plan
        if gate.blocked_by_policy:
            return repair_plan, RepairPatch(
                strategy="policy_gate_abstain",
                instruction=repair_plan.reason or "Automatic repair blocked by policy.",
                target_error_codes=report.error_codes,
                repair_plan=repair_plan,
                loop_signal="stop",
                loop_reason=repair_plan.reason or "Automatic repair blocked by policy.",
            )

        try:
            repair = self.repairer.propose(generated_code, plan, report, evidence_graph=evidence_graph)
        except TypeError:
            repair = self.repairer.propose(generated_code, plan, report)
        return repair.repair_plan or repair_plan, repair


class _ParsedFallback:
    def __init__(self, *, plan, requirement_plan) -> None:
        self.plan = plan
        self.requirement_plan = requirement_plan
        self.raw_response = {}


def _diagnostic_repairability_from_evidence(evidence_graph) -> str | None:
    blocked_scopes = {"diagnose_only", "unsupported", "route_only"}
    failed_atoms = tuple(
        atom for atom in failure_atoms_from_evidence_graph(evidence_graph) if atom.verdict == "fail"
    )
    if not failed_atoms:
        return None
    scopes = tuple(atom.suggested_action_scope for atom in failed_atoms)
    if any(scope and scope not in blocked_scopes for scope in scopes):
        return None
    for scope in ("unsupported", "route_only", "diagnose_only"):
        if scope in scopes:
            return scope
    return None
