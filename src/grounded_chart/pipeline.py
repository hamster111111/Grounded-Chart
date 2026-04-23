from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from grounded_chart.canonical_executor import CanonicalExecutor, Row
from grounded_chart.evidence import (
    attach_figure_requirements,
    build_evidence_graph,
    build_requirement_plan,
    derive_expected_figure,
    merge_expected_figure_specs,
)
from grounded_chart.intent_parser import IntentParser
from grounded_chart.repair_loop import BoundedRepairLoop
from grounded_chart.repairer import Repairer
from grounded_chart.schema import FigureRequirementSpec, FigureTrace, PipelineResult, PlotTrace, TableSchema, VerificationError, VerificationReport
from grounded_chart.trace_runner import MatplotlibTraceRunner
from grounded_chart.verifier import OperatorLevelVerifier


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
    ) -> None:
        self.parser = parser
        self.executor = executor or CanonicalExecutor()
        self.verifier = verifier or OperatorLevelVerifier()
        self.repairer = repairer
        self.trace_runner = trace_runner or MatplotlibTraceRunner()
        self.enable_bounded_repair_loop = enable_bounded_repair_loop
        self.max_repair_rounds = max_repair_rounds

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
    ) -> PipelineResult:
        parsed = self._parse(query, schema)
        plan = parsed.plan
        parser_expected_figure = derive_expected_figure(parsed.requirement_plan)
        merged_expected_figure = merge_expected_figure_specs(parser_expected_figure, expected_figure)
        requirement_plan = attach_figure_requirements(parsed.requirement_plan, expected_figure=expected_figure)
        expected_trace = self.executor.execute(plan, rows)
        report = self.verifier.verify(
            expected_trace,
            actual_trace,
            expected_figure=merged_expected_figure,
            actual_figure=actual_figure,
            verify_data=verify_data,
            enforce_order=plan.sort is not None,
        )
        evidence_graph = build_evidence_graph(
            requirement_plan,
            report,
            expected_trace,
            actual_trace,
            actual_figure,
        )
        repair = None
        repair_plan = None
        if self.repairer is not None and not report.ok:
            repair = self.repairer.propose(generated_code, plan, report)
            repair_plan = repair.repair_plan
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
        )
        if self.enable_bounded_repair_loop and self.repairer is not None and repair_plan is not None and repair_plan.should_repair:
            loop = BoundedRepairLoop(
                parser=self.parser,
                executor=self.executor,
                verifier=self.verifier,
                repairer=self.repairer,
                trace_runner=self.trace_runner,
                max_rounds=self.max_repair_rounds,
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
    ) -> PipelineResult:
        parsed = self._parse(query, schema)
        plan = parsed.plan
        parser_expected_figure = derive_expected_figure(parsed.requirement_plan)
        merged_expected_figure = merge_expected_figure_specs(parser_expected_figure, expected_figure)
        requirement_plan = attach_figure_requirements(parsed.requirement_plan, expected_figure=expected_figure)
        expected_trace = self.executor.execute(plan, rows)
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
        evidence_graph = build_evidence_graph(
            requirement_plan,
            report,
            expected_trace,
            actual_trace,
            None,
        )
        repair = None
        repair_plan = None
        if self.repairer is not None:
            repair = self.repairer.propose(generated_code, plan, report)
            repair_plan = repair.repair_plan
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
            execution_exception_type=type(execution_exception).__name__,
            execution_exception_message=str(execution_exception),
        )
        if self.enable_bounded_repair_loop and self.repairer is not None and repair_plan is not None and repair_plan.should_repair:
            loop = BoundedRepairLoop(
                parser=self.parser,
                executor=self.executor,
                verifier=self.verifier,
                repairer=self.repairer,
                trace_runner=self.trace_runner,
                max_rounds=self.max_repair_rounds,
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


class _ParsedFallback:
    def __init__(self, *, plan, requirement_plan) -> None:
        self.plan = plan
        self.requirement_plan = requirement_plan
