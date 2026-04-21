from __future__ import annotations

from typing import Iterable

from grounded_chart.canonical_executor import CanonicalExecutor, Row
from grounded_chart.evidence import build_evidence_graph, build_requirement_plan
from grounded_chart.intent_parser import IntentParser
from grounded_chart.repair_loop import BoundedRepairLoop
from grounded_chart.repairer import Repairer
from grounded_chart.schema import FigureRequirementSpec, FigureTrace, PipelineResult, PlotTrace, TableSchema
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
        plan = self.parser.parse(query, schema)
        requirement_plan = build_requirement_plan(plan, expected_figure=expected_figure)
        expected_trace = self.executor.execute(plan, rows)
        report = self.verifier.verify(
            expected_trace,
            actual_trace,
            expected_figure=expected_figure,
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
            expected_figure=expected_figure,
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
