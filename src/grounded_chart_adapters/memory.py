from __future__ import annotations

from collections.abc import Iterable

from grounded_chart import GroundedChartPipeline
from grounded_chart.trace_runner import MatplotlibTraceRunner
from grounded_chart_adapters.batch import BatchRunResult, BatchRunner
from grounded_chart_adapters.base import AdapterRunResult, ChartCase


class InMemoryCaseAdapter:
    """Small adapter for tests and hand-written benchmark probes."""

    def __init__(self, cases: Iterable[ChartCase]) -> None:
        self._cases = tuple(cases)

    def iter_cases(self) -> Iterable[ChartCase]:
        yield from self._cases

    def run(self, pipeline: GroundedChartPipeline, trace_runner: MatplotlibTraceRunner | None = None) -> list[AdapterRunResult]:
        runner = trace_runner or MatplotlibTraceRunner()
        results: list[AdapterRunResult] = []
        for case in self.iter_cases():
            source_path = case.metadata.get("source_code")
            execution_dir = case.metadata.get("execution_dir")
            run_trace = runner.run_code_with_figure(
                case.generated_code,
                globals_dict={"rows": list(case.rows)},
                execution_dir=execution_dir,
                file_path=source_path,
            )
            pipeline_result = pipeline.run(
                query=case.query,
                schema=case.schema,
                rows=case.rows,
                actual_trace=run_trace.plot_trace,
                generated_code=case.generated_code,
                expected_figure=case.figure_requirements,
                actual_figure=run_trace.figure_trace,
                verify_data=case.verification_mode == "full",
                execution_dir=execution_dir,
                file_path=source_path,
            )
            results.append(AdapterRunResult(case=case, pipeline_result=pipeline_result))
        return results

    def run_batch(
        self,
        pipeline: GroundedChartPipeline,
        trace_runner: MatplotlibTraceRunner | None = None,
        continue_on_error: bool = True,
    ) -> BatchRunResult:
        return BatchRunner(
            pipeline=pipeline,
            trace_runner=trace_runner,
            continue_on_error=continue_on_error,
        ).run(self)
