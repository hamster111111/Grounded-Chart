from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grounded_chart.api import GroundedChartPipeline
from grounded_chart.runtime.trace_runner import MatplotlibTraceRunner
from grounded_chart_adapters.base import AdapterRunResult, BenchmarkAdapter
from grounded_chart_adapters.reporting import BatchReport, case_report_from_exception, case_report_from_result


@dataclass(frozen=True)
class BatchRunResult:
    """Raw successful pipeline outputs plus benchmark-facing reports."""

    run_results: tuple[AdapterRunResult, ...]
    report: BatchReport


class BatchRunner:
    """Run any benchmark adapter through the GroundedChart pipeline."""

    def __init__(
        self,
        pipeline: GroundedChartPipeline,
        trace_runner: MatplotlibTraceRunner | None = None,
        continue_on_error: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.trace_runner = trace_runner or MatplotlibTraceRunner()
        self.continue_on_error = continue_on_error

    def run(self, adapter: BenchmarkAdapter) -> BatchRunResult:
        run_results: list[AdapterRunResult] = []
        case_reports = []
        for case in adapter.iter_cases():
            try:
                source_path = case.metadata.get("source_code")
                execution_dir = case.metadata.get("execution_dir")
                if execution_dir is None and source_path:
                    execution_dir = str(Path(source_path).resolve().parent)
                verify_data = case.verification_mode in {"full", "figure_and_data"} or case.expected_trace is not None
                run_trace = self.trace_runner.run_code_with_figure(
                    case.generated_code,
                    globals_dict={"rows": list(case.rows)},
                    execution_dir=execution_dir,
                    file_path=source_path,
                )
                pipeline_result = self.pipeline.run(
                    query=case.query,
                    schema=case.schema,
                    rows=case.rows,
                    actual_trace=run_trace.plot_trace,
                    generated_code=case.generated_code,
                    expected_figure=case.figure_requirements,
                    actual_figure=run_trace.figure_trace,
                    verify_data=verify_data,
                    execution_dir=execution_dir,
                    file_path=source_path,
                    case_metadata=case.metadata,
                    parsed_requirements=case.parsed_requirements,
                    parse_source=case.parse_source,
                    expected_trace_override=case.expected_trace,
                )
                run_result = AdapterRunResult(case=case, pipeline_result=pipeline_result)
                run_results.append(run_result)
                case_reports.append(case_report_from_result(run_result))
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                if self.pipeline.enable_bounded_repair_loop and self.pipeline.repairer is not None:
                    source_path = case.metadata.get("source_code")
                    execution_dir = case.metadata.get("execution_dir")
                    if execution_dir is None and source_path:
                        execution_dir = str(Path(source_path).resolve().parent)
                    pipeline_result = self.pipeline.run_with_execution_error(
                        query=case.query,
                        schema=case.schema,
                        rows=case.rows,
                        generated_code=case.generated_code,
                        execution_exception=exc,
                        expected_figure=case.figure_requirements,
                        verify_data=verify_data,
                        execution_dir=execution_dir,
                        file_path=source_path,
                        case_metadata=case.metadata,
                        parsed_requirements=case.parsed_requirements,
                        parse_source=case.parse_source,
                        expected_trace_override=case.expected_trace,
                    )
                    run_result = AdapterRunResult(
                        case=case,
                        pipeline_result=pipeline_result,
                        metadata={
                            "initial_exception_type": type(exc).__name__,
                            "initial_exception_message": str(exc),
                        },
                    )
                    if pipeline_result.report.ok:
                        run_results.append(run_result)
                        case_reports.append(case_report_from_result(run_result))
                    else:
                        case_reports.append(case_report_from_result(run_result))
                else:
                    case_reports.append(case_report_from_exception(case, exc))
        return BatchRunResult(run_results=tuple(run_results), report=BatchReport.from_case_reports(case_reports))
