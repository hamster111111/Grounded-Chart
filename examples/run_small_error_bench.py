from pathlib import Path

from grounded_chart.api import GroundedChartPipeline, HeuristicIntentParser, RuleBasedRepairer
from grounded_chart_adapters import BatchRunner, JsonCaseAdapter, write_batch_report_html


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bench_path = project_root / "benchmarks" / "small_error_bench.json"
    output_dir = project_root / "outputs" / "small_error_bench"
    adapter = JsonCaseAdapter(bench_path)
    pipeline = GroundedChartPipeline(
        parser=HeuristicIntentParser(),
        repairer=RuleBasedRepairer(),
        repair_policy_mode="strict",
    )
    batch = BatchRunner(pipeline, continue_on_error=True).run(adapter)

    output_dir.mkdir(parents=True, exist_ok=True)
    batch.report.write_json(output_dir / "report.json")
    batch.report.write_jsonl(output_dir / "cases.jsonl")
    write_batch_report_html(
        batch.report,
        output_dir / "report.html",
        title="GroundedChart Small Error Bench",
    )

    print("Bench:", bench_path)
    print("Report:", output_dir / "report.json")
    print("HTML:", output_dir / "report.html")
    print("Summary:", batch.report.summary.to_dict())
    for case_report in batch.report.cases:
        print(
            case_report.case_id,
            case_report.status,
            "errors=",
            list(case_report.error_codes),
            "repair=",
            case_report.repair_scope,
        )


if __name__ == "__main__":
    main()
