from pathlib import Path

from grounded_chart import GroundedChartPipeline, HeuristicIntentParser, RuleBasedRepairer
from grounded_chart_adapters import BatchRunner, JsonCaseAdapter, write_batch_report_html


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bench_path = project_root / "benchmarks" / "repair_loop_bench.json"
    output_dir = project_root / "outputs" / "repair_loop_bench"
    adapter = JsonCaseAdapter(bench_path)
    pipeline = GroundedChartPipeline(
        parser=HeuristicIntentParser(),
        repairer=RuleBasedRepairer(),
        repair_policy_mode="strict",
        enable_bounded_repair_loop=True,
        max_repair_rounds=2,
    )
    batch = BatchRunner(pipeline, continue_on_error=True).run(adapter)

    output_dir.mkdir(parents=True, exist_ok=True)
    batch.report.write_json(output_dir / "report.json")
    batch.report.write_jsonl(output_dir / "cases.jsonl")
    write_batch_report_html(
        batch.report,
        output_dir / "report.html",
        title="GroundedChart Repair Loop Bench",
    )

    repaired = 0
    improved = 0
    for case_report in batch.report.cases:
        attempts = list(case_report.repair_attempts)
        if attempts:
            repaired += 1
        if any(attempt.get("resolved_requirement_ids") for attempt in attempts):
            improved += 1

    print("Bench:", bench_path)
    print("Report:", output_dir / "report.json")
    print("HTML:", output_dir / "report.html")
    print("Summary:", batch.report.summary.to_dict())
    print("Cases with repair attempts:", repaired)
    print("Cases with resolved requirements:", improved)
    for case_report in batch.report.cases:
        print(
            case_report.case_id,
            case_report.status,
            "errors=",
            list(case_report.error_codes),
            "repair=",
            case_report.repair_scope,
            "attempts=",
            len(case_report.repair_attempts),
        )


if __name__ == "__main__":
    main()
