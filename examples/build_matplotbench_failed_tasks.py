import json
from pathlib import Path
from typing import Any


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_bench_path = project_root / "benchmarks" / "matplotbench_smoke_wrong.json"
    source_report_path = project_root / "outputs" / "matplotbench_smoke_wrong" / "report.json"
    output_bench_path = project_root / "benchmarks" / "matplotbench_failed_tasks.json"

    source_cases = _load_json_list(source_bench_path)
    source_cases_by_id = {str(case["case_id"]): case for case in source_cases}
    report = _load_json_dict(source_report_path)

    failed_reports = [
        case_report
        for case_report in report.get("cases", [])
        if case_report.get("status") != "passed"
    ]
    failed_cases: list[dict[str, Any]] = []
    missing_case_ids: list[str] = []
    for case_report in failed_reports:
        case_id = str(case_report["case_id"])
        source_case = source_cases_by_id.get(case_id)
        if source_case is None:
            missing_case_ids.append(case_id)
            continue
        failed_case = dict(source_case)
        metadata = dict(failed_case.get("metadata", {}))
        metadata.update(
            {
                "source_failure_report": str(source_report_path),
                "previous_status": case_report.get("status"),
                "previous_error_codes": list(case_report.get("error_codes", [])),
                "previous_repair_scope": case_report.get("repair_scope"),
                "previous_exception_type": case_report.get("exception_type"),
            }
        )
        failed_case["metadata"] = metadata
        failed_cases.append(failed_case)

    if missing_case_ids:
        raise KeyError(f"Failed report contains cases missing from source bench: {missing_case_ids}")

    output_bench_path.write_text(
        json.dumps(failed_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Source bench:", source_bench_path)
    print("Source report:", source_report_path)
    print("Output bench:", output_bench_path)
    print("Failed tasks:", len(failed_cases))
    for case in failed_cases:
        metadata = case.get("metadata", {})
        print(
            case["case_id"],
            "previous_status=",
            metadata.get("previous_status"),
            "previous_errors=",
            metadata.get("previous_error_codes"),
        )


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def _load_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


if __name__ == "__main__":
    main()
