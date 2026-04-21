from pathlib import Path

from grounded_chart_adapters import MatplotBenchInstructionAdapter


def main() -> None:
    default_root = Path(__file__).resolve().parents[2] / "MatPlotAgent" / "benchmark_data"
    adapter = MatplotBenchInstructionAdapter(default_root)
    records = list(adapter.iter_records())
    print("MatPlotBench root:", default_root)
    print("records:", len(records))
    print("with external data:", sum(1 for record in records if record.has_external_data))
    print("with ground truth image:", sum(1 for record in records if record.has_ground_truth_image))
    for record in records[:3]:
        print(record.case_id, record.simple_instruction[:100].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
