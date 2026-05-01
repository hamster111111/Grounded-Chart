from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    matplot_agent_root = Path(args.matplot_agent_root)
    benchmark_root = matplot_agent_root / "benchmark_data"
    workspace_root = matplot_agent_root / "workspace_full"
    instruction_path = benchmark_root / "benchmark_instructions.json"
    eval_results_path = workspace_root / "eval_results.json"
    output_path = project_root / "benchmarks" / args.output

    instructions = {
        str(record["id"]): record
        for record in json.loads(instruction_path.read_text(encoding="utf-8"))
    }
    eval_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
    details = {str(case_id): detail for case_id, detail in eval_results.get("details", {}).items()}

    records: list[dict[str, Any]] = []
    for case_id, detail in sorted(details.items(), key=lambda item: int(item[0])):
        score = detail.get("score")
        if score is None or float(score) >= args.fail_below:
            continue
        instruction = instructions.get(case_id)
        if instruction is None:
            raise KeyError(f"Instruction missing for MatPlotBench case {case_id}")
        workspace_dir = workspace_root / f"example_{case_id}"
        selected_code_path = select_code_path(workspace_dir)
        output_images = [
            path
            for path in (workspace_dir / "novice_final.png", workspace_dir / "novice.png")
            if path.exists()
        ]
        records.append(
            {
                "case_id": f"matplotbench-ds-failed-{case_id}",
                "native_id": int(case_id),
                "benchmark": "MatPlotBench",
                "failure_source": "MatPlotAgent workspace_full DeepSeek run",
                "failure_definition": f"score < {args.fail_below}",
                "score": score,
                "eval_error": detail.get("error"),
                "eval_raw": detail.get("raw"),
                "simple_instruction": instruction["simple_instruction"],
                "expert_instruction": instruction["expert_instruction"],
                "query": instruction["simple_instruction"],
                "query_source": "simple_instruction",
                "schema": {"columns": {}},
                "workspace_dir": str(workspace_dir),
                "selected_code_path": str(selected_code_path) if selected_code_path else None,
                "selected_log_path": str(selected_code_path) + ".log" if selected_code_path and Path(str(selected_code_path) + ".log").exists() else None,
                "output_image_paths": [str(path) for path in output_images],
                "ground_truth_path": str(benchmark_root / "ground_truth" / f"example_{case_id}.png"),
                "data_dir": str(benchmark_root / "data" / case_id) if (benchmark_root / "data" / case_id).exists() else None,
                "metadata": {
                    "source_instruction_path": str(instruction_path),
                    "source_eval_results_path": str(eval_results_path),
                    "source_workspace_root": str(workspace_root),
                    "eval_summary": eval_results.get("summary", {}),
                    "query_source": "simple_instruction",
                    "expert_instruction_role": "metadata_only",
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "output": str(output_path),
        "total_eval_cases": len(details),
        "failed_cases": len(records),
        "fail_below": args.fail_below,
        "native_ids": [record["native_id"] for record in records],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the native MatPlotBench DeepSeek failed-case manifest.")
    parser.add_argument(
        "--matplot-agent-root",
        default=r"D:\Code\autoReaserch\MatPlotAgent",
        help="Path to the MatPlotAgent repository containing benchmark_data and workspace_full.",
    )
    parser.add_argument(
        "--output",
        default="matplotbench_ds_failed_native.json",
        help="Output filename under GroundedChart/benchmarks.",
    )
    parser.add_argument("--fail-below", type=float, default=50.0)
    return parser.parse_args()


def select_code_path(workspace_dir: Path) -> Path | None:
    code_paths = sorted(workspace_dir.glob("code_action_*.py"), key=code_rank)
    return code_paths[-1] if code_paths else None


def code_rank(path: Path) -> tuple[int, int, str]:
    name = path.name
    stage_rank = 1 if "_vis_refined_" in name else 0
    iteration = 0
    try:
        iteration = int(name.rsplit("_", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        pass
    return stage_rank, iteration, name


if __name__ == "__main__":
    main()

