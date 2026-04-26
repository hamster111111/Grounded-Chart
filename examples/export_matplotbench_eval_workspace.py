from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_chart_adapters import MatplotBenchEvalWorkspaceExporter


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    summary_path = _resolve_path(args.summary, project_root)
    workspace_dir = _resolve_path(args.workspace, project_root)
    exporter = MatplotBenchEvalWorkspaceExporter(
        summary_path,
        workspace_dir,
        overwrite=args.overwrite,
    )
    manifest = exporter.export()
    manifest_path = workspace_dir / "groundedchart_eval_export_manifest.json"
    print("Workspace:", workspace_dir)
    print("Export manifest:", manifest_path)
    print("Exported cases:", manifest["exported_cases"], "/", manifest["total_cases"])
    print("Native ids:", manifest["native_ids"])
    print("Run these from MatPlotAgent root for non-contiguous subsets:")
    for command in manifest["eval_qwen_commands_by_id"]:
        print("  ", command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GroundedChart MatPlotBench generation outputs to MatPlotBench evaluator workspace layout.")
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to matplotbench_generation_summary.json from run_matplotbench_generation_pipeline.py.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Output evaluator workspace directory, absolute or relative to project root.",
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args()


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


if __name__ == "__main__":
    main()