from pathlib import Path

from grounded_chart_adapters import MatplotBenchWorkspaceAdapter


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    matplot_agent_root = project_root.parent / "MatPlotAgent"
    benchmark_root = matplot_agent_root / "benchmark_data"
    workspace_root = matplot_agent_root / "workspace_full"
    output_path = project_root / "outputs" / "matplotbench_workspace" / "low_score_candidates.json"

    adapter = MatplotBenchWorkspaceAdapter(benchmark_root=benchmark_root, workspace_root=workspace_root)
    print("Summary:", adapter.summary())
    candidates = adapter.write_candidate_manifest(output_path, score_lte=50, limit=20, include_instruction=True)
    print("Manifest:", output_path)
    for record in candidates[:10]:
        print(
            record.case_id,
            "score=",
            record.score,
            "stage=",
            record.selected_code_stage,
            "iter=",
            record.selected_code_iteration,
            "code=",
            record.selected_code_path,
        )


if __name__ == "__main__":
    main()
