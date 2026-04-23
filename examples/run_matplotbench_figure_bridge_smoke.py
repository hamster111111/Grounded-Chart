from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from grounded_chart import LLMIntentParser, OpenAICompatibleLLMClient, TableSchema, derive_expected_figure, load_ablation_run_config
from grounded_chart.requirements import ChartRequirementPlan, PanelRequirementPlan, RequirementNode
from grounded_chart_adapters import ChartCase, InMemoryCaseAdapter, write_batch_report_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small figure-only smoke on MatPlotBench failed-native cases.")
    parser.add_argument(
        "--config",
        default="configs/llm_ablation.deepseek.yaml",
        help="Config path used to load the OpenAI-compatible parser provider.",
    )
    parser.add_argument(
        "--bench",
        default="benchmarks/matplotbench_ds_failed_native.json",
        help="Benchmark JSON case list.",
    )
    parser.add_argument(
        "--selection-source",
        default="outputs/matplotbench_requirement_extraction/parser_compare/llm_rebuilt_report.json",
        help="Previously rebuilt LLM requirement report used to rank high-signal cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of cases to run.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=(),
        help="Optional explicit case IDs. If provided, ranking is skipped.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/matplotbench_figure_bridge_smoke_deepseek",
        help="Directory for report artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_ablation_run_config(project_root / args.config)
    provider = config.parser_provider or config.repair_provider
    if provider is None:
        raise ValueError("No parser provider found in config. Fill llm.default or llm.parser first.")

    parser = LLMIntentParser(OpenAICompatibleLLMClient(provider))
    pipeline = _build_pipeline(parser)

    bench_cases = _load_native_failed_cases(project_root / args.bench)
    selection = _select_cases(
        bench_cases=bench_cases,
        selection_source=project_root / args.selection_source,
        limit=max(1, int(args.limit)),
        explicit_case_ids=tuple(args.case_ids),
    )

    selected_cases = tuple(
        ChartCase(
            case_id=case.case_id,
            query=case.query,
            schema=case.schema,
            rows=case.rows,
            generated_code=case.generated_code,
            figure_requirements=case.figure_requirements,
            verification_mode="figure_only",
            metadata={
                **case.metadata,
                "smoke_selection": {
                    "bridge_score": item["bridge_score"],
                    "rank": item["rank"],
                    "selection_reason": item["selection_reason"],
                    "expected_figure_preview": item["expected_figure_preview"],
                },
            },
        )
        for item in selection
        for case in (bench_cases[item["case_id"]],)
    )

    adapter = InMemoryCaseAdapter(selected_cases)
    batch = adapter.run_batch(pipeline, continue_on_error=True)

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    batch.report.write_json(output_dir / "report.json")
    batch.report.write_jsonl(output_dir / "cases.jsonl")
    write_batch_report_html(
        batch.report,
        output_dir / "report.html",
        title="GroundedChart MatPlotBench Figure Bridge Smoke",
    )

    smoke_summary = _build_smoke_summary(batch.report.to_dict(), selection)
    (output_dir / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "smoke_summary.json").write_text(json.dumps(smoke_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Config:", project_root / args.config)
    print("Bench:", project_root / args.bench)
    print("Selection source:", project_root / args.selection_source)
    print("Output:", output_dir)
    print("Selected cases:", [item["case_id"] for item in selection])
    print("Summary:", json.dumps(smoke_summary["summary"], ensure_ascii=False))
    for case in smoke_summary["cases"]:
        print(
            case["case_id"],
            case["status"],
            "figure_failed=",
            len(case["failed_figure_requirement_ids"]),
            "codes=",
            case["figure_error_codes"],
        )


def _build_pipeline(parser: LLMIntentParser):
    from grounded_chart import GroundedChartPipeline

    return GroundedChartPipeline(
        parser=parser,
        repairer=None,
        enable_bounded_repair_loop=False,
    )


def _select_cases(
    *,
    bench_cases: dict[str, ChartCase],
    selection_source: Path,
    limit: int,
    explicit_case_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    if explicit_case_ids:
        selection = []
        for rank, case_id in enumerate(explicit_case_ids, start=1):
            if case_id not in bench_cases:
                raise KeyError(f"Case not found in bench: {case_id}")
            selection.append(
                {
                    "case_id": case_id,
                    "rank": rank,
                    "bridge_score": None,
                    "selection_reason": "explicit_case_id",
                    "expected_figure_preview": None,
                }
            )
        return selection

    report = json.loads(selection_source.read_text(encoding="utf-8"))
    ranked: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        case_id = str(case["case_id"])
        if case_id not in bench_cases:
            continue
        expected_figure = _derive_expected_figure_from_raw_plan(case["requirement_plan"])
        bridge_score = _bridge_score(expected_figure)
        if bridge_score <= 0:
            continue
        ranked.append(
            {
                "case_id": case_id,
                "bridge_score": bridge_score,
                "selection_reason": "top_bridge_score_from_saved_llm_report",
                "expected_figure_preview": _figure_preview(expected_figure),
                "benchmark_score": case.get("score"),
                "native_id": case.get("native_id"),
            }
        )

    ranked.sort(
        key=lambda item: (
            -int(item["bridge_score"]),
            float(item["benchmark_score"]) if item.get("benchmark_score") is not None else float("inf"),
            int(item["native_id"]) if item.get("native_id") is not None else 10**9,
            item["case_id"],
        )
    )
    selected = ranked[:limit]
    for index, item in enumerate(selected, start=1):
        item["rank"] = index
    return selected


def _load_native_failed_cases(path: Path) -> dict[str, ChartCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("Native failed-case manifest must be a JSON list.")

    cases: dict[str, ChartCase] = {}
    for raw in raw_cases:
        case_id = str(raw["case_id"])
        code_path = Path(str(raw["selected_code_path"]))
        generated_code = code_path.read_text(encoding="utf-8", errors="replace")
        metadata = dict(raw.get("metadata", {}))
        metadata.update(
            {
                "benchmark": raw.get("benchmark"),
                "native_id": raw.get("native_id"),
                "score": raw.get("score"),
                "ground_truth_path": raw.get("ground_truth_path"),
                "output_image_paths": list(raw.get("output_image_paths", [])),
                "workspace_dir": raw.get("workspace_dir"),
                "source_code": str(code_path),
                "execution_dir": str(code_path.parent),
                "selected_log_path": raw.get("selected_log_path"),
                "failure_definition": raw.get("failure_definition"),
                "eval_raw": raw.get("eval_raw"),
                "eval_error": raw.get("eval_error"),
            }
        )
        cases[case_id] = ChartCase(
            case_id=case_id,
            query=raw["query"],
            schema=TableSchema(columns=dict(raw.get("schema", {}).get("columns", {}))),
            rows=(),
            generated_code=generated_code,
            figure_requirements=None,
            verification_mode="figure_only",
            metadata=metadata,
        )
    return cases


def _derive_expected_figure_from_raw_plan(raw_plan: dict[str, Any]):
    requirements = tuple(
        RequirementNode(
            requirement_id=item["requirement_id"],
            scope=item["scope"],
            type=item["type"],
            name=item["name"],
            value=item["value"],
            source_span=item.get("source_span", ""),
            status=item.get("status", "explicit"),
            confidence=item.get("confidence"),
            depends_on=tuple(item.get("depends_on", [])),
            priority=item.get("priority", "core"),
            panel_id=item.get("panel_id"),
            assumption=item.get("assumption"),
        )
        for item in raw_plan.get("requirements", [])
    )
    panels = tuple(
        PanelRequirementPlan(
            panel_id=panel["panel_id"],
            chart_type=panel.get("chart_type", "unknown"),
            requirement_ids=tuple(panel.get("requirement_ids", [])),
            data_ops=dict(panel.get("data_ops", {})),
            encodings=dict(panel.get("encodings", {})),
            annotations=dict(panel.get("annotations", {})),
            presentation_constraints=dict(panel.get("presentation_constraints", {})),
        )
        for panel in raw_plan.get("panels", [])
    )
    requirement_plan = ChartRequirementPlan(
        requirements=requirements,
        panels=panels,
        figure_requirements=dict(raw_plan.get("figure_requirements", {})),
        shared_requirement_ids=tuple(raw_plan.get("shared_requirement_ids", [])),
        raw_query=raw_plan.get("raw_query", ""),
    )
    return derive_expected_figure(requirement_plan)


def _bridge_score(expected_figure: Any) -> int:
    if expected_figure is None:
        return 0
    score = 0
    if expected_figure.axes_count is not None:
        score += 3
    if expected_figure.figure_title:
        score += 2
    for axis in expected_figure.axes:
        if axis.title:
            score += 2
        if axis.xlabel:
            score += 1
        if axis.ylabel:
            score += 1
        if axis.zlabel:
            score += 1
        score += len(axis.artist_types)
    return score


def _figure_preview(expected_figure: Any) -> dict[str, Any] | None:
    if expected_figure is None:
        return None
    return {
        "axes_count": expected_figure.axes_count,
        "figure_title": expected_figure.figure_title,
        "axes": [
            {
                "axis_index": axis.axis_index,
                "title": axis.title,
                "xlabel": axis.xlabel,
                "ylabel": axis.ylabel,
                "zlabel": axis.zlabel,
                "artist_types": list(axis.artist_types),
            }
            for axis in expected_figure.axes
        ],
    }


def _build_smoke_summary(report: dict[str, Any], selection: list[dict[str, Any]]) -> dict[str, Any]:
    selection_by_case = {item["case_id"]: item for item in selection}
    cases = []
    total_figure_failures = 0
    total_figure_passes = 0
    total_figure_requirements = 0

    for case in report.get("cases", []):
        evidence_graph = case.get("evidence_graph") or {}
        links = evidence_graph.get("links", [])
        figure_links = [
            link
            for link in links
            if link.get("expected_artifact_id") == "expected.figure_requirements"
            or link.get("actual_artifact_id") == "actual.figure_trace"
        ]
        failed_figure_requirement_ids = [
            link["requirement_id"]
            for link in figure_links
            if link.get("verdict") == "fail"
        ]
        passed_figure_requirement_ids = [
            link["requirement_id"]
            for link in figure_links
            if link.get("verdict") == "pass"
        ]
        figure_error_codes = sorted(
            {
                code
                for link in figure_links
                for code in link.get("error_codes", [])
            }
        )
        total_figure_failures += len(failed_figure_requirement_ids)
        total_figure_passes += len(passed_figure_requirement_ids)
        total_figure_requirements += len(figure_links)
        cases.append(
            {
                "case_id": case["case_id"],
                "rank": selection_by_case.get(case["case_id"], {}).get("rank"),
                "bridge_score": selection_by_case.get(case["case_id"], {}).get("bridge_score"),
                "status": case["status"],
                "figure_requirement_count": len(figure_links),
                "failed_figure_requirement_ids": failed_figure_requirement_ids,
                "passed_figure_requirement_ids": passed_figure_requirement_ids,
                "figure_error_codes": figure_error_codes,
                "expected_figure": case.get("figure_requirements"),
                "actual_figure": case.get("actual_figure"),
                "case_metadata": case.get("case_metadata"),
            }
        )

    return {
        "summary": {
            "selected_cases": len(selection),
            "completed_cases": len(report.get("cases", [])),
            "passed_cases": sum(1 for case in cases if case["status"] == "passed"),
            "failed_cases": sum(1 for case in cases if case["status"] == "failed"),
            "errored_cases": sum(1 for case in cases if case["status"] == "error"),
            "total_figure_requirements": total_figure_requirements,
            "total_figure_passes": total_figure_passes,
            "total_figure_failures": total_figure_failures,
        },
        "selection": selection,
        "cases": cases,
    }


if __name__ == "__main__":
    main()
