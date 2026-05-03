from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from grounded_chart.api import LLMIntentParser, OpenAICompatibleLLMClient, load_ablation_run_config
from grounded_chart.core.schema import TableSchema


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_ablation_run_config(args.config)
    if config.parser_provider is None:
        raise ValueError("Parser provider is missing in config.")

    bench_path = project_root / "benchmarks" / args.bench
    output_dir = project_root / "outputs" / args.output_dir
    records = json.loads(bench_path.read_text(encoding="utf-8"))
    selected = select_records(
        records,
        case_ids=args.case_ids if not args.all_cases else [],
        limit=None if args.all_cases else args.limit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.json"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "report.html"
    progress_path = output_dir / "progress.json"

    cases: list[dict[str, Any]] = []
    parser_provider = config.parser_provider
    assert parser_provider is not None

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_record = {
            executor.submit(process_record, parser_provider, record, total=len(selected)): record
            for record in selected
        }
        completed = 0
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            completed += 1
            try:
                case_output = future.result()
            except Exception as exc:
                case_output = error_case_output(record, exc, completed=completed, total=len(selected))
                print(f"[{completed}/{len(selected)}] native_id={record.get('native_id')} error={type(exc).__name__}: {exc}")
            else:
                print(
                    f"[{completed}/{len(selected)}] native_id={record.get('native_id')} "
                    f"score={record.get('score')} requirements={len(case_output['requirements'])}"
                )
            cases.append(case_output)
            write_outputs(cases_path, summary_path, html_path, progress_path, cases)

    cases.sort(key=lambda item: int(item.get("native_id") or 0))
    write_outputs(cases_path, summary_path, html_path, progress_path, cases)

    print(
        json.dumps(
            {
                "cases": str(cases_path),
                "summary": str(summary_path),
                "html": str(html_path),
                "processed_native_ids": [item["native_id"] for item in cases],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect requirement extraction on native MatPlotBench failed cases.")
    parser.add_argument("--config", default="configs/llm_ablation.deepseek.yaml")
    parser.add_argument("--bench", default="matplotbench_ds_failed_native.json")
    parser.add_argument("--output-dir", default="matplotbench_requirement_extraction/full_run")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--all-cases", action="store_true", help="Run all records in the bench file.")
    parser.add_argument(
        "--case-ids",
        nargs="*",
        type=int,
        default=[5, 32, 37, 71],
        help="Native MatPlotBench ids to inspect. Ignored when --all-cases is set.",
    )
    return parser.parse_args()


def process_record(parser_provider, record: dict[str, Any], *, total: int) -> dict[str, Any]:
    parser = LLMIntentParser(OpenAICompatibleLLMClient(parser_provider))
    query = str(record.get("query") or record.get("simple_instruction") or record.get("expert_instruction") or "")
    bundle = parser.parse_requirements(query, TableSchema(columns={}))
    return {
        "case_id": record["case_id"],
        "native_id": record.get("native_id"),
        "score": record.get("score"),
        "simple_instruction": record.get("simple_instruction"),
        "expert_instruction": record.get("expert_instruction"),
        "plan": plan_to_dict(bundle.plan),
        "requirements": [requirement_to_dict(req) for req in bundle.requirement_plan.requirements],
        "raw_response": bundle.raw_response,
        "summary": summarize_case(record, bundle.requirement_plan.requirements),
        "progress": {"total": total},
    }


def error_case_output(record: dict[str, Any], exc: Exception, *, completed: int, total: int) -> dict[str, Any]:
    return {
        "case_id": record["case_id"],
        "native_id": record.get("native_id"),
        "score": record.get("score"),
        "simple_instruction": record.get("simple_instruction"),
        "expert_instruction": record.get("expert_instruction"),
        "plan": {
            "chart_type": "unknown",
            "dimensions": [],
            "measure": {"column": None, "agg": "none"},
            "filters": [],
            "sort": None,
            "limit": None,
            "confidence": None,
        },
        "requirements": [],
        "raw_response": None,
        "summary": {
            "requirement_count": 0,
            "status_counts": {"error": 1},
            "scope_counts": {},
            "explicit_without_span": 0,
            "grounded_span_count": 0,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        "progress": {"index": completed, "total": total, "error": True},
    }


def write_outputs(
    cases_path: Path,
    summary_path: Path,
    html_path: Path,
    progress_path: Path,
    cases: list[dict[str, Any]],
) -> None:
    ordered = sorted(cases, key=lambda item: int(item.get("native_id") or 0))
    summary = build_summary(ordered)
    cases_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(summary, ordered), encoding="utf-8")
    progress_path.write_text(
        json.dumps(
            {
                "completed_cases": len(ordered),
                "native_ids": [item.get("native_id") for item in ordered],
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def select_records(records: list[dict], *, case_ids: list[int], limit: int | None) -> list[dict]:
    if case_ids:
        wanted = {int(case_id) for case_id in case_ids}
        selected = [record for record in records if int(record.get("native_id")) in wanted]
        selected.sort(key=lambda record: int(record["native_id"]))
        return selected
    selected = sorted(records, key=lambda record: (float(record.get("score", 999)), int(record.get("native_id", 0))))
    return selected[:limit] if limit is not None else selected


def summarize_case(record: dict[str, Any], requirements: list[Any]) -> dict[str, Any]:
    expert_instruction = str(record.get("expert_instruction") or "")
    status_counts = Counter(requirement.status for requirement in requirements)
    scope_counts = Counter(requirement.scope for requirement in requirements)
    explicit_without_span = sum(1 for requirement in requirements if requirement.status == "explicit" and not requirement.source_span)
    grounded_span_count = sum(
        1
        for requirement in requirements
        if requirement.source_span and requirement.source_span.lower() in expert_instruction.lower()
    )
    return {
        "requirement_count": len(requirements),
        "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts),
        "explicit_without_span": explicit_without_span,
        "grounded_span_count": grounded_span_count,
    }


def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    chart_type_counts: Counter[str] = Counter()
    score_buckets: Counter[str] = Counter()
    total_requirements = 0
    explicit_without_span = 0
    grounded_spans = 0
    cases_with_unknown_chart_type = 0
    cases_with_only_secondary_requirements = 0

    for case in cases:
        summary = case["summary"]
        total_requirements += summary["requirement_count"]
        explicit_without_span += summary["explicit_without_span"]
        grounded_spans += summary["grounded_span_count"]
        status_counts.update(summary["status_counts"])
        scope_counts.update(summary["scope_counts"])
        chart_type_counts.update([case["plan"]["chart_type"]])
        score_buckets.update([score_bucket(case.get("score"))])
        if case["plan"]["chart_type"] == "unknown":
            cases_with_unknown_chart_type += 1
        core_names = {requirement["name"] for requirement in case["requirements"] if requirement["priority"] == "core"}
        if not core_names:
            cases_with_only_secondary_requirements += 1
        for requirement in case["requirements"]:
            name_counts.update([requirement["name"]])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_cases": len(cases),
        "total_requirements": total_requirements,
        "avg_requirements_per_case": round(total_requirements / len(cases), 2) if cases else 0.0,
        "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts),
        "chart_type_counts": dict(chart_type_counts),
        "score_buckets": dict(score_buckets),
        "explicit_without_span": explicit_without_span,
        "grounded_spans": grounded_spans,
        "cases_with_unknown_chart_type": cases_with_unknown_chart_type,
        "cases_with_only_secondary_requirements": cases_with_only_secondary_requirements,
        "top_requirement_names": dict(name_counts.most_common(25)),
    }


def score_bucket(score: Any) -> str:
    if score is None:
        return "missing"
    numeric = float(score)
    if numeric == 0:
        return "0"
    if numeric <= 20:
        return "1-20"
    if numeric <= 40:
        return "21-40"
    return "41-49"


def plan_to_dict(plan) -> dict:
    return {
        "chart_type": plan.chart_type,
        "dimensions": list(plan.dimensions),
        "measure": {"column": plan.measure.column, "agg": plan.measure.agg},
        "filters": [
            {"column": filter_spec.column, "op": filter_spec.op, "value": filter_spec.value}
            for filter_spec in plan.filters
        ],
        "sort": {"by": plan.sort.by, "direction": plan.sort.direction} if plan.sort is not None else None,
        "limit": plan.limit,
        "confidence": plan.confidence,
    }


def requirement_to_dict(requirement) -> dict:
    return {
        "requirement_id": requirement.requirement_id,
        "scope": requirement.scope,
        "type": requirement.type,
        "name": requirement.name,
        "value": requirement.value,
        "source_span": requirement.source_span,
        "status": requirement.status,
        "confidence": requirement.confidence,
        "depends_on": list(requirement.depends_on),
        "priority": requirement.priority,
        "panel_id": requirement.panel_id,
        "assumption": requirement.assumption,
    }


def render_html(summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    cards = "\n".join(render_case_card(case) for case in cases)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MatPlotBench Requirement Extraction</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffdf8;
      --ink: #1f2a32;
      --muted: #6b7681;
      --line: #ddd5c7;
      --accent: #0d6d77;
      --accent-soft: #d9eef0;
      --warn: #9d5b18;
      --warn-soft: #f6e4cc;
      --shadow: 0 8px 24px rgba(31, 42, 50, 0.08);
      --radius: 16px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #faf7f1, var(--bg));
      color: var(--ink);
    }}
    .shell {{
      width: min(1440px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}
    .hero, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    .sub {{
      color: var(--muted);
      line-height: 1.5;
      max-width: 960px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      background: #fcfaf5;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .metric .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .bands {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .band {{
      background: #fcfaf5;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
    }}
    .band h2 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      border-radius: 999px;
      padding: 6px 10px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
    }}
    .cases {{
      display: grid;
      gap: 14px;
      margin-top: 22px;
    }}
    .card {{
      padding: 18px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      flex-wrap: wrap;
    }}
    .case-id {{
      font-size: 22px;
      font-weight: 700;
    }}
    .score {{
      color: var(--warn);
      background: var(--warn-soft);
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.5fr 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }}
    .block {{
      background: #fcfaf5;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }}
    .block h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .mono {{
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 12px;
      white-space: pre-wrap;
      line-height: 1.5;
    }}
    .req-list {{
      display: grid;
      gap: 8px;
      max-height: 340px;
      overflow: auto;
    }}
    .req {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: white;
    }}
    .req-top {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .req-name {{
      font-weight: 700;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>MatPlotBench Requirement Extraction</h1>
      <div class="sub">Full-run report over the 43 DeepSeek-failed MatPlotBench cases. This is a parser-only evaluation slice: it checks what the LLM-first requirement extractor produces on native instruction-to-figure prompts, not whether downstream verification already covers all requirement families.</div>
      <div class="metrics">
        <div class="metric"><div class="label">Total Cases</div><div class="value">{summary["total_cases"]}</div></div>
        <div class="metric"><div class="label">Total Requirements</div><div class="value">{summary["total_requirements"]}</div></div>
        <div class="metric"><div class="label">Avg Requirements</div><div class="value">{summary["avg_requirements_per_case"]}</div></div>
        <div class="metric"><div class="label">Unknown Chart Type</div><div class="value">{summary["cases_with_unknown_chart_type"]}</div></div>
        <div class="metric"><div class="label">Explicit No Span</div><div class="value">{summary["explicit_without_span"]}</div></div>
        <div class="metric"><div class="label">Generated At</div><div class="value" style="font-size:14px">{escape_html(summary["generated_at"])}</div></div>
      </div>
      <div class="bands">
        <div class="band"><h2>Status Counts</h2><div class="chips">{render_chips(summary["status_counts"])}</div></div>
        <div class="band"><h2>Scope Counts</h2><div class="chips">{render_chips(summary["scope_counts"])}</div></div>
        <div class="band"><h2>Chart Types</h2><div class="chips">{render_chips(summary["chart_type_counts"])}</div></div>
        <div class="band"><h2>Score Buckets</h2><div class="chips">{render_chips(summary["score_buckets"])}</div></div>
        <div class="band"><h2>Top Requirement Names</h2><div class="chips">{render_chips(summary["top_requirement_names"], limit=18)}</div></div>
      </div>
    </section>
    <section class="cases">
      {cards}
    </section>
  </div>
</body>
</html>"""


def render_chips(values: dict[str, Any], limit: int | None = None) -> str:
    items = list(values.items())
    if limit is not None:
        items = items[:limit]
    return "".join(f'<span class="chip">{escape_html(str(key))}: {escape_html(str(value))}</span>' for key, value in items)


def render_case_card(case: dict[str, Any]) -> str:
    top_requirements = "\n".join(render_requirement(requirement) for requirement in case["requirements"][:12])
    return f"""
    <article class="card">
      <div class="card-head">
        <div class="case-id">Case {escape_html(str(case["native_id"]))}</div>
        <div class="score">GPT4V Score {escape_html(str(case["score"]))}</div>
      </div>
      <div class="grid">
        <div class="block">
          <h3>Instruction</h3>
          <div class="mono">{escape_html(str(case.get("query") or case.get("simple_instruction") or case.get("expert_instruction") or ""))}</div>
        </div>
        <div class="block">
          <h3>Plan</h3>
          <div class="mono">{escape_html(json.dumps(case["plan"], ensure_ascii=False, indent=2))}</div>
          <h3 style="margin-top:12px;">Case Summary</h3>
          <div class="mono">{escape_html(json.dumps(case["summary"], ensure_ascii=False, indent=2))}</div>
        </div>
        <div class="block">
          <h3>Top Requirements</h3>
          <div class="req-list">{top_requirements}</div>
        </div>
      </div>
    </article>"""


def render_requirement(requirement: dict[str, Any]) -> str:
    return f"""
    <div class="req">
      <div class="req-top">
        <span class="req-name">{escape_html(requirement["name"])}</span>
        <span class="muted">{escape_html(requirement["scope"])} / {escape_html(requirement["status"])}</span>
      </div>
      <div class="mono">{escape_html(json.dumps(requirement["value"], ensure_ascii=False))}</div>
      <div class="muted" style="margin-top:6px;">span: {escape_html(requirement["source_span"] or "∅")}</div>
    </div>"""


def escape_html(value: str) -> str:
    return html.escape(value, quote=True)


if __name__ == "__main__":
    main()
