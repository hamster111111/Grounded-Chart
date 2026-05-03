from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from grounded_chart.api import (  # noqa: E402
    ChartImageRenderer,
    MatplotlibTraceRunner,
    OpenAICompatibleLLMClient,
    load_ablation_run_config,
)
from grounded_chart_adapters.matplotbench import MatplotBenchEvalWorkspaceExporter  # noqa: E402


DEFAULT_CASE_IDS = (
    "matplotbench-ds-failed-35",
    "matplotbench-ds-failed-98",
    "matplotbench-ds-failed-55",
    "matplotbench-ds-failed-59",
    "matplotbench-ds-failed-48",
)

EXPECTED_SYSTEM_PROMPT = """You are an expected-requirement extraction agent for chart-generation benchmarks.
You do not share hidden memory with other agents. Use only the user-provided instruction.
Extract an intentionally broad but source-grounded checklist of what the chart should contain.
Do not discard requirements just because they may be hard to verify with scripts.
Return only a JSON object."""

ACTUAL_SYSTEM_PROMPT = """You are an actual-chart interpretation agent.
You do not share hidden memory with other agents. Use only the given code, runtime observation, and logs.
Infer what the current generated chart likely contains. Be explicit about uncertainty.
Return only a JSON object."""

DIAGNOSIS_SYSTEM_PROMPT = """You are a chart mismatch critic.
You do not share hidden memory with other agents. Compare only the explicit expected and actual artifacts provided.
Identify mismatches that matter for a benchmark evaluator comparing against a reference image.
Return only a JSON object."""

REPAIR_SYSTEM_PROMPT = """You are a chart-code repair agent for a diagnostic upper-bound experiment.
You do not share hidden memory with other agents. You may make broad edits or full-code rewrites if needed.
Your repaired code must be self-contained relative to the original working directory, preserve the user's intended chart, and save a visible figure.
Prefer saving to OUTPUT_PATH when it exists, otherwise save to novice_final.png. Return only a JSON object."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an LLM-heavy MatPlotBench upper-bound repair experiment with independent agent calls."
    )
    parser.add_argument("--config", default="configs/llm_ablation.deepseek.yaml")
    parser.add_argument("--bench", default="benchmarks/matplotbench_ds_failed_native.json")
    parser.add_argument("--output-dir", default="outputs/llm_agent_upper_bound")
    parser.add_argument("--case-ids", default=None, help="Comma-separated case IDs. Defaults to a 5-case diagnostic set.")
    parser.add_argument("--limit", type=int, default=5, help="Used only when --case-ids is omitted.")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-actual-agent", action="store_true", help="Use script observation directly instead of an LLM actual agent.")
    parser.add_argument("--skip-diagnosis-agent", action="store_true", help="Send expected/actual directly to repair agent.")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-evaluator", action="store_true", help="Call MatPlotAgent/evaluation/eval_qwen.py after rendering.")
    parser.add_argument("--matplot-agent-root", default=r"D:\Code\autoReaserch\MatPlotAgent")
    parser.add_argument("--eval-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_ablation_run_config(_resolve_path(args.config))
    provider = config.repair_provider or config.parser_provider
    if provider is None:
        raise ValueError("No LLM provider found in config. Fill llm.default, llm.parser, or llm.repair first.")
    if not str(provider.api_key or "").strip():
        raise ValueError("LLM provider API key is empty.")

    client = OpenAICompatibleLLMClient(provider)
    bench_path = _resolve_path(args.bench)
    raw_cases = _load_raw_cases(bench_path)
    selected = _select_cases(raw_cases, parse_case_ids(args.case_ids), args.limit)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    for index, raw_case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {raw_case['case_id']} expected-agent")
        try:
            case_summary = run_case(
                raw_case,
                client=client,
                output_dir=output_dir,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                render=bool(args.render),
                skip_actual_agent=bool(args.skip_actual_agent),
                skip_diagnosis_agent=bool(args.skip_diagnosis_agent),
            )
        except Exception as exc:
            case_summary = error_case_summary(raw_case, output_dir=output_dir, error=exc)
        case_summaries.append(case_summary)

    summary = build_summary(
        bench_path=bench_path,
        output_dir=output_dir,
        config_path=_resolve_path(args.config),
        cases=case_summaries,
        run_evaluator=bool(args.run_evaluator),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    export_manifest = None
    evaluator_summary = None
    if args.run_evaluator:
        print("Exporting rendered images for MatPlotBench evaluator")
        export_workspace = output_dir / "eval_workspace"
        export_manifest = MatplotBenchEvalWorkspaceExporter(summary_path, export_workspace).export()
        (output_dir / "eval_export_manifest.json").write_text(
            json.dumps(export_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        evaluator_summary = run_matplotbench_evaluator(
            matplot_agent_root=Path(args.matplot_agent_root),
            workspace=export_workspace,
            native_ids=tuple(export_manifest.get("native_ids", ())),
            workers=int(args.eval_workers),
        )
        (output_dir / "evaluator_summary.json").write_text(
            json.dumps(evaluator_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["evaluator_summary"] = evaluator_summary
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Output:", output_dir)
    print("Summary:", summary_path)
    print(
        "Render ok:",
        f"{sum(1 for case in case_summaries if case.get('render_ok'))}/{len(case_summaries)}",
        "Evaluator:",
        "run" if evaluator_summary is not None else "not_run",
    )


def run_case(
    raw_case: dict[str, Any],
    *,
    client: OpenAICompatibleLLMClient,
    output_dir: Path,
    temperature: float,
    max_tokens: int,
    render: bool,
    skip_actual_agent: bool,
    skip_diagnosis_agent: bool,
) -> dict[str, Any]:
    case_id = str(raw_case["case_id"])
    case_dir = output_dir / "cases" / _safe_name(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    code_path = Path(str(raw_case["selected_code_path"]))
    code = code_path.read_text(encoding="utf-8", errors="replace")
    instruction = _instruction_text(raw_case)
    source_workspace = Path(str(raw_case.get("workspace_dir") or code_path.parent))
    experiment_workspace = case_dir / "workspace"
    _copy_workspace(source_workspace, experiment_workspace)
    original_code_path = case_dir / "original_code.py"
    original_code_path.write_text(code, encoding="utf-8")

    input_payload = {
        "case_id": case_id,
        "native_id": raw_case.get("native_id"),
        "original_score": raw_case.get("score"),
        "instruction": instruction,
        "selected_code_path": str(code_path),
        "workspace_dir": str(source_workspace),
        "ground_truth_path": raw_case.get("ground_truth_path"),
        "output_image_paths": list(raw_case.get("output_image_paths", [])),
    }
    _write_json(case_dir / "input.json", input_payload)

    expected = call_agent_json(
        client,
        agent_name="expected_requirements",
        system_prompt=EXPECTED_SYSTEM_PROMPT,
        user_payload={
            "task": "Extract broad expected chart requirements from the MatPlotBench instruction.",
            "case_id": case_id,
            "instruction": instruction,
            "output_schema": expected_schema(),
        },
        output_path=case_dir / "expected_requirements.json",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    script_observation = build_script_observation(code, code_path=code_path, execution_dir=source_workspace)
    _write_json(case_dir / "script_observation.json", script_observation)

    if skip_actual_agent:
        actual = {
            "agent": "script_observation_only",
            "actual_artifacts": script_observation,
            "uncertainties": ["LLM actual agent skipped by CLI flag."],
        }
        _write_json(case_dir / "actual_description.json", actual)
    else:
        print(f"  {case_id} actual-agent")
        actual = call_agent_json(
            client,
            agent_name="actual_description",
            system_prompt=ACTUAL_SYSTEM_PROMPT,
            user_payload={
                "task": "Describe the actual generated chart from code and runtime observation. Do not compare against the expected requirements.",
                "case_id": case_id,
                "code": _truncate_middle(code, 26000),
                "script_observation": script_observation,
                "original_output_image_paths": list(raw_case.get("output_image_paths", [])),
                "output_schema": actual_schema(),
            },
            output_path=case_dir / "actual_description.json",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if skip_diagnosis_agent:
        diagnosis = {
            "agent": "diagnosis_skipped",
            "mismatches": [],
            "repair_priorities": [],
            "notes": "Repair agent receives expected and actual artifacts directly.",
        }
        _write_json(case_dir / "diagnosis.json", diagnosis)
    else:
        print(f"  {case_id} diagnosis-agent")
        diagnosis = call_agent_json(
            client,
            agent_name="diagnosis",
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            user_payload={
                "task": "Compare expected requirements against actual chart description and identify repair targets.",
                "case_id": case_id,
                "expected_requirements": expected.get("payload"),
                "actual_description": actual.get("payload") if isinstance(actual, dict) and "payload" in actual else actual,
                "output_schema": diagnosis_schema(),
            },
            output_path=case_dir / "diagnosis.json",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    print(f"  {case_id} repair-agent")
    repair = call_agent_json(
        client,
        agent_name="repair",
        system_prompt=REPAIR_SYSTEM_PROMPT,
        user_payload={
            "task": "Repair the chart code. You may return full repaired code. Optimize for the original benchmark evaluator, not for a script verifier.",
            "case_id": case_id,
            "instruction": instruction,
            "original_code": _truncate_middle(code, 36000),
            "expected_requirements": expected.get("payload"),
            "actual_description": actual.get("payload") if isinstance(actual, dict) and "payload" in actual else actual,
            "diagnosis": diagnosis.get("payload") if isinstance(diagnosis, dict) and "payload" in diagnosis else diagnosis,
            "important_output_rule": "The final code should save a visible figure to OUTPUT_PATH if that variable exists; otherwise save to novice_final.png.",
            "output_schema": repair_schema(),
        },
        output_path=case_dir / "repair.json",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    repaired_code = _extract_repaired_code(repair.get("payload")) or code
    repaired_code_path = case_dir / "repaired_code.py"
    repaired_code_path.write_text(repaired_code, encoding="utf-8")
    workspace_code_path = experiment_workspace / "code_action_llm_agent_upper_bound.py"
    workspace_code_path.write_text(repaired_code, encoding="utf-8")

    render_result = None
    if render:
        print(f"  {case_id} render")
        renderer = ChartImageRenderer()
        render_result = renderer.render(
            repaired_code,
            rows=(),
            output_dir=experiment_workspace,
            output_filename="novice_final.png",
            file_path=workspace_code_path,
        )
        _write_json(case_dir / "render_result.json", render_result_to_dict(render_result))

    return {
        "case_id": case_id,
        "native_id": raw_case.get("native_id"),
        "query": raw_case.get("query"),
        "original_score": raw_case.get("score"),
        "ground_truth_path": raw_case.get("ground_truth_path"),
        "original_code_path": str(original_code_path),
        "final_code_path": str(repaired_code_path),
        "workspace_code_path": str(workspace_code_path),
        "image_path": str(render_result.image_path) if render_result and render_result.image_path else None,
        "render_ok": bool(render_result.ok) if render_result is not None else False,
        "render_exception_type": render_result.exception_type if render_result is not None else None,
        "render_exception_message": render_result.exception_message if render_result is not None else None,
        "agent_artifacts": {
            "input": str(case_dir / "input.json"),
            "script_observation": str(case_dir / "script_observation.json"),
            "expected_requirements": str(case_dir / "expected_requirements.json"),
            "actual_description": str(case_dir / "actual_description.json"),
            "diagnosis": str(case_dir / "diagnosis.json"),
            "repair": str(case_dir / "repair.json"),
            "render_result": str(case_dir / "render_result.json") if render else None,
        },
        "metadata": {
            "native_id": raw_case.get("native_id"),
            "source_code": str(code_path),
            "source_workspace": str(source_workspace),
            "experiment_workspace": str(experiment_workspace),
            "original_eval_raw": raw_case.get("eval_raw"),
            "original_eval_error": raw_case.get("eval_error"),
        },
    }


def error_case_summary(raw_case: dict[str, Any], *, output_dir: Path, error: BaseException) -> dict[str, Any]:
    case_id = str(raw_case.get("case_id") or "unknown_case")
    case_dir = output_dir / "cases" / _safe_name(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "native_id": raw_case.get("native_id"),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    _write_json(case_dir / "error.json", payload)
    return {
        "case_id": case_id,
        "native_id": raw_case.get("native_id"),
        "query": raw_case.get("query"),
        "original_score": raw_case.get("score"),
        "ground_truth_path": raw_case.get("ground_truth_path"),
        "final_code_path": None,
        "image_path": None,
        "render_ok": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "agent_artifacts": {"error": str(case_dir / "error.json")},
        "metadata": {"native_id": raw_case.get("native_id")},
    }


def call_agent_json(
    client: OpenAICompatibleLLMClient,
    *,
    agent_name: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    output_path: Path,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    result = client.complete_json_with_trace(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    wrapped = {
        "agent": agent_name,
        "payload": result.payload,
        "trace": llm_trace_to_dict(result.trace),
    }
    _write_json(output_path, wrapped)
    return wrapped


def build_script_observation(code: str, *, code_path: Path, execution_dir: Path) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "execution_dir": str(execution_dir),
        "code_path": str(code_path),
        "code_structure": [],
        "trace_error": None,
        "plot_trace": None,
        "figure_trace": None,
    }
    try:
        run_trace = MatplotlibTraceRunner().run_code_with_figure(
            code,
            globals_dict={"rows": []},
            execution_dir=execution_dir,
            file_path=code_path,
        )
        observation["plot_trace"] = jsonable(run_trace.plot_trace)
        observation["figure_trace"] = jsonable(run_trace.figure_trace)
    except Exception as exc:
        observation["trace_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return observation


def run_matplotbench_evaluator(
    *,
    matplot_agent_root: Path,
    workspace: Path,
    native_ids: tuple[int, ...],
    workers: int,
) -> dict[str, Any]:
    evaluator = matplot_agent_root / "evaluation" / "eval_qwen.py"
    if not evaluator.exists():
        raise FileNotFoundError(f"MatPlotBench evaluator not found: {evaluator}")
    details: dict[str, Any] = {}
    commands: list[dict[str, Any]] = []
    for native_id in native_ids:
        command = [
            sys.executable,
            str(evaluator),
            "--workspace",
            str(workspace),
            "--start_id",
            str(native_id),
            "--end_id",
            str(native_id),
            "--workers",
            str(max(1, workers)),
        ]
        completed = subprocess.run(
            command,
            cwd=str(matplot_agent_root),
            text=True,
            capture_output=True,
            timeout=600,
        )
        commands.append(
            {
                "native_id": native_id,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
        eval_results_path = workspace / "eval_results.json"
        if completed.returncode == 0 and eval_results_path.exists():
            payload = json.loads(eval_results_path.read_text(encoding="utf-8"))
            result = (payload.get("details") or {}).get(str(native_id))
            if result is not None:
                details[str(native_id)] = result
        else:
            details[str(native_id)] = {
                "score": None,
                "raw": None,
                "error": f"evaluator_returncode_{completed.returncode}",
            }
    scores = [item.get("score") for item in details.values() if isinstance(item.get("score"), (int, float))]
    return {
        "workspace": str(workspace),
        "native_ids": list(native_ids),
        "details": details,
        "commands": commands,
        "summary": {
            "evaluated": len(scores),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
            "pass_rate_50": round(sum(1 for score in scores if score >= 50) / len(scores) * 100, 2) if scores else None,
            "high_quality_70": round(sum(1 for score in scores if score >= 70) / len(scores) * 100, 2) if scores else None,
        },
    }


def build_summary(
    *,
    bench_path: Path,
    output_dir: Path,
    config_path: Path,
    cases: list[dict[str, Any]],
    run_evaluator: bool,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "llm_agent_upper_bound",
        "bench_path": str(bench_path),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "run_evaluator": run_evaluator,
        "total_cases": len(cases),
        "render_ok_cases": sum(1 for case in cases if case.get("render_ok")),
        "original_scores": {
            str(case.get("native_id")): case.get("original_score") for case in cases if case.get("native_id") is not None
        },
        "cases": cases,
    }


def _load_raw_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Bench must be a JSON list: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _select_cases(raw_cases: list[dict[str, Any]], explicit_case_ids: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    by_id = {str(case.get("case_id")): case for case in raw_cases}
    if explicit_case_ids:
        missing = [case_id for case_id in explicit_case_ids if case_id not in by_id]
        if missing:
            raise KeyError(f"Cases not found in bench: {missing}")
        return [by_id[case_id] for case_id in explicit_case_ids]
    selected = [by_id[case_id] for case_id in DEFAULT_CASE_IDS if case_id in by_id]
    if len(selected) >= limit:
        return selected[:limit]
    seen = {str(case.get("case_id")) for case in selected}
    for case in raw_cases:
        if str(case.get("case_id")) not in seen:
            selected.append(case)
            seen.add(str(case.get("case_id")))
        if len(selected) >= limit:
            break
    return selected


def _instruction_text(raw_case: dict[str, Any]) -> str:
    for key in ("query", "simple_instruction", "instruction", "expert_instruction"):
        value = raw_case.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _copy_workspace(source: Path, target: Path) -> None:
    if target.exists():
        return
    if source.exists() and source.is_dir():
        ignore = shutil.ignore_patterns("*.log", "__pycache__", ".ipynb_checkpoints")
        shutil.copytree(source, target, ignore=ignore)
    else:
        target.mkdir(parents=True, exist_ok=True)


def _extract_repaired_code(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("repaired_code", "code", "final_code"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_code_fence(value.strip())
    return None


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return text


def parse_case_ids(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def expected_schema() -> dict[str, Any]:
    return {
        "requirements": [
            {
                "id": "short stable id",
                "claim": "source-grounded expected chart requirement",
                "type": "data|layout|encoding|style|annotation|text|legend|axis|3d|other",
                "source_span": "verbatim supporting instruction span",
                "priority": "critical|major|minor",
                "verifiability": "script_supported|vision_needed|human_or_bench_only|uncertain",
                "confidence": 0.0,
            }
        ],
        "global_notes": [],
        "uncertainties": [],
    }


def actual_schema() -> dict[str, Any]:
    return {
        "actual_artifacts": [
            {
                "id": "short stable id",
                "description": "what the current code/trace appears to produce",
                "type": "data|layout|encoding|style|annotation|text|legend|axis|3d|other",
                "evidence": "code line, trace field, or log signal",
                "confidence": 0.0,
            }
        ],
        "likely_rendering_issues": [],
        "uncertainties": [],
    }


def diagnosis_schema() -> dict[str, Any]:
    return {
        "mismatches": [
            {
                "expected_id": "requirement id if known",
                "actual_id": "actual artifact id if known",
                "problem": "concise mismatch statement",
                "severity": "critical|major|minor",
                "repair_hint": "actionable code-level hint",
                "confidence": 0.0,
            }
        ],
        "repair_priorities": [],
        "risk_notes": [],
    }


def repair_schema() -> dict[str, Any]:
    return {
        "summary": "what changed and why",
        "repaired_code": "full Python code string",
        "expected_improvements": [],
        "risk_notes": [],
    }


def render_result_to_dict(render: Any) -> dict[str, Any]:
    return {
        "ok": render.ok,
        "image_path": str(render.image_path) if render.image_path else None,
        "artifact_paths": [str(path) for path in render.artifact_paths],
        "backend": render.backend,
        "stdout": render.stdout,
        "stderr": render.stderr,
        "exception_type": render.exception_type,
        "exception_message": render.exception_message,
        "metadata": jsonable(render.metadata),
    }


def llm_trace_to_dict(trace: Any) -> dict[str, Any] | None:
    if trace is None:
        return None
    usage = getattr(trace, "usage", None)
    return {
        "provider": getattr(trace, "provider", None),
        "model": getattr(trace, "model", None),
        "base_url": getattr(trace, "base_url", None),
        "temperature": getattr(trace, "temperature", None),
        "max_tokens": getattr(trace, "max_tokens", None),
        "usage": jsonable(usage),
        "raw_text": getattr(trace, "raw_text", ""),
        "parsed_json": jsonable(getattr(trace, "parsed_json", None)),
    }


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n# ... <truncated middle for prompt budget> ...\n\n" + text[-tail:]


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe.strip("_") or "case"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
