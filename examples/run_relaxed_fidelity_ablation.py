from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from grounded_chart import (  # noqa: E402
    ChartImageRenderer,
    MatplotlibTraceRunner,
    OpenAICompatibleLLMClient,
    SourceDataPlanner,
    apply_patch_operations,
    load_ablation_run_config,
    parse_patch_operations,
)
from grounded_chart.llm import extract_json_object  # noqa: E402
from grounded_chart_adapters.matplotbench import MatplotBenchEvalWorkspaceExporter  # noqa: E402


DEFAULT_CASE_IDS = (
    "matplotbench-ds-failed-35",
    "matplotbench-ds-failed-98",
    "matplotbench-ds-failed-55",
    "matplotbench-ds-failed-59",
    "matplotbench-ds-failed-48",
)

VALID_VARIANTS = frozenset({"vanilla_repair", "fidelity_repair"})
FINAL_SCORE_RE = re.compile(r"\[FINAL SCORE\]\s*:\s*(\d+(?:\.\d+)?)")
DATA_FILE_SUFFIXES = frozenset({".csv", ".tsv", ".json", ".xlsx", ".xls"})
SYNTHETIC_DATA_PATTERNS = (
    ("np.random", re.compile(r"\bnp\.random\b")),
    ("random_module", re.compile(r"\brandom\.")),
    ("dummy_data", re.compile(r"dummy\s+data", re.IGNORECASE)),
    ("sample_data", re.compile(r"sample\s+data", re.IGNORECASE)),
    ("synthetic_data", re.compile(r"synthetic\s+data", re.IGNORECASE)),
    ("files_not_provided", re.compile(r"(no|not|without)\s+(actual\s+)?(csv|data|files?)\s+(is|are\s+)?provided", re.IGNORECASE)),
    ("create_data", re.compile(r"\b(create|generate)\s+(sample|dummy|synthetic)?\s*data", re.IGNORECASE)),
)
DATA_LOAD_CALL_RE = re.compile(r"\b(pd\.)?(read_csv|read_table|read_excel|read_json)\s*\(|\bopen\s*\(")
INSTRUCTION_FILE_RE = re.compile(r"['\"]?([A-Za-z0-9_\- ./()]+?\.(?:csv|tsv|json|xlsx|xls))['\"]?", re.IGNORECASE)


def _score_number(value: str) -> int | float:
    score = float(value)
    return int(score) if score.is_integer() else score


def normalize_matplotbench_evaluator_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Trust only explicit MatPlotBench [FINAL SCORE] markers, not fallback numbers."""
    normalized = dict(detail)
    raw_score = normalized.get("score")
    normalized["raw_score_from_evaluator"] = raw_score

    raw = normalized.get("raw")
    if isinstance(raw, str):
        match = FINAL_SCORE_RE.search(raw)
        normalized["final_score_marker_present"] = bool(match)
        if match:
            marker_score = _score_number(match.group(1))
            normalized["score"] = marker_score
            normalized["score_parse_status"] = "final_score_marker"
            if raw_score != marker_score:
                normalized["score_parse_note"] = "overrode_evaluator_fallback_with_final_score_marker"
            return normalized
        if raw.strip() == "No image generated":
            normalized["score_parse_status"] = "no_image_score_0"
            normalized["score"] = 0
            return normalized
        normalized["score"] = None
        normalized["score_parse_status"] = "evaluator_parse_error_missing_final_score"
        normalized["error"] = normalized.get("error") or "evaluator_parse_error_missing_final_score"
        return normalized

    normalized["final_score_marker_present"] = False
    normalized["score"] = None
    normalized["score_parse_status"] = (
        "evaluator_error" if normalized.get("error") else "evaluator_parse_error_missing_raw"
    )
    normalized["error"] = normalized.get("error") or "evaluator_parse_error_missing_raw"
    return normalized


EXPECTED_SYSTEM_PROMPT = """You are a chart requirement extraction agent for a cost-controlled fidelity experiment.
Use only the benchmark instruction. Do not use hidden memory, reference images, or benchmark-specific prior knowledge.
Extract broad but source-grounded chart requirements, including data semantics, plot type, layout, axes, labels, legends, annotations, scales, and visual constraints when stated.
Do not invent exact data values unless they are explicitly stated. Mark uncertain or vision-needed requirements honestly.
Return only one valid JSON object. Do not use markdown fences.
Be compact: at most 16 requirements, each claim <=18 words, each source_span <=20 words, no prose outside JSON."""


DIAGNOSIS_SYSTEM_PROMPT = """You are a compact chart-fidelity mismatch critic.
Compare only the extracted expected requirements and the script-observed actual artifacts.
Do not infer from hidden memory. Do not ask for a full rewrite by default.
Return only one valid JSON object. Do not use markdown fences.
Be compact: at most 8 mismatches, each problem/evidence/repair_hint <=18 words."""


VANILLA_REPAIR_SYSTEM_PROMPT = """You are a cost-controlled chart-code repair agent.
You receive the benchmark instruction, compact code context, and script/runtime observation.
Prefer localized patch_ops over full code rewrites. Use full repaired_code only when a patch cannot safely express the fix.
Preserve existing working code and avoid changing unrelated chart structure.
Return only one valid JSON object. Do not use markdown fences. Do not repeat analysis. Keep summary under 60 words."""


FIDELITY_REPAIR_SYSTEM_PROMPT = """You are a cost-controlled evidence-guided chart-code repair agent.
You receive source-grounded expected requirements, available source-file evidence, exact source-schema constraints, compact script-observed actual artifacts, an evidence repair plan, and optional mismatch notes.
Repair the most important chart-fidelity mismatches while preserving requirements that are already satisfied.
Prefer localized patch_ops for local failures. Use full repaired_code when the repair_plan requires data_patch or structural_regeneration and a patch would be fragile.
If available files are listed, do not invent dummy/sample/random data for those sources. Read the provided files by filename.
Respect exact schemas: do not reference absent columns unless your code first creates them through an explicit deterministic transformation.
If evidence is insufficient, abstain or make only a conservative runtime/output fix.
Return only one valid JSON object. Do not use markdown fences. Do not repeat analysis. Keep summary under 60 words."""


RUNTIME_RETRY_SYSTEM_PROMPT = """You are a bounded chart-code runtime repair agent.
Fix only the reported runtime/rendering error while preserving the benchmark instruction and existing chart structure.
If source-schema constraints are provided, respect exact columns and use deterministic transformations when reshaping data.
Prefer patch_ops for local runtime fixes. Use full repaired_code only when the failed code cannot be patched safely.
Return only one valid JSON object. Do not use markdown fences. Keep summary under 60 words."""


@dataclass(frozen=True)
class RepairApplication:
    code: str
    source: str
    patch_applied: bool = False
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a relaxed but token-controlled MatPlotBench repair ablation: "
            "vanilla LLM repair vs fidelity-evidence LLM repair."
        )
    )
    parser.add_argument("--config", default="configs/llm_ablation.deepseek.yaml")
    parser.add_argument("--bench", default="benchmarks/matplotbench_ds_failed_native.json")
    parser.add_argument("--output-dir", default="outputs/relaxed_fidelity_ablation")
    parser.add_argument(
        "--variants",
        default="vanilla_repair,fidelity_repair",
        help=f"Comma-separated variants from: {', '.join(sorted(VALID_VARIANTS))}.",
    )
    parser.add_argument("--case-ids", default=None, help="Comma-separated case IDs. Defaults to a diagnostic set.")
    parser.add_argument("--limit", type=int, default=5, help="Used only when --case-ids is omitted.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--expected-max-tokens", type=int, default=1536)
    parser.add_argument("--diagnosis-max-tokens", type=int, default=1024)
    parser.add_argument("--repair-max-tokens", type=int, default=4096)
    parser.add_argument("--runtime-retry-max-tokens", type=int, default=3072)
    parser.add_argument("--max-code-chars", type=int, default=14000)
    parser.add_argument("--max-observation-chars", type=int, default=7000)
    parser.add_argument("--max-patch-ops", type=int, default=4)
    parser.add_argument("--max-changed-lines", type=int, default=80)
    parser.add_argument("--diagnosis-agent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-full-rewrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-retry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-evaluator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the official MatPlotBench visual evaluator as the experiment scoring standard.",
    )
    parser.add_argument("--matplot-agent-root", default=r"D:\Code\autoReaserch\MatPlotAgent")
    parser.add_argument("--eval-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    config = load_ablation_run_config(_resolve_path(args.config))
    provider = config.repair_provider or config.parser_provider
    if provider is None:
        raise ValueError("No LLM provider found in config.")
    if not str(provider.api_key or "").strip():
        raise ValueError("LLM provider API key is empty.")

    client = OpenAICompatibleLLMClient(provider)
    bench_path = _resolve_path(args.bench)
    raw_cases = _load_raw_cases(bench_path)
    selected_cases = _select_cases(raw_cases, parse_case_ids(args.case_ids), int(args.limit))
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_cases: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    common_cases: list[dict[str, Any]] = []

    for index, raw_case in enumerate(selected_cases, start=1):
        print(f"[{index}/{len(selected_cases)}] {raw_case.get('case_id')}")
        try:
            common = prepare_common_case(
                raw_case,
                client=client,
                output_dir=output_dir,
                need_expected="fidelity_repair" in variants,
                diagnosis_agent=bool(args.diagnosis_agent),
                temperature=float(args.temperature),
                expected_max_tokens=int(args.expected_max_tokens),
                diagnosis_max_tokens=int(args.diagnosis_max_tokens),
                max_code_chars=int(args.max_code_chars),
                max_observation_chars=int(args.max_observation_chars),
            )
        except Exception as exc:
            common = common_error_case(raw_case, output_dir=output_dir, error=exc)
        common_cases.append(common)

        for variant in variants:
            try:
                print(f"  {variant}")
                case_summary = run_variant_case(
                    common,
                    variant=variant,
                    client=client,
                    output_dir=output_dir,
                    temperature=float(args.temperature),
                    repair_max_tokens=int(args.repair_max_tokens),
                    max_patch_ops=int(args.max_patch_ops),
                    max_changed_lines=int(args.max_changed_lines),
                    allow_full_rewrite=bool(args.allow_full_rewrite),
                    runtime_retry=bool(args.runtime_retry),
                    runtime_retry_max_tokens=int(args.runtime_retry_max_tokens),
                    render=bool(args.render),
                )
            except Exception as exc:
                case_summary = variant_error_case(common, variant=variant, output_dir=output_dir, error=exc)
            variant_cases[variant].append(case_summary)

    variant_summaries = {}
    for variant, cases in variant_cases.items():
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        summary = build_variant_summary(
            variant=variant,
            bench_path=bench_path,
            config_path=_resolve_path(args.config),
            output_dir=variant_dir,
            cases=cases,
        )
        summary_path = variant_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.run_evaluator:
            evaluator_summary = run_variant_evaluator(
                variant_summary_path=summary_path,
                variant_dir=variant_dir,
                matplot_agent_root=Path(args.matplot_agent_root),
                workers=int(args.eval_workers),
            )
            summary["evaluator_summary"] = evaluator_summary
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        variant_summaries[variant] = summary

    overall_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "relaxed_fidelity_ablation",
        "bench_path": str(bench_path),
        "output_dir": str(output_dir),
        "evaluation_standard": {
            "primary_metric": "official_evaluator_score_delta",
            "official_evaluator": "MatPlotBench eval_qwen.py",
            "render_failures_score": 0,
            "note": "Internal render status is only a runtime diagnostic; figure quality is judged only by official evaluator score.",
        },
        "variants": list(variants),
        "total_cases": len(common_cases),
        "case_ids": [case.get("case_id") for case in common_cases],
        "common_cases": common_cases,
        "variant_summaries": {
            variant: {
                "summary_path": str(output_dir / variant / "summary.json"),
                "runtime_render_ok_cases_diagnostic": summary["runtime_render_ok_cases_diagnostic"],
                "llm_usage": summary["llm_usage"],
                "official_score_summary": (summary.get("evaluator_summary") or {}).get("summary"),
                "evaluator_summary": summary.get("evaluator_summary"),
            }
            for variant, summary in variant_summaries.items()
        },
        "comparisons": build_relaxed_comparisons(variant_summaries),
    }
    overall_path = output_dir / "summary.json"
    overall_path.write_text(json.dumps(overall_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Output:", output_dir)
    print("Summary:", overall_path)
    for variant, summary in variant_summaries.items():
        print(
            variant,
            "runtime_render_ok(diagnostic)=",
            f"{summary['runtime_render_ok_cases_diagnostic']}/{summary['total_cases']}",
            "llm_calls=",
            summary["llm_usage"]["call_count"],
            "tokens=",
            summary["llm_usage"]["total_tokens"],
            )
        evaluator = summary.get("evaluator_summary")
        if evaluator:
            official = evaluator["summary"]
            print(
                " ",
                "original_avg=",
                official["original_avg_score"],
                "official_final_avg=",
                official["official_final_avg_score"],
                "official_avg_delta=",
                official["official_avg_delta"],
            )
        else:
            print(" ", "official_evaluator=not_run", "score_metrics=unavailable")


def prepare_common_case(
    raw_case: dict[str, Any],
    *,
    client: OpenAICompatibleLLMClient,
    output_dir: Path,
    need_expected: bool,
    diagnosis_agent: bool,
    temperature: float,
    expected_max_tokens: int,
    diagnosis_max_tokens: int,
    max_code_chars: int,
    max_observation_chars: int,
) -> dict[str, Any]:
    case_id = str(raw_case["case_id"])
    common_dir = output_dir / "cases" / _safe_name(case_id) / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    code_path = Path(str(raw_case["selected_code_path"]))
    code = code_path.read_text(encoding="utf-8", errors="replace")
    instruction = _instruction_text(raw_case)
    source_workspace = Path(str(raw_case.get("workspace_dir") or code_path.parent))
    code_context = build_code_context(code, max_chars=max_code_chars)
    source_file_evidence = build_source_file_evidence(
        source_workspace=source_workspace,
        instruction=instruction,
        max_preview_rows=5,
        max_files=12,
    )
    source_schema_constraints = build_source_schema_constraints(source_file_evidence)
    source_data_violation = detect_source_data_violation(
        code,
        source_file_evidence=source_file_evidence,
        instruction=instruction,
    )
    repair_plan = plan_initial_repair(
        source_file_evidence=source_file_evidence,
        source_data_violation=source_data_violation,
    )
    observation = build_compact_script_observation(
        code,
        code_path=code_path,
        execution_dir=source_workspace,
        max_chars=max_observation_chars,
    )

    input_payload = {
        "case_id": case_id,
        "native_id": _native_id(raw_case),
        "original_score": raw_case.get("score"),
        "instruction": instruction,
        "query": raw_case.get("query"),
        "simple_instruction": raw_case.get("simple_instruction"),
        "expert_instruction": raw_case.get("expert_instruction"),
        "selected_code_path": str(code_path),
        "workspace_dir": str(source_workspace),
        "ground_truth_path": raw_case.get("ground_truth_path"),
        "output_image_paths": list(raw_case.get("output_image_paths") or []),
    }
    _write_json(common_dir / "input.json", input_payload)
    _write_json(common_dir / "code_context.json", code_context)
    _write_json(common_dir / "source_file_evidence.json", source_file_evidence)
    _write_json(common_dir / "source_schema_constraints.json", source_schema_constraints)
    _write_json(common_dir / "source_data_violation.json", source_data_violation)
    _write_json(common_dir / "repair_plan.json", repair_plan)
    _write_json(common_dir / "script_observation.json", observation)

    expected = None
    diagnosis = None
    if need_expected:
        expected = call_agent_json(
            client,
            agent_name="expected_requirements",
            system_prompt=EXPECTED_SYSTEM_PROMPT,
            user_payload={
                "task": "Extract source-grounded expected chart requirements from the instruction.",
                "case_id": case_id,
                "instruction": instruction,
                "max_requirements": 16,
                "compactness_rules": [
                    "claim <=18 words",
                    "source_span <=20 words",
                    "global_notes and uncertainties should be empty unless essential",
                ],
                "output_schema": expected_schema(),
            },
            output_path=common_dir / "expected_requirements.json",
            temperature=temperature,
            max_tokens=expected_max_tokens,
        )
        if diagnosis_agent:
            diagnosis = call_agent_json(
                client,
                agent_name="fidelity_diagnosis",
                system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
                user_payload={
                "task": "Find likely fidelity mismatches using only expected requirements and script observations.",
                "case_id": case_id,
                "expected_requirements": expected.get("payload"),
                "script_observation": observation,
                "compactness_rules": [
                    "at most 8 mismatches",
                    "problem/evidence/repair_hint <=18 words each",
                    "repair_priorities <=5 short items",
                    "risk_notes <=3 short items",
                ],
                "output_schema": diagnosis_schema(),
            },
                output_path=common_dir / "diagnosis.json",
                temperature=temperature,
                max_tokens=diagnosis_max_tokens,
            )
        else:
            diagnosis = {
                "agent": "diagnosis_skipped",
                "payload": {
                    "mismatches": [],
                    "repair_priorities": [],
                    "risk_notes": ["Diagnosis agent disabled by CLI flag."],
                },
                "trace": None,
            }
            _write_json(common_dir / "diagnosis.json", diagnosis)

    return {
        "case_id": case_id,
        "native_id": _native_id(raw_case),
        "original_score": raw_case.get("score"),
        "instruction": instruction,
        "selected_code_path": str(code_path),
        "source_workspace": str(source_workspace),
        "ground_truth_path": raw_case.get("ground_truth_path"),
        "output_image_paths": list(raw_case.get("output_image_paths") or []),
        "common_dir": str(common_dir),
        "input_path": str(common_dir / "input.json"),
        "code_context_path": str(common_dir / "code_context.json"),
        "script_observation_path": str(common_dir / "script_observation.json"),
        "expected_requirements_path": str(common_dir / "expected_requirements.json") if expected else None,
        "diagnosis_path": str(common_dir / "diagnosis.json") if diagnosis else None,
        "source_file_evidence_path": str(common_dir / "source_file_evidence.json"),
        "source_schema_constraints_path": str(common_dir / "source_schema_constraints.json"),
        "source_data_violation_path": str(common_dir / "source_data_violation.json"),
        "repair_plan_path": str(common_dir / "repair_plan.json"),
        "code_context": code_context,
        "source_file_evidence": source_file_evidence,
        "source_schema_constraints": source_schema_constraints,
        "source_data_violation": source_data_violation,
        "repair_plan": repair_plan,
        "script_observation": observation,
        "expected_requirements": expected,
        "diagnosis": diagnosis,
        "llm_usage": aggregate_llm_usage(
            [item.get("trace") for item in (expected, diagnosis) if isinstance(item, dict)]
        ),
    }


def run_variant_case(
    common: dict[str, Any],
    *,
    variant: str,
    client: OpenAICompatibleLLMClient,
    output_dir: Path,
    temperature: float,
    repair_max_tokens: int,
    max_patch_ops: int,
    max_changed_lines: int,
    allow_full_rewrite: bool,
    runtime_retry: bool,
    runtime_retry_max_tokens: int,
    render: bool,
) -> dict[str, Any]:
    case_id = str(common["case_id"])
    case_dir = output_dir / variant / "cases" / _safe_name(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    original_code_path = Path(str(common["selected_code_path"]))
    original_code = original_code_path.read_text(encoding="utf-8", errors="replace")
    source_workspace = Path(str(common["source_workspace"]))
    experiment_workspace = case_dir / "workspace"
    _copy_workspace(source_workspace, experiment_workspace)

    system_prompt = VANILLA_REPAIR_SYSTEM_PROMPT if variant == "vanilla_repair" else FIDELITY_REPAIR_SYSTEM_PROMPT
    user_payload = build_repair_user_payload(common, variant=variant, allow_full_rewrite=allow_full_rewrite)
    repair = call_agent_json(
        client,
        agent_name=variant,
        system_prompt=system_prompt,
        user_payload=user_payload,
        output_path=case_dir / "repair.json",
        temperature=temperature,
        max_tokens=repair_max_tokens,
    )
    application = apply_repair_payload(
        original_code,
        repair.get("payload"),
        max_patch_ops=max_patch_ops,
        max_changed_lines=max_changed_lines,
        allow_full_rewrite=allow_full_rewrite,
    )
    repair_artifacts = [case_dir / "repair.json"]
    final_code_path = case_dir / "final_code.py"
    final_code_path.write_text(application.code, encoding="utf-8")
    _write_json(case_dir / "repair_application.json", application.to_dict())

    workspace_code_path = experiment_workspace / f"code_action_relaxed_{variant}_initial_0.py"
    workspace_code_path.write_text(application.code, encoding="utf-8")
    render_result = None
    if render:
        renderer = ChartImageRenderer()
        render_result = renderer.render(
            application.code,
            rows=(),
            output_dir=experiment_workspace,
            output_filename="novice_final.png",
            file_path=workspace_code_path,
        )
        _write_json(case_dir / "render_result.json", render_result_to_dict(render_result))
        if (
            runtime_retry
            and variant == "fidelity_repair"
            and not render_result.ok
            and (render_result.exception_type or render_result.exception_message)
        ):
            retry = run_runtime_retry(
                common,
                client=client,
                case_dir=case_dir,
                failed_code=application.code,
                failed_render_result=render_result_to_dict(render_result),
                temperature=temperature,
                max_tokens=runtime_retry_max_tokens,
                max_patch_ops=max_patch_ops,
                max_changed_lines=max_changed_lines,
                allow_full_rewrite=allow_full_rewrite,
            )
            repair_artifacts.append(case_dir / "runtime_retry_repair.json")
            retry_application = retry["application"]
            retry_code_path = case_dir / "final_code_runtime_retry.py"
            retry_code_path.write_text(retry_application.code, encoding="utf-8")
            workspace_code_path = experiment_workspace / f"code_action_relaxed_{variant}_runtime_retry_0.py"
            workspace_code_path.write_text(retry_application.code, encoding="utf-8")
            application = retry_application
            final_code_path = retry_code_path
            render_result = renderer.render(
                application.code,
                rows=(),
                output_dir=experiment_workspace,
                output_filename="novice_final.png",
                file_path=workspace_code_path,
            )
            _write_json(case_dir / "render_result_after_runtime_retry.json", render_result_to_dict(render_result))

    counted_llm_items: list[Any] = [repair.get("trace")]
    for artifact_path in repair_artifacts[1:]:
        counted_llm_items.append(llm_usage_from_artifact_files(artifact_path))
    if variant == "fidelity_repair":
        counted_llm_items.insert(0, common.get("llm_usage"))
    counted_llm_usage = aggregate_llm_usage(counted_llm_items)
    fidelity_artifacts = variant == "fidelity_repair"

    return {
        "case_id": case_id,
        "native_id": common.get("native_id"),
        "variant": variant,
        "query": common.get("instruction"),
        "original_score": common.get("original_score"),
        "ground_truth_path": common.get("ground_truth_path"),
        "original_code_path": str(original_code_path),
        "final_code_path": str(final_code_path),
        "workspace_code_path": str(workspace_code_path),
        "image_path": str(render_result.image_path) if render_result and render_result.image_path else None,
        "render_ok": bool(render_result.ok) if render_result is not None else False,
        "render_exception_type": render_result.exception_type if render_result is not None else None,
        "render_exception_message": render_result.exception_message if render_result is not None else None,
        "repair_application": application.to_dict(),
        "agent_artifacts": {
            "repair": str(case_dir / "repair.json"),
            "repair_application": str(case_dir / "repair_application.json"),
            "render_result": str(case_dir / "render_result.json") if render else None,
            "common_input": common.get("input_path"),
            "script_observation": common.get("script_observation_path"),
            "expected_requirements": common.get("expected_requirements_path") if fidelity_artifacts else None,
            "diagnosis": common.get("diagnosis_path") if fidelity_artifacts else None,
            "source_file_evidence": common.get("source_file_evidence_path") if fidelity_artifacts else None,
            "source_schema_constraints": common.get("source_schema_constraints_path") if fidelity_artifacts else None,
            "source_data_violation": common.get("source_data_violation_path") if fidelity_artifacts else None,
            "repair_plan": common.get("repair_plan_path") if fidelity_artifacts else None,
            "runtime_retry_repair": str(case_dir / "runtime_retry_repair.json")
            if (case_dir / "runtime_retry_repair.json").exists()
            else None,
            "render_result_after_runtime_retry": str(case_dir / "render_result_after_runtime_retry.json")
            if (case_dir / "render_result_after_runtime_retry.json").exists()
            else None,
        },
        "llm_usage": counted_llm_usage,
        "metadata": {
            "source_workspace": str(source_workspace),
            "experiment_workspace": str(experiment_workspace),
            "repair_kind": (repair.get("payload") or {}).get("repair_kind") if isinstance(repair.get("payload"), dict) else None,
            "repair_summary": (repair.get("payload") or {}).get("summary") if isinstance(repair.get("payload"), dict) else None,
            "runtime_retry_applied": (case_dir / "runtime_retry_repair.json").exists(),
        },
    }


def build_repair_user_payload(common: dict[str, Any], *, variant: str, allow_full_rewrite: bool) -> dict[str, Any]:
    payload = {
        "task": "Repair the chart code for the benchmark instruction.",
        "case_id": common["case_id"],
        "instruction": common["instruction"],
        "original_score": common.get("original_score"),
        "code_context": common["code_context"],
        "script_observation": common["script_observation"],
        "repair_rules": repair_rules(allow_full_rewrite=allow_full_rewrite),
        "output_schema": repair_schema(),
    }
    if variant == "fidelity_repair":
        payload["source_file_evidence"] = common.get("source_file_evidence")
        payload["source_schema_constraints"] = common.get("source_schema_constraints")
        payload["source_data_violation"] = common.get("source_data_violation")
        payload["repair_plan"] = common.get("repair_plan")
        payload["expected_requirements"] = (
            common.get("expected_requirements", {}).get("payload")
            if isinstance(common.get("expected_requirements"), dict)
            else None
        )
        payload["fidelity_diagnosis"] = (
            common.get("diagnosis", {}).get("payload")
            if isinstance(common.get("diagnosis"), dict)
            else None
        )
    return payload


def run_runtime_retry(
    common: dict[str, Any],
    *,
    client: OpenAICompatibleLLMClient,
    case_dir: Path,
    failed_code: str,
    failed_render_result: dict[str, Any],
    temperature: float,
    max_tokens: int,
    max_patch_ops: int,
    max_changed_lines: int,
    allow_full_rewrite: bool,
) -> dict[str, Any]:
    payload = {
        "task": "Repair only the runtime/rendering error in the failed chart code.",
        "case_id": common["case_id"],
        "instruction": common["instruction"],
        "failed_render_result": failed_render_result,
        "failed_code_context": build_code_context(failed_code, max_chars=16000),
        "source_file_evidence": common.get("source_file_evidence"),
        "source_schema_constraints": common.get("source_schema_constraints"),
        "repair_plan": common.get("repair_plan"),
        "repair_rules": repair_rules(allow_full_rewrite=allow_full_rewrite),
        "output_schema": repair_schema(),
    }
    repair = call_agent_json(
        client,
        agent_name="runtime_retry_repair",
        system_prompt=RUNTIME_RETRY_SYSTEM_PROMPT,
        user_payload=payload,
        output_path=case_dir / "runtime_retry_repair.json",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    application = apply_repair_payload(
        failed_code,
        repair.get("payload"),
        max_patch_ops=max_patch_ops,
        max_changed_lines=max_changed_lines,
        allow_full_rewrite=allow_full_rewrite,
    )
    _write_json(case_dir / "runtime_retry_application.json", application.to_dict())
    return {"repair": repair, "application": application}


def build_code_context(code: str, *, max_chars: int) -> dict[str, Any]:
    text = str(code or "")
    lines = text.splitlines()
    if len(text) <= max(2000, int(max_chars)):
        return {
            "mode": "full",
            "char_count": len(text),
            "line_count": len(lines),
            "code": text,
        }
    snippets = _collect_code_snippets(lines, max_chars=max_chars)
    return {
        "mode": "snippets",
        "char_count": len(text),
        "line_count": len(lines),
        "omitted_note": "Code exceeded prompt budget; only relevant raw snippets are included.",
        "patch_instruction": (
            "If returning replace_text patch_ops, anchor.text must exactly match text from one included snippet."
        ),
        "snippets": snippets,
    }


def build_compact_script_observation(
    code: str,
    *,
    code_path: Path,
    execution_dir: Path,
    max_chars: int,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "execution_dir": str(execution_dir),
        "code_path": str(code_path),
        "trace_error": None,
        "plot_trace": None,
        "figure_trace": None,
    }
    try:
        run_trace = MatplotlibTraceRunner().run_code_with_figure(
            code,
            globals_dict={
                "rows": [],
                "OUTPUT_PATH": str(Path(execution_dir) / "groundedchart_trace_dummy.png"),
            },
            execution_dir=execution_dir,
            file_path=code_path,
        )
        observation["plot_trace"] = compact_plot_trace(run_trace.plot_trace)
        observation["figure_trace"] = compact_figure_trace(run_trace.figure_trace)
    except Exception as exc:
        observation["trace_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return compact_payload(observation, max_chars=max_chars)


def compact_plot_trace(trace: Any) -> dict[str, Any]:
    raw = getattr(trace, "raw", {}) or {}
    artifacts = raw.get("actual_intermediate_artifacts") if isinstance(raw, dict) else None
    return {
        "chart_type": getattr(trace, "chart_type", None),
        "point_count": len(getattr(trace, "points", ()) or ()),
        "points_preview": [jsonable(point) for point in list(getattr(trace, "points", ()) or ())[:12]],
        "source": getattr(trace, "source", None),
        "actual_intermediate_artifacts": compact_payload(artifacts or [], max_chars=2500),
        "raw_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
    }


def compact_figure_trace(trace: Any) -> dict[str, Any]:
    axes = list(getattr(trace, "axes", ()) or ())
    raw = getattr(trace, "raw", {}) or {}
    code_structure = []
    if isinstance(raw, dict):
        code_structure = raw.get("code_structure_artifacts") or []
    return {
        "title": getattr(trace, "title", None),
        "size_inches": jsonable(getattr(trace, "size_inches", None)),
        "axes_count": len(axes),
        "axes": [compact_axis_trace(axis) for axis in axes[:8]],
        "code_structure_artifacts": compact_payload(code_structure, max_chars=2500),
        "source": getattr(trace, "source", None),
        "raw_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
    }


def compact_axis_trace(axis: Any) -> dict[str, Any]:
    artists = list(getattr(axis, "artists", ()) or ())
    artist_counts: dict[str, int] = {}
    for artist in artists:
        artist_type = str(getattr(artist, "artist_type", None) or "unknown")
        artist_counts[artist_type] = artist_counts.get(artist_type, 0) + 1
    return {
        "index": getattr(axis, "index", None),
        "title": getattr(axis, "title", None),
        "xlabel": getattr(axis, "xlabel", None),
        "ylabel": getattr(axis, "ylabel", None),
        "zlabel": getattr(axis, "zlabel", None),
        "projection": getattr(axis, "projection", None),
        "xscale": getattr(axis, "xscale", None),
        "yscale": getattr(axis, "yscale", None),
        "zscale": getattr(axis, "zscale", None),
        "bounds": jsonable(getattr(axis, "bounds", None)),
        "legend_labels": list(getattr(axis, "legend_labels", ()) or ())[:16],
        "texts": list(getattr(axis, "texts", ()) or ())[:16],
        "xtick_labels_preview": _nonempty_preview(getattr(axis, "xtick_labels", ()) or (), limit=12),
        "ytick_labels_preview": _nonempty_preview(getattr(axis, "ytick_labels", ()) or (), limit=12),
        "ztick_labels_preview": _nonempty_preview(getattr(axis, "ztick_labels", ()) or (), limit=12),
        "artist_counts": artist_counts,
        "artists_preview": [jsonable(artist) for artist in artists[:12]],
    }


def apply_repair_payload(
    code: str,
    payload: Any,
    *,
    max_patch_ops: int,
    max_changed_lines: int,
    allow_full_rewrite: bool,
) -> RepairApplication:
    if not isinstance(payload, dict):
        return RepairApplication(code=code, source="no_valid_payload", rejected_reason="Repair payload was not a JSON object.")

    patch_ops = parse_patch_operations(payload.get("patch_ops"))
    if patch_ops:
        result = apply_patch_operations(
            code,
            patch_ops,
            max_operations=max_patch_ops,
            max_changed_lines=max_changed_lines,
        )
        if result.applied:
            return RepairApplication(code=result.code, source="patch_ops", patch_applied=True)
        if not allow_full_rewrite:
            return RepairApplication(
                code=code,
                source="patch_ops_rejected",
                rejected_reason=result.rejected_reason,
            )
        rejected_reason = result.rejected_reason
    else:
        rejected_reason = "No structured patch_ops were provided."

    repaired_code = _extract_repaired_code(payload)
    if allow_full_rewrite and repaired_code:
        return RepairApplication(
            code=repaired_code,
            source="repaired_code",
            patch_applied=False,
            rejected_reason=rejected_reason,
        )
    return RepairApplication(code=code, source="no_change", rejected_reason=rejected_reason)


def run_variant_evaluator(
    *,
    variant_summary_path: Path,
    variant_dir: Path,
    matplot_agent_root: Path,
    workers: int,
) -> dict[str, Any]:
    export_workspace = variant_dir / "eval_workspace"
    export_manifest = MatplotBenchEvalWorkspaceExporter(variant_summary_path, export_workspace).export()
    (variant_dir / "eval_export_manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evaluator_result = run_matplotbench_evaluator(
        matplot_agent_root=matplot_agent_root,
        workspace=export_workspace,
        native_ids=tuple(export_manifest.get("native_ids", ())),
        workers=workers,
    )
    summary = json.loads(variant_summary_path.read_text(encoding="utf-8"))
    combined = combine_evaluator_with_cases(summary.get("cases", []), evaluator_result)
    result = {
        "export_manifest": export_manifest,
        "evaluator_result": evaluator_result,
        "summary": combined,
    }
    (variant_dir / "evaluator_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


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
                details[str(native_id)] = normalize_matplotbench_evaluator_detail(result)
            else:
                details[str(native_id)] = {
                    "score": None,
                    "raw": None,
                    "error": "missing_eval_result_for_native_id",
                    "score_parse_status": "missing_eval_result_for_native_id",
                    "final_score_marker_present": False,
                }
        else:
            details[str(native_id)] = {
                "score": None,
                "raw": None,
                "error": f"evaluator_returncode_{completed.returncode}",
                "score_parse_status": f"evaluator_returncode_{completed.returncode}",
                "final_score_marker_present": False,
            }
    scores = [item.get("score") for item in details.values() if isinstance(item.get("score"), (int, float))]
    return {
        "workspace": str(workspace),
        "native_ids": list(native_ids),
        "details": details,
        "commands": commands,
        "summary": {
            "evaluated": len(scores),
            "official_avg_score_on_evaluated": round(sum(scores) / len(scores), 2) if scores else None,
        },
    }


def combine_evaluator_with_cases(cases: list[dict[str, Any]], evaluator_result: dict[str, Any]) -> dict[str, Any]:
    details = evaluator_result.get("details") or {}
    final_scores: list[float] = []
    original_scores: list[float] = []
    evaluated_cases = 0
    render_failed_cases = 0
    evaluator_parse_error_cases = 0
    case_details: dict[str, Any] = {}
    for case in cases:
        native_id = case.get("native_id")
        original = _float_or_none(case.get("original_score"))
        if original is not None:
            original_scores.append(original)
        render_ok = bool(case.get("render_ok"))
        if not render_ok:
            render_failed_cases += 1
            final = 0.0
            score_source = "render_failed_no_image_score_0"
        else:
            detail = details.get(str(native_id), {}) if native_id is not None else {}
            score = _float_or_none(detail.get("score"))
            score_parse_status = detail.get("score_parse_status")
            if score is None:
                final = 0.0
                if score_parse_status:
                    score_source = f"{score_parse_status}_score_0"
                    if str(score_parse_status).startswith("evaluator_parse_error"):
                        evaluator_parse_error_cases += 1
                else:
                    score_source = "missing_evaluator_score_0"
            else:
                final = score
                evaluated_cases += 1
                score_source = (
                    "matplotbench_visual_evaluator_final_score_marker"
                    if score_parse_status == "final_score_marker"
                    else "matplotbench_visual_evaluator"
                )
        final_scores.append(final)
        detail = details.get(str(native_id), {}) if native_id is not None else {}
        case_details[str(native_id or case.get("case_id"))] = {
            "case_id": case.get("case_id"),
            "original_score": original,
            "final_score": final,
            "delta": None if original is None else final - original,
            "render_ok": render_ok,
            "score_source": score_source,
            "evaluator_score_parse_status": detail.get("score_parse_status"),
            "final_score_marker_present": detail.get("final_score_marker_present"),
            "raw_score_from_evaluator": detail.get("raw_score_from_evaluator"),
            "evaluator_error": detail.get("error"),
        }
    total = len(cases)
    original_avg = sum(original_scores) / len(original_scores) if original_scores else None
    final_avg = sum(final_scores) / len(final_scores) if final_scores else None
    return {
        "total_cases": total,
        "primary_metric": "official_avg_delta",
        "scoring_standard": "MatPlotBench official visual evaluator score; valid scores require an explicit [FINAL SCORE] marker. Runtime render status is diagnostic only.",
        "evaluated_cases": evaluated_cases,
        "runtime_render_failed_cases_diagnostic": render_failed_cases,
        "evaluator_parse_error_cases": evaluator_parse_error_cases,
        "original_avg_score": round(original_avg, 4) if original_avg is not None else None,
        "official_final_avg_score": round(final_avg, 4) if final_avg is not None else None,
        "official_avg_delta": round(final_avg - original_avg, 4) if final_avg is not None and original_avg is not None else None,
        "details": case_details,
    }


def build_variant_summary(
    *,
    variant: str,
    bench_path: Path,
    config_path: Path,
    output_dir: Path,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "relaxed_fidelity_ablation",
        "variant": variant,
        "evaluation_standard": {
            "primary_metric": "official_evaluator_score_delta",
            "official_evaluator_required_for_effect_claims": True,
            "runtime_render_ok_cases": "diagnostic_only",
        },
        "bench_path": str(bench_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "total_cases": len(cases),
        "runtime_render_ok_cases_diagnostic": sum(1 for case in cases if case.get("render_ok")),
        "llm_usage": aggregate_llm_usage([case.get("llm_usage") for case in cases]),
        "cases": cases,
    }


def build_relaxed_comparisons(variant_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    vanilla = variant_summaries.get("vanilla_repair")
    fidelity = variant_summaries.get("fidelity_repair")
    if not vanilla or not fidelity:
        return []
    comparison = {
        "comparison_type": "fidelity_vs_vanilla_relaxed_repair",
        "baseline": "vanilla_repair",
        "candidate": "fidelity_repair",
        "llm_call_delta": fidelity["llm_usage"]["call_count"] - vanilla["llm_usage"]["call_count"],
        "llm_token_delta": fidelity["llm_usage"]["total_tokens"] - vanilla["llm_usage"]["total_tokens"],
    }
    vanilla_eval = (((vanilla.get("evaluator_summary") or {}).get("summary")) or {})
    fidelity_eval = (((fidelity.get("evaluator_summary") or {}).get("summary")) or {})
    if vanilla_eval and fidelity_eval:
        comparison["primary_metric"] = "official_avg_score_delta"
        comparison["official_avg_score_delta"] = _safe_numeric_delta(
            fidelity_eval.get("official_final_avg_score"),
            vanilla_eval.get("official_final_avg_score"),
        )
        comparison["official_avg_delta_delta"] = _safe_numeric_delta(
            fidelity_eval.get("official_avg_delta"),
            vanilla_eval.get("official_avg_delta"),
        )
    else:
        comparison["primary_metric"] = "official_avg_score_delta"
        comparison["official_score_status"] = "unavailable_without_official_evaluator"
    return [comparison]


def _safe_numeric_delta(candidate: Any, baseline: Any) -> float | None:
    candidate_value = _float_or_none(candidate)
    baseline_value = _float_or_none(baseline)
    if candidate_value is None or baseline_value is None:
        return None
    return round(candidate_value - baseline_value, 4)


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
    trace = client.complete_text_with_trace(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parse_error = None
    try:
        payload = extract_json_object(trace.raw_text)
    except Exception as exc:
        payload = {}
        parse_error = {"type": type(exc).__name__, "message": str(exc)}
    wrapped = {
        "agent": agent_name,
        "payload": payload,
        "trace": llm_trace_to_dict(trace),
        "parse_error": parse_error,
    }
    _write_json(output_path, wrapped)
    if parse_error is not None:
        raise ValueError(f"{agent_name} returned invalid JSON: {parse_error['message']}")
    return wrapped


def expected_schema() -> dict[str, Any]:
    return {
        "requirements": [
            {
                "id": "short stable id",
                "claim": "source-grounded expected chart requirement",
                "family": "data|chart_type|layout|encoding|axis|legend|annotation|style|scale|composition|other",
                "source_span": "verbatim supporting instruction span",
                "priority": "critical|major|minor",
                "verifiability": "script_supported|vision_needed|human_or_bench_only|uncertain",
                "confidence": 0.0,
            }
        ],
        "global_notes": [],
        "uncertainties": [],
    }


def diagnosis_schema() -> dict[str, Any]:
    return {
        "mismatches": [
            {
                "expected_id": "requirement id if known",
                "problem": "concise mismatch statement, <=18 words",
                "severity": "critical|major|minor",
                "evidence": "actual trace or runtime evidence, <=18 words",
                "repair_hint": "localized code-level hint, <=18 words",
                "confidence": 0.0,
            }
        ],
        "repair_priorities": ["<=5 short items"],
        "risk_notes": ["<=3 short items"],
    }


def repair_schema() -> dict[str, Any]:
    return {
        "summary": "one concise sentence, <=60 words",
        "repair_kind": "patch_ops|full_rewrite|abstain",
        "patch_ops": [
            {
                "op": "replace_text|replace_call_arg|replace_keyword_arg|remove_keyword_arg|insert_after_anchor",
                "anchor": {
                    "kind": "text|method_call|function_call",
                    "text": "exact old text for text anchors",
                    "name": "call name for call anchors",
                    "occurrence": 1,
                },
                "arg_index": 0,
                "keyword": "keyword name when needed",
                "new_value": "replacement value or inserted code",
                "description": "why this edit is local and safe",
            }
        ],
        "repaired_code": "only include full executable Python code when patch_ops cannot safely express the repair",
        "expected_improvements": ["<=3 short items"],
        "risk_notes": ["<=3 short items"],
        "unresolved_requirements": ["<=5 short items"],
    }


def repair_rules(*, allow_full_rewrite: bool) -> list[str]:
    rules = [
        "Return patch_ops when the fix can be localized.",
        "Use replace_text only with anchor.text copied exactly from code_context.",
        "Do not change unrelated subplot layout, chart type, or data generation unless the instruction/diagnosis requires it.",
        "The final code must save a visible figure to OUTPUT_PATH if that variable exists, otherwise to novice_final.png.",
        "Do not call network APIs or require interactive display.",
        "Keep summary and notes short. Do not repeat the same finding.",
        "If no safe repair is clear, return repair_kind='abstain' with empty patch_ops and no repaired_code.",
    ]
    if allow_full_rewrite:
        rules.append("Use repaired_code only for broad structural fixes or when patch_ops would be unsafe.")
    else:
        rules.append("Do not return full repaired_code; use patch_ops or abstain.")
    return rules


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
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
            "raw": jsonable(getattr(usage, "raw", {})) if usage is not None else {},
        },
        "raw_text": getattr(trace, "raw_text", ""),
        "parsed_json": jsonable(getattr(trace, "parsed_json", None)),
    }


def aggregate_llm_usage(items: list[Any]) -> dict[str, int]:
    total = {
        "call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "missing_usage_count": 0,
    }
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict) and "call_count" in item:
            total["call_count"] += int(item.get("call_count") or 0)
            total["prompt_tokens"] += int(item.get("prompt_tokens") or 0)
            total["completion_tokens"] += int(item.get("completion_tokens") or 0)
            total["total_tokens"] += int(item.get("total_tokens") or 0)
            total["missing_usage_count"] += int(item.get("missing_usage_count") or 0)
            continue
        if isinstance(item, dict) and "usage" in item:
            total["call_count"] += 1
            usage = item.get("usage") or {}
            if not usage:
                total["missing_usage_count"] += 1
                continue
            total["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            total["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            total["total_tokens"] += int(usage.get("total_tokens") or 0)
            continue
        total["missing_usage_count"] += 1
    return total


def llm_usage_from_artifact_files(*paths: Path) -> dict[str, int]:
    traces: list[Any] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("trace"):
            traces.append(payload.get("trace"))
    return aggregate_llm_usage(traces)


def compact_payload(value: Any, *, max_chars: int) -> Any:
    normalized = jsonable(value)
    if len(json.dumps(normalized, ensure_ascii=False)) <= max_chars:
        return normalized
    compacted = _compact_recursive(normalized, depth=0)
    if len(json.dumps(compacted, ensure_ascii=False)) <= max_chars:
        return compacted
    text = json.dumps(compacted, ensure_ascii=False)
    return {
        "truncated": True,
        "char_budget": max_chars,
        "preview_json": _truncate_middle(text, max_chars),
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


def common_error_case(raw_case: dict[str, Any], *, output_dir: Path, error: BaseException) -> dict[str, Any]:
    case_id = str(raw_case.get("case_id") or "unknown_case")
    common_dir = output_dir / "cases" / _safe_name(case_id) / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "native_id": _native_id(raw_case),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    _write_json(common_dir / "error.json", payload)
    llm_usage = llm_usage_from_artifact_files(
        common_dir / "expected_requirements.json",
        common_dir / "diagnosis.json",
    )
    return {
        "case_id": case_id,
        "native_id": _native_id(raw_case),
        "original_score": raw_case.get("score"),
        "instruction": _instruction_text(raw_case),
        "selected_code_path": raw_case.get("selected_code_path"),
        "source_workspace": raw_case.get("workspace_dir"),
        "ground_truth_path": raw_case.get("ground_truth_path"),
        "common_dir": str(common_dir),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "llm_usage": llm_usage,
    }


def variant_error_case(common: dict[str, Any], *, variant: str, output_dir: Path, error: BaseException) -> dict[str, Any]:
    case_id = str(common.get("case_id") or "unknown_case")
    case_dir = output_dir / variant / "cases" / _safe_name(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "variant": variant,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    _write_json(case_dir / "error.json", payload)
    llm_usage_items: list[Any] = []
    if variant == "fidelity_repair":
        llm_usage_items.append(common.get("llm_usage"))
    llm_usage_items.append(llm_usage_from_artifact_files(case_dir / "repair.json"))
    return {
        "case_id": case_id,
        "native_id": common.get("native_id"),
        "variant": variant,
        "query": common.get("instruction"),
        "original_score": common.get("original_score"),
        "ground_truth_path": common.get("ground_truth_path"),
        "final_code_path": None,
        "image_path": None,
        "render_ok": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "agent_artifacts": {"error": str(case_dir / "error.json")},
        "llm_usage": aggregate_llm_usage(llm_usage_items),
    }


def parse_variants(raw: str) -> tuple[str, ...]:
    variants = tuple(dict.fromkeys(part.strip() for part in str(raw or "").split(",") if part.strip()))
    if not variants:
        raise ValueError("--variants must include at least one variant.")
    unknown = sorted(set(variants) - VALID_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {sorted(VALID_VARIANTS)}")
    return variants


def parse_case_ids(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


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
    for key in ("query", "simple_instruction", "instruction", "prompt", "expert_instruction"):
        value = raw_case.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _native_id(raw_case: dict[str, Any]) -> int | None:
    for key in ("native_id", "id"):
        value = raw_case.get(key)
        try:
            if value is not None and str(value).strip():
                return int(value)
        except (TypeError, ValueError):
            continue
    text = str(raw_case.get("case_id") or "")
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    if digits:
        return int(digits[-1])
    return None


def _collect_code_snippets(lines: list[str], *, max_chars: int) -> list[dict[str, Any]]:
    keywords = (
        "plt.",
        "ax.",
        "fig.",
        "sns.",
        "plotly",
        "go.",
        "px.",
        "savefig",
        "OUTPUT_PATH",
        "read_csv",
        "open(",
        "np.",
        "pd.",
        "subplots",
        "add_subplot",
        "set_",
        "legend",
        "annotate",
        "hist",
        "bar",
        "scatter",
        "plot(",
        "imshow",
    )
    windows: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if index < 25 or any(keyword in line for keyword in keywords):
            windows.append((max(0, index - 3), min(len(lines), index + 4)))
    merged = _merge_windows(windows)
    snippets: list[dict[str, Any]] = []
    used_chars = 0
    for start, end in merged:
        text = "\n".join(lines[start:end])
        if used_chars + len(text) > max_chars and snippets:
            break
        snippets.append({"start_line": start + 1, "end_line": end, "text": text})
        used_chars += len(text)
    if not snippets:
        snippets.append({"start_line": 1, "end_line": min(len(lines), 80), "text": "\n".join(lines[:80])})
    return snippets


def build_source_file_evidence(
    *,
    source_workspace: Path,
    instruction: str,
    max_preview_rows: int,
    max_files: int,
) -> dict[str, Any]:
    plan = SourceDataPlanner().build_plan(
        workspace=source_workspace,
        instruction=instruction,
        max_preview_rows=max_preview_rows,
        max_files=max_files,
    )
    files = [item.to_dict() for item in plan.files]
    return {
        "workspace": str(source_workspace),
        "mentioned_files": list(plan.mentioned_files),
        "available_files": files,
        "mentioned_available_files": [name for name in plan.mentioned_files if name not in plan.missing_mentioned_files],
        "missing_mentioned_files": list(plan.missing_mentioned_files),
        "has_available_data_files": bool(files),
    }


def build_source_schema_constraints(source_file_evidence: dict[str, Any]) -> dict[str, Any]:
    from grounded_chart.source_data import schema_constraint_for_file

    class _Summary:
        def __init__(self, name: str, columns: list[str]) -> None:
            self.name = name
            self.columns = tuple(columns)

    files = [
        schema_constraint_for_file(_Summary(str(item.get("name") or ""), [str(column) for column in list(item.get("columns") or [])]))
        for item in list(source_file_evidence.get("available_files") or [])
        if isinstance(item, dict)
    ]
    return {
        "files": files,
        "global_constraints": [
            "Treat listed columns as exact unless your code checks and creates derived columns.",
            "Do not pivot on a column unless that column exists or was explicitly created earlier.",
            "For wide tables with Year plus measure columns, use direct column access or melt before long-form plotting.",
            "When renaming columns, update all downstream references consistently.",
        ],
    }


def infer_table_schema_type(columns: list[str]) -> str:
    normalized = {column.strip().lower() for column in columns}
    if "year" in normalized and len(columns) >= 3:
        non_year = [column for column in normalized if column != "year"]
        if {"urban", "rural"}.issubset(normalized) or len(non_year) >= 2:
            return "wide_year_measure_table"
    if {"category", "value"}.issubset(normalized):
        return "long_category_value_table"
    return "unknown_table"


def schema_usage_constraints(*, name: str, columns: list[str]) -> list[str]:
    constraints = [f"{name}: exact columns are {columns}."]
    normalized = {column.strip().lower() for column in columns}
    if "year" in normalized and len(columns) >= 3:
        measure_columns = [column for column in columns if column.strip().lower() != "year"]
        constraints.append(
            f"{name}: this is wide time data; access measure columns {measure_columns} directly or melt before using Category/Value."
        )
        constraints.append(f"{name}: do not call pivot(... columns='Category', values='Value') before creating those columns.")
    category_like = [column for column in columns if any(token in column.strip().lower() for token in ("category", "group", "label", "type", "segment"))]
    value_like = [column for column in columns if any(token in column.strip().lower() for token in ("ratio", "rate", "percent", "share", "value"))]
    if category_like and value_like:
        constraints.append(f"{name}: preserve category columns {category_like} and value/ratio columns {value_like} semantics when renaming.")
    return constraints


def _summarize_source_file(path: Path, *, max_preview_rows: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "columns": [],
        "preview_rows": [],
        "row_count_preview": None,
        "read_error": None,
    }
    try:
        if path.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                summary["columns"] = list(reader.fieldnames or [])
                rows = []
                for index, row in enumerate(reader):
                    if index >= max_preview_rows:
                        break
                    rows.append({str(key): _short_cell(value) for key, value in dict(row).items()})
                summary["preview_rows"] = rows
                summary["row_count_preview"] = len(rows)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            summary["json_type"] = type(payload).__name__
            if isinstance(payload, list):
                summary["row_count_preview"] = min(len(payload), max_preview_rows)
                summary["preview_rows"] = compact_payload(payload[:max_preview_rows], max_chars=1800)
                if payload and isinstance(payload[0], dict):
                    summary["columns"] = list(payload[0].keys())
            elif isinstance(payload, dict):
                summary["columns"] = list(payload.keys())[:24]
                summary["preview_rows"] = compact_payload(payload, max_chars=1800)
        else:
            summary["read_error"] = "preview_not_supported_for_excel"
    except Exception as exc:
        summary["read_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return summary


def detect_source_data_violation(
    code: str,
    *,
    source_file_evidence: dict[str, Any],
    instruction: str,
) -> dict[str, Any]:
    code_text = str(code or "")
    available_files = list(source_file_evidence.get("available_files") or [])
    mentioned_available = list(source_file_evidence.get("mentioned_available_files") or [])
    expected_file_names = mentioned_available or [str(item.get("name")) for item in available_files if item.get("name")]
    used_files = sorted(
        name for name in expected_file_names if _filename_referenced_in_code(code_text, name)
    )
    missing_file_uses = [name for name in expected_file_names if name not in used_files]
    synthetic_hits = [
        {"pattern": name, "matches": sorted(set(match.group(0) for match in pattern.finditer(code_text)))[:5]}
        for name, pattern in SYNTHETIC_DATA_PATTERNS
        if pattern.search(code_text)
    ]
    uses_data_loader = bool(DATA_LOAD_CALL_RE.search(code_text))
    severity = "none"
    if expected_file_names and synthetic_hits and missing_file_uses:
        severity = "critical"
    elif expected_file_names and missing_file_uses:
        severity = "major"
    elif synthetic_hits:
        severity = "minor"
    return {
        "has_violation": severity in {"critical", "major"},
        "severity": severity,
        "expected_file_names": expected_file_names,
        "used_files": used_files,
        "missing_file_uses": missing_file_uses,
        "uses_data_loader": uses_data_loader,
        "synthetic_data_indicators": synthetic_hits,
        "rationale": _source_violation_rationale(
            severity=severity,
            expected_file_names=expected_file_names,
            missing_file_uses=missing_file_uses,
            synthetic_hits=synthetic_hits,
        ),
    }


def plan_initial_repair(
    *,
    source_file_evidence: dict[str, Any],
    source_data_violation: dict[str, Any],
) -> dict[str, Any]:
    if source_data_violation.get("severity") == "critical":
        return {
            "repair_level": 3,
            "repair_action": "structural_regeneration",
            "allowed_edit_scope": "full_chart_code",
            "target_failures": ["source_data_violation", "data_loading", "data_transformation", "chart_structure"],
            "constraints": [
                "Read available source files by filename; do not synthesize replacement data.",
                "Use file schema and preview rows as the source of data truth.",
                "Preserve explicit requirements from the instruction.",
                "Save the final rendered figure to OUTPUT_PATH if available, otherwise novice_final.png.",
            ],
            "reason": source_data_violation.get("rationale"),
        }
    if source_data_violation.get("severity") == "major":
        return {
            "repair_level": 2,
            "repair_action": "data_patch",
            "allowed_edit_scope": "data_loading_and_transformation",
            "target_failures": ["source_data_violation", "data_loading"],
            "constraints": [
                "Patch data loading to use available source files.",
                "Preserve plotting structure unless bindings must change.",
                "Do not synthesize replacement data.",
            ],
            "reason": source_data_violation.get("rationale"),
        }
    return {
        "repair_level": 1,
        "repair_action": "local_patch",
        "allowed_edit_scope": "localized_code_edits",
        "target_failures": [],
        "constraints": [
            "Prefer localized patch operations.",
            "Escalate only if evidence justifies broader changes.",
        ],
        "reason": "No source-data violation was detected by static evidence.",
    }


def _source_violation_rationale(
    *,
    severity: str,
    expected_file_names: list[str],
    missing_file_uses: list[str],
    synthetic_hits: list[dict[str, Any]],
) -> str:
    if severity == "critical":
        return (
            "Source files are available/required, but code omits them and contains synthetic data indicators: "
            f"missing={missing_file_uses}, synthetic={[item['pattern'] for item in synthetic_hits]}."
        )
    if severity == "major":
        return f"Source files are available/required but not all are referenced in code: missing={missing_file_uses}."
    if severity == "minor":
        return f"Synthetic data indicators found without required source-file evidence: {[item['pattern'] for item in synthetic_hits]}."
    return f"No required source-file mismatch detected for files: {expected_file_names}."


def _filename_referenced_in_code(code: str, filename: str) -> bool:
    name = _normalize_filename(filename)
    if not name:
        return False
    code_lower = str(code or "").lower().replace("\\", "/")
    return name.lower().replace("\\", "/") in code_lower


def _normalize_filename(name: str) -> str:
    return Path(str(name).strip().replace("\\", "/")).name


def _short_cell(value: Any) -> Any:
    text = "" if value is None else str(value)
    if len(text) <= 80:
        return text
    return text[:77] + "..."


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for start, end in windows[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _copy_workspace(source: Path, target: Path) -> None:
    if target.exists():
        return
    if source.exists() and source.is_dir():
        ignore = shutil.ignore_patterns("*.log", "__pycache__", ".ipynb_checkpoints")
        shutil.copytree(source, target, ignore=ignore)
    else:
        target.mkdir(parents=True, exist_ok=True)


def _extract_repaired_code(payload: dict[str, Any]) -> str | None:
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


def _compact_recursive(value: Any, *, depth: int) -> Any:
    if depth > 4:
        return "<truncated_depth>"
    if isinstance(value, str):
        return _truncate_middle(value, 900)
    if isinstance(value, list):
        compacted = [_compact_recursive(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            compacted.append({"truncated_items": len(value) - 8})
        return compacted
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 20:
                result["truncated_keys"] = len(value) - 20
                break
            result[str(key)] = _compact_recursive(item, depth=depth + 1)
        return result
    return value


def _nonempty_preview(values: Any, *, limit: int) -> list[str]:
    result = []
    for value in list(values)[: limit * 2]:
        text = str(value or "").strip()
        if text:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n... <truncated> ...\n" + text[-tail:]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
