from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from grounded_chart import (
    AblationRunConfig,
    ChartGenerationPipeline,
    GroundedChartPipeline,
    HeuristicIntentParser,
    LLMChartCodeGenerator,
    LLMIntentParser,
    LLMRepairer,
    OpenAICompatibleLLMClient,
    RuleBasedRepairer,
    StaticChartCodeGenerator,
    TieredRepairer,
    load_ablation_run_config,
)
from grounded_chart_adapters import MatplotBenchGenerationAdapter

STATIC_INSTRUCTION_ONLY_CODE = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, "MatPlotBench generation smoke", ha="center", va="center", fontsize=12)
ax.set_axis_off()
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
""".strip()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = _resolve_path(args.manifest, project_root)
    output_dir = _resolve_path(args.output_dir, project_root) if args.output_dir else project_root / "outputs" / "matplotbench_generation_pipeline"
    config = load_ablation_run_config(_resolve_path(args.config, project_root)) if args.config else AblationRunConfig()

    adapter = MatplotBenchGenerationAdapter(
        manifest_path,
        case_ids=parse_case_ids(args.case_ids),
        limit=args.limit,
    )
    verifier_pipeline = GroundedChartPipeline(
        parser=build_parser(config),
        repairer=build_repairer(config),
        repair_policy_mode=config.repair_policy_mode,
        enable_bounded_repair_loop=config.enable_repair_loop,
        max_repair_rounds=config.max_repair_rounds,
    )
    generation_pipeline = ChartGenerationPipeline(
        code_generator=build_code_generator(config, use_llm=args.llm_codegen, max_tokens=args.max_tokens),
        verifier_pipeline=verifier_pipeline,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    case_summaries: list[dict[str, Any]] = []
    for case in adapter.iter_cases():
        case_dir = output_dir / _safe_name(case.case_id)
        try:
            result = generation_pipeline.run(
                query=case.query,
                schema=case.schema,
                rows=case.rows,
                output_dir=case_dir,
                case_id=case.case_id,
                verification_mode=args.verification_mode,
                generation_mode=case.generation_mode,
                generation_context=case.generation_context(),
            )
            case_summaries.append(case_summary(case, result=result, error=None))
            print(case.case_id, "render_ok=", result.render_result.ok, "verify_ok=", result.pipeline_result.report.ok, "image=", result.image_path)
        except Exception as exc:
            case_summaries.append(case_summary(case, result=None, error=exc, case_dir=case_dir))
            print(case.case_id, "ERROR", type(exc).__name__, str(exc))
            if not args.continue_on_error:
                raise

    summary = build_summary(
        manifest_path=manifest_path,
        output_dir=output_dir,
        config_path=_resolve_path(args.config, project_root) if args.config else None,
        llm_codegen=args.llm_codegen,
        cases=case_summaries,
    )
    summary_path = output_dir / "matplotbench_generation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary:", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MatPlotBench manifests through ChartGenerationPipeline.")
    parser.add_argument(
        "--manifest",
        default="benchmarks/matplotbench_ds_failed_native.json",
        help="MatPlotBench generation manifest JSON, absolute or relative to project root.",
    )
    parser.add_argument("--config", help="YAML/TOML LLM config path, absolute or relative to project root.")
    parser.add_argument("--llm-codegen", action="store_true", help="Use configured LLM for instruction-to-code generation.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override codegen max_tokens.")
    parser.add_argument("--output-dir", help="Output directory, absolute or relative to project root.")
    parser.add_argument("--case-ids", help="Comma-separated case_id or native_id filter.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verification-mode", default="figure_only", choices=("full", "figure_only", "figure_and_data"))
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def build_parser(config: AblationRunConfig):
    if config.parser_backend == "llm":
        return LLMIntentParser(OpenAICompatibleLLMClient(require_provider(config.parser_provider, role="parser")))
    return HeuristicIntentParser()


def build_repairer(config: AblationRunConfig):
    if config.repair_backend == "tiered":
        llm_repairer = LLMRepairer(OpenAICompatibleLLMClient(require_provider(config.repair_provider, role="repair")))
        if config.repair_tier_mode == "llm_only":
            return TieredRepairer(
                deterministic_repairer=RuleBasedRepairer(),
                llm_repairer=llm_repairer,
                llm_scopes=(
                    "local_patch",
                    "data_transformation",
                    "structural_regeneration",
                    "backend_specific_regeneration",
                ),
            )
        if config.repair_tier_mode == "hybrid":
            return TieredRepairer(deterministic_repairer=RuleBasedRepairer(), llm_repairer=llm_repairer)
        return RuleBasedRepairer()
    if config.repair_backend == "llm":
        return LLMRepairer(OpenAICompatibleLLMClient(require_provider(config.repair_provider, role="repair")))
    return RuleBasedRepairer()


def build_code_generator(config: AblationRunConfig, *, use_llm: bool, max_tokens: int | None):
    if not use_llm:
        return StaticChartCodeGenerator(STATIC_INSTRUCTION_ONLY_CODE, generator_name="static_matplotbench_smoke")
    provider = config.codegen_provider or config.parser_provider or config.repair_provider
    return LLMChartCodeGenerator(
        OpenAICompatibleLLMClient(require_provider(provider, role="codegen")),
        max_tokens=max_tokens,
    )


def require_provider(provider, *, role: str):
    if provider is None:
        raise ValueError(f"LLM {role} backend is enabled but no provider configuration was found.")
    if not str(provider.api_key or "").strip():
        raise ValueError(f"LLM {role} backend is enabled but API key is empty.")
    return provider


def case_summary(case, *, result, error: BaseException | None, case_dir: Path | None = None) -> dict[str, Any]:
    if result is None:
        return {
            "case_id": case.case_id,
            "generation_mode": case.generation_mode,
            "query": case.query,
            "output_dir": str(case_dir) if case_dir else None,
            "ok": False,
            "render_ok": False,
            "verification_ok": False,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            "metadata": case.generation_context(),
        }
    return {
        "case_id": case.case_id,
        "generation_mode": case.generation_mode,
        "query": case.query,
        "output_dir": str(result.output_dir),
        "image_path": str(result.image_path) if result.image_path else None,
        "initial_code_path": str(result.initial_code_path),
        "final_code_path": str(result.final_code_path),
        "manifest_path": str(result.manifest_path),
        "report_path": str(result.report_path),
        "ok": bool(result.render_result.ok and result.pipeline_result.report.ok),
        "render_ok": result.render_result.ok,
        "verification_ok": result.pipeline_result.report.ok,
        "error_codes": list(result.pipeline_result.report.error_codes),
        "final_code_source": result.metadata.get("final_code_source"),
        "final_code_verified": result.metadata.get("final_code_verified"),
        "render_exception_type": result.render_result.exception_type,
        "render_exception_message": result.render_result.exception_message,
        "metadata": case.generation_context(),
    }


def build_summary(*, manifest_path: Path, output_dir: Path, config_path: Path | None, llm_codegen: bool, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "config_path": str(config_path) if config_path else None,
        "llm_codegen": llm_codegen,
        "total_cases": len(cases),
        "render_ok": sum(1 for case in cases if case.get("render_ok")),
        "verification_ok": sum(1 for case in cases if case.get("verification_ok")),
        "all_ok": sum(1 for case in cases if case.get("ok")),
        "cases": cases,
    }


def parse_case_ids(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _resolve_path(value: str | None, project_root: Path) -> Path:
    if value is None:
        raise ValueError("Path value is required.")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe.strip("_") or "case"


if __name__ == "__main__":
    main()