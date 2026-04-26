import argparse
from pathlib import Path

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
    TableSchema,
    TieredRepairer,
    load_ablation_run_config,
)

STATIC_BAR_CODE = """
import matplotlib.pyplot as plt

labels = [row["product"] for row in rows]
values = [row["sales"] for row in rows]
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(labels, values, color="#4C78A8")
ax.set_xlabel("product")
ax.set_ylabel("sales")
ax.set_title("Sales by product")
fig.tight_layout()
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
""".strip()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "generation_pipeline_smoke"
    config = load_ablation_run_config(project_root / args.config) if args.config else AblationRunConfig()

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

    result = generation_pipeline.run(
        query=args.query,
        schema=TableSchema(columns={"product": "string", "sales": "number"}),
        rows=(
            {"product": "Alpha", "sales": 12},
            {"product": "Beta", "sales": 18},
            {"product": "Gamma", "sales": 9},
        ),
        output_dir=output_dir,
        case_id=args.case_id,
    )

    print("Output dir:", result.output_dir)
    print("Initial code:", result.initial_code_path)
    print("Final code:", result.final_code_path)
    print("Image:", result.image_path)
    print("Manifest:", result.manifest_path)
    print("Report:", result.report_path)
    print("Verification ok:", result.pipeline_result.report.ok)
    print("Render ok:", result.render_result.ok)
    print("Error codes:", list(result.pipeline_result.report.error_codes))


def parse_args():
    parser = argparse.ArgumentParser(description="Run GroundedChart instruction-to-image generation smoke pipeline.")
    parser.add_argument("--config", help="YAML/TOML LLM config path relative to project root or absolute path.")
    parser.add_argument("--llm-codegen", action="store_true", help="Use configured LLM provider for code generation.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override codegen max_tokens for this run.")
    parser.add_argument("--output-dir", help="Directory for generated code, image, manifest, and report.")
    parser.add_argument("--case-id", default="generation_smoke")
    parser.add_argument(
        "--query",
        default="Create a bar chart of sales by product titled Sales by product.",
    )
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
        return StaticChartCodeGenerator(STATIC_BAR_CODE, generator_name="static_smoke")
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


if __name__ == "__main__":
    main()
