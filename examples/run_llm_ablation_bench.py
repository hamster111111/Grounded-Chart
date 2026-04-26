import argparse
import os
from pathlib import Path

from grounded_chart import (
    AblationRunConfig,
    GroundedChartPipeline,
    HeuristicIntentParser,
    LLMIntentParser,
    LLMRepairer,
    OpenAICompatibleLLMClient,
    RuleBasedRepairer,
    TieredRepairer,
    load_ablation_run_config,
    load_ablation_run_config_from_env,
)
from grounded_chart_adapters import BatchRunner, JsonCaseAdapter, write_batch_report_html


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    bench_name = config.bench_name
    bench_path = project_root / "benchmarks" / f"{bench_name}.json"
    output_name = config.output_name or f"{bench_name}_llm_ablation"
    output_dir = project_root / "outputs" / output_name

    parser = build_parser(config)
    repairer = build_repairer(config)
    pipeline = GroundedChartPipeline(
        parser=parser,
        repairer=repairer,
        repair_policy_mode=config.repair_policy_mode,
        enable_bounded_repair_loop=config.enable_repair_loop,
        max_repair_rounds=config.max_repair_rounds,
    )
    adapter = JsonCaseAdapter(bench_path)
    batch = BatchRunner(pipeline, continue_on_error=True).run(adapter)

    output_dir.mkdir(parents=True, exist_ok=True)
    batch.report.write_json(output_dir / "report.json")
    batch.report.write_jsonl(output_dir / "cases.jsonl")
    write_batch_report_html(
        batch.report,
        output_dir / "report.html",
        title=f"GroundedChart LLM Ablation: {bench_name}",
    )

    print("Bench:", bench_path)
    print("Config:", args.config or "env")
    print("Parser backend:", config.parser_backend)
    print("Repair backend:", config.repair_backend)
    print("Repair policy mode:", config.repair_policy_mode)
    print("Repair loop enabled:", config.enable_repair_loop)
    print("Report:", output_dir / "report.json")
    print("HTML:", output_dir / "report.html")
    print("Summary:", batch.report.summary.to_dict())


def parse_args():
    parser = argparse.ArgumentParser(description="Run GroundedChart parser/repair ablations.")
    parser.add_argument(
        "--config",
        help="Path to a TOML config file. If omitted, environment variables are used.",
    )
    return parser.parse_args()


def load_config(path: str | None) -> AblationRunConfig:
    if path:
        return load_ablation_run_config(path)
    return load_ablation_run_config_from_env()


def build_parser(config: AblationRunConfig):
    if config.parser_backend == "llm":
        return LLMIntentParser(build_client(config.parser_provider, role="parser"))
    return HeuristicIntentParser()


def build_repairer(config: AblationRunConfig, *, include_failure_atoms: bool = True):
    if config.repair_backend == "tiered":
        llm_repairer = LLMRepairer(build_client(config.repair_provider, role="repair"), include_failure_atoms=include_failure_atoms)
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
            return TieredRepairer(
                deterministic_repairer=RuleBasedRepairer(),
                llm_repairer=llm_repairer,
            )
        return RuleBasedRepairer()
    if config.repair_backend == "llm":
        return LLMRepairer(build_client(config.repair_provider, role="repair"), include_failure_atoms=include_failure_atoms)
    return RuleBasedRepairer()


def build_client(config, *, role: str):
    if config is None:
        raise ValueError(f"LLM {role} backend is enabled but no provider configuration was found.")
    if not str(config.api_key or "").strip():
        raise ValueError(
            f"LLM {role} backend is enabled but API key is empty. "
            "Fill `api_key` in the config file or set the configured environment variable."
        )
    return OpenAICompatibleLLMClient(config)


if __name__ == "__main__":
    main()
