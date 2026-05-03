from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from grounded_chart.runtime.llm import OpenAICompatibleConfig


@dataclass(frozen=True)
class AblationRunConfig:
    bench_name: str = "repair_loop_bench"
    output_name: str | None = None
    parser_backend: str = "heuristic"
    repair_backend: str = "rule"
    repair_tier_mode: str = "rule_only"
    repair_policy_mode: str = "strict"
    enable_repair_loop: bool = True
    max_repair_rounds: int = 2
    parser_provider: OpenAICompatibleConfig | None = None
    repair_provider: OpenAICompatibleConfig | None = None
    codegen_provider: OpenAICompatibleConfig | None = None
    layout_provider: OpenAICompatibleConfig | None = None
    plan_provider: OpenAICompatibleConfig | None = None


def load_ablation_run_config(path: str | Path) -> AblationRunConfig:
    config_path = Path(path)
    data = _load_mapping(config_path)
    run = data.get("run", {})
    llm = data.get("llm", {})

    default_provider = _provider_from_mapping(llm.get("default", {}))
    parser_provider = _merge_provider(default_provider, _provider_from_mapping(llm.get("parser", {})))
    repair_provider = _merge_provider(default_provider, _provider_from_mapping(llm.get("repair", {})))
    codegen_provider = _merge_provider(default_provider, _provider_from_mapping(llm.get("codegen", {})))
    layout_provider = _merge_provider(default_provider, _provider_from_mapping(llm.get("layout", {})))
    plan_provider = _merge_provider(default_provider, _provider_from_mapping(llm.get("plan", {})))

    parser_backend = _normalized_backend(run.get("parser_backend", "heuristic"), allowed={"heuristic", "llm"})
    repair_backend = _normalized_backend(run.get("repair_backend", "rule"), allowed={"rule", "llm", "tiered"})
    repair_tier_mode = _normalized_backend(
        run.get("repair_tier_mode", "rule_only"),
        allowed={"rule_only", "hybrid", "llm_only"},
    )
    repair_policy_mode = _normalized_backend(
        run.get("repair_policy_mode", "strict"),
        allowed={"strict", "exploratory"},
    )

    if parser_backend == "llm" and parser_provider is None:
        raise ValueError("Config enables LLM parser but does not define [llm.default] or [llm.parser].")
    if repair_backend in {"llm", "tiered"} and repair_provider is None:
        raise ValueError("Config enables LLM repairer but does not define [llm.default] or [llm.repair].")

    return AblationRunConfig(
        bench_name=str(run.get("bench", "repair_loop_bench")),
        output_name=str(run["output_name"]) if run.get("output_name") else None,
        parser_backend=parser_backend,
        repair_backend=repair_backend,
        repair_tier_mode=repair_tier_mode,
        repair_policy_mode=repair_policy_mode,
        enable_repair_loop=bool(run.get("enable_repair_loop", True)),
        max_repair_rounds=int(run.get("max_repair_rounds", 2)),
        parser_provider=_resolve_provider_secret(parser_provider),
        repair_provider=_resolve_provider_secret(repair_provider),
        codegen_provider=_resolve_provider_secret(codegen_provider),
        layout_provider=_resolve_provider_secret(layout_provider),
        plan_provider=_resolve_provider_secret(plan_provider),
    )


def load_ablation_run_config_from_env() -> AblationRunConfig:
    parser_backend = _normalized_backend(os.environ.get("GCHART_PARSER_BACKEND", "heuristic"), allowed={"heuristic", "llm"})
    repair_backend = _normalized_backend(os.environ.get("GCHART_REPAIR_BACKEND", "rule"), allowed={"rule", "llm", "tiered"})
    repair_tier_mode = _normalized_backend(
        os.environ.get("GCHART_REPAIR_TIER_MODE", "rule_only"),
        allowed={"rule_only", "hybrid", "llm_only"},
    )
    repair_policy_mode = _normalized_backend(
        os.environ.get("GCHART_REPAIR_POLICY_MODE", "strict"),
        allowed={"strict", "exploratory"},
    )
    default_provider = _provider_from_env("GCHART") if "GCHART_MODEL" in os.environ else None
    parser_provider = _provider_from_env("GCHART_PARSER") if "GCHART_PARSER_MODEL" in os.environ else default_provider
    repair_provider = _provider_from_env("GCHART_REPAIR") if "GCHART_REPAIR_MODEL" in os.environ else default_provider
    codegen_provider = _provider_from_env("GCHART_CODEGEN") if "GCHART_CODEGEN_MODEL" in os.environ else default_provider
    layout_provider = _provider_from_env("GCHART_LAYOUT") if "GCHART_LAYOUT_MODEL" in os.environ else default_provider
    plan_provider = _provider_from_env("GCHART_PLAN") if "GCHART_PLAN_MODEL" in os.environ else default_provider
    return AblationRunConfig(
        bench_name=os.environ.get("GCHART_BENCH", "repair_loop_bench"),
        output_name=os.environ.get("GCHART_OUTPUT_NAME"),
        parser_backend=parser_backend,
        repair_backend=repair_backend,
        repair_tier_mode=repair_tier_mode,
        repair_policy_mode=repair_policy_mode,
        enable_repair_loop=os.environ.get("GCHART_ENABLE_REPAIR_LOOP", "1").strip() not in {"0", "false", "False"},
        max_repair_rounds=int(os.environ.get("GCHART_MAX_REPAIR_ROUNDS", "2")),
        parser_provider=parser_provider,
        repair_provider=repair_provider,
        codegen_provider=codegen_provider,
        layout_provider=layout_provider,
        plan_provider=plan_provider,
    )


def _provider_from_mapping(raw: dict[str, Any]) -> OpenAICompatibleConfig | None:
    if not isinstance(raw, dict) or not raw:
        return None
    provider = str(raw.get("provider", "openai_compatible"))
    if provider != "openai_compatible":
        raise ValueError(f"Unsupported provider type in config: {provider}")
    model = str(raw.get("model", "")).strip()
    if not model:
        return None
    api_key = str(raw.get("api_key", "") or "")
    api_key_env = str(raw.get("api_key_env", "") or "")
    resolved_api_key = api_key or (os.environ.get(api_key_env, "") if api_key_env else "")
    return OpenAICompatibleConfig(
        model=model,
        api_key=resolved_api_key,
        base_url=str(raw.get("base_url", "") or "") or None,
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw["max_tokens"]) if raw.get("max_tokens") is not None else None,
        no_proxy=bool(raw.get("no_proxy", False)),
    )


def _provider_from_env(prefix: str) -> OpenAICompatibleConfig | None:
    model = os.environ.get(f"{prefix}_MODEL", "").strip()
    if not model:
        return None
    return OpenAICompatibleConfig.from_env(prefix=prefix)


def _merge_provider(
    base: OpenAICompatibleConfig | None,
    override: OpenAICompatibleConfig | None,
) -> OpenAICompatibleConfig | None:
    if base is None:
        return override
    if override is None:
        return base
    same_endpoint = override.base_url is None or override.base_url == base.base_url
    return OpenAICompatibleConfig(
        model=override.model or base.model,
        api_key=override.api_key or (base.api_key if same_endpoint else ""),
        base_url=override.base_url or base.base_url,
        temperature=override.temperature,
        max_tokens=override.max_tokens if override.max_tokens is not None else base.max_tokens,
        no_proxy=override.no_proxy or base.no_proxy,
    )


def _resolve_provider_secret(config: OpenAICompatibleConfig | None) -> OpenAICompatibleConfig | None:
    if config is None:
        return None
    return replace(config, api_key=str(config.api_key or "").strip())


def _normalized_backend(value: object, *, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported backend {value!r}; allowed values: {sorted(allowed)}")
    return normalized


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".toml":
        data = tomllib.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML config requires PyYAML to be installed.") from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml, .yml, or .toml.")
    if not isinstance(data, dict):
        raise ValueError("Config file must deserialize to a mapping/object at the top level.")
    return data
