from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from grounded_chart.llm import LLMClient, LLMCompletionTrace


@dataclass(frozen=True)
class VisualChannelContract:
    """Minimal fidelity contract for one visual channel.

    This is not a complete style specification. Hard contracts only identify
    semantic bindings that must not be changed by the executor.
    """

    layer_id: str
    channel: str
    semantic_dimension: str
    field: str | None = None
    strength: str = "hard"
    contract_tier: str = "hard_fidelity"
    domain_source: str = "artifact_schema"
    executor_freedom: str = "choose_style"
    constraints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constraints"] = list(self.constraints)
        return payload


@dataclass(frozen=True)
class ChartRenderingProtocol:
    chart_type: str
    protocol_id: str
    version: str = "v1"
    contract_tiers: dict[str, str] = field(default_factory=dict)
    semantic_interpretation: dict[str, Any] = field(default_factory=dict)
    data_artifacts: tuple[dict[str, Any], ...] = ()
    geometry_rules: tuple[str, ...] = ()
    visual_channel_policy: dict[str, Any] = field(default_factory=dict)
    visual_channel_contracts: tuple[VisualChannelContract, ...] = ()
    rendering_rules: tuple[str, ...] = ()
    required_artifact_columns: tuple[str, ...] = ()
    plotting_primitives: tuple[dict[str, Any], ...] = ()
    forbidden_shortcuts: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    source: str = "deterministic_fallback"
    llm_trace: LLMCompletionTrace | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_tiers"] = dict(self.contract_tiers)
        payload["semantic_interpretation"] = dict(self.semantic_interpretation)
        payload["data_artifacts"] = [dict(item) for item in self.data_artifacts]
        payload["geometry_rules"] = list(self.geometry_rules)
        payload["visual_channel_policy"] = dict(self.visual_channel_policy)
        payload["visual_channel_contracts"] = [item.to_dict() for item in self.visual_channel_contracts]
        payload["rendering_rules"] = list(self.rendering_rules)
        payload["required_artifact_columns"] = list(self.required_artifact_columns)
        payload["plotting_primitives"] = [dict(item) for item in self.plotting_primitives]
        payload["forbidden_shortcuts"] = list(self.forbidden_shortcuts)
        payload["uncertainties"] = list(self.uncertainties)
        payload["assumptions"] = list(self.assumptions)
        if self.llm_trace is not None:
            payload["llm_trace"] = {
                "provider": self.llm_trace.provider,
                "model": self.llm_trace.model,
                "base_url": self.llm_trace.base_url,
                "temperature": self.llm_trace.temperature,
                "max_tokens": self.llm_trace.max_tokens,
                "raw_text_preview": self.llm_trace.raw_text[:1200],
                "usage": asdict(self.llm_trace.usage) if self.llm_trace.usage is not None else None,
            }
        return payload


@dataclass(frozen=True)
class ProtocolValidationIssue:
    code: str
    message: str
    severity: str = "error"
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtocolValidationReport:
    ok: bool
    issues: tuple[ProtocolValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": [issue.to_dict() for issue in self.issues]}


class ChartProtocolAgent:
    """Generate case-specific chart rendering protocols.

    The protocol layer defines executable constraints for the current request.
    It should not become a hard-coded chart recipe library. With an LLM client
    configured, every chart type is interpreted from the task context. The
    deterministic fallback is intentionally generic and derived from the
    construction plan/artifact schema only.
    """

    def __init__(self, client: LLMClient | None = None, *, max_tokens: int | None = None) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def build_protocol(self, *, chart_type: str, context: dict[str, Any] | None = None) -> ChartRenderingProtocol:
        normalized = str(chart_type or "").strip().lower()
        if self.client is not None:
            return self._build_llm_protocol(chart_type=normalized, context=context or {})
        return generic_protocol(normalized, context=context or {})

    def _build_llm_protocol(self, *, chart_type: str, context: dict[str, Any]) -> ChartRenderingProtocol:
        compact_context = _compact_protocol_context(context, chart_type)
        result = self.client.complete_json_with_trace(
            system_prompt=(
                "You are ChartProtocolAgent for an evidence-grounded chart generation pipeline. "
                "Return strict JSON only. Generate a case-specific rendering protocol from the request, source schema, "
                "construction plan, prepared artifacts, and feedback. Do not use a memorized universal recipe when the "
                "task context implies a different semantic interpretation. Separate data/geometry roles from visual "
                "channel roles. For example, a terminal total role in a waterfall does not automatically mean a third "
                "fill color when series identity is the primary visual channel. Use deterministic artifacts for "
                "computed values and state uncertainties instead of inventing unsupported semantics."
            ),
            user_prompt=json.dumps(
                {
                    "chart_type": chart_type,
                    "context": compact_context,
                    "required_keys": [
                        "chart_type",
                        "protocol_id",
                        "contract_tiers",
                        "semantic_interpretation",
                        "data_artifacts",
                        "geometry_rules",
                        "visual_channel_policy",
                        "visual_channel_contracts",
                        "rendering_rules",
                        "required_artifact_columns",
                        "plotting_primitives",
                        "forbidden_shortcuts",
                        "uncertainties",
                        "assumptions",
                    ],
                    "field_guidance": {
                        "semantic_interpretation": "State what this chart type means in this specific task.",
                        "data_artifacts": "List prepared artifacts/columns the executor must consume when available. Use only columns present in prepared_artifacts[].schema.columns.",
                        "geometry_rules": "State geometry rules without assigning visual colors unless needed.",
                        "visual_channel_policy": "State which fields control fill color, edge, hatch, alpha, line, marker, legend. Keep this minimal.",
                        "visual_channel_contracts": (
                            "List only fidelity-critical channel bindings. Each item should include layer_id, channel, "
                            "semantic_dimension, field, strength={hard|soft|free}, contract_tier={hard_fidelity|soft_guidance|free_design}, "
                            "domain_source, executor_freedom, constraints. Do not assign concrete color values unless explicitly requested."
                        ),
                        "rendering_rules": "Use concrete executable rules; avoid vague style advice.",
                        "forbidden_shortcuts": "Forbid likely semantic bypasses for this case.",
                        "uncertainties": "List unresolved interpretation risks that should not be hidden.",
                    },
                    "strict_rules": [
                        "Classify every protocol commitment into hard_fidelity, soft_guidance, or free_design. Hard fidelity covers source-grounded data, deterministic geometry, explicit requirements, and semantic channel bindings. Soft guidance covers readability and aesthetics. Free design covers palette, fonts, exact spacing, and API choices unless explicitly requested.",
                        "Do not invent artifact columns. If a prepared artifact schema lists wide per-series columns such as *_fill_top/*_fill_bottom, do not rewrite it as series/value unless you create and document a derived intermediate.",
                        "If a chart needs long-form data but the prepared artifact is wide, say the executor may locally melt it into a new intermediate file and must document that derived artifact.",
                        "When the plan and artifact schema conflict, trust the artifact schema for executable column names and state the conflict in uncertainties.",
                        "Hard visual-channel contracts constrain semantic binding only; leave palette, fonts, spacing, and aesthetics to Executor unless requested.",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            temperature=0.0,
            max_tokens=self.max_tokens or 4096,
        )
        payload = result.payload
        fallback = generic_protocol(chart_type, context=context)
        required_columns = tuple(str(item) for item in payload.get("required_artifact_columns") or ())
        data_artifacts = _protocol_artifact_items(payload.get("data_artifacts"))
        if not required_columns:
            required_columns = _required_columns_from_data_artifacts(data_artifacts)
        visual_channel_contracts = _visual_channel_contract_items(payload.get("visual_channel_contracts")) or fallback.visual_channel_contracts
        protocol = ChartRenderingProtocol(
            chart_type=str(payload.get("chart_type") or chart_type),
            protocol_id=str(payload.get("protocol_id") or f"{chart_type}_protocol"),
            contract_tiers=_protocol_contract_tiers(payload.get("contract_tiers")),
            semantic_interpretation=_protocol_mapping_or_summary(payload.get("semantic_interpretation")),
            data_artifacts=data_artifacts or fallback.data_artifacts,
            geometry_rules=tuple(str(item) for item in payload.get("geometry_rules") or ()) or fallback.geometry_rules,
            visual_channel_policy=dict(payload.get("visual_channel_policy") or {})
            if isinstance(payload.get("visual_channel_policy"), dict)
            else fallback.visual_channel_policy,
            visual_channel_contracts=visual_channel_contracts,
            rendering_rules=tuple(str(item) for item in payload.get("rendering_rules") or ()) or fallback.rendering_rules,
            required_artifact_columns=required_columns or fallback.required_artifact_columns,
            plotting_primitives=tuple(dict(item) for item in payload.get("plotting_primitives") or () if isinstance(item, dict)),
            forbidden_shortcuts=tuple(str(item) for item in payload.get("forbidden_shortcuts") or ()) or fallback.forbidden_shortcuts,
            uncertainties=tuple(str(item) for item in payload.get("uncertainties") or ()),
            assumptions=tuple(
                dict.fromkeys(
                    (
                        *tuple(str(item) for item in payload.get("assumptions") or ()),
                        *(
                            ("LLM protocol was completed with schema-derived fallback fields because some required protocol fields were absent.",)
                            if _protocol_needs_fallback(payload)
                            else ()
                        ),
                    )
                )
            ),
            source="llm",
            llm_trace=result.trace,
        )
        return protocol


def generic_protocol(chart_type: str, context: dict[str, Any] | None = None) -> ChartRenderingProtocol:
    normalized = chart_type or "unknown"
    context = context or {}
    artifact_columns = _artifact_columns_for_chart_type(context, normalized)
    visual_policy = _visual_channel_policy_from_plan(context, normalized)
    rules = [
        "Interpret chart semantics from the construction plan and source schema for this case.",
        "Use prepared artifacts when they are available; do not silently replace them with hidden recomputation.",
        "Keep data roles, geometry roles, and visual-channel roles separate.",
    ]
    forbidden = ["Do not fabricate data or semantic categories not supported by the request/source plan."]
    if artifact_columns:
        rules.append("Prepared artifact columns define the executable data/geometry contract for this chart type.")
    else:
        artifact_columns = tuple(_fallback_required_columns(context, normalized))
    return ChartRenderingProtocol(
        chart_type=normalized,
        protocol_id=f"{normalized}_case_protocol_v1",
        contract_tiers=_default_contract_tiers(),
        semantic_interpretation={
            "source": "construction_plan_context",
            "chart_type": normalized,
            "note": "Fallback protocol is schema-derived, not a hard-coded chart recipe.",
        },
        data_artifacts=tuple(_tier_data_artifacts(_data_artifacts_for_chart_type(context, normalized))),
        visual_channel_policy=visual_policy,
        visual_channel_contracts=tuple(_visual_channel_contracts_from_plan(context, normalized, visual_policy)),
        rendering_rules=tuple(rules),
        required_artifact_columns=tuple(artifact_columns),
        plotting_primitives=tuple(_plotting_primitives_from_plan(context, normalized)),
        forbidden_shortcuts=tuple(forbidden),
        assumptions=("No LLM protocol client was configured; protocol is a minimal case-derived fallback.",),
    )


def validate_protocol(protocol: ChartRenderingProtocol) -> ProtocolValidationReport:
    issues: list[ProtocolValidationIssue] = []
    if not protocol.chart_type:
        issues.append(ProtocolValidationIssue(code="missing_chart_type", message="Protocol has no chart_type."))
    if not protocol.rendering_rules and not protocol.geometry_rules:
        issues.append(
            ProtocolValidationIssue(
                code="missing_rendering_or_geometry_rules",
                message="Protocol must define rendering_rules or geometry_rules.",
            )
        )
    if not protocol.required_artifact_columns and not protocol.data_artifacts:
        issues.append(
            ProtocolValidationIssue(
                code="missing_required_artifact_columns",
                message="Protocol must define required artifact columns or data artifact contracts.",
            )
        )
    if not protocol.visual_channel_policy:
        issues.append(
            ProtocolValidationIssue(
                code="missing_visual_channel_policy",
                message="Protocol should state how visual channels are assigned or intentionally delegated.",
                severity="warning",
            )
        )
    for contract in protocol.visual_channel_contracts:
        if contract.strength == "hard" and contract.contract_tier != "hard_fidelity":
            issues.append(
                ProtocolValidationIssue(
                    code="hard_contract_tier_mismatch",
                    message="Contracts with strength=hard must be classified as hard_fidelity.",
                    severity="warning",
                    evidence=str(contract.to_dict()),
                )
            )
        if contract.strength == "free" and contract.contract_tier == "hard_fidelity":
            issues.append(
                ProtocolValidationIssue(
                    code="free_contract_tier_mismatch",
                    message="Contracts with strength=free cannot be classified as hard_fidelity.",
                    severity="warning",
                    evidence=str(contract.to_dict()),
                )
            )
        if contract.contract_tier == "hard_fidelity" and (not contract.layer_id or not contract.channel or not contract.semantic_dimension):
            issues.append(
                ProtocolValidationIssue(
                    code="invalid_hard_visual_channel_contract",
                    message="Hard visual-channel contracts must include layer_id, channel, and semantic_dimension.",
                    evidence=str(contract.to_dict()),
                )
            )
    return ProtocolValidationReport(ok=not any(issue.severity == "error" for issue in issues), issues=tuple(issues))


def _required_columns_from_data_artifacts(data_artifacts: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    columns: list[str] = []
    for artifact in data_artifacts:
        for column in list(artifact.get("columns") or artifact.get("required_columns") or []):
            value = str(column).strip()
            if value and value not in columns:
                columns.append(value)
    return tuple(columns)


def _protocol_artifact_items(raw: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(raw, dict):
        values = []
        for key, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", key)
                values.append(item)
        return tuple(_tier_data_artifacts(values))
    return tuple(_tier_data_artifacts(tuple(dict(item) for item in list(raw or []) if isinstance(item, dict))))


def _visual_channel_contract_items(raw: Any) -> tuple[VisualChannelContract, ...]:
    values = []
    if isinstance(raw, dict):
        raw_items = raw.values()
    else:
        raw_items = list(raw or [])
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        strength = str(item.get("strength") or "hard").lower()
        if strength not in {"hard", "soft", "free"}:
            strength = "hard"
        contract_tier = _normalize_contract_tier(item.get("contract_tier") or item.get("tier"), default=_contract_tier_from_strength(strength))
        values.append(
            VisualChannelContract(
                layer_id=str(item.get("layer_id") or ""),
                channel=str(item.get("channel") or ""),
                semantic_dimension=str(item.get("semantic_dimension") or ""),
                field=str(item.get("field")) if item.get("field") is not None else None,
                strength=strength,
                contract_tier=contract_tier,
                domain_source=str(item.get("domain_source") or "artifact_schema"),
                executor_freedom=str(item.get("executor_freedom") or "choose_style"),
                constraints=tuple(str(value) for value in list(item.get("constraints") or []) if str(value).strip()),
            )
        )
    return tuple(values)


def _normalize_contract_tier(raw: Any, *, default: str = "hard_fidelity") -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "hard": "hard_fidelity",
        "hard_fidelity": "hard_fidelity",
        "fidelity": "hard_fidelity",
        "required": "hard_fidelity",
        "soft": "soft_guidance",
        "soft_guidance": "soft_guidance",
        "guidance": "soft_guidance",
        "warning": "soft_guidance",
        "free": "free_design",
        "free_design": "free_design",
        "design_space": "free_design",
        "delegated": "free_design",
    }
    return aliases.get(value, default)


def _contract_tier_from_strength(strength: str) -> str:
    if strength == "soft":
        return "soft_guidance"
    if strength == "free":
        return "free_design"
    return "hard_fidelity"


def _tier_data_artifacts(raw: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    artifacts: list[dict[str, Any]] = []
    for item in raw:
        artifact = dict(item)
        artifact.setdefault("contract_tier", "hard_fidelity")
        artifact.setdefault("contract_reason", "Prepared data/geometry artifacts are hard fidelity inputs because they preserve source-grounded values and deterministic transformations.")
        artifacts.append(artifact)
    return tuple(artifacts)


def _protocol_mapping_or_summary(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw or "").strip()
    return {"summary": text} if text else {}


def _protocol_needs_fallback(payload: dict[str, Any]) -> bool:
    return not (
        payload.get("data_artifacts")
        and payload.get("required_artifact_columns")
        and payload.get("rendering_rules")
        and payload.get("visual_channel_policy")
    )


def _default_contract_tiers() -> dict[str, str]:
    return {
        "hard_fidelity": (
            "Blocking requirements grounded in source data, deterministic artifacts, explicit user constraints, "
            "chart type semantics, and semantic visual-channel bindings."
        ),
        "soft_guidance": (
            "Non-blocking readability, layout, and aesthetic preferences used by visual feedback/replanning."
        ),
        "free_design": (
            "Executor-owned design choices such as concrete palette, fonts, line widths, exact spacing, and plotting API "
            "unless explicitly requested."
        ),
    }


def _protocol_contract_tiers(raw: Any) -> dict[str, str]:
    defaults = _default_contract_tiers()
    if not isinstance(raw, dict):
        return defaults
    result = dict(defaults)
    for key, value in raw.items():
        tier = _normalize_contract_tier(key, default=str(key))
        if tier not in defaults:
            continue
        if isinstance(value, str):
            result[tier] = value
        elif isinstance(value, dict):
            result[tier] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return result


def _compact_protocol_context(context: dict[str, Any], chart_type: str) -> dict[str, Any]:
    return {
        "query": context.get("query"),
        "round_id": context.get("round_id"),
        "construction_plan": _compact_construction_plan(context.get("construction_plan"), chart_type),
        "source_data_plan": _compact_source_plan(context.get("source_data_plan")),
        "source_data_execution": _compact_source_execution(context.get("source_data_execution")),
        "prepared_artifacts": _compact_prepared_artifacts(context.get("prepared_artifacts")),
    }


def _compact_construction_plan(raw: Any, chart_type: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    panels = []
    for panel in list(raw.get("panels") or []):
        if not isinstance(panel, dict):
            continue
        layers = [
            {
                "layer_id": layer.get("layer_id"),
                "chart_type": layer.get("chart_type"),
                "role": layer.get("role"),
                "data_source": layer.get("data_source"),
                "x": layer.get("x"),
                "y": layer.get("y"),
                "axis": layer.get("axis"),
                "encoding": layer.get("encoding"),
                "semantic_modifiers": layer.get("semantic_modifiers"),
                "visual_channel_plan": layer.get("visual_channel_plan"),
                "components": layer.get("components"),
            }
            for layer in list(panel.get("layers") or [])
            if isinstance(layer, dict) and str(layer.get("chart_type") or "").lower() == chart_type
        ]
        if layers or panel.get("role") == "inset_pie_chart":
            panels.append(
                {
                    "panel_id": panel.get("panel_id"),
                    "role": panel.get("role"),
                    "anchor": panel.get("anchor"),
                    "placement_policy": panel.get("placement_policy"),
                    "avoid_occlusion": panel.get("avoid_occlusion"),
                    "layers": layers,
                }
            )
    return {
        "plan_type": raw.get("plan_type"),
        "layout_strategy": raw.get("layout_strategy"),
        "figure_size": raw.get("figure_size"),
        "panels": panels,
        "global_elements": raw.get("global_elements"),
        "constraints": raw.get("constraints"),
    }


def _compact_source_plan(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "files": [
            {
                "name": item.get("name"),
                "columns": item.get("columns"),
                "row_count_preview": item.get("row_count_preview"),
                "preview_rows": list(item.get("preview_rows") or [])[:3],
            }
            for item in list(raw.get("files") or [])
            if isinstance(item, dict)
        ],
        "schema_constraints": raw.get("schema_constraints"),
        "global_constraints": raw.get("global_constraints"),
    }


def _compact_source_execution(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "loaded_tables": [
            {
                "name": item.get("name"),
                "columns": item.get("columns"),
                "row_count_loaded": item.get("row_count_loaded"),
                "preview_rows": list(item.get("rows") or [])[:3],
                "truncated": item.get("truncated"),
            }
            for item in list(raw.get("loaded_tables") or [])
            if isinstance(item, dict)
        ]
    }


def _compact_prepared_artifacts(raw: Any) -> list[dict[str, Any]]:
    artifacts = []
    for artifact in list(raw or []):
        if not isinstance(artifact, dict):
            continue
        schema = artifact.get("schema") if isinstance(artifact.get("schema"), dict) else {}
        artifacts.append(
            {
                "name": artifact.get("name"),
                "relative_path": artifact.get("relative_path"),
                "purpose": artifact.get("purpose"),
                "schema": {
                    "columns": list(schema.get("columns") or artifact.get("columns") or []),
                    "row_count": schema.get("row_count"),
                },
            }
        )
    return artifacts


def _data_artifacts_for_chart_type(context: dict[str, Any], chart_type: str) -> tuple[dict[str, Any], ...]:
    artifacts = []
    for artifact in _artifact_workspace_artifacts(context):
        name = str(artifact.get("name") or "")
        if _artifact_matches_chart_type(artifact, chart_type):
            schema = artifact.get("schema") if isinstance(artifact.get("schema"), dict) else {}
            schema_columns = [str(item) for item in list(schema.get("columns") or artifact.get("columns") or [])]
            artifacts.append(
                {
                    "name": name,
                    "relative_path": artifact.get("relative_path"),
                    "purpose": artifact.get("purpose"),
                    "artifact_id": artifact.get("artifact_id"),
                    "artifact_role": artifact.get("artifact_role"),
                    "layer_id": artifact.get("layer_id"),
                    "source_table": artifact.get("source_table"),
                    "columns": schema_columns or _columns_from_artifact_name(name),
                }
            )
    return tuple(artifacts)


def _artifact_columns_for_chart_type(context: dict[str, Any], chart_type: str) -> tuple[str, ...]:
    columns: list[str] = []
    for artifact in _data_artifacts_for_chart_type(context, chart_type):
        for column in list(artifact.get("columns") or []):
            if column not in columns:
                columns.append(column)
    return tuple(columns)


def _columns_from_artifact_name(name: str) -> list[str]:
    lower = str(name or "").lower()
    if "waterfall_geometry" in lower or "waterfall_render_table" in lower:
        return [
            "x_value",
            "series",
            "x_index",
            "x_offset",
            "x_position",
            "source_value",
            "bar_bottom",
            "bar_height",
            "bar_top",
            "role",
            "change_role",
            "fill_color_role",
            "connector_y_start",
            "connector_y_end",
        ]
    if "source_values" in lower or "waterfall_values" in lower:
        return ["x_index", "x_value", "artifact_role"]
    if "area_fill_geometry" in lower or "area_values" in lower:
        return ["x_index", "x_value", "*_fill_bottom", "*_fill_top", "composition_policy"]
    if "categorical_values" in lower or "pie_values" in lower:
        return ["category", "raw_value", "Percentage"]
    return []


def _artifact_workspace_artifacts(context: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = [dict(item) for item in list(context.get("prepared_artifacts") or []) if isinstance(item, dict)]
    workspace = context.get("artifact_workspace") if isinstance(context.get("artifact_workspace"), dict) else {}
    artifacts.extend(dict(item) for item in list(workspace.get("artifacts") or []) if isinstance(item, dict))
    return artifacts


def _artifact_matches_chart_type(artifact: dict[str, Any], chart_type: str) -> bool:
    normalized = str(chart_type or "").lower()
    artifact_chart_type = str(artifact.get("chart_type") or "").lower()
    if artifact_chart_type:
        return artifact_chart_type == normalized
    name = str(artifact.get("name") or "")
    role = str(artifact.get("artifact_role") or "")
    lower = f"{name} {role}".lower()
    if normalized == "waterfall":
        return "waterfall" in lower
    if normalized == "area":
        return "area" in lower
    if normalized == "pie":
        return "pie" in lower or "categorical" in lower
    return normalized and normalized in lower


def _visual_channel_policy_from_plan(context: dict[str, Any], chart_type: str) -> dict[str, Any]:
    for layer in _layers_from_context(context):
        if str(layer.get("chart_type") or "").lower() == chart_type:
            plan = layer.get("visual_channel_plan")
            if isinstance(plan, dict) and plan:
                return plan
            return {
                "source": "construction_plan_layer",
                "layer_id": layer.get("layer_id"),
                "encoding": layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {},
                "note": "Plan did not provide a detailed visual_channel_plan; ExecutorAgent must keep labels/legend faithful to the layer encoding.",
            }
    return {"source": "generic_fallback", "note": "No layer-specific visual channel plan found."}


def _visual_channel_contracts_from_plan(
    context: dict[str, Any],
    chart_type: str,
    visual_policy: dict[str, Any],
) -> list[VisualChannelContract]:
    contracts: list[VisualChannelContract] = []
    artifacts = _data_artifacts_for_chart_type(context, chart_type)
    artifact_columns = {column for artifact in artifacts for column in list(artifact.get("columns") or [])}
    for layer in _layers_from_context(context):
        if str(layer.get("chart_type") or "").lower() != chart_type:
            continue
        layer_id = str(layer.get("layer_id") or chart_type or "layer")
        policy = layer.get("visual_channel_plan") if isinstance(layer.get("visual_channel_plan"), dict) else visual_policy
        channel_allocation = policy.get("channel_allocation") if isinstance(policy.get("channel_allocation"), dict) else {}
        fill_field = _channel_field(channel_allocation.get("fill_color")) or _channel_field(policy.get("fill_color"))
        if not fill_field and "fill_color_role" in artifact_columns:
            fill_field = "fill_color_role"
        if not fill_field:
            encoding = layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {}
            series = encoding.get("series")
            if series:
                fill_field = "series"
        if fill_field:
            contracts.append(
                VisualChannelContract(
                    layer_id=layer_id,
                    channel="fill_color",
                    semantic_dimension=_semantic_dimension_for_field(fill_field),
                    field=fill_field,
                    strength="hard",
                    domain_source="artifact_schema" if fill_field in artifact_columns else "construction_plan",
                    executor_freedom="choose_palette",
                    constraints=(
                        "same field value must use the same visual encoding within the layer",
                        "do not introduce extra fill-color legend categories unless requested or supported by the contract field",
                    ),
                )
            )
        offset_field = _channel_field(channel_allocation.get("x_group_offset"))
        if offset_field:
            contracts.append(
                VisualChannelContract(
                    layer_id=layer_id,
                    channel="x_offset",
                    semantic_dimension=_semantic_dimension_for_field(offset_field),
                    field=offset_field,
                    strength="hard",
                    domain_source="artifact_schema" if offset_field in artifact_columns else "construction_plan",
                    executor_freedom="choose_group_spacing",
                    constraints=("same field value should keep a stable offset within each x group",),
                )
            )
        auxiliary = _channel_field(channel_allocation.get("auxiliary_change_cue"))
        if auxiliary:
            contracts.append(
                VisualChannelContract(
                    layer_id=layer_id,
                    channel="edge_style_or_hatch",
                    semantic_dimension=_semantic_dimension_for_field(auxiliary),
                    field=auxiliary,
                    strength="soft",
                    domain_source="artifact_schema" if auxiliary in artifact_columns else "construction_plan",
                    executor_freedom="may_choose_or_omit_if_readability_suffers",
                    constraints=("auxiliary cues must not override hard fill_color semantics",),
                )
            )
    return contracts


def _channel_field(raw: Any) -> str | None:
    if isinstance(raw, dict):
        value = raw.get("field")
        return str(value).strip() if value is not None and str(value).strip() else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _semantic_dimension_for_field(field: str) -> str:
    normalized = str(field or "").lower()
    if "series" in normalized or "fill_color_role" in normalized:
        return "series_identity"
    if "change" in normalized or "role" in normalized:
        return "geometry_or_change_role"
    if "category" in normalized or "label" in normalized or "group" in normalized:
        return "category_identity"
    return normalized or "semantic_field"


def _plotting_primitives_from_plan(context: dict[str, Any], chart_type: str) -> list[dict[str, Any]]:
    primitives = []
    for layer in _layers_from_context(context):
        if str(layer.get("chart_type") or "").lower() != chart_type:
            continue
        encoding = layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {}
        primitive = {"chart_type": chart_type, "layer_id": layer.get("layer_id")}
        if encoding:
            primitive["encoding"] = encoding
        primitives.append(primitive)
    return primitives


def _fallback_required_columns(context: dict[str, Any], chart_type: str) -> list[str]:
    for layer in _layers_from_context(context):
        if str(layer.get("chart_type") or "").lower() != chart_type:
            continue
        columns = []
        for key in ("x", "y"):
            value = layer.get(key)
            if isinstance(value, str) and value:
                columns.append(value)
            elif isinstance(value, list):
                columns.extend(str(item) for item in value if str(item).strip())
        if columns:
            return columns
    return ["x", "y"]


def _layers_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    plan = context.get("construction_plan") if isinstance(context.get("construction_plan"), dict) else {}
    layers: list[dict[str, Any]] = []
    for panel in list(plan.get("panels") or []):
        if not isinstance(panel, dict):
            continue
        for layer in list(panel.get("layers") or []):
            if isinstance(layer, dict):
                layers.append(layer)
    return layers
