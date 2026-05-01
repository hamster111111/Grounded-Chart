from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from grounded_chart.llm import LLMClient, LLMCompletionTrace


@dataclass(frozen=True)
class ChartRenderingProtocol:
    chart_type: str
    protocol_id: str
    version: str = "v1"
    rendering_rules: tuple[str, ...] = ()
    required_artifact_columns: tuple[str, ...] = ()
    plotting_primitives: tuple[dict[str, Any], ...] = ()
    forbidden_shortcuts: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    source: str = "deterministic_fallback"
    llm_trace: LLMCompletionTrace | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rendering_rules"] = list(self.rendering_rules)
        payload["required_artifact_columns"] = list(self.required_artifact_columns)
        payload["plotting_primitives"] = [dict(item) for item in self.plotting_primitives]
        payload["forbidden_shortcuts"] = list(self.forbidden_shortcuts)
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
    """Generate chart-type rendering protocols.

    The default path is deterministic so benchmark runs are reproducible. An
    LLM client can be added later for unknown chart types, with the same schema.
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def build_protocol(self, *, chart_type: str, context: dict[str, Any] | None = None) -> ChartRenderingProtocol:
        normalized = str(chart_type or "").strip().lower()
        if normalized == "waterfall":
            return waterfall_protocol()
        if normalized == "area":
            return area_protocol(context or {})
        if self.client is not None:
            return self._build_llm_protocol(chart_type=normalized, context=context or {})
        return generic_protocol(normalized)

    def _build_llm_protocol(self, *, chart_type: str, context: dict[str, Any]) -> ChartRenderingProtocol:
        result = self.client.complete_json_with_trace(
            system_prompt=(
                "You generate chart rendering protocols. Return strict JSON only. "
                "A protocol must be executable and verifiable: no vague words such as generally/normally/appropriate."
            ),
            user_prompt=json.dumps(
                {
                    "chart_type": chart_type,
                    "context": context,
                    "required_keys": [
                        "chart_type",
                        "protocol_id",
                        "rendering_rules",
                        "required_artifact_columns",
                        "plotting_primitives",
                        "forbidden_shortcuts",
                        "assumptions",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            temperature=0.0,
            max_tokens=1800,
        )
        payload = result.payload
        return ChartRenderingProtocol(
            chart_type=str(payload.get("chart_type") or chart_type),
            protocol_id=str(payload.get("protocol_id") or f"{chart_type}_protocol"),
            rendering_rules=tuple(str(item) for item in payload.get("rendering_rules") or ()),
            required_artifact_columns=tuple(str(item) for item in payload.get("required_artifact_columns") or ()),
            plotting_primitives=tuple(dict(item) for item in payload.get("plotting_primitives") or () if isinstance(item, dict)),
            forbidden_shortcuts=tuple(str(item) for item in payload.get("forbidden_shortcuts") or ()),
            assumptions=tuple(str(item) for item in payload.get("assumptions") or ()),
            source="llm",
            llm_trace=result.trace,
        )


def waterfall_protocol() -> ChartRenderingProtocol:
    return ChartRenderingProtocol(
        chart_type="waterfall",
        protocol_id="waterfall_grouped_series_v1",
        rendering_rules=(
            "Each series has an independent cumulative path.",
            "The initial bar starts at y=0 and ends at the initial source value.",
            "Each delta bar starts at the previous cumulative value for the same series.",
            "A positive delta extends upward; a negative delta extends downward.",
            "The total bar starts at y=0 and ends at the final cumulative total.",
            "The color_role column uses exactly these semantic values: increase, decrease, total.",
            "Color mapping must branch on color_role values increase/decrease/total, not positive/negative.",
            "Connector lines connect consecutive cumulative tops for the same series.",
            "Grouped waterfall renders series side-by-side with stable x offsets.",
        ),
        required_artifact_columns=(
            "Year",
            "series",
            "x_index",
            "x_offset",
            "source_value",
            "bar_bottom",
            "bar_height",
            "bar_top",
            "role",
            "color_role",
            "connector_y_start",
            "connector_y_end",
        ),
        plotting_primitives=(
            {
                "primitive": "bar",
                "required_kwargs": ["x", "height", "bottom", "color"],
                "x": "x_index + x_offset",
                "height": "bar_height",
                "bottom": "bar_bottom",
                "color": "map color_role: increase -> positive/up color, decrease -> negative/down color, total -> total color",
            },
            {
                "primitive": "plot",
                "role": "connector",
                "x": "connector_x_start to connector_x_end",
                "y": "connector_y_start to connector_y_end",
            },
        ),
        forbidden_shortcuts=(
            "Do not draw all waterfall values as ordinary bars from y=0.",
            "Do not apply adjacent-row differencing unless the source explicitly stores cumulative totals.",
            "Do not compute bar bottoms inside plotting code when a render table provides bar_bottom.",
        ),
        assumptions=(
            "The source values represent initial, delta, and final total rows according to the requirement roles.",
        ),
    )


def generic_protocol(chart_type: str) -> ChartRenderingProtocol:
    normalized = chart_type or "unknown"
    return ChartRenderingProtocol(
        chart_type=normalized,
        protocol_id=f"{normalized}_generic_protocol_v1",
        rendering_rules=("Use the prepared artifact columns without hidden recomputation.",),
        required_artifact_columns=("x", "y"),
        forbidden_shortcuts=("Do not fabricate data.",),
    )


def area_protocol(context: dict[str, Any]) -> ChartRenderingProtocol:
    modifiers = _area_modifiers_from_context(context)
    composition = str(modifiers.get("composition") or "independent")
    scale_policy = modifiers.get("scale_policy") if isinstance(modifiers.get("scale_policy"), dict) else {}
    required_columns = (
        "Year",
        "x_index",
        "Urban",
        "Rural",
        "Urban_area",
        "Rural_area",
        "Urban_fill_bottom",
        "Urban_fill_top",
        "Rural_fill_bottom",
        "Rural_fill_top",
        "composition_policy",
    )
    rules = [
        "Area layers must preserve each source series before any optional cumulative geometry is created.",
        "The composition_policy column determines whether areas are independent, overlapping, or additive_stack.",
    ]
    forbidden = ["Do not choose stackplot or cumulative fill geometry without checking composition_policy."]
    primitives: list[dict[str, Any]] = [
        {
            "primitive": "fill_between",
            "x": "x_index or shared year coordinate",
            "y1": "Urban_fill_bottom",
            "y2": "Urban_fill_top",
            "alpha": modifiers.get("alpha", 0.35),
        },
        {
            "primitive": "fill_between",
            "x": "x_index or shared year coordinate",
            "y1": "Rural_fill_bottom",
            "y2": "Rural_fill_top",
            "alpha": modifiers.get("alpha", 0.35),
        },
    ]
    if composition == "overlap":
        rules.append("Overlapping area composition draws each series as an independent translucent filled area, not as Urban + Rural.")
        forbidden.append("Do not use Urban + Rural as Rural_fill_top when composition_policy is overlap.")
    elif composition == "additive_stack":
        rules.append("Additive stack composition draws later series from previous cumulative tops.")
    if scale_policy.get("type") == "explicit_range":
        rules.append(f"Use explicit axis range {scale_policy.get('min')} to {scale_policy.get('max')} for the bound axis.")
    return ChartRenderingProtocol(
        chart_type="area",
        protocol_id=f"area_{composition}_protocol_v1",
        rendering_rules=tuple(rules),
        required_artifact_columns=required_columns,
        plotting_primitives=tuple(primitives),
        forbidden_shortcuts=tuple(forbidden),
        assumptions=("Composition and scale are resolved from semantic modifiers in the construction plan.",),
    )


def validate_protocol(protocol: ChartRenderingProtocol) -> ProtocolValidationReport:
    issues: list[ProtocolValidationIssue] = []
    if not protocol.chart_type:
        issues.append(ProtocolValidationIssue(code="missing_chart_type", message="Protocol has no chart_type."))
    if not protocol.required_artifact_columns:
        issues.append(
            ProtocolValidationIssue(
                code="missing_required_artifact_columns",
                message="Protocol must define required artifact columns.",
            )
        )
    if protocol.chart_type == "waterfall":
        required = {"bar_bottom", "bar_height", "bar_top", "role", "series"}
        missing = sorted(required - set(protocol.required_artifact_columns))
        if missing:
            issues.append(
                ProtocolValidationIssue(
                    code="waterfall_protocol_missing_render_columns",
                    message=f"Waterfall protocol is missing render columns: {missing}.",
                )
            )
    if protocol.chart_type == "area":
        required = {"Urban_fill_bottom", "Urban_fill_top", "Rural_fill_bottom", "Rural_fill_top", "composition_policy"}
        missing = sorted(required - set(protocol.required_artifact_columns))
        if missing:
            issues.append(
                ProtocolValidationIssue(
                    code="area_protocol_missing_render_columns",
                    message=f"Area protocol is missing render columns: {missing}.",
                )
            )
    return ProtocolValidationReport(ok=not any(issue.severity == "error" for issue in issues), issues=tuple(issues))


def _area_modifiers_from_context(context: dict[str, Any]) -> dict[str, Any]:
    plan = context.get("construction_plan") if isinstance(context.get("construction_plan"), dict) else {}
    for panel in list(plan.get("panels") or []):
        if not isinstance(panel, dict):
            continue
        for layer in list(panel.get("layers") or []):
            if isinstance(layer, dict) and str(layer.get("chart_type") or "").lower() == "area":
                modifiers = layer.get("semantic_modifiers")
                if isinstance(modifiers, dict):
                    return modifiers
    return {}
