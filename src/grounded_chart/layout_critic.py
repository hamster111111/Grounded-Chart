from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from grounded_chart.llm import LLMClient, LLMCompletionTrace, LLMUsage
from grounded_chart.plan_feedback import normalize_layout_plan_feedback


@dataclass(frozen=True)
class LayoutCritique:
    """Layout-only critique over a rendered chart and its construction plan.

    This intentionally excludes data correctness, chart-type correctness, and
    semantic value judgments. Those belong to requirement verification or the
    benchmark evaluator, not to layout replanning.
    """

    ok: bool
    failed_contracts: tuple[str, ...] = ()
    diagnosis: str = ""
    recommended_plan_updates: tuple[dict[str, Any], ...] = ()
    plan_feedback: tuple[dict[str, Any], ...] = ()
    confidence: float = 0.0
    llm_trace: LLMCompletionTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failed_contracts": list(self.failed_contracts),
            "diagnosis": self.diagnosis,
            "recommended_plan_updates": [_jsonable(item) for item in self.recommended_plan_updates],
            "plan_feedback": [_jsonable(item) for item in self.normalized_plan_feedback()],
            "confidence": self.confidence,
            "llm_trace": _llm_trace_to_dict(self.llm_trace),
            "metadata": _jsonable(self.metadata),
        }

    def normalized_plan_feedback(self) -> tuple[dict[str, Any], ...]:
        source_agent = str(
            self.metadata.get("critic_name")
            or self.metadata.get("agent_name")
            or "LayoutAgent"
        )
        return normalize_layout_plan_feedback(
            source_agent=source_agent,
            failed_contracts=self.failed_contracts,
            diagnosis=self.diagnosis,
            recommended_plan_updates=self.recommended_plan_updates,
            raw_plan_feedback=self.plan_feedback,
            confidence=self.confidence,
        )


class LayoutCriticAgent(Protocol):
    def critique(
        self,
        *,
        query: str,
        construction_plan: dict[str, Any],
        generated_code: str,
        render_result: Any,
        actual_figure: Any | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> LayoutCritique:
        """Return layout-only critique and bounded plan-update suggestions."""


class LLMLayoutCriticAgent:
    """OpenAI-compatible text critic for layout contracts.

    The current LLM client is text-only, so this critic cannot directly inspect
    image pixels. It uses the construction plan, generated code, render metadata,
    and extracted figure layout trace. If a VLM client is added later, it should
    implement the same `LayoutCriticAgent` protocol.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        critic_name: str = "llm_layout_critic_v1",
        max_tokens: int | None = 2048,
    ) -> None:
        self.client = client
        self.critic_name = critic_name
        self.max_tokens = max_tokens

    def critique(
        self,
        *,
        query: str,
        construction_plan: dict[str, Any],
        generated_code: str,
        render_result: Any,
        actual_figure: Any | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> LayoutCritique:
        result = self.client.complete_json_with_trace(
            system_prompt=_system_prompt(),
            user_prompt=_user_prompt(
                query=query,
                construction_plan=construction_plan,
                generated_code=generated_code,
                render_result=render_result,
                actual_figure=actual_figure,
                generation_context=generation_context,
            ),
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        payload = result.payload
        return LayoutCritique(
            ok=bool(payload.get("ok", False)),
            failed_contracts=_string_tuple(payload.get("failed_contracts")),
            diagnosis=str(payload.get("diagnosis") or ""),
            recommended_plan_updates=tuple(
                dict(item)
                for item in list(payload.get("recommended_plan_updates") or [])
                if isinstance(item, dict)
            ),
            plan_feedback=tuple(
                dict(item)
                for item in list(payload.get("plan_feedback") or payload.get("findings") or [])
                if isinstance(item, dict)
            ),
            confidence=_confidence(payload.get("confidence")),
            llm_trace=result.trace,
            metadata={
                "critic_name": self.critic_name,
                "mode": "text_only",
                "image_pixels_available": False,
                "limitation": "Current OpenAI-compatible client sends text only; image_path is provided as metadata, not pixels.",
            },
        )


class VLMLayoutAgent:
    """Vision-first LayoutAgent for layout and composition critique.

    The image is the primary evidence. The plan, code excerpt, and figure trace
    constrain the critique so the agent reports layout, composition,
    readability, and visual-quality issues at the plan level. Coordinate-level
    placement must be left to ExecutorAgent during figure execution.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        agent_name: str = "vlm_layout_agent_v1",
        max_tokens: int | None = 2048,
    ) -> None:
        self.client = client
        self.agent_name = agent_name
        self.max_tokens = max_tokens

    def critique(
        self,
        *,
        query: str,
        construction_plan: dict[str, Any],
        generated_code: str,
        render_result: Any,
        actual_figure: Any | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> LayoutCritique:
        image_path = getattr(render_result, "image_path", None)
        if image_path is None:
            return LayoutCritique(
                ok=False,
                failed_contracts=("layout.image_missing",),
                diagnosis="LayoutAgent requires a rendered image but render_result.image_path is missing.",
                confidence=1.0,
                metadata={
                    "critic_name": self.agent_name,
                    "mode": "vlm",
                    "image_pixels_available": False,
                },
            )
        complete_with_image = getattr(self.client, "complete_json_with_image_trace", None)
        if not callable(complete_with_image):
            raise TypeError("VLMLayoutAgent requires an LLM client with complete_json_with_image_trace().")
        try:
            result = complete_with_image(
                system_prompt=_vlm_system_prompt(),
                user_prompt=_user_prompt(
                    query=query,
                    construction_plan=construction_plan,
                    generated_code=generated_code,
                    render_result=render_result,
                    actual_figure=actual_figure,
                    generation_context=generation_context,
                ),
                image_path=image_path,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                "VLMLayoutAgent image critique failed. The configured layout provider likely does not support "
                "OpenAI-compatible image_url messages. Configure a vision-capable model under llm.layout "
                "or run with --layout-agent-backend text."
            ) from exc
        payload = result.payload
        return LayoutCritique(
            ok=bool(payload.get("ok", False)),
            failed_contracts=_string_tuple(payload.get("failed_contracts")),
            diagnosis=str(payload.get("diagnosis") or ""),
            recommended_plan_updates=tuple(
                dict(item)
                for item in list(payload.get("recommended_plan_updates") or [])
                if isinstance(item, dict)
            ),
            plan_feedback=tuple(
                dict(item)
                for item in list(payload.get("plan_feedback") or payload.get("findings") or [])
                if isinstance(item, dict)
            ),
            confidence=_confidence(payload.get("confidence")),
            llm_trace=result.trace,
            metadata={
                "critic_name": self.agent_name,
                "mode": "vlm",
                "image_pixels_available": True,
                "image_path": str(image_path),
            },
        )


def _system_prompt() -> str:
    return (
        "You are LayoutCriticAgent for a grounded chart-generation pipeline. "
        "Your task is layout, composition, and presentation criticism. Do not judge data values, data transformations, chart semantics, or exact style taste. "
        "Check whether the generated code and extracted figure layout obey the construction plan's panel relationships, inset anchoring intent, "
        "occlusion-avoidance rules, reserved regions, legends, titles, whitespace, alignment, hierarchy, and manual-axis constraints. "
        "If the evidence is insufficient, do not invent a failure. "
        "Return only a JSON object with keys: ok, failed_contracts, diagnosis, plan_feedback, recommended_plan_updates, confidence. "
        "Use failed_contracts such as layout.panel_bounds, layout.inset_anchor_alignment, layout.no_occlusion, "
        "layout.reserved_region, layout.legend_collision, layout.title_collision, layout.manual_axes. "
        "plan_feedback is preferred and must contain one item per issue with issue_id, source_agent='LayoutAgent', issue_type, severity, evidence, affected_region, related_plan_ref, and suggested_plan_action. "
        "Every suggested_plan_action must target PlanAgent, not ExecutorAgent, and must describe a plan-level layout/composition contract revision. "
        "Do not output normalized bounds, exact coordinates, pixel positions, or numeric panel sizes. Say what layout relationship should change, not the coordinates. "
        "recommended_plan_updates is a backward-compatible optional field; if used, prefer panel layout_notes, placement_policy, avoid_occlusion, reserved-region, hierarchy, or legend placement intent rather than bounds. "
        "Never recommend changing data, chart type, values, labels, or source files."
    )


def _vlm_system_prompt() -> str:
    return (
        "You are LayoutAgent for a grounded chart-generation pipeline. "
        "You inspect the rendered chart image as the primary evidence, then use the construction plan, code excerpt, and figure trace as constraints. "
        "Your task has two parts: layout/composition fidelity and visual-quality critique. "
        "Layout/composition fidelity checks whether explicit or inferred layout requirements are followed: panel count, main/inset relationships, anchor intent, legends, titles, reserved regions, occlusion avoidance, and readable composition. "
        "Visual quality checks objective readability proxies: overlap, crowding, excessive whitespace, misalignment, unreadable inset size, clipped labels, poor hierarchy, and awkward placement. "
        "Do not judge data values, data transformations, chart semantics, chart type correctness, or color semantics unless they create a layout/readability issue. "
        "Never recommend changing data, chart type, plotted values, labels, legends categories, annotations, or source files. "
        "Return only a JSON object with keys: ok, failed_contracts, diagnosis, plan_feedback, recommended_plan_updates, confidence. "
        "Use failed_contracts such as layout.panel_bounds, layout.inset_anchor_alignment, layout.no_occlusion, "
        "layout.reserved_region, layout.legend_collision, layout.title_collision, layout.manual_axes, "
        "layout.visual_crowding, layout.excess_whitespace, layout.inset_readability, layout.visual_hierarchy. "
        "plan_feedback is preferred and must contain one item per layout/readability issue with issue_id, source_agent='LayoutAgent', issue_type, severity, evidence, affected_region, related_plan_ref, and suggested_plan_action. "
        "Every suggested_plan_action must target PlanAgent, not ExecutorAgent, and must describe a plan-level layout/composition contract revision. "
        "Do not output normalized bounds, exact coordinates, pixel positions, or numeric panel sizes. Say what layout relationship should change, not the coordinates. "
        "recommended_plan_updates is a backward-compatible optional field; if used, prefer panel layout_notes, placement_policy, avoid_occlusion, reserved-region, hierarchy, or legend placement intent rather than bounds. "
        "If the image is acceptable or evidence is insufficient, return ok=true and no updates."
    )


def _user_prompt(
    *,
    query: str,
    construction_plan: dict[str, Any],
    generated_code: str,
    render_result: Any,
    actual_figure: Any | None,
    generation_context: dict[str, Any] | None,
) -> str:
    payload = {
        "query": query,
        "generation_context": _compact_generation_context(generation_context or {}),
        "construction_plan_layout": _compact_plan_layout(construction_plan),
        "render_result": _render_payload(render_result),
        "actual_figure_layout_trace": _figure_payload(actual_figure),
        "generated_code_layout_excerpt": _code_layout_excerpt(generated_code),
        "critic_scope": {
            "allowed": [
                "panel composition and hierarchy",
                "inset placement intent",
                "anchor alignment intent",
                "occlusion avoidance",
                "legend/title placement",
                "reserved regions",
                "visual balance and whitespace",
                "manual axes layout stability",
            ],
            "forbidden": [
                "data correctness",
                "chart type correctness",
                "series values",
                "source file changes",
                "label text changes",
                "exact normalized bounds",
                "pixel coordinates",
                "numeric panel sizes",
            ],
        },
    }
    return "Critique the layout contract for this generated chart.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _compact_plan_layout(plan: dict[str, Any]) -> dict[str, Any]:
    panels = []
    for panel in list(plan.get("panels") or []):
        if not isinstance(panel, dict):
            continue
        panels.append(
            {
                "panel_id": panel.get("panel_id"),
                "role": panel.get("role"),
                "bounds": panel.get("bounds"),
                "anchor": panel.get("anchor"),
                "placement_policy": panel.get("placement_policy"),
                "avoid_occlusion": panel.get("avoid_occlusion"),
                "layout_notes": panel.get("layout_notes"),
                "layers": [
                    {
                        "layer_id": layer.get("layer_id"),
                        "chart_type": layer.get("chart_type"),
                        "role": layer.get("role"),
                        "axis": layer.get("axis"),
                    }
                    for layer in list(panel.get("layers") or [])
                    if isinstance(layer, dict)
                ],
            }
        )
    return {
        "plan_type": plan.get("plan_type"),
        "layout_strategy": plan.get("layout_strategy"),
        "figure_size": plan.get("figure_size"),
        "panels": panels,
        "global_elements": plan.get("global_elements"),
        "constraints": [
            item
            for item in list(plan.get("constraints") or [])
            if any(keyword in str(item).lower() for keyword in ("layout", "occlusion", "inset", "legend", "title", "axes"))
        ],
    }


def _compact_generation_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in context.items()
        if key in {"simple_instruction", "expert_instruction", "metadata", "source", "native_id"}
    }


def _render_payload(render_result: Any) -> dict[str, Any]:
    image_path = getattr(render_result, "image_path", None)
    return {
        "ok": bool(getattr(render_result, "ok", False)),
        "image_path": str(image_path) if image_path is not None else None,
        "backend": getattr(render_result, "backend", None),
        "exception_type": getattr(render_result, "exception_type", None),
        "exception_message": getattr(render_result, "exception_message", None),
        "metadata": _jsonable(getattr(render_result, "metadata", {})),
    }


def _figure_payload(actual_figure: Any | None) -> dict[str, Any] | None:
    if actual_figure is None:
        return None
    axes = []
    for axis in list(getattr(actual_figure, "axes", ()) or ()):
        axes.append(
            {
                "index": getattr(axis, "index", None),
                "title": getattr(axis, "title", ""),
                "xlabel": getattr(axis, "xlabel", ""),
                "ylabel": getattr(axis, "ylabel", ""),
                "bounds": list(getattr(axis, "bounds", None) or []),
                "legend_labels": list(getattr(axis, "legend_labels", ()) or ()),
                "texts": list(getattr(axis, "texts", ()) or ()),
                "artists": [
                    {
                        "artist_type": getattr(artist, "artist_type", None),
                        "label": getattr(artist, "label", None),
                        "count": getattr(artist, "count", None),
                    }
                    for artist in list(getattr(axis, "artists", ()) or ())
                ],
            }
        )
    return {
        "title": getattr(actual_figure, "title", ""),
        "size_inches": list(getattr(actual_figure, "size_inches", None) or []),
        "axes": axes,
        "raw": _jsonable(getattr(actual_figure, "raw", {})),
    }


def _code_layout_excerpt(code: str, *, max_chars: int = 12000) -> str:
    lines = str(code or "").splitlines()
    keywords = (
        "add_axes",
        "inset",
        "GridSpec",
        "subplots",
        "subplots_adjust",
        "tight_layout",
        "constrained_layout",
        "legend",
        "suptitle",
        ".set_title",
        ".pie",
        "pie(",
        "bbox_to_anchor",
        "set_position",
    )
    selected: list[str] = []
    for index, line in enumerate(lines, start=1):
        if any(keyword in line for keyword in keywords):
            selected.append(f"{index}: {line}")
    excerpt = "\n".join(selected)
    if not excerpt:
        excerpt = str(code or "")
    if len(excerpt) > max_chars:
        return excerpt[:max_chars] + "\n... [truncated]"
    return excerpt


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _llm_trace_to_dict(trace: LLMCompletionTrace | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    return {
        "provider": trace.provider,
        "model": trace.model,
        "base_url": trace.base_url,
        "temperature": trace.temperature,
        "max_tokens": trace.max_tokens,
        "usage": _usage_to_dict(trace.usage),
        "parsed_json": _jsonable(trace.parsed_json),
        "raw_text_preview": str(trace.raw_text or "")[:1200],
    }


def _usage_to_dict(usage: LLMUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "raw": _jsonable(usage.raw),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)
