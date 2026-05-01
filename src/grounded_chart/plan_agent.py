from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from grounded_chart.construction_plan import (
    ChartConstructionPlan,
    chart_construction_plan_from_dict,
)
from grounded_chart.llm import LLMClient, LLMCompletionTrace
from grounded_chart.requirements import ChartRequirementPlan


@dataclass(frozen=True)
class PlanAgentRequest:
    query: str
    requirement_plan: ChartRequirementPlan | None = None
    source_data_plan: Any | None = None
    source_data_execution: Any | None = None
    context: dict[str, Any] = field(default_factory=dict)
    scaffold_plan: ChartConstructionPlan | None = None
    previous_plan: ChartConstructionPlan | None = None
    feedback_bundle: dict[str, Any] | None = None
    round_index: int = 1


@dataclass(frozen=True)
class PlanAgentResult:
    plan: ChartConstructionPlan
    agent_name: str
    feedback_resolution: tuple[dict[str, Any], ...] = ()
    rationale: str = ""
    llm_trace: LLMCompletionTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChartPlanAgent(Protocol):
    def build_plan(self, request: PlanAgentRequest) -> PlanAgentResult:
        """Return a complete construction plan for a chart-generation round."""


class HeuristicPlanAgent:
    """Adapter for deterministic construction planning."""

    def __init__(self, planner) -> None:
        self.planner = planner

    def build_plan(self, request: PlanAgentRequest) -> PlanAgentResult:
        plan = self.planner.build_plan(
            query=request.query,
            requirement_plan=request.requirement_plan,
            source_data_plan=request.source_data_plan,
            context=request.context,
        )
        return PlanAgentResult(
            plan=plan,
            agent_name="heuristic_plan_agent",
            metadata={"round_index": request.round_index},
        )


class LLMPlanAgent:
    """LLM PlanAgent that outputs a complete typed ChartConstructionPlan.

    The LLM receives a deterministic scaffold, source-data summaries, and
    optional visual feedback. It must revise the construction plan, not patch
    plotting code directly.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        agent_name: str = "llm_plan_agent_v1",
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.agent_name = agent_name
        self.max_tokens = max_tokens

    def build_plan(self, request: PlanAgentRequest) -> PlanAgentResult:
        result = self.client.complete_json_with_trace(
            system_prompt=_system_prompt(),
            user_prompt=_user_prompt(request),
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        payload = result.payload
        raw_plan = payload.get("construction_plan") or payload.get("plan")
        if not isinstance(raw_plan, dict):
            raise ValueError("LLMPlanAgent response missing object `construction_plan`.")
        plan = chart_construction_plan_from_dict(raw_plan)
        raw_resolution = tuple(
            dict(item)
            for item in list(payload.get("feedback_resolution") or [])
            if isinstance(item, dict)
        )
        feedback_resolution = _complete_feedback_resolution(
            raw_resolution,
            feedback_bundle=request.feedback_bundle,
            revised_plan=plan,
        )
        return PlanAgentResult(
            plan=plan,
            agent_name=str(payload.get("agent_name") or self.agent_name),
            feedback_resolution=feedback_resolution,
            rationale=str(payload.get("rationale") or ""),
            llm_trace=result.trace,
            metadata={
                "round_index": request.round_index,
                "used_scaffold": request.scaffold_plan is not None,
                "used_feedback": bool(request.feedback_bundle),
                "feedback_resolution_source": _feedback_resolution_source(raw_resolution, feedback_resolution),
                "feedback_resolution_count": len(feedback_resolution),
            },
        )


def _system_prompt() -> str:
    return (
        "You are PlanAgent for GroundedChart, an evidence-grounded language-to-chart generation pipeline. "
        "Your job is to produce a complete chart_construction_plan_v2 for ExecutorAgent. "
        "Do not write plotting code. Do not output prose outside JSON. "
        "Return only a JSON object with keys: agent_name, construction_plan, feedback_resolution, rationale. "
        "The construction_plan must include plan_type, layout_strategy, figure_size, panels, global_elements, "
        "decisions, assumptions, constraints, data_transform_plan, execution_steps, and style_policy. "
        "Each panel should include panel_id, role, layers, axes, layout_notes, anchor, placement_policy, "
        "avoid_occlusion, style_policy, and z_order. "
        "Each layer should include layer_id, chart_type, role, data_source, x, y, axis, status, rationale, "
        "encoding, data_transform, components, semantic_modifiers, visual_channel_plan, style_policy, placement_policy, and z_order. "
        "Use the original request and source schema as truth. Preserve explicit chart requirements unless they conflict with verified source facts. "
        "Use deterministic artifacts for computations; never ask ExecutorAgent to use an LLM for arithmetic. "
        "Do not invent hard numeric bounds for panels or insets unless the user explicitly requested exact coordinates, "
        "the previous plan already contains source-grounded bounds that must be preserved, or a hard layout constraint requires them. "
        "Prefer semantic layout contracts in placement_policy, anchor, avoid_occlusion, layout_notes, and style_policy. "
        "ExecutorAgent is responsible for computing concrete matplotlib/plotly layout coordinates from these contracts while recording its layout decisions. "
        "When feedback_bundle is present, revise the whole plan to address each feedback item with executable layout, "
        "composition, visual-channel, legend, inset, or chart-protocol commitments. Do not merely copy feedback into notes. "
        "feedback_resolution is mandatory when feedback_bundle.feedback_items is non-empty: it must contain exactly one item for each feedback issue_id. "
        "Each feedback_resolution item must include issue_id, status, evidence_summary, plan_change, affected_plan_refs, and rationale. "
        "Use status='addressed' when the revised plan changes something, status='deferred' when it is postponed, and status='rejected' when the feedback conflicts with source-grounded requirements. "
        "If you reject or defer feedback, explain why in feedback_resolution."
    )


def _user_prompt(request: PlanAgentRequest) -> str:
    payload = {
        "query": request.query,
        "round_index": request.round_index,
        "source_data_plan": _to_dict(request.source_data_plan),
        "source_data_execution": _compact_source_execution(request.source_data_execution),
        "requirement_plan": _requirement_plan_payload(request.requirement_plan),
        "context": _compact_context(request.context),
        "scaffold_plan": request.scaffold_plan.to_dict() if request.scaffold_plan is not None else None,
        "previous_plan": request.previous_plan.to_dict() if request.previous_plan is not None else None,
        "feedback_bundle": request.feedback_bundle,
        "planning_rules": [
            "Output a full replacement plan, not a diff.",
            "Every explicit visual requirement should map to a panel, layer, global element, or constraint.",
            "Represent uncertain interpretations as assumptions or decisions with rationale.",
            "Use executable chart-type protocols where needed, especially waterfall, stacked/overlap area, grouped bars, facets, and insets.",
            "For multi-layer figures, specify layout strategy, relative placement, anchoring, reserved regions, and occlusion avoidance clearly enough for ExecutorAgent.",
            "Do not output numeric panel bounds unless the request explicitly specifies exact placement or the bound is a source-grounded hard constraint.",
            "If concrete coordinates are needed, describe them as executor-computed layout decisions rather than PlanAgent-authored requirements.",
            "For feedback-driven rounds, feedback_resolution must have exactly one item per feedback issue_id with status addressed, deferred, or rejected.",
            "Each feedback_resolution item must include affected_plan_refs pointing to concrete plan locations such as panels.panel.pie_2008.placement_policy, global_elements.legend, or layer.import_waterfall.visual_channel_plan.",
        ],
    }
    return "Build or revise the chart construction plan.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _to_dict(value: Any) -> Any:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, dict):
        return value
    return str(value)


def _compact_source_execution(value: Any) -> Any:
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return payload
    tables = []
    for table in list(payload.get("loaded_tables") or []):
        if not isinstance(table, dict):
            continue
        tables.append(
            {
                "name": table.get("name"),
                "columns": table.get("columns"),
                "row_count_loaded": table.get("row_count_loaded"),
                "preview_rows": list(table.get("rows") or [])[:8],
                "read_error": table.get("read_error"),
            }
        )
    return {"loaded_tables": tables}


def _requirement_plan_payload(plan: ChartRequirementPlan | None) -> Any:
    if plan is None:
        return None
    return {
        "raw_query": plan.raw_query,
        "figure_requirements": dict(plan.figure_requirements),
        "requirements": [
            {
                "requirement_id": req.requirement_id,
                "scope": req.scope,
                "type": req.type,
                "name": req.name,
                "value": req.value,
                "source_span": req.source_span,
                "status": req.status,
                "priority": req.priority,
                "severity": req.severity,
            }
            for req in plan.requirements
        ],
        "panels": [
            {
                "panel_id": panel.panel_id,
                "chart_type": panel.chart_type,
                "requirement_ids": list(panel.requirement_ids),
            }
            for panel in plan.panels
        ],
    }


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    result = dict(context or {})
    # Avoid recursively sending large previous code through PlanAgent. Executor
    # receives the prior code separately during layout_replanning.
    replanning = result.get("plan_replanning")
    if isinstance(replanning, dict):
        replanning = dict(replanning)
        replanning.pop("previous_code_excerpt", None)
        result["plan_replanning"] = replanning
    return result


def _complete_feedback_resolution(
    raw_resolution: tuple[dict[str, Any], ...],
    *,
    feedback_bundle: dict[str, Any] | None,
    revised_plan: ChartConstructionPlan,
) -> tuple[dict[str, Any], ...]:
    feedback_items = _feedback_items(feedback_bundle)
    if not feedback_items:
        return raw_resolution
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_resolution:
        issue_id = str(item.get("issue_id") or "").strip()
        if issue_id:
            by_id[issue_id] = _normalize_resolution_item(item, source="model")
    completed: list[dict[str, Any]] = []
    plan_refs = _candidate_plan_refs(revised_plan)
    for feedback in feedback_items:
        issue_id = str(feedback.get("issue_id") or "").strip()
        if not issue_id:
            continue
        if issue_id in by_id:
            completed.append(by_id[issue_id])
            continue
        completed.append(
            {
                "issue_id": issue_id,
                "status": "missing_from_model",
                "evidence_summary": str(feedback.get("evidence") or ""),
                "plan_change": (
                    "LLM PlanAgent did not provide an explicit feedback_resolution item for this issue; "
                    "inspect revised_plan and plan diff for actual handling."
                ),
                "affected_plan_refs": plan_refs,
                "rationale": "Auto-filled by framework to preserve feedback-to-plan trace completeness.",
                "source": "framework_autofill",
            }
        )
    return tuple(completed)


def _normalize_resolution_item(item: dict[str, Any], *, source: str) -> dict[str, Any]:
    affected = item.get("affected_plan_refs")
    if isinstance(affected, str):
        affected_refs = [affected]
    elif isinstance(affected, (list, tuple)):
        affected_refs = [str(ref) for ref in affected if str(ref)]
    else:
        affected_refs = []
    return {
        "issue_id": str(item.get("issue_id") or ""),
        "status": str(item.get("status") or "addressed"),
        "evidence_summary": str(item.get("evidence_summary") or item.get("evidence") or ""),
        "plan_change": str(item.get("plan_change") or item.get("change") or item.get("resolution") or ""),
        "affected_plan_refs": affected_refs,
        "rationale": str(item.get("rationale") or item.get("reason") or ""),
        "source": str(item.get("source") or source),
    }


def _feedback_items(feedback_bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(feedback_bundle, dict):
        return []
    return [dict(item) for item in list(feedback_bundle.get("feedback_items") or []) if isinstance(item, dict)]


def _candidate_plan_refs(plan: ChartConstructionPlan) -> list[str]:
    refs: list[str] = []
    for element in plan.global_elements:
        element_type = str(element.get("type") or "").strip()
        if element_type:
            refs.append(f"global_elements.{element_type}")
    for panel in plan.panels:
        refs.append(f"panels.{panel.panel_id}")
        if panel.bounds is not None:
            refs.append(f"panels.{panel.panel_id}.bounds")
        for layer in panel.layers:
            refs.append(f"layers.{layer.layer_id}")
            if layer.visual_channel_plan:
                refs.append(f"layers.{layer.layer_id}.visual_channel_plan")
            if layer.style_policy:
                refs.append(f"layers.{layer.layer_id}.style_policy")
    return refs[:12]


def _feedback_resolution_source(
    raw_resolution: tuple[dict[str, Any], ...],
    completed_resolution: tuple[dict[str, Any], ...],
) -> str:
    if not completed_resolution:
        return "none"
    if len(raw_resolution) == len(completed_resolution) and all(
        str(item.get("source") or "model") == "model" for item in completed_resolution
    ):
        return "model"
    if raw_resolution:
        return "model_plus_framework_autofill"
    return "framework_autofill"
