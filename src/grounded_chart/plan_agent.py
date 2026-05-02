from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from grounded_chart.artifact_workspace import PLAN_AGENT_DIR
from grounded_chart.construction_plan import (
    ChartConstructionPlan,
    chart_construction_plan_from_dict,
)
from grounded_chart.llm import LLMClient, LLMCompletionTrace
from grounded_chart.requirements import ChartRequirementPlan


@dataclass(frozen=True)
class PlanAgentRequest:
    query: str
    case_id: str = ""
    output_root: str | Path | None = None
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
        workspace = _plan_agent_workspace(request)
        input_package = _build_input_package(request, agent_name="heuristic_plan_agent", workspace=workspace)
        _write_input_artifacts(workspace, input_package)
        plan = self.planner.build_plan(
            query=request.query,
            requirement_plan=request.requirement_plan,
            source_data_plan=request.source_data_plan,
            context=request.context,
        )
        self_check = _self_check_plan(plan, request=request, feedback_resolution=())
        result = PlanAgentResult(
            plan=plan,
            agent_name="heuristic_plan_agent",
            metadata={
                "round_index": request.round_index,
                "workspace_dir": str(workspace) if workspace is not None else None,
                "self_check_ok": self_check["ok"],
                "state_mode": "file_backed" if workspace is not None else "stateless",
            },
        )
        _write_result_artifacts(workspace, result=result, self_check=self_check, input_package=input_package)
        return result


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
        workspace = _plan_agent_workspace(request)
        input_package = _build_input_package(request, agent_name=self.agent_name, workspace=workspace)
        _write_input_artifacts(workspace, input_package)
        prompt_payload = _prompt_payload(request, input_package=input_package)
        _write_prompt_payload(workspace, prompt_payload)
        result = self.client.complete_json_with_trace(
            system_prompt=_system_prompt(),
            user_prompt=_user_prompt(prompt_payload),
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
        self_check = _self_check_plan(plan, request=request, feedback_resolution=feedback_resolution)
        plan_result = PlanAgentResult(
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
                "workspace_dir": str(workspace) if workspace is not None else None,
                "self_check_ok": self_check["ok"],
                "state_mode": "file_backed" if workspace is not None else "stateless",
                "prompt_payload_path": str(workspace / "prompt_payload.json") if workspace is not None else None,
            },
        )
        _write_result_artifacts(workspace, result=plan_result, self_check=self_check, input_package=input_package)
        return plan_result


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


def _user_prompt(payload: dict[str, Any]) -> str:
    return "Build or revise the chart construction plan.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _prompt_payload(request: PlanAgentRequest, *, input_package: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": request.query,
        "case_id": request.case_id,
        "round_index": request.round_index,
        "workspace": input_package.get("workspace"),
        "loaded_memory": input_package.get("loaded_memory"),
        "source_cards": input_package.get("source_cards"),
        "requirement_index": input_package.get("requirement_index"),
        "context": _compact_context(request.context),
        "scaffold_plan": _compact_plan_for_prompt(request.scaffold_plan),
        "previous_plan_summary": _plan_summary(request.previous_plan),
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


def _to_dict(value: Any) -> Any:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, dict):
        return value
    return str(value)


def _plan_agent_workspace(request: PlanAgentRequest) -> Path | None:
    if request.output_root is None:
        return None
    root = Path(request.output_root).resolve()
    workspace = root / PLAN_AGENT_DIR / f"round_{request.round_index}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _build_input_package(
    request: PlanAgentRequest,
    *,
    agent_name: str,
    workspace: Path | None,
) -> dict[str, Any]:
    loaded_memory = _load_previous_memory(workspace, request.round_index)
    source_cards = _source_cards(request.source_data_plan, request.source_data_execution)
    requirement_index = _requirement_index(request.requirement_plan)
    return {
        "agent_name": agent_name,
        "case_id": request.case_id,
        "round_index": request.round_index,
        "workspace": None
        if workspace is None
        else {
            "dir": str(workspace),
            "round_id": f"round_{request.round_index}",
            "memory_file": str(workspace / "task_memory.json"),
            "input_manifest_file": str(workspace / "input_manifest.json"),
            "self_check_file": str(workspace / "self_check.json"),
        },
        "input_manifest": {
            "query": request.query,
            "has_requirement_plan": request.requirement_plan is not None,
            "has_source_data_plan": request.source_data_plan is not None,
            "has_source_data_execution": request.source_data_execution is not None,
            "has_scaffold_plan": request.scaffold_plan is not None,
            "has_previous_plan": request.previous_plan is not None,
            "feedback_issue_ids": _feedback_issue_ids(request.feedback_bundle),
            "state_policy": "file_backed_compact_memory",
        },
        "loaded_memory": loaded_memory,
        "source_cards": source_cards,
        "requirement_index": requirement_index,
        "previous_plan_summary": _plan_summary(request.previous_plan),
        "scaffold_plan_summary": _plan_summary(request.scaffold_plan),
        "feedback_bundle": _compact_feedback_bundle(request.feedback_bundle),
    }


def _write_input_artifacts(workspace: Path | None, input_package: dict[str, Any]) -> None:
    if workspace is None:
        return
    _write_json(workspace / "input_manifest.json", input_package.get("input_manifest") or {})
    _write_json(workspace / "loaded_memory.json", input_package.get("loaded_memory") or {})
    _write_json(workspace / "source_cards.json", input_package.get("source_cards") or {})
    _write_json(workspace / "requirement_index.json", input_package.get("requirement_index") or {})
    feedback = input_package.get("feedback_bundle")
    if feedback:
        _write_json(workspace / "feedback_bundle.json", feedback)


def _write_prompt_payload(workspace: Path | None, payload: dict[str, Any]) -> None:
    if workspace is None:
        return
    _write_json(workspace / "prompt_payload.json", payload)


def _write_result_artifacts(
    workspace: Path | None,
    *,
    result: PlanAgentResult,
    self_check: dict[str, Any],
    input_package: dict[str, Any],
) -> None:
    if workspace is None:
        return
    _write_json(workspace / "plan.json", result.plan.to_dict())
    _write_json(workspace / "feedback_resolution.json", [dict(item) for item in result.feedback_resolution])
    _write_json(workspace / "self_check.json", self_check)
    memory = _next_memory(
        result.plan,
        input_package=input_package,
        result=result,
        self_check=self_check,
    )
    _write_json(workspace / "task_memory.json", memory)


def _load_previous_memory(workspace: Path | None, round_index: int) -> dict[str, Any]:
    if workspace is None or round_index <= 1:
        return {}
    parent = workspace.parent
    for index in range(round_index - 1, 0, -1):
        memory_path = parent / f"round_{index}" / "task_memory.json"
        if not memory_path.exists():
            continue
        try:
            payload = json.loads(memory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return _compact_loaded_memory(payload)
    return {}


def _compact_loaded_memory(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "case_id": payload.get("case_id"),
            "query_digest": payload.get("query_digest"),
            "source_tables": payload.get("source_tables"),
            "requirement_families": payload.get("requirement_families"),
            "chart_types": payload.get("chart_types"),
            "panel_ids": payload.get("panel_ids"),
            "layer_ids": payload.get("layer_ids"),
            "layout_strategy": payload.get("layout_strategy"),
            "unresolved_feedback": payload.get("unresolved_feedback"),
            "last_self_check_ok": payload.get("self_check_ok"),
        }.items()
        if value not in (None, "", [], {})
    }


def _next_memory(
    plan: ChartConstructionPlan,
    *,
    input_package: dict[str, Any],
    result: PlanAgentResult,
    self_check: dict[str, Any],
) -> dict[str, Any]:
    source_cards = input_package.get("source_cards") if isinstance(input_package.get("source_cards"), dict) else {}
    requirement_index = input_package.get("requirement_index") if isinstance(input_package.get("requirement_index"), dict) else {}
    return {
        "case_id": input_package.get("case_id"),
        "round_index": input_package.get("round_index"),
        "query_digest": _query_digest(str((input_package.get("input_manifest") or {}).get("query") or "")),
        "source_tables": [
            {
                "name": table.get("name"),
                "columns": table.get("columns"),
                "row_count_loaded": table.get("row_count_loaded"),
            }
            for table in list(source_cards.get("tables") or [])
            if isinstance(table, dict)
        ],
        "requirement_families": requirement_index.get("families"),
        "chart_types": _plan_chart_types(plan),
        "panel_ids": [panel.panel_id for panel in plan.panels],
        "layer_ids": [layer.layer_id for panel in plan.panels for layer in panel.layers],
        "layout_strategy": plan.layout_strategy,
        "unresolved_feedback": [
            dict(item)
            for item in result.feedback_resolution
            if str(item.get("status") or "").lower() not in {"addressed"}
        ],
        "self_check_ok": self_check.get("ok"),
        "self_check_issue_count": len(self_check.get("issues") or []),
    }


def _source_cards(source_data_plan: Any, source_data_execution: Any) -> dict[str, Any]:
    plan_payload = _to_dict(source_data_plan)
    execution_payload = _compact_source_execution(source_data_execution)
    files = []
    if isinstance(plan_payload, dict):
        for item in list(plan_payload.get("files") or []):
            if not isinstance(item, dict):
                continue
            files.append(
                {
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "kind": item.get("kind"),
                    "status": item.get("status"),
                }
            )
    tables = []
    if isinstance(execution_payload, dict):
        for table in list(execution_payload.get("loaded_tables") or []):
            if not isinstance(table, dict):
                continue
            tables.append(
                {
                    "name": table.get("name"),
                    "columns": table.get("columns"),
                    "row_count_loaded": table.get("row_count_loaded"),
                    "preview_rows": list(table.get("preview_rows") or [])[:5],
                    "read_error": table.get("read_error"),
                }
            )
    return {"files": files, "tables": tables}


def _requirement_index(plan: ChartRequirementPlan | None) -> dict[str, Any]:
    payload = _requirement_plan_payload(plan)
    if not isinstance(payload, dict):
        return {}
    requirements = list(payload.get("requirements") or [])
    families: dict[str, int] = {}
    compact_requirements: list[dict[str, Any]] = []
    for req in requirements:
        if not isinstance(req, dict):
            continue
        family = str(req.get("type") or req.get("scope") or "unknown")
        families[family] = families.get(family, 0) + 1
        compact_requirements.append(
            {
                "requirement_id": req.get("requirement_id"),
                "scope": req.get("scope"),
                "type": req.get("type"),
                "name": req.get("name"),
                "value": req.get("value"),
                "source_span": req.get("source_span"),
                "priority": req.get("priority"),
                "severity": req.get("severity"),
            }
        )
    return {
        "raw_query": payload.get("raw_query"),
        "figure_requirements": payload.get("figure_requirements"),
        "families": families,
        "requirements": compact_requirements,
        "panels": payload.get("panels"),
    }


def _compact_feedback_bundle(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(bundle, dict):
        return None
    return {
        "round_index": bundle.get("round_index"),
        "mode": bundle.get("mode"),
        "summary": bundle.get("summary"),
        "failed_contracts": list(bundle.get("failed_contracts") or []),
        "feedback_items": [
            {
                "issue_id": item.get("issue_id"),
                "source_agent": item.get("source_agent"),
                "issue_type": item.get("issue_type"),
                "severity": item.get("severity"),
                "evidence": item.get("evidence"),
                "affected_region": item.get("affected_region"),
                "related_plan_ref": item.get("related_plan_ref"),
                "suggested_plan_action": item.get("suggested_plan_action"),
            }
            for item in list(bundle.get("feedback_items") or [])
            if isinstance(item, dict)
        ],
    }


def _feedback_issue_ids(bundle: dict[str, Any] | None) -> list[str]:
    return [
        str(item.get("issue_id"))
        for item in _feedback_items(bundle)
        if str(item.get("issue_id") or "")
    ]


def _compact_plan_for_prompt(plan: ChartConstructionPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return plan.to_dict()


def _plan_summary(plan: ChartConstructionPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "plan_type": plan.plan_type,
        "layout_strategy": plan.layout_strategy,
        "figure_size": list(plan.figure_size) if plan.figure_size is not None else None,
        "chart_types": _plan_chart_types(plan),
        "panels": [
            {
                "panel_id": panel.panel_id,
                "role": panel.role,
                "bounds": list(panel.bounds) if panel.bounds is not None else None,
                "layout_notes": list(panel.layout_notes),
                "anchor": dict(panel.anchor),
                "placement_policy": dict(panel.placement_policy),
                "avoid_occlusion": list(panel.avoid_occlusion),
                "layers": [
                    {
                        "layer_id": layer.layer_id,
                        "chart_type": layer.chart_type,
                        "role": layer.role,
                        "data_source": layer.data_source,
                        "x": layer.x,
                        "y": list(layer.y),
                        "axis": layer.axis,
                        "semantic_modifiers": dict(layer.semantic_modifiers),
                        "visual_channel_plan": dict(layer.visual_channel_plan),
                        "placement_policy": dict(layer.placement_policy),
                    }
                    for layer in panel.layers
                ],
            }
            for panel in plan.panels
        ],
        "global_elements": list(plan.global_elements),
        "constraints": list(plan.constraints),
        "assumptions": list(plan.assumptions),
    }


def _raw_plan_summary(raw: dict[str, Any]) -> dict[str, Any]:
    panels = []
    chart_types: set[str] = set()
    for panel in list(raw.get("panels") or []):
        if not isinstance(panel, dict):
            continue
        layers = []
        for layer in list(panel.get("layers") or []):
            if not isinstance(layer, dict):
                continue
            chart_type = str(layer.get("chart_type") or "")
            if chart_type:
                chart_types.add(chart_type)
            layers.append(
                {
                    "layer_id": layer.get("layer_id"),
                    "chart_type": layer.get("chart_type"),
                    "role": layer.get("role"),
                    "data_source": layer.get("data_source"),
                    "x": layer.get("x"),
                    "y": layer.get("y"),
                    "axis": layer.get("axis"),
                }
            )
        panels.append(
            {
                "panel_id": panel.get("panel_id"),
                "role": panel.get("role"),
                "bounds": panel.get("bounds"),
                "placement_policy": panel.get("placement_policy"),
                "avoid_occlusion": panel.get("avoid_occlusion"),
                "layers": layers,
            }
        )
    return {
        "plan_type": raw.get("plan_type"),
        "layout_strategy": raw.get("layout_strategy"),
        "figure_size": raw.get("figure_size"),
        "chart_types": sorted(chart_types),
        "panels": panels,
        "global_elements": list(raw.get("global_elements") or []),
    }


def _self_check_plan(
    plan: ChartConstructionPlan,
    *,
    request: PlanAgentRequest,
    feedback_resolution: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not plan.panels:
        issues.append({"code": "no_panels", "severity": "error", "message": "Plan has no panels."})
    if not any(panel.layers for panel in plan.panels):
        issues.append({"code": "no_layers", "severity": "error", "message": "Plan has no visual layers."})
    for panel in plan.panels:
        if panel.bounds is not None and not _numeric_bounds_allowed(request):
            issues.append(
                {
                    "code": "unjustified_numeric_bounds",
                    "severity": "warning",
                    "plan_ref": f"panels.{panel.panel_id}.bounds",
                    "message": "PlanAgent produced numeric bounds without an explicit coordinate requirement.",
                }
            )
    missing_feedback = sorted(set(_feedback_issue_ids(request.feedback_bundle)) - {str(item.get("issue_id")) for item in feedback_resolution})
    if missing_feedback:
        issues.append(
            {
                "code": "missing_feedback_resolution",
                "severity": "error",
                "message": "Feedback issues are missing feedback_resolution entries.",
                "issue_ids": missing_feedback,
            }
        )
    vague_terms = ("nice", "beautiful", "clear if possible", "appropriate")
    vague_refs = []
    plan_text = json.dumps(plan.to_dict(), ensure_ascii=False).lower()
    for term in vague_terms:
        if term in plan_text:
            vague_refs.append(term)
    if vague_refs:
        issues.append(
            {
                "code": "vague_plan_language",
                "severity": "warning",
                "message": "Plan contains vague presentation language that may be hard for ExecutorAgent to execute.",
                "terms": vague_refs,
            }
        )
    return {
        "ok": not any(str(issue.get("severity")) == "error" for issue in issues),
        "issues": issues,
        "checked_contracts": [
            "panel_presence",
            "layer_presence",
            "numeric_bounds_delegation",
            "feedback_resolution_completeness",
            "vague_language_scan",
        ],
    }


def _numeric_bounds_allowed(request: PlanAgentRequest) -> bool:
    query = request.query.lower()
    explicit_terms = ("bounds", "coordinate", "position [", "x0", "y0", "left=", "bottom=")
    if any(term in query for term in explicit_terms):
        return True
    if request.previous_plan is not None and any(panel.bounds is not None for panel in request.previous_plan.panels):
        return True
    return False


def _plan_chart_types(plan: ChartConstructionPlan) -> list[str]:
    chart_types = sorted(
        {
            str(layer.chart_type)
            for panel in plan.panels
            for layer in panel.layers
            if str(layer.chart_type or "")
        }
    )
    return chart_types


def _query_digest(query: str, *, max_chars: int = 360) -> str:
    text = " ".join(str(query or "").split())
    return text[:max_chars]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
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
        replanning.pop("previous_code", None)
        previous_plan = replanning.pop("previous_construction_plan", None)
        if isinstance(previous_plan, dict):
            replanning["previous_plan_summary"] = _raw_plan_summary(previous_plan)
        feedback_bundle = replanning.pop("feedback_bundle", None)
        if isinstance(feedback_bundle, dict):
            replanning["feedback_summary"] = {
                "round_index": feedback_bundle.get("round_index"),
                "summary": feedback_bundle.get("summary"),
                "failed_contracts": list(feedback_bundle.get("failed_contracts") or []),
                "feedback_issue_ids": [
                    str(item.get("issue_id"))
                    for item in list(feedback_bundle.get("feedback_items") or [])
                    if isinstance(item, dict) and str(item.get("issue_id") or "")
                ],
            }
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
