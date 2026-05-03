from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from grounded_chart.artifact_workspace import PLAN_AGENT_DIR
from grounded_chart.construction_plan import (
    ChartConstructionPlan,
    PlanDecision,
    chart_construction_plan_from_dict,
)
from grounded_chart.llm import LLMClient, LLMCompletionTrace, LLMJsonResult
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
    feedback_handling: tuple[dict[str, Any], ...] = ()
    plan_brief: dict[str, Any] = field(default_factory=dict)
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
        result = PlanAgentResult(
            plan=plan,
            agent_name="heuristic_plan_agent",
            metadata={
                "round_index": request.round_index,
                "workspace_dir": str(workspace) if workspace is not None else None,
                "plan_mode": "internal_legacy_scaffold",
                "state_mode": "file_backed" if workspace is not None else "stateless",
            },
        )
        _write_result_artifacts(workspace, result=result, input_package=input_package)
        return result


class LLMPlanAgent:
    """LLM PlanAgent that outputs a compact freeform execution brief.

    The typed ChartConstructionPlan is now only an internal compatibility
    scaffold for existing ExecutorAgent/artifact code. It is not the PlanAgent's
    research-facing contract.
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
        _write_llm_response_artifact(workspace, result)
        raw_plan, plan_payload_key = _extract_raw_plan(payload)
        plan_brief, plan_brief_payload_key = _extract_plan_brief(payload)
        if isinstance(raw_plan, dict):
            try:
                plan = chart_construction_plan_from_dict(raw_plan)
            except Exception as exc:
                _write_plan_parse_error_artifact(
                    workspace,
                    payload=payload,
                    result=result,
                    message=str(exc),
                    error_type=type(exc).__name__,
                    plan_payload_key=plan_payload_key,
                )
                raise
            plan_mode = "typed_construction_plan"
        elif plan_brief:
            plan_payload_key = plan_brief_payload_key
            try:
                plan = _bridge_plan_brief_to_construction_plan(request, plan_brief, workspace=workspace)
            except Exception as exc:
                _write_plan_parse_error_artifact(
                    workspace,
                    payload=payload,
                    result=result,
                    message=str(exc),
                    error_type=type(exc).__name__,
                    plan_payload_key=plan_payload_key,
                )
                raise
            plan_mode = "freeform_plan_brief_bridge"
        else:
            message = "LLMPlanAgent response missing a typed plan or recognizable freeform plan brief."
            _write_plan_parse_error_artifact(
                workspace,
                payload=payload,
                result=result,
                message=message,
                error_type="missing_plan_object",
                plan_payload_key=plan_payload_key,
            )
            raise ValueError(message)
        feedback_handling = _extract_feedback_handling(payload, plan_brief)
        plan_result = PlanAgentResult(
            plan=plan,
            agent_name=str(payload.get("agent_name") or self.agent_name),
            feedback_handling=feedback_handling,
            plan_brief=plan_brief,
            rationale=str(payload.get("rationale") or ""),
            llm_trace=result.trace,
            metadata={
                "round_index": request.round_index,
                "used_scaffold": request.scaffold_plan is not None,
                "used_feedback": bool(request.feedback_bundle),
                "plan_mode": plan_mode,
                "feedback_handling_count": len(feedback_handling),
                "workspace_dir": str(workspace) if workspace is not None else None,
                "state_mode": "file_backed" if workspace is not None else "stateless",
                "prompt_payload_path": str(workspace / "prompt_payload.json") if workspace is not None else None,
                "llm_response_path": str(workspace / "llm_response.json") if workspace is not None else None,
                "plan_payload_key": plan_payload_key,
                "plan_brief_path": str(workspace / "plan_brief.json") if workspace is not None and plan_brief else None,
                "plan_brief_keys": sorted(plan_brief.keys()) if plan_brief else [],
            },
        )
        _write_result_artifacts(workspace, result=plan_result, input_package=input_package)
        return plan_result


def _system_prompt() -> str:
    return (
        "You are PlanAgent in a multi-agent chart generation pipeline. "
        "Your job is to produce a compact freeform execution plan brief for ExecutorAgent. "
        "Do not write plotting code. Do not output prose outside JSON. "
        "Return one JSON object. You may choose clear field names; do not overfit to a fixed schema. "
        "A preferred shape is: agent_name, plan_brief, feedback_handling, execution_plan, assumptions, rationale. "
        "execution_plan should be a numbered list of executable steps with enough detail for ExecutorAgent to implement the chart. "
        "Treat the original task and later visual feedback as planning inputs. "
        "If feedback_bundle is present, cover each feedback issue_id and point to the step that should try to handle it. "
        "Do not claim a feedback item is fixed; only state the planned handling. "
        "Avoid fragile coordinate-level patches unless exact placement is explicitly requested. "
        "Let ExecutorAgent compute concrete data transforms and layout with code/scripts, and ask it to record useful implementation notes."
    )


def _user_prompt(payload: dict[str, Any]) -> str:
    return "Build or revise the chart execution plan brief.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _prompt_payload(request: PlanAgentRequest, *, input_package: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": request.query,
        "case_id": request.case_id,
        "round_index": request.round_index,
        "workspace": input_package.get("workspace"),
        "loaded_memory": input_package.get("loaded_memory"),
        "source_cards": input_package.get("source_cards"),
        "context": _compact_context(request.context),
        "previous_plan_summary": _plan_summary(request.previous_plan),
        "feedback_bundle": request.feedback_bundle,
        "planning_rules": [
            "Output a compact freeform plan brief; do not output a typed construction_plan unless this is a legacy compatibility test.",
            "You may choose field names, but make the numbered execution steps and feedback handling easy to find.",
            "Use the plan to guide ExecutorAgent, not to certify fidelity or verification success.",
            "Represent uncertain interpretations as assumptions with rationale.",
            "For multi-layer figures, specify chart structure, data sources, layout intent, and readability intent clearly enough for ExecutorAgent.",
            "If concrete coordinates are needed, describe them as ExecutorAgent-computed layout decisions rather than PlanAgent-authored coordinates.",
            "For feedback-driven rounds, include every feedback issue_id somewhere in feedback handling.",
            "Use planned/try/defer/reject language. Do not use fixed/resolved wording.",
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
    feedback = input_package.get("feedback_bundle")
    if feedback:
        _write_json(workspace / "feedback_bundle.json", feedback)


def _write_prompt_payload(workspace: Path | None, payload: dict[str, Any]) -> None:
    if workspace is None:
        return
    _write_json(workspace / "prompt_payload.json", payload)


def _write_llm_response_artifact(workspace: Path | None, result: LLMJsonResult) -> None:
    if workspace is None:
        return
    _write_json(
        workspace / "llm_response.json",
        {
            "payload": result.payload,
            "trace": _llm_trace_payload(result.trace),
        },
    )


def _write_plan_parse_error_artifact(
    workspace: Path | None,
    *,
    payload: dict[str, Any],
    result: LLMJsonResult,
    message: str,
    error_type: str,
    plan_payload_key: str | None,
) -> None:
    if workspace is None:
        return
    _write_json(
        workspace / "parse_error.json",
        {
            "stage": "plan_payload_parse",
            "error_type": error_type,
            "message": message,
            "accepted_plan_keys": list(_PLAN_PAYLOAD_KEYS),
            "plan_payload_key": plan_payload_key,
            "payload_keys": sorted(str(key) for key in payload.keys()),
            "payload": payload,
            "trace": _llm_trace_payload(result.trace),
        },
    )


def _write_result_artifacts(
    workspace: Path | None,
    *,
    result: PlanAgentResult,
    input_package: dict[str, Any],
) -> None:
    if workspace is None:
        return
    _write_json(workspace / "plan.json", result.plan.to_dict())
    if result.plan_brief:
        _write_json(workspace / "plan_brief.json", result.plan_brief)
    _write_json(workspace / "feedback_handling.json", [dict(item) for item in result.feedback_handling])
    memory = _next_memory(
        result.plan,
        input_package=input_package,
        result=result,
    )
    _write_json(workspace / "task_memory.json", memory)


_PLAN_PAYLOAD_KEYS = ("construction_plan", "plan", "revised_plan")
_PLAN_BRIEF_KEYS = ("plan_brief", "execution_plan", "numbered_execution_plan", "steps")


def _extract_raw_plan(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    for key in _PLAN_PAYLOAD_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            return value, key
    for key in _PLAN_PAYLOAD_KEYS:
        if key in payload:
            return None, key
    return None, None


def _extract_plan_brief(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    merged = _top_level_plan_brief_fields(payload)
    for key in _PLAN_BRIEF_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            return {**dict(value), **merged}, key
        if isinstance(value, list):
            return {**merged, "execution_plan": list(value)}, key
    candidate = merged
    if candidate:
        return candidate, "top_level_plan_brief_fields"
    return {}, None


def _top_level_plan_brief_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(payload.get(key))
        for key in (
            "feedback_handling",
            "execution_plan",
            "numbered_execution_plan",
            "steps",
            "hard_constraints",
            "assumptions",
        )
        if payload.get(key) not in (None, "", [], {})
    }


def _bridge_plan_brief_to_construction_plan(
    request: PlanAgentRequest,
    plan_brief: dict[str, Any],
    *,
    workspace: Path | None,
) -> ChartConstructionPlan:
    scaffold = request.scaffold_plan or request.previous_plan
    if scaffold is None:
        raise ValueError("Freeform plan_brief requires scaffold_plan or previous_plan for typed pipeline compatibility.")
    brief_summary = _plan_brief_summary(plan_brief)
    decision = PlanDecision(
        decision_id=f"plan_agent.plan_brief.round_{request.round_index}",
        category="plan_brief",
        value={
            "summary": brief_summary,
            "execution_step_count": len(_execution_steps_from_brief(plan_brief)),
            "feedback_handling_count": len(_feedback_handling_from_brief(plan_brief)),
            "workspace_plan_brief_path": str(workspace / "plan_brief.json") if workspace is not None else None,
        },
        status="model_authored",
        rationale=str(plan_brief.get("rationale") or plan_brief.get("summary") or "PlanAgent supplied a freeform execution brief."),
    )
    constraints = tuple(
        dict.fromkeys(
            (
                *scaffold.constraints,
                "PlanAgent supplied a freeform execution brief; ExecutorAgent should treat it as implementation guidance, not verified completion evidence.",
            )
        )
    )
    execution_steps = tuple(_normalized_execution_steps_from_brief(plan_brief) or scaffold.execution_steps)
    return replace(
        scaffold,
        decisions=(*scaffold.decisions, decision),
        constraints=constraints,
        execution_steps=execution_steps,
    )


def _extract_feedback_handling(payload: dict[str, Any], plan_brief: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_items = _feedback_handling_from_brief(plan_brief)
    if not raw_items:
        raw_items = _feedback_handling_from_brief(payload)
    return _feedback_handling_from_items(raw_items)


def _feedback_handling_from_items(raw_items: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    items = []
    for raw in raw_items:
        issue_id = str(raw.get("issue_id") or raw.get("id") or "").strip()
        if not issue_id:
            continue
        decision = str(raw.get("decision") or raw.get("status") or "accepted").strip().lower()
        status = {
            "accept": "planned",
            "accepted": "planned",
            "address": "planned",
            "addressed": "planned",
            "plan": "planned",
            "planned": "planned",
            "defer": "deferred",
            "deferred": "deferred",
            "reject": "rejected",
            "rejected": "rejected",
        }.get(decision, decision or "planned")
        step_refs = raw.get("execution_step_ids") or raw.get("step_ids") or raw.get("steps") or raw.get("handled_by")
        if isinstance(step_refs, str):
            refs = [step_refs]
        elif isinstance(step_refs, (list, tuple)):
            refs = [str(item) for item in step_refs if str(item)]
        else:
            refs = []
        items.append(
            {
                "issue_id": issue_id,
                "status": status,
                "evidence_summary": str(raw.get("evidence") or raw.get("evidence_summary") or ""),
                "plan_change": str(
                    raw.get("instruction")
                    or raw.get("planned_action")
                    or raw.get("action")
                    or raw.get("proposal")
                    or raw.get("plan_change")
                    or ""
                ),
                "affected_plan_refs": refs,
                "rationale": str(raw.get("rationale") or raw.get("reason") or ""),
                "source": "model_plan_brief",
            }
        )
    return tuple(items)


def _feedback_handling_from_brief(plan_brief: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("feedback_handling", "feedback_decisions", "feedback"):
        value = plan_brief.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [dict(item) for item in value.values() if isinstance(item, dict)]
    return []


def _execution_steps_from_brief(plan_brief: dict[str, Any]) -> list[Any]:
    for key in ("execution_plan", "numbered_execution_plan", "steps", "plan_steps"):
        value = plan_brief.get(key)
        if isinstance(value, list):
            return list(value)
    return []


def _normalized_execution_steps_from_brief(plan_brief: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    for index, raw in enumerate(_execution_steps_from_brief(plan_brief), start=1):
        if isinstance(raw, dict):
            payload = dict(raw)
            step_id = str(payload.get("step_id") or payload.get("id") or f"step_{index}")
            payload["step_id"] = step_id
            payload.setdefault("source", "plan_brief")
            normalized.append(payload)
        elif str(raw).strip():
            normalized.append(
                {
                    "step_id": f"step_{index}",
                    "action": str(raw).strip(),
                    "source": "plan_brief",
                }
            )
    return normalized


def _plan_brief_summary(plan_brief: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": sorted(str(key) for key in plan_brief.keys()),
        "execution_steps": [
            _compact_step_for_summary(step)
            for step in _normalized_execution_steps_from_brief(plan_brief)[:8]
        ],
        "feedback_issue_ids": [
            str(item.get("issue_id") or item.get("id") or "")
            for item in _feedback_handling_from_brief(plan_brief)
            if str(item.get("issue_id") or item.get("id") or "")
        ],
        "hard_constraints": _string_items_from_any(plan_brief.get("hard_constraints"))[:8],
    }


def _compact_step_for_summary(step: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(step.get(key))
        for key in ("step_id", "goal", "action", "purpose")
        if step.get(key) not in (None, "", [], {})
    }


def _string_items_from_any(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items()]
    return [str(value)]


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
            "chart_types": payload.get("chart_types"),
            "panel_ids": payload.get("panel_ids"),
            "layer_ids": payload.get("layer_ids"),
            "layout_strategy": payload.get("layout_strategy"),
            "open_feedback": payload.get("open_feedback"),
        }.items()
        if value not in (None, "", [], {})
    }


def _next_memory(
    plan: ChartConstructionPlan,
    *,
    input_package: dict[str, Any],
    result: PlanAgentResult,
) -> dict[str, Any]:
    source_cards = input_package.get("source_cards") if isinstance(input_package.get("source_cards"), dict) else {}
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
        "chart_types": _plan_chart_types(plan),
        "panel_ids": [panel.panel_id for panel in plan.panels],
        "layer_ids": [layer.layer_id for panel in plan.panels for layer in panel.layers],
        "layout_strategy": plan.layout_strategy,
        "open_feedback": [
            dict(item)
            for item in result.feedback_handling
            if str(item.get("status") or "").lower() not in {"planned", "accepted"}
        ],
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


def _llm_trace_payload(trace: LLMCompletionTrace | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    usage = trace.usage
    return {
        "provider": trace.provider,
        "model": trace.model,
        "base_url": trace.base_url,
        "temperature": trace.temperature,
        "max_tokens": trace.max_tokens,
        "raw_text": trace.raw_text,
        "parsed_json": trace.parsed_json,
        "usage": None
        if usage is None
        else {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "raw": usage.raw,
        },
        "raw_response": trace.raw_response,
    }


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


def _feedback_items(feedback_bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(feedback_bundle, dict):
        return []
    return [dict(item) for item in list(feedback_bundle.get("feedback_items") or []) if isinstance(item, dict)]
