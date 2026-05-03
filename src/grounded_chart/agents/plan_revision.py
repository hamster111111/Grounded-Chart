from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from grounded_chart.construction_plan import ChartConstructionPlan, PlanDecision, VisualPanelPlan
from grounded_chart.artifact_workspace import PLAN_REVISION_AGENT_DIR
from grounded_chart.agents.feedback import plan_updates_from_feedback
from grounded_chart.agents.layout import LayoutCritique


@dataclass(frozen=True)
class PlanRevisionResult:
    """Result of a bounded visual-presentation plan revision."""

    revised_plan: ChartConstructionPlan
    applied: bool
    applied_updates: tuple[dict[str, Any], ...] = ()
    rejected_updates: tuple[dict[str, Any], ...] = ()
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "applied_updates": [dict(item) for item in self.applied_updates],
            "rejected_updates": [dict(item) for item in self.rejected_updates],
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
            "revised_plan": self.revised_plan.to_dict(),
        }


class PlanRevisionAgent(Protocol):
    def revise(
        self,
        *,
        construction_plan: ChartConstructionPlan,
        critique: LayoutCritique,
        query: str,
        generation_context: dict[str, Any] | None = None,
    ) -> PlanRevisionResult:
        """Return a revised plan whose edits are limited to bounded presentation fields."""


class LayoutOnlyPlanRevisionAgent:
    """Apply visual-presentation suggestions without changing chart semantics.

    This agent is deliberately conservative. It accepts normalized bounds and a
    small allowlist of placement/style fields. If the critic only reports
    a generic inset occlusion/anchor problem, it can apply a generic top-band
    redistribution for inset panels. This is not case-specific: it is keyed by
    panel role and layout contracts.
    """

    ALLOWED_PANEL_FIELDS = {"bounds", "placement_policy", "layout_notes", "avoid_occlusion", "style_policy"}
    ALLOWED_PLAN_FIELDS = {"figure_size"}
    ALLOWED_GLOBAL_FIELDS = {"placement", "style_policy", "avoid_occlusion"}

    def revise(
        self,
        *,
        construction_plan: ChartConstructionPlan,
        critique: LayoutCritique,
        query: str,
        generation_context: dict[str, Any] | None = None,
    ) -> PlanRevisionResult:
        explicit_updates = tuple(critique.recommended_plan_updates)
        feedback_updates = (
            plan_updates_from_feedback(critique.normalized_plan_feedback())
            if critique.plan_feedback or not explicit_updates
            else ()
        )
        plan_updates = _merge_plan_updates(explicit_updates, feedback_updates)
        if critique.ok or not plan_updates:
            return PlanRevisionResult(
                revised_plan=construction_plan,
                applied=False,
                rationale="No layout plan revision requested.",
                metadata={"revision_agent": "layout_only_v1"},
            )

        plan = construction_plan
        applied: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for raw_update in plan_updates:
            update = dict(raw_update)
            plan, accepted, reason = self._apply_update(plan, update)
            record = {**update, "decision": "accepted" if accepted else "rejected", "decision_reason": reason}
            if accepted:
                applied.append(record)
            else:
                rejected.append(record)

        if not applied and _needs_inset_redistribution(critique):
            plan, fallback_updates = _redistribute_inset_top_band(plan)
            applied.extend(fallback_updates)

        if applied:
            plan = _append_revision_decision(
                plan,
                critique=critique,
                applied_updates=tuple(applied),
            )
        return PlanRevisionResult(
            revised_plan=plan,
            applied=bool(applied),
            applied_updates=tuple(applied),
            rejected_updates=tuple(rejected),
            rationale=critique.diagnosis,
            metadata={
                "revision_agent": "layout_only_v1",
                "source_failed_contracts": list(critique.failed_contracts),
                "query_preview": str(query or "")[:240],
            },
        )

    def _apply_update(
        self,
        plan: ChartConstructionPlan,
        update: dict[str, Any],
    ) -> tuple[ChartConstructionPlan, bool, str]:
        target = str(update.get("target") or "").strip()
        field = str(update.get("field") or "").strip()
        operation = str(update.get("operation") or "set").strip()
        value = update.get("value")
        if not target:
            return plan, False, "missing target"
        if target == "figure_size" and field in {"", "value", "figure_size"}:
            target = "figure"
            field = "figure_size"
        if target == "global_elements" and "." in field:
            element_type, global_field = field.split(".", 1)
            if element_type:
                target = f"global.{element_type}"
                field = global_field
        if target.startswith("global_elements."):
            target = "global." + target.split(".", 1)[1]
        if _looks_semantic_field(field):
            return plan, False, f"semantic field is outside layout revision scope: {field}"
        if target.startswith("panel."):
            if field not in self.ALLOWED_PANEL_FIELDS:
                return plan, False, f"panel field not allowed: {field}"
            return self._apply_panel_update(plan, target=target, field=field, operation=operation, value=value)
        if target in {"construction_plan", "plan", "figure"}:
            if field not in self.ALLOWED_PLAN_FIELDS:
                return plan, False, f"plan field not allowed: {field}"
            if field == "figure_size":
                size = _valid_figure_size(value)
                if size is None:
                    return plan, False, "invalid figure_size"
                return replace(plan, figure_size=size), True, "figure_size updated"
        if target.startswith("global.") or target == "global_elements":
            if field not in self.ALLOWED_GLOBAL_FIELDS:
                return plan, False, f"global field not allowed: {field}"
            return self._apply_global_update(plan, target=target, field=field, operation=operation, value=value)
        return plan, False, f"unsupported target: {target}"

    def _apply_panel_update(
        self,
        plan: ChartConstructionPlan,
        *,
        target: str,
        field: str,
        operation: str,
        value: Any,
    ) -> tuple[ChartConstructionPlan, bool, str]:
        panels = list(plan.panels)
        panel_index = _find_panel_index(panels, target)
        if panel_index is None:
            return plan, False, f"panel target not found: {target}"
        panel = panels[panel_index]
        if field == "bounds":
            bounds = _valid_bounds(value)
            if bounds is None:
                return plan, False, "invalid normalized bounds"
            panels[panel_index] = replace(panel, bounds=bounds)
            return replace(plan, panels=tuple(panels)), True, "panel bounds updated"
        if field in {"placement_policy", "style_policy"}:
            current = dict(getattr(panel, field))
            patch = value if isinstance(value, dict) else {}
            if not patch:
                return plan, False, f"invalid {field} update"
            merged = {**current, **patch}
            panels[panel_index] = replace(panel, **{field: merged})
            return replace(plan, panels=tuple(panels)), True, f"panel {field} merged"
        if field in {"layout_notes", "avoid_occlusion"}:
            current_tuple = tuple(getattr(panel, field) or ())
            additions = _string_items(value)
            if not additions:
                return plan, False, f"invalid {field} update"
            if operation in {"append", "add", "merge", "set"}:
                merged_tuple = tuple(dict.fromkeys((*current_tuple, *additions)))
                panels[panel_index] = replace(panel, **{field: merged_tuple})
                return replace(plan, panels=tuple(panels)), True, f"panel {field} merged"
        return plan, False, f"unsupported panel operation: {operation}"

    def _apply_global_update(
        self,
        plan: ChartConstructionPlan,
        *,
        target: str,
        field: str,
        operation: str,
        value: Any,
    ) -> tuple[ChartConstructionPlan, bool, str]:
        element_type = target.split(".", 1)[1] if target.startswith("global.") and "." in target else ""
        if target == "global_elements" and "." in field:
            element_type, field = field.split(".", 1)
        elements = []
        updated = False
        for element in plan.global_elements:
            payload = dict(element)
            if not element_type or str(payload.get("type") or "") == element_type:
                if field in {"style_policy"}:
                    patch = value if isinstance(value, dict) else {}
                    if not patch:
                        return plan, False, "invalid global style_policy update"
                    payload[field] = {**dict(payload.get(field) or {}), **patch}
                elif field == "avoid_occlusion":
                    payload[field] = list(dict.fromkeys([*list(payload.get(field) or []), *_string_items(value)]))
                else:
                    payload[field] = value
                updated = True
            elements.append(payload)
        if not updated:
            return plan, False, f"global target not found: {target}"
        return replace(plan, global_elements=tuple(elements)), True, f"global {field} updated"


def write_plan_revision_artifacts(
    *,
    output_root: str | Path,
    round_index: int,
    critique: LayoutCritique,
    revision: PlanRevisionResult,
) -> dict[str, str]:
    agent_dir = Path(output_root).resolve() / PLAN_REVISION_AGENT_DIR / f"round_{round_index}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    critique_path = agent_dir / "layout_critique.json"
    revision_path = agent_dir / "plan_revision.json"
    revised_plan_path = agent_dir / "revised_plan.json"
    critique_path.write_text(json.dumps(critique.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    revision_path.write_text(json.dumps(revision.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    revised_plan_path.write_text(json.dumps(revision.revised_plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "agent_dir": str(agent_dir),
        "layout_critique_path": str(critique_path),
        "plan_revision_path": str(revision_path),
        "revised_plan_path": str(revised_plan_path),
    }


def _merge_plan_updates(
    explicit_updates: tuple[dict[str, Any], ...],
    feedback_updates: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for update in (*explicit_updates, *feedback_updates):
        if not isinstance(update, dict):
            continue
        key = _update_identity(update)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(update))
    return tuple(merged)


def _update_identity(update: dict[str, Any]) -> tuple[str, str, str, str]:
    value = update.get("value")
    try:
        value_key = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        value_key = str(value)
    return (
        str(update.get("target") or ""),
        str(update.get("field") or ""),
        str(update.get("operation") or ""),
        value_key,
    )


def _append_revision_decision(
    plan: ChartConstructionPlan,
    *,
    critique: LayoutCritique,
    applied_updates: tuple[dict[str, Any], ...],
) -> ChartConstructionPlan:
    decision = PlanDecision(
        decision_id=f"layout.revision.{len(plan.decisions) + 1}",
        category="layout",
        value={
            "failed_contracts": list(critique.failed_contracts),
            "applied_update_count": len(applied_updates),
        },
        status="inferred",
        rationale=critique.diagnosis or "Layout critic requested bounded layout-only plan revision.",
    )
    constraints = tuple(
        dict.fromkeys(
            (
                *plan.constraints,
                "Visual-presentation revisions may only change figure size, panel bounds, placement policies, notes, style policies, and global placement/style.",
                "Visual-presentation revisions must not change data sources, chart types, plotted values, required labels, or source files.",
            )
        )
    )
    return replace(plan, decisions=(*plan.decisions, decision), constraints=constraints)


def _needs_inset_redistribution(critique: LayoutCritique) -> bool:
    text = " ".join([*critique.failed_contracts, critique.diagnosis]).lower()
    return "inset" in text and any(keyword in text for keyword in ("overlap", "occlusion", "anchor", "collision", "crowd"))


def _redistribute_inset_top_band(plan: ChartConstructionPlan) -> tuple[ChartConstructionPlan, list[dict[str, Any]]]:
    panels = list(plan.panels)
    inset_indices = [
        index
        for index, panel in enumerate(panels)
        if "inset" in str(panel.role).lower() or "pie" in str(panel.role).lower()
    ]
    if not inset_indices:
        return plan, []
    count = len(inset_indices)
    width = min(0.14, max(0.08, 0.62 / max(count, 1)))
    height = width
    left_margin = 0.12
    usable = 0.68
    if count == 1:
        lefts = [left_margin + usable / 2 - width / 2]
    else:
        step = (usable - width) / max(count - 1, 1)
        lefts = [left_margin + step * index for index in range(count)]
    bottom = 0.845 if count <= 4 else 0.82
    updates: list[dict[str, Any]] = []
    for slot, panel_index in enumerate(inset_indices):
        panel = panels[panel_index]
        bounds = (_round(lefts[slot]), _round(bottom), _round(width), _round(height))
        policy = {
            **dict(panel.placement_policy),
            "layout_revision": "redistributed_top_band",
            "avoid_overlap_with_main_axes": True,
            "allow_anchor_line_or_label_if_exact_x_alignment_is_not_readable": True,
        }
        notes = tuple(
            dict.fromkeys(
                (
                    *panel.layout_notes,
                    "Layout revision: keep inset readable in top band; use a connector/label if exact anchor overlap would occlude the main chart.",
                )
            )
        )
        panels[panel_index] = replace(panel, bounds=bounds, placement_policy=policy, layout_notes=notes)
        updates.append(
            {
                "target": panel.panel_id,
                "field": "bounds",
                "operation": "fallback_redistribute_top_band",
                "value": list(bounds),
                "reason": "Generic fallback for inset anchor/occlusion layout critique.",
                "decision": "accepted",
                "decision_reason": "fallback inset redistribution",
            }
        )
    return replace(plan, panels=tuple(panels)), updates


def _find_panel_index(panels: list[VisualPanelPlan], target: str) -> int | None:
    normalized_target = target.strip()
    for index, panel in enumerate(panels):
        if panel.panel_id == normalized_target:
            return index
    suffix = normalized_target.split(".", 1)[1] if normalized_target.startswith("panel.") else normalized_target
    for index, panel in enumerate(panels):
        if panel.panel_id == f"panel.{suffix}" or panel.panel_id.endswith(suffix):
            return index
    return None


def _valid_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, bottom, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if left < 0 or bottom < 0 or width <= 0 or height <= 0:
        return None
    if left + width > 1.02 or bottom + height > 1.02:
        return None
    return (_round(left), _round(bottom), _round(width), _round(height))


def _valid_figure_size(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (3 <= width <= 30 and 3 <= height <= 30):
        return None
    return (_round(width), _round(height))


def _string_items(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _looks_semantic_field(field: str) -> bool:
    layout_safe_fields = {
        "bounds",
        "figure_size",
        "placement_policy",
        "layout_notes",
        "avoid_occlusion",
        "style_policy",
        "placement",
    }
    if str(field or "").lower() in layout_safe_fields:
        return False
    semantic_tokens = {
        "data_source",
        "data_transform",
        "chart_type",
        "encoding",
        "x",
        "y",
        "axis",
        "value",
        "values",
        "label",
        "labels",
        "source_file",
        "semantic_modifiers",
    }
    normalized = str(field or "").lower()
    return any(token in normalized for token in semantic_tokens)


def _round(value: float) -> float:
    return round(float(value), 4)
