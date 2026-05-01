from __future__ import annotations

import re
from typing import Any


PLAN_AGENT = "PlanAgent"


def normalize_plan_feedback_items(
    raw_items: Any,
    *,
    source_agent: str,
    default_confidence: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_items, (list, tuple)):
        return ()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        feedback = _normalize_existing_feedback(
            item,
            source_agent=source_agent,
            index=index,
            default_confidence=default_confidence,
        )
        if feedback:
            normalized.append(feedback)
    return tuple(normalized)


def normalize_layout_plan_feedback(
    *,
    source_agent: str,
    failed_contracts: tuple[str, ...],
    diagnosis: str,
    recommended_plan_updates: tuple[dict[str, Any], ...],
    raw_plan_feedback: Any = None,
    confidence: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    explicit = normalize_plan_feedback_items(
        raw_plan_feedback,
        source_agent=source_agent,
        default_confidence=confidence,
    )
    if explicit:
        return explicit
    feedback: list[dict[str, Any]] = []
    for index, update in enumerate(recommended_plan_updates, start=1):
        if not isinstance(update, dict):
            continue
        target_ref = str(update.get("target") or "construction_plan").strip()
        field = str(update.get("field") or "").strip()
        value = update.get("value")
        reason = str(update.get("reason") or diagnosis or "LayoutAgent suggested a plan-level layout revision.").strip()
        issue_type = _issue_type_for_layout_update(update, failed_contracts)
        sanitized_update = _sanitize_layout_legacy_update(update, reason=reason)
        target_ref = str(sanitized_update.get("target") or target_ref)
        field = str(sanitized_update.get("field") or field)
        value = sanitized_update.get("value")
        proposal = _layout_update_proposal(target_ref=target_ref, field=field, value=value, reason=reason)
        feedback.append(
            _feedback_issue(
                source_agent=source_agent,
                issue_id=f"{_slug(source_agent)}.{_slug(issue_type)}.{index}",
                issue_type=issue_type,
                severity=str(update.get("severity") or "warning"),
                evidence=reason,
                affected_region=target_ref,
                related_plan_ref=target_ref,
                suggested_plan_action={
                    "target_agent": PLAN_AGENT,
                    "action_type": "revise_layout_contract",
                    "target_ref": target_ref,
                    "proposal": proposal,
                    "legacy_plan_update": sanitized_update,
                },
                confidence=confidence,
            )
        )
    if feedback:
        return tuple(feedback)
    if not failed_contracts and not diagnosis:
        return ()
    issue_type = failed_contracts[0] if failed_contracts else "layout.issue"
    return (
        _feedback_issue(
            source_agent=source_agent,
            issue_id=f"{_slug(source_agent)}.{_slug(issue_type)}.1",
            issue_type=issue_type,
            severity="warning",
            evidence=diagnosis or "LayoutAgent reported a layout issue.",
            affected_region="figure",
            related_plan_ref="construction_plan",
            suggested_plan_action={
                "target_agent": PLAN_AGENT,
                "action_type": "revise_layout_contract",
                "target_ref": "construction_plan",
                "proposal": diagnosis or "Review and revise the layout plan.",
            },
            confidence=confidence,
        ),
    )


def normalize_figure_plan_feedback(
    *,
    summary: str,
    issue_groups: dict[str, tuple[dict[str, Any], ...]],
    recommended_plan_notes: tuple[dict[str, Any], ...],
    raw_plan_feedback: Any = None,
    confidence: float = 0.0,
    source_agent: str = "FigureReaderAgent",
) -> tuple[dict[str, Any], ...]:
    explicit = normalize_plan_feedback_items(
        raw_plan_feedback,
        source_agent=source_agent,
        default_confidence=confidence,
    )
    if explicit:
        return explicit
    feedback: list[dict[str, Any]] = []
    index = 1
    for family, issues in issue_groups.items():
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            issue_type = str(issue.get("issue_type") or family or "figure_audit_issue").strip()
            recommendation = str(issue.get("recommendation") or issue.get("evidence") or "").strip()
            evidence = str(issue.get("evidence") or recommendation or summary or "").strip()
            related_plan_ref = str(issue.get("related_plan_ref") or "").strip()
            feedback.append(
                _feedback_issue(
                    source_agent=source_agent,
                    issue_id=f"{_slug(source_agent)}.{_slug(issue_type)}.{index}",
                    issue_type=issue_type,
                    severity=str(issue.get("severity") or "warning"),
                    evidence=evidence,
                    affected_region=str(issue.get("affected_region") or "").strip(),
                    related_plan_ref=related_plan_ref,
                    suggested_plan_action={
                        "target_agent": PLAN_AGENT,
                        "action_type": _figure_action_type(family=family, issue_type=issue_type),
                        "target_ref": related_plan_ref or _default_target_ref_for_family(family),
                        "proposal": recommendation or evidence or "Revise the construction plan to address this visual audit issue.",
                    },
                    confidence=confidence,
                )
            )
            index += 1
    for note in recommended_plan_notes:
        recommendation = str(note.get("recommendation") or note.get("note") or note.get("evidence") or "").strip()
        if not recommendation:
            continue
        original_target_agent = str(note.get("original_target_agent") or note.get("target_agent") or "").strip()
        action: dict[str, Any] = {
            "target_agent": PLAN_AGENT,
            "action_type": "revise_plan_contract",
            "target_ref": str(note.get("related_plan_ref") or "construction_plan").strip(),
            "proposal": recommendation,
        }
        if original_target_agent and original_target_agent != PLAN_AGENT:
            action["original_target_agent"] = original_target_agent
        feedback.append(
            _feedback_issue(
                source_agent=source_agent,
                issue_id=f"{_slug(source_agent)}.plan_note.{index}",
                issue_type=str(note.get("issue_type") or "figure_audit_note"),
                severity=str(note.get("severity") or "warning"),
                evidence=str(note.get("evidence") or recommendation),
                affected_region=str(note.get("affected_region") or "").strip(),
                related_plan_ref=str(note.get("related_plan_ref") or "").strip(),
                suggested_plan_action=action,
                confidence=confidence,
            )
        )
        index += 1
    return tuple(feedback)


def plan_updates_from_feedback(feedback_items: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    updates: list[dict[str, Any]] = []
    for item in feedback_items:
        action = item.get("suggested_plan_action")
        if not isinstance(action, dict):
            continue
        legacy_update = action.get("legacy_plan_update")
        if isinstance(legacy_update, dict):
            update = dict(legacy_update)
            update["source"] = item.get("source_agent") or update.get("source") or "plan_feedback"
            update["plan_feedback_id"] = item.get("issue_id")
            update["suggested_plan_action"] = {
                key: value
                for key, value in action.items()
                if key != "legacy_plan_update"
            }
            updates.append(update)
            continue
        proposal = str(action.get("proposal") or item.get("evidence") or "").strip()
        if not proposal:
            continue
        target_ref = str(action.get("target_ref") or item.get("related_plan_ref") or "").strip()
        target = _panel_target_from_ref(target_ref)
        updates.append(
            {
                "target": target,
                "field": "layout_notes",
                "operation": "append",
                "value": [proposal],
                "reason": str(item.get("evidence") or proposal),
                "source": item.get("source_agent") or "plan_feedback",
                "plan_feedback_id": item.get("issue_id"),
                "suggested_plan_action": {
                    **action,
                    "target_agent": PLAN_AGENT,
                },
            }
        )
    return tuple(updates)


def _normalize_existing_feedback(
    item: dict[str, Any],
    *,
    source_agent: str,
    index: int,
    default_confidence: float,
) -> dict[str, Any]:
    action = item.get("suggested_plan_action")
    if not isinstance(action, dict):
        action = {
            "target_agent": PLAN_AGENT,
            "action_type": str(item.get("action_type") or "revise_plan_contract"),
            "target_ref": str(item.get("target_ref") or item.get("related_plan_ref") or "construction_plan"),
            "proposal": str(
                item.get("proposal")
                or item.get("recommendation")
                or item.get("note")
                or item.get("evidence")
                or ""
            ),
        }
    else:
        action = dict(action)
    original_target_agent = str(action.get("target_agent") or "").strip()
    action["target_agent"] = PLAN_AGENT
    if original_target_agent and original_target_agent != PLAN_AGENT:
        action["original_target_agent"] = original_target_agent
    issue_type = str(item.get("issue_type") or item.get("type") or action.get("action_type") or "plan_feedback").strip()
    return _feedback_issue(
        source_agent=str(item.get("source_agent") or source_agent),
        issue_id=str(item.get("issue_id") or f"{_slug(source_agent)}.{_slug(issue_type)}.{index}"),
        issue_type=issue_type,
        severity=str(item.get("severity") or "warning"),
        evidence=str(item.get("evidence") or item.get("reason") or action.get("proposal") or ""),
        affected_region=str(item.get("affected_region") or item.get("region") or ""),
        related_plan_ref=str(item.get("related_plan_ref") or action.get("target_ref") or ""),
        suggested_plan_action=action,
        confidence=_coerce_confidence(item.get("confidence"), default_confidence),
    )


def _feedback_issue(
    *,
    source_agent: str,
    issue_id: str,
    issue_type: str,
    severity: str,
    evidence: str,
    affected_region: str,
    related_plan_ref: str,
    suggested_plan_action: dict[str, Any],
    confidence: float,
) -> dict[str, Any]:
    action = dict(suggested_plan_action)
    original_target_agent = str(action.get("target_agent") or "").strip()
    action["target_agent"] = PLAN_AGENT
    if original_target_agent and original_target_agent != PLAN_AGENT:
        action["original_target_agent"] = original_target_agent
    return {
        "issue_id": issue_id,
        "source_agent": source_agent,
        "issue_type": issue_type,
        "severity": _normalize_severity(severity),
        "evidence": evidence,
        "affected_region": affected_region,
        "related_plan_ref": related_plan_ref,
        "suggested_plan_action": action,
        "confidence": _coerce_confidence(confidence, 0.0),
    }


def _issue_type_for_layout_update(update: dict[str, Any], failed_contracts: tuple[str, ...]) -> str:
    raw = str(update.get("issue_type") or "").strip()
    if raw:
        return raw
    reason = str(update.get("reason") or "").lower()
    target = str(update.get("target") or "").lower()
    for contract in failed_contracts:
        normalized = str(contract).lower()
        if "inset" in reason and "inset" in normalized:
            return str(contract)
        if "legend" in reason and "legend" in normalized:
            return str(contract)
        if "crowd" in reason and "crowd" in normalized:
            return str(contract)
        if "panel" in target and "panel" in normalized:
            return str(contract)
    return failed_contracts[0] if failed_contracts else "layout.plan_update"


def _layout_update_proposal(*, target_ref: str, field: str, value: Any, reason: str) -> str:
    if field and field != "layout_notes":
        return f"Revise `{target_ref}.{field}` to `{value}` at the plan level. Rationale: {reason}"
    return f"Revise the layout/composition contract for `{target_ref}` at the plan level. Rationale: {reason}"


def _sanitize_layout_legacy_update(update: dict[str, Any], *, reason: str) -> dict[str, Any]:
    field = str(update.get("field") or "").strip()
    target = str(update.get("target") or "panel.main").strip()
    if field == "bounds":
        return {
            "target": target,
            "field": "layout_notes",
            "operation": "append",
            "value": [
                (
                    "LayoutAgent reported a coordinate-level bounds issue, but coordinate selection is delegated "
                    f"to ExecutorAgent during figure execution. Preserve the issue as layout intent: {reason}"
                )
            ],
            "reason": reason,
            "source": update.get("source") or "LayoutAgent",
            "legacy_field_suppressed": "bounds",
        }
    return dict(update)


def _figure_action_type(*, family: str, issue_type: str) -> str:
    text = f"{family} {issue_type}".lower()
    if "scale" in text or "axis" in text:
        return "enforce_existing_requirement"
    if "label" in text or "legend" in text or "mapping" in text:
        return "clarify_visual_semantics"
    if "clipping" in text or "occlusion" in text or "crowd" in text:
        return "revise_visual_layout_contract"
    if "series" in text or "encoding" in text:
        return "clarify_visual_channel_contract"
    return "revise_plan_contract"


def _default_target_ref_for_family(family: str) -> str:
    if "layout" in family or "readability" in family:
        return "construction_plan"
    if "encoding" in family:
        return "construction_plan.visual_channel_plan"
    return "construction_plan"


def _panel_target_from_ref(ref: str) -> str:
    match = re.search(r"panel\.[A-Za-z0-9_.-]+", ref)
    if match:
        return match.group(0)
    return "panel.main"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return slug or "issue"


def _normalize_severity(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"info", "warning", "error"}:
        return normalized
    if normalized in {"warn", "medium"}:
        return "warning"
    if normalized in {"critical", "high", "severe"}:
        return "error"
    if normalized in {"low", "minor"}:
        return "info"
    return "warning"


def _coerce_confidence(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return max(0.0, min(1.0, numeric))
