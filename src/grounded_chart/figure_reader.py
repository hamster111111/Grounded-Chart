from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from grounded_chart.llm import LLMClient, LLMCompletionTrace, LLMUsage
from grounded_chart.plan_feedback import normalize_figure_plan_feedback


@dataclass(frozen=True)
class FigureAudit:
    """Vision-side audit of whether a rendered chart is understandable.

    This is diagnostic evidence for replanning. It is deliberately weaker than a
    deterministic verifier: the audit may flag visible confusion, missing visual
    semantics, or suspicious artifacts, but it must not become the numerical
    truth authority for the chart.
    """

    ok: bool
    summary: str = ""
    readability_issues: tuple[dict[str, Any], ...] = ()
    encoding_confusions: tuple[dict[str, Any], ...] = ()
    data_semantic_warnings: tuple[dict[str, Any], ...] = ()
    suspicious_artifacts: tuple[dict[str, Any], ...] = ()
    unclear_regions: tuple[dict[str, Any], ...] = ()
    recommended_plan_notes: tuple[dict[str, Any], ...] = ()
    plan_feedback: tuple[dict[str, Any], ...] = ()
    confidence: float = 0.0
    llm_trace: LLMCompletionTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "readability_issues": [_jsonable(item) for item in self.readability_issues],
            "encoding_confusions": [_jsonable(item) for item in self.encoding_confusions],
            "data_semantic_warnings": [_jsonable(item) for item in self.data_semantic_warnings],
            "suspicious_artifacts": [_jsonable(item) for item in self.suspicious_artifacts],
            "unclear_regions": [_jsonable(item) for item in self.unclear_regions],
            "recommended_plan_notes": [_jsonable(item) for item in self.recommended_plan_notes],
            "plan_feedback": [_jsonable(item) for item in self.normalized_plan_feedback()],
            "confidence": self.confidence,
            "llm_trace": _llm_trace_to_dict(self.llm_trace),
            "metadata": _jsonable(self.metadata),
        }

    def normalized_plan_feedback(self) -> tuple[dict[str, Any], ...]:
        return normalize_figure_plan_feedback(
            summary=self.summary,
            issue_groups={
                "readability_issues": self.readability_issues,
                "encoding_confusions": self.encoding_confusions,
                "data_semantic_warnings": self.data_semantic_warnings,
                "suspicious_artifacts": self.suspicious_artifacts,
                "unclear_regions": self.unclear_regions,
            },
            recommended_plan_notes=self.recommended_plan_notes,
            raw_plan_feedback=self.plan_feedback,
            confidence=self.confidence,
            source_agent=str(self.metadata.get("agent_name") or "FigureReaderAgent"),
        )


class FigureReaderAgent(Protocol):
    def audit(
        self,
        *,
        query: str,
        construction_plan: dict[str, Any],
        generated_code: str,
        render_result: Any,
        actual_figure: Any | None = None,
        artifact_workspace_report: Any | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> FigureAudit:
        """Return visual-semantic audit evidence for a rendered chart."""


class VLMFigureReaderAgent:
    """Vision-backed FigureReaderAgent for semantic readability audits."""

    def __init__(
        self,
        client: LLMClient,
        *,
        agent_name: str = "vlm_figure_reader_v1",
        max_tokens: int | None = 2048,
    ) -> None:
        self.client = client
        self.agent_name = agent_name
        self.max_tokens = max_tokens

    def audit(
        self,
        *,
        query: str,
        construction_plan: dict[str, Any],
        generated_code: str,
        render_result: Any,
        actual_figure: Any | None = None,
        artifact_workspace_report: Any | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> FigureAudit:
        image_path = getattr(render_result, "image_path", None)
        if image_path is None:
            return FigureAudit(
                ok=False,
                summary="FigureReaderAgent requires a rendered image but render_result.image_path is missing.",
                suspicious_artifacts=(
                    {
                        "issue_type": "image_missing",
                        "severity": "error",
                        "evidence": "render_result.image_path is missing",
                    },
                ),
                confidence=1.0,
                metadata={
                    "agent_name": self.agent_name,
                    "mode": "vlm",
                    "image_pixels_available": False,
                },
            )
        complete_with_image = getattr(self.client, "complete_json_with_image_trace", None)
        if not callable(complete_with_image):
            raise TypeError("VLMFigureReaderAgent requires an LLM client with complete_json_with_image_trace().")
        try:
            result = complete_with_image(
                system_prompt=_system_prompt(),
                user_prompt=_user_prompt(
                    query=query,
                    construction_plan=construction_plan,
                    generated_code=generated_code,
                    render_result=render_result,
                    actual_figure=actual_figure,
                    artifact_workspace_report=artifact_workspace_report,
                    generation_context=generation_context,
                ),
                image_path=image_path,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                "VLMFigureReaderAgent image audit failed. Configure a vision-capable model for figure reading "
                "or run without --enable-figure-reader."
            ) from exc
        payload = result.payload
        return FigureAudit(
            ok=bool(payload.get("ok", False)),
            summary=str(payload.get("summary") or ""),
            readability_issues=_dict_tuple(payload.get("readability_issues")),
            encoding_confusions=_dict_tuple(payload.get("encoding_confusions")),
            data_semantic_warnings=_dict_tuple(payload.get("data_semantic_warnings")),
            suspicious_artifacts=_dict_tuple(payload.get("suspicious_artifacts")),
            unclear_regions=_dict_tuple(payload.get("unclear_regions")),
            recommended_plan_notes=_normalize_note_tuple(payload.get("recommended_plan_notes")),
            plan_feedback=tuple(
                dict(item)
                for item in list(payload.get("plan_feedback") or payload.get("findings") or [])
                if isinstance(item, dict)
            ),
            confidence=_confidence(payload.get("confidence")),
            llm_trace=result.trace,
            metadata={
                "agent_name": self.agent_name,
                "mode": "vlm",
                "image_pixels_available": True,
                "image_path": str(image_path),
                "scope": "visual_semantic_readability_audit",
            },
        )


def write_figure_audit_artifact(
    *,
    output_root: str | Path,
    round_index: int,
    audit: FigureAudit,
) -> str:
    repair_dir = Path(output_root).resolve() / "repair" / f"layout_round_{round_index}"
    repair_dir.mkdir(parents=True, exist_ok=True)
    audit_path = repair_dir / "figure_audit.json"
    audit_path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(audit_path)


def figure_audit_plan_feedback(audit: FigureAudit | None) -> dict[str, Any] | None:
    if audit is None:
        return None
    normalized_notes = normalized_figure_audit_notes(audit)
    plan_feedback = audit.normalized_plan_feedback()
    issue_count = (
        len(audit.readability_issues)
        + len(audit.encoding_confusions)
        + len(audit.data_semantic_warnings)
        + len(audit.suspicious_artifacts)
        + len(audit.unclear_regions)
    )
    return {
        "ok": audit.ok,
        "summary": audit.summary,
        "confidence": audit.confidence,
        "issue_count": issue_count,
        "readability_issues": [_feedback_item(item) for item in audit.readability_issues],
        "encoding_confusions": [_feedback_item(item) for item in audit.encoding_confusions],
        "data_semantic_warnings": [_feedback_item(item) for item in audit.data_semantic_warnings],
        "suspicious_artifacts": [_feedback_item(item) for item in audit.suspicious_artifacts],
        "unclear_regions": [_feedback_item(item) for item in audit.unclear_regions],
        "recommended_plan_notes": [_feedback_item(item) for item in normalized_notes],
        "plan_feedback": [_jsonable(item) for item in plan_feedback],
        "scope_rules": [
            "Treat this audit as visual evidence, not as a numerical oracle.",
            "Send all proposed actions to PlanAgent for adjudication before ExecutorAgent sees them.",
            "Use it to clarify layout, legend, visual channel, label, and readability intent at the plan level.",
            "Do not change source data, plotted values, chart types, or required labels unless a deterministic plan or artifact already supports the change.",
        ],
    }


def normalized_figure_audit_notes(audit: FigureAudit | None) -> tuple[dict[str, Any], ...]:
    if audit is None:
        return ()
    return _normalize_note_tuple(audit.recommended_plan_notes)


def _system_prompt() -> str:
    return (
        "You are FigureReaderAgent for a grounded chart-generation pipeline. "
        "Inspect the rendered chart image as the primary evidence, then use the request, construction plan, figure trace, and artifact workspace as constraints. "
        "Your job is to report whether the chart is understandable and visually faithful at a semantic level: readable marks, distinguishable encodings, non-confusing legends, visible annotations, sensible composition, and suspicious visual artifacts. "
        "You may flag that a visible trend, mark geometry, or color/channel mapping appears suspicious, but you must not assert exact numeric data is wrong unless the provided deterministic artifacts or figure trace support that claim. "
        "Do not propose code edits. Do not invent missing data. Do not rewrite the task. "
        "Return only a JSON object with keys: ok, summary, readability_issues, encoding_confusions, data_semantic_warnings, suspicious_artifacts, unclear_regions, plan_feedback, recommended_plan_notes, confidence. "
        "Each issue or note should be an object with issue_type, severity, evidence, affected_region, related_plan_ref, and recommendation when applicable. "
        "plan_feedback is preferred and must contain one item per issue with issue_id, source_agent='FigureReaderAgent', issue_type, severity, evidence, affected_region, related_plan_ref, and suggested_plan_action. "
        "Every suggested_plan_action must target PlanAgent, not ExecutorAgent, and must describe a plan-level contract revision or clarification. "
        "recommended_plan_notes is a backward-compatible optional field; if used, it must still be intended for PlanAgent adjudication, not direct code patches. "
        "Use severity values info, warning, or error. If the image is understandable and no evidence-backed concern exists, return ok=true and empty arrays."
    )


def _user_prompt(
    *,
    query: str,
    construction_plan: dict[str, Any],
    generated_code: str,
    render_result: Any,
    actual_figure: Any | None,
    artifact_workspace_report: Any | None,
    generation_context: dict[str, Any] | None,
) -> str:
    payload = {
        "query": query,
        "generation_context": _compact_generation_context(generation_context or {}),
        "construction_plan_visual_semantics": _compact_plan_visual_semantics(construction_plan),
        "artifact_workspace": _compact_artifact_workspace(artifact_workspace_report),
        "render_result": _render_payload(render_result),
        "actual_figure_trace": _figure_payload(actual_figure),
        "generated_code_visual_excerpt": _code_visual_excerpt(generated_code),
        "audit_scope": {
            "allowed": [
                "visual readability",
                "legend/channel ambiguity",
                "mark overlap or clutter",
                "missing or unreadable labels/annotations",
                "suspicious visible data geometry when constrained by artifacts",
                "strange artifacts, clipping, occlusion, or misleading composition",
            ],
            "forbidden": [
                "exact numeric judging without artifact evidence",
                "source data edits",
                "direct code patch proposals",
                "changing required chart type or required labels by preference",
            ],
        },
    }
    return "Audit the rendered chart for visual-semantic understandability.\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _compact_plan_visual_semantics(plan: dict[str, Any]) -> dict[str, Any]:
    panels = []
    for panel in list(plan.get("panels") or []):
        if not isinstance(panel, dict):
            continue
        layers = []
        for layer in list(panel.get("layers") or []):
            if not isinstance(layer, dict):
                continue
            layers.append(
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
                    "style_policy": layer.get("style_policy"),
                    "components": layer.get("components"),
                }
            )
        panels.append(
            {
                "panel_id": panel.get("panel_id"),
                "role": panel.get("role"),
                "bounds": panel.get("bounds"),
                "axes": panel.get("axes"),
                "anchor": panel.get("anchor"),
                "layout_notes": panel.get("layout_notes"),
                "placement_policy": panel.get("placement_policy"),
                "style_policy": panel.get("style_policy"),
                "layers": layers,
            }
        )
    return {
        "plan_type": plan.get("plan_type"),
        "layout_strategy": plan.get("layout_strategy"),
        "figure_size": plan.get("figure_size"),
        "panels": panels,
        "global_elements": plan.get("global_elements"),
        "constraints": plan.get("constraints"),
        "style_policy": plan.get("style_policy"),
    }


def _compact_artifact_workspace(report: Any | None) -> dict[str, Any] | None:
    if report is None:
        return None
    payload = report.to_dict() if hasattr(report, "to_dict") else _jsonable(report)
    if not isinstance(payload, dict):
        return {"raw": payload}
    artifacts = []
    for item in list(payload.get("artifacts") or []):
        if not isinstance(item, dict):
            continue
        artifacts.append(
            {
                "name": item.get("name"),
                "kind": item.get("kind"),
                "relative_path": item.get("relative_path"),
                "description": item.get("description"),
                "columns": item.get("columns"),
                "row_count": item.get("row_count"),
                "preview": item.get("preview"),
            }
        )
    return {
        "ok": payload.get("ok"),
        "execution_dir": payload.get("execution_dir"),
        "artifacts": artifacts[:12],
        "metadata": payload.get("metadata"),
    }


def _compact_generation_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in context.items()
        if key in {"simple_instruction", "expert_instruction", "metadata", "source", "native_id", "generation_mode"}
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
    }


def _code_visual_excerpt(code: str, *, max_chars: int = 12000) -> str:
    lines = str(code or "").splitlines()
    keywords = (
        "bar",
        "fill_between",
        "plot(",
        "scatter",
        "pie",
        "hist",
        "imshow",
        "legend",
        "label",
        "color",
        "cmap",
        "alpha",
        "hatch",
        "annotate",
        "text(",
        "set_title",
        "set_xlabel",
        "set_ylabel",
        "set_xlim",
        "set_ylim",
        "add_axes",
        "twinx",
    )
    selected: list[str] = []
    for index, line in enumerate(lines, start=1):
        if any(keyword in line for keyword in keywords):
            selected.append(f"{index}: {line}")
    excerpt = "\n".join(selected) or str(code or "")
    if len(excerpt) > max_chars:
        return excerpt[:max_chars] + "\n... [truncated]"
    return excerpt


def _dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _normalize_note_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    notes = []
    for item in value:
        if isinstance(item, str):
            normalized = _normalize_note_dict({"note": item})
        elif isinstance(item, dict):
            normalized = _normalize_note_dict(item)
        else:
            continue
        if normalized:
            notes.append(normalized)
    return tuple(notes)


def _normalize_note_dict(item: dict[str, Any]) -> dict[str, Any]:
    recommendation = _first_text(
        item,
        "recommendation",
        "note",
        "suggestion",
        "action",
        "message",
        "description",
        "fix",
    )
    evidence = _first_text(item, "evidence", "reason", "rationale", "observation")
    related_plan_ref = _first_text(
        item,
        "related_plan_ref",
        "plan_ref",
        "plan_target",
        "target_ref",
    )
    if not related_plan_ref:
        raw_target = _first_text(item, "target")
        if raw_target and not raw_target.lower().endswith("agent"):
            related_plan_ref = raw_target
    if not recommendation and evidence:
        recommendation = evidence
    if not recommendation:
        return {}
    normalized = {
        "issue_type": _first_text(item, "issue_type", "type", "category") or "figure_audit_note",
        "severity": _normalize_severity(_first_text(item, "severity", "level")),
        "evidence": evidence or recommendation,
        "affected_region": _first_text(item, "affected_region", "region", "location"),
        "related_plan_ref": related_plan_ref,
        "recommendation": recommendation,
        "target_agent": _first_text(item, "target_agent", "agent", "owner"),
        "source": _first_text(item, "source") or "figure_audit",
    }
    return {key: value for key, value in normalized.items() if value not in {None, ""}}


def _first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_severity(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"info", "warning", "error"}:
        return normalized
    if normalized in {"warn", "medium"}:
        return "warning"
    if normalized in {"critical", "high", "severe"}:
        return "error"
    return "warning"


def _feedback_item(item: dict[str, Any], *, max_text: int = 700) -> dict[str, Any]:
    keys = (
        "issue_type",
        "severity",
        "evidence",
        "affected_region",
        "related_plan_ref",
        "recommendation",
        "target_agent",
        "source",
    )
    payload = {key: _jsonable(item.get(key)) for key in keys if item.get(key) is not None}
    for key, value in list(payload.items()):
        if isinstance(value, str) and len(value) > max_text:
            payload[key] = value[:max_text] + "... [truncated]"
    return payload


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
