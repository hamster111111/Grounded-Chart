from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from grounded_chart.workspace.artifact_workspace import FIGURE_READER_AGENT_DIR
from grounded_chart.runtime.llm import LLMClient, LLMCompletionTrace, LLMUsage
from grounded_chart.agents.feedback import normalize_figure_plan_feedback
from grounded_chart.data.source_data import parse_scalar, short_cell


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
        source_data_plan: Any | None = None,
    ) -> FigureAudit:
        """Return visual-semantic audit evidence for a rendered chart."""


class VLMFigureReaderAgent:
    """Vision-backed FigureReaderAgent for semantic readability audits."""

    def __init__(
        self,
        client: LLMClient,
        *,
        agent_name: str = "vlm_figure_reader_v1",
        max_tokens: int | None = 1024,
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
        source_data_plan: Any | None = None,
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
                    source_data_plan=source_data_plan,
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
        payload = _normalize_figure_reader_payload(result.payload)
        issues = _dict_tuple(payload.get("issues"))
        return FigureAudit(
            ok=bool(payload.get("ok", False)),
            summary=str(payload.get("summary") or ""),
            readability_issues=_issues_by_family(issues, "readability", payload.get("readability_issues")),
            encoding_confusions=_issues_by_family(issues, "encoding", payload.get("encoding_confusions")),
            data_semantic_warnings=_issues_by_family(issues, "data_semantic", payload.get("data_semantic_warnings")),
            suspicious_artifacts=_issues_by_family(issues, "artifact", payload.get("suspicious_artifacts")),
            unclear_regions=_issues_by_family(issues, "unclear", payload.get("unclear_regions")),
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
                "scope": "source_aware_visual_semantic_readability_audit",
                "input_policy": "image_original_task_and_original_source_files_only",
            },
        )


def write_figure_audit_artifact(
    *,
    output_root: str | Path,
    round_index: int,
    audit: FigureAudit,
) -> str:
    agent_dir = Path(output_root).resolve() / FIGURE_READER_AGENT_DIR / f"round_{round_index}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    audit_path = agent_dir / "figure_audit.json"
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
        "Inspect the rendered chart image as the primary evidence, then use only the original task and original attached source-file cards as constraints. "
        "You are independent from PlanAgent and ExecutorAgent: do not rely on construction plans, generated code, artifact workspaces, chart protocols, or pipeline traces. "
        "Your job is to report whether the chart is understandable and visibly faithful to the original task and source files at a semantic level: readable marks, distinguishable encodings, non-confusing legends, visible annotations, sensible composition, and suspicious visual artifacts. "
        "You may flag that a visible trend, mark geometry, source-file value range, or color/channel mapping appears suspicious, but you must not assert exact numeric data is wrong unless the supplied source-file cards directly support that claim. "
        "Do not propose code edits. Do not invent missing data. Do not rewrite the task. "
        "Return only a compact JSON object with keys: ok, summary, issues, confidence. "
        "issues must contain at most five objects. Each issue should include family, issue_type, severity, evidence, affected_region, related_plan_ref, and recommendation. "
        "Use family values readability, encoding, data_semantic, artifact, or unclear. "
        "Do not output plan_feedback, code patches, or nested action objects; the pipeline will convert issues into PlanAgent feedback. "
        "Use severity values info, warning, or error. If the image is understandable and no evidence-backed concern exists, return ok=true and empty arrays."
    )


def _user_prompt(
    *,
    query: str,
    source_data_plan: Any | None,
    generation_context: dict[str, Any] | None,
) -> str:
    payload = {
        "original_task": query,
        "generation_context": _compact_generation_context(generation_context or {}),
        "original_source_file_cards": _source_file_cards(
            query=query,
            source_data_plan=source_data_plan,
        ),
        "input_policy": {
            "provided_to_agent": [
                "rendered_image",
                "original_task",
                "original_source_file_cards",
            ],
            "internal_pipeline_state": "omitted",
        },
        "audit_scope": {
            "allowed": [
                "visual readability",
                "legend/channel ambiguity",
                "mark overlap or clutter",
                "missing or unreadable labels/annotations",
                "suspicious visible data geometry when constrained by original source-file cards",
                "strange artifacts, clipping, occlusion, or misleading composition",
            ],
            "forbidden": [
                "exact numeric judging without source-file-card evidence",
                "source data edits",
                "direct code patch proposals",
                "changing required chart type or required labels by preference",
            ],
            "max_issues": 5,
        },
    }
    return "Audit the rendered chart using only the original task, original source files, and image.\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _source_file_cards(
    *,
    query: str,
    source_data_plan: Any | None,
    max_files: int = 8,
) -> list[dict[str, Any]]:
    payload = source_data_plan.to_dict() if hasattr(source_data_plan, "to_dict") else _jsonable(source_data_plan)
    if not isinstance(payload, dict):
        return []
    requested_years = _requested_years(query)
    cards = []
    for item in list(payload.get("files") or [])[:max_files]:
        if not isinstance(item, dict):
            continue
        cards.append(_source_file_card(item, requested_years=requested_years))
    return cards


def _source_file_card(item: dict[str, Any], *, requested_years: set[int]) -> dict[str, Any]:
    path = Path(str(item.get("path") or ""))
    suffix = str(item.get("suffix") or path.suffix).lower()
    base = {
        "name": item.get("name") or path.name,
        "suffix": suffix,
        "columns": list(item.get("columns") or []),
        "size_bytes": item.get("size_bytes"),
    }
    if not path.exists() or not path.is_file():
        return {**base, "read_error": "source_file_not_found"}
    try:
        if suffix in {".csv", ".tsv"}:
            return {**base, **_tabular_source_card(path, suffix=suffix, requested_years=requested_years)}
        if suffix == ".json":
            return {**base, **_json_source_card(path)}
        return {**base, "read_error": f"unsupported_source_preview:{suffix}"}
    except Exception as exc:
        return {**base, "read_error": f"{type(exc).__name__}: {str(exc)[:180]}"}


def _tabular_source_card(path: Path, *, suffix: str, requested_years: set[int]) -> dict[str, Any]:
    delimiter = "\t" if suffix == ".tsv" else ","
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    requested_rows: list[dict[str, Any]] = []
    numeric: dict[str, dict[str, Any]] = {}
    row_count = 0
    columns: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = [str(item) for item in list(reader.fieldnames or [])]
        for raw_row in reader:
            row_count += 1
            row = {str(key): parse_scalar(value) for key, value in dict(raw_row).items()}
            compact = _compact_row(row)
            if len(first_rows) < 3:
                first_rows.append(compact)
            last_rows.append(compact)
            if len(last_rows) > 3:
                last_rows.pop(0)
            if requested_years and _row_matches_requested_year(row, requested_years) and len(requested_rows) < 12:
                requested_rows.append(compact)
            _update_numeric_profile(numeric, row)
    return {
        "columns": columns,
        "row_count": row_count,
        "first_rows": first_rows,
        "last_rows": last_rows,
        "requested_year_rows": requested_rows,
        "numeric_profile": _finalize_numeric_profile(numeric),
    }


def _json_source_card(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        rows = [_jsonable(item) for item in payload[:6]]
        return {
            "json_type": "list",
            "row_count": len(payload),
            "preview_rows": rows,
        }
    if isinstance(payload, dict):
        return {
            "json_type": "dict",
            "keys": [str(key) for key in list(payload.keys())[:24]],
            "preview_items": [
                {"key": str(key), "value": short_cell(value)}
                for key, value in list(payload.items())[:8]
            ],
        }
    return {"json_type": type(payload).__name__, "preview": short_cell(payload)}


def _requested_years(query: str) -> set[int]:
    years = set()
    for token in re.findall(r"\b(?:19|20)\d{2}\b", str(query or "")):
        try:
            years.add(int(token))
        except ValueError:
            pass
    return years


def _row_matches_requested_year(row: dict[str, Any], requested_years: set[int]) -> bool:
    for value in row.values():
        try:
            if int(value) in requested_years:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): short_cell(value) for key, value in row.items()}


def _update_numeric_profile(profile: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            entry = profile.setdefault(str(key), {"min": value, "max": value, "count": 0})
            entry["min"] = min(entry["min"], value)
            entry["max"] = max(entry["max"], value)
            entry["count"] += 1


def _finalize_numeric_profile(profile: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "min": value.get("min"),
            "max": value.get("max"),
            "count": value.get("count"),
        }
        for key, value in profile.items()
    }


def _compact_generation_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in context.items()
        if key in {"query_source", "native_id", "generation_mode"}
    }


def _dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _normalize_figure_reader_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if "issues" in payload or "summary" in payload or "ok" in payload:
        return payload
    if payload.get("issue_type") or payload.get("family"):
        return {
            "ok": False,
            "summary": "",
            "issues": [payload],
            "confidence": payload.get("confidence"),
        }
    return payload


def _issues_by_family(issues: tuple[dict[str, Any], ...], family: str, fallback: Any) -> tuple[dict[str, Any], ...]:
    fallback_items = _dict_tuple(fallback)
    if fallback_items:
        return fallback_items
    aliases = {
        "readability": {"readability", "visual_readability", "text"},
        "encoding": {"encoding", "encoding_confusion", "channel"},
        "data_semantic": {"data", "data_semantic", "semantic"},
        "artifact": {"artifact", "suspicious_artifact", "visual_artifact"},
        "unclear": {"unclear", "occlusion", "ambiguous"},
    }.get(family, {family})
    return tuple(
        {key: value for key, value in item.items() if key != "family"}
        for item in issues
        if str(item.get("family") or "").strip().lower() in aliases
    )


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
