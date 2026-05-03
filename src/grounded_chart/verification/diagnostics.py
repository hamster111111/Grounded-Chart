from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grounded_chart.core.requirements import Artifact, EvidenceGraph, EvidenceLink, RequirementNode


@dataclass(frozen=True)
class FailureAtom:
    """Actionable, evidence-grounded diagnosis for one failed requirement."""

    requirement_id: str
    requirement_name: str | None
    requirement_type: str | None
    requirement_severity: str | None
    requirement_match_policy: str | None
    verdict: str
    error_codes: tuple[str, ...]
    expected_artifact_id: str | None
    actual_artifact_id: str | None
    expected_preview: Any
    actual_preview: Any
    mismatch_type: str
    suggested_action_scope: str
    evidence_summary: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_name": self.requirement_name,
            "requirement_type": self.requirement_type,
            "requirement_severity": self.requirement_severity,
            "requirement_match_policy": self.requirement_match_policy,
            "verdict": self.verdict,
            "error_codes": list(self.error_codes),
            "expected_artifact_id": self.expected_artifact_id,
            "actual_artifact_id": self.actual_artifact_id,
            "expected_preview": self.expected_preview,
            "actual_preview": self.actual_preview,
            "mismatch_type": self.mismatch_type,
            "suggested_action_scope": self.suggested_action_scope,
            "evidence_summary": self.evidence_summary,
            "message": self.message,
        }


def failure_atoms_from_evidence_graph(graph: EvidenceGraph | None) -> tuple[FailureAtom, ...]:
    if graph is None:
        return ()
    requirements = {requirement.requirement_id: requirement for requirement in graph.requirements}
    expected_artifacts = {artifact.artifact_id: artifact for artifact in graph.expected_artifacts}
    actual_artifacts = {artifact.artifact_id: artifact for artifact in graph.actual_artifacts}
    atoms: list[FailureAtom] = []
    for link in graph.links:
        if link.verdict not in {"fail", "abstain", "unsupported"}:
            continue
        requirement = requirements.get(link.requirement_id)
        expected_artifact = expected_artifacts.get(link.expected_artifact_id or "")
        actual_artifact = actual_artifacts.get(link.actual_artifact_id or "")
        mismatch_type = _mismatch_type(link, requirement, expected_artifact, actual_artifact)
        atoms.append(
            FailureAtom(
                requirement_id=link.requirement_id,
                requirement_name=requirement.name if requirement is not None else None,
                requirement_type=requirement.type if requirement is not None else None,
                requirement_severity=requirement.severity if requirement is not None else None,
                requirement_match_policy=requirement.match_policy if requirement is not None else None,
                verdict=link.verdict,
                error_codes=tuple(link.error_codes),
                expected_artifact_id=link.expected_artifact_id,
                actual_artifact_id=link.actual_artifact_id,
                expected_preview=_payload_preview(expected_artifact.payload if expected_artifact is not None else None),
                actual_preview=_payload_preview(actual_artifact.payload if actual_artifact is not None else None),
                mismatch_type=mismatch_type,
                suggested_action_scope=_suggested_action_scope(mismatch_type, link, requirement),
                evidence_summary=_evidence_summary(link, expected_artifact, actual_artifact, mismatch_type),
                message=link.message,
            )
        )
    return tuple(atoms)


def failure_atoms_to_dicts(atoms: tuple[FailureAtom, ...] | list[FailureAtom]) -> list[dict[str, Any]]:
    return [atom.to_dict() for atom in atoms]


def _mismatch_type(
    link: EvidenceLink,
    requirement: RequirementNode | None,
    expected_artifact: Artifact | None,
    actual_artifact: Artifact | None,
) -> str:
    codes = set(link.error_codes)
    requirement_name = str(requirement.name if requirement is not None else "").strip().lower()
    if link.verdict == "unsupported":
        return "unsupported_requirement"
    if link.verdict == "abstain" or (expected_artifact is None or actual_artifact is None):
        return "missing_artifact_binding"
    if "execution_error" in codes:
        return "runtime_error"
    if "wrong_chart_type" in codes:
        return "wrong_chart_type"
    if "wrong_order" in codes:
        return "wrong_sort_order"
    if "length_mismatch_extra_points" in codes and ("wrong_aggregation_value" in codes or requirement_name == "aggregation"):
        return "missing_groupby_or_aggregation"
    if "wrong_aggregation_value" in codes:
        return "wrong_aggregation_value"
    if codes & {"length_mismatch_missing_points", "data_point_not_found", "unexpected_data_point"}:
        return "wrong_filter_or_dimension_values"
    if codes & {"wrong_axes_count", "wrong_axis_layout"}:
        return "figure_composition_mismatch"
    if "wrong_projection" in codes:
        return "wrong_projection"
    if "missing_legend_label" in codes:
        return "missing_legend"
    if "missing_annotation_text" in codes:
        return "missing_annotation"
    if codes & {"wrong_figure_title", "wrong_axis_title", "wrong_x_label", "wrong_y_label", "wrong_z_label"}:
        return "presentation_text_mismatch"
    if codes & {"wrong_x_tick_labels", "wrong_y_tick_labels", "wrong_z_tick_labels"}:
        return "tick_label_mismatch"
    if codes & {"wrong_x_scale", "wrong_y_scale", "wrong_z_scale"}:
        return "scale_mismatch"
    if "missing_artist_type" in codes:
        return "missing_artist_type"
    if requirement_name:
        return f"{requirement_name}_mismatch"
    return "unknown_requirement_mismatch"


def _suggested_action_scope(mismatch_type: str, link: EvidenceLink, requirement: RequirementNode | None) -> str:
    if mismatch_type in {
        "missing_groupby_or_aggregation",
        "wrong_aggregation_value",
        "wrong_filter_or_dimension_values",
        "wrong_sort_order",
    }:
        return "data_transformation"
    if mismatch_type in {
        "presentation_text_mismatch",
        "tick_label_mismatch",
        "scale_mismatch",
        "missing_legend",
        "missing_annotation",
        "wrong_projection",
        "runtime_error",
    }:
        return "local_patch"
    if mismatch_type in {"wrong_chart_type", "figure_composition_mismatch", "missing_artist_type"}:
        return "structural_regeneration"
    if mismatch_type in {"unsupported_requirement", "missing_artifact_binding"}:
        return "diagnose_only"
    if requirement is not None and requirement.type == "data_operation":
        return "data_transformation"
    return "diagnose_only"


def _evidence_summary(
    link: EvidenceLink,
    expected_artifact: Artifact | None,
    actual_artifact: Artifact | None,
    mismatch_type: str,
) -> str:
    expected_label = _artifact_label(expected_artifact)
    actual_label = _artifact_label(actual_artifact)
    codes = ", ".join(link.error_codes) if link.error_codes else "no_error_code"
    return (
        f"{link.requirement_id}: {mismatch_type}; "
        f"expected={expected_label}; actual={actual_label}; error_codes={codes}."
    )


def _artifact_label(artifact: Artifact | None) -> str:
    if artifact is None:
        return "none"
    if artifact.program:
        return f"{artifact.artifact_id}/{artifact.program}"
    return artifact.artifact_id


def _payload_preview(payload: Any, *, max_items: int = 8, max_depth: int = 4) -> Any:
    return _jsonable_preview(payload, max_items=max_items, max_depth=max_depth)


def _jsonable_preview(value: Any, *, max_items: int, max_depth: int) -> Any:
    if max_depth <= 0:
        return _short_repr(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, tuple):
        return [_jsonable_preview(item, max_items=max_items, max_depth=max_depth - 1) for item in value[:max_items]]
    if isinstance(value, list):
        return [_jsonable_preview(item, max_items=max_items, max_depth=max_depth - 1) for item in value[:max_items]]
    if isinstance(value, dict):
        return {
            str(key): _jsonable_preview(item, max_items=max_items, max_depth=max_depth - 1)
            for key, item in list(value.items())[:max_items]
        }
    if hasattr(value, "item"):
        try:
            return _jsonable_preview(value.item(), max_items=max_items, max_depth=max_depth - 1)
        except Exception:
            pass
    if hasattr(value, "to_dict") and hasattr(value, "head"):
        try:
            return _jsonable_preview(value.head(max_items).to_dict(orient="records"), max_items=max_items, max_depth=max_depth - 1)
        except Exception:
            pass
    return _short_repr(value)


def _short_repr(value: Any, *, max_chars: int = 240) -> str:
    text = str(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text

