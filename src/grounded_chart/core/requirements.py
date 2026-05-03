from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RequirementScope = Literal["figure", "panel", "shared", "any_visible"]
RequirementType = Literal["data_operation", "encoding", "annotation", "presentation_constraint", "figure_composition"]
RequirementStatus = Literal["explicit", "inferred", "assumed", "ambiguous", "unsupported"]
RequirementPriority = Literal["core", "secondary"]
RequirementSeverity = Literal["error", "warning", "info"]
RequirementMatchPolicy = Literal[
    "exact",
    "contains",
    "normalized_contains",
    "numeric_close",
    "sequence_exact",
    "presence",
]


@dataclass(frozen=True)
class RequirementNode:
    """Atomic chart requirement with source provenance and verifier policy."""

    requirement_id: str
    scope: RequirementScope
    type: RequirementType
    name: str
    value: Any
    source_span: str = ""
    status: RequirementStatus = "explicit"
    confidence: float | None = None
    depends_on: tuple[str, ...] = ()
    priority: RequirementPriority = "core"
    panel_id: str | None = None
    assumption: str | None = None
    severity: RequirementSeverity = "error"
    match_policy: RequirementMatchPolicy = "exact"

    @property
    def is_verifiable(self) -> bool:
        return self.status in {"explicit", "inferred", "assumed"}


@dataclass(frozen=True)
class PanelRequirementPlan:
    panel_id: str
    chart_type: str
    requirement_ids: tuple[str, ...] = ()
    data_ops: dict[str, Any] = field(default_factory=dict)
    encodings: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    presentation_constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChartRequirementPlan:
    """Figure-aware requirement plan.

    The current MVP can keep using ChartIntentPlan for execution, while this
    object carries the traceable requirement representation described in the
    design document.
    """

    requirements: tuple[RequirementNode, ...]
    panels: tuple[PanelRequirementPlan, ...]
    figure_requirements: dict[str, Any] = field(default_factory=dict)
    shared_requirement_ids: tuple[str, ...] = ()
    raw_query: str = ""

    @property
    def verifiable_requirements(self) -> tuple[RequirementNode, ...]:
        return tuple(req for req in self.requirements if req.is_verifiable)

    @property
    def ambiguous_requirements(self) -> tuple[RequirementNode, ...]:
        return tuple(req for req in self.requirements if req.status == "ambiguous")

    @property
    def unsupported_requirements(self) -> tuple[RequirementNode, ...]:
        return tuple(req for req in self.requirements if req.status == "unsupported")


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: Literal["expected", "actual"]
    requirement_ids: tuple[str, ...]
    payload: Any
    source: str
    program: str | None = None
    input_hash: str | None = None
    artifact_hash: str | None = None
    panel_id: str | None = None


@dataclass(frozen=True)
class EvidenceLink:
    requirement_id: str
    expected_artifact_id: str | None
    actual_artifact_id: str | None
    verdict: Literal["pass", "fail", "abstain", "unsupported"]
    error_codes: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class EvidenceGraph:
    requirements: tuple[RequirementNode, ...]
    expected_artifacts: tuple[Artifact, ...] = ()
    actual_artifacts: tuple[Artifact, ...] = ()
    links: tuple[EvidenceLink, ...] = ()

    @property
    def failed_requirement_ids(self) -> tuple[str, ...]:
        return tuple(link.requirement_id for link in self.links if link.verdict == "fail")

    @property
    def passed_requirement_ids(self) -> tuple[str, ...]:
        return tuple(link.requirement_id for link in self.links if link.verdict == "pass")
