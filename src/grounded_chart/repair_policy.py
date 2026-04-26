from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from grounded_chart.backend import infer_backend_profile
from grounded_chart.schema import VerificationReport

RepairLevel = Literal[0, 1, 2, 3, 4]
RepairPolicyMode = Literal["strict", "exploratory"]
RepairActionClass = Literal["none", "local_patch", "data_block_regeneration", "structural_regeneration", "abstain"]


@dataclass(frozen=True)
class RepairPlan:
    repair_level: RepairLevel
    scope: str
    target_error_codes: tuple[str, ...]
    target_requirements: tuple[str, ...] = ()
    allowed_edits: tuple[str, ...] = ()
    forbidden_edits: tuple[str, ...] = ()
    reason: str = ""

    @property
    def should_repair(self) -> bool:
        return self.repair_level in {1, 2, 3}


@dataclass(frozen=True)
class RepairGateDecision:
    mode: RepairPolicyMode
    repairability: str | None
    blocked_by_policy: bool
    effective_plan: RepairPlan

    @property
    def should_repair(self) -> bool:
        return self.effective_plan.should_repair


STRICT_BLOCKED_REPAIRABILITIES = frozenset({"diagnose_only", "unsupported", "route_only"})


def normalize_repair_policy_mode(value: str | None) -> RepairPolicyMode:
    normalized = str(value or "exploratory").strip().lower()
    if normalized not in {"strict", "exploratory"}:
        raise ValueError(f"Unsupported repair policy mode {value!r}; allowed values: ['exploratory', 'strict']")
    return normalized  # type: ignore[return-value]


def repair_action_class_for_scope(scope: str | None) -> RepairActionClass:
    normalized = str(scope or "none").strip().lower()
    if normalized in {"", "none"}:
        return "none"
    if normalized == "local_patch":
        return "local_patch"
    if normalized == "data_transformation":
        return "data_block_regeneration"
    if normalized in {"structural_regeneration", "backend_specific_regeneration"}:
        return "structural_regeneration"
    if normalized in {"policy_blocked", "abstain", "diagnose_only", "route_only", "unsupported"}:
        return "abstain"
    return "abstain"


def repair_action_class_for_plan(repair_plan: RepairPlan | None) -> RepairActionClass:
    if repair_plan is None:
        return "none"
    return repair_action_class_for_scope(repair_plan.scope)


def apply_auto_repair_gate(
    repair_plan: RepairPlan,
    *,
    case_metadata: dict[str, Any] | None = None,
    mode: str | None = None,
) -> RepairGateDecision:
    normalized_mode = normalize_repair_policy_mode(mode)
    metadata = case_metadata or {}
    repairability_raw = metadata.get("repairability")
    repairability = str(repairability_raw).strip().lower() if repairability_raw is not None else None

    if (
        not repair_plan.should_repair
        or normalized_mode != "strict"
        or repairability not in STRICT_BLOCKED_REPAIRABILITIES
    ):
        return RepairGateDecision(
            mode=normalized_mode,
            repairability=repairability,
            blocked_by_policy=False,
            effective_plan=repair_plan,
        )

    blocked_plan = replace(
        repair_plan,
        repair_level=4,
        scope="policy_blocked",
        allowed_edits=(),
        forbidden_edits=tuple(dict.fromkeys((*repair_plan.forbidden_edits, "automatic repair in strict mode"))),
        reason=(
            f"Strict repair policy blocked automatic repair for repairability='{repairability}'. "
            "Keep this case as diagnose-only, unsupported, or route-only in benchmark evaluation."
        ),
    )
    return RepairGateDecision(
        mode=normalized_mode,
        repairability=repairability,
        blocked_by_policy=True,
        effective_plan=blocked_plan,
    )


def override_repair_plan_scope(repair_plan: RepairPlan, *, scope: str | None = None, reason: str | None = None) -> RepairPlan:
    desired_scope = str(scope or "").strip()
    if not desired_scope or desired_scope == repair_plan.scope:
        if reason:
            return replace(repair_plan, reason=reason)
        return repair_plan

    if desired_scope == "data_transformation":
        return replace(
            repair_plan,
            repair_level=max(int(repair_plan.repair_level), 2),  # type: ignore[arg-type]
            scope="data_transformation",
            allowed_edits=("data preparation block", "x/y binding if needed", "groupby / aggregation / sorting logic"),
            forbidden_edits=("data loading", "unrelated style", "unrelated labels"),
            reason=reason or f"Escalated from {repair_plan.scope} to data_transformation.",
        )
    if desired_scope == "structural_regeneration":
        return replace(
            repair_plan,
            repair_level=3,
            scope="structural_regeneration",
            allowed_edits=("full plotting code",),
            forbidden_edits=("inventing requirements", "using values outside expected artifacts"),
            reason=reason or f"Escalated from {repair_plan.scope} to structural_regeneration.",
        )
    if desired_scope == "backend_specific_regeneration":
        return replace(
            repair_plan,
            repair_level=3,
            scope="backend_specific_regeneration",
            allowed_edits=("full plotting code", "backend-specific figure construction"),
            forbidden_edits=("inventing requirements", "claiming hard fidelity beyond supported backend capabilities"),
            reason=reason or f"Escalated from {repair_plan.scope} to backend_specific_regeneration.",
        )
    return replace(repair_plan, reason=reason or repair_plan.reason)


def _has_explicit_expected_plot_points(report: VerificationReport) -> bool:
    expected_trace = getattr(report, "expected_trace", None)
    points = getattr(expected_trace, "points", ()) if expected_trace is not None else ()
    return bool(points)

class RuleBasedRepairPlanner:
    """Map verifier errors into a scope-controlled repair plan."""

    LOCAL_ERRORS = {
        "execution_error",
        "wrong_order",
        "wrong_axis_label",
        "wrong_axis_title",
        "wrong_x_label",
        "wrong_y_label",
        "wrong_z_label",
        "wrong_figure_title",
        "wrong_figure_size",
        "wrong_projection",
        "wrong_x_scale",
        "wrong_y_scale",
        "wrong_z_scale",
        "wrong_x_tick_labels",
        "wrong_y_tick_labels",
        "wrong_z_tick_labels",
        "missing_legend_label",
        "missing_annotation_text",
        "missing_requested_constraint",
    }
    DATA_TRANSFORM_ERRORS = {
        "length_mismatch_extra_points",
        "length_mismatch_missing_points",
        "wrong_aggregation_value",
        "data_point_not_found",
        "unexpected_data_point",
    }
    STRUCTURAL_ERRORS = {
        "wrong_axis_layout",
        "wrong_chart_type",
        "missing_artist_type",
    }
    ARTIST_CARDINALITY_ERRORS = {
        "wrong_artist_count",
        "insufficient_artist_count",
    }

    def plan(self, report: VerificationReport, generated_code: str = "") -> RepairPlan:
        hard_errors = tuple(error for error in report.errors if getattr(error, "severity", "error") == "error")
        codes = {error.code for error in hard_errors}
        backend_profile = infer_backend_profile(
            actual_trace=report.actual_trace,
            actual_figure=report.actual_figure,
            generated_code=generated_code,
        )
        trace_source = str(report.actual_trace.source) if report.actual_trace is not None else ""
        figure_source = str(report.actual_figure.source) if report.actual_figure is not None else ""
        has_backend_evidence = bool(generated_code.strip()) or (
            report.actual_trace is not None
            and (
                bool(report.actual_trace.raw.get("backend")) if isinstance(report.actual_trace.raw, dict) else False
            or trace_source.startswith("matplotlib")
            or trace_source.startswith("plotly")
            )
        ) or (
            report.actual_figure is not None
            and (
                bool(report.actual_figure.raw.get("backend")) if isinstance(report.actual_figure.raw, dict) else False
            or figure_source in {"matplotlib_figure", "plotly_figure"}
            or any(str(axis.projection).lower() in {"3d", "plotly"} for axis in report.actual_figure.axes)
            )
        )
        if not codes:
            warning_codes = tuple(error.code for error in report.errors)
            reason = (
                "Only warning/info-level requirements failed; skip automatic repair for hard-score evaluation."
                if warning_codes
                else "All verifiable requirements passed."
            )
            return RepairPlan(
                repair_level=0,
                scope="none",
                target_error_codes=warning_codes,
                reason=reason,
            )
        if codes <= self.LOCAL_ERRORS:
            return RepairPlan(
                repair_level=1,
                scope="local_patch",
                target_error_codes=tuple(sorted(codes)),
                allowed_edits=("smallest relevant code region", "runtime-unsafe plotting kwargs or calls"),
                forbidden_edits=("data loading", "unrelated data transformations", "unrelated styling"),
                reason="Only localized presentation or encoding requirements failed.",
            )
        if has_backend_evidence and (backend_profile.verification_mode in {"soft", "none"} or backend_profile.support_tier != "native"):
            return RepairPlan(
                repair_level=3,
                scope="backend_specific_regeneration",
                target_error_codes=tuple(sorted(codes)),
                allowed_edits=("full plotting code", f"{backend_profile.backend_name} backend construction"),
                forbidden_edits=("inventing requirements", "claiming hard fidelity beyond supported backend capabilities"),
                reason=(
                    f"Backend '{backend_profile.backend_name}' only supports {backend_profile.verification_mode} verification; "
                    "use backend-specific regeneration only after localized repairs are ruled out."
                ),
            )
        if codes & self.STRUCTURAL_ERRORS:
            return RepairPlan(
                repair_level=3,
                scope="structural_regeneration",
                target_error_codes=tuple(sorted(codes)),
                allowed_edits=("full plotting code", "subplot layout or visual artist family"),
                forbidden_edits=("inventing requirements", "using values outside expected artifacts"),
                reason="Structural chart, artist-family, or axis-layout failures require regeneration rather than local patching.",
            )
        if codes & self.ARTIST_CARDINALITY_ERRORS:
            if _has_explicit_expected_plot_points(report):
                return RepairPlan(
                    repair_level=2,
                    scope="data_transformation",
                    target_error_codes=tuple(sorted(codes)),
                    allowed_edits=("plotted data cardinality", "x/y binding", "data preparation block"),
                    forbidden_edits=("data loading", "unrelated style", "unrelated layout"),
                    reason="Artist cardinality can be grounded to explicit expected plot points; repair plotted data cardinality.",
                )
            return RepairPlan(
                repair_level=4,
                scope="diagnose_only",
                target_error_codes=tuple(sorted(codes)),
                allowed_edits=(),
                forbidden_edits=("automatic local patch without expected plot-point evidence",),
                reason=(
                    "Artist-count failures are not safely auto-repairable without explicit expected plot-point artifacts; "
                    "treat this as diagnose-only or route to a specialized visual-structure repairer."
                ),
            )
        if codes & self.DATA_TRANSFORM_ERRORS and not (codes - self.DATA_TRANSFORM_ERRORS - self.LOCAL_ERRORS):
            return RepairPlan(
                repair_level=2,
                scope="data_transformation",
                target_error_codes=tuple(sorted(codes)),
                allowed_edits=("data preparation block", "x/y binding if needed"),
                forbidden_edits=("data loading", "unrelated style", "unrelated labels"),
                reason="Failures are attributable to data preparation or plotted data mismatch.",
            )
        return RepairPlan(
            repair_level=3,
            scope="structural_regeneration",
            target_error_codes=tuple(sorted(codes)),
            allowed_edits=("full plotting code",),
            forbidden_edits=("inventing requirements", "using values outside expected artifacts"),
            reason="Multiple or unknown failure families make localized repair unreliable.",
        )
