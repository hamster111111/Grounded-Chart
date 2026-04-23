from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from grounded_chart.backend import infer_backend_profile
from grounded_chart.schema import VerificationReport

RepairLevel = Literal[0, 1, 2, 3, 4]


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


class RuleBasedRepairPlanner:
    """Map verifier errors into a scope-controlled repair plan."""

    LOCAL_ERRORS = {
        "execution_error",
        "wrong_order",
        "wrong_chart_type",
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
        "wrong_axis_layout",
        "missing_legend_label",
        "missing_artist_type",
        "wrong_artist_count",
        "insufficient_artist_count",
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

    def plan(self, report: VerificationReport, generated_code: str = "") -> RepairPlan:
        codes = set(report.error_codes)
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
            return RepairPlan(
                repair_level=0,
                scope="none",
                target_error_codes=(),
                reason="All verifiable requirements passed.",
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
                    "use backend-specific regeneration instead of localized patching."
                ),
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
