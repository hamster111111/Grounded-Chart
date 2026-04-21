from __future__ import annotations

from typing import Protocol

from grounded_chart.llm import LLMClient
from grounded_chart.repair_policy import RepairPlan, RuleBasedRepairPlanner
from grounded_chart.schema import ChartIntentPlan, RepairPatch, VerificationReport


class Repairer(Protocol):
    def propose(self, code: str, plan: ChartIntentPlan, report: VerificationReport) -> RepairPatch:
        """Return a repair patch or repair instruction for failed verification."""


class RuleBasedRepairer:
    """Generate deterministic repair guidance from verifier error codes.

    This is a placeholder for an LLM-backed localized repairer. Keeping the
    first version deterministic makes ablations and error analysis easier.
    """

    def __init__(self, planner: RuleBasedRepairPlanner | None = None) -> None:
        self.planner = planner or RuleBasedRepairPlanner()

    def propose(self, code: str, plan: ChartIntentPlan, report: VerificationReport) -> RepairPatch:
        repair_plan = self.planner.plan(report, generated_code=code)
        codes = report.error_codes
        hints: list[str] = []
        if not repair_plan.should_repair:
            return RepairPatch(
                strategy="rule_based_abstain",
                instruction=repair_plan.reason or "No repair should be attempted.",
                target_error_codes=codes,
                repair_plan=repair_plan,
            )
        if "length_mismatch_extra_points" in codes:
            hints.append("Aggregate rows before plotting; add the missing group-by over the requested dimension(s).")
        if "length_mismatch_missing_points" in codes:
            hints.append("Check filters, joins, and grouping columns; expected categories are missing from the plot.")
        if "wrong_aggregation_value" in codes:
            hints.append("Use the requested aggregation operator and measure column before plotting.")
        if "data_point_not_found" in codes or "unexpected_data_point" in codes:
            hints.append("Align plotted x-values with the canonical expected data table; inspect filter and join logic.")
        if "wrong_order" in codes:
            hints.append("Sort the plotted data according to the requested order before rendering.")
        if "wrong_chart_type" in codes:
            hints.append(f"Use chart type '{plan.chart_type}' as requested by the intent plan.")
        if any(code in codes for code in ("wrong_axis_title", "wrong_x_label", "wrong_y_label", "wrong_z_label", "wrong_figure_title")):
            hints.append("Update only the requested title or axis label calls; keep plotted data unchanged.")
        if any(code in codes for code in ("wrong_x_tick_labels", "wrong_y_tick_labels", "wrong_z_tick_labels")):
            hints.append("Adjust only the requested tick locations or tick labels.")
        if any(code in codes for code in ("wrong_x_scale", "wrong_y_scale", "wrong_z_scale")):
            hints.append("Use the requested axis scaling without changing the underlying data.")
        if "wrong_axis_layout" in codes:
            hints.append("Recreate the subplot layout so the axes occupy the required positions.")
        if "missing_legend_label" in codes:
            hints.append("Add the required legend labels and call legend without changing the plotted values.")
        if "wrong_projection" in codes:
            hints.append("Use the requested axis projection when constructing the subplot.")
        if "missing_annotation_text" in codes:
            hints.append("Add the missing annotation or text element at the requested scope.")
        if "missing_artist_type" in codes:
            hints.append("Add the required visual artist type while preserving existing data bindings.")
        if not hints:
            if repair_plan.scope == "backend_specific_regeneration":
                hints.append("Regenerate the figure using backend-specific APIs and preserve the grounded requirements.")
            else:
                hints.append("No repair needed.")
        constraints = self._constraints(repair_plan)
        instruction = " ".join([*hints, constraints]).strip()
        return RepairPatch(
            strategy="rule_based_scoped_instruction",
            instruction=instruction,
            target_error_codes=codes,
            repair_plan=repair_plan,
        )

    def _constraints(self, repair_plan: RepairPlan) -> str:
        allowed = ", ".join(repair_plan.allowed_edits) or "the failed requirement only"
        forbidden = ", ".join(repair_plan.forbidden_edits) or "unrelated requirements"
        return f"Repair scope: {repair_plan.scope}. Allowed edits: {allowed}. Forbidden edits: {forbidden}."


class LLMRepairer:
    """Scoped code repairer backed by an OpenAI-compatible JSON-returning LLM."""

    def __init__(self, client: LLMClient, planner: RuleBasedRepairPlanner | None = None) -> None:
        self.client = client
        self.planner = planner or RuleBasedRepairPlanner()

    def propose(self, code: str, plan: ChartIntentPlan, report: VerificationReport) -> RepairPatch:
        repair_plan = self.planner.plan(report, generated_code=code)
        if not repair_plan.should_repair:
            return RepairPatch(
                strategy="llm_abstain",
                instruction=repair_plan.reason or "No repair should be attempted.",
                target_error_codes=report.error_codes,
                repair_plan=repair_plan,
            )
        payload = self.client.complete_json(
            system_prompt=_repairer_system_prompt(),
            user_prompt=_repairer_user_prompt(code=code, plan=plan, report=report, repair_plan=repair_plan),
            temperature=0.0,
        )
        instruction = str(payload.get("instruction") or repair_plan.reason or "").strip()
        repaired_code = payload.get("repaired_code")
        if repaired_code is not None:
            repaired_code = str(repaired_code)
        return RepairPatch(
            strategy="llm_scoped_repair",
            instruction=instruction or repair_plan.reason or "Scoped repair proposed by LLM.",
            target_error_codes=report.error_codes,
            repair_plan=repair_plan,
            repaired_code=repaired_code,
        )


def _repairer_system_prompt() -> str:
    return (
        "You are a scoped chart-code repair assistant. "
        "You must preserve all satisfied requirements and only repair the failed requirements described in the report. "
        "Return JSON with keys: instruction, repaired_code. "
        "Do not add explanations outside JSON."
    )


def _repairer_user_prompt(code: str, plan: ChartIntentPlan, report: VerificationReport, repair_plan: RepairPlan) -> str:
    plan_payload = {
        "chart_type": plan.chart_type,
        "dimensions": list(plan.dimensions),
        "measure": {"column": plan.measure.column, "agg": plan.measure.agg},
        "sort": {"by": plan.sort.by, "direction": plan.sort.direction} if plan.sort is not None else None,
        "limit": plan.limit,
        "raw_query": plan.raw_query,
    }
    error_payload = [
        {
            "code": error.code,
            "message": error.message,
            "expected": error.expected,
            "actual": error.actual,
            "operator": error.operator,
        }
        for error in report.errors
    ]
    repair_payload = {
        "repair_level": repair_plan.repair_level,
        "scope": repair_plan.scope,
        "allowed_edits": list(repair_plan.allowed_edits),
        "forbidden_edits": list(repair_plan.forbidden_edits),
        "reason": repair_plan.reason,
    }
    return (
        "Requirement plan:\n"
        f"{plan_payload}\n\n"
        "Verification errors:\n"
        f"{error_payload}\n\n"
        "Repair constraints:\n"
        f"{repair_payload}\n\n"
        "Original code:\n"
        f"{code}\n"
    )
