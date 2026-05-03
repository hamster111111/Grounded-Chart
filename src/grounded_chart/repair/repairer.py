from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from textwrap import dedent
from typing import Any, Protocol

from grounded_chart.runtime.llm import LLMClient, LLMJsonResult
from grounded_chart.verification.diagnostics import failure_atoms_from_evidence_graph, failure_atoms_to_dicts
from grounded_chart.repair.patch_ops import PatchAnchor, PatchOperation, parse_patch_operations
from grounded_chart.repair.policy import RepairPlan, RuleBasedRepairPlanner, override_repair_plan_scope
from grounded_chart.core.requirements import EvidenceGraph
from grounded_chart.core.schema import ChartIntentPlan, RepairPatch, VerificationReport


class Repairer(Protocol):
    def propose(
        self,
        code: str,
        plan: ChartIntentPlan,
        report: VerificationReport,
        escalation_scope: str | None = None,
        evidence_graph: EvidenceGraph | None = None,
    ) -> RepairPatch:
        """Return a repair patch or repair instruction for failed verification."""


class RuleBasedRepairer:
    """Generate deterministic repair guidance from verifier error codes.

    This is a placeholder for an LLM-backed localized repairer. Keeping the
    first version deterministic makes ablations and error analysis easier.
    """

    def __init__(self, planner: RuleBasedRepairPlanner | None = None) -> None:
        self.planner = planner or RuleBasedRepairPlanner()

    def propose(
        self,
        code: str,
        plan: ChartIntentPlan,
        report: VerificationReport,
        escalation_scope: str | None = None,
        evidence_graph: EvidenceGraph | None = None,
    ) -> RepairPatch:
        repair_plan = self.planner.plan(report, generated_code=code)
        if escalation_scope:
            repair_plan = override_repair_plan_scope(
                repair_plan,
                scope=escalation_scope,
                reason=f"Escalated to {escalation_scope} because a previous round made insufficient progress.",
            )
        codes = repair_plan.target_error_codes or report.error_codes
        hints: list[str] = []
        if not repair_plan.should_repair:
            return RepairPatch(
                strategy="rule_based_abstain",
                instruction=repair_plan.reason or "No repair should be attempted.",
                target_error_codes=codes,
                repair_plan=repair_plan,
                loop_signal="stop",
                loop_reason=repair_plan.reason or "No repair should be attempted.",
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
        if "execution_error" in codes:
            hints.append("Remove or replace only the runtime-unsafe plotting argument or call that caused execution to fail.")
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
        patch_ops = self._scoped_patch_ops(code=code, plan=plan, report=report, repair_plan=repair_plan)
        return RepairPatch(
            strategy="rule_based_scoped_instruction",
            instruction=instruction,
            target_error_codes=codes,
            repair_plan=repair_plan,
            loop_signal="continue",
            loop_reason=(
                f"Applied {repair_plan.scope}; if failures remain, another verifier-guided round may be useful."
                if not escalation_scope
                else f"Applied escalated scope {repair_plan.scope}; stop if this broader repair still makes no progress."
            ),
            patch_ops=patch_ops,
        )

    def _constraints(self, repair_plan: RepairPlan) -> str:
        allowed = ", ".join(repair_plan.allowed_edits) or "the failed requirement only"
        forbidden = ", ".join(repair_plan.forbidden_edits) or "unrelated requirements"
        return f"Repair scope: {repair_plan.scope}. Allowed edits: {allowed}. Forbidden edits: {forbidden}."

    def _scoped_patch_ops(
        self,
        *,
        code: str,
        plan: ChartIntentPlan,
        report: VerificationReport,
        repair_plan: RepairPlan,
    ) -> tuple[PatchOperation, ...]:
        if repair_plan.scope == "local_patch":
            return self._local_patch_ops(code=code, report=report, repair_plan=repair_plan)
        if repair_plan.scope == "data_transformation":
            return self._data_transformation_patch_ops(code=code, plan=plan, report=report, repair_plan=repair_plan)
        return ()

    def _local_patch_ops(self, *, code: str, report: VerificationReport, repair_plan: RepairPlan) -> tuple[PatchOperation, ...]:
        if repair_plan.scope != "local_patch":
            return ()
        operations: list[PatchOperation] = []
        target_codes = set(repair_plan.target_error_codes)
        for error in report.errors:
            if target_codes and error.code not in target_codes:
                continue
            expected = error.expected
            if error.code == "execution_error":
                keyword = _extract_unsupported_keyword(error.message)
                anchor = _infer_keyword_patch_anchor(code, keyword) if keyword else None
                if keyword and anchor is not None:
                    operations.append(
                        PatchOperation(
                            op="remove_keyword_arg",
                            anchor=anchor,
                            keyword=keyword,
                            description=f"Remove only the unsupported keyword argument '{keyword}'.",
                        )
                    )
                continue
            if expected is None:
                continue
            if error.code == "wrong_figure_title":
                operations.append(
                    PatchOperation(
                        op="replace_call_arg",
                        anchor=PatchAnchor(kind="method_call", name="suptitle", occurrence=1),
                        arg_index=0,
                        new_value=expected,
                        description="Update the figure title only.",
                    )
                )
            elif error.code == "wrong_axis_title":
                operations.append(
                    PatchOperation(
                        op="replace_call_arg",
                        anchor=PatchAnchor(kind="method_call", name="set_title", occurrence=_axis_occurrence(error.requirement_id)),
                        arg_index=0,
                        new_value=expected,
                        description="Update the axis title only.",
                    )
                )
            elif error.code == "wrong_x_label":
                operations.append(
                    PatchOperation(
                        op="replace_call_arg",
                        anchor=PatchAnchor(kind="method_call", name="set_xlabel", occurrence=_axis_occurrence(error.requirement_id)),
                        arg_index=0,
                        new_value=expected,
                        description="Update the x-axis label only.",
                    )
                )
            elif error.code == "wrong_y_label":
                operations.append(
                    PatchOperation(
                        op="replace_call_arg",
                        anchor=PatchAnchor(kind="method_call", name="set_ylabel", occurrence=_axis_occurrence(error.requirement_id)),
                        arg_index=0,
                        new_value=expected,
                        description="Update the y-axis label only.",
                    )
                )
            elif error.code == "wrong_z_label":
                operations.append(
                    PatchOperation(
                        op="replace_call_arg",
                        anchor=PatchAnchor(kind="method_call", name="set_zlabel", occurrence=_axis_occurrence(error.requirement_id)),
                        arg_index=0,
                        new_value=expected,
                        description="Update the z-axis label only.",
                    )
                )
            elif error.code == "wrong_x_tick_labels":
                operations.append(
                    PatchOperation(
                        op="replace_call_arg",
                        anchor=PatchAnchor(kind="method_call", name="set_xticklabels", occurrence=_axis_occurrence(error.requirement_id)),
                        arg_index=0,
                        new_value=list(expected) if isinstance(expected, (list, tuple)) else expected,
                        description="Update x tick labels only.",
                    )
                )
            elif error.code == "wrong_y_tick_labels":
                operations.append(
                    PatchOperation(
                        op="replace_call_arg",
                        anchor=PatchAnchor(kind="method_call", name="set_yticklabels", occurrence=_axis_occurrence(error.requirement_id)),
                        arg_index=0,
                        new_value=list(expected) if isinstance(expected, (list, tuple)) else expected,
                        description="Update y tick labels only.",
                    )
                )
            elif error.code == "wrong_z_tick_labels":
                operations.append(
                    PatchOperation(
                        op="replace_call_arg",
                        anchor=PatchAnchor(kind="method_call", name="set_zticklabels", occurrence=_axis_occurrence(error.requirement_id)),
                        arg_index=0,
                        new_value=list(expected) if isinstance(expected, (list, tuple)) else expected,
                        description="Update z tick labels only.",
                    )
                )
        return tuple(operations)

    def _data_transformation_patch_ops(
        self,
        *,
        code: str,
        plan: ChartIntentPlan,
        report: VerificationReport,
        repair_plan: RepairPlan,
    ) -> tuple[PatchOperation, ...]:
        if repair_plan.scope != "data_transformation":
            return ()
        supported_codes = {
            "length_mismatch_extra_points",
            "length_mismatch_missing_points",
            "wrong_aggregation_value",
            "data_point_not_found",
            "unexpected_data_point",
            "wrong_order",
        }
        if not report.error_codes or any(code not in supported_codes for code in report.error_codes):
            return ()
        snippet = _find_simple_row_projection_snippet(
            code,
            dimension_column=plan.dimensions[0] if plan.dimensions else None,
            measure_column=plan.measure.column,
        )
        if snippet is None:
            return ()
        replacement = _build_grouped_data_prep_block(
            dimension_var=snippet["dimension_var"],
            measure_var=snippet["measure_var"],
            dimension_column=snippet["dimension_column"],
            measure_column=snippet["measure_column"],
            agg=plan.measure.agg,
            sort=plan.sort,
            limit=plan.limit,
        )
        if replacement is None:
            return ()
        return (
            PatchOperation(
                op="replace_text",
                anchor=PatchAnchor(kind="text", text=snippet["text"], occurrence=1),
                new_value=replacement,
                description="Replace raw row projections with a grouped data-preparation block.",
            ),
        )


class LLMRepairer:
    """Scoped code repairer backed by an OpenAI-compatible JSON-returning LLM."""

    def __init__(self, client: LLMClient, planner: RuleBasedRepairPlanner | None = None, include_failure_atoms: bool = True) -> None:
        self.client = client
        self.planner = planner or RuleBasedRepairPlanner()
        self.include_failure_atoms = bool(include_failure_atoms)

    def propose(
        self,
        code: str,
        plan: ChartIntentPlan,
        report: VerificationReport,
        escalation_scope: str | None = None,
        evidence_graph: EvidenceGraph | None = None,
    ) -> RepairPatch:
        repair_plan = self.planner.plan(report, generated_code=code)
        if escalation_scope:
            repair_plan = override_repair_plan_scope(
                repair_plan,
                scope=escalation_scope,
                reason=f"Escalated to {escalation_scope} because a previous round made insufficient progress.",
            )
        if not repair_plan.should_repair:
            return RepairPatch(
                strategy="llm_abstain",
                instruction=repair_plan.reason or "No repair should be attempted.",
                target_error_codes=report.error_codes,
                repair_plan=repair_plan,
                loop_signal="stop",
                loop_reason=repair_plan.reason or "No repair should be attempted.",
            )
        result = _complete_json_with_trace(
            self.client,
            system_prompt=_repairer_system_prompt(),
            user_prompt=_repairer_user_prompt(
                code=code,
                plan=plan,
                report=report,
                repair_plan=repair_plan,
                escalation_scope=escalation_scope,
                evidence_graph=evidence_graph if self.include_failure_atoms else None,
            ),
            temperature=0.0,
        )
        payload = result.payload
        instruction = str(payload.get("instruction") or repair_plan.reason or "").strip()
        repaired_code = payload.get("repaired_code")
        if repaired_code is not None:
            repaired_code = str(repaired_code)
        patch_ops = _normalize_llm_patch_ops(parse_patch_operations(payload.get("patch_ops")), report=report)
        loop_signal = _normalize_loop_signal(payload.get("loop_signal"))
        loop_reason = str(payload.get("loop_reason") or "").strip() or None
        return RepairPatch(
            strategy="llm_scoped_repair",
            instruction=instruction or repair_plan.reason or "Scoped repair proposed by LLM.",
            target_error_codes=report.error_codes,
            repair_plan=repair_plan,
            repaired_code=repaired_code,
            loop_signal=loop_signal or "continue",
            loop_reason=loop_reason,
            llm_trace=result.trace,
            patch_ops=patch_ops,
        )


class TieredRepairer:
    """Route repairs across deterministic and LLM tiers by scope.

    Default behavior:
    - local_patch -> deterministic rule-based repair
    - data/structural/backend-specific -> LLM repair
    """

    def __init__(
        self,
        deterministic_repairer: Repairer | None = None,
        llm_repairer: Repairer | None = None,
        llm_scopes: tuple[str, ...] = (
            "data_transformation",
            "structural_regeneration",
            "backend_specific_regeneration",
        ),
    ) -> None:
        self.deterministic_repairer = deterministic_repairer or RuleBasedRepairer()
        self.llm_repairer = llm_repairer
        self.llm_scopes = tuple(llm_scopes)

    def propose(
        self,
        code: str,
        plan: ChartIntentPlan,
        report: VerificationReport,
        escalation_scope: str | None = None,
        evidence_graph: EvidenceGraph | None = None,
    ) -> RepairPatch:
        deterministic_patch = self.deterministic_repairer.propose(code, plan, report, escalation_scope=escalation_scope, evidence_graph=evidence_graph)
        repair_plan = deterministic_patch.repair_plan
        if repair_plan is None or not repair_plan.should_repair:
            return deterministic_patch
        if repair_plan.scope in self.llm_scopes and self.llm_repairer is not None:
            llm_patch = self.llm_repairer.propose(code, plan, report, escalation_scope=escalation_scope, evidence_graph=evidence_graph)
            return RepairPatch(
                strategy=f"tiered::{llm_patch.strategy}",
                instruction=llm_patch.instruction,
                target_error_codes=llm_patch.target_error_codes,
                repair_plan=llm_patch.repair_plan or repair_plan,
                repaired_code=llm_patch.repaired_code,
                loop_signal=llm_patch.loop_signal,
                loop_reason=llm_patch.loop_reason,
                llm_trace=llm_patch.llm_trace,
                patch_ops=llm_patch.patch_ops,
            )
        return RepairPatch(
            strategy=f"tiered::{deterministic_patch.strategy}",
            instruction=deterministic_patch.instruction,
            target_error_codes=deterministic_patch.target_error_codes,
            repair_plan=repair_plan,
            repaired_code=deterministic_patch.repaired_code,
            loop_signal=deterministic_patch.loop_signal,
            loop_reason=deterministic_patch.loop_reason,
            llm_trace=deterministic_patch.llm_trace,
            patch_ops=deterministic_patch.patch_ops,
        )


def _repairer_system_prompt() -> str:
    return (
        "You are a scoped chart-code repair assistant. "
        "You must preserve all satisfied requirements and only repair the failed requirements described in the report. "
        "Treat parser-plan fields as context, not repair targets, unless they are repeated in verifier errors or failure atoms. "
        "Do not change chart type, backend, artist family, data source, or unrelated layout unless the verifier errors explicitly target them. "
        "For localized edits, prefer structured patch_ops over full repaired_code. "
        "Supported patch ops are replace_call_arg, replace_keyword_arg, remove_keyword_arg, insert_after_anchor, replace_text. "
        "Return JSON with keys: instruction, repaired_code, patch_ops, loop_signal, loop_reason. "
        "loop_signal must be one of continue, stop, escalate. "
        "Do not add explanations outside JSON."
    )


def _repairer_user_prompt(
    code: str,
    plan: ChartIntentPlan,
    report: VerificationReport,
    repair_plan: RepairPlan,
    escalation_scope: str | None = None,
    evidence_graph: EvidenceGraph | None = None,
) -> str:
    plan_confidence = getattr(plan, "confidence", None)
    chart_type_is_low_confidence = plan_confidence is not None and float(plan_confidence) < 0.5
    plan_payload = {
        "chart_type": None if chart_type_is_low_confidence else plan.chart_type,
        "chart_type_note": (
            "low-confidence parser default omitted; preserve existing chart family unless a failed verifier error targets chart type or artist family"
            if chart_type_is_low_confidence
            else None
        ),
        "confidence": plan_confidence,
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
            "requirement_id": error.requirement_id,
            "severity": error.severity,
            "match_policy": error.match_policy,
        }
        for error in report.errors
    ]
    preservation_section = _repair_preservation_section(report)
    repair_payload = {
        "repair_level": repair_plan.repair_level,
        "scope": repair_plan.scope,
        "allowed_edits": list(repair_plan.allowed_edits),
        "forbidden_edits": list(repair_plan.forbidden_edits),
        "reason": repair_plan.reason,
    }
    failure_atom_payload = failure_atoms_to_dicts(failure_atoms_from_evidence_graph(evidence_graph))
    failure_atom_section = ""
    if failure_atom_payload:
        failure_atom_section = (
            "Evidence-grounded failure atoms:\n"
            f"{json.dumps(failure_atom_payload, ensure_ascii=False, indent=2)}\n\n"
            "Use these atoms as the primary repair target: match expected artifacts, inspect actual artifacts, "
            "and preserve requirements that are not listed here.\n\n"
        )
    escalation_note = (
        f"Previous round escalation request: broaden to {escalation_scope} because prior repair made insufficient progress.\n\n"
        if escalation_scope
        else ""
    )
    return (
        "Parser context (not repair targets unless repeated in failed verifier errors or failure atoms):\n"
        f"{plan_payload}\n\n"
        "Verification errors:\n"
        f"{error_payload}\n\n"
        f"{failure_atom_section}"
        "Repair constraints:\n"
        f"{repair_payload}\n\n"
        "Preservation policy:\n"
        f"{preservation_section}\n\n"
        f"{escalation_note}"
        "Loop signaling policy:\n"
        "- continue: this patch is still on-track and another verifier-guided round could help if failures remain.\n"
        "- stop: further automatic repair would likely be speculative, unsafe, or wasteful.\n"
        "- escalate: the current failure likely needs a broader rewrite/regeneration next round.\n\n"
        "Structured patch guidance:\n"
        "- For local_patch scopes, prefer patch_ops that touch only the failed call/keyword/text anchor.\n"
        "- patch_ops format: [{'op': 'replace_call_arg', 'anchor': {'kind': 'method_call', 'name': 'set_title', 'occurrence': 1}, 'arg_index': 0, 'new_value': 'Expected Title'}]\n"
        "- Matplotlib setter methods such as set_title/set_xlabel/set_ylabel/set_zlabel/set_*ticklabels take the requested text/list as positional arg_index=0; do not use replace_keyword_arg with title/xlabel/ylabel/zlabel.\n"
        "- Plotly fig.update_layout should use replace_keyword_arg with keyword 'title', 'annotations', 'xaxis_title', or 'yaxis_title'; do not use replace_call_arg with one large dict.\n"
        "- If an execution error says an unexpected keyword argument is unsupported, use remove_keyword_arg for that exact keyword instead of setting it to null/None.\n"
        "- Sort fix example: {'op': 'replace_keyword_arg', 'anchor': {'kind': 'function_call', 'name': 'sorted', 'occurrence': 1}, 'keyword': 'reverse', 'new_value': False}.\n"
        "- Legend fix example: add label with replace_keyword_arg on bar/plot/scatter, then insert_after_anchor with new_value='ax.legend()'.\n"
        "- Annotation fix example: insert_after_anchor with a concrete ax.annotate(...) or ax.text(...) line.\n"
        "- Projection fix example: replace_keyword_arg on subplots/add_subplot with keyword 'subplot_kw' or 'projection'.\n"
        "- insert_after_anchor can use method_call/function_call anchors by name; do not leave new_value empty.\n"
        "- Use repaired_code only when a safe localized patch cannot express the repair.\n\n"
        "Original code:\n"
        f"{code}\n"
    )


def _normalize_loop_signal(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"continue", "stop", "escalate"}:
        return normalized
    return None


_STRUCTURAL_ERROR_CODES = frozenset(
    {
        "wrong_chart_type",
        "missing_artist_type",
        "wrong_artist_count",
        "insufficient_artist_count",
    }
)

_LAYOUT_ERROR_CODES = frozenset({"wrong_axis_layout", "wrong_projection"})


def _repair_preservation_section(report: VerificationReport) -> str:
    codes = set(report.error_codes)
    lines = [
        "Treat verification errors and figure requirements as authoritative for this repair; the predicted requirement plan may be incomplete or wrong.",
        "Preserve every requirement that is not listed as failed by the verifier or by any provided failure atom.",
        "Do not use non-failed parser defaults, such as an assumed chart_type, as a reason to rewrite the chart.",
    ]
    if not (codes & _STRUCTURAL_ERROR_CODES):
        lines.append(
            "Do not change chart type or visual artist family; for example, keep a sunburst as sunburst and a 3D bar chart as 3D bar. A repaired_code that changes chart constructors without a structural chart/artist error will be rejected by the safety gate."
        )
    if not (codes & _LAYOUT_ERROR_CODES):
        lines.append("Do not change subplot layout, projection, or backend unless required by an execution error.")
    if "execution_error" not in codes:
        lines.append("Do not switch plotting backend or replace the whole chart to fix text-only/style-only failures.")
    return "\n".join(f"- {line}" for line in lines)


_POSITIONAL_SETTER_NAMES = frozenset(
    {
        "set_title",
        "suptitle",
        "set_xlabel",
        "set_ylabel",
        "set_zlabel",
        "set_xticklabels",
        "set_yticklabels",
        "set_zticklabels",
    }
)


def _normalize_llm_patch_ops(
    patch_ops: tuple[PatchOperation, ...],
    *,
    report: VerificationReport,
) -> tuple[PatchOperation, ...]:
    """Map common LLM patch-op slips onto the safe patch substrate.

    This deliberately normalizes operation shape only. It does not invent new
    anchors or broaden the repair scope, so strict policy validation still owns
    whether an operation is allowed for the current verifier errors.
    """
    if not patch_ops:
        return ()
    unsupported_keywords = _execution_error_unsupported_keywords(report)
    normalized_ops: list[PatchOperation] = []
    for operation in patch_ops:
        normalized_new_value = _normalize_llm_new_value(operation.new_value)
        normalized = operation
        if operation.new_value is not normalized_new_value:
            normalized = replace(normalized, new_value=normalized_new_value)

        anchor_name = str(getattr(operation.anchor, "name", "") or "").strip()
        if normalized.op == "replace_keyword_arg" and anchor_name in _POSITIONAL_SETTER_NAMES:
            normalized = replace(
                normalized,
                op="replace_call_arg",
                arg_index=0 if normalized.arg_index is None else normalized.arg_index,
                keyword=None,
                new_value=normalized_new_value,
                description=normalized.description or "Normalized setter keyword patch to positional arg_index=0.",
            )
        elif (
            normalized.op == "replace_call_arg"
            and anchor_name == "update_layout"
            and normalized.arg_index in {0, None}
            and isinstance(normalized_new_value, dict)
        ):
            layout_keywords = _plotly_layout_keywords_for_errors(report.error_codes)
            emitted = False
            for keyword, value in normalized_new_value.items():
                if str(keyword) not in layout_keywords:
                    continue
                normalized_ops.append(
                    replace(
                        normalized,
                        op="replace_keyword_arg",
                        arg_index=None,
                        keyword=str(keyword),
                        new_value=value,
                        description=normalized.description or "Normalized Plotly update_layout dict patch to keyword patch.",
                    )
                )
                emitted = True
            if emitted:
                continue
        elif (
            normalized.op == "replace_keyword_arg"
            and _is_none_like(normalized_new_value)
            and normalized.keyword in unsupported_keywords
        ):
            normalized = replace(
                normalized,
                op="remove_keyword_arg",
                new_value=None,
                description=normalized.description or "Normalized null keyword replacement to unsupported keyword removal.",
            )
        elif (
            normalized.op in {"replace_keyword_arg", "remove_keyword_arg"}
            and not normalized.keyword
            and len(unsupported_keywords) == 1
            and (normalized.op == "remove_keyword_arg" or _is_none_like(normalized_new_value))
        ):
            keyword = next(iter(unsupported_keywords))
            normalized = replace(
                normalized,
                op="remove_keyword_arg",
                keyword=keyword,
                new_value=None,
                description=normalized.description or f"Inferred unsupported keyword '{keyword}' from execution error.",
            )
        normalized_ops.append(normalized)
    return tuple(normalized_ops)


def _execution_error_unsupported_keywords(report: VerificationReport) -> frozenset[str]:
    keywords: list[str] = []
    for error in report.errors:
        if error.code != "execution_error":
            continue
        keyword = _extract_unsupported_keyword(error.message)
        if keyword:
            keywords.append(keyword)
    return frozenset(dict.fromkeys(keywords))


def _plotly_layout_keywords_for_errors(error_codes: tuple[str, ...]) -> frozenset[str]:
    keywords: set[str] = set()
    codes = set(error_codes)
    if codes & {"wrong_figure_title", "wrong_axis_title"}:
        keywords.update({"title", "title_text"})
    if "missing_annotation_text" in codes:
        keywords.add("annotations")
    if "wrong_x_label" in codes:
        keywords.add("xaxis_title")
    if "wrong_y_label" in codes:
        keywords.add("yaxis_title")
    if "wrong_z_label" in codes:
        keywords.update({"zaxis_title", "scene_zaxis_title"})
    return frozenset(keywords)


def _normalize_llm_new_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    should_parse_literal = (
        (len(stripped) >= 2 and stripped[0] in {"'", '"'} and stripped[-1] == stripped[0])
        or (stripped[0] in {"[", "{", "("} and stripped[-1] in {"]", "}", ")"})
    )
    if not should_parse_literal:
        return value
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value
    if isinstance(parsed, tuple):
        return list(parsed)
    if isinstance(parsed, (str, int, float, bool, list, dict)) or parsed is None:
        return parsed
    return value


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"none", "null"}


def _complete_json_with_trace(client: LLMClient, **kwargs: Any) -> LLMJsonResult:
    traced = getattr(client, "complete_json_with_trace", None)
    if callable(traced):
        result = traced(**kwargs)
        if isinstance(result, LLMJsonResult):
            return result
        payload = getattr(result, "payload", None)
        trace = getattr(result, "trace", None)
        if isinstance(payload, dict):
            return LLMJsonResult(payload=dict(payload), trace=trace)
        if isinstance(result, dict):
            return LLMJsonResult(payload=dict(result), trace=None)
    return LLMJsonResult(payload=dict(client.complete_json(**kwargs)), trace=None)


def _axis_occurrence(requirement_id: str | None) -> int:
    match = re.search(r"axis_(\d+)", str(requirement_id or ""))
    if match:
        return int(match.group(1)) + 1
    return 1


def _extract_unsupported_keyword(message: str | None) -> str | None:
    match = re.search(r"unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", str(message or ""))
    if match:
        return match.group(1)
    return None


def _infer_keyword_patch_anchor(code: str, keyword: str | None) -> PatchAnchor | None:
    normalized_keyword = str(keyword or "").strip()
    if not normalized_keyword:
        return None
    method_anchor = _infer_call_anchor(code, keyword=normalized_keyword, kind="method_call")
    if method_anchor is not None:
        return method_anchor
    return _infer_call_anchor(code, keyword=normalized_keyword, kind="function_call")


def _infer_call_anchor(code: str, *, keyword: str, kind: str) -> PatchAnchor | None:
    if kind == "method_call":
        pattern = re.compile(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    else:
        pattern = re.compile(r"(?<!\.)\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    occurrences: dict[str, int] = {}
    for match in pattern.finditer(code):
        name = match.group(1)
        occurrences[name] = occurrences.get(name, 0) + 1
        open_paren = code.find("(", match.start())
        if open_paren == -1:
            continue
        close_paren = _find_matching_paren(code, open_paren)
        if close_paren is None:
            continue
        call_body = code[open_paren + 1 : close_paren]
        if re.search(rf"\b{re.escape(keyword)}\s*=", call_body):
            return PatchAnchor(kind=kind, name=name, occurrence=occurrences[name])
    return None


def _find_matching_paren(text: str, open_index: int) -> int | None:
    depth = 0
    in_string = False
    quote_char = ""
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                in_string = False
                quote_char = ""
            continue
        if char in {"'", '"'}:
            in_string = True
            quote_char = char
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _find_simple_row_projection_snippet(
    code: str,
    *,
    dimension_column: str | None,
    measure_column: str | None,
) -> dict[str, str] | None:
    if not dimension_column or not measure_column:
        return None
    pattern = re.compile(
        r"(?ms)^(?P<dim_var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[row\[(?P<dim_quote>['\"])(?P<dim_col>.+?)(?P=dim_quote)\]\s+for\s+row\s+in\s+rows\]\s*\r?\n"
        r"(?P<measure_var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[row\[(?P<measure_quote>['\"])(?P<measure_col>.+?)(?P=measure_quote)\]\s+for\s+row\s+in\s+rows\]\s*$"
    )
    for match in pattern.finditer(code):
        if match.group("dim_col") != dimension_column or match.group("measure_col") != measure_column:
            continue
        return {
            "text": match.group(0),
            "dimension_var": match.group("dim_var"),
            "measure_var": match.group("measure_var"),
            "dimension_column": match.group("dim_col"),
            "measure_column": match.group("measure_col"),
        }
    return None


def _build_grouped_data_prep_block(
    *,
    dimension_var: str,
    measure_var: str,
    dimension_column: str,
    measure_column: str,
    agg: str,
    sort,
    limit: int | None,
) -> str | None:
    normalized_agg = str(agg or "none").strip().lower()
    if normalized_agg not in {"sum", "count", "mean", "min", "max"}:
        return None
    lines: list[str] = []
    if normalized_agg == "sum":
        lines.extend(
            [
                "totals = {}",
                "for row in rows:",
                f"    key = row[{dimension_column!r}]",
                f"    totals[key] = totals.get(key, 0) + row[{measure_column!r}]",
                "items = list(totals.items())",
            ]
        )
    elif normalized_agg == "count":
        lines.extend(
            [
                "counts = {}",
                "for row in rows:",
                f"    key = row[{dimension_column!r}]",
                "    counts[key] = counts.get(key, 0) + 1",
                "items = list(counts.items())",
            ]
        )
    elif normalized_agg == "mean":
        lines.extend(
            [
                "stats = {}",
                "for row in rows:",
                f"    key = row[{dimension_column!r}]",
                f"    value = row[{measure_column!r}]",
                "    total, count = stats.get(key, (0, 0))",
                "    stats[key] = (total + value, count + 1)",
                "items = [(key, total / count) for key, (total, count) in stats.items()]",
            ]
        )
    elif normalized_agg == "min":
        lines.extend(
            [
                "mins = {}",
                "for row in rows:",
                f"    key = row[{dimension_column!r}]",
                f"    value = row[{measure_column!r}]",
                "    mins[key] = value if key not in mins else min(mins[key], value)",
                "items = list(mins.items())",
            ]
        )
    elif normalized_agg == "max":
        lines.extend(
            [
                "maxs = {}",
                "for row in rows:",
                f"    key = row[{dimension_column!r}]",
                f"    value = row[{measure_column!r}]",
                "    maxs[key] = value if key not in maxs else max(maxs[key], value)",
                "items = list(maxs.items())",
            ]
        )
    sort_key = "item[0]"
    if sort is not None and str(getattr(sort, "by", "")).strip().lower() == "measure":
        sort_key = "item[1]"
    reverse = bool(sort is not None and str(getattr(sort, "direction", "")).strip().lower() == "desc")
    lines.append(f"items = sorted(items, key=lambda item: {sort_key}, reverse={str(reverse)})")
    if limit is not None and int(limit) > 0:
        lines.append(f"items = items[:{int(limit)}]")
    lines.extend(
        [
            f"{dimension_var} = [key for key, value in items]",
            f"{measure_var} = [value for key, value in items]",
        ]
    )
    return dedent("\n".join(lines))
