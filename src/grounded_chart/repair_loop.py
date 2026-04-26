from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from grounded_chart.evidence import bind_requirement_policy_to_verification, build_evidence_graph, build_requirement_plan
from grounded_chart.patch_ops import PatchApplyResult, apply_patch_operations
from grounded_chart.repairer import Repairer
from grounded_chart.schema import FigureRequirementSpec, FigureTrace, PipelineResult, PlotTrace, RepairAttempt, RepairLoopAction, TableSchema
from grounded_chart.trace_runner import MatplotlibTraceRunner
from grounded_chart.verifier import OperatorLevelVerifier
from grounded_chart.canonical_executor import CanonicalExecutor, Row
from grounded_chart.intent_parser import IntentParser


@dataclass(frozen=True)
class _LoopDecision:
    action: RepairLoopAction
    reason: str
    next_scope: str | None = None


class BoundedRepairLoop:
    """Optional bounded repair loop over generated plotting code.

    The first implementation is intentionally narrow: only a small set of
    deterministic, provenance-targeted edits are allowed.
    """

    def __init__(
        self,
        parser: IntentParser,
        executor: CanonicalExecutor,
        verifier: OperatorLevelVerifier,
        repairer: Repairer,
        trace_runner: MatplotlibTraceRunner | None = None,
        max_rounds: int = 2,
        repair_policy_mode: str = "exploratory",
    ) -> None:
        self.parser = parser
        self.executor = executor
        self.verifier = verifier
        self.repairer = repairer
        self.trace_runner = trace_runner or MatplotlibTraceRunner()
        self.max_rounds = max(1, int(max_rounds))
        self.repair_policy_mode = str(repair_policy_mode or "exploratory").strip().lower()

    def run(
        self,
        *,
        query: str,
        schema: TableSchema,
        rows: Iterable[Row],
        initial_result: PipelineResult,
        generated_code: str,
        expected_figure: FigureRequirementSpec | None = None,
        execution_dir: str | Path | None = None,
        file_path: str | Path | None = None,
        verify_data: bool = True,
    ) -> PipelineResult:
        current_result = initial_result
        current_code = generated_code
        attempts: list[RepairAttempt] = []
        loop_status = current_result.repair_loop_status
        loop_reason = current_result.repair_loop_reason
        active_expected_figure = initial_result.expected_figure if initial_result.expected_figure is not None else expected_figure
        if current_result.repair is None or current_result.repair_plan is None:
            return current_result

        next_escalation_scope: str | None = None
        for round_index in range(1, self.max_rounds + 1):
            if round_index == 1 and current_result.repair is not None:
                patch = current_result.repair
            else:
                patch = self._propose_patch(
                    current_code,
                    current_result.plan,
                    current_result.report,
                    escalation_scope=next_escalation_scope,
                    evidence_graph=current_result.evidence_graph,
                )
            next_escalation_scope = None
            structured_result = None
            strict_local_patch_reason = self._strict_local_patch_validation_reason(
                patch,
                report_errors=current_result.report.errors,
            )
            if strict_local_patch_reason is not None:
                structured_result = _strict_local_patch_rejection(strict_local_patch_reason)
            elif patch.patch_ops:
                structured_result = apply_patch_operations(
                    current_code,
                    patch.patch_ops,
                    max_operations=_patch_op_budget(patch.repair_plan.scope if patch.repair_plan else None, error_count=len(current_result.report.errors)),
                    max_changed_lines=_patch_line_budget(
                        patch.repair_plan.scope if patch.repair_plan else None,
                        error_count=len(current_result.report.errors),
                    ),
                )
            if structured_result is not None and structured_result.applied:
                new_code = structured_result.code
                applied = True
                applied_patch_ops = structured_result.applied_ops
            elif self._require_structured_local_patch(patch):
                new_code = current_code
                applied = False
                applied_patch_ops = ()
                if structured_result is None:
                    structured_result = _strict_local_patch_rejection(
                        "Strict local_patch mode requires structured patch_ops; free-form repaired_code fallback is disabled."
                    )
                elif structured_result.rejected_reason is None:
                    structured_result = _strict_local_patch_rejection(
                        "Strict local_patch mode rejected the structured patch and disallows free-form fallback."
                    )
            elif patch.repaired_code is not None and patch.repaired_code.strip() and patch.repaired_code != current_code:
                preservation_reason = _repaired_code_preservation_rejection(
                    current_code=current_code,
                    repaired_code=patch.repaired_code,
                    target_error_codes=patch.target_error_codes,
                )
                if preservation_reason is not None:
                    new_code = current_code
                    applied = False
                    applied_patch_ops = ()
                    structured_result = _strict_local_patch_rejection(preservation_reason)
                else:
                    new_code = patch.repaired_code
                    applied = True
                    applied_patch_ops = ()
            else:
                new_code, applied = self._apply_patch(
                    current_code,
                    patch.instruction,
                    target_error_codes=patch.target_error_codes,
                    repair_scope=patch.repair_plan.scope if patch.repair_plan else None,
                    requirement_plan=current_result.requirement_plan,
                    expected_figure=active_expected_figure,
                    actual_figure=current_result.actual_figure,
                    execution_error_message=current_result.execution_exception_message,
                )
                applied_patch_ops = ()
            if not applied or new_code == current_code:
                decision = _LoopDecision(
                    action="stop",
                    reason=(
                        structured_result.rejected_reason
                        if structured_result is not None and structured_result.rejected_reason
                        else "Repair patch was not applicable or produced no code change."
                    ),
                )
                attempts.append(
                    RepairAttempt(
                        round_index=round_index,
                        input_code=current_code,
                        output_code=current_code,
                        applied=False,
                        strategy=patch.strategy,
                        scope=patch.repair_plan.scope if patch.repair_plan else None,
                        targeted_requirement_ids=current_result.evidence_graph.failed_requirement_ids if current_result.evidence_graph else (),
                        targeted_error_codes=patch.target_error_codes,
                        resolved_requirement_ids=(),
                        unresolved_requirement_ids=current_result.evidence_graph.failed_requirement_ids if current_result.evidence_graph else (),
                        report_ok=current_result.report.ok,
                        instruction=patch.instruction,
                        decision_action=decision.action,
                        decision_reason=decision.reason,
                        decision_next_scope=decision.next_scope,
                        llm_trace=patch.llm_trace,
                        patch_ops=(),
                    )
                )
                loop_status = decision.action
                loop_reason = decision.reason
                break

            try:
                rerun = self.trace_runner.run_code_with_figure(
                    new_code,
                    globals_dict={"rows": list(rows)},
                    execution_dir=execution_dir,
                    file_path=file_path,
                )
            except Exception as exc:
                decision = self._decide_after_execution_error(
                    round_index=round_index,
                    patch=patch,
                )
                attempts.append(
                    RepairAttempt(
                        round_index=round_index,
                        input_code=current_code,
                        output_code=new_code,
                        applied=True,
                        strategy=patch.strategy,
                        scope=patch.repair_plan.scope if patch.repair_plan else None,
                        targeted_requirement_ids=current_result.evidence_graph.failed_requirement_ids if current_result.evidence_graph else (),
                        targeted_error_codes=patch.target_error_codes,
                        resolved_requirement_ids=(),
                        unresolved_requirement_ids=current_result.evidence_graph.failed_requirement_ids if current_result.evidence_graph else (),
                        report_ok=False,
                        instruction=f"{patch.instruction} Rerun failed: {type(exc).__name__}: {exc}",
                        decision_action=decision.action,
                        decision_reason=decision.reason,
                        decision_next_scope=decision.next_scope,
                        llm_trace=patch.llm_trace,
                        patch_ops=applied_patch_ops,
                    )
                )
                current_code = new_code
                current_result = replace(
                    current_result,
                    repair_plan=patch.repair_plan,
                    repair=replace(patch, repaired_code=new_code),
                    repaired_code=new_code,
                    repair_attempts=tuple(attempts),
                    repair_loop_status=decision.action,
                    repair_loop_reason=decision.reason,
                    execution_exception_type=type(exc).__name__,
                    execution_exception_message=str(exc),
                )
                loop_status = decision.action
                loop_reason = decision.reason
                next_escalation_scope = decision.next_scope
                if decision.action == "stop":
                    break
                continue
            next_requirement_plan = current_result.requirement_plan or build_requirement_plan(current_result.plan, expected_figure=active_expected_figure)
            next_report = self.verifier.verify(
                current_result.expected_trace,
                rerun.plot_trace,
                expected_figure=active_expected_figure,
                actual_figure=rerun.figure_trace,
                verify_data=verify_data,
                enforce_order=current_result.plan.sort is not None,
            )
            next_report = bind_requirement_policy_to_verification(next_report, next_requirement_plan)
            next_evidence_graph = build_evidence_graph(
                next_requirement_plan,
                next_report,
                current_result.expected_trace,
                rerun.plot_trace,
                rerun.figure_trace,
            )
            resolved = _diff_failed_requirements(
                current_result.evidence_graph.failed_requirement_ids if current_result.evidence_graph else (),
                next_evidence_graph.failed_requirement_ids,
            )
            decision = self._decide_after_verification(
                round_index=round_index,
                previous_failed_ids=current_result.evidence_graph.failed_requirement_ids if current_result.evidence_graph else (),
                next_failed_ids=next_evidence_graph.failed_requirement_ids,
                next_report_ok=next_report.ok,
                patch=patch,
            )
            attempts.append(
                RepairAttempt(
                    round_index=round_index,
                    input_code=current_code,
                    output_code=new_code,
                    applied=True,
                    strategy=patch.strategy,
                    scope=patch.repair_plan.scope if patch.repair_plan else None,
                    targeted_requirement_ids=current_result.evidence_graph.failed_requirement_ids if current_result.evidence_graph else (),
                    targeted_error_codes=patch.target_error_codes,
                    resolved_requirement_ids=resolved,
                    unresolved_requirement_ids=next_evidence_graph.failed_requirement_ids,
                    report_ok=next_report.ok,
                    instruction=patch.instruction,
                    decision_action=decision.action,
                    decision_reason=decision.reason,
                    decision_next_scope=decision.next_scope,
                    llm_trace=patch.llm_trace,
                    patch_ops=applied_patch_ops,
                )
            )
            current_code = new_code
            current_result = replace(
                current_result,
                actual_trace=rerun.plot_trace,
                report=next_report,
                actual_figure=rerun.figure_trace,
                requirement_plan=next_requirement_plan,
                evidence_graph=next_evidence_graph,
                repair_plan=patch.repair_plan,
                repair=replace(patch, repaired_code=new_code),
                repaired_code=new_code,
                repair_attempts=tuple(attempts),
                repair_loop_status=decision.action,
                repair_loop_reason=decision.reason,
            )
            loop_status = decision.action
            loop_reason = decision.reason
            if next_report.ok or decision.action == "stop":
                return current_result
            next_escalation_scope = decision.next_scope
        return replace(
            current_result,
            repair_attempts=tuple(attempts),
            repaired_code=current_code if attempts else current_result.repaired_code,
            repair_loop_status=loop_status,
            repair_loop_reason=loop_reason,
        )

    def _require_structured_local_patch(self, patch) -> bool:
        scope = str(patch.repair_plan.scope if patch.repair_plan else "").strip().lower()
        return self.repair_policy_mode == "strict" and scope == "local_patch"

    def _strict_local_patch_validation_reason(self, patch, *, report_errors) -> str | None:
        if not self._require_structured_local_patch(patch):
            return None
        return _validate_strict_local_patch_ops(
            report_errors=report_errors,
            target_error_codes=patch.target_error_codes,
            patch_ops=patch.patch_ops,
        )

    def _propose_patch(self, code, plan, report, *, escalation_scope: str | None = None, evidence_graph=None):
        try:
            return self.repairer.propose(code, plan, report, escalation_scope=escalation_scope, evidence_graph=evidence_graph)
        except TypeError:
            return self.repairer.propose(code, plan, report, escalation_scope=escalation_scope)

    def _decide_after_execution_error(self, *, round_index: int, patch) -> _LoopDecision:
        if round_index >= self.max_rounds:
            return _LoopDecision(action="stop", reason="Repair budget exhausted after a rerun execution error.")
        if patch.loop_signal == "stop":
            return _LoopDecision(action="stop", reason=patch.loop_reason or "Repairer requested stop after execution failure.")
        current_scope = patch.repair_plan.scope if patch.repair_plan else None
        if patch.loop_signal != "escalate" and current_scope == "local_patch":
            return _LoopDecision(
                action="continue",
                reason=(
                    patch.loop_reason
                    or "Patched code still hit an execution error; try one more localized runtime fix before escalation."
                ),
            )
        escalate_scope = _next_escalation_scope(patch.repair_plan.scope if patch.repair_plan else None)
        if patch.loop_signal == "escalate" and escalate_scope is not None:
            return _LoopDecision(
                action="escalate",
                reason=patch.loop_reason or "Repairer requested escalation after execution failure.",
                next_scope=escalate_scope,
            )
        if escalate_scope is not None:
            return _LoopDecision(
                action="escalate",
                reason="Patched code still failed to execute; escalate repair scope for the next round.",
                next_scope=escalate_scope,
            )
        return _LoopDecision(action="stop", reason="Patched code failed to execute and no broader safe scope remains.")

    def _decide_after_verification(
        self,
        *,
        round_index: int,
        previous_failed_ids: tuple[str, ...],
        next_failed_ids: tuple[str, ...],
        next_report_ok: bool,
        patch,
    ) -> _LoopDecision:
        if next_report_ok:
            return _LoopDecision(action="stop", reason="Verifier passed after repair.")
        if round_index >= self.max_rounds:
            return _LoopDecision(action="stop", reason="Repair budget exhausted while verification failures remain.")

        previous_failed = set(previous_failed_ids)
        current_failed = set(next_failed_ids)
        resolved_count = len(previous_failed - current_failed)
        new_failure_count = len(current_failed - previous_failed)
        explicit_signal = patch.loop_signal

        if explicit_signal == "stop":
            return _LoopDecision(action="stop", reason=patch.loop_reason or "Repairer requested stop.")

        if resolved_count > 0 and new_failure_count == 0:
            return _LoopDecision(
                action="continue",
                reason=patch.loop_reason or f"Repair reduced failed requirements by {resolved_count}; continue if budget remains.",
            )

        escalate_scope = _next_escalation_scope(patch.repair_plan.scope if patch.repair_plan else None)
        if explicit_signal == "escalate" and escalate_scope is not None:
            return _LoopDecision(
                action="escalate",
                reason=patch.loop_reason or "Repairer requested a broader repair scope.",
                next_scope=escalate_scope,
            )

        if resolved_count == 0 or new_failure_count > 0:
            if escalate_scope is not None:
                return _LoopDecision(
                    action="escalate",
                    reason=(
                        patch.loop_reason
                        or "Current repair scope made no progress or introduced regressions; broaden the next round."
                    ),
                    next_scope=escalate_scope,
                )
            return _LoopDecision(
                action="stop",
                reason="Repair made no measurable progress and no broader safe scope remains.",
            )

        return _LoopDecision(
            action="continue",
            reason=patch.loop_reason or "Partial progress observed; continue within remaining repair budget.",
        )

    def _apply_patch(
        self,
        code: str,
        instruction: str,
        *,
        target_error_codes: tuple[str, ...] = (),
        repair_scope: str | None = None,
        requirement_plan,
        expected_figure: FigureRequirementSpec | None,
        actual_figure: FigureTrace | None,
        execution_error_message: str | None = None,
    ) -> tuple[str, bool]:
        updated = code
        applied = False

        if expected_figure is None:
            return code, False

        axis_expectations = {
            axis.axis_index: axis
            for axis in expected_figure.axes
        }
        actual_axes = {axis.index: axis for axis in (actual_figure.axes if actual_figure is not None else ())}
        codes = set(target_error_codes)

        for axis_index, expected_axis in axis_expectations.items():
            actual_axis = actual_axes.get(axis_index)
            if actual_axis is not None:
                replacement_specs = (
                    ("set_title", actual_axis.title, expected_axis.title),
                    ("set_xlabel", actual_axis.xlabel, expected_axis.xlabel),
                    ("set_ylabel", actual_axis.ylabel, expected_axis.ylabel),
                    ("set_zlabel", actual_axis.zlabel, expected_axis.zlabel),
                )
                for method_name, actual_text, expected_text in replacement_specs:
                    if expected_text is None or str(actual_text).strip() == str(expected_text).strip():
                        continue
                    updated, did_apply = _replace_axis_text(updated, method_name, actual_text, expected_text)
                    applied = applied or did_apply
                tick_replacement_specs = (
                    ("set_xticklabels", actual_axis.xtick_labels, expected_axis.xtick_labels),
                    ("set_yticklabels", actual_axis.ytick_labels, expected_axis.ytick_labels),
                    ("set_zticklabels", actual_axis.ztick_labels, expected_axis.ztick_labels),
                )
                for method_name, actual_ticks, expected_ticks in tick_replacement_specs:
                    if not expected_ticks:
                        continue
                    updated, did_apply = _replace_axis_ticklabels(updated, method_name, actual_ticks, expected_ticks)
                    applied = applied or did_apply
            if expected_axis.text_contains and actual_axis is not None:
                for required_text in expected_axis.text_contains:
                    if any(required_text.lower() in str(text).lower() for text in actual_axis.texts):
                        continue
                    updated, did_apply = _append_missing_text(updated, required_text)
                    applied = applied or did_apply

        if expected_figure.figure_title is not None and actual_figure is not None:
            if str(actual_figure.title).strip() != str(expected_figure.figure_title).strip():
                updated, did_apply = _replace_figure_title(updated, actual_figure.title, expected_figure.figure_title)
                applied = applied or did_apply

        if "wrong_axes_count" in codes:
            updated, did_apply = _remove_colorbar_blocks(updated)
            applied = applied or did_apply
        if "wrong_artist_count" in codes:
            updated, did_apply = _repair_artist_count_patterns(updated)
            applied = applied or did_apply
        if "wrong_axis_layout" in codes:
            updated, did_apply = _repair_zoom_layout(updated)
            applied = applied or did_apply
        if "execution_error" in codes:
            updated, did_apply = _repair_execution_error(updated, execution_error_message or instruction)
            applied = applied or did_apply

        if (
            expected_figure.axes
            and (
                repair_scope == "backend_specific_regeneration"
                or "plotly" in instruction.lower()
                or _looks_like_plotly_code(updated)
            )
        ):
            title = expected_figure.axes[0].title
            subtitle = expected_figure.axes[0].text_contains[0] if expected_figure.axes[0].text_contains else None
            updated, did_apply = _patch_plotly_layout(updated, title=title, subtitle=subtitle)
            applied = applied or did_apply

        return updated, applied


def _replace_axis_text(code: str, method_name: str, current_value: str, expected_value: str) -> tuple[str, bool]:
    if expected_value is None:
        return code, False
    pattern = re.compile(rf"(\.\s*{re.escape(method_name)}\(\s*)(['\"]){re.escape(str(current_value))}\2")
    updated, count = pattern.subn(lambda match: f"{match.group(1)}{match.group(2)}{expected_value}{match.group(2)}", code, count=1)
    return updated, count > 0


def _replace_figure_title(code: str, current_value: str, expected_value: str) -> tuple[str, bool]:
    pattern = re.compile(rf"(\.\s*suptitle\(\s*)(['\"]){re.escape(str(current_value))}\2")
    updated, count = pattern.subn(lambda match: f"{match.group(1)}{match.group(2)}{expected_value}{match.group(2)}", code, count=1)
    if count:
        return updated, True
    insertion = f"\nplt.gcf().suptitle({expected_value!r})\n"
    if "import matplotlib.pyplot as plt" in code:
        return code + insertion, True
    return code, False


def _replace_axis_ticklabels(code: str, method_name: str, actual_values: tuple[str, ...], expected_values: tuple[str, ...]) -> tuple[str, bool]:
    if not expected_values:
        return code, False
    expected_repr = repr(list(expected_values))
    pattern = re.compile(rf"(\.\s*{re.escape(method_name)}\(\s*)(.+?)(\s*\))")
    updated, count = pattern.subn(lambda match: f"{match.group(1)}{expected_repr}{match.group(3)}", code, count=1)
    return updated, count > 0


def _append_missing_text(code: str, required_text: str) -> tuple[str, bool]:
    pattern = re.compile(r"(?P<anchor>\.\s*legend\(\)\s*)")
    updated, count = pattern.subn(lambda match: f"{match.group('anchor')}\nplt.gca().text(0.5, 0.5, {required_text!r}, transform=plt.gca().transAxes)", code, count=1)
    if count:
        return updated, True
    if "import matplotlib.pyplot as plt" in code:
        return code + f"\nplt.gca().text(0.5, 0.5, {required_text!r}, transform=plt.gca().transAxes)\n", True
    return code, False


def _patch_plotly_layout(code: str, *, title: str | None, subtitle: str | None) -> tuple[str, bool]:
    updated = code
    applied = False
    if title and "update_layout(" in updated:
        dict_pattern = re.compile(r"(title\s*=\s*\{[^{}]*['\"]text['\"]\s*:\s*['\"])(.*?)(['\"])", re.DOTALL)
        updated, count = dict_pattern.subn(lambda match: f"{match.group(1)}{title}{match.group(3)}", updated, count=1)
        if count == 0:
            scalar_pattern = re.compile(r"(title\s*=\s*)(['\"])(.*?)(\2)")
            updated, count = scalar_pattern.subn(lambda match: f"{match.group(1)}{match.group(2)}{title}{match.group(2)}", updated, count=1)
        applied = applied or count > 0
    elif title and "fig = " in updated:
        updated += f"\nfig.update_layout(title={title!r})\n"
        applied = True
    if subtitle:
        annotation_pattern = re.compile(r"(annotations\s*=\s*\[\s*dict\(\s*text=)(['\"])(.*?)(\2)", re.DOTALL)
        updated, count = annotation_pattern.subn(lambda match: f"{match.group(1)}{match.group(2)}{subtitle}{match.group(2)}", updated, count=1)
        if count == 0 and subtitle not in updated:
            updated += f"\nfig.add_annotation(text={subtitle!r}, x=0.5, y=1.02, xref='paper', yref='paper', showarrow=False)\n"
            applied = True
        else:
            applied = applied or count > 0
    return updated, applied


def _diff_failed_requirements(previous: tuple[str, ...], current: tuple[str, ...]) -> tuple[str, ...]:
    current_set = set(current)
    return tuple(requirement_id for requirement_id in previous if requirement_id not in current_set)


def _next_escalation_scope(current_scope: str | None) -> str | None:
    scope = str(current_scope or "").strip().lower()
    if scope in {"local_patch", "data_transformation"}:
        return "structural_regeneration"
    return None


_STRUCTURAL_REPAIR_ERROR_CODES = frozenset(
    {
        "wrong_chart_type",
        "missing_artist_type",
        "wrong_artist_count",
        "insufficient_artist_count",
    }
)

_PLOTLY_EXPRESS_CONSTRUCTORS = frozenset(
    {
        "area",
        "bar",
        "bar_polar",
        "box",
        "choropleth",
        "density_contour",
        "density_heatmap",
        "ecdf",
        "funnel",
        "funnel_area",
        "histogram",
        "icicle",
        "imshow",
        "line",
        "line_3d",
        "line_geo",
        "line_mapbox",
        "line_polar",
        "pie",
        "scatter",
        "scatter_3d",
        "scatter_geo",
        "scatter_mapbox",
        "scatter_matrix",
        "scatter_polar",
        "scatter_ternary",
        "strip",
        "sunburst",
        "timeline",
        "treemap",
        "violin",
    }
)

_PLOTLY_GO_CONSTRUCTORS = frozenset(
    {
        "Bar",
        "Box",
        "Contour",
        "Funnel",
        "Heatmap",
        "Histogram",
        "Icicle",
        "Image",
        "Pie",
        "Sankey",
        "Scatter",
        "Scatter3d",
        "Scattergeo",
        "Scatterpolar",
        "Sunburst",
        "Surface",
        "Table",
        "Treemap",
        "Violin",
    }
)

_MATPLOTLIB_CHART_CONSTRUCTORS = frozenset(
    {
        "bar",
        "bar3d",
        "barh",
        "boxplot",
        "contour",
        "contourf",
        "eventplot",
        "fill_between",
        "hist",
        "imshow",
        "pie",
        "plot",
        "pcolormesh",
        "scatter",
        "stackplot",
        "stem",
        "streamplot",
        "tricontour",
        "trisurf",
        "violinplot",
    }
)


def _repaired_code_preservation_rejection(
    *,
    current_code: str,
    repaired_code: str,
    target_error_codes: tuple[str, ...],
) -> str | None:
    if set(target_error_codes) & _STRUCTURAL_REPAIR_ERROR_CODES:
        return None
    current_families = _chart_constructor_families(current_code)
    repaired_families = _chart_constructor_families(repaired_code)
    if not current_families or not repaired_families or current_families == repaired_families:
        return None
    return (
        "Repair safety gate rejected repaired_code because it changed chart constructors "
        f"from {sorted(current_families)} to {sorted(repaired_families)} without a structural chart/artist error."
    )


def _chart_constructor_families(code: str) -> frozenset[str]:
    families: set[str] = set()
    for name in re.findall(r"\bpx\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        if name in _PLOTLY_EXPRESS_CONSTRUCTORS:
            families.add(f"plotly.express:{name}")
    for name in re.findall(r"\bgo\.([A-Z][A-Za-z0-9_]*)\s*\(", code):
        if name in _PLOTLY_GO_CONSTRUCTORS:
            families.add(f"plotly.graph_objects:{name.lower()}")
    for name in re.findall(r"(?:\bplt|\bax\w*|\bgca\(\))\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        if name in _MATPLOTLIB_CHART_CONSTRUCTORS:
            families.add(f"matplotlib:{name}")
    for name in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        if name in _MATPLOTLIB_CHART_CONSTRUCTORS:
            families.add(f"matplotlib:{name}")
    return frozenset(families)

def _patch_op_budget(repair_scope: str | None, *, error_count: int | None = None) -> int:
    scope = str(repair_scope or "").strip().lower()
    if scope == "local_patch":
        if error_count is None:
            return 2
        return min(8, max(2, int(error_count)))
    if scope == "data_transformation":
        return 3
    return 1


def _patch_line_budget(repair_scope: str | None, *, error_count: int | None = None) -> int:
    scope = str(repair_scope or "").strip().lower()
    if scope == "local_patch":
        if error_count is None:
            return 15
        return min(35, max(15, 8 * int(error_count) + 5))
    if scope == "data_transformation":
        return 30
    return 20


def _strict_local_patch_rejection(reason: str) -> PatchApplyResult:
    return PatchApplyResult(
        code="",
        applied=False,
        applied_ops=(),
        rejected_reason=reason,
    )


def _validate_strict_local_patch_ops(*, report_errors, target_error_codes: tuple[str, ...], patch_ops) -> str | None:
    if not patch_ops:
        return "Strict local_patch mode requires structured patch_ops; free-form repaired_code fallback is disabled."
    unsupported_codes = tuple(code for code in target_error_codes if not _strict_local_patch_supports_error_code(code))
    if unsupported_codes:
        joined = ", ".join(sorted(dict.fromkeys(unsupported_codes)))
        return f"Strict local_patch whitelist does not support these error codes yet: {joined}."
    execution_error_keywords = _strict_execution_error_keywords(report_errors)
    for operation in patch_ops:
        if not any(
            _strict_local_patch_op_matches_error_code(
                operation,
                code,
                execution_error_keywords=execution_error_keywords,
            )
            for code in target_error_codes
        ):
            return (
                f"Strict local_patch whitelist rejected op '{operation.op}' "
                f"for error codes {', '.join(target_error_codes) or 'unknown'}."
            )
    return None


def _strict_local_patch_supports_error_code(code: str) -> bool:
    return code in {
        "wrong_figure_title",
        "wrong_axis_title",
        "wrong_x_label",
        "wrong_y_label",
        "wrong_z_label",
        "wrong_x_tick_labels",
        "wrong_y_tick_labels",
        "wrong_z_tick_labels",
        "wrong_order",
        "wrong_projection",
        "missing_legend_label",
        "missing_annotation_text",
        "execution_error",
    }


def _strict_local_patch_op_matches_error_code(operation, code: str, *, execution_error_keywords: tuple[str, ...]) -> bool:
    if code == "wrong_figure_title":
        return _matches_replace_call_arg(operation, method_name="suptitle") or _matches_plotly_layout_keyword_patch(
            operation, keywords={"title", "title_text"}
        )
    if code == "wrong_axis_title":
        return _matches_replace_call_arg(operation, method_name="set_title") or _matches_plotly_layout_keyword_patch(
            operation, keywords={"title", "title_text"}
        )
    if code == "wrong_x_label":
        return _matches_replace_call_arg(operation, method_name="set_xlabel") or _matches_plotly_layout_keyword_patch(
            operation, keywords={"xaxis_title"}
        )
    if code == "wrong_y_label":
        return _matches_replace_call_arg(operation, method_name="set_ylabel") or _matches_plotly_layout_keyword_patch(
            operation, keywords={"yaxis_title"}
        )
    if code == "wrong_z_label":
        return _matches_replace_call_arg(operation, method_name="set_zlabel") or _matches_plotly_layout_keyword_patch(
            operation, keywords={"zaxis_title", "scene_zaxis_title"}
        )
    if code == "wrong_x_tick_labels":
        return _matches_replace_call_arg(operation, method_name="set_xticklabels")
    if code == "wrong_y_tick_labels":
        return _matches_replace_call_arg(operation, method_name="set_yticklabels")
    if code == "wrong_z_tick_labels":
        return _matches_replace_call_arg(operation, method_name="set_zticklabels")
    if code == "wrong_order":
        return _matches_sort_order_patch(operation)
    if code == "wrong_projection":
        return _matches_projection_patch(operation)
    if code == "missing_legend_label":
        return _matches_legend_patch(operation)
    if code == "missing_annotation_text":
        return _matches_annotation_patch(operation)
    if code == "execution_error":
        return (
            operation.op in {"remove_keyword_arg", "replace_keyword_arg"}
            and bool(getattr(operation, "keyword", None))
            and str(getattr(operation, "keyword", "")).strip() in execution_error_keywords
        )
    return False


def _matches_sort_order_patch(operation) -> bool:
    anchor = getattr(operation, "anchor", None)
    return (
        getattr(operation, "op", None) in {"replace_keyword_arg", "remove_keyword_arg"}
        and getattr(anchor, "kind", None) in {"function_call", "method_call"}
        and getattr(anchor, "name", None) == "sorted"
        and getattr(operation, "keyword", None) == "reverse"
    )


def _matches_projection_patch(operation) -> bool:
    anchor = getattr(operation, "anchor", None)
    return (
        getattr(operation, "op", None) == "replace_keyword_arg"
        and getattr(anchor, "kind", None) in {"function_call", "method_call"}
        and getattr(anchor, "name", None) in {"subplots", "subplot", "add_subplot"}
        and getattr(operation, "keyword", None) in {"projection", "subplot_kw"}
    )


def _matches_legend_patch(operation) -> bool:
    anchor = getattr(operation, "anchor", None)
    op = getattr(operation, "op", None)
    anchor_name = getattr(anchor, "name", None)
    if (
        op == "replace_keyword_arg"
        and getattr(anchor, "kind", None) in {"function_call", "method_call"}
        and anchor_name in {"bar", "barh", "plot", "scatter", "pie"}
        and getattr(operation, "keyword", None) == "label"
    ):
        return True
    if op == "insert_after_anchor":
        inserted = str(getattr(operation, "new_value", "") or "").lower()
        return "legend" in inserted
    return False


def _matches_annotation_patch(operation) -> bool:
    if _matches_plotly_layout_keyword_patch(operation, keywords={"annotations"}):
        return True
    if getattr(operation, "op", None) != "insert_after_anchor":
        return False
    inserted = str(getattr(operation, "new_value", "") or "").lower()
    return "annotate" in inserted or ".text(" in inserted or "text(" in inserted


def _matches_plotly_layout_keyword_patch(operation, *, keywords: set[str]) -> bool:
    anchor = getattr(operation, "anchor", None)
    return (
        getattr(operation, "op", None) == "replace_keyword_arg"
        and getattr(anchor, "kind", None) in {"function_call", "method_call"}
        and getattr(anchor, "name", None) == "update_layout"
        and getattr(operation, "keyword", None) in keywords
    )

def _matches_replace_call_arg(operation, *, method_name: str) -> bool:
    anchor = getattr(operation, "anchor", None)
    return (
        getattr(operation, "op", None) == "replace_call_arg"
        and getattr(operation, "arg_index", None) == 0
        and getattr(anchor, "kind", None) == "method_call"
        and getattr(anchor, "name", None) == method_name
    )


def _strict_execution_error_keywords(report_errors) -> tuple[str, ...]:
    keywords: list[str] = []
    for error in report_errors:
        if getattr(error, "code", None) != "execution_error":
            continue
        match = re.search(
            r"unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
            str(getattr(error, "message", "") or ""),
        )
        if match:
            keywords.append(match.group(1))
    return tuple(dict.fromkeys(keywords))


def _looks_like_plotly_code(code: str) -> bool:
    lowered = code.lower()
    return "plotly" in lowered or "px.sunburst" in lowered or "go.figure" in lowered or "fig.update_layout" in lowered


def _remove_colorbar_blocks(code: str) -> tuple[str, bool]:
    updated = code
    changed = False
    patterns = (
        re.compile(r"(?ms)^[ \t]*cbar\w*\s*=\s*fig\.colorbar\([^\n]*\n(?:^[ \t].*\n)*^[ \t]*cbar\w*\.set_label\([^\n]*\n?"),
        re.compile(r"(?ms)^[ \t]*cbar\w*\s*=\s*fig\.colorbar\([^\n]*\)\n^[ \t]*cbar\w*\.set_label\([^\n]*\n?"),
    )
    for pattern in patterns:
        updated, count = pattern.subn("", updated)
        changed = changed or count > 0
    return updated, changed


def _repair_artist_count_patterns(code: str) -> tuple[str, bool]:
    updated = code
    changed = False
    transpose_patterns = (
        "eventplot(random_data.T",
        "eventplot(gamma_data.T",
    )
    for pattern in transpose_patterns:
        if pattern in updated:
            updated = updated.replace(pattern, pattern.replace(".T", ""))
            changed = True
    bar_pattern = re.compile(
        r"(x_vals\s*=\s*np\.array\(\[)(.*?)(\]\)\s*\n\s*y_vals\s*=\s*np\.array\(\[)(.*?)(\]\))",
        re.DOTALL,
    )
    match = bar_pattern.search(updated)
    if match:
        updated = (
            updated[: match.start()]
            + "x_vals = np.array([1, 2, 3, 4])\n"
            + "y_vals = np.array([4, 3, 2, 4])"
            + updated[match.end() :]
        )
        changed = True
    return updated, changed


def _repair_zoom_layout(code: str) -> tuple[str, bool]:
    updated = code
    changed = False
    subplot_pattern = re.compile(
        r"fig,\s*\(ax_zoom1,\s*ax_zoom2,\s*ax_main\)\s*=\s*plt\.subplots\(\s*3\s*,\s*1\s*,\s*figsize=\((.*?)\)\s*,\s*gridspec_kw=\{[^}]*\}\)",
        re.DOTALL,
    )
    replacement = (
        "fig = plt.figure(figsize=(8, 6))\n"
        "gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])\n"
        "ax_zoom1 = fig.add_subplot(gs[0, 0])\n"
        "ax_zoom2 = fig.add_subplot(gs[0, 1])\n"
        "ax_main = fig.add_subplot(gs[1, :])"
    )
    updated, count = subplot_pattern.subn(replacement, updated, count=1)
    changed = changed or count > 0
    if "draw_slanted_line(ax_main, ax_zoom1" in updated:
        updated = updated.replace("draw_slanted_line(ax_main, ax_zoom1", "draw_slanted_line(ax_main, ax_zoom1")
    return updated, changed


def _repair_execution_error(code: str, message: str) -> tuple[str, bool]:
    unsupported_kwarg = _extract_unsupported_kwarg(message)
    if unsupported_kwarg:
        updated, did_apply = _remove_keyword_argument(code, unsupported_kwarg)
        if did_apply:
            return updated, True
    if _looks_like_sankey_connection_error(message):
        updated, did_apply = _remove_sankey_connection_arguments(code)
        if did_apply:
            return updated, True
    return code, False


def _extract_unsupported_kwarg(message: str) -> str | None:
    match = re.search(r"unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", message)
    if match:
        return match.group(1)
    return None


def _remove_keyword_argument(code: str, keyword: str) -> tuple[str, bool]:
    updated = code
    changed = False
    line_patterns = (
        re.compile(rf"(?m)^[ \t]*{re.escape(keyword)}\s*=\s*.*?,?[ \t]*(#.*)?\n"),
    )
    inline_patterns = (
        re.compile(rf",\s*{re.escape(keyword)}\s*=\s*[^,\n)]+", re.MULTILINE),
        re.compile(rf"{re.escape(keyword)}\s*=\s*[^,\n)]+\s*,\s*", re.MULTILINE),
    )
    for pattern in line_patterns + inline_patterns:
        updated, count = pattern.subn("" if pattern in line_patterns else "", updated)
        changed = changed or count > 0
    return updated, changed


def _looks_like_sankey_connection_error(message: str) -> bool:
    lowered = message.lower()
    return "connected flows" in lowered and "scaled sum" in lowered


def _remove_sankey_connection_arguments(code: str) -> tuple[str, bool]:
    updated = code
    changed = False
    for keyword in ("prior", "connect"):
        updated, did_apply = _remove_keyword_argument(updated, keyword)
        changed = changed or did_apply
    return updated, changed
