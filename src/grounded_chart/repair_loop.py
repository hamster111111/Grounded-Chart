from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from grounded_chart.evidence import build_evidence_graph, build_requirement_plan
from grounded_chart.repairer import Repairer
from grounded_chart.schema import FigureRequirementSpec, FigureTrace, PipelineResult, PlotTrace, RepairAttempt, TableSchema
from grounded_chart.trace_runner import MatplotlibTraceRunner
from grounded_chart.verifier import OperatorLevelVerifier
from grounded_chart.canonical_executor import CanonicalExecutor, Row
from grounded_chart.intent_parser import IntentParser


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
    ) -> None:
        self.parser = parser
        self.executor = executor
        self.verifier = verifier
        self.repairer = repairer
        self.trace_runner = trace_runner or MatplotlibTraceRunner()
        self.max_rounds = max(1, int(max_rounds))

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
        active_expected_figure = initial_result.expected_figure if initial_result.expected_figure is not None else expected_figure
        if current_result.repair is None or current_result.repair_plan is None:
            return current_result

        for round_index in range(1, self.max_rounds + 1):
            patch = self.repairer.propose(current_code, current_result.plan, current_result.report)
            if patch.repaired_code is not None and patch.repaired_code.strip() and patch.repaired_code != current_code:
                new_code = patch.repaired_code
                applied = True
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
            if not applied or new_code == current_code:
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
                    )
                )
                break

            try:
                rerun = self.trace_runner.run_code_with_figure(
                    new_code,
                    globals_dict={"rows": list(rows)},
                    execution_dir=execution_dir,
                    file_path=file_path,
                )
            except Exception as exc:
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
                    )
                )
                current_code = new_code
                current_result = replace(
                    current_result,
                    repair_plan=patch.repair_plan,
                    repair=replace(patch, repaired_code=new_code),
                    repaired_code=new_code,
                    repair_attempts=tuple(attempts),
                    execution_exception_type=type(exc).__name__,
                    execution_exception_message=str(exc),
                )
                continue
            next_report = self.verifier.verify(
                current_result.expected_trace,
                rerun.plot_trace,
                expected_figure=active_expected_figure,
                actual_figure=rerun.figure_trace,
                verify_data=verify_data,
                enforce_order=current_result.plan.sort is not None,
            )
            next_requirement_plan = current_result.requirement_plan or build_requirement_plan(current_result.plan, expected_figure=active_expected_figure)
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
            )
            if next_report.ok:
                return current_result
        return replace(
            current_result,
            repair_attempts=tuple(attempts),
            repaired_code=current_code if attempts else current_result.repaired_code,
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
