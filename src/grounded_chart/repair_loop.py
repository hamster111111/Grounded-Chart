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
                    requirement_plan=current_result.requirement_plan,
                    expected_figure=expected_figure,
                    actual_figure=current_result.actual_figure,
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

            rerun = self.trace_runner.run_code_with_figure(
                new_code,
                globals_dict={"rows": list(rows)},
                execution_dir=execution_dir,
                file_path=file_path,
            )
            next_report = self.verifier.verify(
                current_result.expected_trace,
                rerun.plot_trace,
                expected_figure=expected_figure,
                actual_figure=rerun.figure_trace,
                verify_data=verify_data,
                enforce_order=current_result.plan.sort is not None,
            )
            next_requirement_plan = build_requirement_plan(current_result.plan, expected_figure=expected_figure)
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
        requirement_plan,
        expected_figure: FigureRequirementSpec | None,
        actual_figure: FigureTrace | None,
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

        if not applied and "backend-specific" in instruction.lower() and expected_figure.axes:
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


def _append_missing_text(code: str, required_text: str) -> tuple[str, bool]:
    if required_text in code:
        return code, False
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
        pattern = re.compile(r"(title\s*=\s*)(['\"])(.*?)(\2)")
        updated, count = pattern.subn(lambda match: f"{match.group(1)}{match.group(2)}{title}{match.group(2)}", updated, count=1)
        applied = applied or count > 0
    elif title and "fig = " in updated:
        updated += f"\nfig.update_layout(title={title!r})\n"
        applied = True
    if subtitle and subtitle not in updated:
        updated += f"\nfig.add_annotation(text={subtitle!r}, x=0.5, y=1.02, xref='paper', yref='paper', showarrow=False)\n"
        applied = True
    return updated, applied


def _diff_failed_requirements(previous: tuple[str, ...], current: tuple[str, ...]) -> tuple[str, ...]:
    current_set = set(current)
    return tuple(requirement_id for requirement_id in previous if requirement_id not in current_set)
