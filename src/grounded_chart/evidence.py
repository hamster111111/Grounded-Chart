from __future__ import annotations

from typing import Any

from grounded_chart.requirements import Artifact, ChartRequirementPlan, EvidenceGraph, EvidenceLink, PanelRequirementPlan, RequirementNode
from grounded_chart.schema import ChartIntentPlan, FigureRequirementSpec, FigureTrace, PlotTrace, VerificationReport


def build_requirement_plan(
    plan: ChartIntentPlan,
    expected_figure: FigureRequirementSpec | None = None,
) -> ChartRequirementPlan:
    requirements: list[RequirementNode] = []
    panel_id = "panel_0"
    panel_requirement_ids: list[str] = []
    shared_requirement_ids: list[str] = []
    figure_requirement_payload: dict[str, Any] = {}

    def add_requirement(
        requirement_id: str,
        scope: str,
        type_: str,
        name: str,
        value: Any,
        source_span: str,
        *,
        panel_ref: str | None = None,
        status: str = "explicit",
        priority: str = "core",
    ) -> None:
        requirements.append(
            RequirementNode(
                requirement_id=requirement_id,
                scope=scope,
                type=type_,
                name=name,
                value=value,
                source_span=source_span,
                status=status,
                confidence=plan.confidence,
                priority=priority,
                panel_id=panel_ref,
            )
        )
        if panel_ref == panel_id:
            panel_requirement_ids.append(requirement_id)

    add_requirement(
        "panel_0.chart_type",
        "panel",
        "encoding",
        "chart_type",
        plan.chart_type,
        plan.raw_query or plan.chart_type,
        panel_ref=panel_id,
    )
    add_requirement(
        "panel_0.aggregation",
        "panel",
        "data_operation",
        "aggregation",
        plan.measure.agg,
        plan.measure.agg,
        panel_ref=panel_id,
    )
    if plan.measure.column is not None:
        add_requirement(
            "panel_0.measure_column",
            "panel",
            "data_operation",
            "measure_column",
            plan.measure.column,
            plan.measure.column,
            panel_ref=panel_id,
        )
    if plan.dimensions:
        add_requirement(
            "panel_0.dimensions",
            "panel",
            "data_operation",
            "dimensions",
            tuple(plan.dimensions),
            ", ".join(plan.dimensions),
            panel_ref=panel_id,
        )
    if plan.sort is not None:
        add_requirement(
            "panel_0.sort",
            "panel",
            "presentation_constraint",
            "sort",
            {"by": plan.sort.by, "direction": plan.sort.direction},
            f"{plan.sort.by}:{plan.sort.direction}",
            panel_ref=panel_id,
        )
    if plan.limit is not None:
        add_requirement(
            "panel_0.limit",
            "panel",
            "presentation_constraint",
            "limit",
            plan.limit,
            str(plan.limit),
            panel_ref=panel_id,
        )
    for index, filter_spec in enumerate(plan.filters):
        add_requirement(
            f"panel_0.filter_{index}",
            "panel",
            "data_operation",
            "filter",
            {"column": filter_spec.column, "op": filter_spec.op, "value": filter_spec.value},
            f"{filter_spec.column} {filter_spec.op} {filter_spec.value}",
            panel_ref=panel_id,
        )

    if expected_figure is not None:
        if expected_figure.axes_count is not None:
            add_requirement(
                "figure.axes_count",
                "figure",
                "figure_composition",
                "axes_count",
                expected_figure.axes_count,
                str(expected_figure.axes_count),
            )
            figure_requirement_payload["axes_count"] = expected_figure.axes_count
        if expected_figure.figure_title is not None:
            add_requirement(
                "figure.title",
                "figure",
                "annotation",
                "figure_title",
                expected_figure.figure_title,
                expected_figure.figure_title,
            )
            figure_requirement_payload["figure_title"] = expected_figure.figure_title
        if expected_figure.size_inches is not None:
            add_requirement(
                "figure.size_inches",
                "figure",
                "presentation_constraint",
                "size_inches",
                tuple(expected_figure.size_inches),
                str(tuple(expected_figure.size_inches)),
            )
            figure_requirement_payload["size_inches"] = tuple(expected_figure.size_inches)
        for axis in expected_figure.axes:
            axis_prefix = f"panel_0.axis_{axis.axis_index}"
            axis_fields = (
                ("title", axis.title, "annotation"),
                ("xlabel", axis.xlabel, "annotation"),
                ("ylabel", axis.ylabel, "annotation"),
                ("zlabel", axis.zlabel, "annotation"),
                ("projection", axis.projection, "presentation_constraint"),
                ("xscale", axis.xscale, "presentation_constraint"),
                ("yscale", axis.yscale, "presentation_constraint"),
                ("zscale", axis.zscale, "presentation_constraint"),
                ("bounds", axis.bounds, "figure_composition"),
            )
            for field_name, field_value, req_type in axis_fields:
                if field_value is None:
                    continue
                add_requirement(
                    f"{axis_prefix}.{field_name}",
                    "panel",
                    req_type,
                    field_name,
                    field_value,
                    str(field_value),
                    panel_ref=panel_id,
                )
            if axis.legend_labels:
                add_requirement(
                    f"{axis_prefix}.legend_labels",
                    "panel",
                    "annotation",
                    "legend_labels",
                    tuple(axis.legend_labels),
                    ", ".join(axis.legend_labels),
                    panel_ref=panel_id,
                )
            if axis.artist_types:
                add_requirement(
                    f"{axis_prefix}.artist_types",
                    "panel",
                    "encoding",
                    "artist_types",
                    tuple(axis.artist_types),
                    ", ".join(axis.artist_types),
                    panel_ref=panel_id,
                )
            if axis.artist_counts:
                add_requirement(
                    f"{axis_prefix}.artist_counts",
                    "panel",
                    "encoding",
                    "artist_counts",
                    dict(axis.artist_counts),
                    str(dict(axis.artist_counts)),
                    panel_ref=panel_id,
                )
            if axis.min_artist_counts:
                add_requirement(
                    f"{axis_prefix}.min_artist_counts",
                    "panel",
                    "encoding",
                    "min_artist_counts",
                    dict(axis.min_artist_counts),
                    str(dict(axis.min_artist_counts)),
                    panel_ref=panel_id,
                )
            if axis.text_contains:
                add_requirement(
                    f"{axis_prefix}.text_contains",
                    "panel",
                    "annotation",
                    "text_contains",
                    tuple(axis.text_contains),
                    ", ".join(axis.text_contains),
                    panel_ref=panel_id,
                )

    panel = PanelRequirementPlan(
        panel_id=panel_id,
        chart_type=plan.chart_type,
        requirement_ids=tuple(panel_requirement_ids),
        data_ops={
            "dimensions": tuple(plan.dimensions),
            "measure_column": plan.measure.column,
            "aggregation": plan.measure.agg,
            "filters": tuple(
                {"column": filter_spec.column, "op": filter_spec.op, "value": filter_spec.value}
                for filter_spec in plan.filters
            ),
        },
        encodings={"chart_type": plan.chart_type},
        annotations={},
        presentation_constraints={
            "sort": {"by": plan.sort.by, "direction": plan.sort.direction} if plan.sort is not None else None,
            "limit": plan.limit,
        },
    )
    return ChartRequirementPlan(
        requirements=tuple(requirements),
        panels=(panel,),
        figure_requirements=figure_requirement_payload,
        shared_requirement_ids=tuple(shared_requirement_ids),
        raw_query=plan.raw_query,
    )


def build_evidence_graph(
    requirement_plan: ChartRequirementPlan,
    verification: VerificationReport,
    expected_trace: PlotTrace,
    actual_trace: PlotTrace,
    actual_figure: FigureTrace | None,
) -> EvidenceGraph:
    expected_artifacts = [
        Artifact(
            artifact_id="expected.plot_trace",
            kind="expected",
            requirement_ids=_trace_requirement_ids(requirement_plan),
            payload=expected_trace,
            source=expected_trace.source,
            panel_id="panel_0",
        )
    ]
    actual_artifacts = [
        Artifact(
            artifact_id="actual.plot_trace",
            kind="actual",
            requirement_ids=_trace_requirement_ids(requirement_plan),
            payload=actual_trace,
            source=actual_trace.source,
            panel_id="panel_0",
        )
    ]
    if verification.expected_figure is not None:
        expected_artifacts.append(
            Artifact(
                artifact_id="expected.figure_requirements",
                kind="expected",
                requirement_ids=_figure_requirement_ids(requirement_plan),
                payload=verification.expected_figure,
                source="figure_requirements",
            )
        )
    if actual_figure is not None:
        actual_artifacts.append(
            Artifact(
                artifact_id="actual.figure_trace",
                kind="actual",
                requirement_ids=_figure_requirement_ids(requirement_plan),
                payload=actual_figure,
                source=actual_figure.source,
            )
        )

    errors_by_requirement: dict[str, list[Any]] = {}
    for error in verification.errors:
        requirement_id = error.requirement_id or infer_requirement_id(error.code, requirement_plan)
        errors_by_requirement.setdefault(requirement_id, []).append(error)

    trace_requirement_ids = set(_trace_requirement_ids(requirement_plan))
    figure_requirement_ids = set(_figure_requirement_ids(requirement_plan))
    links: list[EvidenceLink] = []
    for requirement in requirement_plan.requirements:
        linked_errors = errors_by_requirement.get(requirement.requirement_id, [])
        if requirement.requirement_id in trace_requirement_ids:
            expected_artifact_id = "expected.plot_trace"
            actual_artifact_id = "actual.plot_trace"
        elif requirement.requirement_id in figure_requirement_ids:
            expected_artifact_id = "expected.figure_requirements" if verification.expected_figure is not None else None
            actual_artifact_id = "actual.figure_trace" if actual_figure is not None else None
        else:
            expected_artifact_id = None
            actual_artifact_id = None

        if not requirement.is_verifiable:
            verdict = "unsupported"
            message = f"Requirement status is {requirement.status}."
            error_codes = ()
        elif linked_errors:
            verdict = "fail"
            message = "; ".join(error.message for error in linked_errors)
            error_codes = tuple(error.code for error in linked_errors)
        elif expected_artifact_id is None and actual_artifact_id is None:
            verdict = "abstain"
            message = "No artifact binding available for this requirement yet."
            error_codes = ()
        else:
            verdict = "pass"
            message = "Requirement satisfied by current artifacts."
            error_codes = ()

        links.append(
            EvidenceLink(
                requirement_id=requirement.requirement_id,
                expected_artifact_id=expected_artifact_id,
                actual_artifact_id=actual_artifact_id,
                verdict=verdict,
                error_codes=error_codes,
                message=message,
            )
        )
    return EvidenceGraph(
        requirements=tuple(requirement_plan.requirements),
        expected_artifacts=tuple(expected_artifacts),
        actual_artifacts=tuple(actual_artifacts),
        links=tuple(links),
    )


def infer_requirement_id(error_code: str, requirement_plan: ChartRequirementPlan) -> str:
    direct_map = {
        "wrong_chart_type": "panel_0.chart_type",
        "wrong_order": "panel_0.sort",
        "length_mismatch_extra_points": "panel_0.aggregation",
        "length_mismatch_missing_points": "panel_0.filter_0",
        "data_point_not_found": "panel_0.filter_0",
        "unexpected_data_point": "panel_0.filter_0",
        "wrong_aggregation_value": "panel_0.aggregation",
        "wrong_axes_count": "figure.axes_count",
        "wrong_figure_title": "figure.title",
        "wrong_figure_size": "figure.size_inches",
    }
    direct_candidate = direct_map.get(error_code)
    if direct_candidate is not None and any(req.requirement_id == direct_candidate for req in requirement_plan.requirements):
        return direct_candidate

    suffix_map = {
        "wrong_axis_title": ".title",
        "wrong_x_label": ".xlabel",
        "wrong_y_label": ".ylabel",
        "wrong_z_label": ".zlabel",
        "wrong_projection": ".projection",
        "wrong_x_scale": ".xscale",
        "wrong_y_scale": ".yscale",
        "wrong_z_scale": ".zscale",
        "wrong_axis_layout": ".bounds",
        "missing_legend_label": ".legend_labels",
        "missing_annotation_text": ".text_contains",
        "missing_artist_type": ".artist_types",
        "wrong_artist_count": ".artist_counts",
        "insufficient_artist_count": ".min_artist_counts",
        "wrong_x_tick_labels": ".xtick_labels",
        "wrong_y_tick_labels": ".ytick_labels",
        "wrong_z_tick_labels": ".ztick_labels",
    }
    suffix = suffix_map.get(error_code)
    if suffix is not None:
        for requirement in requirement_plan.requirements:
            if requirement.requirement_id.endswith(suffix):
                return requirement.requirement_id

    for requirement in requirement_plan.requirements:
        if requirement.requirement_id == "panel_0.chart_type":
            return requirement.requirement_id
    return requirement_plan.requirements[0].requirement_id


def _trace_requirement_ids(requirement_plan: ChartRequirementPlan) -> tuple[str, ...]:
    return tuple(
        requirement.requirement_id
        for requirement in requirement_plan.requirements
        if requirement.scope == "panel"
        and requirement.name in {"chart_type", "aggregation", "measure_column", "dimensions", "sort", "limit", "filter"}
    )


def _figure_requirement_ids(requirement_plan: ChartRequirementPlan) -> tuple[str, ...]:
    return tuple(
        requirement.requirement_id
        for requirement in requirement_plan.requirements
        if requirement.name
        in {
            "axes_count",
            "figure_title",
            "size_inches",
            "title",
            "xlabel",
            "ylabel",
            "zlabel",
            "projection",
            "xscale",
            "yscale",
            "zscale",
            "bounds",
            "legend_labels",
            "artist_types",
            "artist_counts",
            "min_artist_counts",
            "text_contains",
        }
    )
