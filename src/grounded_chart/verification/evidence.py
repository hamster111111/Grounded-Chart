from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from grounded_chart.core.requirements import Artifact, ChartRequirementPlan, EvidenceGraph, EvidenceLink, PanelRequirementPlan, RequirementNode
from grounded_chart.core.schema import AxisRequirementSpec, ChartIntentPlan, FigureRequirementSpec, FigureTrace, PlotTrace, VerificationReport
from grounded_chart.data.visual_artifacts import extract_actual_visual_artifacts


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
        severity: str | None = None,
        match_policy: str | None = None,
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
                severity=severity or _default_requirement_severity(name),
                match_policy=match_policy or _default_requirement_match_policy(name),
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
                _figure_source_span(expected_figure, "axes_count", str(expected_figure.axes_count)),
            )
            figure_requirement_payload["axes_count"] = expected_figure.axes_count
        if expected_figure.figure_title is not None:
            add_requirement(
                "figure.title",
                "figure",
                "annotation",
                "figure_title",
                expected_figure.figure_title,
                _figure_source_span(expected_figure, "figure_title", expected_figure.figure_title),
            )
            figure_requirement_payload["figure_title"] = expected_figure.figure_title
        if expected_figure.size_inches is not None:
            add_requirement(
                "figure.size_inches",
                "figure",
                "presentation_constraint",
                "size_inches",
                tuple(expected_figure.size_inches),
                _figure_source_span(expected_figure, "size_inches", str(tuple(expected_figure.size_inches))),
            )
            figure_requirement_payload["size_inches"] = tuple(expected_figure.size_inches)
        for contract in expected_figure.artifact_contracts:
            fields = _requirement_fields_from_artifact_contract(contract)
            if fields is None:
                continue
            add_requirement(**fields)
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
                    _axis_source_span(axis, field_name, str(field_value)),
                    panel_ref=panel_id,
                )
            if axis.legend_labels:
                add_requirement(
                    f"{axis_prefix}.legend_labels",
                    "panel",
                    "annotation",
                    "legend_labels",
                    tuple(axis.legend_labels),
                    _axis_source_span(axis, "legend_labels", ", ".join(axis.legend_labels)),
                    panel_ref=panel_id,
                )
            if axis.artist_types:
                add_requirement(
                    f"{axis_prefix}.artist_types",
                    "panel",
                    "encoding",
                    "artist_types",
                    tuple(axis.artist_types),
                    _axis_source_span(axis, "artist_types", ", ".join(axis.artist_types)),
                    panel_ref=panel_id,
                )
            if axis.artist_counts:
                add_requirement(
                    f"{axis_prefix}.artist_counts",
                    "panel",
                    "encoding",
                    "artist_counts",
                    dict(axis.artist_counts),
                    _axis_source_span(axis, "artist_counts", str(dict(axis.artist_counts))),
                    panel_ref=panel_id,
                )
            if axis.min_artist_counts:
                add_requirement(
                    f"{axis_prefix}.min_artist_counts",
                    "panel",
                    "encoding",
                    "min_artist_counts",
                    dict(axis.min_artist_counts),
                    _axis_source_span(axis, "min_artist_counts", str(dict(axis.min_artist_counts))),
                    panel_ref=panel_id,
                )
            if axis.text_contains:
                add_requirement(
                    f"{axis_prefix}.text_contains",
                    "panel",
                    "annotation",
                    "text_contains",
                    tuple(axis.text_contains),
                    _axis_source_span(axis, "text_contains", ", ".join(axis.text_contains)),
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


def derive_expected_figure(requirement_plan: ChartRequirementPlan) -> FigureRequirementSpec | None:
    """Build verifier-consumable figure expectations from parser-native requirements.

    This bridge is intentionally narrow: it only materializes requirement
    families that the current figure trace + verifier stack can check
    reliably. Unsupported parser-native figure families stay in the
    requirement plan, but remain unbound until a verifier is added.
    """

    verifiable_requirements = tuple(
        requirement
        for requirement in requirement_plan.requirements
        if requirement.is_verifiable
    )
    if not verifiable_requirements:
        return None

    figure_provenance: dict[str, tuple[str, ...]] = {}
    figure_source_spans: dict[str, str] = {}
    figure_title: str | None = None
    axes_count: int | None = None

    panel_axis_label_requirements: dict[str, list[RequirementNode]] = {}
    panel_title_requirements: dict[str, list[RequirementNode]] = {}
    panel_artist_type_requirements: dict[str, list[RequirementNode]] = {}

    for requirement in verifiable_requirements:
        if requirement.scope == "figure":
            normalized_title = _normalize_text_value(requirement.value) if requirement.name in {"title", "figure_title"} else None
            if requirement.name in {"title", "figure_title"} and normalized_title is not None and figure_title is None:
                figure_title = normalized_title
                figure_provenance["figure_title"] = (requirement.requirement_id,)
                figure_source_spans["figure_title"] = requirement.source_span or normalized_title
            elif requirement.name in {"axes_count", "subplot_count"} and axes_count is None:
                parsed_axes_count = _as_int(requirement.value)
                if parsed_axes_count is not None:
                    axes_count = parsed_axes_count
                    figure_provenance["axes_count"] = (requirement.requirement_id,)
                    figure_source_spans["axes_count"] = requirement.source_span or str(parsed_axes_count)
            elif requirement.name == "subplot_layout" and axes_count is None:
                parsed_axes_count = _axes_count_from_layout(requirement.value)
                if parsed_axes_count is not None:
                    axes_count = parsed_axes_count
                    figure_provenance["axes_count"] = (requirement.requirement_id,)
                    figure_source_spans["axes_count"] = requirement.source_span or str(parsed_axes_count)
            continue

        if requirement.scope != "panel":
            continue
        panel_id = requirement.panel_id or "panel_0"
        if requirement.name == "axis_label":
            panel_axis_label_requirements.setdefault(panel_id, []).append(requirement)
        elif requirement.name == "title":
            panel_title_requirements.setdefault(panel_id, []).append(requirement)
        elif requirement.name == "artist_type":
            panel_artist_type_requirements.setdefault(panel_id, []).append(requirement)

    panel_ids = sorted(
        {
            *panel_axis_label_requirements.keys(),
            *panel_title_requirements.keys(),
            *panel_artist_type_requirements.keys(),
        },
        key=_panel_sort_key,
    )

    panel_index_offset = _panel_number_offset(panel_ids)
    axes: list[AxisRequirementSpec] = []
    for dense_index, panel_id in enumerate(panel_ids):
        axis_index = _panel_axis_index(panel_id, dense_index, panel_index_offset)
        axis_title: str | None = None
        axis_provenance: dict[str, tuple[str, ...]] = {}
        axis_source_spans: dict[str, str] = {}

        title_requirements = panel_title_requirements.get(panel_id, [])
        if title_requirements:
            primary_title = title_requirements[0]
            normalized_title = _normalize_text_value(primary_title.value)
            if normalized_title is not None:
                axis_title = normalized_title
                axis_provenance["title"] = (primary_title.requirement_id,)
                axis_source_spans["title"] = primary_title.source_span or normalized_title

        label_mapping = _map_axis_label_requirements(panel_axis_label_requirements.get(panel_id, ()))
        for field_name, requirement in label_mapping.items():
            axis_provenance[field_name] = (requirement.requirement_id,)
            axis_source_spans[field_name] = requirement.source_span or str(requirement.value)

        artist_types = tuple(
            artist_type
            for artist_type in (_normalize_artist_type_value(requirement.value) for requirement in panel_artist_type_requirements.get(panel_id, ()))
            if artist_type is not None
        )
        if artist_types:
            artist_provenance: dict[str, tuple[str, ...]] = {}
            for requirement in panel_artist_type_requirements.get(panel_id, ()):
                artist_type = _normalize_artist_type_value(requirement.value)
                if artist_type is None:
                    continue
                artist_provenance.setdefault(f"artist_type:{artist_type}", ())
                artist_provenance[f"artist_type:{artist_type}"] = _merge_provenance_ids(
                    artist_provenance[f"artist_type:{artist_type}"],
                    (requirement.requirement_id,),
                )
            axis_provenance.update(artist_provenance)
            axis_source_spans["artist_types"] = ", ".join(artist_types)
            axis_provenance["artist_types"] = tuple(
                requirement.requirement_id
                for requirement in panel_artist_type_requirements.get(panel_id, ())
                if _normalize_artist_type_value(requirement.value) is not None
            )

        axis = AxisRequirementSpec(
            axis_index=axis_index,
            title=axis_title,
            xlabel=label_mapping.get("xlabel").value if "xlabel" in label_mapping else None,
            ylabel=label_mapping.get("ylabel").value if "ylabel" in label_mapping else None,
            zlabel=label_mapping.get("zlabel").value if "zlabel" in label_mapping else None,
            artist_types=tuple(dict.fromkeys(artist_types)),
            provenance=axis_provenance,
            source_spans=axis_source_spans,
        )
        if _axis_requirement_has_expectations(axis):
            axes.append(axis)

    if figure_title is None and axes_count is None and not axes:
        return None

    return FigureRequirementSpec(
        axes_count=axes_count,
        figure_title=figure_title,
        axes=tuple(axes),
        provenance=figure_provenance,
        source_spans=figure_source_spans,
    )


def merge_expected_figure_specs(
    base: FigureRequirementSpec | None,
    overlay: FigureRequirementSpec | None,
) -> FigureRequirementSpec | None:
    if base is None:
        return overlay
    if overlay is None:
        return base

    merged_axes_by_index: dict[int, AxisRequirementSpec] = {
        axis.axis_index: axis
        for axis in base.axes
    }
    for axis in overlay.axes:
        existing = merged_axes_by_index.get(axis.axis_index)
        if existing is None:
            merged_axes_by_index[axis.axis_index] = axis
            continue
        merged_axis_provenance = _merge_axis_provenance(existing, axis)
        merged_axis_source_spans = _merge_axis_source_spans(existing, axis)
        merged_axes_by_index[axis.axis_index] = AxisRequirementSpec(
            axis_index=axis.axis_index,
            title=axis.title if axis.title is not None else existing.title,
            xlabel=axis.xlabel if axis.xlabel is not None else existing.xlabel,
            ylabel=axis.ylabel if axis.ylabel is not None else existing.ylabel,
            zlabel=axis.zlabel if axis.zlabel is not None else existing.zlabel,
            projection=axis.projection if axis.projection is not None else existing.projection,
            xscale=axis.xscale if axis.xscale is not None else existing.xscale,
            yscale=axis.yscale if axis.yscale is not None else existing.yscale,
            zscale=axis.zscale if axis.zscale is not None else existing.zscale,
            xtick_labels=axis.xtick_labels if axis.xtick_labels else existing.xtick_labels,
            ytick_labels=axis.ytick_labels if axis.ytick_labels else existing.ytick_labels,
            ztick_labels=axis.ztick_labels if axis.ztick_labels else existing.ztick_labels,
            bounds=axis.bounds if axis.bounds is not None else existing.bounds,
            legend_labels=axis.legend_labels if axis.legend_labels else existing.legend_labels,
            artist_types=axis.artist_types if axis.artist_types else existing.artist_types,
            artist_counts=axis.artist_counts if axis.artist_counts else existing.artist_counts,
            min_artist_counts=axis.min_artist_counts if axis.min_artist_counts else existing.min_artist_counts,
            text_contains=axis.text_contains if axis.text_contains else existing.text_contains,
            provenance=merged_axis_provenance,
            source_spans=merged_axis_source_spans,
        )

    merged_figure_provenance = _merge_figure_provenance(base, overlay)
    merged_figure_source_spans = _merge_figure_source_spans(base, overlay)
    return FigureRequirementSpec(
        axes_count=overlay.axes_count if overlay.axes_count is not None else base.axes_count,
        figure_title=overlay.figure_title if overlay.figure_title is not None else base.figure_title,
        size_inches=overlay.size_inches if overlay.size_inches is not None else base.size_inches,
        axes=tuple(sorted(merged_axes_by_index.values(), key=lambda axis: axis.axis_index)),
        provenance=merged_figure_provenance,
        source_spans=merged_figure_source_spans,
        artifact_contracts=_merge_artifact_contracts(base.artifact_contracts, overlay.artifact_contracts),
    )



def _merge_artifact_contracts(
    base: tuple[dict[str, Any], ...],
    overlay: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for contract in (*base, *overlay):
        if not isinstance(contract, dict):
            continue
        key = str(contract.get("artifact_id") or contract)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(contract))
    return tuple(merged)
def _axis_requirement_has_expectations(axis: AxisRequirementSpec) -> bool:
    return any(
        (
            axis.title is not None,
            axis.xlabel is not None,
            axis.ylabel is not None,
            axis.zlabel is not None,
            axis.artist_types,
            axis.projection is not None,
            axis.xscale is not None,
            axis.yscale is not None,
            axis.zscale is not None,
            axis.xtick_labels,
            axis.ytick_labels,
            axis.ztick_labels,
            axis.bounds is not None,
            axis.legend_labels,
            axis.artist_counts,
            axis.min_artist_counts,
            axis.text_contains,
        )
    )


def _map_axis_label_requirements(requirements: list[RequirementNode] | tuple[RequirementNode, ...]) -> dict[str, RequirementNode]:
    ordered = sorted(requirements, key=_requirement_sort_key)
    mapped: dict[str, RequirementNode] = {}
    unresolved: list[RequirementNode] = []
    for requirement in ordered:
        normalized_value = _normalize_text_value(requirement.value)
        if normalized_value is None or _is_generic_axis_label_text(normalized_value):
            continue
        normalized_requirement = replace(requirement, value=normalized_value)
        slot = _axis_label_slot(normalized_requirement)
        if slot is None or slot in mapped:
            unresolved.append(normalized_requirement)
            continue
        mapped[slot] = normalized_requirement

    if len(ordered) >= 2:
        for slot in ("xlabel", "ylabel", "zlabel"):
            if slot in mapped or not unresolved:
                continue
            mapped[slot] = unresolved.pop(0)
    return mapped


def _axis_label_slot(requirement: RequirementNode) -> str | None:
    name = requirement.name.lower()
    if _looks_like_axis_label_name(name, "x"):
        return "xlabel"
    if _looks_like_axis_label_name(name, "y"):
        return "ylabel"
    if _looks_like_axis_label_name(name, "z"):
        return "zlabel"

    grounded_text = " ".join(
        value
        for value in (requirement.source_span, requirement.assumption or "", requirement.requirement_id)
        if value
    ).lower()
    for axis_name, slot in (("x", "xlabel"), ("y", "ylabel"), ("z", "zlabel")):
        patterns = (
            rf"\b{axis_name}\s*axis\b",
            rf"\b{axis_name}axis\b",
            rf"\b{axis_name}\s*label\b",
            rf"\b{axis_name}label\b",
        )
        if any(re.search(pattern, grounded_text) for pattern in patterns):
            return slot
    return None


def _looks_like_axis_label_name(name: str, axis_name: str) -> bool:
    return name in {
        f"{axis_name}label",
        f"{axis_name}_label",
        f"{axis_name}_axis_label",
        f"{axis_name}axislabel",
    } or (name.startswith(f"{axis_name}_") and "label" in name)


def _axes_count_from_layout(value: Any) -> int | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        first = _as_int(value[0])
        second = _as_int(value[1])
        if first is not None and second is not None:
            return first * second
    if isinstance(value, dict):
        first = _as_int(value.get("rows"))
        second = _as_int(value.get("cols") or value.get("columns"))
        if first is not None and second is not None:
            return first * second
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"side-by-side", "side_by_side", "side by side"}:
        return 2
    match = re.search(r"(\d+)\s*[x闂傚倸鍊烽懗鍫曞储瑜旈、姘额敇閵忊€充罕婵犻潧顦介幗?(\d+)", text)
    if match:
        return int(match.group(1)) * int(match.group(2))
    row_col_match = re.search(
        r"\b(\d+|one|two|three|four|five|six)\s+rows?\s*,?\s*(\d+|one|two|three|four|five|six)\s+columns?\b",
        text,
    )
    if row_col_match:
        rows = _word_or_int(row_col_match.group(1))
        columns = _word_or_int(row_col_match.group(2))
        if rows is not None and columns is not None:
            return rows * columns
    row_match = re.search(r"\b(\d+|one|two|three|four|five|six)\s+rows?\b", text)
    if row_match:
        rows = _word_or_int(row_match.group(1))
        if rows is not None:
            return rows
    subplot_counts = [
        _word_or_int(match.group(1))
        for match in re.finditer(r"\b(\d+|one|two|three|four|five|six)\b(?=[^.;,\n]{0,24}\bsubplots?\b)", text)
    ]
    subplot_counts = [count for count in subplot_counts if count is not None]
    if subplot_counts:
        return sum(subplot_counts)
    return None


def _normalize_artist_type_value(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bar_chart": "bar",
        "line_chart": "line",
        "pie_chart": "pie",
        "scatter_plot": "scatter",
        "area_chart": "area",
        "stacked_area": "area",
    }
    normalized = aliases.get(text, text)
    if normalized in {"bar", "line", "pie", "scatter", "area", "heatmap", "box", "waterfall", "errorbar", "hist2d", "patch", "image", "table"}:
        return normalized
    if "pie" in normalized:
        return "pie"
    if "scatter" in normalized:
        return "scatter"
    if "bar" in normalized:
        return "bar"
    if "area" in normalized:
        return "area"
    if "line" in normalized or "log" in normalized:
        return "line"
    if "hist" in normalized:
        return "hist2d"
    if "waterfall" in normalized:
        return "waterfall"
    return None


def _normalize_text_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    while len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1].strip()
    return normalized or None


def _is_generic_axis_label_text(value: str) -> bool:
    normalized = value.strip().lower().replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in {
        "x",
        "y",
        "z",
        "x axis",
        "y axis",
        "z axis",
        "the x axis",
        "the y axis",
        "the z axis",
        "x label",
        "y label",
        "z label",
        "the x label",
        "the y label",
        "the z label",
    }


def _panel_sort_key(panel_id: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"panel_(\d+)", panel_id)
    if match:
        return (0, int(match.group(1)))
    return (1, panel_id)


def _panel_number_offset(panel_ids: list[str] | tuple[str, ...]) -> int:
    numeric_indices = [_panel_numeric_index(panel_id) for panel_id in panel_ids]
    numeric_indices = [index for index in numeric_indices if index is not None]
    if numeric_indices and min(numeric_indices) >= 1:
        return 1
    return 0


def _panel_axis_index(panel_id: str, dense_index: int, offset: int) -> int:
    numeric_index = _panel_numeric_index(panel_id)
    if numeric_index is None:
        return dense_index
    return max(0, numeric_index - offset)


def _panel_numeric_index(panel_id: str) -> int | None:
    match = re.fullmatch(r"panel_(\d+)", panel_id)
    if match:
        return int(match.group(1))
    return None


def _requirement_sort_key(requirement: RequirementNode) -> tuple[int, str]:
    match = re.search(r"_(\d+)$", requirement.requirement_id)
    if match:
        return (int(match.group(1)), requirement.requirement_id)
    return (10_000, requirement.requirement_id)


def _merge_provenance_ids(existing: tuple[str, ...], new_ids: tuple[str, ...]) -> tuple[str, ...]:
    merged = list(existing)
    for requirement_id in new_ids:
        if requirement_id not in merged:
            merged.append(requirement_id)
    return tuple(merged)


def _merge_figure_provenance(
    base: FigureRequirementSpec,
    overlay: FigureRequirementSpec,
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {}
    figure_fields = (
        ("axes_count", overlay.axes_count is not None),
        ("figure_title", overlay.figure_title is not None),
        ("size_inches", overlay.size_inches is not None),
    )
    for field_name, overlay_active in figure_fields:
        if overlay_active:
            provenance = overlay.provenance.get(field_name, ())
        else:
            provenance = base.provenance.get(field_name, ())
        if provenance:
            merged[field_name] = provenance
    return merged


def _merge_axis_provenance(
    base: AxisRequirementSpec,
    overlay: AxisRequirementSpec,
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {}
    scalar_fields = (
        ("title", overlay.title is not None),
        ("xlabel", overlay.xlabel is not None),
        ("ylabel", overlay.ylabel is not None),
        ("zlabel", overlay.zlabel is not None),
        ("projection", overlay.projection is not None),
        ("xscale", overlay.xscale is not None),
        ("yscale", overlay.yscale is not None),
        ("zscale", overlay.zscale is not None),
        ("bounds", overlay.bounds is not None),
    )
    for field_name, overlay_active in scalar_fields:
        provenance = overlay.provenance.get(field_name, ()) if overlay_active else base.provenance.get(field_name, ())
        if provenance:
            merged[field_name] = provenance

    tuple_fields = (
        ("xtick_labels", bool(overlay.xtick_labels)),
        ("ytick_labels", bool(overlay.ytick_labels)),
        ("ztick_labels", bool(overlay.ztick_labels)),
        ("legend_labels", bool(overlay.legend_labels)),
        ("text_contains", bool(overlay.text_contains)),
        ("artist_types", bool(overlay.artist_types)),
        ("artist_counts", bool(overlay.artist_counts)),
        ("min_artist_counts", bool(overlay.min_artist_counts)),
    )
    for field_name, overlay_active in tuple_fields:
        source = overlay.provenance if overlay_active else base.provenance
        for key, value in source.items():
            if key == field_name or key.startswith(field_name[:-1] if field_name.endswith("s") else field_name):
                merged[key] = value
        if field_name in source:
            merged[field_name] = source[field_name]
    return merged


def _merge_figure_source_spans(
    base: FigureRequirementSpec,
    overlay: FigureRequirementSpec,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    figure_fields = (
        ("axes_count", overlay.axes_count is not None),
        ("figure_title", overlay.figure_title is not None),
        ("size_inches", overlay.size_inches is not None),
    )
    for field_name, overlay_active in figure_fields:
        source = overlay.source_spans if overlay_active else base.source_spans
        source_span = source.get(field_name, "")
        if source_span:
            merged[field_name] = source_span
    return merged


def _merge_axis_source_spans(
    base: AxisRequirementSpec,
    overlay: AxisRequirementSpec,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    scalar_fields = (
        ("title", overlay.title is not None),
        ("xlabel", overlay.xlabel is not None),
        ("ylabel", overlay.ylabel is not None),
        ("zlabel", overlay.zlabel is not None),
        ("projection", overlay.projection is not None),
        ("xscale", overlay.xscale is not None),
        ("yscale", overlay.yscale is not None),
        ("zscale", overlay.zscale is not None),
        ("bounds", overlay.bounds is not None),
    )
    for field_name, overlay_active in scalar_fields:
        source = overlay.source_spans if overlay_active else base.source_spans
        source_span = source.get(field_name, "")
        if source_span:
            merged[field_name] = source_span

    tuple_fields = (
        ("xtick_labels", bool(overlay.xtick_labels)),
        ("ytick_labels", bool(overlay.ytick_labels)),
        ("ztick_labels", bool(overlay.ztick_labels)),
        ("legend_labels", bool(overlay.legend_labels)),
        ("text_contains", bool(overlay.text_contains)),
        ("artist_types", bool(overlay.artist_types)),
        ("artist_counts", bool(overlay.artist_counts)),
        ("min_artist_counts", bool(overlay.min_artist_counts)),
    )
    for field_name, overlay_active in tuple_fields:
        source = overlay.source_spans if overlay_active else base.source_spans
        source_span = source.get(field_name, "")
        if source_span:
            merged[field_name] = source_span
    return merged


def _figure_source_span(expected_figure: FigureRequirementSpec, field_name: str, fallback: str) -> str:
    return expected_figure.source_spans.get(field_name) or fallback


def _axis_source_span(axis: AxisRequirementSpec, field_name: str, fallback: str) -> str:
    return axis.source_spans.get(field_name) or fallback


def _default_requirement_severity(name: str) -> str:
    if name in {"size_inches", "bounds"}:
        return "warning"
    return "error"


def _default_requirement_match_policy(name: str) -> str:
    if name in {"size_inches", "bounds"}:
        return "numeric_close"
    if name in {"legend_labels", "text_contains", "legend"}:
        return "contains"
    if name == "figure_title":
        return "normalized_contains"
    if name in {"artist_type", "artist_types", "min_artist_counts"}:
        return "presence"
    if name in {"xtick_labels", "ytick_labels", "ztick_labels"}:
        return "sequence_exact"
    return "exact"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _word_or_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = _as_int(value)
    if parsed is not None:
        return parsed
    mapping = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    return mapping.get(str(value).strip().lower())


def attach_figure_requirements(
    requirement_plan: ChartRequirementPlan,
    expected_figure: FigureRequirementSpec | None = None,
) -> ChartRequirementPlan:
    """Attach externally supplied figure requirements without rewriting parser output.

    Query-derived data and encoding requirements must come from the parser.
    Figure requirements are currently supplied by adapters/traces, so this
    function only appends those external requirements and preserves existing
    source spans, status, assumptions, and confidence values.
    """

    if expected_figure is None:
        return requirement_plan

    requirements = list(requirement_plan.requirements)
    existing_ids = {requirement.requirement_id for requirement in requirements}
    figure_requirement_payload: dict[str, Any] = dict(requirement_plan.figure_requirements)
    panel_requirement_ids = {
        panel.panel_id: list(panel.requirement_ids)
        for panel in requirement_plan.panels
    }

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
        priority: str = "secondary",
        severity: str | None = None,
        match_policy: str | None = None,
    ) -> None:
        if requirement_id in existing_ids:
            return
        requirements.append(
            RequirementNode(
                requirement_id=requirement_id,
                scope=scope,
                type=type_,
                name=name,
                value=value,
                source_span=source_span,
                status=status,
                priority=priority,
                panel_id=panel_ref,
                severity=severity or _default_requirement_severity(name),
                match_policy=match_policy or _default_requirement_match_policy(name),
            )
        )
        existing_ids.add(requirement_id)
        if panel_ref is not None:
            panel_requirement_ids.setdefault(panel_ref, []).append(requirement_id)

    if expected_figure.axes_count is not None:
        add_requirement(
            "figure.axes_count",
            "figure",
            "figure_composition",
            "axes_count",
            expected_figure.axes_count,
            _figure_source_span(expected_figure, "axes_count", str(expected_figure.axes_count)),
        )
        figure_requirement_payload["axes_count"] = expected_figure.axes_count
    if expected_figure.figure_title is not None:
        add_requirement(
            "figure.title",
            "figure",
            "annotation",
            "figure_title",
            expected_figure.figure_title,
            _figure_source_span(expected_figure, "figure_title", expected_figure.figure_title),
        )
        figure_requirement_payload["figure_title"] = expected_figure.figure_title
    if expected_figure.size_inches is not None:
        add_requirement(
            "figure.size_inches",
            "figure",
            "presentation_constraint",
            "size_inches",
            tuple(expected_figure.size_inches),
            _figure_source_span(expected_figure, "size_inches", str(tuple(expected_figure.size_inches))),
        )
        figure_requirement_payload["size_inches"] = tuple(expected_figure.size_inches)

    for contract in expected_figure.artifact_contracts:
        fields = _requirement_fields_from_artifact_contract(contract)
        if fields is None:
            continue
        add_requirement(**fields)

    for axis in expected_figure.axes:
        panel_id = "panel_0"
        axis_prefix = f"{panel_id}.axis_{axis.axis_index}"
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
                _axis_source_span(axis, field_name, str(field_value)),
                panel_ref=panel_id,
            )
        if axis.legend_labels:
            add_requirement(
                f"{axis_prefix}.legend_labels",
                "panel",
                "annotation",
                "legend_labels",
                tuple(axis.legend_labels),
                _axis_source_span(axis, "legend_labels", ", ".join(axis.legend_labels)),
                panel_ref=panel_id,
            )
        if axis.artist_types:
            add_requirement(
                f"{axis_prefix}.artist_types",
                "panel",
                "encoding",
                "artist_types",
                tuple(axis.artist_types),
                _axis_source_span(axis, "artist_types", ", ".join(axis.artist_types)),
                panel_ref=panel_id,
            )
        if axis.artist_counts:
            add_requirement(
                f"{axis_prefix}.artist_counts",
                "panel",
                "encoding",
                "artist_counts",
                dict(axis.artist_counts),
                _axis_source_span(axis, "artist_counts", str(dict(axis.artist_counts))),
                panel_ref=panel_id,
            )
        if axis.min_artist_counts:
            add_requirement(
                f"{axis_prefix}.min_artist_counts",
                "panel",
                "encoding",
                "min_artist_counts",
                dict(axis.min_artist_counts),
                _axis_source_span(axis, "min_artist_counts", str(dict(axis.min_artist_counts))),
                panel_ref=panel_id,
            )
        if axis.text_contains:
            add_requirement(
                f"{axis_prefix}.text_contains",
                "panel",
                "annotation",
                "text_contains",
                tuple(axis.text_contains),
                _axis_source_span(axis, "text_contains", ", ".join(axis.text_contains)),
                panel_ref=panel_id,
            )

    panels = tuple(
        replace(panel, requirement_ids=tuple(panel_requirement_ids.get(panel.panel_id, panel.requirement_ids)))
        for panel in requirement_plan.panels
    )
    return replace(
        requirement_plan,
        requirements=tuple(requirements),
        panels=panels,
        figure_requirements=figure_requirement_payload,
    )

def bind_requirement_policy_to_verification(
    verification: VerificationReport,
    requirement_plan: ChartRequirementPlan,
) -> VerificationReport:
    """Attach requirement-level policy metadata to verifier errors.

    The verifier operates on expected/actual artifacts and only knows optional
    requirement ids through provenance. This pass resolves missing ids, then
    applies the RequirementNode policy so downstream reporting and repair can
    distinguish hard errors from soft constraints.
    """

    requirements_by_id = {
        requirement.requirement_id: requirement
        for requirement in requirement_plan.requirements
    }
    bound_errors = []
    for error in verification.errors:
        requirement_id = error.requirement_id or infer_requirement_id(error.code, requirement_plan)
        requirement = requirements_by_id.get(requirement_id)
        if requirement is None:
            bound_errors.append(error if error.requirement_id == requirement_id else replace(error, requirement_id=requirement_id))
            continue
        bound_errors.append(
            replace(
                error,
                requirement_id=requirement_id,
                severity=requirement.severity,
                match_policy=requirement.match_policy,
            )
        )
    return replace(verification, errors=tuple(bound_errors))


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
    expected_artifacts.extend(_expected_intermediate_artifacts(requirement_plan, expected_trace))

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
    actual_artifacts.extend(_actual_trace_artifacts(requirement_plan, actual_trace))

    if verification.expected_figure is not None:
        expected_artifacts.append(
            Artifact(
                artifact_id="expected.figure_requirements",
                kind="expected",
                requirement_ids=_figure_requirement_ids(requirement_plan, verification.expected_figure),
                payload=verification.expected_figure,
                source="figure_requirements",
            )
        )
        for contract in verification.expected_figure.artifact_contracts:
            if not isinstance(contract, dict):
                continue
            artifact_id = str(contract.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            requirement_id = str(contract.get("source_requirement_id") or "").strip()
            expected_artifacts.append(
                Artifact(
                    artifact_id=artifact_id,
                    kind="expected",
                    requirement_ids=(requirement_id,) if requirement_id else (),
                    payload=contract,
                    source="expected_visual_artifact_contract",
                    panel_id=_artifact_contract_panel_id(contract),
                )
            )
    if actual_figure is not None:
        actual_artifacts.append(
            Artifact(
                artifact_id="actual.figure_trace",
                kind="actual",
                requirement_ids=_figure_requirement_ids(requirement_plan, verification.expected_figure),
                payload=actual_figure,
                source=actual_figure.source,
            )
        )
        for visual_artifact in extract_actual_visual_artifacts(actual_figure):
            actual_artifacts.append(
                Artifact(
                    artifact_id=visual_artifact.artifact_id,
                    kind="actual",
                    requirement_ids=_visual_artifact_requirement_ids(requirement_plan, visual_artifact),
                    payload=visual_artifact.to_dict(),
                    source=visual_artifact.source,
                    panel_id=visual_artifact.locator.get("panel_id"),
                )
            )

    expected_artifact_ids = {artifact.artifact_id for artifact in expected_artifacts}
    actual_artifact_ids = {artifact.artifact_id for artifact in actual_artifacts}
    expected_artifact_ids_by_requirement = _artifact_ids_by_requirement(expected_artifacts)
    actual_artifact_ids_by_requirement = _artifact_ids_by_requirement(actual_artifacts)
    errors_by_requirement: dict[str, list[Any]] = {}
    for error in verification.errors:
        requirement_id = error.requirement_id or infer_requirement_id(error.code, requirement_plan)
        errors_by_requirement.setdefault(requirement_id, []).append(error)

    trace_requirement_ids = set(_trace_requirement_ids(requirement_plan))
    figure_requirement_ids = set(_figure_requirement_ids(requirement_plan, verification.expected_figure))
    missing_figure_trace = any(error.code == "missing_figure_trace" for error in verification.errors)
    links: list[EvidenceLink] = []
    for requirement in requirement_plan.requirements:
        linked_errors = errors_by_requirement.get(requirement.requirement_id, [])
        if requirement.requirement_id in trace_requirement_ids:
            expected_artifact_id = _preferred_expected_artifact_id_for_requirement(
                requirement,
                expected_artifact_ids_by_requirement,
                expected_artifact_ids,
                fallback=_expected_trace_artifact_id_for_requirement(requirement, expected_artifact_ids),
            )
            actual_artifact_id = _preferred_actual_artifact_id_for_requirement(
                requirement,
                actual_artifact_ids_by_requirement,
                actual_artifact_ids,
                fallback=_actual_trace_artifact_id_for_requirement(requirement, actual_artifact_ids),
            )
        elif requirement.requirement_id in figure_requirement_ids:
            expected_artifact_id = _preferred_expected_figure_artifact_id_for_requirement(
                requirement,
                expected_artifact_ids_by_requirement,
                expected_artifact_ids,
                has_expected_figure=verification.expected_figure is not None,
            )
            actual_artifact_id = _preferred_actual_figure_artifact_id_for_requirement(
                requirement,
                actual_artifact_ids_by_requirement,
                actual_artifact_ids,
                has_actual_figure=actual_figure is not None,
            )
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
        elif expected_artifact_id is not None and actual_artifact_id is None:
            if missing_figure_trace:
                verdict = "fail"
                message = "Figure trace was required but not captured."
                error_codes = ("missing_figure_trace",)
            else:
                verdict = "abstain"
                message = "Expected figure requirement exists, but no actual figure artifact was captured."
                error_codes = ()
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



def _artifact_contract_panel_id(contract: dict[str, Any]) -> str | None:
    locator = contract.get("locator") if isinstance(contract.get("locator"), dict) else {}
    panel_id = locator.get("panel_id")
    return str(panel_id) if panel_id is not None and str(panel_id).strip() else None

def _requirement_fields_from_artifact_contract(contract: Any) -> dict[str, Any] | None:
    if not isinstance(contract, dict):
        return None
    artifact_type = str(contract.get("artifact_type") or "").strip().lower()
    if not artifact_type:
        return None
    expected = contract.get("expected") if isinstance(contract.get("expected"), dict) else {}
    locator = contract.get("locator") if isinstance(contract.get("locator"), dict) else {}
    axis_index = _as_int(locator.get("axis_index"))
    panel_id = str(locator.get("panel_id") or "").strip() or (f"panel_{axis_index}" if axis_index is not None else None)
    source_requirement_id = str(contract.get("source_requirement_id") or "").strip()
    source_span = str(contract.get("source_span") or contract.get("artifact_id") or artifact_type)
    match_policy = str(contract.get("match_policy") or "presence")
    criticality = str(contract.get("criticality") or "hard").strip().lower()
    severity = "warning" if criticality in {"soft", "secondary", "warning", "info"} else "error"

    if artifact_type == "panel_chart_type":
        requirement_id = source_requirement_id or f"{panel_id or 'panel_0'}.chart_type"
        return {
            "requirement_id": requirement_id,
            "scope": "panel",
            "type_": "encoding",
            "name": "chart_type",
            "value": expected.get("chart_type"),
            "source_span": source_span,
            "panel_ref": panel_id or "panel_0",
            "priority": "core",
            "severity": severity,
            "match_policy": match_policy,
        }
    if artifact_type == "connector":
        return {
            "requirement_id": source_requirement_id or "figure.visual_relation.connector",
            "scope": "figure",
            "type_": "figure_composition",
            "name": "visual_relation",
            "value": {"relation_type": "connector", **dict(expected)},
            "source_span": source_span,
            "priority": "core",
            "severity": severity,
            "match_policy": match_policy,
        }
    if artifact_type == "layout":
        return {
            "requirement_id": source_requirement_id or "figure.layout",
            "scope": "figure",
            "type_": "figure_composition",
            "name": "subplot_layout",
            "value": dict(expected),
            "source_span": source_span,
            "priority": "core",
            "severity": severity,
            "match_policy": match_policy,
        }
    if artifact_type == "text":
        return {
            "requirement_id": source_requirement_id or f"{panel_id or 'panel_0'}.visual_text",
            "scope": "panel" if panel_id is not None else "figure",
            "type_": "annotation",
            "name": "text_contains",
            "value": expected.get("text"),
            "source_span": source_span,
            "panel_ref": panel_id,
            "priority": "secondary",
            "severity": severity,
            "match_policy": match_policy,
        }
    return {
        "requirement_id": source_requirement_id or f"figure.visual.{artifact_type}",
        "scope": "figure",
        "type_": "figure_composition",
        "name": artifact_type,
        "value": dict(expected),
        "source_span": source_span,
        "priority": "secondary",
        "severity": severity,
        "match_policy": match_policy,
    }


def _visual_artifact_requirement_ids(requirement_plan: ChartRequirementPlan, visual_artifact: Any) -> tuple[str, ...]:
    artifact_type = str(getattr(visual_artifact, "artifact_type", "") or "").strip().lower()
    value = getattr(visual_artifact, "value", {}) if isinstance(getattr(visual_artifact, "value", {}), dict) else {}
    locator = getattr(visual_artifact, "locator", {}) if isinstance(getattr(visual_artifact, "locator", {}), dict) else {}
    panel_id = _panel_id_from_locator(locator)
    names: set[str] = set()
    exact_ids: list[str] = []

    if artifact_type == "layout":
        names.update({"axes_count", "subplot_layout", "subplot_count", "bounds"})
    elif artifact_type == "connector":
        names.update({"visual_relation", "connector"})
        exact_ids.append("figure.visual_relation.connector")
    elif artifact_type == "panel_chart_type":
        names.update({"chart_type", "artist_type", "artist_types"})
    elif artifact_type == "text":
        names.update({"title", "figure_title", "xlabel", "ylabel", "zlabel", "axis_label", "legend_labels", "text_contains"})
    elif artifact_type == "code_structure":
        structure = str(value.get("structure") or "").strip().lower()
        if structure in {"stacked_bar", "grouped_bar", "exploded_pie"}:
            names.update({"chart_type", "artist_type", "artist_types"})
        elif structure in {"connector", "visual_relation"}:
            names.update({"visual_relation", "connector"})
            exact_ids.append("figure.visual_relation.connector")
        elif structure in {"subplot_layout", "layout"}:
            names.update({"axes_count", "subplot_layout", "subplot_count", "bounds"})

    ids: list[str] = []
    existing = {requirement.requirement_id for requirement in requirement_plan.requirements}
    for requirement_id in exact_ids:
        if requirement_id in existing and requirement_id not in ids:
            ids.append(requirement_id)
    for requirement in requirement_plan.requirements:
        if requirement.name not in names:
            continue
        if not _requirement_matches_visual_panel(requirement, panel_id):
            continue
        if requirement.requirement_id not in ids:
            ids.append(requirement.requirement_id)
    return tuple(ids)


def _panel_id_from_locator(locator: dict[str, Any]) -> str | None:
    panel_id = str(locator.get("panel_id") or "").strip()
    if panel_id:
        return panel_id
    axis_index = _as_int(locator.get("axis_index"))
    if axis_index is not None:
        return f"panel_{axis_index}"
    return None


def _requirement_matches_visual_panel(requirement: RequirementNode, panel_id: str | None) -> bool:
    if requirement.scope in {"figure", "shared", "any_visible"}:
        return True
    if panel_id is None:
        return requirement.panel_id in {None, "", "panel_0"}
    return requirement.panel_id in {None, "", panel_id}


def _artifact_ids_by_requirement(artifacts: list[Artifact]) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for artifact in artifacts:
        for requirement_id in artifact.requirement_ids:
            if not requirement_id:
                continue
            mapping.setdefault(requirement_id, []).append(artifact.artifact_id)
    return {requirement_id: tuple(ids) for requirement_id, ids in mapping.items()}


def _preferred_expected_artifact_id_for_requirement(
    requirement: RequirementNode,
    ids_by_requirement: dict[str, tuple[str, ...]],
    available_artifact_ids: set[str],
    *,
    fallback: str | None,
) -> str | None:
    candidates = ids_by_requirement.get(requirement.requirement_id, ())
    preferred = ("expected.visual",) if requirement.name in {"chart_type", "artist_type", "artist_types", "visual_relation", "connector", "subplot_layout"} else ()
    return _select_preferred_artifact_id(candidates, preferred, available_artifact_ids, fallback=fallback)


def _preferred_actual_artifact_id_for_requirement(
    requirement: RequirementNode,
    ids_by_requirement: dict[str, tuple[str, ...]],
    available_artifact_ids: set[str],
    *,
    fallback: str | None,
) -> str | None:
    candidates = ids_by_requirement.get(requirement.requirement_id, ())
    preferred = _actual_artifact_preferences(requirement)
    return _select_preferred_artifact_id(candidates, preferred, available_artifact_ids, fallback=fallback)


def _preferred_expected_figure_artifact_id_for_requirement(
    requirement: RequirementNode,
    ids_by_requirement: dict[str, tuple[str, ...]],
    available_artifact_ids: set[str],
    *,
    has_expected_figure: bool,
) -> str | None:
    fallback = "expected.figure_requirements" if has_expected_figure else None
    return _preferred_expected_artifact_id_for_requirement(requirement, ids_by_requirement, available_artifact_ids, fallback=fallback)


def _preferred_actual_figure_artifact_id_for_requirement(
    requirement: RequirementNode,
    ids_by_requirement: dict[str, tuple[str, ...]],
    available_artifact_ids: set[str],
    *,
    has_actual_figure: bool,
) -> str | None:
    fallback = "actual.figure_trace" if has_actual_figure else None
    return _preferred_actual_artifact_id_for_requirement(requirement, ids_by_requirement, available_artifact_ids, fallback=fallback)


def _actual_artifact_preferences(requirement: RequirementNode) -> tuple[str, ...]:
    if requirement.name in {"dimensions", "measure_column", "aggregation"}:
        return ("actual.aggregated_table", "actual.candidate_aggregate_tables", "actual.candidate_point_tables", "actual.plot_points")
    if requirement.name == "sort":
        return ("actual.sorted_table", "actual.plot_points")
    if requirement.name == "limit":
        return ("actual.limited_table", "actual.plot_points")
    if requirement.name == "filter":
        return ("actual.plot_points", "actual.x_values")
    if requirement.name in {"chart_type", "artist_type", "artist_types"}:
        return ("actual.code_structure", "actual.panel")
    if requirement.name in {"visual_relation", "connector"}:
        return ("actual.figure.connector", "actual.code_structure")
    if requirement.name in {"axes_count", "subplot_layout", "subplot_count", "bounds"}:
        return ("actual.figure.layout", "actual.code_structure")
    if requirement.name in {"title", "figure_title", "xlabel", "ylabel", "zlabel", "axis_label", "legend_labels", "text_contains"}:
        return ("actual.panel", "actual.figure")
    return ()


def _select_preferred_artifact_id(
    candidates: tuple[str, ...],
    preferred_prefixes: tuple[str, ...],
    available_artifact_ids: set[str],
    *,
    fallback: str | None,
) -> str | None:
    candidate_list = [candidate for candidate in candidates if candidate in available_artifact_ids]
    for prefix in preferred_prefixes:
        for candidate in candidate_list:
            if candidate.startswith(prefix):
                return candidate
    if fallback is not None and fallback in available_artifact_ids:
        return fallback
    return candidate_list[0] if candidate_list else None
def _expected_intermediate_artifacts(requirement_plan: ChartRequirementPlan, expected_trace: PlotTrace) -> list[Artifact]:
    raw_artifacts = expected_trace.raw.get("intermediate_artifacts", []) if isinstance(expected_trace.raw, dict) else []
    artifacts: list[Artifact] = []
    if not isinstance(raw_artifacts, list):
        return artifacts
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            continue
        artifact_id = str(raw.get("artifact_id") or "").strip()
        if not artifact_id:
            continue
        requirement_names = tuple(str(name) for name in raw.get("requirement_names", ()) if str(name))
        artifacts.append(
            Artifact(
                artifact_id=artifact_id,
                kind="expected",
                requirement_ids=_requirement_ids_for_names(requirement_plan, requirement_names),
                payload=_jsonable_value(raw.get("payload")),
                source=f"{expected_trace.source}:{raw.get('stage') or artifact_id}",
                panel_id="panel_0",
            )
        )
    return artifacts


def _actual_trace_artifacts(requirement_plan: ChartRequirementPlan, actual_trace: PlotTrace) -> list[Artifact]:
    raw_artifacts = actual_trace.raw.get("actual_intermediate_artifacts", []) if isinstance(actual_trace.raw, dict) else []
    artifacts: list[Artifact] = []
    if isinstance(raw_artifacts, list):
        for raw in raw_artifacts:
            if not isinstance(raw, dict):
                continue
            artifact_id = str(raw.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            requirement_names = tuple(str(name) for name in raw.get("requirement_names", ()) if str(name))
            artifacts.append(
                Artifact(
                    artifact_id=artifact_id,
                    kind="actual",
                    requirement_ids=_requirement_ids_for_names(requirement_plan, requirement_names),
                    payload=_jsonable_value(raw.get("payload")),
                    source=f"{actual_trace.source}:{raw.get('stage') or artifact_id}",
                    panel_id="panel_0",
                )
            )
    points = [{"x": _jsonable_value(point.x), "y": _jsonable_value(point.y)} for point in actual_trace.points]
    ordered_points = [
        {"x": _jsonable_value(point.x), "y": _jsonable_value(point.y), "meta": _jsonable_value(point.meta)}
        for point in actual_trace.points
    ]
    artifacts.extend([
        Artifact(
            artifact_id="actual.x_values",
            kind="actual",
            requirement_ids=_requirement_ids_for_names(requirement_plan, ("dimensions", "filter", "sort")),
            payload=[point["x"] for point in points],
            source=f"{actual_trace.source}:x_values",
            panel_id="panel_0",
        ),
        Artifact(
            artifact_id="actual.y_values",
            kind="actual",
            requirement_ids=_requirement_ids_for_names(requirement_plan, ("aggregation", "measure_column")),
            payload=[point["y"] for point in points],
            source=f"{actual_trace.source}:y_values",
            panel_id="panel_0",
        ),
        Artifact(
            artifact_id="actual.plot_points",
            kind="actual",
            requirement_ids=_trace_requirement_ids(requirement_plan),
            payload=points,
            source=f"{actual_trace.source}:plot_points",
            panel_id="panel_0",
        ),
        Artifact(
            artifact_id="actual.aggregated_table",
            kind="actual",
            requirement_ids=_requirement_ids_for_names(requirement_plan, ("dimensions", "measure_column", "aggregation")),
            payload=points,
            source=f"{actual_trace.source}:observed_aggregated_table",
            panel_id="panel_0",
        ),
        Artifact(
            artifact_id="actual.sorted_table",
            kind="actual",
            requirement_ids=_requirement_ids_for_names(requirement_plan, ("sort",)),
            payload=ordered_points,
            source=f"{actual_trace.source}:observed_sorted_table",
            panel_id="panel_0",
        ),
        Artifact(
            artifact_id="actual.limited_table",
            kind="actual",
            requirement_ids=_requirement_ids_for_names(requirement_plan, ("limit",)),
            payload=ordered_points,
            source=f"{actual_trace.source}:observed_limited_table",
            panel_id="panel_0",
        ),
    ])
    return artifacts


def _requirement_ids_for_names(requirement_plan: ChartRequirementPlan, names: tuple[str, ...]) -> tuple[str, ...]:
    name_set = set(names)
    return tuple(
        requirement.requirement_id
        for requirement in requirement_plan.requirements
        if requirement.name in name_set
    )


def _expected_trace_artifact_id_for_requirement(requirement: RequirementNode, available_artifact_ids: set[str]) -> str:
    preferred_by_name = {
        "filter": ("expected.filtered_rows", "expected.plot_points"),
        "dimensions": ("expected.grouped_rows", "expected.aggregated_table", "expected.plot_points"),
        "measure_column": ("expected.aggregated_table", "expected.plot_points"),
        "aggregation": ("expected.aggregated_table", "expected.plot_points"),
        "sort": ("expected.sorted_table", "expected.plot_points"),
        "limit": ("expected.limited_table", "expected.plot_points"),
        "chart_type": ("expected.plot_trace",),
    }
    for artifact_id in preferred_by_name.get(requirement.name, ("expected.plot_trace",)):
        if artifact_id in available_artifact_ids:
            return artifact_id
    return "expected.plot_trace"


def _actual_trace_artifact_id_for_requirement(requirement: RequirementNode, available_artifact_ids: set[str]) -> str:
    preferred_by_name = {
        "dimensions": ("actual.aggregated_table", "actual.plot_points"),
        "measure_column": ("actual.aggregated_table", "actual.y_values", "actual.plot_points"),
        "aggregation": ("actual.aggregated_table", "actual.plot_points"),
        "sort": ("actual.sorted_table", "actual.plot_points"),
        "limit": ("actual.limited_table", "actual.plot_points"),
        "filter": ("actual.plot_points", "actual.x_values"),
        "chart_type": ("actual.plot_trace",),
    }
    for artifact_id in preferred_by_name.get(requirement.name, ("actual.plot_trace", "actual.plot_points")):
        if artifact_id in available_artifact_ids:
            return artifact_id
    return "actual.plot_trace" if "actual.plot_trace" in available_artifact_ids else "actual.plot_points"


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return _jsonable_value(value.item())
        except Exception:
            pass
    return value

def _first_existing_requirement_id(
    requirement_plan: ChartRequirementPlan,
    candidates: tuple[str, ...],
) -> str | None:
    existing_ids = {requirement.requirement_id for requirement in requirement_plan.requirements}
    for candidate in candidates:
        if candidate in existing_ids:
            return candidate
    return None


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
    if error_code in {"data_point_not_found", "unexpected_data_point", "length_mismatch_missing_points"}:
        fallback = _first_existing_requirement_id(
            requirement_plan,
            ("panel_0.filter_0", "panel_0.dimensions", "panel_0.aggregation", "panel_0.measure_column"),
        )
        if fallback is not None:
            return fallback

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


def _figure_requirement_ids(
    requirement_plan: ChartRequirementPlan,
    expected_figure: FigureRequirementSpec | None,
) -> tuple[str, ...]:
    bound_ids: list[str] = []

    def add(requirement_id: str) -> None:
        if requirement_id and requirement_id not in bound_ids:
            bound_ids.append(requirement_id)

    static_names = {
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
        "subplot_layout",
        "subplot_count",
        "visual_relation",
        "connector",
    }
    for requirement in requirement_plan.requirements:
        if requirement.name in static_names:
            add(requirement.requirement_id)

    if expected_figure is not None:
        for requirement_ids in expected_figure.provenance.values():
            for requirement_id in requirement_ids:
                add(requirement_id)
        for axis in expected_figure.axes:
            for requirement_ids in axis.provenance.values():
                for requirement_id in requirement_ids:
                    add(requirement_id)
    return tuple(bound_ids)
