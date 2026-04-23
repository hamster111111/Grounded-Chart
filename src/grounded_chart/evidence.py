from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from grounded_chart.requirements import Artifact, ChartRequirementPlan, EvidenceGraph, EvidenceLink, PanelRequirementPlan, RequirementNode
from grounded_chart.schema import AxisRequirementSpec, ChartIntentPlan, FigureRequirementSpec, FigureTrace, PlotTrace, VerificationReport


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
    figure_title: str | None = None
    axes_count: int | None = None

    panel_axis_label_requirements: dict[str, list[RequirementNode]] = {}
    panel_title_requirements: dict[str, list[RequirementNode]] = {}
    panel_artist_type_requirements: dict[str, list[RequirementNode]] = {}

    for requirement in verifiable_requirements:
        if requirement.scope == "figure":
            normalized_title = _normalize_text_value(requirement.value) if requirement.name == "title" else None
            if requirement.name == "title" and normalized_title is not None and figure_title is None:
                figure_title = normalized_title
                figure_provenance["figure_title"] = (requirement.requirement_id,)
            elif requirement.name in {"axes_count", "subplot_count"} and axes_count is None:
                parsed_axes_count = _as_int(requirement.value)
                if parsed_axes_count is not None:
                    axes_count = parsed_axes_count
                    figure_provenance["axes_count"] = (requirement.requirement_id,)
            elif requirement.name == "subplot_layout" and axes_count is None:
                parsed_axes_count = _axes_count_from_layout(requirement.value)
                if parsed_axes_count is not None:
                    axes_count = parsed_axes_count
                    figure_provenance["axes_count"] = (requirement.requirement_id,)
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

        title_requirements = panel_title_requirements.get(panel_id, [])
        if title_requirements:
            primary_title = title_requirements[0]
            normalized_title = _normalize_text_value(primary_title.value)
            if normalized_title is not None:
                axis_title = normalized_title
                axis_provenance["title"] = (primary_title.requirement_id,)

        label_mapping = _map_axis_label_requirements(panel_axis_label_requirements.get(panel_id, ()))
        for field_name, requirement in label_mapping.items():
            axis_provenance[field_name] = (requirement.requirement_id,)

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
        )

    merged_figure_provenance = _merge_figure_provenance(base, overlay)
    return FigureRequirementSpec(
        axes_count=overlay.axes_count if overlay.axes_count is not None else base.axes_count,
        figure_title=overlay.figure_title if overlay.figure_title is not None else base.figure_title,
        size_inches=overlay.size_inches if overlay.size_inches is not None else base.size_inches,
        axes=tuple(sorted(merged_axes_by_index.values(), key=lambda axis: axis.axis_index)),
        provenance=merged_figure_provenance,
    )


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
    match = re.search(r"(\d+)\s*[x×]\s*(\d+)", text)
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
                requirement_ids=_figure_requirement_ids(requirement_plan, verification.expected_figure),
                payload=verification.expected_figure,
                source="figure_requirements",
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
