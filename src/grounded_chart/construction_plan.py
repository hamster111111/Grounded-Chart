from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from grounded_chart.requirements import ChartRequirementPlan


@dataclass(frozen=True)
class PlanDecision:
    decision_id: str
    category: str
    value: Any
    status: str = "inferred"
    rationale: str = ""
    source_span: str = ""
    conflicts_with: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conflicts_with"] = list(self.conflicts_with)
        return payload


@dataclass(frozen=True)
class PlanValidationIssue:
    code: str
    message: str
    severity: str = "error"
    plan_ref: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanValidationReport:
    ok: bool
    issues: tuple[PlanValidationIssue, ...] = ()
    checked_contracts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "checked_contracts": list(self.checked_contracts),
        }


@dataclass(frozen=True)
class VisualLayerPlan:
    layer_id: str
    chart_type: str
    role: str
    data_source: str | None = None
    x: str | None = None
    y: tuple[str, ...] = ()
    axis: str = "primary"
    status: str = "explicit"
    rationale: str = ""
    encoding: dict[str, Any] = field(default_factory=dict)
    data_transform: tuple[dict[str, Any], ...] = ()
    components: tuple[dict[str, Any], ...] = ()
    semantic_modifiers: dict[str, Any] = field(default_factory=dict)
    visual_channel_plan: dict[str, Any] = field(default_factory=dict)
    style_policy: dict[str, Any] = field(default_factory=dict)
    placement_policy: dict[str, Any] = field(default_factory=dict)
    z_order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["y"] = list(self.y)
        payload["data_transform"] = [dict(item) for item in self.data_transform]
        payload["components"] = [dict(item) for item in self.components]
        payload["visual_channel_plan"] = dict(self.visual_channel_plan)
        return payload


@dataclass(frozen=True)
class VisualPanelPlan:
    panel_id: str
    role: str
    bounds: tuple[float, float, float, float] | None = None
    layers: tuple[VisualLayerPlan, ...] = ()
    axes: dict[str, Any] = field(default_factory=dict)
    layout_notes: tuple[str, ...] = ()
    anchor: dict[str, Any] = field(default_factory=dict)
    placement_policy: dict[str, Any] = field(default_factory=dict)
    avoid_occlusion: tuple[str, ...] = ()
    style_policy: dict[str, Any] = field(default_factory=dict)
    z_order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "role": self.role,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "layers": [layer.to_dict() for layer in self.layers],
            "axes": dict(self.axes),
            "layout_notes": list(self.layout_notes),
            "anchor": dict(self.anchor),
            "placement_policy": dict(self.placement_policy),
            "avoid_occlusion": list(self.avoid_occlusion),
            "style_policy": dict(self.style_policy),
            "z_order": self.z_order,
        }


@dataclass(frozen=True)
class ChartConstructionPlan:
    plan_type: str
    layout_strategy: str
    figure_size: tuple[float, float] | None = None
    panels: tuple[VisualPanelPlan, ...] = ()
    global_elements: tuple[dict[str, Any], ...] = ()
    decisions: tuple[PlanDecision, ...] = ()
    assumptions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    data_transform_plan: tuple[dict[str, Any], ...] = ()
    execution_steps: tuple[dict[str, Any], ...] = ()
    style_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_type": self.plan_type,
            "layout_strategy": self.layout_strategy,
            "figure_size": list(self.figure_size) if self.figure_size is not None else None,
            "panels": [panel.to_dict() for panel in self.panels],
            "global_elements": [dict(item) for item in self.global_elements],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "assumptions": list(self.assumptions),
            "constraints": list(self.constraints),
            "data_transform_plan": [dict(item) for item in self.data_transform_plan],
            "execution_steps": [dict(item) for item in self.execution_steps],
            "style_policy": dict(self.style_policy),
        }


def chart_construction_plan_from_dict(raw: dict[str, Any]) -> ChartConstructionPlan:
    """Parse a JSON-like construction plan payload into typed plan objects."""

    if not isinstance(raw, dict):
        raise TypeError("construction plan payload must be a mapping")
    figure_size = _float_pair(raw.get("figure_size"))
    return ChartConstructionPlan(
        plan_type=str(raw.get("plan_type") or "chart_construction_plan_v2"),
        layout_strategy=str(raw.get("layout_strategy") or "single_panel"),
        figure_size=figure_size,
        panels=tuple(_visual_panel_from_dict(item) for item in _dict_items(raw.get("panels"))),
        global_elements=tuple(dict(item) for item in _dict_items(raw.get("global_elements"))),
        decisions=tuple(_plan_decision_from_dict(item) for item in _dict_items(raw.get("decisions"))),
        assumptions=tuple(_string_items(raw.get("assumptions"))),
        constraints=tuple(_string_items(raw.get("constraints"))),
        data_transform_plan=tuple(dict(item) for item in _dict_items(raw.get("data_transform_plan"))),
        execution_steps=tuple(dict(item) for item in _dict_items(raw.get("execution_steps"))),
        style_policy=dict(raw.get("style_policy") or {}) if isinstance(raw.get("style_policy"), dict) else {},
    )


def _plan_decision_from_dict(raw: dict[str, Any]) -> PlanDecision:
    return PlanDecision(
        decision_id=str(raw.get("decision_id") or "decision"),
        category=str(raw.get("category") or "planning"),
        value=raw.get("value"),
        status=str(raw.get("status") or "inferred"),
        rationale=str(raw.get("rationale") or ""),
        source_span=str(raw.get("source_span") or ""),
        conflicts_with=tuple(_string_items(raw.get("conflicts_with"))),
    )


def _visual_panel_from_dict(raw: dict[str, Any]) -> VisualPanelPlan:
    return VisualPanelPlan(
        panel_id=str(raw.get("panel_id") or "panel"),
        role=str(raw.get("role") or "panel"),
        bounds=_bounds_tuple(raw.get("bounds")),
        layers=tuple(_visual_layer_from_dict(item) for item in _dict_items(raw.get("layers"))),
        axes=dict(raw.get("axes") or {}) if isinstance(raw.get("axes"), dict) else {},
        layout_notes=tuple(_string_items(raw.get("layout_notes"))),
        anchor=dict(raw.get("anchor") or {}) if isinstance(raw.get("anchor"), dict) else {},
        placement_policy=dict(raw.get("placement_policy") or {}) if isinstance(raw.get("placement_policy"), dict) else {},
        avoid_occlusion=tuple(_string_items(raw.get("avoid_occlusion"))),
        style_policy=dict(raw.get("style_policy") or {}) if isinstance(raw.get("style_policy"), dict) else {},
        z_order=_optional_int(raw.get("z_order")),
    )


def _visual_layer_from_dict(raw: dict[str, Any]) -> VisualLayerPlan:
    return VisualLayerPlan(
        layer_id=str(raw.get("layer_id") or raw.get("chart_type") or "layer"),
        chart_type=str(raw.get("chart_type") or ""),
        role=str(raw.get("role") or "visual_layer"),
        data_source=str(raw.get("data_source")) if raw.get("data_source") is not None else None,
        x=str(raw.get("x")) if raw.get("x") is not None else None,
        y=tuple(_string_items(raw.get("y"))),
        axis=str(raw.get("axis") or "primary"),
        status=str(raw.get("status") or "explicit"),
        rationale=str(raw.get("rationale") or ""),
        encoding=dict(raw.get("encoding") or {}) if isinstance(raw.get("encoding"), dict) else {},
        data_transform=tuple(dict(item) for item in _dict_items(raw.get("data_transform"))),
        components=tuple(dict(item) for item in _dict_items(raw.get("components"))),
        semantic_modifiers=dict(raw.get("semantic_modifiers") or {}) if isinstance(raw.get("semantic_modifiers"), dict) else {},
        visual_channel_plan=dict(raw.get("visual_channel_plan") or {}) if isinstance(raw.get("visual_channel_plan"), dict) else {},
        style_policy=dict(raw.get("style_policy") or {}) if isinstance(raw.get("style_policy"), dict) else {},
        placement_policy=dict(raw.get("placement_policy") or {}) if isinstance(raw.get("placement_policy"), dict) else {},
        z_order=_optional_int(raw.get("z_order")),
    )


def _dict_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _string_items(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw if str(item)]
    return [str(raw)]


def _float_pair(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError):
        return None


def _bounds_tuple(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None


def _optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class HeuristicChartConstructionPlanner:
    """Build an executable whole-figure plan from requirements and source context.

    This is intentionally conservative: it may add inferred layout decisions that
    make execution practical, but it records them as inferred and does not treat
    them as source-grounded requirements.
    """

    def build_plan(
        self,
        *,
        query: str,
        requirement_plan: ChartRequirementPlan | None = None,
        source_data_plan: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChartConstructionPlan:
        text = str(query or "")
        evidence_text = _planning_evidence_text(query, context=context)
        normalized = evidence_text.lower()
        files = _source_file_names(source_data_plan)
        explicit_chart_types = _chart_types_from_text(normalized)
        wants_pies = "pie" in explicit_chart_types
        wants_area = "area" in explicit_chart_types or "stacked area" in normalized
        wants_bar_or_waterfall = "waterfall" in explicit_chart_types or "bar" in explicit_chart_types
        wants_dual_axis = any(phrase in normalized for phrase in ("secondary y-axis", "dual y", "dual-axis", "twinx"))
        wants_overlay = any(phrase in normalized for phrase in ("overlay", "embed", "inset", "multi-layer", "multi layered", "cohesive"))

        decisions: list[PlanDecision] = []
        panels: list[VisualPanelPlan] = []
        global_elements: list[dict[str, Any]] = []
        assumptions: list[str] = []
        data_transform_plan = _data_transform_plan(explicit_chart_types, files, normalized)
        execution_steps = _execution_steps(explicit_chart_types, files, normalized)
        style_policy = _style_policy(explicit_chart_types, normalized)

        layout_strategy = "single_panel"
        figure_size = (12.0, 7.0)
        if wants_pies and (wants_overlay or wants_bar_or_waterfall or wants_area):
            layout_strategy = "main_axes_with_top_insets"
            figure_size = (16.0, 10.0)
            decisions.append(
                PlanDecision(
                    decision_id="layout.main_with_top_insets",
                    category="layout",
                    value={
                        "main_panel": "time_series_composite",
                        "inset_role": "pie_charts",
                        "placement": "top_band_aligned_to_relevant_x_values",
                    },
                    status="inferred",
                    rationale="Composite requests with embedded pies need reserved inset space for readable execution.",
                )
            )
            assumptions.append("Place pie charts in a top band unless the query gives exact coordinates.")
        elif _mentions_multiple_panels(normalized):
            layout_strategy = "multi_panel_grid"
            figure_size = (14.0, 9.0)
            decisions.append(
                PlanDecision(
                    decision_id="layout.multi_panel_grid",
                    category="layout",
                    value={"grid": _infer_grid(normalized)},
                    status="explicit" if re.search(r"\d+\s*(x|by)\s*\d+", normalized) else "inferred",
                    rationale="Multiple panels need an explicit grid for deterministic execution.",
                    source_span=_first_layout_span(text),
                )
            )

        main_layers = _main_layers(explicit_chart_types, files, normalized)
        if wants_area:
            area_modifiers = _area_semantic_modifiers(evidence_text)
            composition_policy = str(area_modifiers.get("composition") or "additive_stack")
            main_layers.append(
                VisualLayerPlan(
                    layer_id="layer.consumption_area",
                    chart_type="area",
                    role="stacked_area",
                    data_source=_match_file(files, "consumption"),
                    x="Year",
                    y=("Urban", "Rural"),
                    axis="secondary" if wants_dual_axis else "primary",
                    status="explicit",
                    rationale="Area/stacked area requirement detected in instruction.",
                    encoding={
                        "x": "Year",
                        "series": ["Urban", "Rural"],
                        "mark": "area",
                        "composition": composition_policy,
                    },
                    data_transform=(
                        {"op": "sort", "by": "Year", "direction": "ascending", "status": "inferred_from_time_axis"},
                        {
                            "op": "preserve_or_melt_wide_series",
                            "x": "Year",
                            "series_columns": ["Urban", "Rural"],
                            "status": "inferred_from_chart_type",
                        },
                        {
                            "op": "derive_area_fill_geometry",
                            "composition": composition_policy,
                            "status": "derived_from_semantic_modifiers",
                        },
                    ),
                    semantic_modifiers=area_modifiers,
                    style_policy={
                        "alpha": area_modifiers.get("alpha", 0.35),
                        "stacked": composition_policy == "additive_stack",
                        "opacity": area_modifiers.get("opacity"),
                    },
                    z_order=1,
                )
            )
        panels.append(
            VisualPanelPlan(
                panel_id="panel.main",
                role="primary_composite_chart",
                bounds=(0.08, 0.12, 0.74, 0.74) if layout_strategy == "main_axes_with_top_insets" else None,
                layers=tuple(main_layers),
                axes={
                    "x": "common year axis" if "year" in normalized or any(_has_column_hint(source_data_plan, "Year") for _ in [0]) else None,
                    "primary_y": "main quantitative axis",
                    "secondary_y": "secondary quantitative axis" if wants_dual_axis else None,
                },
                layout_notes=(
                    "Keep the main chart readable before adding inset or legend elements.",
                    "Use shared x positions for layers that refer to years.",
                ),
                placement_policy={
                    "reserve_top_band_for_insets": layout_strategy == "main_axes_with_top_insets",
                    "reserve_right_margin_for_legend": "legend" in normalized,
                    "shared_x_coordinate_system": "use one coordinate basis across primary axis, secondary axis, and inset anchors",
                },
                avoid_occlusion=("title", "legend", "inset_pie_chart"),
                style_policy={
                    "background": "plain",
                    "grid": "light_y_grid",
                    "main_axis_priority": "waterfall_or_bar_then_overlays",
                },
                z_order=0,
            )
        )

        if wants_pies:
            pie_years = _extract_years(text)
            if not pie_years:
                assumptions.append("Use evenly spaced inset positions for pie charts because exact target years are not specified.")
            for index, year in enumerate(pie_years or ()):
                left = 0.18 + min(index, 4) * 0.18
                panels.append(
                    VisualPanelPlan(
                        panel_id=f"panel.pie_{year}",
                        role="inset_pie_chart",
                        bounds=(left, 0.76, 0.12, 0.12),
                        layers=(
                            VisualLayerPlan(
                                layer_id=f"layer.pie_{year}",
                                chart_type="pie",
                                role="ratio_breakdown",
                                data_source=_match_file(files, "ratio"),
                                x="Age Group",
                                y=("Consumption Ratio",),
                                axis="inset",
                                status="explicit",
                                rationale=f"Pie chart required for {year}.",
                                encoding={"filter": {"Year": year}, "labels": "Age Group", "values": "Consumption Ratio"},
                                data_transform=(
                                    {"op": "filter", "column": "Year", "value": year, "status": "explicit"},
                                    {"op": "preserve_category_value", "category": "Age Group", "value": "Consumption Ratio"},
                                ),
                                style_policy={
                                    "labels": "compact",
                                    "autopct": "percentage_if_readable",
                                    "wedge_width": "default",
                                },
                                z_order=3,
                            ),
                        ),
                        axes={"title": str(year), "show_ticks": False},
                        anchor={"type": "x_value", "value": year, "axis_ref": "panel.main.x"},
                        placement_policy={
                            "preferred_region": "top_band",
                            "align_to_anchor_x": True,
                            "fallback": "even_spacing_without_overlap",
                        },
                        avoid_occlusion=("waterfall_bars", "stacked_area", "legend", "title"),
                        layout_notes=("Inset pie should be visually tied to its corresponding year.",),
                        z_order=3,
                    )
                )

        if "legend" in normalized:
            global_elements.append(
                {
                    "type": "legend",
                    "placement": "bottom_or_inside_free_space",
                    "status": "explicit",
                    "avoid_occlusion": ["right_edge_crop", "inset_pie_chart", "title"],
                    "style_policy": {"compact": True, "frame": True, "columns": "auto"},
                }
            )
        else:
            global_elements.append(
                {
                    "type": "legend",
                    "placement": "bottom_or_inside_free_space",
                    "status": "inferred",
                    "avoid_occlusion": ["right_edge_crop", "inset_pie_chart", "title"],
                    "style_policy": {"compact": True, "frame": True, "columns": "auto"},
                }
            )
        title = _quoted_title(text)
        if title:
            global_elements.append({"type": "title", "text": title, "status": "explicit"})

        constraints = [
            "Inferred layout decisions may support execution but must not contradict explicit requirements.",
            "Do not replace explicit data-source requirements with synthetic data.",
            "Prefer a readable whole-figure layout over placing every element in one crowded axis.",
            "If exact placement is unspecified, choose stable normalized bounds and record the decision as inferred.",
            "For overlaid or twinx layers, use one shared x-coordinate basis; do not mix categorical/index positions with raw year values.",
            "When using manual axes/insets, avoid layout managers that can reposition axes unpredictably unless explicitly verified.",
            "Do not use vague operations such as 'compute yearly change' unless the input semantic, precondition, output artifact, and assertion are explicit.",
            "If a data semantic is uncertain, mark the step as needs_evidence instead of inventing a transformation.",
            "Preserve visual semantic modifiers such as composition, opacity, axis range, layer order, overlap, grouping, and normalization as executable plan fields.",
        ]
        return ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy=layout_strategy,
            figure_size=figure_size,
            panels=tuple(panels),
            global_elements=tuple(global_elements),
            decisions=tuple(decisions),
            assumptions=tuple(assumptions),
            constraints=tuple(constraints),
            data_transform_plan=tuple(data_transform_plan),
            execution_steps=tuple(execution_steps),
            style_policy=style_policy,
        )


class PlanValidator:
    """Validate whether a construction plan is executable and evidence-aligned."""

    def validate(
        self,
        plan: ChartConstructionPlan | dict[str, Any],
        *,
        query: str = "",
        source_data_plan: Any | None = None,
    ) -> PlanValidationReport:
        payload = plan.to_dict() if isinstance(plan, ChartConstructionPlan) else dict(plan)
        normalized = str(query or "").lower()
        explicit_chart_types = _chart_types_from_text(normalized)
        issues: list[PlanValidationIssue] = []
        checked = [
            "plan_type",
            "visual_layer_bindings",
            "source_data_bindings",
            "inset_anchors",
            "waterfall_components",
            "global_layout",
        ]

        if payload.get("plan_type") != "chart_construction_plan_v2":
            issues.append(
                PlanValidationIssue(
                    code="outdated_construction_plan_type",
                    message="Construction plan should use chart_construction_plan_v2.",
                    severity="warning",
                    plan_ref="construction_plan.plan_type",
                    evidence=str(payload.get("plan_type")),
                )
            )

        panels = [item for item in list(payload.get("panels") or []) if isinstance(item, dict)]
        source_files = set(_source_file_names(source_data_plan))
        source_columns = _source_columns_by_file(source_data_plan)
        layers = _layers_from_panels(panels)
        if _needs_visual_layer(normalized) and not layers:
            issues.append(
                PlanValidationIssue(
                    code="missing_visual_layers",
                    message="The request asks for a plot/chart, but the construction plan has no visual layers.",
                    severity="warning",
                    plan_ref="construction_plan.panels",
                )
            )

        for layer in layers:
            issues.extend(
                _validate_layer_plan(
                    layer,
                    normalized=normalized,
                    explicit_chart_types=explicit_chart_types,
                    source_files=source_files,
                    source_columns=source_columns,
                )
            )

        for panel in panels:
            issues.extend(_validate_panel_plan(panel, normalized=normalized))

        issues.extend(_validate_global_elements(payload))
        blocking = [issue for issue in issues if issue.severity == "error"]
        return PlanValidationReport(ok=not blocking, issues=tuple(issues), checked_contracts=tuple(checked))


def validate_construction_plan(
    plan: ChartConstructionPlan | dict[str, Any],
    *,
    query: str = "",
    source_data_plan: Any | None = None,
) -> PlanValidationReport:
    return PlanValidator().validate(plan, query=query, source_data_plan=source_data_plan)


def _source_file_names(source_data_plan: Any | None) -> list[str]:
    if source_data_plan is None:
        return []
    files = getattr(source_data_plan, "files", None)
    if files is not None:
        return [str(getattr(item, "name", "")) for item in files if getattr(item, "name", "")]
    if isinstance(source_data_plan, dict):
        return [str(item.get("name") or "") for item in list(source_data_plan.get("files") or []) if isinstance(item, dict)]
    return []


def _source_columns_by_file(source_data_plan: Any | None) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    if source_data_plan is None:
        return result
    files = getattr(source_data_plan, "files", None)
    if files is not None:
        for item in files:
            name = str(getattr(item, "name", ""))
            if name:
                result[name] = {str(col) for col in getattr(item, "columns", ())}
        return result
    if isinstance(source_data_plan, dict):
        for item in list(source_data_plan.get("files") or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if name:
                result[name] = {str(col) for col in item.get("columns", ())}
    return result


def _layers_from_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for panel in panels:
        for layer in list(panel.get("layers") or []):
            if isinstance(layer, dict):
                layers.append(layer)
    return layers


def _needs_visual_layer(normalized: str) -> bool:
    return bool(_chart_types_from_text(normalized)) or any(word in normalized for word in ("plot", "graph", "chart", "visualize"))


def _validate_layer_plan(
    layer: dict[str, Any],
    *,
    normalized: str,
    explicit_chart_types: set[str],
    source_files: set[str],
    source_columns: dict[str, set[str]],
) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    layer_id = str(layer.get("layer_id") or layer.get("chart_type") or "layer")
    chart_type = str(layer.get("chart_type") or "").lower()
    status = str(layer.get("status") or "explicit")
    data_source = str(layer.get("data_source") or "")
    if status == "explicit" and chart_type and chart_type not in explicit_chart_types and chart_type != "waterfall":
        if not _explicit_layer_is_supported_by_semantics(chart_type, normalized):
            issues.append(
                PlanValidationIssue(
                    code="explicit_layer_not_supported_by_query",
                    message=f"Layer `{layer_id}` is marked explicit, but its chart type is not explicitly requested.",
                    severity="error",
                    plan_ref=f"construction_plan.layers.{layer_id}.status",
                    evidence=chart_type,
                )
            )
    if status == "explicit" and source_files and not data_source:
        issues.append(
            PlanValidationIssue(
                code="missing_layer_data_source",
                message=f"Explicit layer `{layer_id}` has no data_source despite available source files.",
                severity="error",
                plan_ref=f"construction_plan.layers.{layer_id}.data_source",
            )
        )
    if data_source and source_files and data_source not in source_files:
        issues.append(
            PlanValidationIssue(
                code="unknown_layer_data_source",
                message=f"Layer `{layer_id}` references unknown data source `{data_source}`.",
                severity="error",
                plan_ref=f"construction_plan.layers.{layer_id}.data_source",
            )
        )
    expected_columns = source_columns.get(data_source, set()) if data_source else set()
    if expected_columns:
        for column in _layer_columns(layer):
            if column and column not in expected_columns:
                issues.append(
                    PlanValidationIssue(
                        code="layer_column_not_in_source",
                        message=f"Layer `{layer_id}` references column `{column}` not found in `{data_source}`.",
                        severity="warning",
                        plan_ref=f"construction_plan.layers.{layer_id}",
                        evidence=f"{data_source}: {sorted(expected_columns)}",
                    )
                )
    if chart_type == "waterfall":
        components = {str(item.get("type") or "") for item in list(layer.get("components") or []) if isinstance(item, dict)}
        required = {"start_bars", "delta_bars", "final_total_bars", "connector_lines"}
        missing = sorted(required - components)
        if missing:
            issues.append(
                PlanValidationIssue(
                    code="underspecified_waterfall_components",
                    message=f"Waterfall layer `{layer_id}` is missing component plan entries: {missing}.",
                    severity="warning",
                    plan_ref=f"construction_plan.layers.{layer_id}.components",
                )
            )
    if chart_type in {"bar", "line", "area", "waterfall"} and "year" in normalized and not layer.get("x"):
        issues.append(
            PlanValidationIssue(
                code="missing_time_axis_binding",
                message=f"Layer `{layer_id}` should bind the year/time axis explicitly.",
                severity="warning",
                plan_ref=f"construction_plan.layers.{layer_id}.x",
            )
        )
    return issues


def _validate_panel_plan(panel: dict[str, Any], *, normalized: str) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    panel_id = str(panel.get("panel_id") or "panel")
    role = str(panel.get("role") or "")
    anchor = panel.get("anchor") if isinstance(panel.get("anchor"), dict) else {}
    bounds = panel.get("bounds")
    if role == "inset_pie_chart":
        if "corresponding year" in normalized or "corresponding years" in normalized or "align" in normalized:
            if not anchor or anchor.get("type") != "x_value":
                issues.append(
                    PlanValidationIssue(
                        code="missing_inset_x_value_anchor",
                        message=f"Inset panel `{panel_id}` should be anchored to an x-axis value instead of only fixed bounds.",
                        severity="error",
                        plan_ref=f"construction_plan.panels.{panel_id}.anchor",
                    )
                )
        if bounds is None:
            issues.append(
                PlanValidationIssue(
                    code="missing_inset_bounds",
                    message=f"Inset panel `{panel_id}` has no fallback bounds.",
                    severity="warning",
                    plan_ref=f"construction_plan.panels.{panel_id}.bounds",
                )
            )
    return issues


def _validate_global_elements(plan: dict[str, Any]) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    for element in list(plan.get("global_elements") or []):
        if not isinstance(element, dict):
            continue
        if str(element.get("type") or "").lower() == "legend":
            placement = str(element.get("placement") or "")
            if placement == "outside_right_or_bottom":
                issues.append(
                    PlanValidationIssue(
                        code="ambiguous_legend_placement",
                        message="Legend placement should not be an ambiguous right-or-bottom choice.",
                        severity="warning",
                        plan_ref="construction_plan.global_elements.legend.placement",
                    )
                )
    return issues


def _layer_columns(layer: dict[str, Any]) -> list[str]:
    columns = []
    if layer.get("x"):
        columns.append(str(layer.get("x")))
    for item in list(layer.get("y") or []):
        columns.append(str(item))
    encoding = layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {}
    for key in ("x", "labels", "values"):
        value = encoding.get(key)
        if isinstance(value, str):
            columns.append(value)
    filter_spec = encoding.get("filter")
    if isinstance(filter_spec, dict):
        columns.extend(str(key) for key in filter_spec.keys())
    return sorted(set(columns))


def _explicit_layer_is_supported_by_semantics(chart_type: str, normalized: str) -> bool:
    if chart_type == "line":
        return _wants_trend_line(normalized)
    if chart_type == "area":
        return "stacked area" in normalized
    return False


def _chart_types_from_text(normalized: str) -> set[str]:
    chart_types = set()
    phrases = {
        "waterfall": (r"\bwaterfall(?:\s+chart)?\b",),
        "bar": (r"\bbar(?:\s+chart)?\b", r"\bbars\b"),
        "area": (r"\bstacked\s+area(?:\s+chart)?\b", r"\barea(?:\s+chart)?\b"),
        "line": (r"\bline(?:\s+chart)?\b", r"\btrend\s+line\b"),
        "pie": (r"\bpie(?:\s+chart)?s?\b",),
        "scatter": (r"\bscatter(?:\s+plot)?\b",),
        "heatmap": (r"\bheat\s*map\b", r"\bheatmap\b"),
        "box": (r"\bbox(?:\s+plot)?\b",),
    }
    for name, patterns in phrases.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            chart_types.add(name)
    return chart_types


def _main_layers(chart_types: set[str], files: list[str], normalized: str) -> list[VisualLayerPlan]:
    layers: list[VisualLayerPlan] = []
    if "waterfall" in chart_types:
        layers.append(
            VisualLayerPlan(
                layer_id="layer.import_waterfall",
                chart_type="waterfall",
                role="yearly_change_bars",
                data_source=_match_file(files, "import"),
                x="Year",
                y=("Urban", "Rural"),
                axis="primary",
                status="explicit",
                rationale="Waterfall requirement detected in instruction.",
                encoding={
                    "x": "Year",
                    "series": ["Urban", "Rural"],
                    "mark": "bar",
                    "semantics": "direct source values with initial/change/final-total roles",
                },
                data_transform=(
                    {"op": "sort", "by": "Year", "direction": "ascending", "status": "inferred_from_time_axis"},
                    {
                        "op": "direct_use_source_values",
                        "columns": ["Urban", "Rural"],
                        "precondition": "Imports.csv values are the values to plot for each year; do not apply adjacent-row differencing unless the source explicitly stores cumulative totals.",
                        "output": "execution/round_1/step_02_imports_waterfall_values.csv",
                    },
                    {
                        "op": "assign_waterfall_roles",
                        "roles": {"first_year": "initial", "middle_years": "annual_change", "last_year": "cumulative_total"},
                    },
                    {
                        "op": "compute_waterfall_render_geometry",
                        "input": "execution/round_1/step_02_imports_waterfall_values.csv",
                        "output": "execution/round_1/step_02_imports_waterfall_render_table.csv",
                        "semantics": "convert source initial/delta/total rows into bar_bottom, bar_height, and bar_top for plotting",
                    },
                ),
                components=(
                    {"type": "start_bars", "status": "inferred_from_chart_type"},
                    {"type": "delta_bars", "positive_color": "imports_positive", "negative_color": "imports_negative"},
                    {"type": "final_total_bars", "status": "inferred_from_chart_type"},
                    {"type": "connector_lines", "style": "dotted", "status": "inferred_from_chart_type"},
                    {"type": "cumulative_markers", "marker": "square", "status": "inferred_from_chart_type"},
                ),
                visual_channel_plan={
                    "channel_contract": "multi_semantic_waterfall_encoding_v1",
                    "dimensions": {
                        "series_identity": {
                            "field": "series",
                            "values": ["Urban", "Rural"],
                            "source": "Imports.csv wide columns",
                            "status": "explicit_data_schema",
                        },
                        "change_role": {
                            "field": "color_role",
                            "values": ["increase", "decrease", "total"],
                            "source": "waterfall render geometry",
                            "status": "inferred_from_waterfall_protocol",
                        },
                    },
                    "channel_allocation": {
                        "x_group_offset": {
                            "field": "series",
                            "purpose": "distinguish Urban and Rural side-by-side within each year",
                            "required": True,
                        },
                        "fill_color": {
                            "field": "color_role",
                            "purpose": "distinguish increase, decrease, and final total bars",
                            "required": True,
                        },
                        "secondary_series_cue": {
                            "field": "series",
                            "preferred_channels": ["edge_style", "hatch", "compact_series_legend"],
                            "required_if_series_identity_is_not_visually_clear": True,
                        },
                    },
                    "legend_policy": {
                        "avoid_flattening_dimensions": True,
                        "represent_change_role_and_series_identity_as_separate_semantics": True,
                    },
                },
                style_policy={
                    "positive_color": "muted_green",
                    "negative_color": "muted_red_or_blue",
                    "connector_style": "dotted",
                    "marker": "square",
                    "alpha": 0.85,
                },
                z_order=2,
            )
        )
    elif "bar" in chart_types:
        layers.append(
            VisualLayerPlan(
                layer_id="layer.bar",
                chart_type="bar",
                role="main_bar_layer",
                data_source=_match_file(files, "import") or _first_file(files),
                x="Year" if "year" in normalized else None,
                axis="primary",
                status="explicit",
                rationale="Bar requirement detected in instruction.",
                encoding={"x": "Year" if "year" in normalized else None, "mark": "bar"},
                style_policy={"alpha": 0.85},
                z_order=2,
            )
        )
    if "line" in chart_types or _wants_trend_line(normalized):
        layers.append(
            VisualLayerPlan(
                layer_id="layer.trend_line",
                chart_type="line",
                role="trend_or_cumulative_total",
                data_source=_match_file(files, "import") or _first_file(files),
                x="Year" if "year" in normalized else None,
                axis="primary",
                status="explicit" if "line" in chart_types else "inferred",
                rationale="Trend/cumulative wording benefits from a line layer for executable visual structure.",
                encoding={"x": "Year" if "year" in normalized else None, "mark": "line", "semantics": "trend_or_cumulative_total"},
                data_transform=(
                    {"op": "sort", "by": "Year", "direction": "ascending", "status": "inferred_from_time_axis"},
                ),
                style_policy={"line_style": "dashed", "marker": "diamond", "linewidth": 1.8},
                z_order=4,
            )
        )
    return layers


def _data_transform_plan(chart_types: set[str], files: list[str], normalized: str) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = []
    if "year" in normalized:
        transforms.append(
            {
                "transform_id": "transform.sort_by_year",
                "op": "sort",
                "by": "Year",
                "direction": "ascending",
                "status": "inferred_from_time_axis",
                "applies_to": files or "all_tables_with_year",
            }
        )
    if "waterfall" in chart_types:
        transforms.extend(
            [
                {
                    "transform_id": "transform.imports_waterfall_direct_values",
                    "op": "direct_use_source_values",
                    "columns": ["Urban", "Rural"],
                    "status": "inferred_from_chart_type",
                    "input": _match_file(files, "import") or "Imports.csv",
                    "output": "execution/round_1/step_02_imports_waterfall_values.csv",
                    "precondition": "The source table columns are the values to display for each year.",
                    "assertion": "plotted Urban/Rural values must equal source Urban/Rural values row by row.",
                },
                {
                    "transform_id": "transform.imports_waterfall_render_geometry",
                    "op": "compute_waterfall_render_geometry",
                    "columns": ["Urban", "Rural"],
                    "status": "inferred_from_chart_type_protocol",
                    "input": "execution/round_1/step_02_imports_waterfall_values.csv",
                    "output": "execution/round_1/step_02_imports_waterfall_render_table.csv",
                    "assertion": "delta bars start at the previous cumulative total; final total bars start from zero.",
                },
            ]
        )
    if "area" in chart_types:
        area_modifiers = _area_semantic_modifiers(normalized)
        transforms.append(
            {
                "transform_id": "transform.area_wide_to_series",
                "op": "preserve_or_melt_wide_series",
                "x": "Year",
                "series_columns": ["Urban", "Rural"],
                "status": "inferred_from_chart_type",
                "semantic_modifiers": area_modifiers,
            }
        )
    if "pie" in chart_types:
        transforms.append(
            {
                "transform_id": "transform.pie_filter_by_anchor",
                "op": "filter_per_requested_anchor",
                "anchor_column": "Year",
                "category": "Age Group",
                "value": "Consumption Ratio",
                "status": "explicit_if_years_mentioned_else_inferred",
            }
        )
    return transforms


def _execution_steps(chart_types: set[str], files: list[str], normalized: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if files:
        steps.append(
            {
                "step_id": "step_01_load_sources",
                "stage": "execution",
                "action": "load_and_summarize_sources",
                "inputs": files,
                "outputs": ["execution/round_1/step_01_sources_summary.json"],
                "purpose": "Load source files deterministically and expose schema/row counts before plotting.",
                "script": "execution/round_1/step_01_load_sources.py",
            }
        )
    if "waterfall" in chart_types:
        steps.append(
            {
                "step_id": "step_02_prepare_imports_waterfall",
                "stage": "execution",
                "action": "sort_copy_import_values_and_compute_waterfall_geometry",
                "inputs": [_match_file(files, "import") or "Imports.csv"],
                "outputs": [
                    "execution/round_1/step_02_imports_waterfall_values.csv",
                    "execution/round_1/step_02_imports_waterfall_render_table.csv",
                ],
                "purpose": "Prepare source values and protocol-grounded bar geometry for the import waterfall layer.",
                "script": "execution/round_1/step_02_transform_imports.py",
                "assertions": [
                    "row_count equals source row_count",
                    "Year values are sorted ascending",
                    "Urban_plot equals source Urban",
                    "Rural_plot equals source Rural",
                    "render table has bar_bottom, bar_height, bar_top, role, color_role, and series",
                    "delta bar bottoms equal the previous cumulative value for the same series",
                    "no adjacent-row differencing is applied unless a future plan step explicitly justifies it",
                ],
            }
        )
    if "area" in chart_types:
        area_modifiers = _area_semantic_modifiers(normalized)
        composition_policy = str(area_modifiers.get("composition") or "additive_stack")
        steps.append(
            {
                "step_id": "step_03_prepare_consumption_area",
                "stage": "execution",
                "action": "sort_and_prepare_area_fill_geometry",
                "inputs": [_match_file(files, "consumption") or "Consumption.csv"],
                "outputs": ["execution/round_1/step_03_consumption_area_values.csv"],
                "purpose": "Prepare Urban/Rural consumption values and modifier-grounded fill geometry for the secondary area layer.",
                "script": "execution/round_1/step_03_transform_consumption.py",
                "semantic_modifiers": area_modifiers,
                "assertions": [
                    "Urban_area equals source Urban",
                    "Rural_area equals source Rural",
                    f"composition_policy equals {composition_policy}",
                    "overlap composition uses independent fill intervals rather than Urban + Rural stacking",
                    "additive_stack composition uses cumulative stack tops",
                    "x_index is shared with other year-based layers",
                ],
            }
        )
    if "pie" in chart_types:
        steps.append(
            {
                "step_id": "step_04_prepare_pie_values",
                "stage": "execution",
                "action": "filter_ratio_table_for_requested_years",
                "inputs": [_match_file(files, "ratio") or "Grain_Consumption_Ratio.csv"],
                "outputs": ["execution/round_1/step_04_pie_values.csv"],
                "purpose": "Prepare age-group ratio values for each requested pie chart.",
                "script": "execution/round_1/step_04_transform_pies.py",
                "assertions": [
                    "only requested years are included",
                    "Age Group labels are preserved",
                    "Consumption Ratio values are preserved",
                    "normalized pie percentages are recorded separately from raw values",
                ],
            }
        )
    if _needs_visual_layer(normalized):
        steps.append(
            {
                "step_id": "step_05_plot_figure",
                "stage": "execution",
                "action": "plot_from_prepared_artifacts",
                "inputs": [
                    "execution/round_1/step_02_imports_waterfall_render_table.csv",
                    "execution/round_1/step_03_consumption_area_values.csv",
                    "execution/round_1/step_04_pie_values.csv",
                ],
                "outputs": ["execution/round_1/plot_spec.md", "execution/round_1/plot.py", "figure.png"],
                "purpose": "Render the final figure using prepared artifacts rather than recomputing hidden transformations inside plotting code.",
                "script": "execution/round_1/plot.py",
            }
        )
    return steps


def _style_policy(chart_types: set[str], normalized: str) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "readability": "prioritize clear figure structure over dense decoration",
        "legend": "compact_non_occluding",
        "title": "figure_level_when_global",
        "axis_labels": "preserve_explicit_or_infer_from_columns",
    }
    if "waterfall" in chart_types:
        policy["waterfall"] = {
            "positive_negative_contrast": True,
            "connector_lines": "dotted",
            "cumulative_markers": "square_or_diamond",
        }
    if "area" in chart_types:
        modifiers = _area_semantic_modifiers(normalized)
        policy["area"] = {
            "alpha": modifiers.get("alpha", 0.35),
            "composition": modifiers.get("composition"),
            "opacity": modifiers.get("opacity"),
        }
    if "pie" in chart_types:
        policy["pie"] = {"compact_labels": True, "avoid_main_layer_occlusion": True}
    if "secondary y-axis" in normalized or "dual" in normalized:
        policy["dual_axis"] = {"separate_axis_colors": True, "avoid_scale_confusion": True}
    return policy


def _wants_trend_line(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(trend\s+line|cumulative\s+(?:line|total)|moving\s+average|rolling\s+average)\b",
            normalized,
        )
    )


def _match_file(files: list[str], keyword: str) -> str | None:
    for name in files:
        if keyword.lower() in name.lower():
            return name
    return None


def _first_file(files: list[str]) -> str | None:
    return files[0] if files else None


def _mentions_multiple_panels(normalized: str) -> bool:
    return any(word in normalized for word in ("subplot", "subplots", "panel", "panels", "grid", "facet", "mosaic"))


def _infer_grid(normalized: str) -> dict[str, int] | None:
    match = re.search(r"(\d+)\s*(?:x|by)\s*(\d+)", normalized)
    if match:
        return {"rows": int(match.group(1)), "cols": int(match.group(2))}
    return None


def _extract_years(text: str) -> tuple[int, ...]:
    years = []
    for match in re.finditer(r"\b(19\d{2}|20\d{2})\b", text):
        year = int(match.group(1))
        if year not in years:
            years.append(year)
    return tuple(years[:8])


def _quoted_title(text: str) -> str | None:
    title_match = re.search(r'title[^"\']*["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
    if title_match:
        return title_match.group(1)
    return None


def _first_layout_span(text: str) -> str:
    lowered = text.lower()
    for keyword in ("subplot", "panel", "grid", "layout", "facet", "mosaic"):
        index = lowered.find(keyword)
        if index >= 0:
            return text[max(0, index - 40) : index + 80].strip()
    return ""


def _has_column_hint(source_data_plan: Any | None, column: str) -> bool:
    if source_data_plan is None:
        return False
    expected = column.lower()
    files = getattr(source_data_plan, "files", None)
    if files is not None:
        return any(expected in {str(col).lower() for col in getattr(item, "columns", ())} for item in files)
    if isinstance(source_data_plan, dict):
        for item in list(source_data_plan.get("files") or []):
            if isinstance(item, dict) and expected in {str(col).lower() for col in item.get("columns", ())}:
                return True
    return False


def _planning_evidence_text(query: str, *, context: dict[str, Any] | None = None) -> str:
    texts: list[str] = []
    for value in (
        query,
        *(str((context or {}).get(key) or "") for key in ("instruction", "simple_instruction", "expert_instruction", "source_instruction")),
    ):
        text = str(value or "").strip()
        if text and text not in texts:
            texts.append(text)
    return "\n".join(texts)


def _area_semantic_modifiers(text: str) -> dict[str, Any]:
    normalized = str(text or "").lower()
    evidence: list[dict[str, Any]] = []
    composition = "additive_stack" if "stacked" in normalized else "independent"
    if _has_any(normalized, ("overlap", "overlapping", "superimpose", "superimposed")):
        composition = "overlap"
        evidence.append({"modifier": "composition", "value": "overlap", "source_span": _first_matching_span(text, ("overlap", "overlapping", "superimpose", "superimposed"))})
    elif "stacked" in normalized:
        evidence.append({"modifier": "composition", "value": "additive_stack", "source_span": _first_matching_span(text, ("stacked",))})
    opacity = "translucent" if _has_any(normalized, ("translucent", "transparent", "semi-transparent", "alpha")) else "opaque"
    if opacity == "translucent":
        evidence.append({"modifier": "opacity", "value": "translucent", "source_span": _first_matching_span(text, ("translucent", "transparent", "semi-transparent", "alpha"))})
    axis_range = _explicit_numeric_range_near_axis(text)
    modifiers: dict[str, Any] = {
        "base_mark": "area",
        "composition": composition,
        "opacity": opacity,
        "alpha": 0.35 if opacity == "translucent" else 0.8,
        "axis_binding": "secondary" if _has_any(normalized, ("secondary y-axis", "secondary y axis", "secondary axis", "dual y", "dual-axis")) else "primary",
        "fill_baseline": "axis_min" if composition == "overlap" and axis_range else "zero",
        "evidence": [item for item in evidence if item.get("source_span")],
    }
    if axis_range:
        modifiers["scale_policy"] = {"type": "explicit_range", "min": axis_range[0], "max": axis_range[1]}
        modifiers["evidence"].append({"modifier": "scale_policy", "value": "explicit_range", "source_span": axis_range[2]})
    else:
        modifiers["scale_policy"] = {"type": "data_driven"}
    return modifiers


def _explicit_numeric_range_near_axis(text: str) -> tuple[float, float, str] | None:
    for match in re.finditer(r"(?:from|between)\s+(-?\d+(?:\.\d+)?)\s+(?:to|and)\s+(-?\d+(?:\.\d+)?)", str(text or ""), flags=re.IGNORECASE):
        span = text[max(0, match.start() - 80) : min(len(text), match.end() + 80)]
        if re.search(r"\b(axis|scale|range|y-axis|y axis)\b", span, flags=re.IGNORECASE):
            return float(match.group(1)), float(match.group(2)), span.strip()
    return None


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _first_matching_span(text: str, needles: tuple[str, ...]) -> str:
    lowered = str(text or "").lower()
    for needle in needles:
        index = lowered.find(needle)
        if index >= 0:
            return str(text)[max(0, index - 80) : min(len(str(text)), index + len(needle) + 80)].strip()
    return ""
