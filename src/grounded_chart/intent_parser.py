from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Protocol

from grounded_chart.llm import LLMClient
from grounded_chart.requirements import ChartRequirementPlan, PanelRequirementPlan, RequirementNode, RequirementStatus
from grounded_chart.schema import (
    ChartIntentPlan,
    ChartType,
    FilterOp,
    FilterSpec,
    MeasureSpec,
    ParsedRequirementBundle,
    SortDirection,
    SortSpec,
    TableSchema,
)


class IntentParser(Protocol):
    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        """Parse a natural-language chart request into a legacy ChartIntentPlan."""

    def parse_requirements(self, query: str, schema: TableSchema) -> ParsedRequirementBundle:
        """Parse a request into a plan plus parser-native requirement provenance."""


@dataclass(frozen=True)
class _ParsedValue:
    value: Any
    source_span: str
    status: RequirementStatus
    confidence: float | None = None
    assumption: str | None = None


@dataclass(frozen=True)
class _FilterExtraction:
    spec: FilterSpec
    source_span: str
    status: RequirementStatus = "explicit"
    confidence: float | None = None
    assumption: str | None = None


_CORE_EXECUTABLE_NAMES = {"chart_type", "aggregation", "measure_column", "dimensions", "sort", "limit", "filter"}
_CORE_FALLBACK_ORDER = (
    "panel_0.chart_type",
    "panel_0.aggregation",
    "panel_0.measure_column",
    "panel_0.dimensions",
    "panel_0.sort",
    "panel_0.limit",
)
_SCHEMALESS_NAME_ALIASES = {
    "plot_type": "artist_type",
    "chart_type": "artist_type",
    "subplot_grid": "subplot_layout",
    "subplots": "subplot_layout",
    "subplot_mosaic": "subplot_layout",
    "mosaic": "subplot_layout",
    "x_label": "axis_label",
    "y_label": "axis_label",
    "z_label": "axis_label",
    "xlabel": "axis_label",
    "ylabel": "axis_label",
    "zlabel": "axis_label",
    "axis_labels": "axis_label",
    "y1_label": "axis_label",
    "y2_label": "axis_label",
    "x_axis_label": "axis_label",
    "y_axis_label": "axis_label",
    "z_axis_label": "axis_label",
    "x_limit": "axis_limit",
    "y_limit": "axis_limit",
    "z_limit": "axis_limit",
    "xlim": "axis_limit",
    "ylim": "axis_limit",
    "zlim": "axis_limit",
    "y_lim_lower": "axis_limit",
    "secondary_y_scale": "axis_limit",
    "x_scale": "axis_scale",
    "y_scale": "axis_scale",
    "z_scale": "axis_scale",
    "x_axis": "axis_spec",
    "y_axis": "axis_spec",
    "z_axis": "axis_spec",
    "title_text": "title",
    "plot_title": "title",
    "main_title": "title",
    "panel_title": "title",
    "figure_title": "title",
    "size": "figure_size",
    "size_inches": "figure_size",
    "projection_type": "projection",
    "subplot_type": "projection",
    "show_legend": "legend",
    "legend_labels": "legend",
    "dimension": "data_dimension",
    "dimensions": "data_dimension",
}
_SCHEMALESS_META_REQUIREMENT_NAMES = {
    "libraries",
    "library",
    "plotting_library",
    "visualization_library",
    "programming_language",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "python",
    "python_script",
    "generate_python_script",
    "create_python_script",
    "write_python_code",
    "write_code",
}
_SCHEMALESS_DISPLAY_REQUIREMENT_NAMES = {"show_plot", "display_plot", "display", "plot"}


class HeuristicIntentParser:
    """Small parser baseline with parser-native requirement provenance.

    This remains intentionally conservative. It extracts only requirements that
    can be tied to the query or clearly labelled as schema-driven assumptions.
    """

    CHART_TYPES: tuple[ChartType, ...] = ("bar", "line", "pie", "scatter", "area", "heatmap", "box")

    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        return self.parse_requirements(query, schema).plan

    def parse_requirements(self, query: str, schema: TableSchema) -> ParsedRequirementBundle:
        chart_type = self._chart_type(query)
        aggregation = self._aggregation(query)
        measure = self._measure_column(query, schema)
        dimensions = self._dimension_columns(query, schema, exclude=measure.value)
        limit, implied_sort_direction = self._limit(query)
        sort = self._sort(query)
        if sort.value is None and implied_sort_direction is not None:
            sort = _ParsedValue(
                SortSpec(by="measure", direction=implied_sort_direction),
                limit.source_span,
                "explicit",
                confidence=0.75,
            )
        filters = self._filters(query, schema)
        plan = ChartIntentPlan(
            chart_type=chart_type.value,
            dimensions=dimensions.value,
            measure=MeasureSpec(column=measure.value, agg=aggregation.value),
            filters=tuple(filter_extraction.spec for filter_extraction in filters),
            sort=sort.value,
            limit=limit.value,
            raw_query=query,
            confidence=0.35,
        )
        requirement_plan = _build_requirement_plan_from_parse(
            plan,
            fields={
                "chart_type": chart_type,
                "aggregation": aggregation,
                "measure_column": measure,
                "dimensions": dimensions,
                "sort": sort,
                "limit": limit,
            },
            filters=filters,
        )
        return ParsedRequirementBundle(plan=plan, requirement_plan=requirement_plan)

    def _chart_type(self, query: str) -> _ParsedValue:
        for chart_type in self.CHART_TYPES:
            span = _first_span(query, (rf"\b{re.escape(chart_type)}\s+(?:chart|plot|graph)\b", rf"\b{re.escape(chart_type)}\b"))
            if span:
                return _ParsedValue(chart_type, span, "explicit", confidence=0.8)
        return _ParsedValue(
            "bar",
            "",
            "assumed",
            confidence=0.25,
            assumption="Defaulted to a bar chart because the query did not name a chart type.",
        )

    def _aggregation(self, query: str) -> _ParsedValue:
        patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("count", (r"\bcount\b", r"\bnumber of\b", r"\bhow many\b")),
            ("mean", (r"\bavg\b", r"\baverage\b", r"\bmean\b")),
            ("max", (r"\bmax\b", r"\bmaximum\b", r"\bhighest\b")),
            ("min", (r"\bmin\b", r"\bminimum\b", r"\blowest\b")),
            ("sum", (r"\bsum\b", r"\btotal\b")),
        )
        for aggregation, regexes in patterns:
            span = _first_span(query, regexes)
            if span:
                return _ParsedValue(aggregation, span, "explicit", confidence=0.8)
        return _ParsedValue(
            "none",
            "",
            "assumed",
            confidence=0.25,
            assumption="No aggregation operator was explicit in the query.",
        )

    def _measure_column(self, query: str, schema: TableSchema) -> _ParsedValue:
        numeric_columns = [column for column, dtype in schema.columns.items() if dtype in {"int", "float", "number"}]
        by_match = re.search(r"(?P<prefix>.+?)(?:\bby\b|\bper\b|\bfor each\b|\bgrouped by\b)\s+[\w ]+", query, re.IGNORECASE)
        if by_match:
            prefix = query[: by_match.end("prefix")]
            for column in numeric_columns:
                span = _column_span(prefix, column)
                if span:
                    return _ParsedValue(column, span, "explicit", confidence=0.75)
        for column in numeric_columns:
            span = _column_span(query, column)
            if span:
                return _ParsedValue(column, span, "explicit", confidence=0.7)
        if numeric_columns:
            return _ParsedValue(
                numeric_columns[0],
                "",
                "assumed",
                confidence=0.25,
                assumption=f"Selected '{numeric_columns[0]}' because it is the first numeric column in the schema.",
            )
        return _ParsedValue(
            None,
            "",
            "unsupported",
            confidence=0.0,
            assumption="No numeric measure column is available in the schema.",
        )

    def _dimension_columns(self, query: str, schema: TableSchema, exclude: str | None) -> _ParsedValue:
        by_match = re.search(r"(?:\bby\b|\bper\b|\bfor each\b|\bgrouped by\b)\s+([\w ]+)", query, re.IGNORECASE)
        if by_match:
            phrase = by_match.group(1)
            matches: list[tuple[int, str, str]] = []
            for column in schema.columns:
                if column == exclude:
                    continue
                column_match = _column_match(phrase, column)
                if column_match is not None:
                    matches.append((column_match.start(), column, phrase[column_match.start() : column_match.end()]))
            if matches:
                matches.sort(key=lambda item: item[0])
                return _ParsedValue(
                    tuple(column for _, column, _ in matches),
                    ", ".join(span for _, _, span in matches),
                    "explicit",
                    confidence=0.75,
                )
        categorical = [
            column
            for column, dtype in schema.columns.items()
            if dtype not in {"int", "float", "number"} and column != exclude
        ]
        if categorical:
            return _ParsedValue(
                (categorical[0],),
                "",
                "assumed",
                confidence=0.25,
                assumption=f"Selected '{categorical[0]}' because it is the first categorical column in the schema.",
            )
        return _ParsedValue((), "", "unsupported", confidence=0.0, assumption="No dimension column is available.")

    def _sort(self, query: str) -> _ParsedValue:
        patterns: tuple[tuple[SortDirection, tuple[str, ...]], ...] = (
            ("asc", (r"\bascending\b", r"\blow to high\b", r"\basc\b")),
            ("desc", (r"\bdescending\b", r"\bhigh to low\b", r"\bdesc\b")),
        )
        for direction, regexes in patterns:
            span = _first_span(query, regexes)
            if span:
                return _ParsedValue(SortSpec(by="measure", direction=direction), span, "explicit", confidence=0.8)
        return _ParsedValue(None, "", "unsupported", confidence=0.0)

    def _limit(self, query: str) -> tuple[_ParsedValue, SortDirection | None]:
        patterns: tuple[tuple[str, SortDirection | None, str], ...] = (
            (r"\btop\s+(?P<limit>\d+)\b", "desc", "top"),
            (r"\bbottom\s+(?P<limit>\d+)\b", "asc", "bottom"),
            (r"\bfirst\s+(?P<limit>\d+)\b", None, "first"),
            (r"\blast\s+(?P<limit>\d+)\b", None, "last"),
        )
        for pattern, implied_sort, _ in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return (
                    _ParsedValue(int(match.group("limit")), query[match.start() : match.end()], "explicit", confidence=0.8),
                    implied_sort,
                )
        return _ParsedValue(None, "", "unsupported", confidence=0.0), None

    def _filters(self, query: str, schema: TableSchema) -> tuple[_FilterExtraction, ...]:
        filters: list[_FilterExtraction] = []
        for column in schema.columns:
            column_regex = rf"(?:{_column_regex(column)})"
            comparison_patterns: tuple[tuple[FilterOp, str], ...] = (
                ("gte", rf"\b{column_regex}\s+(?:at least|greater than or equal to)\s+(?P<value>-?\d+(?:\.\d+)?)"),
                ("lte", rf"\b{column_regex}\s+(?:at most|less than or equal to)\s+(?P<value>-?\d+(?:\.\d+)?)"),
                ("gt", rf"\b{column_regex}\s+(?:greater than|more than|above|over)\s+(?P<value>-?\d+(?:\.\d+)?)"),
                ("lt", rf"\b{column_regex}\s+(?:less than|below|under)\s+(?P<value>-?\d+(?:\.\d+)?)"),
                ("eq", rf"\b{column_regex}\s*(?:=|==|equals|is)\s*['\"]?(?P<value>[\w .-]+?)['\"]?(?=,|\.|;|\band\b|\bby\b|$)"),
            )
            for op, pattern in comparison_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if not match:
                    continue
                value = _coerce_filter_value(match.group("value").strip())
                filters.append(
                    _FilterExtraction(
                        spec=FilterSpec(column=column, op=op, value=value),
                        source_span=query[match.start() : match.end()],
                        confidence=0.7,
                    )
                )
                break
        return tuple(filters)


class LLMIntentParser:
    """Schema-aware parser backed by an OpenAI-compatible JSON-returning LLM."""

    def __init__(self, client: LLMClient, default_confidence: float = 0.7) -> None:
        self.client = client
        self.default_confidence = default_confidence

    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        return self.parse_requirements(query, schema).plan

    def parse_requirements(self, query: str, schema: TableSchema) -> ParsedRequirementBundle:
        payload = self.client.complete_json(
            system_prompt=_intent_parser_system_prompt(),
            user_prompt=_intent_parser_user_prompt(query, schema),
            temperature=0.0,
        )
        return _build_llm_requirement_bundle(query, schema, payload, default_confidence=self.default_confidence)


def _intent_parser_system_prompt() -> str:
    return (
        "You are the requirement extractor for an evidence-grounded chart generation framework. "
        "Convert the natural-language request into atomic chart requirements. "
        "Only use columns that exist in the provided schema. "
        "Do not infer filters unless they are explicit. "
        "If the request is an instruction-to-figure task with no table schema, do not force table fields; "
        "extract figure, layout, artist, axis, data-generation, color, style, legend, annotation, and projection requirements instead. "
        "The requirements array is the source of truth; top-level fields are only a compact executable summary. "
        "Return JSON with keys: chart_type, dimensions, measure_column, aggregation, filters, sort, limit, confidence, requirements. "
        "Each requirements item must include scope, type, name, value, source_span, status, confidence, and optional assumption, depends_on, priority, panel_id. "
        "Use source_span only for text that appears verbatim in the user query. "
        "Allowed status values are explicit, inferred, assumed, ambiguous, unsupported. "
        "Mark ambiguous user language as ambiguous instead of silently choosing a ground-truth value. "
        "Use null when a field is not justified."
    )


def _intent_parser_user_prompt(query: str, schema: TableSchema) -> str:
    columns = [{"name": name, "dtype": dtype} for name, dtype in schema.columns.items()]
    return (
        "Query:\n"
        f"{query}\n\n"
        "Schema:\n"
        f"{columns}\n\n"
        "Return JSON only."
    )


def _build_llm_requirement_bundle(
    query: str,
    schema: TableSchema,
    payload: dict[str, Any],
    *,
    default_confidence: float,
) -> ParsedRequirementBundle:
    confidence = _normalize_confidence(payload.get("confidence"), fallback=default_confidence)
    raw_requirements = payload.get("requirements")
    requirements = _normalize_llm_requirement_nodes(raw_requirements, query, schema, default_confidence=confidence)
    requirements = _postprocess_requirement_nodes(requirements, query, schema, payload)
    fields = _fields_from_requirements(requirements)
    fallback_plan = _plan_from_top_level_payload(query, schema, payload, confidence=confidence)
    fallback_bundle = _bundle_from_plan_with_fallback_provenance(
        fallback_plan,
        query,
        schema,
        payload,
        default_confidence=confidence,
    )
    fallback_requirements = _select_core_fallback_requirements(
        schema,
        payload,
        requirements,
        fallback_bundle.requirement_plan.requirements,
    )
    requirements = _merge_core_fallback_requirements(requirements, fallback_requirements)
    fields = _fields_from_requirements(requirements)
    plan = _plan_from_requirement_fields(query, fields, requirements, fallback_plan, confidence=confidence)
    requirement_plan = _chart_requirement_plan_from_nodes(plan, requirements)
    return ParsedRequirementBundle(plan=plan, requirement_plan=requirement_plan, raw_response=dict(payload))


def _postprocess_requirement_nodes(
    requirements: tuple[RequirementNode, ...],
    query: str,
    schema: TableSchema,
    payload: dict[str, Any],
) -> tuple[RequirementNode, ...]:
    if schema.columns:
        return requirements
    return _cleanup_schemaless_requirements(requirements, query, payload)


def _cleanup_schemaless_requirements(
    requirements: tuple[RequirementNode, ...],
    query: str,
    payload: dict[str, Any],
) -> tuple[RequirementNode, ...]:
    normalized: list[RequirementNode] = []
    seen_fingerprints: set[tuple[str, str | None, str, str]] = set()
    for requirement in requirements:
        cleaned = _cleanup_schemaless_requirement_node(requirement, query, payload)
        if cleaned is None:
            continue
        fingerprint = (
            cleaned.name,
            cleaned.panel_id,
            cleaned.source_span.lower(),
            repr(cleaned.value),
        )
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        normalized.append(cleaned)
    normalized = _relax_shape_only_artist_requirements(normalized)
    return tuple(_reassign_requirement_ids(normalized))


def _cleanup_schemaless_requirement_node(
    requirement: RequirementNode,
    query: str,
    payload: dict[str, Any],
) -> RequirementNode | None:
    name = _normalize_schemaless_requirement_name(requirement.name)
    status = requirement.status
    assumption = requirement.assumption
    if name in _SCHEMALESS_META_REQUIREMENT_NAMES:
        return None
    if name in _SCHEMALESS_DISPLAY_REQUIREMENT_NAMES and _is_display_only_requirement(requirement, query):
        return None
    if name == "aggregation":
        if status == "assumed" or requirement.value == "none":
            return None
    if name in {"measure_column", "dimensions"} and status in {"ambiguous", "unsupported"}:
        return None
    if name == "artist_type":
        artist_type = _normalize_artist_type(requirement.value)
        if artist_type is None:
            return None
        name = "artist_type"
        value = artist_type
        type_ = "encoding"
    else:
        value = requirement.value
        type_ = requirement.type
    scope, panel_id = _normalize_schemaless_scope(requirement.scope, requirement.panel_id)
    priority = _normalize_schemaless_priority(name, requirement.priority)
    if name == "axis_scale":
        type_ = "presentation_constraint"
    elif name in {"axis_label", "title", "legend"}:
        type_ = "annotation"
    elif name in {"subplot_layout", "figure_size"}:
        type_ = "figure_composition"
    elif name in {"projection", "axis_limit", "layout_type"}:
        type_ = "presentation_constraint"
    extra_assumption = _schemaless_assumption_for_unknown_chart_type(name, payload)
    if extra_assumption is not None:
        assumption = _append_assumption(assumption, extra_assumption)
    return replace(
        requirement,
        scope=scope,
        panel_id=panel_id,
        name=name,
        type=type_,
        value=value,
        priority=priority,
        assumption=assumption,
    )


def _normalize_schemaless_requirement_name(name: str) -> str:
    normalized = _SCHEMALESS_NAME_ALIASES.get(name, name)
    return normalized


def _normalize_schemaless_scope(scope: str, panel_id: str | None) -> tuple[str, str | None]:
    if scope == "figure":
        return "figure", None
    normalized_panel_id = str(panel_id or "panel_0").strip()
    if normalized_panel_id.isdigit():
        normalized_panel_id = f"panel_{normalized_panel_id}"
    return "panel", normalized_panel_id or "panel_0"


def _normalize_schemaless_priority(name: str, current: str) -> str:
    if name in {"artist_type", "subplot_layout", "projection", "axis_scale", "axis_limit", "axis_label", "title", "figure_size", "legend"}:
        return "core"
    if current == "core":
        return current if name in _CORE_EXECUTABLE_NAMES else "secondary"
    return current


def _relax_shape_only_artist_requirements(requirements: list[RequirementNode]) -> list[RequirementNode]:
    relaxed: list[RequirementNode] = []
    for requirement in requirements:
        if requirement.name != "artist_type" or not _is_shape_only_artist_span(requirement.source_span):
            relaxed.append(requirement)
            continue
        relaxed.append(
            replace(
                requirement,
                status="ambiguous",
                assumption=_append_assumption(
                    requirement.assumption,
                    "Shape-only language does not uniquely determine the rendering primitive, so artist_type was downgraded to ambiguous.",
                ),
            )
        )
    return relaxed


def _is_shape_only_artist_span(source_span: str) -> bool:
    normalized = str(source_span or "").strip().lower()
    if not normalized:
        return False
    if not any(term in normalized for term in ("square", "squares", "rectangle", "rectangles", "circle", "circles", "triangle", "triangles")):
        return False
    if any(term in normalized for term in ("scatter", "bar", "line", "pie", "heatmap", "hist", "box", "errorbar", "plot", "graph", "chart")):
        return False
    return True


def _normalize_artist_type(value: object) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "stacked_area": "area",
        "pie_chart": "pie",
        "pie_charts": "pie",
        "scatter_plot": "scatter",
        "bar_plot": "bar",
        "errorbar": "errorbar",
        "semilogy": "line",
        "semilogx": "line",
        "loglog": "line",
        "2d_histogram": "hist2d",
        "waterfall_chart": "waterfall",
        "area_chart": "area",
    }
    normalized = aliases.get(normalized, normalized)
    if not normalized:
        return None
    if normalized in {"bar", "line", "pie", "scatter", "area", "heatmap", "box", "waterfall", "errorbar", "hist2d"}:
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


def _is_display_only_requirement(requirement: RequirementNode, query: str) -> bool:
    source = (requirement.source_span or "").lower()
    if not source:
        return True
    if "display" in source or "show" in source:
        return True
    lowered = query.lower()
    return source in lowered and ("display" in lowered or "show" in lowered)


def _schemaless_assumption_for_unknown_chart_type(name: str, payload: dict[str, Any]) -> str | None:
    if name != "artist_type":
        return None
    top_level_chart_type = str(payload.get("chart_type") or "").strip().lower()
    if top_level_chart_type in {"", "unknown", "multi_panel_figure", "multi_panel_composite", "composite"}:
        return "Schema-less task uses artist_type requirements; no single canonical chart_type was enforced."
    return None


def _reassign_requirement_ids(requirements: list[RequirementNode]) -> tuple[RequirementNode, ...]:
    counters: dict[str, int] = {}
    reassigned: list[RequirementNode] = []
    for requirement in requirements:
        requirement_id = _default_requirement_id(requirement.name, requirement.panel_id, counters)
        reassigned.append(replace(requirement, requirement_id=requirement_id))
    return tuple(reassigned)


def _normalize_llm_requirement_nodes(
    value: object,
    query: str,
    schema: TableSchema,
    *,
    default_confidence: float,
) -> tuple[RequirementNode, ...]:
    if not isinstance(value, list):
        return ()
    normalized: list[RequirementNode] = []
    seen_ids: set[str] = set()
    counters: dict[str, int] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        node = _normalize_llm_requirement_node(
            item,
            query,
            schema,
            default_confidence=default_confidence,
            counters=counters,
        )
        if node.requirement_id in seen_ids:
            node = replace(node, requirement_id=_next_requirement_id(node.name, counters))
        seen_ids.add(node.requirement_id)
        normalized.append(node)
    return tuple(normalized)


def _normalize_llm_requirement_node(
    item: dict[str, Any],
    query: str,
    schema: TableSchema,
    *,
    default_confidence: float,
    counters: dict[str, int],
) -> RequirementNode:
    raw_name = str(item.get("name") or "unknown").strip().lower()
    name = _normalize_requirement_name(raw_name)
    raw_scope = item.get("scope")
    scope = _normalize_scope(raw_scope)
    type_ = _normalize_requirement_type(item.get("type"), name)
    panel_id = _normalize_requirement_panel_id(raw_scope, item.get("panel_id")) if scope == "panel" else None
    requirement_id = str(item.get("requirement_id") or "").strip() or _default_requirement_id(name, panel_id, counters)
    value = _normalize_requirement_value(name, item.get("value"), schema)
    source_span = str(item.get("source_span") or "")
    source_span_is_grounded = _span_is_exact_query_text(query, source_span)
    status = _normalize_status(item.get("status"), fallback="inferred")
    assumption = item.get("assumption")
    assumption_text = str(assumption) if assumption is not None else None
    if source_span and not source_span_is_grounded:
        source_span = ""
        if status == "explicit":
            status = "inferred"
        assumption_text = assumption_text or "LLM-provided source_span was removed because it was not found verbatim in the query."
    if status == "explicit" and not source_span:
        status = "inferred"
        assumption_text = assumption_text or "Requirement was not marked explicit because no verbatim source span was grounded in the query."
    value, status, assumption_text = _validate_requirement_value(name, value, status, assumption_text, schema)
    return RequirementNode(
        requirement_id=requirement_id,
        scope=scope,
        type=type_,
        name=name,
        value=value,
        source_span=source_span,
        status=status,
        confidence=_normalize_confidence(item.get("confidence"), fallback=default_confidence),
        depends_on=_normalize_depends_on(item.get("depends_on")),
        priority=_normalize_priority(item.get("priority"), name),
        panel_id=panel_id,
        assumption=assumption_text,
    )


def _plan_from_top_level_payload(
    query: str,
    schema: TableSchema,
    payload: dict[str, Any],
    *,
    confidence: float,
) -> ChartIntentPlan:
    columns = dict(schema.columns)
    chart_type = _normalize_chart_type(payload.get("chart_type") if payload.get("chart_type") not in (None, "") else "unknown")
    measure_column = payload.get("measure_column")
    if measure_column not in columns:
        measure_column = None
    raw_dimensions = payload.get("dimensions")
    if not isinstance(raw_dimensions, list):
        raw_dimensions = []
    dimensions = tuple(
        column
        for column in raw_dimensions
        if isinstance(column, str) and column in columns and column != measure_column
    )
    aggregation = _normalize_aggregation(payload.get("aggregation"))
    filters = _normalize_filter_extractions(payload.get("filters"), schema)
    return ChartIntentPlan(
        chart_type=chart_type,
        dimensions=dimensions,
        measure=MeasureSpec(column=measure_column, agg=aggregation),
        filters=tuple(filter_extraction.spec for filter_extraction in filters),
        sort=_normalize_sort(payload.get("sort")),
        limit=_normalize_limit(payload.get("limit")),
        raw_query=query,
        confidence=confidence,
    )


def _should_use_core_fallback(schema: TableSchema, payload: dict[str, Any]) -> bool:
    if schema.columns:
        return True
    chart_type = str(payload.get("chart_type") or "").strip().lower()
    if chart_type and chart_type not in {"unknown", "composite", "multi_panel_figure", "multi_panel_composite"}:
        return True
    return False


def _select_core_fallback_requirements(
    schema: TableSchema,
    payload: dict[str, Any],
    requirements: tuple[RequirementNode, ...],
    fallback_nodes: tuple[RequirementNode, ...],
) -> dict[str, RequirementNode]:
    if not _should_use_core_fallback(schema, payload):
        return {}
    available = {requirement.requirement_id: requirement for requirement in fallback_nodes}
    if schema.columns:
        return available
    has_artist_type = any(
        requirement.name == "artist_type" and requirement.status not in {"ambiguous", "unsupported"}
        for requirement in requirements
    )
    chart_type = _normalize_chart_type(payload.get("chart_type") if payload.get("chart_type") not in (None, "") else "unknown")
    if has_artist_type or chart_type == "unknown":
        return {}
    chart_type_requirement = available.get("panel_0.chart_type")
    if chart_type_requirement is None:
        return {}
    return {chart_type_requirement.requirement_id: chart_type_requirement}


def _bundle_from_plan_with_fallback_provenance(
    plan: ChartIntentPlan,
    query: str,
    schema: TableSchema,
    payload: dict[str, Any],
    *,
    default_confidence: float,
) -> ParsedRequirementBundle:
    fields = {
        "chart_type": _fallback_field(
            "chart_type",
            plan.chart_type,
            query,
            default_confidence=default_confidence,
            status="assumed" if payload.get("chart_type") in (None, "") else "inferred",
            source_span=_first_span(query, (rf"\b{re.escape(plan.chart_type)}\s+(?:chart|plot|graph)\b", rf"\b{re.escape(plan.chart_type)}\b")),
        ),
        "aggregation": _fallback_field(
            "aggregation",
            plan.measure.agg,
            query,
            default_confidence=default_confidence,
            status="assumed" if plan.measure.agg == "none" else "inferred",
            source_span=_aggregation_span(query, plan.measure.agg),
            assumption="No aggregation operator was explicit in the query." if plan.measure.agg == "none" else None,
        ),
        "measure_column": _fallback_field(
            "measure_column",
            plan.measure.column,
            query,
            default_confidence=default_confidence,
            status="inferred",
            source_span=_column_span(query, plan.measure.column) if plan.measure.column is not None else "",
        ),
        "dimensions": _fallback_field(
            "dimensions",
            plan.dimensions,
            query,
            default_confidence=default_confidence,
            status="inferred",
            source_span=_joined_column_spans(query, plan.dimensions),
        ),
        "sort": _fallback_field(
            "sort",
            plan.sort,
            query,
            default_confidence=default_confidence,
            status="inferred",
            source_span=_sort_span(query, plan.sort),
        ),
        "limit": _fallback_field(
            "limit",
            plan.limit,
            query,
            default_confidence=default_confidence,
            status="inferred",
            source_span=_limit_span(query, plan.limit),
        ),
    }
    return ParsedRequirementBundle(
        plan=plan,
        requirement_plan=_build_requirement_plan_from_parse(
            plan,
            fields=fields,
            filters=_normalize_filter_extractions(payload.get("filters"), schema),
        ),
    )


def _fallback_field(
    name: str,
    value: Any,
    query: str,
    *,
    default_confidence: float,
    status: RequirementStatus,
    source_span: str = "",
    assumption: str | None = None,
) -> _ParsedValue:
    normalized_status = status
    if source_span and _span_is_exact_query_text(query, source_span):
        normalized_status = "explicit"
    elif normalized_status == "explicit":
        normalized_status = "inferred"
    return _ParsedValue(value, source_span, normalized_status, confidence=default_confidence, assumption=assumption)


def _merge_core_fallback_requirements(
    requirements: tuple[RequirementNode, ...],
    fallback_requirements: dict[str, RequirementNode],
) -> tuple[RequirementNode, ...]:
    existing_by_name = {requirement.name for requirement in requirements if requirement.name in _CORE_EXECUTABLE_NAMES}
    merged = list(requirements)
    for requirement_id in _CORE_FALLBACK_ORDER:
        fallback = fallback_requirements.get(requirement_id)
        if fallback is None or fallback.name in existing_by_name:
            continue
        if fallback.value in (None, (), []):
            continue
        merged.append(fallback)
        existing_by_name.add(fallback.name)
    return tuple(merged)


def _fields_from_requirements(requirements: tuple[RequirementNode, ...]) -> dict[str, RequirementNode]:
    fields: dict[str, RequirementNode] = {}
    for requirement in requirements:
        if requirement.name not in _CORE_EXECUTABLE_NAMES:
            continue
        if requirement.status in {"ambiguous", "unsupported"}:
            continue
        fields.setdefault(requirement.name, requirement)
    return fields


def _plan_from_requirement_fields(
    query: str,
    fields: dict[str, RequirementNode],
    requirements: tuple[RequirementNode, ...],
    fallback_plan: ChartIntentPlan,
    *,
    confidence: float,
) -> ChartIntentPlan:
    blocked_names = {
        requirement.name
        for requirement in requirements
        if requirement.name in _CORE_EXECUTABLE_NAMES and requirement.status in {"ambiguous", "unsupported"}
    }
    chart_type_value = fields.get("chart_type").value if "chart_type" in fields else ("unknown" if "chart_type" in blocked_names else fallback_plan.chart_type)
    chart_type = _normalize_chart_type(chart_type_value)
    aggregation_value = fields.get("aggregation").value if "aggregation" in fields else ("none" if "aggregation" in blocked_names else fallback_plan.measure.agg)
    aggregation = _normalize_aggregation(aggregation_value)
    measure_column = fields.get("measure_column").value if "measure_column" in fields else (None if "measure_column" in blocked_names else fallback_plan.measure.column)
    if not isinstance(measure_column, str):
        measure_column = None
    dimensions_value = fields.get("dimensions").value if "dimensions" in fields else (() if "dimensions" in blocked_names else fallback_plan.dimensions)
    dimensions = tuple(item for item in _as_tuple(dimensions_value) if isinstance(item, str))
    sort = _normalize_sort(fields.get("sort").value if "sort" in fields else None)
    if sort is None and "sort" not in blocked_names:
        sort = fallback_plan.sort
    limit = _normalize_limit(fields.get("limit").value if "limit" in fields else None)
    if limit is None and "limit" not in blocked_names:
        limit = fallback_plan.limit
    has_filter_requirement = any(requirement.name == "filter" for requirement in requirements)
    filters = tuple(
        filter_spec
        for requirement in sorted(requirements, key=lambda req: req.requirement_id)
        if requirement.name == "filter"
        and requirement.status not in {"ambiguous", "unsupported"}
        for filter_spec in (_filter_spec_from_requirement(requirement),)
        if filter_spec is not None
    )
    return ChartIntentPlan(
        chart_type=chart_type,
        dimensions=dimensions,
        measure=MeasureSpec(column=measure_column, agg=aggregation),
        filters=filters if has_filter_requirement else fallback_plan.filters,
        sort=sort,
        limit=limit,
        raw_query=query,
        confidence=confidence,
    )


def _chart_requirement_plan_from_nodes(
    plan: ChartIntentPlan,
    requirements: tuple[RequirementNode, ...],
) -> ChartRequirementPlan:
    panel_ids = sorted({requirement.panel_id or "panel_0" for requirement in requirements if requirement.scope == "panel"}) or ["panel_0"]
    panels = []
    for panel_id in panel_ids:
        panel_requirement_ids = tuple(
            requirement.requirement_id
            for requirement in requirements
            if requirement.scope == "panel" and (requirement.panel_id or "panel_0") == panel_id
        )
        panels.append(
            PanelRequirementPlan(
                panel_id=panel_id,
                chart_type=plan.chart_type,
                requirement_ids=panel_requirement_ids,
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
        )
    shared_ids = tuple(requirement.requirement_id for requirement in requirements if requirement.scope == "shared")
    figure_requirements = {
        requirement.name: requirement.value
        for requirement in requirements
        if requirement.scope == "figure"
    }
    return ChartRequirementPlan(
        requirements=requirements,
        panels=tuple(panels),
        figure_requirements=figure_requirements,
        shared_requirement_ids=shared_ids,
        raw_query=plan.raw_query,
    )


def _build_requirement_plan_from_parse(
    plan: ChartIntentPlan,
    *,
    fields: dict[str, _ParsedValue],
    filters: tuple[_FilterExtraction, ...],
) -> ChartRequirementPlan:
    requirements: list[RequirementNode] = []
    panel_id = "panel_0"
    panel_requirement_ids: list[str] = []

    def add_requirement(
        requirement_id: str,
        type_: str,
        name: str,
        value: Any,
        field: _ParsedValue,
        *,
        depends_on: tuple[str, ...] = (),
        priority: str = "core",
    ) -> None:
        requirements.append(
            RequirementNode(
                requirement_id=requirement_id,
                scope="panel",
                type=type_,
                name=name,
                value=value,
                source_span=field.source_span,
                status=field.status,
                confidence=field.confidence if field.confidence is not None else plan.confidence,
                depends_on=depends_on,
                priority=priority,
                panel_id=panel_id,
                assumption=field.assumption,
            )
        )
        panel_requirement_ids.append(requirement_id)

    chart_type_field = fields["chart_type"]
    aggregation_field = fields["aggregation"]
    add_requirement("panel_0.chart_type", "encoding", "chart_type", plan.chart_type, chart_type_field)
    aggregation_depends_on = ("panel_0.measure_column",) if plan.measure.column is not None else ()
    add_requirement(
        "panel_0.aggregation",
        "data_operation",
        "aggregation",
        plan.measure.agg,
        aggregation_field,
        depends_on=aggregation_depends_on,
    )
    if plan.measure.column is not None:
        add_requirement(
            "panel_0.measure_column",
            "data_operation",
            "measure_column",
            plan.measure.column,
            fields["measure_column"],
        )
    if plan.dimensions:
        add_requirement(
            "panel_0.dimensions",
            "data_operation",
            "dimensions",
            tuple(plan.dimensions),
            fields["dimensions"],
        )
    if plan.sort is not None:
        add_requirement(
            "panel_0.sort",
            "presentation_constraint",
            "sort",
            {"by": plan.sort.by, "direction": plan.sort.direction},
            fields["sort"],
            depends_on=("panel_0.aggregation",),
        )
    if plan.limit is not None:
        add_requirement(
            "panel_0.limit",
            "presentation_constraint",
            "limit",
            plan.limit,
            fields["limit"],
            depends_on=("panel_0.sort",) if plan.sort is not None else ("panel_0.aggregation",),
        )
    for index, filter_extraction in enumerate(filters):
        add_requirement(
            f"panel_0.filter_{index}",
            "data_operation",
            "filter",
            {
                "column": filter_extraction.spec.column,
                "op": filter_extraction.spec.op,
                "value": filter_extraction.spec.value,
            },
            _ParsedValue(
                filter_extraction.spec,
                filter_extraction.source_span,
                filter_extraction.status,
                confidence=filter_extraction.confidence,
                assumption=filter_extraction.assumption,
            ),
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
    return ChartRequirementPlan(requirements=tuple(requirements), panels=(panel,), raw_query=plan.raw_query)


def _requirement_metadata(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            metadata[name] = item
    return metadata


def _normalize_requirement_name(value: str) -> str:
    aliases = {
        "chart": "chart_type",
        "chart type": "chart_type",
        "measure": "measure_column",
        "metric": "measure_column",
        "x": "dimensions",
        "x_axis": "dimensions",
        "group_by": "dimensions",
        "groupby": "dimensions",
        "dimension": "dimensions",
        "y": "measure_column",
        "y_axis": "measure_column",
        "agg": "aggregation",
        "top_k": "limit",
        "topk": "limit",
    }
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    return aliases.get(normalized, normalized)


def _normalize_scope(value: object) -> str:
    normalized = str(value or "panel").strip().lower()
    return normalized if normalized in {"figure", "panel", "shared"} else "panel"


def _normalize_requirement_panel_id(scope: object, panel_id: object) -> str:
    explicit = str(panel_id or "").strip().lower()
    if explicit:
        return explicit if not explicit.isdigit() else f"panel_{explicit}"
    normalized_scope = str(scope or "").strip().lower()
    if re.fullmatch(r"panel_\d+", normalized_scope):
        return normalized_scope
    return "panel_0"


def _normalize_requirement_type(value: object, name: str) -> str:
    normalized = str(value or "").strip().lower()
    allowed = {"data_operation", "encoding", "annotation", "presentation_constraint", "figure_composition"}
    if normalized in allowed:
        return normalized
    if name in {"chart_type", "artist_types", "artist_counts", "min_artist_counts"}:
        return "encoding"
    if name in {"sort", "limit", "xscale", "yscale", "zscale", "projection", "size_inches"}:
        return "presentation_constraint"
    if name in {"title", "figure_title", "xlabel", "ylabel", "zlabel", "legend_labels", "text_contains"}:
        return "annotation"
    if name in {"axes_count", "bounds"}:
        return "figure_composition"
    return "data_operation"


def _normalize_requirement_value(name: str, value: object, schema: TableSchema) -> Any:
    if name == "chart_type":
        return _normalize_chart_type(value)
    if name == "aggregation":
        return _normalize_aggregation(value)
    if name == "measure_column":
        return str(value) if isinstance(value, str) else None
    if name == "dimensions":
        return tuple(item for item in _as_tuple(value) if isinstance(item, str))
    if name == "sort":
        return _normalize_sort(value)
    if name == "limit":
        return _normalize_limit(value)
    if name == "filter":
        if isinstance(value, dict):
            column = value.get("column")
            op = str(value.get("op") or "").strip().lower()
            if isinstance(column, str) and column in schema.columns and op in {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}:
                return {"column": column, "op": op, "value": value.get("value")}
        return value
    return value


def _validate_requirement_value(
    name: str,
    value: Any,
    status: RequirementStatus,
    assumption: str | None,
    schema: TableSchema,
) -> tuple[Any, RequirementStatus, str | None]:
    if status in {"ambiguous", "unsupported"}:
        return value, status, assumption
    if name == "measure_column":
        if value is None:
            return value, "ambiguous", assumption or "No measure column was selected."
        if value not in schema.columns:
            return None, "unsupported", _append_assumption(assumption, f"Column '{value}' is not present in the schema.")
    if name == "dimensions":
        valid_dimensions = tuple(column for column in _as_tuple(value) if isinstance(column, str) and column in schema.columns)
        if valid_dimensions != tuple(_as_tuple(value)):
            assumption = _append_assumption(assumption, "One or more dimension columns were removed because they are not present in the schema.")
        value = valid_dimensions
        if not value:
            return value, "ambiguous", assumption or "No valid dimension column was selected."
    if name == "filter":
        filter_spec = _filter_spec_from_value(value)
        if filter_spec is None:
            return value, "unsupported", assumption or "Filter requirement could not be normalized."
        if filter_spec.column not in schema.columns:
            return value, "unsupported", _append_assumption(assumption, f"Filter column '{filter_spec.column}' is not present in the schema.")
        value = {"column": filter_spec.column, "op": filter_spec.op, "value": filter_spec.value}
    if name == "sort" and value is None:
        return value, "unsupported", assumption or "Sort requirement could not be normalized."
    if name == "limit" and value is None:
        return value, "unsupported", assumption or "Limit requirement could not be normalized."
    return value, status, assumption


def _append_assumption(current: str | None, addition: str) -> str:
    if not current:
        return addition
    if addition in current:
        return current
    return f"{current} {addition}"


def _normalize_depends_on(value: object) -> tuple[str, ...]:
    return tuple(item for item in _as_tuple(value) if isinstance(item, str))


def _normalize_priority(value: object, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"core", "secondary"}:
        return normalized
    return "core" if name in _CORE_EXECUTABLE_NAMES else "secondary"


def _default_requirement_id(name: str, panel_id: str | None, counters: dict[str, int]) -> str:
    if name in _CORE_EXECUTABLE_NAMES and name != "filter":
        return f"{panel_id or 'panel_0'}.{name}"
    return _next_requirement_id(name, counters, panel_id=panel_id)


def _next_requirement_id(name: str, counters: dict[str, int], *, panel_id: str | None = "panel_0") -> str:
    count = counters.get(name, 0)
    counters[name] = count + 1
    prefix = panel_id or "shared"
    return f"{prefix}.{name}_{count}"


def _as_tuple(value: object) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _filter_spec_from_requirement(requirement: RequirementNode) -> FilterSpec | None:
    return _filter_spec_from_value(requirement.value)


def _filter_spec_from_value(value: object) -> FilterSpec | None:
    if isinstance(value, FilterSpec):
        return value
    if not isinstance(value, dict):
        return None
    column = value.get("column")
    op = str(value.get("op") or "").strip().lower()
    if not isinstance(column, str) or op not in {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}:
        return None
    return FilterSpec(column=column, op=op, value=value.get("value"))


def _llm_field(
    name: str,
    value: Any,
    query: str,
    metadata: dict[str, dict[str, Any]],
    *,
    default_confidence: float,
    default_status: RequirementStatus,
    fallback_span: str = "",
    fallback_assumption: str | None = None,
) -> _ParsedValue:
    item = metadata.get(name, {})
    source_span = str(item.get("source_span") or fallback_span or "")
    status = _normalize_status(item.get("status"), fallback=default_status if source_span else default_status)
    confidence = _normalize_confidence(item.get("confidence"), fallback=default_confidence)
    assumption = item.get("assumption")
    if assumption is not None:
        assumption = str(assumption)
    if source_span and status in {"assumed", "inferred"} and _span_is_exact_query_text(query, source_span):
        status = "explicit"
    return _ParsedValue(value, source_span, status, confidence=confidence, assumption=assumption or fallback_assumption)


def _normalize_chart_type(value: object) -> ChartType:
    normalized = str(value or "unknown").strip().lower()
    allowed: tuple[ChartType, ...] = ("bar", "line", "pie", "scatter", "area", "heatmap", "box", "unknown")
    return normalized if normalized in allowed else "unknown"


def _normalize_aggregation(value: object) -> str:
    normalized = str(value or "none").strip().lower()
    return normalized if normalized in {"none", "count", "sum", "mean", "min", "max"} else "none"


def _normalize_sort(value: object) -> SortSpec | None:
    if not isinstance(value, dict):
        return None
    by = str(value.get("by", "") or "").strip().lower()
    direction = str(value.get("direction", "") or "").strip().lower()
    if by not in {"dimension", "measure"} or direction not in {"asc", "desc"}:
        return None
    return SortSpec(by=by, direction=direction)


def _normalize_filter_extractions(value: object, schema: TableSchema) -> tuple[_FilterExtraction, ...]:
    if not isinstance(value, list):
        return ()
    filters: list[_FilterExtraction] = []
    allowed_ops = {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}
    for item in value:
        if not isinstance(item, dict):
            continue
        column = item.get("column")
        op = str(item.get("op") or "").strip().lower()
        if not isinstance(column, str) or column not in schema.columns or op not in allowed_ops:
            continue
        status = _normalize_status(item.get("status"), fallback="explicit")
        filters.append(
            _FilterExtraction(
                spec=FilterSpec(column=column, op=op, value=item.get("value")),
                source_span=str(item.get("source_span") or ""),
                status=status,
                confidence=_normalize_confidence(item.get("confidence"), fallback=0.7),
                assumption=str(item.get("assumption")) if item.get("assumption") is not None else None,
            )
        )
    return tuple(filters)


def _normalize_status(value: object, fallback: RequirementStatus) -> RequirementStatus:
    normalized = str(value or fallback).strip().lower()
    allowed: tuple[RequirementStatus, ...] = ("explicit", "inferred", "assumed", "ambiguous", "unsupported")
    return normalized if normalized in allowed else fallback


def _normalize_confidence(value: object, fallback: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, confidence))


def _normalize_limit(value: object) -> int | None:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _first_span(query: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return query[match.start() : match.end()]
    return ""


def _column_span(query: str, column: str | None) -> str:
    if column is None:
        return ""
    match = _column_match(query, column)
    if match is None:
        return ""
    return query[match.start() : match.end()]


def _column_match(text: str, column: str) -> re.Match[str] | None:
    return re.search(rf"\b(?:{_column_regex(column)})\b", text, re.IGNORECASE)


def _column_regex(column: str) -> str:
    variants = {re.escape(column), re.escape(column.replace("_", " "))}
    return "|".join(sorted(variants, key=len, reverse=True))


def _joined_column_spans(query: str, columns: tuple[str, ...]) -> str:
    spans = [_column_span(query, column) for column in columns]
    spans = [span for span in spans if span]
    return ", ".join(spans)


def _aggregation_span(query: str, aggregation: str) -> str:
    patterns = {
        "count": (r"\bcount\b", r"\bnumber of\b", r"\bhow many\b"),
        "mean": (r"\bavg\b", r"\baverage\b", r"\bmean\b"),
        "max": (r"\bmax\b", r"\bmaximum\b", r"\bhighest\b"),
        "min": (r"\bmin\b", r"\bminimum\b", r"\blowest\b"),
        "sum": (r"\bsum\b", r"\btotal\b"),
    }
    return _first_span(query, patterns.get(aggregation, ()))


def _sort_span(query: str, sort: SortSpec | None) -> str:
    if sort is None:
        return ""
    if sort.direction == "asc":
        return _first_span(query, (r"\bascending\b", r"\blow to high\b", r"\basc\b", r"\bbottom\s+\d+\b"))
    return _first_span(query, (r"\bdescending\b", r"\bhigh to low\b", r"\bdesc\b", r"\btop\s+\d+\b"))


def _limit_span(query: str, limit: int | None) -> str:
    if limit is None:
        return ""
    return _first_span(query, (rf"\btop\s+{limit}\b", rf"\bbottom\s+{limit}\b", rf"\bfirst\s+{limit}\b", rf"\blast\s+{limit}\b"))


def _span_is_exact_query_text(query: str, span: str) -> bool:
    return bool(span) and span.lower() in query.lower()


def _coerce_filter_value(value: str) -> Any:
    cleaned = value.strip().strip("'\"")
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned
