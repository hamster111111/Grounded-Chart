from __future__ import annotations

import re
from typing import Protocol

from grounded_chart.llm import LLMClient
from grounded_chart.schema import ChartIntentPlan, ChartType, MeasureSpec, SortSpec, TableSchema


class IntentParser(Protocol):
    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        """Parse a natural-language chart request into a ChartIntentPlan."""


class HeuristicIntentParser:
    """Small baseline parser for smoke tests.

    This is not the final research parser. It exists so the framework has a
    runnable local baseline before an LLM parser is connected.
    """

    CHART_TYPES: tuple[ChartType, ...] = ("bar", "line", "pie", "scatter", "area", "heatmap", "box")

    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        lowered = query.lower()
        chart_type = self._chart_type(lowered)
        agg = self._aggregation(lowered)
        measure = self._measure_column(lowered, schema)
        dimensions = self._dimension_columns(lowered, schema, exclude=measure)
        sort = self._sort(lowered)
        return ChartIntentPlan(
            chart_type=chart_type,
            dimensions=dimensions,
            measure=MeasureSpec(column=measure, agg=agg),
            sort=sort,
            raw_query=query,
            confidence=0.35,
        )

    def _chart_type(self, lowered: str) -> ChartType:
        for chart_type in self.CHART_TYPES:
            if chart_type in lowered:
                return chart_type
        return "bar"

    def _aggregation(self, lowered: str) -> str:
        if re.search(r"\b(count|number of|how many)\b", lowered):
            return "count"
        if re.search(r"\b(avg|average|mean)\b", lowered):
            return "mean"
        if re.search(r"\b(max|maximum|highest)\b", lowered):
            return "max"
        if re.search(r"\b(min|minimum|lowest)\b", lowered):
            return "min"
        if re.search(r"\b(sum|total)\b", lowered):
            return "sum"
        return "none"

    def _measure_column(self, lowered: str, schema: TableSchema) -> str | None:
        numeric_columns = [column for column, dtype in schema.columns.items() if dtype in {"int", "float", "number"}]
        by_match = re.search(r"(?P<prefix>.+?)(?:by|per|for each|grouped by)\s+[a-zA-Z0-9_ ]+", lowered)
        if by_match:
            prefix = by_match.group("prefix")
            for column in numeric_columns:
                if column.lower().replace("_", " ") in prefix or column.lower() in prefix:
                    return column
        for column in numeric_columns:
            if column.lower().replace("_", " ") in lowered or column.lower() in lowered:
                return column
        return numeric_columns[0] if numeric_columns else None

    def _dimension_columns(self, lowered: str, schema: TableSchema, exclude: str | None) -> tuple[str, ...]:
        by_match = re.search(r"(?:by|per|for each|grouped by)\s+([a-zA-Z0-9_ ]+)", lowered)
        if by_match:
            phrase = by_match.group(1)
            matches: list[tuple[int, str]] = []
            for column in schema.columns:
                if column == exclude:
                    continue
                variants = {column.lower(), column.lower().replace("_", " ")}
                positions = [phrase.find(variant) for variant in variants if phrase.find(variant) != -1]
                if positions:
                    matches.append((min(positions), column))
            if matches:
                matches.sort(key=lambda item: item[0])
                return tuple(column for _, column in matches)
        categorical = [column for column, dtype in schema.columns.items() if dtype not in {"int", "float", "number"} and column != exclude]
        return (categorical[0],) if categorical else ()

    def _sort(self, lowered: str) -> SortSpec | None:
        if "ascending" in lowered or "low to high" in lowered or "asc" in lowered:
            return SortSpec(by="measure", direction="asc")
        if "descending" in lowered or "high to low" in lowered or "desc" in lowered:
            return SortSpec(by="measure", direction="desc")
        return None


class LLMIntentParser:
    """Schema-aware parser backed by an OpenAI-compatible JSON-returning LLM."""

    def __init__(self, client: LLMClient, default_confidence: float = 0.7) -> None:
        self.client = client
        self.default_confidence = default_confidence

    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        payload = self.client.complete_json(
            system_prompt=_intent_parser_system_prompt(),
            user_prompt=_intent_parser_user_prompt(query, schema),
            temperature=0.0,
        )
        columns = dict(schema.columns)
        chart_type = _normalize_chart_type(payload.get("chart_type"))
        measure_column = payload.get("measure_column")
        if measure_column not in columns:
            measure_column = None
        dimensions = tuple(
            column
            for column in payload.get("dimensions", [])
            if isinstance(column, str) and column in columns and column != measure_column
        )
        aggregation = _normalize_aggregation(payload.get("aggregation"))
        sort = _normalize_sort(payload.get("sort"))
        confidence = _normalize_confidence(payload.get("confidence"), fallback=self.default_confidence)
        return ChartIntentPlan(
            chart_type=chart_type,
            dimensions=dimensions,
            measure=MeasureSpec(column=measure_column, agg=aggregation),
            filters=(),
            sort=sort,
            limit=_normalize_limit(payload.get("limit")),
            raw_query=query,
            confidence=confidence,
        )


def _intent_parser_system_prompt() -> str:
    return (
        "You convert natural-language chart requests into a minimal ChartIntentPlan JSON object. "
        "Only use columns that exist in the provided schema. "
        "Do not infer filters unless they are explicit. "
        "Return JSON with keys: chart_type, dimensions, measure_column, aggregation, sort, limit, confidence. "
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


def _normalize_chart_type(value: object) -> ChartType:
    normalized = str(value or "bar").strip().lower()
    allowed: tuple[ChartType, ...] = ("bar", "line", "pie", "scatter", "area", "heatmap", "box", "unknown")
    return normalized if normalized in allowed else "bar"


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
