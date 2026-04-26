from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from grounded_chart.schema import ChartIntentPlan, DataPoint, FilterSpec, PlotTrace

Row = dict[str, Any]


class CanonicalExecutor:
    """Compute expected plotted data from a ChartIntentPlan.

    The MVP uses an in-memory executor to keep the verifier deterministic and
    dependency-free. A DuckDB-backed implementation can replace this later.
    """

    def execute(self, plan: ChartIntentPlan, rows: Iterable[Row]) -> PlotTrace:
        source_rows = [dict(row) for row in rows]
        filtered = [row for row in source_rows if self._passes_filters(row, plan.filters)]
        grouped_rows = self._group_rows_by_key(filtered, plan)
        grouped = [(key, self._aggregate(group, plan)) for key, group in grouped_rows]
        points = [DataPoint(x=key, y=value) for key, value in grouped]
        sorted_points = self._sort_points(points, plan)
        points = list(sorted_points)
        if plan.limit is not None:
            points = points[: plan.limit]
        return PlotTrace(
            chart_type=plan.chart_type,
            points=tuple(points),
            source="canonical_executor",
            raw={
                "intermediate_artifacts": self._intermediate_artifacts(
                    plan=plan,
                    source_rows=source_rows,
                    filtered_rows=filtered,
                    grouped_rows=grouped_rows,
                    aggregated=grouped,
                    sorted_points=sorted_points,
                    limited_points=points,
                )
            },
        )

    def _passes_filters(self, row: Row, filters: tuple[FilterSpec, ...]) -> bool:
        return all(self._passes_filter(row, filter_spec) for filter_spec in filters)

    def _passes_filter(self, row: Row, filter_spec: FilterSpec) -> bool:
        actual = row.get(filter_spec.column)
        expected = filter_spec.value
        op = filter_spec.op
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "contains":
            return str(expected).lower() in str(actual).lower()
        try:
            actual_num = float(actual)
            expected_num = float(expected)
        except (TypeError, ValueError):
            return False
        if op == "gt":
            return actual_num > expected_num
        if op == "gte":
            return actual_num >= expected_num
        if op == "lt":
            return actual_num < expected_num
        if op == "lte":
            return actual_num <= expected_num
        raise ValueError(f"Unsupported filter op: {op}")

    def _group_rows(self, rows: list[Row], plan: ChartIntentPlan) -> list[tuple[Any, Any]]:
        return [(key, self._aggregate(group, plan)) for key, group in self._group_rows_by_key(rows, plan)]

    def _group_rows_by_key(self, rows: list[Row], plan: ChartIntentPlan) -> list[tuple[Any, list[Row]]]:
        if not plan.dimensions:
            key_fn = lambda row: "__all__"
        elif len(plan.dimensions) == 1:
            key_fn = lambda row: row.get(plan.dimensions[0])
        else:
            key_fn = lambda row: tuple(row.get(dim) for dim in plan.dimensions)

        groups: dict[Any, list[Row]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)

        return list(groups.items())

    def _intermediate_artifacts(
        self,
        *,
        plan: ChartIntentPlan,
        source_rows: list[Row],
        filtered_rows: list[Row],
        grouped_rows: list[tuple[Any, list[Row]]],
        aggregated: list[tuple[Any, Any]],
        sorted_points: list[DataPoint],
        limited_points: list[DataPoint],
    ) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = [
            {
                "artifact_id": "expected.source_rows",
                "stage": "source_rows",
                "requirement_names": ["measure_column", "dimensions"],
                "payload": [_jsonable_row(row) for row in source_rows],
            },
            {
                "artifact_id": "expected.filtered_rows",
                "stage": "filter",
                "requirement_names": ["filter"],
                "payload": [_jsonable_row(row) for row in filtered_rows],
            },
            {
                "artifact_id": "expected.grouped_rows",
                "stage": "groupby",
                "requirement_names": ["dimensions"],
                "payload": [
                    {"key": _jsonable_value(key), "row_count": len(group), "rows": [_jsonable_row(row) for row in group]}
                    for key, group in grouped_rows
                ],
            },
            {
                "artifact_id": "expected.aggregated_table",
                "stage": "aggregation",
                "requirement_names": ["aggregation", "measure_column", "dimensions"],
                "payload": [
                    {"x": _jsonable_value(key), "y": _jsonable_value(value)}
                    for key, value in aggregated
                ],
            },
        ]
        if plan.sort is not None:
            artifacts.append(
                {
                    "artifact_id": "expected.sorted_table",
                    "stage": "sort",
                    "requirement_names": ["sort"],
                    "payload": [_point_to_dict(point) for point in sorted_points],
                }
            )
        if plan.limit is not None:
            artifacts.append(
                {
                    "artifact_id": "expected.limited_table",
                    "stage": "limit",
                    "requirement_names": ["limit"],
                    "payload": [_point_to_dict(point) for point in limited_points],
                }
            )
        artifacts.append(
            {
                "artifact_id": "expected.plot_points",
                "stage": "plot_points",
                "requirement_names": ["chart_type", "aggregation", "measure_column", "dimensions", "filter", "sort", "limit"],
                "payload": [_point_to_dict(point) for point in limited_points],
            }
        )
        return artifacts

    def _aggregate(self, group: list[Row], plan: ChartIntentPlan) -> Any:
        agg = plan.measure.agg
        column = plan.measure.column
        if agg == "count":
            return len(group)
        if column is None:
            if len(group) == 1:
                return 1
            return len(group)
        values = [self._as_number(row.get(column)) for row in group]
        values = [value for value in values if value is not None]
        if not values:
            return None
        if agg in ("none", "sum"):
            if agg == "none" and len(values) == 1:
                return values[0]
            return sum(values)
        if agg == "mean":
            return sum(values) / len(values)
        if agg == "min":
            return min(values)
        if agg == "max":
            return max(values)
        raise ValueError(f"Unsupported aggregation: {agg}")

    def _sort_points(self, points: list[DataPoint], plan: ChartIntentPlan) -> list[DataPoint]:
        if plan.sort is None:
            return points
        reverse = plan.sort.direction == "desc"
        if plan.sort.by == "measure":
            return sorted(points, key=lambda point: self._sort_value(point.y), reverse=reverse)
        return sorted(points, key=lambda point: str(point.x), reverse=reverse)

    def _sort_value(self, value: Any) -> tuple[int, Any]:
        number = self._as_number(value)
        if number is not None:
            return (0, number)
        return (1, str(value))

    def _as_number(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _point_to_dict(point: DataPoint) -> dict[str, Any]:
    return {"x": _jsonable_value(point.x), "y": _jsonable_value(point.y), "meta": _jsonable_value(point.meta)}


def _jsonable_row(row: Row) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in row.items()}


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
