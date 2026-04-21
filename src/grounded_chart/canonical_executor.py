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
        filtered = [row for row in rows if self._passes_filters(row, plan.filters)]
        grouped = self._group_rows(filtered, plan)
        points = [DataPoint(x=key, y=value) for key, value in grouped]
        points = self._sort_points(points, plan)
        if plan.limit is not None:
            points = points[: plan.limit]
        return PlotTrace(chart_type=plan.chart_type, points=tuple(points), source="canonical_executor")

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
        if not plan.dimensions:
            key_fn = lambda row: "__all__"
        elif len(plan.dimensions) == 1:
            key_fn = lambda row: row.get(plan.dimensions[0])
        else:
            key_fn = lambda row: tuple(row.get(dim) for dim in plan.dimensions)

        groups: dict[Any, list[Row]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)

        return [(key, self._aggregate(group, plan)) for key, group in groups.items()]

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
