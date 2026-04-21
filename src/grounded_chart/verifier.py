from __future__ import annotations

from math import isclose
from typing import Any

from grounded_chart.schema import AxisTrace, FigureRequirementSpec, FigureTrace, PlotTrace, VerificationError, VerificationReport


class OperatorLevelVerifier:
    """Compare expected and actual plotted data at operator-level granularity."""

    def __init__(self, numeric_tolerance: float = 1e-6) -> None:
        self.numeric_tolerance = numeric_tolerance

    def verify(
        self,
        expected: PlotTrace,
        actual: PlotTrace,
        expected_figure: FigureRequirementSpec | None = None,
        actual_figure: FigureTrace | None = None,
        verify_data: bool = True,
        enforce_order: bool = True,
    ) -> VerificationReport:
        errors: list[VerificationError] = []
        if verify_data:
            if expected.chart_type != actual.chart_type:
                errors.append(
                    VerificationError(
                        code="wrong_chart_type",
                        operator="chart_type",
                        message="Actual chart type does not match expected chart type.",
                        expected=expected.chart_type,
                        actual=actual.chart_type,
                    )
                )

            if len(expected.points) != len(actual.points):
                operator = "groupby" if len(actual.points) > len(expected.points) else "filter"
                code = "length_mismatch_extra_points" if len(actual.points) > len(expected.points) else "length_mismatch_missing_points"
                errors.append(
                    VerificationError(
                        code=code,
                        operator=operator,
                        message="Actual plotted data has a different number of points than expected.",
                        expected=len(expected.points),
                        actual=len(actual.points),
                    )
                )

            expected_by_x = {self._normalize_key(point.x): point.y for point in expected.points}
            actual_by_x = {self._normalize_key(point.x): point.y for point in actual.points}

            missing = [point.x for point in expected.points if self._normalize_key(point.x) not in actual_by_x]
            extra = [point.x for point in actual.points if self._normalize_key(point.x) not in expected_by_x]
            if missing:
                errors.append(
                    VerificationError(
                        code="data_point_not_found",
                        operator="filter_or_join",
                        message="Expected data points are missing from the actual plot.",
                        expected=missing,
                        actual=list(actual_by_x.keys()),
                    )
                )
            if extra:
                errors.append(
                    VerificationError(
                        code="unexpected_data_point",
                        operator="filter_or_join",
                        message="Actual plot contains data points that are not expected.",
                        expected=list(expected_by_x.keys()),
                        actual=extra,
                    )
                )

            for key, expected_y in expected_by_x.items():
                if key not in actual_by_x:
                    continue
                actual_y = actual_by_x[key]
                if not self._same_value(expected_y, actual_y):
                    errors.append(
                        VerificationError(
                            code="wrong_aggregation_value",
                            operator="aggregation",
                            message="A plotted value does not match the expected aggregated value.",
                            expected={key: expected_y},
                            actual={key: actual_y},
                        )
                    )

            if enforce_order and self._same_point_set(expected, actual) and not self._same_order(expected, actual):
                errors.append(
                    VerificationError(
                        code="wrong_order",
                        operator="sort",
                        message="Actual plot contains the right data points but in the wrong order.",
                        expected=[point.x for point in expected.points],
                        actual=[point.x for point in actual.points],
                        severity="warning",
                    )
                )

        if expected_figure is not None:
            errors.extend(self._verify_figure(expected_figure, actual_figure))

        return VerificationReport(
            ok=not errors,
            errors=tuple(errors),
            expected_trace=expected,
            actual_trace=actual,
            expected_figure=expected_figure,
            actual_figure=actual_figure,
        )

    def _same_point_set(self, expected: PlotTrace, actual: PlotTrace) -> bool:
        return {self._normalize_key(point.x) for point in expected.points} == {self._normalize_key(point.x) for point in actual.points}

    def _same_order(self, expected: PlotTrace, actual: PlotTrace) -> bool:
        return [self._normalize_key(point.x) for point in expected.points] == [self._normalize_key(point.x) for point in actual.points]

    def _same_value(self, expected: Any, actual: Any) -> bool:
        expected_num = self._as_number(expected)
        actual_num = self._as_number(actual)
        if expected_num is not None and actual_num is not None:
            return isclose(expected_num, actual_num, rel_tol=self.numeric_tolerance, abs_tol=self.numeric_tolerance)
        return expected == actual

    def _normalize_key(self, value: Any) -> str:
        if isinstance(value, tuple):
            return "|".join(str(item).strip().lower() for item in value)
        return str(value).strip().lower()

    def _as_number(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _verify_figure(self, expected: FigureRequirementSpec, actual: FigureTrace | None) -> list[VerificationError]:
        errors: list[VerificationError] = []
        if actual is None:
            return [
                VerificationError(
                    code="missing_figure_trace",
                    operator="figure_trace",
                    message="Figure-level requirements were provided, but no figure trace was captured.",
                    expected="figure trace",
                    actual=None,
                )
            ]

        if expected.axes_count is not None and expected.axes_count != actual.axes_count:
            errors.append(
                VerificationError(
                    code="wrong_axes_count",
                    operator="figure_composition",
                    message="Actual figure has a different number of axes than required.",
                    expected=expected.axes_count,
                    actual=actual.axes_count,
                )
            )
        if expected.figure_title is not None and not self._same_text(expected.figure_title, actual.title):
            errors.append(
                VerificationError(
                    code="wrong_figure_title",
                    operator="figure_title",
                    message="Actual figure title does not match the required figure title.",
                    expected=expected.figure_title,
                    actual=actual.title,
                )
            )
        if expected.size_inches is not None and not self._same_size(expected.size_inches, actual.size_inches):
            errors.append(
                VerificationError(
                    code="wrong_figure_size",
                    operator="figure_size",
                    message="Actual figure size does not match the required size.",
                    expected=expected.size_inches,
                    actual=actual.size_inches,
                    severity="warning",
                )
            )

        axes_by_index = {axis.index: axis for axis in actual.axes}
        for axis_requirement in expected.axes:
            axis = axes_by_index.get(axis_requirement.axis_index)
            if axis is None:
                errors.append(
                    VerificationError(
                        code="missing_axis",
                        operator="figure_composition",
                        message="A required axis was not found.",
                        expected=axis_requirement.axis_index,
                        actual=tuple(axes_by_index),
                    )
                )
                continue
            errors.extend(self._verify_axis(axis_requirement, axis))
        return errors

    def _verify_axis(self, expected: Any, actual: AxisTrace) -> list[VerificationError]:
        errors: list[VerificationError] = []
        text_fields = (
            ("title", "wrong_axis_title", "axis_title"),
            ("xlabel", "wrong_x_label", "axis_label"),
            ("ylabel", "wrong_y_label", "axis_label"),
            ("zlabel", "wrong_z_label", "axis_label"),
            ("projection", "wrong_projection", "projection"),
            ("xscale", "wrong_x_scale", "axis_scale"),
            ("yscale", "wrong_y_scale", "axis_scale"),
            ("zscale", "wrong_z_scale", "axis_scale"),
        )
        for field_name, code, operator in text_fields:
            expected_value = getattr(expected, field_name)
            if expected_value is not None and not self._same_text(expected_value, getattr(actual, field_name)):
                errors.append(
                    VerificationError(
                        code=code,
                        operator=operator,
                        message=f"Actual axis {field_name} does not match the required value.",
                        expected=expected_value,
                        actual=getattr(actual, field_name),
                    )
                )

        for label in expected.legend_labels:
            if not self._contains_text(label, actual.legend_labels):
                errors.append(
                    VerificationError(
                        code="missing_legend_label",
                        operator="legend",
                        message="A required legend label is missing.",
                        expected=label,
                        actual=list(actual.legend_labels),
                    )
                )

        tick_fields = (
            ("xtick_labels", "wrong_x_tick_labels", "ticks"),
            ("ytick_labels", "wrong_y_tick_labels", "ticks"),
            ("ztick_labels", "wrong_z_tick_labels", "ticks"),
        )
        for field_name, code, operator in tick_fields:
            expected_ticks = tuple(getattr(expected, field_name))
            if expected_ticks and not self._same_tick_labels(expected_ticks, getattr(actual, field_name)):
                errors.append(
                    VerificationError(
                        code=code,
                        operator=operator,
                        message=f"Actual axis {field_name} do not match the required tick labels.",
                        expected=list(expected_ticks),
                        actual=list(getattr(actual, field_name)),
                    )
                )

        if expected.bounds is not None and not self._same_bounds(expected.bounds, actual.bounds):
            errors.append(
                VerificationError(
                    code="wrong_axis_layout",
                    operator="figure_layout",
                    message="Actual axis position/layout does not match the required bounds.",
                    expected=list(expected.bounds),
                    actual=list(actual.bounds) if actual.bounds is not None else None,
                )
            )

        actual_artist_types = tuple(artist.artist_type for artist in actual.artists)
        for artist_type in expected.artist_types:
            if artist_type not in actual_artist_types:
                errors.append(
                    VerificationError(
                        code="missing_artist_type",
                        operator="artist",
                        message="A required visual artist type is missing.",
                        expected=artist_type,
                        actual=list(actual_artist_types),
                    )
                )

        actual_artist_counts: dict[str, int] = {}
        for artist in actual.artists:
            count = artist.count if artist.count is not None else 1
            actual_artist_counts[artist.artist_type] = actual_artist_counts.get(artist.artist_type, 0) + int(count)
        for artist_type, expected_count in expected.artist_counts.items():
            actual_count = actual_artist_counts.get(artist_type, 0)
            if actual_count != expected_count:
                errors.append(
                    VerificationError(
                        code="wrong_artist_count",
                        operator="artist",
                        message="An axis has a different number of visual artists than required.",
                        expected={artist_type: expected_count},
                        actual={artist_type: actual_count},
                    )
                )
        for artist_type, minimum in expected.min_artist_counts.items():
            actual_count = actual_artist_counts.get(artist_type, 0)
            if actual_count < minimum:
                errors.append(
                    VerificationError(
                        code="insufficient_artist_count",
                        operator="artist",
                        message="An axis has fewer visual artists than required.",
                        expected={artist_type: minimum},
                        actual={artist_type: actual_count},
                    )
                )

        for text in expected.text_contains:
            if not self._contains_text(text, actual.texts):
                errors.append(
                    VerificationError(
                        code="missing_annotation_text",
                        operator="annotation",
                        message="A required annotation/text string is missing from the axis.",
                        expected=text,
                        actual=list(actual.texts),
                    )
                )
        return errors

    def _same_text(self, expected: str, actual: str) -> bool:
        return str(expected).strip().lower() == str(actual).strip().lower()

    def _contains_text(self, expected: str, actual_values: tuple[str, ...]) -> bool:
        expected_normalized = str(expected).strip().lower()
        return any(expected_normalized in str(actual).strip().lower() for actual in actual_values)

    def _same_size(self, expected: tuple[float, float], actual: tuple[float, float] | None) -> bool:
        if actual is None:
            return False
        return len(expected) == len(actual) and all(isclose(float(exp), float(act), rel_tol=1e-3, abs_tol=1e-3) for exp, act in zip(expected, actual))

    def _same_tick_labels(self, expected: tuple[str, ...], actual: tuple[str, ...]) -> bool:
        normalized_expected = tuple(item.strip().lower() for item in expected if item.strip())
        normalized_actual = tuple(item.strip().lower() for item in actual if item.strip())
        return normalized_expected == normalized_actual

    def _same_bounds(self, expected: tuple[float, float, float, float], actual: tuple[float, float, float, float] | None) -> bool:
        if actual is None:
            return False
        return all(isclose(float(exp), float(act), rel_tol=5e-2, abs_tol=5e-2) for exp, act in zip(expected, actual))
