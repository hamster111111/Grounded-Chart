from __future__ import annotations

import os
import math
import json
import types
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

from grounded_chart.runtime.code_structure import extract_code_structure_artifacts
from grounded_chart.core.schema import ArtistTrace, AxisTrace, ChartType, DataPoint, FigureTrace, PlotTrace


@dataclass(frozen=True)
class MatplotlibRunTrace:
    plot_trace: PlotTrace
    figure_trace: FigureTrace


class ManualTraceRunner:
    """Build PlotTrace objects from already extracted plotted data.

    The next implementation step is a Matplotlib monkey-patch runner. This
    manual runner keeps the verifier and pipeline testable immediately.
    """

    def from_xy(self, chart_type: ChartType, x_values: Iterable[Any], y_values: Iterable[Any], source: str = "manual") -> PlotTrace:
        points = tuple(DataPoint(x=x, y=y) for x, y in zip(x_values, y_values))
        return PlotTrace(chart_type=chart_type, points=points, source=source)

    def from_points(self, chart_type: ChartType, points: Iterable[tuple[Any, Any]], source: str = "manual") -> PlotTrace:
        normalized = tuple(DataPoint(x=x, y=y) for x, y in points)
        return PlotTrace(chart_type=chart_type, points=normalized, source=source)


class MatplotlibTraceRunner:
    """Execute Matplotlib code and capture the first supported plotted trace.

    This is intentionally a minimal MVP runner. It is not a security sandbox.
    It should be used on benchmark-generated code in a controlled environment.
    """

    def run_code(
        self,
        code: str,
        globals_dict: dict[str, Any] | None = None,
        execution_dir: str | Path | None = None,
        file_path: str | Path | None = None,
    ) -> PlotTrace:
        return self.run_code_with_figure(
            code,
            globals_dict=globals_dict,
            execution_dir=execution_dir,
            file_path=file_path,
        ).plot_trace

    def run_code_with_figure(
        self,
        code: str,
        globals_dict: dict[str, Any] | None = None,
        execution_dir: str | Path | None = None,
        file_path: str | Path | None = None,
    ) -> MatplotlibRunTrace:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.axes
        import matplotlib.figure
        import matplotlib.pyplot as plt

        plt.close("all")
        records: list[PlotTrace] = []
        resolved_execution_dir = Path(execution_dir).resolve() if execution_dir is not None else None
        resolved_file_path = Path(file_path).resolve() if file_path is not None else None
        code_structure_artifacts = extract_code_structure_artifacts(code)
        plotly_state: dict[str, Any] = {
            "backend_used": False,
            "figure": None,
            "trace_types": (),
        }

        original_axes_bar = matplotlib.axes.Axes.bar
        original_axes_barh = matplotlib.axes.Axes.barh
        original_axes_plot = matplotlib.axes.Axes.plot
        original_axes_scatter = matplotlib.axes.Axes.scatter
        original_axes_pie = matplotlib.axes.Axes.pie
        original_figure_savefig = matplotlib.figure.Figure.savefig
        original_pyplot_savefig = plt.savefig
        original_pyplot_show = plt.show
        original_pyplot_close = plt.close

        def record(trace: PlotTrace) -> None:
            records.append(trace)

        def axes_bar(ax: Any, x: Any, height: Any, *args: Any, **kwargs: Any) -> Any:
            record(self._trace_xy("bar", x, height, source="matplotlib.axes.Axes.bar", raw={"kwargs": kwargs}))
            return original_axes_bar(ax, x, height, *args, **kwargs)

        def axes_barh(ax: Any, y: Any, width: Any, *args: Any, **kwargs: Any) -> Any:
            record(self._trace_xy("bar", y, width, source="matplotlib.axes.Axes.barh", raw={"orientation": "horizontal", "kwargs": kwargs}))
            return original_axes_barh(ax, y, width, *args, **kwargs)

        def axes_plot(ax: Any, *args: Any, **kwargs: Any) -> Any:
            x_values, y_values = self._xy_from_plot_args(args)
            record(self._trace_xy("line", x_values, y_values, source="matplotlib.axes.Axes.plot", raw={"kwargs": kwargs}))
            return original_axes_plot(ax, *args, **kwargs)

        def axes_scatter(ax: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
            record(self._trace_xy("scatter", x, y, source="matplotlib.axes.Axes.scatter", raw={"kwargs": kwargs}))
            return original_axes_scatter(ax, x, y, *args, **kwargs)

        def axes_pie(ax: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
            labels = kwargs.get("labels")
            x_values = labels if labels is not None else range(len(self._as_list(x)))
            record(self._trace_xy("pie", x_values, x, source="matplotlib.axes.Axes.pie", raw={"kwargs": kwargs}))
            return original_axes_pie(ax, x, *args, **kwargs)

        def pyplot_bar(x: Any, height: Any, *args: Any, **kwargs: Any) -> Any:
            return axes_bar(plt.gca(), x, height, *args, **kwargs)

        def pyplot_barh(y: Any, width: Any, *args: Any, **kwargs: Any) -> Any:
            return axes_barh(plt.gca(), y, width, *args, **kwargs)

        def pyplot_plot(*args: Any, **kwargs: Any) -> Any:
            return axes_plot(plt.gca(), *args, **kwargs)

        def pyplot_scatter(x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
            return axes_scatter(plt.gca(), x, y, *args, **kwargs)

        def pyplot_pie(x: Any, *args: Any, **kwargs: Any) -> Any:
            return axes_pie(plt.gca(), x, *args, **kwargs)

        def noop_savefig(*args: Any, **kwargs: Any) -> None:
            return None

        def noop_show(*args: Any, **kwargs: Any) -> None:
            return None

        def noop_close(*args: Any, **kwargs: Any) -> None:
            return None

        def capture_plotly_figure(fig: Any) -> Any:
            plotly_state["backend_used"] = True
            plotly_state["figure"] = fig
            try:
                plotly_state["trace_types"] = tuple(self._plotly_trace_descriptor(trace) for trace in getattr(fig, "data", ()))
            except Exception:
                plotly_state["trace_types"] = ()
            return fig

        def plotly_express_wrapper(factory: Any):
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return capture_plotly_figure(factory(*args, **kwargs))

            return wrapped

        def plotly_show_wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
            capture_plotly_figure(self)
            return None

        def plotly_write_image_wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
            capture_plotly_figure(self)
            return None

        exec_globals = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "__file__": str(resolved_file_path) if resolved_file_path is not None else "<grounded_chart_exec>",
            "plt": plt,
        }
        if globals_dict:
            exec_globals.update(globals_dict)
        initial_global_keys = set(exec_globals)

        patchers = [
            patch.object(matplotlib.axes.Axes, "bar", axes_bar),
            patch.object(matplotlib.axes.Axes, "barh", axes_barh),
            patch.object(matplotlib.axes.Axes, "plot", axes_plot),
            patch.object(matplotlib.axes.Axes, "scatter", axes_scatter),
            patch.object(matplotlib.axes.Axes, "pie", axes_pie),
            patch.object(plt, "bar", pyplot_bar),
            patch.object(plt, "barh", pyplot_barh),
            patch.object(plt, "plot", pyplot_plot),
            patch.object(plt, "scatter", pyplot_scatter),
            patch.object(plt, "pie", pyplot_pie),
            patch.object(matplotlib.figure.Figure, "savefig", lambda self, *args, **kwargs: None),
            patch.object(plt, "savefig", noop_savefig),
            patch.object(plt, "show", noop_show),
            patch.object(plt, "close", noop_close),
        ]
        plotly_patchers = self._plotly_patchers(plotly_express_wrapper, plotly_show_wrapper, plotly_write_image_wrapper)
        patchers.extend(plotly_patchers)

        with self._working_directory(resolved_execution_dir), self._patch_all(patchers):
            exec(code, exec_globals)

        if plotly_state["backend_used"] and plotly_state["figure"] is not None:
            figure_trace = self._plotly_figure_trace(plotly_state["figure"], trace_types=tuple(plotly_state["trace_types"]))
            plot_trace = self._plotly_plot_trace(plotly_state["figure"], trace_types=tuple(plotly_state["trace_types"]))
        else:
            figure_trace = self._figure_trace(plt)
            if not records:
                plot_trace = PlotTrace(chart_type="unknown", points=(), source="matplotlib_trace_runner", raw={"trace_error": "no_supported_plot_call"})
            else:
                plot_trace = records[0]
        plot_trace = self._attach_actual_intermediate_artifacts(plot_trace, exec_globals, initial_global_keys)
        figure_trace = self._attach_code_structure_artifacts(figure_trace, code_structure_artifacts)
        return MatplotlibRunTrace(plot_trace=plot_trace, figure_trace=figure_trace)


    def _attach_code_structure_artifacts(self, figure_trace: FigureTrace, artifacts: tuple[dict[str, Any], ...]) -> FigureTrace:
        if not artifacts:
            return figure_trace
        raw = dict(figure_trace.raw)
        raw["code_structure_artifacts"] = [dict(artifact) for artifact in artifacts]
        return replace(figure_trace, raw=raw)

    def _trace_xy(self, chart_type: ChartType, x_values: Any, y_values: Any, source: str, raw: dict[str, Any] | None = None) -> PlotTrace:
        xs = self._as_list(x_values)
        ys = self._as_list(y_values)
        points = tuple(DataPoint(x=x, y=y) for x, y in zip(xs, ys))
        return PlotTrace(chart_type=chart_type, points=points, source=source, raw=raw or {})

    def _xy_from_plot_args(self, args: tuple[Any, ...]) -> tuple[list[Any], list[Any]]:
        if len(args) == 0:
            return [], []
        if len(args) == 1 or (len(args) >= 2 and isinstance(args[1], str)):
            y_values = self._as_list(args[0])
            return list(range(len(y_values))), y_values
        return self._as_list(args[0]), self._as_list(args[1])

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, range):
            return list(value)
        if isinstance(value, (str, bytes)):
            return [value]
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, list):
                return [self._normalize_scalar(item) for item in converted]
            return [self._normalize_scalar(converted)]
        try:
            return [self._normalize_scalar(item) for item in list(value)]
        except TypeError:
            return [self._normalize_scalar(value)]

    def _normalize_scalar(self, value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        return value

    def _attach_actual_intermediate_artifacts(
        self,
        plot_trace: PlotTrace,
        exec_globals: dict[str, Any],
        initial_global_keys: set[str],
    ) -> PlotTrace:
        variables = self._extract_data_variables(exec_globals, initial_global_keys)
        if not variables:
            return plot_trace
        artifacts = self._semantic_data_variable_artifacts(variables)
        artifacts.append(
            {
                "artifact_id": "actual.data_variables",
                "stage": "execution_globals",
                "requirement_names": ["execution_globals"],
                "payload": variables,
            }
        )
        raw = dict(plot_trace.raw)
        raw["actual_intermediate_artifacts"] = artifacts
        return replace(plot_trace, raw=raw)

    def _semantic_data_variable_artifacts(self, variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sequence_candidates: list[dict[str, Any]] = []
        table_candidates: list[dict[str, Any]] = []
        aggregate_candidates: list[dict[str, Any]] = []
        for variable in variables:
            name = str(variable.get("name") or "")
            preview = variable.get("preview")
            if isinstance(preview, list):
                sequence_role = self._sequence_role(preview)
                if sequence_role is not None:
                    sequence_candidates.append(
                        {
                            "name": name,
                            "role": sequence_role,
                            "length": len(preview),
                            "preview": preview,
                        }
                    )
                point_table = self._point_table_from_sequence(preview)
                if point_table:
                    table_candidates.append({"name": name, "points": point_table})
            elif isinstance(preview, dict):
                point_table = self._point_table_from_mapping(preview)
                if point_table:
                    aggregate_candidates.append({"name": name, "points": point_table})

        payload: dict[str, Any] = {}
        if sequence_candidates:
            payload["sequence_candidates"] = sequence_candidates
        if table_candidates:
            payload["point_table_candidates"] = table_candidates
        if aggregate_candidates:
            payload["aggregate_table_candidates"] = aggregate_candidates
        if not payload:
            return []
        artifacts = [
            {
                "artifact_id": "actual.semantic_data_variables",
                "stage": "execution_globals_semantic",
                "requirement_names": ["execution_globals_semantic"],
                "payload": payload,
            }
        ]
        if sequence_candidates:
            artifacts.append(
                {
                    "artifact_id": "actual.sequence_candidates",
                    "stage": "execution_globals_sequences",
                    "requirement_names": ["dimensions", "measure_column"],
                    "payload": sequence_candidates,
                }
            )
        if table_candidates:
            artifacts.append(
                {
                    "artifact_id": "actual.candidate_point_tables",
                    "stage": "execution_globals_point_tables",
                    "requirement_names": ["dimensions", "measure_column", "aggregation"],
                    "payload": table_candidates,
                }
            )
        if aggregate_candidates:
            artifacts.append(
                {
                    "artifact_id": "actual.candidate_aggregate_tables",
                    "stage": "execution_globals_aggregate_tables",
                    "requirement_names": ["dimensions", "measure_column", "aggregation"],
                    "payload": aggregate_candidates,
                }
            )
        return artifacts

    def _sequence_role(self, values: list[Any]) -> str | None:
        if not values:
            return None
        scalar_values = [value for value in values if self._is_scalar_preview(value)]
        if len(scalar_values) != len(values):
            return None
        numeric_count = sum(1 for value in scalar_values if self._is_number_like(value))
        if numeric_count == len(scalar_values):
            return "numeric_sequence"
        if numeric_count == 0:
            return "categorical_sequence"
        return "mixed_sequence"

    def _point_table_from_mapping(self, values: dict[Any, Any]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for key, value in values.items():
            if not self._is_number_like(value):
                return []
            points.append({"x": key, "y": value})
        return points

    def _point_table_from_sequence(self, values: list[Any]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for item in values:
            if isinstance(item, dict):
                if "x" in item and "y" in item and self._is_number_like(item.get("y")):
                    points.append({"x": item.get("x"), "y": item.get("y")})
                    continue
                return []
            if isinstance(item, (list, tuple)) and len(item) >= 2 and self._is_number_like(item[1]):
                points.append({"x": item[0], "y": item[1]})
                continue
            return []
        return points

    def _is_number_like(self, value: Any) -> bool:
        if isinstance(value, bool) or value is None:
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

    def _extract_data_variables(self, exec_globals: dict[str, Any], initial_global_keys: set[str]) -> list[dict[str, Any]]:
        variables: list[dict[str, Any]] = []
        for name, value in sorted(exec_globals.items(), key=lambda item: item[0]):
            if name in initial_global_keys or name.startswith("__"):
                continue
            preview = self._data_variable_preview(value)
            if preview is None:
                continue
            variables.append(
                {
                    "name": name,
                    "type": type(value).__name__,
                    "preview": preview,
                }
            )
        return variables[:30]

    def _data_variable_preview(self, value: Any) -> Any | None:
        if isinstance(value, types.ModuleType) or callable(value):
            return None
        if self._is_scalar_preview(value):
            return value
        if isinstance(value, range):
            return list(value)[:20]
        if isinstance(value, dict):
            return {str(key): self._jsonable_preview(item) for key, item in list(value.items())[:20]}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable_preview(item) for item in list(value)[:20]]
        if hasattr(value, "head") and hasattr(value, "to_dict"):
            try:
                return self._jsonable_preview(value.head(20).to_dict(orient="records"))
            except TypeError:
                try:
                    return self._jsonable_preview(value.head(20).to_dict())
                except Exception:
                    return None
            except Exception:
                return None
        if hasattr(value, "tolist"):
            try:
                converted = value.tolist()
            except Exception:
                return None
            return self._jsonable_preview(converted[:20] if isinstance(converted, list) else converted)
        return None

    def _jsonable_preview(self, value: Any) -> Any:
        if self._is_scalar_preview(value):
            return value
        if isinstance(value, range):
            return list(value)[:20]
        if isinstance(value, dict):
            return {str(key): self._jsonable_preview(item) for key, item in list(value.items())[:20]}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable_preview(item) for item in list(value)[:20]]
        if hasattr(value, "item"):
            try:
                return self._jsonable_preview(value.item())
            except Exception:
                pass
        return str(value)

    def _is_scalar_preview(self, value: Any) -> bool:
        return value is None or isinstance(value, (str, bool, int, float))

    def _figure_trace(self, plt: Any) -> FigureTrace:
        fig = plt.gcf()
        semantic_axes = [ax for ax in fig.axes if not self._is_helper_axis(ax)]
        helper_axes = [ax for ax in fig.axes if self._is_helper_axis(ax)]
        axes = tuple(self._axis_trace(index, ax) for index, ax in enumerate(semantic_axes))
        width, height = fig.get_size_inches()
        title = ""
        if getattr(fig, "_suptitle", None) is not None:
            title = fig._suptitle.get_text()
        figure_lines = tuple(getattr(fig, "lines", ()))
        figure_patches = tuple(getattr(fig, "patches", ()))
        figure_patch_types = [type(patch_item).__name__ for patch_item in figure_patches]
        connection_patch_count = sum(1 for name in figure_patch_types if name in {"ConnectionPatch", "FancyArrowPatch"})
        return FigureTrace(
            title=title,
            size_inches=(round(float(width), 4), round(float(height), 4)),
            axes=axes,
            source="matplotlib_figure",
            raw={
                "total_axes_count": len(fig.axes),
                "helper_axes_count": len(helper_axes),
                "helper_axis_labels": [str(ax.get_label() or "") for ax in helper_axes],
                "figure_line_count": len(figure_lines),
                "figure_line_styles": [
                    {
                        "color": str(line.get_color()) if hasattr(line, "get_color") else None,
                        "linewidth": float(line.get_linewidth()) if hasattr(line, "get_linewidth") else None,
                    }
                    for line in figure_lines
                ],
                "figure_patch_count": len(figure_patches),
                "figure_connection_patch_count": connection_patch_count,
                "figure_patch_types": figure_patch_types,
            },
        )

    def _plotly_plot_trace(self, fig: Any, trace_types: tuple[str, ...]) -> PlotTrace:
        chart_type = self._plotly_chart_type(trace_types)
        points = self._plotly_points(fig, chart_type)
        return PlotTrace(
            chart_type=chart_type,
            points=points,
            source="plotly_figure",
            raw={
                "backend": "plotly",
                "trace_types": list(trace_types),
                "trace_count": len(trace_types),
            },
        )

    def _plotly_figure_trace(self, fig: Any, trace_types: tuple[str, ...]) -> FigureTrace:
        title = ""
        width = None
        height = None
        axis_count = 1
        xlabel = ""
        ylabel = ""
        xscale = "linear"
        yscale = "linear"
        xtick_labels: tuple[str, ...] = ()
        ytick_labels: tuple[str, ...] = ()
        legend_labels: tuple[str, ...] = ()
        texts: tuple[str, ...] = ()
        try:
            layout = getattr(fig, "layout", None)
            layout_title = getattr(layout, "title", None)
            if layout_title is not None:
                title = str(getattr(layout_title, "text", "") or "")
            width = getattr(layout, "width", None)
            height = getattr(layout, "height", None)
            axis_count = self._plotly_axes_count(layout)
            xlabel = self._plotly_axis_title(layout, "xaxis")
            ylabel = self._plotly_axis_title(layout, "yaxis")
            xscale = self._plotly_axis_scale(layout, "xaxis")
            yscale = self._plotly_axis_scale(layout, "yaxis")
            xtick_labels = self._plotly_tick_labels(layout, "xaxis")
            ytick_labels = self._plotly_tick_labels(layout, "yaxis")
            legend_labels = self._plotly_legend_labels(fig)
            texts = self._plotly_visible_texts(fig, layout)
        except Exception:
            layout = None
        artists = [
            ArtistTrace(artist_type=trace_type or "plotly_trace", count=1) for trace_type in trace_types
        ]
        shape_count = self._plotly_shape_count(layout) if layout is not None else 0
        if shape_count:
            artists.append(ArtistTrace(artist_type="shape", count=shape_count))
        axis = AxisTrace(
            index=0,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            projection="plotly",
            xscale=xscale,
            yscale=yscale,
            xtick_labels=xtick_labels,
            ytick_labels=ytick_labels,
            legend_labels=legend_labels,
            texts=texts,
            artists=tuple(artists),
        )
        axes = (axis,) + tuple(
            AxisTrace(index=index, projection="plotly")
            for index in range(1, max(1, axis_count))
        )
        return FigureTrace(
            title=title,
            size_inches=self._plotly_size_inches(width, height),
            axes=axes,
            source="plotly_figure",
            raw={
                "backend": "plotly",
                "trace_types": list(trace_types),
                "trace_count": len(trace_types),
                "layout_json": self._plotly_layout_json(fig),
            },
        )

    def _plotly_chart_type(self, trace_types: tuple[str, ...]) -> ChartType:
        normalized = {trace_type.lower() for trace_type in trace_types if trace_type}
        if "bar" in normalized:
            return "bar"
        if "line" in normalized:
            return "line"
        if "scatter" in normalized:
            return "scatter"
        if "pie" in normalized or "sunburst" in normalized:
            return "pie"
        if "heatmap" in normalized:
            return "heatmap"
        return "unknown"

    def _plotly_trace_descriptor(self, trace: Any) -> str:
        trace_type = str(getattr(trace, "type", "") or "").strip().lower()
        if trace_type == "scatter":
            mode = str(getattr(trace, "mode", "") or "").strip().lower()
            if "lines" in mode and "markers" not in mode:
                return "line"
            if "lines" in mode and "markers" in mode:
                return "line"
            return "scatter"
        return trace_type or "unknown"

    def _plotly_points(self, fig: Any, chart_type: ChartType) -> tuple[DataPoint, ...]:
        if chart_type not in {"bar", "line", "scatter", "pie"}:
            return ()
        try:
            traces = tuple(getattr(fig, "data", ()))
        except Exception:
            return ()
        trace = traces[0]
        try:
            if chart_type == "pie":
                if len(traces) != 1:
                    return ()
                labels = self._as_list(getattr(trace, "labels", ()))
                values = self._as_list(getattr(trace, "values", ()))
                return tuple(DataPoint(x=label, y=value) for label, value in zip(labels, values))
            if len(traces) == 1:
                x_values = self._as_list(getattr(trace, "x", ()))
                y_values = self._as_list(getattr(trace, "y", ()))
                return tuple(DataPoint(x=x, y=y) for x, y in zip(x_values, y_values))
            facet_map = self._plotly_trace_facet_map(fig)
            points: list[DataPoint] = []
            for trace in traces:
                series_name = self._clean_plotly_label(getattr(trace, "name", None))
                facet_label = facet_map.get(self._plotly_trace_axis_key(trace))
                facet_value = self._plotly_facet_value(facet_label)
                x_values = self._as_list(getattr(trace, "x", ()))
                y_values = self._as_list(getattr(trace, "y", ()))
                for x, y in zip(x_values, y_values):
                    key_parts = [x]
                    meta: dict[str, Any] = {}
                    if series_name:
                        key_parts.append(series_name)
                        meta["series"] = series_name
                    if facet_value:
                        key_parts.append(facet_value)
                        meta["facet"] = facet_value
                    if facet_label and facet_label != facet_value:
                        meta["facet_label"] = facet_label
                    point_key: Any = tuple(key_parts) if len(key_parts) > 1 else x
                    points.append(DataPoint(x=point_key, y=y, meta=meta))
            return tuple(points)
        except Exception:
            return ()

    def _plotly_trace_axis_key(self, trace: Any) -> tuple[str, str]:
        xaxis = str(getattr(trace, "xaxis", "") or "x")
        yaxis = str(getattr(trace, "yaxis", "") or "y")
        return xaxis, yaxis

    def _plotly_trace_facet_map(self, fig: Any) -> dict[tuple[str, str], str]:
        layout_json = self._plotly_layout_json(fig) or {}
        axis_domains = self._plotly_axis_domains(layout_json)
        if not axis_domains:
            return {}
        annotations = layout_json.get("annotations", []) or []
        if not isinstance(annotations, list):
            return {}
        facet_map: dict[tuple[str, str], str] = {}
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            text = str(annotation.get("text", "") or "").strip()
            if "=" not in text:
                continue
            xref = str(annotation.get("xref", "") or "")
            yref = str(annotation.get("yref", "") or "")
            if xref != "paper" or yref != "paper":
                continue
            annotation_x = annotation.get("x")
            annotation_y = annotation.get("y")
            if annotation_x is None or annotation_y is None:
                continue
            matched_key = self._nearest_plotly_axis_key(axis_domains, float(annotation_x), float(annotation_y))
            if matched_key is not None and matched_key not in facet_map:
                facet_map[matched_key] = text
        return facet_map

    def _plotly_axis_domains(self, layout_json: dict[str, Any]) -> dict[tuple[str, str], tuple[float, float]]:
        axis_keys: dict[tuple[str, str], tuple[float, float]] = {}
        x_axes = {str(key): value for key, value in layout_json.items() if str(key).startswith("xaxis") and isinstance(value, dict)}
        y_axes = {str(key): value for key, value in layout_json.items() if str(key).startswith("yaxis") and isinstance(value, dict)}
        for x_name, x_axis in x_axes.items():
            suffix = x_name[5:]
            y_name = f"yaxis{suffix}"
            y_axis = y_axes.get(y_name)
            if y_axis is None:
                continue
            x_domain = x_axis.get("domain")
            y_domain = y_axis.get("domain")
            if not isinstance(x_domain, list) or not isinstance(y_domain, list) or len(x_domain) != 2 or len(y_domain) != 2:
                continue
            try:
                x_center = (float(x_domain[0]) + float(x_domain[1])) / 2.0
                y_center = (float(y_domain[0]) + float(y_domain[1])) / 2.0
            except (TypeError, ValueError):
                continue
            trace_x = "x" if not suffix else f"x{suffix}"
            trace_y = "y" if not suffix else f"y{suffix}"
            axis_keys[(trace_x, trace_y)] = (x_center, y_center)
        return axis_keys

    def _nearest_plotly_axis_key(
        self,
        axis_domains: dict[tuple[str, str], tuple[float, float]],
        annotation_x: float,
        annotation_y: float,
    ) -> tuple[str, str] | None:
        best_key: tuple[str, str] | None = None
        best_distance: float | None = None
        for axis_key, (x_center, y_center) in axis_domains.items():
            distance = ((annotation_x - x_center) ** 2 + (annotation_y - y_center) ** 2) ** 0.5
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_key = axis_key
        return best_key

    def _plotly_size_inches(self, width: Any, height: Any) -> tuple[float, float] | None:
        try:
            if width is None or height is None:
                return None
            return (round(float(width) / 100.0, 4), round(float(height) / 100.0, 4))
        except (TypeError, ValueError):
            return None

    def _plotly_layout_json(self, fig: Any) -> dict[str, Any] | None:
        try:
            layout = getattr(fig, "layout", None)
            if layout is None:
                return None
            if hasattr(layout, "to_plotly_json"):
                return layout.to_plotly_json()
            return json.loads(json.dumps(layout))
        except Exception:
            return None

    def _plotly_annotation_texts(self, layout: Any) -> tuple[str, ...]:
        try:
            annotations = getattr(layout, "annotations", None)
            if annotations is None:
                return ()
        except Exception:
            return ()
        texts: list[str] = []
        for annotation in annotations:
            try:
                text = getattr(annotation, "text", "")
            except Exception:
                text = ""
            normalized = str(text or "").strip()
            if normalized:
                texts.append(normalized)
        return tuple(texts)

    def _plotly_axes_count(self, layout: Any) -> int:
        try:
            layout_json = layout.to_plotly_json() if hasattr(layout, "to_plotly_json") else {}
        except Exception:
            return 1
        x_axes = [key for key in layout_json.keys() if str(key).startswith("xaxis")]
        y_axes = [key for key in layout_json.keys() if str(key).startswith("yaxis")]
        return max(1, len(x_axes), len(y_axes))

    def _plotly_axis_title(self, layout: Any, axis_name: str) -> str:
        try:
            axis = getattr(layout, axis_name, None)
            if axis is None:
                return ""
            title = getattr(axis, "title", None)
            return str(getattr(title, "text", "") or "")
        except Exception:
            return ""

    def _plotly_axis_scale(self, layout: Any, axis_name: str) -> str:
        try:
            axis = getattr(layout, axis_name, None)
            if axis is None:
                return "linear"
            axis_type = str(getattr(axis, "type", "") or "").strip().lower()
            return axis_type or "linear"
        except Exception:
            return "linear"

    def _plotly_tick_labels(self, layout: Any, axis_name: str) -> tuple[str, ...]:
        try:
            axis = getattr(layout, axis_name, None)
            if axis is None:
                return ()
            tick_text = getattr(axis, "ticktext", None)
            if tick_text:
                return tuple(str(item) for item in self._as_list(tick_text) if str(item).strip())
            tick_vals = getattr(axis, "tickvals", None)
            if tick_vals:
                return tuple(str(item) for item in self._as_list(tick_vals) if str(item).strip())
        except Exception:
            return ()
        return ()

    def _plotly_legend_labels(self, fig: Any) -> tuple[str, ...]:
        labels: list[str] = []
        try:
            for trace in getattr(fig, "data", ()):
                showlegend = getattr(trace, "showlegend", True)
                if showlegend is False:
                    continue
                label = self._clean_plotly_label(getattr(trace, "name", None))
                if label:
                    labels.append(label)
        except Exception:
            return ()
        return tuple(dict.fromkeys(labels))

    def _plotly_colorbar_titles(self, fig: Any) -> tuple[str, ...]:
        texts: list[str] = []
        try:
            for trace in getattr(fig, "data", ()):
                colorbar = getattr(trace, "colorbar", None)
                if colorbar is None:
                    continue
                title = getattr(colorbar, "title", None)
                text = str(getattr(title, "text", "") or "").strip()
                if text:
                    texts.append(text)
        except Exception:
            return ()
        return tuple(dict.fromkeys(texts))

    def _plotly_trace_texts(self, fig: Any) -> tuple[str, ...]:
        texts: list[str] = []
        try:
            for trace in getattr(fig, "data", ()):
                raw_text = getattr(trace, "text", None)
                if raw_text is None:
                    continue
                for item in self._as_list(raw_text):
                    normalized = str(item or "").strip()
                    if normalized:
                        texts.append(normalized)
        except Exception:
            return ()
        return tuple(texts)

    def _plotly_visible_texts(self, fig: Any, layout: Any) -> tuple[str, ...]:
        merged = list(self._plotly_annotation_texts(layout))
        merged.extend(self._plotly_trace_texts(fig))
        merged.extend(self._plotly_colorbar_titles(fig))
        return tuple(dict.fromkeys(text for text in merged if str(text).strip()))

    def _plotly_shape_count(self, layout: Any) -> int:
        try:
            shapes = getattr(layout, "shapes", None)
            if shapes is None:
                return 0
            return len(shapes)
        except Exception:
            return 0

    def _axis_trace(self, index: int, ax: Any) -> AxisTrace:
        projection = getattr(ax, "name", None) or "rectilinear"
        legend = ax.get_legend()
        legend_labels = tuple(text.get_text() for text in legend.get_texts()) if legend is not None else ()
        texts = tuple(text.get_text() for text in getattr(ax, "texts", ()))
        artists = tuple(self._artist_traces(ax))
        return AxisTrace(
            index=index,
            title=ax.get_title(),
            xlabel=ax.get_xlabel(),
            ylabel=ax.get_ylabel(),
            zlabel=ax.get_zlabel() if hasattr(ax, "get_zlabel") else "",
            projection=str(projection),
            xscale=ax.get_xscale() if hasattr(ax, "get_xscale") else "linear",
            yscale=ax.get_yscale() if hasattr(ax, "get_yscale") else "linear",
            zscale=ax.get_zscale() if hasattr(ax, "get_zscale") else "linear",
            xtick_labels=tuple(label.get_text() for label in ax.get_xticklabels()),
            ytick_labels=tuple(label.get_text() for label in ax.get_yticklabels()),
            ztick_labels=tuple(label.get_text() for label in ax.get_zticklabels()) if hasattr(ax, "get_zticklabels") else (),
            bounds=tuple(round(float(item), 4) for item in ax.get_position().bounds),
            legend_labels=legend_labels,
            texts=texts,
            artists=artists,
        )

    def _artist_traces(self, ax: Any) -> list[ArtistTrace]:
        artists: list[ArtistTrace] = []
        for line in getattr(ax, "lines", ()):
            label = self._clean_label(line.get_label())
            artists.append(
                ArtistTrace(
                    artist_type="line",
                    label=label,
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    marker=line.get_marker(),
                    count=len(line.get_xdata()) if hasattr(line, "get_xdata") else None,
                )
            )
        for container in getattr(ax, "containers", ()):
            container_name = type(container).__name__
            if container_name == "ErrorbarContainer":
                data_line = container.lines[0] if getattr(container, "lines", None) else None
                label = self._clean_label(container.get_label() if hasattr(container, "get_label") else None)
                count = len(data_line.get_xdata()) if data_line is not None and hasattr(data_line, "get_xdata") else None
                artists.append(ArtistTrace(artist_type="errorbar", label=label, count=count))
                continue
            if container_name == "BarContainer":
                label = self._clean_label(container.get_label() if hasattr(container, "get_label") else None)
                count = len(container) if hasattr(container, "__len__") else None
                artists.append(ArtistTrace(artist_type="bar", label=label, count=count))
        for collection in getattr(ax, "collections", ()):
            offsets = collection.get_offsets() if hasattr(collection, "get_offsets") else ()
            count = len(offsets) if hasattr(offsets, "__len__") else None
            collection_type = type(collection).__name__
            artist_type = "scatter" if collection_type == "PathCollection" and count not in {None, 0} else "collection"
            artists.append(ArtistTrace(artist_type=artist_type, label=self._clean_label(collection.get_label()), count=count))
        wedge_count = len(
            [
                patch_item
                for patch_item in getattr(ax, "patches", ())
                if patch_item.get_visible() and type(patch_item).__name__ == "Wedge"
            ]
        )
        if wedge_count:
            artists.append(ArtistTrace(artist_type="pie", count=wedge_count))
        patch_count = len([patch_item for patch_item in getattr(ax, "patches", ()) if patch_item.get_visible()])
        if patch_count:
            artists.append(ArtistTrace(artist_type="patch", count=patch_count))
        table_count = len(getattr(ax, "tables", {}))
        if table_count:
            artists.append(ArtistTrace(artist_type="table", count=table_count))
        image_count = len(getattr(ax, "images", ()))
        if image_count:
            artists.append(ArtistTrace(artist_type="image", count=image_count))
        return artists

    def _is_helper_axis(self, ax: Any) -> bool:
        label = str(ax.get_label() or "").strip().lower()
        if label == "<colorbar>":
            return True
        return any(type(child).__name__ == "_ColorbarSpine" for child in ax.get_children())

    def _clean_label(self, label: Any) -> str | None:
        if label is None:
            return None
        label = str(label)
        if not label or label.startswith("_"):
            return None
        return label

    def _clean_plotly_label(self, label: Any) -> str | None:
        if label is None:
            return None
        normalized = str(label).strip()
        if not normalized or normalized.lower().startswith("trace "):
            return None
        return normalized

    def _plotly_facet_value(self, label: str | None) -> str | None:
        if label is None:
            return None
        normalized = str(label).strip()
        if not normalized:
            return None
        if "=" not in normalized:
            return normalized
        _, value = normalized.split("=", 1)
        value = value.strip()
        return value or normalized

    def _plotly_patchers(self, plotly_express_wrapper: Any, plotly_show_wrapper: Any, plotly_write_image_wrapper: Any) -> list[Any]:
        patchers: list[Any] = []
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.io as pio
        except Exception:
            return patchers

        for name in ("bar", "line", "scatter", "pie", "sunburst", "treemap"):
            if hasattr(px, name):
                patchers.append(patch.object(px, name, plotly_express_wrapper(getattr(px, name))))
        patchers.append(patch.object(go.Figure, "show", plotly_show_wrapper))
        patchers.append(patch.object(go.Figure, "write_image", plotly_write_image_wrapper))
        if hasattr(pio, "write_image"):
            patchers.append(
                patch.object(
                    pio,
                    "write_image",
                    lambda fig, *args, **kwargs: plotly_write_image_wrapper(fig, *args, **kwargs),
                )
            )
        return patchers

    @contextmanager
    def _patch_all(self, patchers: list[Any]):
        started = []
        try:
            for patcher in patchers:
                started.append(patcher)
                patcher.start()
            yield
        finally:
            for patcher in reversed(started):
                patcher.stop()

    @contextmanager
    def _working_directory(self, execution_dir: Path | None):
        if execution_dir is None:
            yield
            return
        previous = Path.cwd()
        os.chdir(execution_dir)
        try:
            yield
        finally:
            os.chdir(previous)
