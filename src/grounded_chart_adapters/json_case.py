from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from grounded_chart import (
    AxisRequirementSpec,
    ChartIntentPlan,
    DataPoint,
    FigureRequirementSpec,
    FilterSpec,
    MeasureSpec,
    ParsedRequirementBundle,
    PlotTrace,
    SortSpec,
    TableSchema,
    build_requirement_plan,
)
from grounded_chart.expected_artifacts import extract_expected_trace_from_texts
from grounded_chart.requirements import ChartRequirementPlan, PanelRequirementPlan, RequirementNode
from grounded_chart_adapters.base import ChartCase


class JsonCaseAdapter:
    """Load GroundedChart `ChartCase` objects from a JSON case list."""

    def __init__(self, path: str | Path, *, parse_source_mode: str = "predicted") -> None:
        self.path = Path(path)
        normalized = str(parse_source_mode or "predicted").strip().lower()
        if normalized not in {"predicted", "oracle"}:
            raise ValueError(f"Unsupported parse_source_mode: {parse_source_mode}")
        self.parse_source_mode = normalized

    def iter_cases(self) -> Iterable[ChartCase]:
        for raw in self._load_raw_cases():
            schema = raw["schema"]
            generated_code = raw.get("generated_code", "")
            if not generated_code and raw.get("generated_code_path"):
                generated_code = self._resolve_path(str(raw["generated_code_path"])).read_text(encoding="utf-8", errors="replace")
            table_schema = TableSchema(
                columns=dict(schema["columns"]),
                table_name=schema.get("table_name", "table"),
            )
            yield ChartCase(
                case_id=str(raw["case_id"]),
                query=raw["query"],
                schema=table_schema,
                rows=tuple(dict(row) for row in raw["rows"]),
                generated_code=generated_code,
                figure_requirements=self._figure_requirements(raw.get("figure_requirements")),
                verification_mode=raw.get("verification_mode", "full"),
                expected_trace=self._expected_trace(raw),
                parsed_requirements=self._parsed_requirements(raw, schema=table_schema),
                parse_source=self.parse_source_mode,
                metadata=dict(raw.get("metadata", {})),
            )

    def supports_oracle_requirements(self) -> bool:
        return any(self._has_oracle_payload(raw) for raw in self._load_raw_cases())

    def _load_raw_cases(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"JSON case file not found: {self.path}")
        cases = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if not isinstance(cases, list):
            raise ValueError("JSON case file must contain a list of cases.")
        for index, case in enumerate(cases):
            missing = {"case_id", "query", "schema", "rows"} - set(case)
            if missing:
                raise ValueError(f"Case at index {index} is missing fields: {sorted(missing)}")
            if not case.get("generated_code") and not case.get("generated_code_path"):
                raise ValueError(f"Case at index {index} needs either `generated_code` or `generated_code_path`.")
            if "columns" not in case["schema"]:
                raise ValueError(f"Case at index {index} schema is missing `columns`.")
            verification_mode = case.get("verification_mode", "full")
            if verification_mode not in {"full", "figure_only", "figure_and_data"}:
                raise ValueError(f"Case at index {index} has unsupported verification_mode: {verification_mode}")
        return cases


    def _expected_trace(self, raw_case: dict[str, Any]) -> PlotTrace | None:
        raw_trace = raw_case.get("expected_trace")
        if raw_trace is None and raw_case.get("expected_points") is not None:
            raw_trace = {
                "chart_type": raw_case.get("expected_chart_type") or "unknown",
                "points": raw_case.get("expected_points"),
                "source": "benchmark_expected_points",
            }
        if raw_trace is None:
            return extract_expected_trace_from_texts(
                self._expected_trace_source_texts(raw_case),
                default_chart_type=self._default_expected_chart_type(raw_case),
            )
        if not isinstance(raw_trace, dict):
            raise ValueError(f"Case {raw_case.get('case_id')} expected_trace must be an object.")
        points_raw = raw_trace.get("points")
        if points_raw is None:
            points_raw = raw_trace.get("expected_points")
        if not isinstance(points_raw, list):
            raise ValueError(f"Case {raw_case.get('case_id')} expected_trace.points must be a list.")
        points: list[DataPoint] = []
        for index, point in enumerate(points_raw):
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ValueError(
                    f"Case {raw_case.get('case_id')} expected_trace point {index} must contain x and y."
                )
            meta = point.get("meta") if isinstance(point.get("meta"), dict) else {}
            points.append(DataPoint(x=point.get("x"), y=point.get("y"), meta=dict(meta)))
        return PlotTrace(
            chart_type=str(raw_trace.get("chart_type") or raw_case.get("expected_chart_type") or "unknown"),
            points=tuple(points),
            source=str(raw_trace.get("source") or "benchmark_expected_trace"),
            raw={"expected_trace": dict(raw_trace)},
        )


    def _expected_trace_source_texts(self, raw_case: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        sources: list[tuple[str, str]] = []
        for key in ("expert_instruction", "simple_instruction", "instruction", "raw_instruction", "prompt", "query"):
            value = raw_case.get(key)
            if isinstance(value, str) and value.strip():
                sources.append((key, value))
        metadata = raw_case.get("metadata") if isinstance(raw_case.get("metadata"), dict) else {}
        for key in ("expert_instruction", "simple_instruction", "instruction", "raw_instruction", "source_instruction"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                sources.append((f"metadata.{key}", value))
        return tuple(sources)

    def _default_expected_chart_type(self, raw_case: dict[str, Any]) -> str:
        if raw_case.get("expected_chart_type"):
            return str(raw_case["expected_chart_type"])
        raw_plan = raw_case.get("oracle_plan")
        if isinstance(raw_plan, dict) and raw_plan.get("chart_type"):
            return str(raw_plan["chart_type"])
        raw_bundle = raw_case.get("oracle_bundle") if isinstance(raw_case.get("oracle_bundle"), dict) else {}
        bundled_plan = raw_bundle.get("plan") if isinstance(raw_bundle.get("plan"), dict) else None
        if isinstance(bundled_plan, dict) and bundled_plan.get("chart_type"):
            return str(bundled_plan["chart_type"])
        raw_requirement_plan = raw_case.get("oracle_requirement_plan")
        if not isinstance(raw_requirement_plan, dict):
            raw_requirement_plan = raw_bundle.get("requirement_plan") if isinstance(raw_bundle.get("requirement_plan"), dict) else None
        panels = raw_requirement_plan.get("panels") if isinstance(raw_requirement_plan, dict) else None
        if isinstance(panels, list) and panels and isinstance(panels[0], dict) and panels[0].get("chart_type"):
            return str(panels[0]["chart_type"])
        return "unknown"

    def _figure_requirements(self, raw: dict[str, Any] | None) -> FigureRequirementSpec | None:
        if raw is None:
            return None
        axes = tuple(
            AxisRequirementSpec(
                axis_index=int(axis.get("axis_index", 0)),
                title=axis.get("title"),
                xlabel=axis.get("xlabel"),
                ylabel=axis.get("ylabel"),
                zlabel=axis.get("zlabel"),
                projection=axis.get("projection"),
                xscale=axis.get("xscale"),
                yscale=axis.get("yscale"),
                zscale=axis.get("zscale"),
                xtick_labels=tuple(axis.get("xtick_labels", ())),
                ytick_labels=tuple(axis.get("ytick_labels", ())),
                ztick_labels=tuple(axis.get("ztick_labels", ())),
                bounds=tuple(float(item) for item in axis["bounds"]) if axis.get("bounds") is not None else None,
                legend_labels=tuple(axis.get("legend_labels", ())),
                artist_types=tuple(axis.get("artist_types", ())),
                artist_counts={str(key): int(value) for key, value in axis.get("artist_counts", {}).items()},
                min_artist_counts={str(key): int(value) for key, value in axis.get("min_artist_counts", {}).items()},
                text_contains=tuple(axis.get("text_contains", ())),
            )
            for axis in raw.get("axes", ())
        )
        size_inches = raw.get("size_inches")
        return FigureRequirementSpec(
            axes_count=raw.get("axes_count"),
            figure_title=raw.get("figure_title"),
            size_inches=tuple(float(item) for item in size_inches) if size_inches is not None else None,
            axes=axes,
        )

    def _parsed_requirements(self, raw: dict[str, Any], *, schema: TableSchema) -> ParsedRequirementBundle | None:
        if self.parse_source_mode != "oracle":
            return None
        if not self._has_oracle_payload(raw):
            raise ValueError(
                f"Case {raw.get('case_id')} has no oracle_plan/oracle_requirement_plan payload, "
                "but parse_source_mode='oracle' was requested."
            )

        raw_bundle = raw.get("oracle_bundle") if isinstance(raw.get("oracle_bundle"), dict) else {}
        raw_plan = raw_bundle.get("plan") if isinstance(raw_bundle.get("plan"), dict) else raw.get("oracle_plan")
        raw_requirement_plan = (
            raw_bundle.get("requirement_plan")
            if isinstance(raw_bundle.get("requirement_plan"), dict)
            else raw.get("oracle_requirement_plan")
        )
        raw_response = dict(raw_bundle.get("raw_response", {})) if isinstance(raw_bundle, dict) else {}

        plan = (
            self._plan_from_dict(raw_plan)
            if isinstance(raw_plan, dict)
            else self._plan_from_requirement_plan_dict(raw_requirement_plan, query=raw["query"])
        )
        requirement_plan = (
            self._requirement_plan_from_dict(raw_requirement_plan, query=raw["query"])
            if isinstance(raw_requirement_plan, dict)
            else build_requirement_plan(plan)
        )
        return ParsedRequirementBundle(
            plan=plan,
            requirement_plan=requirement_plan,
            raw_response=raw_response,
        )

    def _has_oracle_payload(self, raw: dict[str, Any]) -> bool:
        return any(
            raw.get(key) is not None
            for key in ("oracle_bundle", "oracle_plan", "oracle_requirement_plan")
        )

    def _plan_from_dict(self, raw: dict[str, Any]) -> ChartIntentPlan:
        measure = raw.get("measure") if isinstance(raw.get("measure"), dict) else {}
        sort = raw.get("sort") if isinstance(raw.get("sort"), dict) else None
        filters = raw.get("filters") if isinstance(raw.get("filters"), list) else []
        return ChartIntentPlan(
            chart_type=str(raw.get("chart_type") or "unknown"),
            dimensions=tuple(str(item) for item in raw.get("dimensions", ()) if item is not None),
            measure=MeasureSpec(
                column=measure.get("column"),
                agg=str(measure.get("agg") or "none"),
            ),
            filters=tuple(
                FilterSpec(
                    column=str(item["column"]),
                    op=str(item["op"]),
                    value=item.get("value"),
                )
                for item in filters
                if isinstance(item, dict) and "column" in item and "op" in item
            ),
            sort=(
                SortSpec(by=str(sort.get("by")), direction=str(sort.get("direction")))
                if sort is not None and sort.get("by") and sort.get("direction")
                else None
            ),
            limit=raw.get("limit"),
            raw_query=str(raw.get("raw_query") or ""),
            confidence=raw.get("confidence"),
        )

    def _plan_from_requirement_plan_dict(self, raw: dict[str, Any] | None, *, query: str) -> ChartIntentPlan:
        if not isinstance(raw, dict):
            return ChartIntentPlan(
                chart_type="unknown",
                dimensions=(),
                measure=MeasureSpec(column=None, agg="none"),
                raw_query=query,
                confidence=None,
            )
        panels = raw.get("panels", [])
        first_panel = panels[0] if isinstance(panels, list) and panels else {}
        data_ops = first_panel.get("data_ops") if isinstance(first_panel.get("data_ops"), dict) else {}
        presentation_constraints = (
            first_panel.get("presentation_constraints")
            if isinstance(first_panel.get("presentation_constraints"), dict)
            else {}
        )
        filters = data_ops.get("filters") if isinstance(data_ops.get("filters"), (list, tuple)) else []
        sort = presentation_constraints.get("sort") if isinstance(presentation_constraints.get("sort"), dict) else None
        return ChartIntentPlan(
            chart_type=str(first_panel.get("chart_type") or "unknown"),
            dimensions=tuple(str(item) for item in data_ops.get("dimensions", ()) if item is not None),
            measure=MeasureSpec(
                column=data_ops.get("measure_column"),
                agg=str(data_ops.get("aggregation") or "none"),
            ),
            filters=tuple(
                FilterSpec(
                    column=str(item["column"]),
                    op=str(item["op"]),
                    value=item.get("value"),
                )
                for item in filters
                if isinstance(item, dict) and "column" in item and "op" in item
            ),
            sort=(
                SortSpec(by=str(sort.get("by")), direction=str(sort.get("direction")))
                if sort is not None and sort.get("by") and sort.get("direction")
                else None
            ),
            limit=presentation_constraints.get("limit"),
            raw_query=str(raw.get("raw_query") or query),
            confidence=None,
        )

    def _requirement_plan_from_dict(self, raw: dict[str, Any] | None, *, query: str) -> ChartRequirementPlan:
        if not isinstance(raw, dict):
            fallback_plan = self._plan_from_requirement_plan_dict(raw, query=query)
            return build_requirement_plan(fallback_plan)
        requirements = tuple(
            RequirementNode(
                requirement_id=str(item["requirement_id"]),
                scope=str(item.get("scope", "panel")),
                type=str(item.get("type", "data_operation")),
                name=str(item.get("name", "")),
                value=item.get("value"),
                source_span=str(item.get("source_span", "")),
                status=str(item.get("status", "explicit")),
                confidence=item.get("confidence"),
                depends_on=tuple(str(dep) for dep in item.get("depends_on", [])),
                priority=str(item.get("priority", "core")),
                panel_id=item.get("panel_id"),
                                assumption=item.get("assumption"),
                severity=str(item.get("severity", "error")),
                match_policy=str(item.get("match_policy", "exact")),            )
            for item in raw.get("requirements", [])
            if isinstance(item, dict) and item.get("requirement_id")
        )
        panels = tuple(
            PanelRequirementPlan(
                panel_id=str(panel.get("panel_id") or "panel_0"),
                chart_type=str(panel.get("chart_type") or "unknown"),
                requirement_ids=tuple(str(item) for item in panel.get("requirement_ids", [])),
                data_ops=dict(panel.get("data_ops", {})),
                encodings=dict(panel.get("encodings", {})),
                annotations=dict(panel.get("annotations", {})),
                presentation_constraints=dict(panel.get("presentation_constraints", {})),
            )
            for panel in raw.get("panels", [])
            if isinstance(panel, dict)
        )
        return ChartRequirementPlan(
            requirements=requirements,
            panels=panels,
            figure_requirements=dict(raw.get("figure_requirements", {})),
            shared_requirement_ids=tuple(str(item) for item in raw.get("shared_requirement_ids", [])),
            raw_query=str(raw.get("raw_query") or query),
        )

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.path.parent / path
        return path
