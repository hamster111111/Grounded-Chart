from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from grounded_chart import AxisRequirementSpec, FigureRequirementSpec, TableSchema
from grounded_chart_adapters.base import ChartCase


class JsonCaseAdapter:
    """Load GroundedChart `ChartCase` objects from a JSON case list."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_cases(self) -> Iterable[ChartCase]:
        for raw in self._load_raw_cases():
            schema = raw["schema"]
            generated_code = raw.get("generated_code", "")
            if not generated_code and raw.get("generated_code_path"):
                generated_code = self._resolve_path(str(raw["generated_code_path"])).read_text(encoding="utf-8", errors="replace")
            yield ChartCase(
                case_id=str(raw["case_id"]),
                query=raw["query"],
                schema=TableSchema(
                    columns=dict(schema["columns"]),
                    table_name=schema.get("table_name", "table"),
                ),
                rows=tuple(dict(row) for row in raw["rows"]),
                generated_code=generated_code,
                figure_requirements=self._figure_requirements(raw.get("figure_requirements")),
                verification_mode=raw.get("verification_mode", "full"),
                metadata=dict(raw.get("metadata", {})),
            )

    def _load_raw_cases(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"JSON case file not found: {self.path}")
        cases = json.loads(self.path.read_text(encoding="utf-8"))
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
            if verification_mode not in {"full", "figure_only"}:
                raise ValueError(f"Case at index {index} has unsupported verification_mode: {verification_mode}")
        return cases

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

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.path.parent / path
        return path
