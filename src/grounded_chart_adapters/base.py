from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol

from grounded_chart.core.schema import FigureRequirementSpec, ParsedRequirementBundle, PipelineResult, PlotTrace, TableSchema


@dataclass(frozen=True)
class ChartCase:
    """Benchmark-neutral chart generation case.

    Adapters should convert benchmark-native records into this shape before
    calling the core GroundedChart pipeline.
    """

    case_id: str
    query: str
    schema: TableSchema
    rows: tuple[dict[str, Any], ...]
    generated_code: str
    figure_requirements: FigureRequirementSpec | None = None
    verification_mode: Literal["full", "figure_only", "figure_and_data"] = "full"
    expected_trace: PlotTrace | None = None
    parsed_requirements: ParsedRequirementBundle | None = None
    parse_source: Literal["predicted", "oracle"] = "predicted"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterRunResult:
    case: ChartCase
    pipeline_result: PipelineResult
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    def iter_cases(self) -> Iterable[ChartCase]:
        """Yield benchmark-neutral chart cases."""
