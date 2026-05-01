from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from grounded_chart.llm import LLMCompletionTrace
from grounded_chart.patch_ops import PatchOperation
from grounded_chart.requirements import ChartRequirementPlan, EvidenceGraph

if TYPE_CHECKING:
    from grounded_chart.repair_policy import RepairPlan

Aggregation = Literal["none", "count", "sum", "mean", "min", "max"]
ChartType = Literal["bar", "line", "pie", "scatter", "area", "heatmap", "box", "unknown"]
FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains"]
SortTarget = Literal["dimension", "measure"]
SortDirection = Literal["asc", "desc"]
RepairLoopAction = Literal["continue", "stop", "escalate"]
PipelineStage = Literal["plan", "execution", "repair"]


@dataclass(frozen=True)
class TableSchema:
    """Minimal table schema exposed to the intent parser."""

    columns: dict[str, str]
    table_name: str = "table"


@dataclass(frozen=True)
class FilterSpec:
    column: str
    op: FilterOp
    value: Any


@dataclass(frozen=True)
class MeasureSpec:
    column: str | None
    agg: Aggregation = "none"


@dataclass(frozen=True)
class SortSpec:
    by: SortTarget = "dimension"
    direction: SortDirection = "asc"


@dataclass(frozen=True)
class ChartIntentPlan:
    """Operator-level representation of the user's chart intent.

    This is the main research object: it makes language-to-chart fidelity
    checkable before and after plotting code execution.
    """

    chart_type: ChartType
    dimensions: tuple[str, ...]
    measure: MeasureSpec
    filters: tuple[FilterSpec, ...] = ()
    sort: SortSpec | None = None
    limit: int | None = None
    raw_query: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class ParsedRequirementBundle:
    """Parser-native chart intent plus traceable requirement nodes."""

    plan: ChartIntentPlan
    requirement_plan: ChartRequirementPlan
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataPoint:
    x: Any
    y: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlotTrace:
    """Normalized view of what the generated code actually plotted."""

    chart_type: ChartType
    points: tuple[DataPoint, ...]
    source: str = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtistTrace:
    """Lightweight summary of one visual artist family on an axis."""

    artist_type: str
    label: str | None = None
    color: Any = None
    linestyle: str | None = None
    marker: str | None = None
    count: int | None = None


@dataclass(frozen=True)
class AxisTrace:
    """Requirement-relevant properties extracted from one Matplotlib axis."""

    index: int
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    projection: str = "rectilinear"
    xscale: str = "linear"
    yscale: str = "linear"
    zscale: str = "linear"
    xtick_labels: tuple[str, ...] = ()
    ytick_labels: tuple[str, ...] = ()
    ztick_labels: tuple[str, ...] = ()
    bounds: tuple[float, float, float, float] | None = None
    legend_labels: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    artists: tuple[ArtistTrace, ...] = ()


@dataclass(frozen=True)
class FigureTrace:
    """Requirement-relevant properties extracted from a rendered figure."""

    title: str = ""
    size_inches: tuple[float, float] | None = None
    axes: tuple[AxisTrace, ...] = ()
    source: str = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def axes_count(self) -> int:
        return len(self.axes)


@dataclass(frozen=True)
class AxisRequirementSpec:
    """Explicit figure-level requirements for one axis.

    `None` means unconstrained. Empty strings are valid expected values.
    """

    axis_index: int = 0
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    zlabel: str | None = None
    projection: str | None = None
    xscale: str | None = None
    yscale: str | None = None
    zscale: str | None = None
    xtick_labels: tuple[str, ...] = ()
    ytick_labels: tuple[str, ...] = ()
    ztick_labels: tuple[str, ...] = ()
    bounds: tuple[float, float, float, float] | None = None
    legend_labels: tuple[str, ...] = ()
    artist_types: tuple[str, ...] = ()
    artist_counts: dict[str, int] = field(default_factory=dict)
    min_artist_counts: dict[str, int] = field(default_factory=dict)
    text_contains: tuple[str, ...] = ()
    provenance: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_spans: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FigureRequirementSpec:
    """Explicit figure-level requirements independent of plotted data."""

    axes_count: int | None = None
    figure_title: str | None = None
    size_inches: tuple[float, float] | None = None
    axes: tuple[AxisRequirementSpec, ...] = ()
    provenance: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_spans: dict[str, str] = field(default_factory=dict)
    artifact_contracts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class VerificationError:
    code: str
    message: str
    expected: Any = None
    actual: Any = None
    operator: str | None = None
    requirement_id: str | None = None
    severity: Literal["info", "warning", "error"] = "error"
    match_policy: str | None = None


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    errors: tuple[VerificationError, ...]
    expected_trace: PlotTrace
    actual_trace: PlotTrace
    expected_figure: FigureRequirementSpec | None = None
    actual_figure: FigureTrace | None = None

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(error.code for error in self.errors)


@dataclass(frozen=True)
class RepairPatch:
    strategy: str
    instruction: str
    target_error_codes: tuple[str, ...]
    repair_plan: RepairPlan | None = None
    repaired_code: str | None = None
    loop_signal: RepairLoopAction | None = None
    loop_reason: str | None = None
    llm_trace: LLMCompletionTrace | None = None
    patch_ops: tuple[PatchOperation, ...] = ()


@dataclass(frozen=True)
class RepairAttempt:
    round_index: int
    input_code: str
    output_code: str
    applied: bool
    strategy: str
    scope: str | None = None
    targeted_requirement_ids: tuple[str, ...] = ()
    targeted_error_codes: tuple[str, ...] = ()
    resolved_requirement_ids: tuple[str, ...] = ()
    unresolved_requirement_ids: tuple[str, ...] = ()
    report_ok: bool = False
    instruction: str = ""
    decision_action: RepairLoopAction | None = None
    decision_reason: str = ""
    decision_next_scope: str | None = None
    llm_trace: LLMCompletionTrace | None = None
    patch_ops: tuple[PatchOperation, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    plan: ChartIntentPlan
    expected_trace: PlotTrace
    actual_trace: PlotTrace
    report: VerificationReport
    expected_figure: FigureRequirementSpec | None = None
    actual_figure: FigureTrace | None = None
    requirement_plan: ChartRequirementPlan | None = None
    evidence_graph: EvidenceGraph | None = None
    repair_plan: RepairPlan | None = None
    repair: RepairPatch | None = None
    repaired_code: str | None = None
    repair_attempts: tuple[RepairAttempt, ...] = ()
    repair_loop_status: str | None = None
    repair_loop_reason: str | None = None
    execution_exception_type: str | None = None
    execution_exception_message: str | None = None
    parse_source: str = "predicted"
    parser_raw_response: dict[str, Any] = field(default_factory=dict)
