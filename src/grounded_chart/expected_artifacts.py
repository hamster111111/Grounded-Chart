from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from grounded_chart.llm import LLMClient, LLMCompletionTrace, LLMJsonResult
from grounded_chart.schema import AxisRequirementSpec, DataPoint, FigureRequirementSpec, PlotTrace
from grounded_chart.visual_artifacts import compile_expected_visual_artifacts


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LIST_SEP = r"(?:\s*,\s*|\s+and\s+|\s*,\s*and\s*)"
_NUMBER_LIST = rf"{_NUMBER}(?:{_LIST_SEP}{_NUMBER})+"


@dataclass(frozen=True)
class ExpectedArtifactNode:
    """A source-grounded expected artifact candidate extracted from task text."""

    artifact_type: str
    value: Any
    source_span: str
    confidence: float
    panel_id: str | None = None
    chart_type: str | None = None
    role: str | None = None
    source: str = "instruction"
    extractor: str = "unknown"
    status: str = "accepted"
    reason: str = ""


@dataclass(frozen=True)
class StructuredExpectedArtifacts:
    """LLM/rule extracted expected artifacts plus provider trace."""

    artifacts: tuple[ExpectedArtifactNode, ...] = ()
    plot_traces: tuple[PlotTrace, ...] = ()
    figure_requirements: FigureRequirementSpec | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    llm_trace: LLMCompletionTrace | None = None

    @property
    def primary_trace(self) -> PlotTrace | None:
        return self.plot_traces[0] if self.plot_traces else None

@dataclass(frozen=True)
class ExpectedTraceExtraction:
    """Traceable expected plotted-data artifact extracted from task text."""

    trace: PlotTrace
    extractor: str
    matched_text: str
    confidence: float


class ExplicitPointExpectedTraceExtractor:
    """Extract explicit x/y plotted points from natural-language instructions.

    This intentionally handles only high-precision cases where the instruction
    gives both x-values and y-values as concrete numeric lists. Ambiguous data
    transformations should be handled by a stronger parser/LLM extractor, not by
    broad regex guesses.
    """

    extractor_name = "explicit_point_sequence_v1"

    def extract(self, text: str, *, default_chart_type: str = "unknown", source: str = "instruction") -> ExpectedTraceExtraction | None:
        normalized = _normalize_text(text)
        if not normalized:
            return None
        match = _match_xy_values_after_labels(normalized) or _match_lists_before_labels(normalized)
        if match is None:
            return None
        x_values = _parse_number_list(match.group("x"))
        y_values = _parse_number_list(match.group("y"))
        if not x_values or len(x_values) != len(y_values):
            return None
        matched_text = match.group(0).strip()
        chart_type = _infer_chart_type(normalized, match.start(), default_chart_type=default_chart_type)
        trace = PlotTrace(
            chart_type=chart_type,
            points=tuple(DataPoint(x=x, y=y) for x, y in zip(x_values, y_values)),
            source=f"{source}:{self.extractor_name}",
            raw={
                "expected_artifact_extractor": self.extractor_name,
                "matched_text": matched_text,
                "x_values": list(x_values),
                "y_values": list(y_values),
            },
        )
        return ExpectedTraceExtraction(
            trace=trace,
            extractor=self.extractor_name,
            matched_text=matched_text,
            confidence=0.9,
        )



class LLMExpectedArtifactExtractor:
    """LLM-assisted expected artifact extractor with source-span validation.

    The model proposes structured candidates; this class only accepts candidates
    whose `source_span` is grounded in the input instruction. Plot-point traces
    are accepted only when x/y pairs are explicit and well-formed.
    """

    extractor_name = "llm_expected_artifact_v1"

    def __init__(self, client: LLMClient, *, max_tokens: int | None = 2500, min_confidence: float = 0.55) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.min_confidence = min_confidence

    def extract(self, text: str, *, source: str = "instruction", case_id: str | None = None) -> StructuredExpectedArtifacts:
        normalized = _normalize_text(text)
        if not normalized:
            return StructuredExpectedArtifacts()
        result = _complete_json_with_trace(
            self.client,
            system_prompt=_llm_expected_artifact_system_prompt(),
            user_prompt=_llm_expected_artifact_user_prompt(normalized, source=source, case_id=case_id),
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return self._normalize_payload(result.payload, normalized, source=source, trace=result.trace)

    def _normalize_payload(
        self,
        payload: dict[str, Any],
        text: str,
        *,
        source: str,
        trace: LLMCompletionTrace | None,
    ) -> StructuredExpectedArtifacts:
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raw_artifacts = []
        nodes: list[ExpectedArtifactNode] = []
        traces: list[PlotTrace] = []
        for raw in raw_artifacts:
            if not isinstance(raw, dict):
                continue
            node = self._artifact_node(raw, text, source=source)
            if node is None:
                continue
            nodes.append(node)
            if node.artifact_type == "plot_points":
                trace_obj = self._plot_trace_from_node(node)
                if trace_obj is not None:
                    traces.append(trace_obj)
        return StructuredExpectedArtifacts(
            artifacts=tuple(nodes),
            plot_traces=tuple(traces),
            figure_requirements=compile_expected_artifacts_to_figure(nodes),
            raw_response=dict(payload),
            llm_trace=trace,
        )

    def _artifact_node(self, raw: dict[str, Any], text: str, *, source: str) -> ExpectedArtifactNode | None:
        artifact_type = str(raw.get("artifact_type") or raw.get("type") or "").strip().lower()
        if artifact_type not in _ALLOWED_LLM_ARTIFACT_TYPES:
            return None
        source_span = str(raw.get("source_span") or "").strip()
        if not source_span or not _span_appears_in_text(source_span, text):
            return None
        confidence = _normalize_confidence(raw.get("confidence"), fallback=0.0)
        if confidence < self.min_confidence:
            return None
        value = raw.get("value")
        role = _normalize_artifact_role(raw, value if isinstance(value, dict) else None)
        if artifact_type == "plot_points":
            value = _normalize_plot_points_value(raw)
            if not value:
                return None
            role = _normalize_plot_points_role(raw, value)
            value["role"] = role
        return ExpectedArtifactNode(
            artifact_type=artifact_type,
            value=value,
            source_span=source_span,
            confidence=confidence,
            panel_id=str(raw.get("panel_id")) if raw.get("panel_id") is not None else None,
            chart_type=str(raw.get("chart_type")) if raw.get("chart_type") else None,
            role=role,
            source=source,
            extractor=self.extractor_name,
        )

    def _plot_trace_from_node(self, node: ExpectedArtifactNode) -> PlotTrace | None:
        if node.role != "main_data":
            return None
        points_raw = node.value.get("points") if isinstance(node.value, dict) else None
        if not isinstance(points_raw, list) or not points_raw:
            return None
        points: list[DataPoint] = []
        for point in points_raw:
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                return None
            meta = {"source_span": node.source_span, "confidence": node.confidence, "role": node.role}
            if node.panel_id is not None:
                meta["panel_id"] = node.panel_id
            points.append(DataPoint(x=point.get("x"), y=point.get("y"), meta=meta))
        return PlotTrace(
            chart_type=node.chart_type or "unknown",
            points=tuple(points),
            source=f"{node.source}:{self.extractor_name}",
            raw={
                "expected_artifact_extractor": self.extractor_name,
                "source_span": node.source_span,
                "confidence": node.confidence,
                "panel_id": node.panel_id,
                "artifact_type": node.artifact_type,
                "role": node.role,
            },
        )

def extract_expected_trace_from_text(
    text: str,
    *,
    default_chart_type: str = "unknown",
    source: str = "instruction",
) -> PlotTrace | None:
    extraction = ExplicitPointExpectedTraceExtractor().extract(
        text,
        default_chart_type=default_chart_type,
        source=source,
    )
    return extraction.trace if extraction is not None else None


def extract_expected_trace_from_texts(
    texts: Iterable[tuple[str, str]],
    *,
    default_chart_type: str = "unknown",
) -> PlotTrace | None:
    extractor = ExplicitPointExpectedTraceExtractor()
    for source, text in texts:
        extraction = extractor.extract(text, default_chart_type=default_chart_type, source=source)
        if extraction is not None:
            return extraction.trace
    return None


@dataclass
class _AxisRequirementDraft:
    axis_index: int
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    zlabel: str | None = None
    projection: str | None = None
    xscale: str | None = None
    yscale: str | None = None
    zscale: str | None = None
    xtick_labels: list[str] = field(default_factory=list)
    ytick_labels: list[str] = field(default_factory=list)
    ztick_labels: list[str] = field(default_factory=list)
    legend_labels: list[str] = field(default_factory=list)
    artist_types: list[str] = field(default_factory=list)
    text_contains: list[str] = field(default_factory=list)
    provenance: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_spans: dict[str, str] = field(default_factory=dict)

    def set_scalar(self, field_name: str, value: str, source_span: str) -> None:
        if getattr(self, field_name) is not None:
            return
        setattr(self, field_name, value)
        self.provenance[field_name] = (_axis_requirement_id(self.axis_index, field_name),)
        self.source_spans[field_name] = source_span

    def extend_unique(self, field_name: str, values: Iterable[str], source_span: str) -> None:
        target = getattr(self, field_name)
        changed = False
        for value in values:
            normalized = _clean_artifact_text(value)
            if normalized is None or normalized in target:
                continue
            target.append(normalized)
            changed = True
        if changed:
            requirement_id = _axis_requirement_id(self.axis_index, field_name)
            self.provenance[field_name] = (requirement_id,)
            self.source_spans[field_name] = source_span
            if field_name == "legend_labels":
                for value in target:
                    self.provenance[f"legend_label:{value}"] = (requirement_id,)
            if field_name == "text_contains":
                for value in target:
                    self.provenance[f"text:{value}"] = (requirement_id,)
            if field_name == "artist_types":
                for value in target:
                    self.provenance[f"artist_type:{value}"] = (requirement_id,)

    def to_spec(self) -> AxisRequirementSpec | None:
        spec = AxisRequirementSpec(
            axis_index=self.axis_index,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            zlabel=self.zlabel,
            projection=self.projection,
            xscale=self.xscale,
            yscale=self.yscale,
            zscale=self.zscale,
            xtick_labels=tuple(self.xtick_labels),
            ytick_labels=tuple(self.ytick_labels),
            ztick_labels=tuple(self.ztick_labels),
            legend_labels=tuple(self.legend_labels),
            artist_types=tuple(self.artist_types),
            text_contains=tuple(self.text_contains),
            provenance=dict(self.provenance),
            source_spans=dict(self.source_spans),
        )
        if _axis_spec_has_expectations(spec):
            return spec
        return None


def compile_expected_artifacts_to_figure(artifacts: Iterable[ExpectedArtifactNode]) -> FigureRequirementSpec | None:
    """Compile source-grounded non-trace artifacts into verifier-consumable figure requirements.

    The compiler is deliberately conservative: only artifact fields that are
    directly observable in FigureTrace/AxisTrace are materialized. Unsupported
    artifact details remain available as evidence artifacts but do not affect
    verification.
    """

    compiler = _ExpectedArtifactFigureCompiler()
    return compiler.compile(artifacts)


class _ExpectedArtifactFigureCompiler:
    def __init__(self) -> None:
        self.axes_count: int | None = None
        self.figure_title: str | None = None
        self.size_inches: tuple[float, float] | None = None
        self.figure_provenance: dict[str, tuple[str, ...]] = {}
        self.figure_source_spans: dict[str, str] = {}
        self.axes: dict[int, _AxisRequirementDraft] = {}
        self.next_title_axis_index = 0
        self.artifact_contracts: list[dict[str, Any]] = []

    def compile(self, artifacts: Iterable[ExpectedArtifactNode]) -> FigureRequirementSpec | None:
        artifact_tuple = tuple(artifacts)
        self.artifact_contracts.extend(compile_expected_visual_artifacts(artifact_tuple))
        for artifact in artifact_tuple:
            if artifact.status != "accepted":
                continue
            if artifact.artifact_type == "subplot_layout":
                self._apply_subplot_layout(artifact)
            elif artifact.artifact_type == "title_requirement":
                self._apply_title_requirement(artifact)
            elif artifact.artifact_type == "legend_requirement":
                self._apply_legend_requirement(artifact)
            elif artifact.artifact_type == "axis_requirement":
                self._apply_axis_requirement(artifact)
            elif artifact.artifact_type == "annotation_requirement":
                self._apply_annotation_requirement(artifact)

        axes = tuple(
            spec
            for spec in (draft.to_spec() for draft in sorted(self.axes.values(), key=lambda item: item.axis_index))
            if spec is not None
        )
        axes = _normalize_axis_indices(axes, self.axes_count)
        if self.axes_count is None and self.figure_title is None and self.size_inches is None and not axes and not self.artifact_contracts:
            return None
        return FigureRequirementSpec(
            axes_count=self.axes_count,
            figure_title=self.figure_title,
            size_inches=self.size_inches,
            axes=axes,
            provenance=dict(self.figure_provenance),
            source_spans=dict(self.figure_source_spans),
            artifact_contracts=tuple(self.artifact_contracts),
        )

    def _axis(self, axis_index: int) -> _AxisRequirementDraft:
        if axis_index not in self.axes:
            self.axes[axis_index] = _AxisRequirementDraft(axis_index=axis_index)
        return self.axes[axis_index]

    def _apply_subplot_layout(self, artifact: ExpectedArtifactNode) -> None:
        value = _artifact_value_dict(artifact)
        axes_count = _axes_count_from_artifact(value, artifact.source_span)
        if axes_count is not None and self.axes_count is None:
            self.axes_count = axes_count
            self.figure_provenance["axes_count"] = (_figure_requirement_id("axes_count"),)
            self.figure_source_spans["axes_count"] = artifact.source_span
        size_inches = _figure_size_from_artifact(value, artifact.source_span)
        if size_inches is not None and self.size_inches is None:
            self.size_inches = size_inches
            self.figure_provenance["size_inches"] = (_figure_requirement_id("size_inches"),)
            self.figure_source_spans["size_inches"] = artifact.source_span

    def _apply_title_requirement(self, artifact: ExpectedArtifactNode) -> None:
        title = _artifact_text(artifact, ("text", "title", "figure_title", "axis_title"))
        if title is None:
            return
        if not _span_has_title_cue(artifact.source_span):
            return
        if not _span_supports_text_value(artifact.source_span, title):
            return
        axis_index = _axis_index_from_artifact(artifact)
        if axis_index is None:
            axis_index = self.next_title_axis_index
            self.next_title_axis_index += 1
        self._axis(axis_index).set_scalar("title", title, artifact.source_span)
    def _apply_legend_requirement(self, artifact: ExpectedArtifactNode) -> None:
        labels = _artifact_text_list(artifact, ("legend_labels", "labels", "entries", "order"))
        if not labels:
            return
        labels = tuple(label for label in labels if _span_supports_text_value(artifact.source_span, label))
        if not labels:
            return
        axis_index = _axis_index_from_artifact(artifact)
        self._axis(axis_index if axis_index is not None else 0).extend_unique("legend_labels", labels, artifact.source_span)

    def _apply_axis_requirement(self, artifact: ExpectedArtifactNode) -> None:
        value = _artifact_value_dict(artifact)
        axis_index = _axis_index_from_artifact(artifact)
        axis = self._axis(axis_index if axis_index is not None else 0)
        scalar_mappings = (
            ("xlabel", ("xlabel", "x_label", "x_axis_label")),
            ("ylabel", ("ylabel", "y_label", "y_axis_label")),
            ("zlabel", ("zlabel", "z_label", "z_axis_label")),
            ("projection", ("projection",)),
            ("xscale", ("xscale", "x_scale")),
            ("yscale", ("yscale", "y_scale")),
            ("zscale", ("zscale", "z_scale")),
        )
        for field_name, keys in scalar_mappings:
            text = _text_from_mapping(value, keys)
            if text is not None and _span_supports_axis_scalar(field_name, artifact.source_span, text):
                axis.set_scalar(field_name, text, artifact.source_span)
        tick_mappings = (
            ("xtick_labels", ("xtick_labels", "x_tick_labels", "x_ticks")),
            ("ytick_labels", ("ytick_labels", "y_tick_labels", "y_ticks")),
            ("ztick_labels", ("ztick_labels", "z_tick_labels", "z_ticks")),
        )
        for field_name, keys in tick_mappings:
            ticks = _text_list_from_mapping(value, keys)
            ticks = tuple(tick for tick in ticks if _span_supports_text_value(artifact.source_span, tick))
            if ticks:
                axis.extend_unique(field_name, ticks, artifact.source_span)
    def _apply_annotation_requirement(self, artifact: ExpectedArtifactNode) -> None:
        texts = _artifact_text_list(artifact, ("text", "texts", "annotation_text", "annotation_texts"))
        if not texts:
            return
        texts = tuple(text for text in texts if _span_supports_text_value(artifact.source_span, text))
        if not texts:
            return
        axis_index = _axis_index_from_artifact(artifact)
        self._axis(axis_index if axis_index is not None else 0).extend_unique("text_contains", texts, artifact.source_span)


def _span_supports_text_value(source_span: str, value: str) -> bool:
    normalized_span = _normalize_text(str(source_span or "")).lower()
    normalized_value = _normalize_text(str(value or "")).lower()
    return bool(normalized_value) and normalized_value in normalized_span


def _span_has_title_cue(source_span: str) -> bool:
    normalized = _normalize_text(str(source_span or "")).lower()
    return bool(re.search(r"\b(title|titled|titles|titling)\b", normalized))


def _span_supports_axis_scalar(field_name: str, source_span: str, value: str) -> bool:
    normalized_span = _normalize_text(str(source_span or "")).lower()
    normalized_value = _normalize_text(str(value or "")).lower()
    if not normalized_value:
        return False
    if field_name in {"xlabel", "ylabel", "zlabel"}:
        return _span_supports_text_value(source_span, value)
    if field_name == "projection" and normalized_value in {"3d", "3-d"}:
        return any(cue in normalized_span for cue in ("3d", "3-d", "three-dimensional"))
    if field_name in {"xscale", "yscale", "zscale"}:
        if normalized_value == "log":
            return "log" in normalized_span or "logarithmic" in normalized_span
        return normalized_value in normalized_span
    return _span_supports_text_value(source_span, value)

def _figure_requirement_id(field_name: str) -> str:
    if field_name == "figure_title":
        return "figure.title"
    return f"figure.{field_name}"


def _axis_requirement_id(axis_index: int, field_name: str) -> str:
    return f"panel_0.axis_{axis_index}.{field_name}"


def _normalize_axis_indices(axes: tuple[AxisRequirementSpec, ...], axes_count: int | None) -> tuple[AxisRequirementSpec, ...]:
    if axes_count is None or not axes:
        return axes
    indices = tuple(axis.axis_index for axis in axes)
    if 0 in indices:
        return axes
    if min(indices) < 1 or max(indices) > axes_count:
        return axes
    return tuple(_shift_axis_requirement_spec(axis, axis.axis_index - 1) for axis in axes)


def _shift_axis_requirement_spec(axis: AxisRequirementSpec, new_index: int) -> AxisRequirementSpec:
    old_fragment = f".axis_{axis.axis_index}."
    new_fragment = f".axis_{new_index}."
    provenance = {
        key: tuple(item.replace(old_fragment, new_fragment) for item in value)
        for key, value in axis.provenance.items()
    }
    return AxisRequirementSpec(
        axis_index=new_index,
        title=axis.title,
        xlabel=axis.xlabel,
        ylabel=axis.ylabel,
        zlabel=axis.zlabel,
        projection=axis.projection,
        xscale=axis.xscale,
        yscale=axis.yscale,
        zscale=axis.zscale,
        xtick_labels=axis.xtick_labels,
        ytick_labels=axis.ytick_labels,
        ztick_labels=axis.ztick_labels,
        bounds=axis.bounds,
        legend_labels=axis.legend_labels,
        artist_types=axis.artist_types,
        artist_counts=dict(axis.artist_counts),
        min_artist_counts=dict(axis.min_artist_counts),
        text_contains=axis.text_contains,
        provenance=provenance,
        source_spans=dict(axis.source_spans),
    )

def _artifact_value_dict(artifact: ExpectedArtifactNode) -> dict[str, Any]:
    return artifact.value if isinstance(artifact.value, dict) else {}


def _artifact_text(artifact: ExpectedArtifactNode, keys: tuple[str, ...]) -> str | None:
    value = _artifact_value_dict(artifact)
    text = _text_from_mapping(value, keys)
    if text is not None:
        return text
    if isinstance(artifact.value, str):
        return _clean_artifact_text(artifact.value)
    return None


def _artifact_text_list(artifact: ExpectedArtifactNode, keys: tuple[str, ...]) -> tuple[str, ...]:
    value = _artifact_value_dict(artifact)
    return _text_list_from_mapping(value, keys)


def _text_from_mapping(value: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in value:
            text = _clean_artifact_text(value.get(key))
            if text is not None:
                return text
    return None


def _text_list_from_mapping(value: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    for key in keys:
        if key not in value:
            continue
        texts = _clean_artifact_text_list(value.get(key))
        if texts:
            return texts
    return ()


def _clean_artifact_text(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _clean_artifact_text_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        cleaned = tuple(
            text
            for text in (_clean_artifact_text(item) for item in value)
            if text is not None
        )
        return tuple(dict.fromkeys(cleaned))
    text = _clean_artifact_text(value)
    return (text,) if text is not None else ()


def _axis_index_from_artifact(artifact: ExpectedArtifactNode) -> int | None:
    value = _artifact_value_dict(artifact)
    for raw in (artifact.panel_id, value.get("panel_id")):
        parsed = _parse_index(raw, one_based=False)
        if parsed is not None:
            return parsed
    for raw in (value.get("axis_index"), value.get("panel_index")):
        parsed = _parse_index(raw, one_based=True)
        if parsed is not None:
            return parsed
    return None


def _parse_index(value: Any, *, one_based: bool = False) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value < 0:
            return None
        if one_based and value > 0:
            return value - 1
        return value
    text = str(value).strip().lower()
    match = re.search(r"(?:panel|axis|subplot)[_\s-]*(\d+)", text)
    if match:
        parsed = int(match.group(1))
        return max(parsed - 1, 0) if one_based and "_" not in text else parsed
    ordinal_match = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", text)
    if ordinal_match:
        ordinal_map = {
            "first": 0,
            "second": 1,
            "third": 2,
            "fourth": 3,
            "fifth": 4,
            "sixth": 5,
            "seventh": 6,
            "eighth": 7,
            "ninth": 8,
            "tenth": 9,
        }
        return ordinal_map[ordinal_match.group(1)]
    if text.isdigit():
        parsed = int(text)
        return max(parsed - 1, 0) if one_based and parsed > 0 else parsed
    return None

def _axes_count_from_artifact(value: dict[str, Any], source_span: str) -> int | None:
    direct = _positive_int(value.get("axes_count") or value.get("subplot_count"))
    if direct is not None and _span_has_subplot_count_cue(source_span):
        return direct
    for row_key, col_key in (("nrows", "ncols"), ("rows", "columns"), ("row_count", "column_count")):
        rows = _positive_int(value.get(row_key))
        cols = _positive_int(value.get(col_key))
        if rows is not None and cols is not None and _span_has_layout_grid_cue(source_span):
            return rows * cols
    for key in ("mosaic", "layout"):
        count = _axes_count_from_layout_value(value.get(key))
        if count is not None and _span_has_layout_grid_cue(source_span):
            return count
    return _axes_count_from_layout_text(source_span)


def _axes_count_from_layout_value(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        labels: list[str] = []
        for row in value:
            if isinstance(row, (list, tuple)):
                for item in row:
                    label = _clean_artifact_text(item)
                    if label and label not in {".", "none", "null"}:
                        labels.append(label)
            else:
                label = _clean_artifact_text(row)
                if label and label not in {".", "none", "null"}:
                    labels.append(label)
        return len(set(labels)) if labels else None
    if isinstance(value, str):
        return _axes_count_from_layout_text(value)
    return None


def _axes_count_from_layout_text(text: str) -> int | None:
    normalized = str(text or "").lower()
    count_match = re.search(r"\b(\d+)\s+(?:subplots|plots|panels|sections|axes)\b", normalized)
    if count_match:
        return int(count_match.group(1))
    word_counts = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    for word, count in word_counts.items():
        if re.search(rf"\b{word}\s+(?:subplots|plots|panels|sections|axes)\b", normalized):
            return count
    if not _span_has_layout_grid_cue(normalized):
        return None
    match = re.search(r"(\d+)\s*(?:x|by)\s*(\d+)\s*(?:grid|layout|mosaic|subplots|plots|panels|sections|axes)", normalized)
    if not match:
        match = re.search(r"(?:grid|layout|mosaic|subplots|plots|panels|sections|axes)[^\d]{0,32}(\d+)\s*(?:x|by)\s*(\d+)", normalized)
    if match:
        rows = _positive_int(match.group(1))
        cols = _positive_int(match.group(2))
        if rows is not None and cols is not None:
            return rows * cols
    if "side by side" in normalized or "two sections" in normalized or "two subplots" in normalized:
        return 2
    return None


def _figure_size_from_artifact(value: dict[str, Any], source_span: str) -> tuple[float, float] | None:
    if not _span_has_figure_size_cue(source_span):
        return None
    for key in ("figure_size", "figsize", "size_inches"):
        size = _number_pair(value.get(key))
        if size is not None:
            return size
    normalized = str(source_span or "").lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)", normalized)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))


def _span_has_subplot_count_cue(source_span: str) -> bool:
    normalized = str(source_span or "").lower()
    return bool(re.search(r"\b\d+\s+(?:subplots|plots|panels|sections|axes)\b", normalized)) or any(
        phrase in normalized for phrase in ("side by side", "subplots", "panels", "sections", "axes", "grid", "layout", "mosaic")
    )


def _span_has_layout_grid_cue(source_span: str) -> bool:
    normalized = str(source_span or "").lower()
    return any(
        phrase in normalized
        for phrase in (
            "grid",
            "layout",
            "mosaic",
            "arranged",
            "subplots",
            "plots",
            "panels",
            "sections",
            "axes",
            "rows",
            "columns",
            "side by side",
        )
    )


def _span_has_figure_size_cue(source_span: str) -> bool:
    normalized = str(source_span or "").lower()
    return any(
        phrase in normalized
        for phrase in (
            "figure size",
            "figsize",
            "size_inches",
            "size of",
            "set the size",
            "with a size",
            "visual space to",
            "sized",
        )
    )

def _number_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)", value.lower())
        if match:
            return (float(match.group(1)), float(match.group(2)))
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _axis_spec_has_expectations(axis: AxisRequirementSpec) -> bool:
    return any(
        (
            axis.title is not None,
            axis.xlabel is not None,
            axis.ylabel is not None,
            axis.zlabel is not None,
            axis.projection is not None,
            axis.xscale is not None,
            axis.yscale is not None,
            axis.zscale is not None,
            axis.xtick_labels,
            axis.ytick_labels,
            axis.ztick_labels,
            axis.legend_labels,
            axis.artist_types,
            axis.text_contains,
        )
    )

_ALLOWED_LLM_ARTIFACT_TYPES = {
    "plot_points",
    "data_file_constraint",
    "data_generation_constraint",
    "subplot_layout",
    "axis_requirement",
    "title_requirement",
    "annotation_requirement",
    "legend_requirement",
    "style_requirement",
    "randomness_constraint",
    "panel_chart_type",
    "chart_type_requirement",
    "visual_relation",
}


def _llm_expected_artifact_system_prompt() -> str:
    return (
        "You extract source-grounded expected artifacts for chart-generation evaluation. "
        "Return JSON only. Do not solve by looking at generated code. "
        "Every artifact must be directly supported by a verbatim source_span from the instruction. "
        "If a requirement is implied but not explicitly stated, either omit it or mark confidence below 0.55. "
        "Allowed artifact_type values: plot_points, data_file_constraint, data_generation_constraint, "
        "subplot_layout, axis_requirement, title_requirement, annotation_requirement, legend_requirement, "
        "style_requirement, randomness_constraint, panel_chart_type, chart_type_requirement, visual_relation. "
        "If the instruction explicitly states x-values and y-values, coordinate pairs, or exact plotted point values, output a plot_points artifact first. "
        "For panel-specific chart types, output panel_chart_type with value {chart_type: pie|bar|stacked_bar|grouped_bar|line|scatter|heatmap}. "
        "For explicit relationships such as connector lines or arrows between panels/marks, output visual_relation with value {relation_type: connector, count: number|null, color: string|null, linewidth: number|null}. "
        "For plot_points, only output points when the instruction explicitly gives x/y or coordinate values. "
        "Each plot_points value must include role: main_data, annotation_marker, reference_marker, or derived_constraint. "
        "Use role=main_data only when the points define the primary plotted data series; highlighted or special marker points should be annotation_requirement or plot_points with role=annotation_marker. "
        "Example plot_points value: {\"role\":\"main_data\",\"points\":[{\"x\":1,\"y\":4},{\"x\":2,\"y\":3}]}. "
        "Use this schema: {\"artifacts\":[{\"artifact_type\":string,\"panel_id\":string|null,"
        "\"chart_type\":string|null,\"value\":object,\"source_span\":string,\"confidence\":number}],"
        "\"notes\":[string]}."
    )


def _llm_expected_artifact_user_prompt(text: str, *, source: str, case_id: str | None) -> str:
    return (
        f"Case id: {case_id or 'unknown'}\n"
        f"Source label: {source}\n\n"
        "Instruction:\n"
        f"{text}\n\n"
        "Return JSON only. Do not include artifacts without verbatim source_span support."
    )


def _complete_json_with_trace(client: LLMClient, **kwargs: Any) -> LLMJsonResult:
    traced = getattr(client, "complete_json_with_trace", None)
    if callable(traced):
        result = traced(**kwargs)
        if isinstance(result, LLMJsonResult):
            return result
        if isinstance(result, dict):
            return LLMJsonResult(payload=dict(result), trace=None)
    return LLMJsonResult(payload=dict(client.complete_json(**kwargs)), trace=None)


def _normalize_plot_points_value(raw: dict[str, Any]) -> dict[str, Any] | None:
    value = raw.get("value") if isinstance(raw.get("value"), dict) else {}
    points = value.get("points") if isinstance(value, dict) else None
    if points is None:
        points = raw.get("points")
    if not isinstance(points, list) or not points:
        x_values = value.get("x_values") if isinstance(value, dict) else raw.get("x_values")
        y_values = value.get("y_values") if isinstance(value, dict) else raw.get("y_values")
        if isinstance(x_values, list) and isinstance(y_values, list) and len(x_values) == len(y_values):
            points = [{"x": x, "y": y} for x, y in zip(x_values, y_values)]
    if not isinstance(points, list) or not points:
        return None
    normalized_points = []
    for point in points:
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            return None
        if not _is_concrete_plot_coordinate(point.get("x")) or not _is_concrete_plot_coordinate(point.get("y")):
            return None
        normalized_points.append({"x": point.get("x"), "y": point.get("y")})
    normalized = dict(value) if isinstance(value, dict) else {}
    normalized["points"] = normalized_points
    normalized["role"] = _normalize_plot_points_role(raw, normalized)
    return normalized


def _is_concrete_plot_coordinate(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


_PLOT_POINT_ROLE_ALIASES = {
    "main": "main_data",
    "primary": "main_data",
    "primary_data": "main_data",
    "primary_trace": "main_data",
    "data": "main_data",
    "main_data": "main_data",
    "annotation": "annotation_marker",
    "annotation_marker": "annotation_marker",
    "highlight": "annotation_marker",
    "highlighted_marker": "annotation_marker",
    "marker": "annotation_marker",
    "reference": "reference_marker",
    "reference_marker": "reference_marker",
    "reference_line": "reference_marker",
    "derived": "derived_constraint",
    "derived_constraint": "derived_constraint",
}


_PLOT_POINT_ROLES = {"main_data", "annotation_marker", "reference_marker", "derived_constraint"}


def _normalize_artifact_role(raw: dict[str, Any], value: dict[str, Any] | None = None) -> str | None:
    role = raw.get("role") or raw.get("trace_role")
    if role is None and value is not None:
        role = value.get("role") or value.get("trace_role")
    if role is None:
        return None
    normalized = str(role).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def _normalize_plot_points_role(raw: dict[str, Any], value: dict[str, Any]) -> str:
    role = _normalize_artifact_role(raw, value)
    if role is None:
        return "unknown"
    normalized = _PLOT_POINT_ROLE_ALIASES.get(role, role)
    return normalized if normalized in _PLOT_POINT_ROLES else "unknown"


def _span_appears_in_text(span: str, text: str) -> bool:
    normalized_span = _normalize_text(span).lower()
    normalized_text = _normalize_text(text).lower()
    return bool(normalized_span) and normalized_span in normalized_text


def _normalize_confidence(value: Any, *, fallback: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return fallback
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))

def _match_xy_values_after_labels(text: str) -> re.Match[str] | None:
    pattern = re.compile(
        rf"x[\s-]*values?\s*(?:are|as|:|=)?\s*(?P<x>{_NUMBER_LIST})"
        rf"(?:(?!\b(?:x|y)[\s-]*values?\b).){{0,160}}?"
        rf"y[\s-]*values?\s*(?:are|as|:|=)?\s*(?P<y>{_NUMBER_LIST})",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return pattern.search(text)


def _match_lists_before_labels(text: str) -> re.Match[str] | None:
    pattern = re.compile(
        rf"(?P<x>{_NUMBER_LIST})\s*(?:for|as)\s*(?:the\s+)?x[\s-]*values?"
        rf"(?:(?!\b(?:x|y)[\s-]*values?\b).){{0,200}}?"
        rf"(?P<y>{_NUMBER_LIST})\s*(?:for|as)\s*(?:the\s+)?y[\s-]*values?",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return pattern.search(text)


def _parse_number_list(raw: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for item in re.findall(_NUMBER, raw):
        number = float(item)
        values.append(int(number) if number.is_integer() else number)
    return tuple(values)


def _infer_chart_type(text: str, match_start: int, *, default_chart_type: str) -> str:
    window = text[max(0, match_start - 240) : match_start + 80].lower()
    for chart_type, patterns in (
        ("bar", ("bar plot", "bar chart", "bar x", "bars")),
        ("scatter", ("scatter plot", "scatter chart", "scatter")),
        ("line", ("line plot", "line chart", "plot line")),
        ("area", ("area plot", "area chart")),
    ):
        if any(pattern in window for pattern in patterns):
            return chart_type
    return default_chart_type


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\u2013", "-").replace("\u2014", "-").split())







