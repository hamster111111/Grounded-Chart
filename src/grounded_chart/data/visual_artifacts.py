from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from grounded_chart.core.schema import FigureTrace, VerificationError


@dataclass(frozen=True)
class ExpectedVisualArtifact:
    """Verifier-facing contract compiled from source-grounded requirements."""

    artifact_id: str
    artifact_type: str
    expected: dict[str, Any]
    locator: dict[str, Any] = field(default_factory=dict)
    source_requirement_id: str | None = None
    criticality: str = "hard"
    match_policy: str = "presence"
    supported_by_verifier: bool = True
    source_span: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActualVisualArtifact:
    """Observable artifact extracted from FigureTrace."""

    artifact_id: str
    artifact_type: str
    value: dict[str, Any]
    locator: dict[str, Any] = field(default_factory=dict)
    source: str = "figure_trace"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_expected_visual_artifacts(artifacts: Iterable[Any]) -> tuple[dict[str, Any], ...]:
    """Compile extracted expected artifacts into verifier contracts.

    This is intentionally conservative. Requirements not represented here should
    remain in provenance metadata rather than silently becoming weak checks.
    """

    contracts: list[ExpectedVisualArtifact] = []
    for index, artifact in enumerate(artifacts):
        if str(getattr(artifact, "status", "accepted")) != "accepted":
            continue
        contract = _contract_from_expected_artifact(artifact, index=index)
        if contract is not None:
            contracts.append(contract)
    return tuple(contract.to_dict() for contract in contracts)


def extract_actual_visual_artifacts(figure: FigureTrace | None) -> tuple[ActualVisualArtifact, ...]:
    if figure is None:
        return ()
    actuals: list[ActualVisualArtifact] = [
        ActualVisualArtifact(
            artifact_id="actual.figure.layout",
            artifact_type="layout",
            value={
                "axes_count": figure.axes_count,
                "orientation": _layout_orientation(figure),
                "bounds": [list(axis.bounds) if axis.bounds is not None else None for axis in figure.axes],
            },
            source=figure.source,
        ),
        ActualVisualArtifact(
            artifact_id="actual.figure.connector",
            artifact_type="connector",
            value=_connector_summary(figure),
            source=figure.source,
        ),
    ]
    for axis in figure.axes:
        actuals.append(
            ActualVisualArtifact(
                artifact_id=f"actual.panel_{axis.index}.chart_type",
                artifact_type="panel_chart_type",
                value={"chart_types": list(_axis_chart_types(axis)), "artist_types": [artist.artist_type for artist in axis.artists]},
                locator={"axis_index": axis.index, "panel_id": f"panel_{axis.index}"},
                source=figure.source,
            )
        )
        actuals.append(
            ActualVisualArtifact(
                artifact_id=f"actual.panel_{axis.index}.text",
                artifact_type="text",
                value={
                    "title": axis.title,
                    "xlabel": axis.xlabel,
                    "ylabel": axis.ylabel,
                    "zlabel": axis.zlabel,
                    "legend_labels": list(axis.legend_labels),
                    "texts": list(axis.texts),
                },
                locator={"axis_index": axis.index, "panel_id": f"panel_{axis.index}"},
                source=figure.source,
            )
        )
    actuals.extend(_code_structure_actuals(figure))
    return tuple(actuals)




def _code_structure_actuals(figure: FigureTrace) -> tuple[ActualVisualArtifact, ...]:
    raw = figure.raw if isinstance(figure.raw, dict) else {}
    items = raw.get("code_structure_artifacts", ())
    actuals: list[ActualVisualArtifact] = []
    if not isinstance(items, (list, tuple)):
        return ()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        artifact_type = str(item.get("artifact_type") or "code_structure").strip() or "code_structure"
        value = item.get("value") if isinstance(item.get("value"), dict) else {}
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
        artifact_id = str(item.get("artifact_id") or f"actual.code_structure.{index}").strip()
        actuals.append(
            ActualVisualArtifact(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                value=dict(value),
                locator=dict(locator),
                source=str(item.get("source") or "code_structure"),
            )
        )
    return tuple(actuals)
def verify_expected_visual_artifacts(
    contracts: Iterable[Any],
    actual_figure: FigureTrace | None,
) -> tuple[VerificationError, ...]:
    expected = tuple(_normalize_contract(contract) for contract in contracts)
    expected = tuple(contract for contract in expected if contract is not None)
    if not expected:
        return ()
    actuals = extract_actual_visual_artifacts(actual_figure)
    errors: list[VerificationError] = []
    for contract in expected:
        if not contract.supported_by_verifier:
            errors.append(_under_verified_error(contract, reason="No verifier support for this expected artifact type."))
            continue
        if actual_figure is None:
            errors.append(
                VerificationError(
                    code="missing_actual_visual_artifact",
                    operator="visual_artifact",
                    message="Expected visual artifact exists, but no actual figure trace was captured.",
                    expected=contract.expected,
                    actual=None,
                    requirement_id=contract.source_requirement_id,
                    severity=_severity_for_contract(contract),
                    match_policy=contract.match_policy,
                )
            )
            continue
        if contract.artifact_type == "layout":
            errors.extend(_verify_layout_contract(contract, actuals))
        elif contract.artifact_type == "panel_chart_type":
            errors.extend(_verify_panel_chart_type_contract(contract, actuals))
        elif contract.artifact_type == "connector":
            errors.extend(_verify_connector_contract(contract, actuals))
        elif contract.artifact_type == "text":
            errors.extend(_verify_text_contract(contract, actuals))
        else:
            errors.append(_under_verified_error(contract, reason="Unknown visual artifact contract."))
    return tuple(errors)


def _contract_from_expected_artifact(artifact: Any, *, index: int) -> ExpectedVisualArtifact | None:
    artifact_type = str(getattr(artifact, "artifact_type", "") or "").strip().lower()
    value = _value_dict(getattr(artifact, "value", {}))
    source_span = str(getattr(artifact, "source_span", "") or "")
    panel_id = getattr(artifact, "panel_id", None)
    chart_type = _normalize_chart_type(value.get("chart_type") or getattr(artifact, "chart_type", None) or value.get("type"))
    axis_index = _axis_index_from_panel_id(panel_id)

    if artifact_type in {"panel_chart_type", "chart_type_requirement"}:
        if chart_type is None:
            return None
        return ExpectedVisualArtifact(
            artifact_id=f"expected.visual.panel_chart_type.{index}",
            artifact_type="panel_chart_type",
            expected={"chart_type": chart_type},
            locator=_locator(panel_id, axis_index),
            source_requirement_id=_chart_type_requirement_id(axis_index),
            match_policy="chart_type_family",
            supported_by_verifier=True,
            source_span=source_span,
        )

    if artifact_type == "visual_relation":
        relation_type = str(value.get("relation_type") or value.get("type") or getattr(artifact, "role", "") or "").strip().lower()
        if not relation_type and _looks_like_connector_span(source_span):
            relation_type = "connector"
        if relation_type in {"connector", "connection", "connecting_line", "arrow"}:
            return ExpectedVisualArtifact(
                artifact_id=f"expected.visual.connector.{index}",
                artifact_type="connector",
                expected={
                    "count": _positive_int(value.get("count")) or _connector_count_from_span(source_span) or 1,
                    "color": _optional_text(value.get("color")),
                    "linewidth": _optional_number(value.get("linewidth") or value.get("line_width")),
                },
                locator=_locator(panel_id, axis_index),
                source_requirement_id="figure.visual_relation.connector",
                match_policy="count_at_least",
                supported_by_verifier=True,
                source_span=source_span,
            )
        return ExpectedVisualArtifact(
            artifact_id=f"expected.visual.unsupported_relation.{index}",
            artifact_type="visual_relation",
            expected={"relation_type": relation_type or "unknown"},
            locator=_locator(panel_id, axis_index),
            source_requirement_id="figure.visual_relation.unsupported",
            supported_by_verifier=False,
            source_span=source_span,
        )

    if artifact_type == "subplot_layout":
        axes_count = _positive_int(value.get("axes_count") or value.get("subplot_count") or value.get("panels"))
        if axes_count is None:
            rows = _positive_int(value.get("rows"))
            cols = _positive_int(value.get("cols") or value.get("columns"))
            if rows is not None and cols is not None:
                axes_count = rows * cols
        orientation = _layout_orientation_from_span(source_span)
        if axes_count is None and orientation is None:
            return None
        return ExpectedVisualArtifact(
            artifact_id=f"expected.visual.layout.{index}",
            artifact_type="layout",
            expected={"axes_count": axes_count, "orientation": orientation},
            locator={"scope": "figure"},
            source_requirement_id="figure.axes_count" if axes_count is not None else "figure.layout",
            match_policy="layout_structure",
            supported_by_verifier=True,
            source_span=source_span,
        )

    return None


def _verify_layout_contract(contract: ExpectedVisualArtifact, actuals: tuple[ActualVisualArtifact, ...]) -> list[VerificationError]:
    layout = _first_actual(actuals, "layout")
    if layout is None:
        return [_missing_error(contract, actual=None)]
    errors: list[VerificationError] = []
    expected_axes_count = contract.expected.get("axes_count")
    actual_axes_count = layout.value.get("axes_count")
    if expected_axes_count is not None and expected_axes_count != actual_axes_count:
        errors.append(
            VerificationError(
                code="wrong_visual_layout",
                operator="visual_layout",
                message="Actual figure layout does not match expected visual artifact layout.",
                expected=contract.expected,
                actual=layout.value,
                requirement_id=contract.source_requirement_id,
                severity=_severity_for_contract(contract),
                match_policy=contract.match_policy,
            )
        )
    expected_orientation = contract.expected.get("orientation")
    actual_orientation = layout.value.get("orientation")
    if expected_orientation and actual_orientation and expected_orientation != actual_orientation:
        errors.append(
            VerificationError(
                code="wrong_visual_layout",
                operator="visual_layout",
                message="Actual panel orientation does not match expected visual artifact layout.",
                expected=contract.expected,
                actual=layout.value,
                requirement_id=contract.source_requirement_id,
                severity=_severity_for_contract(contract),
                match_policy=contract.match_policy,
            )
        )
    return errors


def _verify_panel_chart_type_contract(contract: ExpectedVisualArtifact, actuals: tuple[ActualVisualArtifact, ...]) -> list[VerificationError]:
    expected_chart_type = _normalize_chart_type(contract.expected.get("chart_type"))
    if expected_chart_type is None:
        return []
    actual = _actual_for_locator(actuals, "panel_chart_type", contract.locator)
    if actual is None:
        return [_missing_error(contract, actual=None)]
    actual_types = {_normalize_chart_type(item) for item in actual.value.get("chart_types", [])}
    actual_types.discard(None)
    if expected_chart_type in {"stacked_bar", "grouped_bar"}:
        if "bar" not in actual_types:
            return [_wrong_panel_chart_type_error(contract, actual.value)]
        code_structure = _matching_code_structure(actuals, expected_chart_type, contract.locator)
        if code_structure is not None:
            return []
        return [
            _under_verified_error(
                contract,
                reason=(
                    f"Detected a bar chart, but {expected_chart_type!r} requires static code-structure "
                    "evidence such as stacking baselines or grouped x-offsets."
                ),
                actual=actual.value,
            )
        ]
    if expected_chart_type == "exploded_pie":
        if "pie" not in actual_types:
            return [_wrong_panel_chart_type_error(contract, actual.value)]
        code_structure = _matching_code_structure(actuals, "exploded_pie", contract.locator)
        if code_structure is not None:
            return []
        return [
            _under_verified_error(
                contract,
                reason="Detected a pie chart, but exploded-slice structure is not observable without code evidence.",
                actual=actual.value,
            )
        ]
    if expected_chart_type not in actual_types:
        return [_wrong_panel_chart_type_error(contract, actual.value)]
    return []


def _verify_connector_contract(contract: ExpectedVisualArtifact, actuals: tuple[ActualVisualArtifact, ...]) -> list[VerificationError]:
    actual = _first_actual(actuals, "connector")
    if actual is None:
        return [_missing_error(contract, actual=None)]
    expected_count = _positive_int(contract.expected.get("count")) or 1
    actual_count = _positive_int(actual.value.get("count")) or 0
    if actual_count < expected_count:
        return [
            VerificationError(
                code="missing_visual_relation",
                operator="visual_relation",
                message="Expected connector/relationship artifact is missing or under-counted.",
                expected=contract.expected,
                actual=actual.value,
                requirement_id=contract.source_requirement_id,
                severity=_severity_for_contract(contract),
                match_policy=contract.match_policy,
            )
        ]
    return []


def _verify_text_contract(contract: ExpectedVisualArtifact, actuals: tuple[ActualVisualArtifact, ...]) -> list[VerificationError]:
    expected_text = _optional_text(contract.expected.get("text"))
    if not expected_text:
        return []
    actual = _actual_for_locator(actuals, "text", contract.locator) or _first_actual(actuals, "text")
    if actual is None:
        return [_missing_error(contract, actual=None)]
    values = _flatten_text_values(actual.value)
    if any(expected_text.lower() in value.lower() for value in values):
        return []
    return [
        VerificationError(
            code="missing_visual_text",
            operator="visual_text",
            message="Expected visual text artifact is missing.",
            expected=contract.expected,
            actual=actual.value,
            requirement_id=contract.source_requirement_id,
            severity=_severity_for_contract(contract),
            match_policy=contract.match_policy,
        )
    ]



def _matching_code_structure(
    actuals: tuple[ActualVisualArtifact, ...],
    structure: str,
    expected_locator: dict[str, Any],
) -> ActualVisualArtifact | None:
    for actual in actuals:
        if actual.artifact_type != "code_structure":
            continue
        if str(actual.value.get("structure") or "").strip().lower() != structure:
            continue
        if _locator_is_compatible(actual.locator, expected_locator, actuals):
            return actual
    return None


def _locator_is_compatible(
    actual_locator: dict[str, Any],
    expected_locator: dict[str, Any],
    actuals: tuple[ActualVisualArtifact, ...],
) -> bool:
    expected_axis = _optional_int(expected_locator.get("axis_index"))
    actual_axis = _optional_int(actual_locator.get("axis_index"))
    if expected_axis is None:
        expected_panel = _optional_text(expected_locator.get("panel_id"))
        actual_panel = _optional_text(actual_locator.get("panel_id"))
        return expected_panel is None or actual_panel is None or expected_panel == actual_panel
    if actual_axis == expected_axis:
        return True
    if actual_axis is None:
        axes_count = _actual_axes_count(actuals)
        return axes_count in {None, 0, 1}
    return False


def _actual_axes_count(actuals: tuple[ActualVisualArtifact, ...]) -> int | None:
    layout = _first_actual(actuals, "layout")
    if layout is None:
        return None
    return _optional_int(layout.value.get("axes_count"))


def _connector_summary(figure: FigureTrace) -> dict[str, Any]:
    raw = figure.raw if isinstance(figure.raw, dict) else {}
    figure_line_count = _positive_int(raw.get("figure_line_count")) or 0
    connection_patch_count = _positive_int(raw.get("figure_connection_patch_count")) or 0
    axis_line_count = 0
    for axis in figure.axes:
        for artist in axis.artists:
            if str(artist.artist_type).lower() == "line":
                axis_line_count += int(artist.count or 1)
    count = figure_line_count + connection_patch_count
    return {
        "count": count,
        "figure_line_count": figure_line_count,
        "figure_connection_patch_count": connection_patch_count,
        "axis_line_artist_count": axis_line_count,
        "figure_line_styles": raw.get("figure_line_styles", []),
        "figure_patch_types": raw.get("figure_patch_types", []),
    }


def _axis_chart_types(axis: Any) -> tuple[str, ...]:
    artist_types = {str(artist.artist_type).strip().lower() for artist in axis.artists}
    chart_types: list[str] = []
    for artist_type, chart_type in (
        ("pie", "pie"),
        ("bar", "bar"),
        ("line", "line"),
        ("scatter", "scatter"),
        ("image", "heatmap"),
        ("box", "box"),
        ("errorbar", "errorbar"),
    ):
        if artist_type in artist_types:
            chart_types.append(chart_type)
    return tuple(dict.fromkeys(chart_types))


def _layout_orientation(figure: FigureTrace) -> str | None:
    if len(figure.axes) != 2:
        return None
    first, second = figure.axes
    if first.bounds is None or second.bounds is None:
        return None
    x0, y0, w0, h0 = first.bounds
    x1, y1, w1, h1 = second.bounds
    horizontal_gap = abs((x0 + w0 / 2) - (x1 + w1 / 2))
    vertical_gap = abs((y0 + h0 / 2) - (y1 + h1 / 2))
    if horizontal_gap > vertical_gap:
        return "side_by_side"
    if vertical_gap > horizontal_gap:
        return "stacked_vertical"
    return None


def _normalize_contract(value: Any) -> ExpectedVisualArtifact | None:
    if isinstance(value, ExpectedVisualArtifact):
        return value
    if not isinstance(value, dict):
        return None
    artifact_id = str(value.get("artifact_id") or "").strip()
    artifact_type = str(value.get("artifact_type") or "").strip()
    if not artifact_id or not artifact_type:
        return None
    expected = value.get("expected") if isinstance(value.get("expected"), dict) else {}
    locator = value.get("locator") if isinstance(value.get("locator"), dict) else {}
    return ExpectedVisualArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        expected=dict(expected),
        locator=dict(locator),
        source_requirement_id=_optional_text(value.get("source_requirement_id")),
        criticality=str(value.get("criticality") or "hard"),
        match_policy=str(value.get("match_policy") or "presence"),
        supported_by_verifier=bool(value.get("supported_by_verifier", True)),
        source_span=str(value.get("source_span") or ""),
    )


def _missing_error(contract: ExpectedVisualArtifact, *, actual: Any) -> VerificationError:
    return VerificationError(
        code="missing_actual_visual_artifact",
        operator="visual_artifact",
        message="Expected visual artifact is missing from the actual figure trace.",
        expected=contract.expected,
        actual=actual,
        requirement_id=contract.source_requirement_id,
        severity=_severity_for_contract(contract),
        match_policy=contract.match_policy,
    )


def _wrong_panel_chart_type_error(contract: ExpectedVisualArtifact, actual: Any) -> VerificationError:
    return VerificationError(
        code="wrong_panel_chart_type",
        operator="visual_artifact",
        message="Actual panel chart type does not match expected visual artifact contract.",
        expected=contract.expected,
        actual=actual,
        requirement_id=contract.source_requirement_id,
        severity=_severity_for_contract(contract),
        match_policy=contract.match_policy,
    )


def _under_verified_error(contract: ExpectedVisualArtifact, *, reason: str, actual: Any = None) -> VerificationError:
    return VerificationError(
        code="under_verified_visual_artifact",
        operator="visual_coverage_gate",
        message=reason,
        expected=contract.expected,
        actual=actual,
        requirement_id=contract.source_requirement_id,
        severity=_severity_for_contract(contract),
        match_policy=contract.match_policy,
    )


def _first_actual(actuals: tuple[ActualVisualArtifact, ...], artifact_type: str) -> ActualVisualArtifact | None:
    for actual in actuals:
        if actual.artifact_type == artifact_type:
            return actual
    return None


def _actual_for_locator(actuals: tuple[ActualVisualArtifact, ...], artifact_type: str, locator: dict[str, Any]) -> ActualVisualArtifact | None:
    axis_index = _optional_int(locator.get("axis_index"))
    panel_id = _optional_text(locator.get("panel_id"))
    for actual in actuals:
        if actual.artifact_type != artifact_type:
            continue
        if axis_index is not None and _optional_int(actual.locator.get("axis_index")) == axis_index:
            return actual
        if panel_id is not None and actual.locator.get("panel_id") == panel_id:
            return actual
    return None


def _value_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return {"text": value, "chart_type": value, "type": value}
    return {}


def _locator(panel_id: Any, axis_index: int | None) -> dict[str, Any]:
    locator: dict[str, Any] = {}
    if panel_id is not None:
        locator["panel_id"] = str(panel_id)
    if axis_index is not None:
        locator["axis_index"] = axis_index
    return locator


def _axis_index_from_panel_id(panel_id: Any) -> int | None:
    text = str(panel_id or "").strip().lower()
    if not text:
        return None
    if text.startswith("panel_"):
        return _optional_int(text.split("_", 1)[1])
    return None


def _chart_type_requirement_id(axis_index: int | None) -> str:
    return f"panel_{axis_index if axis_index is not None else 0}.chart_type"


def _normalize_chart_type(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "exploded_pie_chart": "exploded_pie",
        "exploded_pie": "exploded_pie",
        "pie_chart": "pie",
        "bar_chart": "bar",
        "stacked_bar_chart": "stacked_bar",
        "grouped_bar_chart": "grouped_bar",
        "line_chart": "line",
        "scatter_plot": "scatter",
        "scatter_chart": "scatter",
        "heat_map": "heatmap",
    }
    text = aliases.get(text, text)
    for known in ("stacked_bar", "grouped_bar", "exploded_pie", "pie", "bar", "line", "scatter", "area", "heatmap", "box"):
        if text == known:
            return known
    if "stack" in text and "bar" in text:
        return "stacked_bar"
    if "group" in text and "bar" in text:
        return "grouped_bar"
    if "explode" in text and "pie" in text:
        return "exploded_pie"
    if "pie" in text:
        return "pie"
    if "bar" in text:
        return "bar"
    if "line" in text:
        return "line"
    if "scatter" in text:
        return "scatter"
    if "heat" in text:
        return "heatmap"
    return None


def _layout_orientation_from_span(source_span: str) -> str | None:
    text = source_span.lower().replace("-", " ")
    if "side by side" in text or "side-by-side" in text or "horizontal" in text:
        return "side_by_side"
    if "vertical" in text or "stacked subplots" in text:
        return "stacked_vertical"
    return None


def _looks_like_connector_span(source_span: str) -> bool:
    text = source_span.lower()
    return any(token in text for token in ("connect", "connecting line", "connector", "arrow", "line between"))


def _connector_count_from_span(source_span: str) -> int | None:
    text = source_span.lower()
    if any(token in text for token in ("two", "2", "top and bottom")):
        return 2
    if any(token in text for token in ("three", "3")):
        return 3
    if _looks_like_connector_span(source_span):
        return 1
    return None


def _severity_for_contract(contract: ExpectedVisualArtifact) -> str:
    return "error" if str(contract.criticality).lower() in {"hard", "core", "error"} else "warning"


def _flatten_text_values(value: dict[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    for item in value.values():
        if isinstance(item, str) and item.strip():
            texts.append(item.strip())
        elif isinstance(item, (list, tuple)):
            texts.extend(str(text).strip() for text in item if str(text).strip())
    return tuple(texts)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None




