from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutorFidelityIssue:
    code: str
    message: str
    severity: str = "error"
    plan_ref: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutorFidelityReport:
    ok: bool
    issues: tuple[ExecutorFidelityIssue, ...]
    checked_contracts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "checked_contracts": list(self.checked_contracts),
        }


SYNTHETIC_DATA_PATTERNS = (
    ("np.random", re.compile(r"\bnp\.random\b")),
    ("random_module", re.compile(r"\brandom\.")),
    ("dummy_data", re.compile(r"dummy\s+data", re.IGNORECASE)),
    ("sample_data", re.compile(r"sample\s+data", re.IGNORECASE)),
    ("synthetic_data", re.compile(r"synthetic\s+data", re.IGNORECASE)),
    ("create_data", re.compile(r"\b(create|generate)\s+(sample|dummy|synthetic)?\s*data", re.IGNORECASE)),
)

READ_FILE_RE = re.compile(r"\b(?:pd\.)?read_(?:csv|table|excel|json)\s*\(\s*['\"]([^'\"]+)['\"]")


class ExecutorFidelityValidator:
    """Static contract checker for the plan-to-code executor stage."""

    def validate(self, code: str, *, context: dict[str, Any]) -> ExecutorFidelityReport:
        code_text = str(code or "")
        issues: list[ExecutorFidelityIssue] = []
        checked: list[str] = []
        source_plan = _dict(context.get("source_data_plan"))
        construction_plan = _dict(context.get("construction_plan"))
        artifact_workspace = _dict(context.get("artifact_workspace"))

        if source_plan:
            checked.append("source_data_plan")
            issues.extend(_validate_source_contract(code_text, source_plan, artifact_workspace=artifact_workspace))
        if construction_plan:
            checked.append("construction_plan")
            issues.extend(_validate_construction_contract(code_text, construction_plan))
            issues.extend(_validate_overlay_coordinate_contract(code_text, construction_plan))
            issues.extend(_validate_computed_layout_record_contract(code_text, construction_plan))
        if artifact_workspace:
            checked.append("artifact_workspace")
            issues.extend(_validate_artifact_workspace_contract(code_text, artifact_workspace))
            issues.extend(_validate_chart_protocol_contract(code_text, artifact_workspace))
            issues.extend(_validate_area_modifier_contract(code_text, artifact_workspace))
            issues.extend(_validate_visual_channel_decision_record_contract(code_text, artifact_workspace))
        checked.append("output_contract")
        if "OUTPUT_PATH" not in code_text:
            issues.append(
                ExecutorFidelityIssue(
                    code="missing_output_path_contract",
                    message="Executor code must save the final artifact to OUTPUT_PATH.",
                    plan_ref="output_contract",
                )
            )

        blocking = [issue for issue in issues if issue.severity == "error"]
        return ExecutorFidelityReport(ok=not blocking, issues=tuple(issues), checked_contracts=tuple(checked))


def validate_executor_fidelity(code: str, *, context: dict[str, Any]) -> ExecutorFidelityReport:
    return ExecutorFidelityValidator().validate(code, context=context)


def _validate_source_contract(
    code: str,
    source_plan: dict[str, Any],
    *,
    artifact_workspace: dict[str, Any] | None = None,
) -> list[ExecutorFidelityIssue]:
    issues: list[ExecutorFidelityIssue] = []
    files = [str(item.get("name") or "") for item in list(source_plan.get("files") or []) if isinstance(item, dict)]
    mentioned = [str(item) for item in list(source_plan.get("mentioned_files") or [])]
    required = mentioned or files
    used_files = set(_normalize_filename(match.group(1)) for match in READ_FILE_RE.finditer(code))
    prepared_artifacts = _prepared_artifact_names(artifact_workspace or {})
    uses_prepared_artifacts = any(name in used_files or name in code for name in prepared_artifacts)
    for name in required:
        normalized = _normalize_filename(name)
        if uses_prepared_artifacts:
            continue
        if normalized and normalized not in used_files and normalized not in code:
            issues.append(
                ExecutorFidelityIssue(
                    code="missing_required_source_file",
                    message=f"Executor code does not read required source file: {normalized}.",
                    plan_ref=f"source_data_plan.files.{normalized}",
                )
            )
    if required:
        for label, pattern in SYNTHETIC_DATA_PATTERNS:
            match = pattern.search(code)
            if match:
                issues.append(
                    ExecutorFidelityIssue(
                        code="synthetic_data_with_required_source",
                        message="Executor code uses synthetic/random data despite required source files.",
                        plan_ref="source_data_plan",
                        evidence=f"{label}: {match.group(0)}",
                    )
                )
    planned = {_normalize_filename(name) for name in files} | prepared_artifacts
    for used in used_files:
        if planned and used not in planned:
            issues.append(
                ExecutorFidelityIssue(
                    code="unplanned_source_file_read",
                    message=f"Executor code reads an unplanned source file: {used}.",
                    severity="warning",
                    plan_ref="source_data_plan",
                )
            )
    return issues


def _validate_construction_contract(code: str, construction_plan: dict[str, Any]) -> list[ExecutorFidelityIssue]:
    issues: list[ExecutorFidelityIssue] = []
    panels = [item for item in list(construction_plan.get("panels") or []) if isinstance(item, dict)]
    layers = []
    for panel in panels:
        for layer in list(panel.get("layers") or []):
            if isinstance(layer, dict):
                layers.append(layer)
    for layer in layers:
        chart_type = str(layer.get("chart_type") or "").lower()
        layer_id = str(layer.get("layer_id") or chart_type or "layer")
        status = str(layer.get("status") or "explicit")
        if chart_type and not _code_has_chart_type(code, chart_type):
            issues.append(
                ExecutorFidelityIssue(
                    code="missing_planned_visual_layer",
                    message=f"Executor code does not implement planned visual layer `{layer_id}` ({chart_type}).",
                    severity="error" if status == "explicit" else "warning",
                    plan_ref=f"construction_plan.layers.{layer_id}",
                )
            )
    if _requires_secondary_axis(panels) and not _has_secondary_axis(code):
        issues.append(
            ExecutorFidelityIssue(
                code="missing_secondary_axis",
                message="Construction plan requires a secondary axis, but executor code does not create one.",
                plan_ref="construction_plan.axes.secondary_y",
            )
        )
    for element in list(construction_plan.get("global_elements") or []):
        if not isinstance(element, dict):
            continue
        element_type = str(element.get("type") or "").lower()
        status = str(element.get("status") or "explicit")
        if element_type == "legend" and status == "explicit" and "legend(" not in code:
            issues.append(
                ExecutorFidelityIssue(
                    code="missing_explicit_legend",
                    message="Construction plan requires a legend, but executor code does not create one.",
                    plan_ref="construction_plan.global_elements.legend",
                )
            )
        if element_type == "title":
            title = str(element.get("text") or "")
            if title and title not in code:
                issues.append(
                    ExecutorFidelityIssue(
                        code="missing_explicit_title_text",
                        message=f"Construction plan requires title text `{title}`, but it is absent from code.",
                        plan_ref="construction_plan.global_elements.title",
                    )
                )
    return issues


def _validate_overlay_coordinate_contract(code: str, construction_plan: dict[str, Any]) -> list[ExecutorFidelityIssue]:
    issues: list[ExecutorFidelityIssue] = []
    panels = [item for item in list(construction_plan.get("panels") or []) if isinstance(item, dict)]
    if not _requires_shared_x_contract(panels):
        return issues
    if _mixes_index_and_raw_year_x(code):
        issues.append(
            ExecutorFidelityIssue(
                code="mixed_overlay_x_coordinate_basis",
                message=(
                    "Executor code appears to plot overlaid/twinx layers with mixed x-coordinate bases "
                    "(index positions for one layer and raw Year values for another)."
                ),
                plan_ref="construction_plan.panels.panel.main.placement_policy.shared_x_coordinate_system",
            )
        )
    if "add_axes(" in code and "tight_layout(" in code:
        issues.append(
            ExecutorFidelityIssue(
                code="manual_axes_with_tight_layout",
                message="Executor code uses manual axes/insets together with tight_layout, which can distort a planned layout.",
                severity="warning",
                plan_ref="construction_plan.layout",
            )
        )
    return issues


def _validate_computed_layout_record_contract(code: str, construction_plan: dict[str, Any]) -> list[ExecutorFidelityIssue]:
    if not _requires_executor_computed_layout(construction_plan):
        return []
    has_json_record = "computed_layout.json" in code
    has_md_record = "layout_decisions.md" in code
    issues: list[ExecutorFidelityIssue] = []
    if not has_json_record:
        issues.append(
            ExecutorFidelityIssue(
                code="missing_computed_layout_json",
                message=(
                    "Construction plan delegates concrete layout coordinates to ExecutorAgent, "
                    "but executor code does not write computed_layout.json."
                ),
                severity="warning",
                plan_ref="construction_plan.layout",
            )
        )
    if not has_md_record:
        issues.append(
            ExecutorFidelityIssue(
                code="missing_layout_decisions_md",
                message=(
                    "Construction plan delegates concrete layout coordinates to ExecutorAgent, "
                    "but executor code does not write layout_decisions.md."
                ),
                severity="warning",
                plan_ref="construction_plan.layout",
            )
        )
    return issues


def _requires_executor_computed_layout(construction_plan: dict[str, Any]) -> bool:
    panels = [item for item in list(construction_plan.get("panels") or []) if isinstance(item, dict)]
    if not panels:
        return False
    semantic_layout = bool(str(construction_plan.get("layout_strategy") or "").strip())
    for panel in panels:
        placement_policy = panel.get("placement_policy")
        anchor = panel.get("anchor")
        avoid = panel.get("avoid_occlusion")
        if (isinstance(placement_policy, dict) and placement_policy) or (isinstance(anchor, dict) and anchor) or avoid:
            semantic_layout = True
        if panel.get("bounds") is None and semantic_layout:
            return True
    return False


def _validate_artifact_workspace_contract(code: str, artifact_workspace: dict[str, Any]) -> list[ExecutorFidelityIssue]:
    issues: list[ExecutorFidelityIssue] = []
    plotting_artifacts = _prepared_plotting_artifacts(artifact_workspace)
    prepared_csvs = {str(item.get("name") or "") for item in plotting_artifacts}
    for artifact in plotting_artifacts:
        name = str(artifact.get("name") or "")
        escaped = re.escape(name)
        writes_artifact = bool(
            re.search(rf"\.to_csv\s*\(\s*['\"][^'\"]*{escaped}['\"]", code)
            or re.search(rf"\bopen\s*\(\s*['\"][^'\"]*{escaped}['\"][^)]*['\"]w", code)
            or re.search(rf"Path\s*\([^)]*{escaped}[^)]*\)\.write_", code)
        )
        if writes_artifact:
            issues.append(
                ExecutorFidelityIssue(
                    code="prepared_artifact_overwrite",
                    message=f"Executor code overwrites framework-prepared artifact `{name}` instead of treating it as read-only.",
                    plan_ref=f"artifact_workspace.artifacts.{name}",
                )
            )
    for artifact in plotting_artifacts:
        name = str(artifact.get("name") or "")
        if not _artifact_or_alias_is_read(code, artifact_workspace, artifact):
            issues.append(
                ExecutorFidelityIssue(
                    code="prepared_artifact_not_read",
                    message=f"Executor code does not read prepared plotting artifact `{name}`.",
                    severity="error",
                    plan_ref=f"artifact_workspace.artifacts.{name}",
                )
            )
        elif not _uses_artifact_or_alias_relative_path(code, artifact_workspace, artifact):
            issues.append(
                ExecutorFidelityIssue(
                    code="prepared_artifact_bare_filename_read",
                    message=f"Executor code mentions prepared artifact `{name}` but not its artifact_workspace relative path.",
                    severity="error",
                    plan_ref=f"artifact_workspace.artifacts.{name}.relative_path",
                )
            )
    return issues


def _validate_chart_protocol_contract(code: str, artifact_workspace: dict[str, Any]) -> list[ExecutorFidelityIssue]:
    issues: list[ExecutorFidelityIssue] = []
    protocols = _chart_protocols(artifact_workspace)
    waterfall_artifacts = _artifacts_by_role(artifact_workspace, "waterfall_geometry")
    has_waterfall_protocol = "waterfall" in protocols or any("waterfall_protocol" in name for name in _artifact_names(artifact_workspace))
    if not (has_waterfall_protocol or waterfall_artifacts):
        return issues
    for artifact in waterfall_artifacts:
        name = str(artifact.get("name") or "")
        if name and not _artifact_or_alias_is_read(code, artifact_workspace, artifact):
            issues.append(
                ExecutorFidelityIssue(
                    code="waterfall_render_table_not_read",
                    message=f"Waterfall protocol requires reading prepared geometry artifact `{name}`.",
                    plan_ref=f"artifact_workspace.artifacts.{name}",
                )
            )
    if waterfall_artifacts and not _uses_waterfall_bottom_height(code):
        issues.append(
            ExecutorFidelityIssue(
                code="waterfall_render_protocol_not_used",
                message="Waterfall protocol requires plotting bars with bottom=bar_bottom and height=bar_height from the prepared geometry artifact.",
                plan_ref="artifact_workspace.chart_protocols.waterfall",
            )
        )
    source_artifact_names = [
        str(item.get("name") or "")
        for item in _artifacts_by_role(artifact_workspace, "source_values")
        if str(item.get("chart_type") or "").lower() in {"", "waterfall"}
    ]
    if waterfall_artifacts and _uses_values_table_as_zero_based_waterfall(code, source_artifact_names=source_artifact_names):
        issues.append(
            ExecutorFidelityIssue(
                code="waterfall_values_table_used_as_ordinary_bars",
                message="Executor appears to draw waterfall source values as ordinary zero-based bars instead of using protocol geometry.",
                plan_ref="artifact_workspace.artifacts.waterfall_geometry",
            )
        )
    if waterfall_artifacts and _uses_legacy_waterfall_change_color_mapping(code):
        issues.append(
            ExecutorFidelityIssue(
                code="waterfall_visual_channel_policy_bypass",
                message=(
                    "Executor appears to hard-code waterfall change-direction colors while the protocol/artifact "
                    "may assign fill color to a different field such as fill_color_role or series."
                ),
                plan_ref="artifact_workspace.chart_protocols.waterfall.visual_channel_policy",
                severity="warning",
            )
        )
    return issues


def _validate_area_modifier_contract(code: str, artifact_workspace: dict[str, Any]) -> list[ExecutorFidelityIssue]:
    issues: list[ExecutorFidelityIssue] = []
    area_artifacts = _artifacts_by_role(artifact_workspace, "area_fill_geometry")
    if not area_artifacts:
        return issues
    has_area_protocol = "area" in _chart_protocols(artifact_workspace) or any("area_protocol" in name for name in _artifact_names(artifact_workspace))
    if not has_area_protocol:
        return issues
    mentioned_area_artifact = any(_artifact_or_alias_is_read(code, artifact_workspace, item) for item in area_artifacts)
    if mentioned_area_artifact and not _uses_area_fill_columns(code, area_artifacts=area_artifacts):
        issues.append(
            ExecutorFidelityIssue(
                code="area_fill_geometry_not_used",
                message="Area protocol requires plotting from *_fill_bottom and *_fill_top columns, not hidden recomputation.",
                plan_ref="artifact_workspace.chart_protocols.area",
            )
        )
    if _uses_additive_area_without_policy(code):
        issues.append(
            ExecutorFidelityIssue(
                code="area_additive_stack_without_policy_check",
                message="Executor appears to use hard-coded additive stacking without checking composition_policy from the prepared artifact.",
                plan_ref="artifact_workspace.artifacts.area_fill_geometry",
                severity="warning",
            )
        )
    return issues


def _validate_visual_channel_decision_record_contract(code: str, artifact_workspace: dict[str, Any]) -> list[ExecutorFidelityIssue]:
    hard_contracts = _hard_visual_channel_contracts(artifact_workspace)
    if not hard_contracts:
        return []
    if "visual_channel_decisions.json" in code:
        return []
    channels = {str(item.get("channel") or "") for item in hard_contracts}
    channel_tokens = {
        "fill_color": ("color=", "c=", "facecolor=", "fill_color", "palette"),
        "line_color": ("color=", "line_color", "palette"),
        "marker_shape": ("marker=", "marker_shape"),
        "hatch": ("hatch=",),
        "edge_style": ("edgecolor=", "linestyle=", "edge_style"),
        "alpha": ("alpha=",),
        "x_offset": ("x_offset", "x_position", "+ row", "+ offset"),
    }
    lower = str(code or "").lower()
    likely_implements_contract = any(
        any(token.lower() in lower for token in channel_tokens.get(channel, (channel,)))
        for channel in channels
        if channel
    )
    if not likely_implements_contract:
        return []
    return [
        ExecutorFidelityIssue(
            code="missing_visual_channel_decisions_record",
            message=(
                "Hard visual-channel contracts are present, but executor code does not record "
                "its actual semantic channel mappings in visual_channel_decisions.json."
            ),
            severity="warning",
            plan_ref="artifact_workspace.chart_protocols.visual_channel_contracts",
        )
    ]


def _prepared_artifact_names(artifact_workspace: dict[str, Any]) -> set[str]:
    artifact_names = {
        str(item.get("name") or "")
        for item in list(artifact_workspace.get("artifacts") or [])
        if isinstance(item, dict)
    }
    return {name for name in artifact_names if name.endswith(".csv")}


def _prepared_plotting_artifacts(artifact_workspace: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for item in list(artifact_workspace.get("artifacts") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name.endswith(".csv"):
            continue
        if item.get("required_for_plotting") is False:
            continue
        if item.get("legacy_alias"):
            continue
        if not _is_hard_fidelity_contract(item):
            continue
        if not item.get("artifact_role") and not name.startswith("step_"):
            continue
        artifacts.append(item)
    return artifacts


def _artifacts_by_role(artifact_workspace: dict[str, Any], role: str) -> list[dict[str, Any]]:
    result = []
    for item in list(artifact_workspace.get("artifacts") or []):
        if not isinstance(item, dict):
            continue
        if item.get("legacy_alias"):
            continue
        if not _is_hard_fidelity_contract(item):
            continue
        if str(item.get("artifact_role") or "") == role:
            result.append(item)
            continue
        name = str(item.get("name") or "").lower()
        if role == "waterfall_geometry" and ("waterfall_geometry" in name or "waterfall_render_table" in name):
            result.append(item)
        elif role == "area_fill_geometry" and ("area_fill_geometry" in name or "area_values" in name):
            result.append(item)
        elif role == "source_values" and ("source_values" in name or "waterfall_values" in name):
            result.append(item)
    return result


def _artifact_or_alias_is_read(code: str, artifact_workspace: dict[str, Any], artifact: dict[str, Any]) -> bool:
    for name in _artifact_and_alias_names(artifact_workspace, artifact):
        if name and name in code:
            return True
    return False


def _uses_artifact_or_alias_relative_path(code: str, artifact_workspace: dict[str, Any], artifact: dict[str, Any]) -> bool:
    for name in _artifact_and_alias_names(artifact_workspace, artifact):
        if name and name in code and _uses_artifact_relative_path(code, artifact_workspace, name):
            return True
    return False


def _artifact_and_alias_names(artifact_workspace: dict[str, Any], artifact: dict[str, Any]) -> list[str]:
    name = str(artifact.get("name") or "")
    names = [name] if name else []
    for item in list(artifact_workspace.get("artifacts") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("alias_for") or "") == name:
            alias = str(item.get("name") or "")
            if alias and alias not in names:
                names.append(alias)
    return names


def _uses_artifact_relative_path(code: str, artifact_workspace: dict[str, Any], name: str) -> bool:
    expected = _artifact_relative_path(artifact_workspace, name)
    if not expected:
        return True
    normalized_code = str(code or "").replace("\\", "/")
    if expected in normalized_code:
        return True
    parts = expected.split("/")
    if len(parts) >= 3 and _uses_joined_artifact_path(normalized_code, parts, name):
        return True
    return False


def _uses_joined_artifact_path(code: str, parts: list[str], name: str) -> bool:
    if name not in code:
        return False
    round_id = parts[-2]
    root_dir = parts[-3]
    if (
        "os.path.join" in code
        and re.search(rf"['\"]{re.escape(root_dir)}['\"]", code)
        and re.search(rf"['\"]{re.escape(round_id)}['\"]", code)
    ):
        return True
    if re.search(
        rf"=\s*['\"]{re.escape(root_dir)}/{re.escape(round_id)}['\"]",
        code,
    ) and re.search(rf"os\.path\.join\s*\([^)]*['\"]{re.escape(name)}['\"]", code, re.DOTALL):
        return True
    join_pattern = re.compile(
        rf"os\.path\.join\s*\([^)]*['\"]{re.escape(root_dir)}['\"][^)]*['\"]{re.escape(round_id)}['\"][^)]*\)",
        re.DOTALL,
    )
    if join_pattern.search(code):
        return True
    pathlib_pattern = re.compile(
        rf"/\s*['\"]{re.escape(root_dir)}['\"]\s*/\s*['\"]{re.escape(round_id)}['\"]",
        re.DOTALL,
    )
    return bool(pathlib_pattern.search(code))


def _artifact_relative_path(artifact_workspace: dict[str, Any], name: str) -> str | None:
    for item in list(artifact_workspace.get("artifacts") or []):
        if not isinstance(item, dict) or str(item.get("name") or "") != name:
            continue
        relative = str(item.get("relative_path") or "").replace("\\", "/").strip()
        if relative:
            return relative
        path = str(item.get("path") or "").replace("\\", "/")
        marker = f"/{name}"
        if marker in path:
            for prefix in ("/execution/", "/plan/", "/repair/"):
                index = path.find(prefix)
                if index >= 0:
                    return path[index + 1 :]
    return None


def _artifact_names(artifact_workspace: dict[str, Any]) -> set[str]:
    return {
        str(item.get("name") or "")
        for item in list(artifact_workspace.get("artifacts") or [])
        if isinstance(item, dict)
    }


def _chart_protocols(artifact_workspace: dict[str, Any]) -> set[str]:
    metadata = _dict(artifact_workspace.get("metadata"))
    protocols = []
    for item in list(metadata.get("chart_protocols") or []):
        if isinstance(item, dict):
            protocols.append(str(item.get("chart_type") or "").lower())
    return {item for item in protocols if item}


def _hard_visual_channel_contracts(artifact_workspace: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = _dict(artifact_workspace.get("metadata"))
    contracts: list[dict[str, Any]] = []
    for protocol in list(metadata.get("chart_protocols") or []):
        if not isinstance(protocol, dict):
            continue
        for contract in list(protocol.get("visual_channel_contracts") or []):
            if isinstance(contract, dict) and _is_hard_fidelity_contract(contract):
                contracts.append(contract)
    return contracts


def _is_hard_fidelity_contract(item: dict[str, Any]) -> bool:
    tier = str(item.get("contract_tier") or item.get("tier") or "").strip().lower()
    if tier:
        return tier in {"hard_fidelity", "hard", "fidelity", "required"}
    return str(item.get("strength") or "hard").strip().lower() == "hard"


def _uses_waterfall_bottom_height(code: str) -> bool:
    lower = code.lower()
    has_bottom_arg = bool(re.search(r"\bbar\s*\([^)]*\bbottom\s*=", code, re.DOTALL))
    mentions_bottom_col = "bar_bottom" in lower
    mentions_height_col = "bar_height" in lower
    return has_bottom_arg and mentions_bottom_col and mentions_height_col


def _uses_values_table_as_zero_based_waterfall(code: str, *, source_artifact_names: list[str]) -> bool:
    if not any(name and name in code for name in source_artifact_names):
        return False
    if _uses_waterfall_bottom_height(code):
        return False
    return bool(re.search(r"\bbar\s*\(", code))


def _uses_legacy_waterfall_change_color_mapping(code: str) -> bool:
    lower = str(code or "").lower()
    if "fill_color_role" in lower or "series_color_role" in lower:
        return False
    if "change_role" not in lower and "color_role" not in lower:
        return False
    legacy_terms = ("increase", "decrease", "total", "positive", "negative")
    return sum(1 for term in legacy_terms if term in lower) >= 2


def _uses_area_fill_columns(code: str, *, area_artifacts: list[dict[str, Any]]) -> bool:
    lower = code.lower()
    schema_columns: list[str] = []
    for artifact in area_artifacts:
        schema = artifact.get("schema") if isinstance(artifact.get("schema"), dict) else {}
        for column in list(schema.get("columns") or artifact.get("columns") or []):
            value = str(column)
            if value and value not in schema_columns:
                schema_columns.append(value)
    bottom_cols = [column for column in schema_columns if column.lower().endswith("_fill_bottom")]
    top_cols = [column for column in schema_columns if column.lower().endswith("_fill_top")]
    if bottom_cols and top_cols:
        return any(column.lower() in lower for column in bottom_cols) and any(column.lower() in lower for column in top_cols)
    return "_fill_bottom" in lower and "_fill_top" in lower


def _uses_additive_area_without_policy(code: str) -> bool:
    lower = code.lower()
    if "composition_policy" in lower:
        return False
    return "rural_stack_top" in lower or bool(re.search(r"\burban_vals?\s*\+\s*rural_vals?\b", lower))


def _code_has_chart_type(code: str, chart_type: str) -> bool:
    checks = {
        "waterfall": ("bar(", ".bar(", "barh(", ".barh("),
        "bar": ("bar(", ".bar(", "barh(", ".barh("),
        "area": ("stackplot(", "fill_between(", ".area("),
        "line": ("plot(", ".plot("),
        "pie": ("pie(", ".pie("),
        "scatter": ("scatter(", ".scatter("),
        "heatmap": ("imshow(", "pcolormesh(", "heatmap("),
        "box": ("boxplot(", ".boxplot("),
    }
    return any(token in code for token in checks.get(chart_type, (chart_type,)))


def _requires_secondary_axis(panels: list[dict[str, Any]]) -> bool:
    for panel in panels:
        axes = _dict(panel.get("axes"))
        if axes.get("secondary_y"):
            return True
        for layer in list(panel.get("layers") or []):
            if isinstance(layer, dict) and str(layer.get("axis") or "").lower() == "secondary":
                return True
    return False


def _requires_shared_x_contract(panels: list[dict[str, Any]]) -> bool:
    for panel in panels:
        placement_policy = _dict(panel.get("placement_policy"))
        if placement_policy.get("shared_x_coordinate_system"):
            return True
        axes = _dict(panel.get("axes"))
        if axes.get("secondary_y"):
            return True
        layer_count = 0
        for layer in list(panel.get("layers") or []):
            if isinstance(layer, dict) and str(layer.get("x") or "").lower() in {"year", "date", "time"}:
                layer_count += 1
        if layer_count >= 2:
            return True
    return False


def _mixes_index_and_raw_year_x(code: str) -> bool:
    text = str(code or "")
    index_vars = set()
    raw_year_vars = set()
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*np\.arange\s*\(", text):
        index_vars.add(match.group(1))
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^\n]+)", text):
        name = match.group(1)
        expression = match.group(2)
        if _x_expression_uses_index_basis(expression):
            index_vars.add(name)
        if _x_expression_uses_raw_year_basis(expression):
            raw_year_vars.add(name)
    uses_index = _x_basis_used_as_coordinate(text, index_vars, basis="index")
    uses_raw_year = _x_basis_used_as_coordinate(text, raw_year_vars, basis="raw_year")
    return uses_index and uses_raw_year


def _x_basis_used_as_coordinate(code: str, variables: set[str], *, basis: str) -> bool:
    call_pattern = re.compile(
        r"\b(?:ax\w*|plt)\.(?:bar|barh|plot|fill_between|scatter|stackplot|set_xticks)\s*\((?P<args>[^)]*)\)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in call_pattern.finditer(code):
        first_arg = _first_call_arg(match.group("args"))
        if basis == "index" and (
            _x_expression_uses_index_basis(first_arg) or _expression_mentions_variable(first_arg, variables)
        ):
            return True
        if basis == "raw_year" and (
            _x_expression_uses_raw_year_basis(first_arg) or _expression_mentions_variable(first_arg, variables)
        ):
            return True
    return False


def _first_call_arg(args: str) -> str:
    depth = 0
    quote: str | None = None
    for index, char in enumerate(str(args or "")):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            continue
        if char == "," and depth == 0:
            return args[:index]
    return str(args or "")


def _expression_mentions_variable(expression: str, variables: set[str]) -> bool:
    if not variables:
        return False
    return any(re.search(rf"\b{re.escape(name)}\b", expression) for name in variables)


def _x_expression_uses_index_basis(expression: str) -> bool:
    lower = str(expression or "").lower()
    return "x_position" in lower or "x_index" in lower or "np.arange" in lower


def _x_expression_uses_raw_year_basis(expression: str) -> bool:
    lower = str(expression or "").lower()
    return "x_value" in lower or bool(re.search(r"\[['\"]year['\"]\]|\.year\b|\byears?\b", lower))


def _has_secondary_axis(code: str) -> bool:
    return any(token in code for token in ("twinx(", "secondary_yaxis(", "make_subplots(", "secondary_y=True"))


def _normalize_filename(name: str) -> str:
    return str(name).replace("\\", "/").split("/")[-1].strip()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
