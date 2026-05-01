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
    prepared_csvs = sorted(_prepared_artifact_names(artifact_workspace))
    for name in prepared_csvs:
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
    expected_inputs = [
        "step_03_consumption_area_values.csv",
        "step_04_pie_values.csv",
    ]
    if "step_02_imports_waterfall_render_table.csv" in prepared_csvs:
        expected_inputs.append("step_02_imports_waterfall_render_table.csv")
    elif "step_02_imports_waterfall_values.csv" in prepared_csvs:
        expected_inputs.append("step_02_imports_waterfall_values.csv")
    for name in expected_inputs:
        if name in prepared_csvs and name not in code:
            issues.append(
                ExecutorFidelityIssue(
                    code="prepared_artifact_not_read",
                    message=f"Executor code does not read prepared plotting artifact `{name}`.",
                    severity="error",
                    plan_ref=f"artifact_workspace.artifacts.{name}",
                )
            )
        elif name in prepared_csvs and not _uses_artifact_relative_path(code, artifact_workspace, name):
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
    prepared_csvs = _prepared_artifact_names(artifact_workspace)
    protocols = _chart_protocols(artifact_workspace)
    has_waterfall_protocol = "waterfall" in protocols or any("waterfall_protocol" in name for name in _artifact_names(artifact_workspace))
    has_waterfall_render_table = "step_02_imports_waterfall_render_table.csv" in prepared_csvs
    if not (has_waterfall_protocol or has_waterfall_render_table):
        return issues
    if has_waterfall_render_table and "step_02_imports_waterfall_render_table.csv" not in code:
        issues.append(
            ExecutorFidelityIssue(
                code="waterfall_render_table_not_read",
                message="Waterfall protocol requires reading step_02_imports_waterfall_render_table.csv.",
                plan_ref="artifact_workspace.chart_protocols.waterfall",
            )
        )
    if has_waterfall_render_table and not _uses_waterfall_bottom_height(code):
        issues.append(
            ExecutorFidelityIssue(
                code="waterfall_render_protocol_not_used",
                message="Waterfall protocol requires plotting bars with bottom=bar_bottom and height=bar_height from the render table.",
                plan_ref="artifact_workspace.chart_protocols.waterfall",
            )
        )
    if has_waterfall_render_table and _uses_values_table_as_zero_based_waterfall(code):
        issues.append(
            ExecutorFidelityIssue(
                code="waterfall_values_table_used_as_ordinary_bars",
                message="Executor appears to draw waterfall source values as ordinary zero-based bars instead of using protocol geometry.",
                plan_ref="artifact_workspace.artifacts.step_02_imports_waterfall_render_table.csv",
            )
        )
    if has_waterfall_render_table and _uses_non_protocol_waterfall_color_roles(code):
        issues.append(
            ExecutorFidelityIssue(
                code="waterfall_color_role_enum_mismatch",
                message=(
                    "Waterfall render table uses color_role values increase/decrease/total, "
                    "but executor code appears to branch on positive/negative."
                ),
                plan_ref="artifact_workspace.chart_protocols.waterfall.color_role",
            )
        )
    return issues


def _validate_area_modifier_contract(code: str, artifact_workspace: dict[str, Any]) -> list[ExecutorFidelityIssue]:
    issues: list[ExecutorFidelityIssue] = []
    prepared_csvs = _prepared_artifact_names(artifact_workspace)
    if "step_03_consumption_area_values.csv" not in prepared_csvs:
        return issues
    has_area_protocol = "area" in _chart_protocols(artifact_workspace) or any("area_protocol" in name for name in _artifact_names(artifact_workspace))
    if not has_area_protocol:
        return issues
    if "step_03_consumption_area_values.csv" in code and not _uses_area_fill_columns(code):
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
                message="Executor appears to use Urban + Rural stacking without checking composition_policy from the prepared artifact.",
                plan_ref="artifact_workspace.artifacts.step_03_consumption_area_values.csv",
                severity="warning",
            )
        )
    return issues


def _prepared_artifact_names(artifact_workspace: dict[str, Any]) -> set[str]:
    artifact_names = {
        str(item.get("name") or "")
        for item in list(artifact_workspace.get("artifacts") or [])
        if isinstance(item, dict)
    }
    return {name for name in artifact_names if name.startswith("step_") and name.endswith(".csv")}


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


def _uses_waterfall_bottom_height(code: str) -> bool:
    lower = code.lower()
    has_bottom_arg = bool(re.search(r"\bbar\s*\([^)]*\bbottom\s*=", code, re.DOTALL))
    mentions_bottom_col = "bar_bottom" in lower
    mentions_height_col = "bar_height" in lower
    return has_bottom_arg and mentions_bottom_col and mentions_height_col


def _uses_values_table_as_zero_based_waterfall(code: str) -> bool:
    if "step_02_imports_waterfall_values.csv" not in code:
        return False
    if "step_02_imports_waterfall_render_table.csv" in code and _uses_waterfall_bottom_height(code):
        return False
    return bool(re.search(r"\bbar\s*\(", code))


def _uses_non_protocol_waterfall_color_roles(code: str) -> bool:
    lower = str(code or "").lower()
    if "color_role" not in lower:
        return False
    if "positive" not in lower and "negative" not in lower:
        return False
    return not ("increase" in lower and "decrease" in lower)


def _uses_area_fill_columns(code: str) -> bool:
    lower = code.lower()
    return all(name.lower() in lower for name in ("Urban_fill_bottom", "Urban_fill_top", "Rural_fill_bottom", "Rural_fill_top"))


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
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*np\.arange\s*\(", text):
        index_vars.add(match.group(1))
    if not index_vars:
        return False
    uses_index = any(re.search(rf"\b(?:ax\w*|plt)\.bar\s*\(\s*{re.escape(name)}\b", text) for name in index_vars)
    uses_year_column = bool(
        re.search(r"\b(?:ax\w*|plt)\.(?:fill_between|plot|scatter)\s*\([^)]*(?:\[['\"]Year['\"]\]|\.Year\b|year[s]?\b)", text, re.IGNORECASE | re.DOTALL)
    )
    sets_index_ticklabels_to_years = (
        any(re.search(rf"set_xticks\s*\(\s*{re.escape(name)}\b", text) for name in index_vars)
        and "set_xticklabels" in text
    )
    return uses_index and uses_year_column and sets_index_ticklabels_to_years


def _has_secondary_axis(code: str) -> bool:
    return any(token in code for token in ("twinx(", "secondary_yaxis(", "make_subplots(", "secondary_y=True"))


def _normalize_filename(name: str) -> str:
    return str(name).replace("\\", "/").split("/")[-1].strip()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
