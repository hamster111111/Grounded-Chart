from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from grounded_chart.chart_protocol import ChartProtocolAgent, ChartRenderingProtocol, validate_protocol
from grounded_chart.construction_plan import ChartConstructionPlan, PlanValidationReport
from grounded_chart.source_data import SourceDataExecution, SourceDataPlan


@dataclass(frozen=True)
class ArtifactWorkspace:
    root: Path
    plan_dir: Path
    execution_dir: Path
    repair_dir: Path
    round_id: str = "round_1"


@dataclass(frozen=True)
class ArtifactWorkspaceReport:
    ok: bool
    root: str
    plan_dir: str
    execution_dir: str
    repair_dir: str
    artifacts: tuple[dict[str, Any], ...] = ()
    issues: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "root": self.root,
            "plan_dir": self.plan_dir,
            "execution_dir": self.execution_dir,
            "repair_dir": self.repair_dir,
            "artifacts": [dict(item) for item in self.artifacts],
            "issues": [dict(item) for item in self.issues],
            "metadata": dict(self.metadata),
        }


class ArtifactWorkspaceBuilder:
    """Create human- and machine-readable artifacts for one pipeline round."""

    def build(
        self,
        *,
        output_root: str | Path,
        case_id: str,
        query: str,
        construction_plan: ChartConstructionPlan,
        plan_validation_report: PlanValidationReport,
        source_plan: SourceDataPlan,
        source_execution: SourceDataExecution | None,
        round_id: str = "round_1",
    ) -> ArtifactWorkspaceReport:
        workspace = create_artifact_workspace(output_root, round_id=round_id)
        artifacts: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []

        plan_json = workspace.plan_dir / "plan.json"
        _write_json(plan_json, construction_plan.to_dict())
        artifacts.append(_artifact("plan.json", plan_json, "canonical structured construction plan"))

        plan_md = workspace.plan_dir / "plan.md"
        plan_md.write_text(
            render_plan_markdown(
                case_id=case_id,
                query=query,
                construction_plan=construction_plan,
                plan_validation_report=plan_validation_report,
            ),
            encoding="utf-8",
        )
        artifacts.append(_artifact("plan.md", plan_md, "human-readable construction plan"))

        plan_validation_path = workspace.plan_dir / "plan_validation.json"
        _write_json(plan_validation_path, plan_validation_report.to_dict())
        artifacts.append(_artifact("plan_validation.json", plan_validation_path, "plan validation report"))

        steps_md = workspace.execution_dir / "steps.md"
        steps_md.write_text(render_execution_steps_markdown(construction_plan), encoding="utf-8")
        artifacts.append(_artifact("steps.md", steps_md, "execution step contract"))

        protocol_result = build_chart_protocol_artifacts(
            workspace=workspace,
            construction_plan=construction_plan,
        )
        artifacts.extend(protocol_result.artifacts)
        issues.extend(protocol_result.issues)
        protocols = tuple(protocol_result.metadata.get("protocols") or ())

        if source_execution is not None:
            generated = build_execution_artifacts(
                workspace=workspace,
                source_execution=source_execution,
                construction_plan=construction_plan,
            )
            artifacts.extend(generated.artifacts)
            issues.extend(generated.issues)
        else:
            issues.append(
                {
                    "code": "no_source_execution",
                    "message": "No source data execution is available; intermediate CSV artifacts were not generated.",
                    "severity": "warning",
                }
            )

        plot_spec = workspace.execution_dir / "plot_spec.md"
        plot_spec.write_text(render_plot_spec_markdown(construction_plan, protocols=protocols), encoding="utf-8")
        artifacts.append(_artifact("plot_spec.md", plot_spec, "plotting contract over prepared artifacts"))

        manifest = workspace.root / "artifact_workspace_manifest.json"
        report = ArtifactWorkspaceReport(
            ok=not any(str(issue.get("severity")) == "error" for issue in issues),
            root=str(workspace.root),
            plan_dir=str(workspace.plan_dir),
            execution_dir=str(workspace.execution_dir),
            repair_dir=str(workspace.repair_dir),
            artifacts=tuple(artifacts),
            issues=tuple(issues),
            metadata={
                "round_id": round_id,
                "case_id": case_id,
                "source_files": [item.name for item in source_plan.files],
                "chart_protocols": [dict(item) for item in protocols],
            },
        )
        _write_json(manifest, report.to_dict())
        return report


@dataclass(frozen=True)
class _GeneratedArtifacts:
    artifacts: tuple[dict[str, Any], ...] = ()
    issues: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def create_artifact_workspace(output_root: str | Path, *, round_id: str = "round_1") -> ArtifactWorkspace:
    root = Path(output_root).resolve()
    plan_dir = root / "plan" / round_id
    execution_dir = root / "execution" / round_id
    repair_dir = root / "repair" / round_id
    for path in (plan_dir, execution_dir, repair_dir):
        path.mkdir(parents=True, exist_ok=True)
    return ArtifactWorkspace(root=root, plan_dir=plan_dir, execution_dir=execution_dir, repair_dir=repair_dir, round_id=round_id)


def build_execution_artifacts(
    *,
    workspace: ArtifactWorkspace,
    source_execution: SourceDataExecution,
    construction_plan: ChartConstructionPlan,
) -> _GeneratedArtifacts:
    tables = {str(item.get("name")): item for item in source_execution.to_dict().get("loaded_tables", []) if isinstance(item, dict)}
    artifacts: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    summary_path = workspace.execution_dir / "step_01_sources_summary.json"
    _write_json(summary_path, _source_summary_payload(tables))
    artifacts.append(_artifact("step_01_sources_summary.json", summary_path, "loaded source schema and row-count summary"))

    imports_name = _layer_source(construction_plan, role="yearly_change_bars") or _match_table_name(tables, "import")
    if imports_name and imports_name in tables:
        path, issue = _write_imports_waterfall_values(workspace.execution_dir, tables[imports_name])
        artifacts.append(_artifact("step_02_imports_waterfall_values.csv", path, "direct source values for import waterfall layer"))
        if issue:
            issues.append(issue)
        if _has_chart_type(construction_plan, "waterfall"):
            render_path, render_issue = _write_imports_waterfall_render_table(workspace.execution_dir, tables[imports_name])
            artifacts.append(
                _artifact(
                    "step_02_imports_waterfall_render_table.csv",
                    render_path,
                    "protocol-grounded bar geometry for import waterfall layer",
                )
            )
            if render_issue:
                issues.append(render_issue)
    elif _has_chart_type(construction_plan, "waterfall"):
        issues.append(_missing_table_issue("imports_waterfall", imports_name or "Imports.csv"))

    consumption_name = _layer_source(construction_plan, role="stacked_area") or _match_table_name(tables, "consumption")
    if consumption_name and consumption_name in tables:
        path, issue = _write_consumption_area_values(
            workspace.execution_dir,
            tables[consumption_name],
            construction_plan=construction_plan,
        )
        artifacts.append(_artifact("step_03_consumption_area_values.csv", path, "stacked area values for consumption layer"))
        if issue:
            issues.append(issue)
    elif _has_chart_type(construction_plan, "area"):
        issues.append(_missing_table_issue("consumption_area", consumption_name or "Consumption.csv"))

    ratio_name = _layer_source(construction_plan, role="ratio_breakdown") or _match_table_name(tables, "ratio")
    if ratio_name and ratio_name in tables:
        years = _pie_years(construction_plan)
        path, issue = _write_pie_values(workspace.execution_dir, tables[ratio_name], years=years)
        artifacts.append(_artifact("step_04_pie_values.csv", path, "pie values with raw and normalized percentages"))
        if issue:
            issues.append(issue)
    elif _has_chart_type(construction_plan, "pie"):
        issues.append(_missing_table_issue("pie_values", ratio_name or "Grain_Consumption_Ratio.csv"))

    return _GeneratedArtifacts(artifacts=tuple(artifacts), issues=tuple(issues))


def build_chart_protocol_artifacts(
    *,
    workspace: ArtifactWorkspace,
    construction_plan: ChartConstructionPlan,
    agent: ChartProtocolAgent | None = None,
) -> _GeneratedArtifacts:
    chart_types = _chart_types(construction_plan)
    protocol_agent = agent or ChartProtocolAgent()
    artifacts: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
    protocol_dir = workspace.execution_dir / "chart_protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    for chart_type in chart_types:
        protocol = protocol_agent.build_protocol(chart_type=chart_type, context={"construction_plan": construction_plan.to_dict()})
        validation = validate_protocol(protocol)
        base_name = f"{_safe_filename(chart_type)}_protocol"
        json_path = protocol_dir / f"{base_name}.json"
        md_path = protocol_dir / f"{base_name}.md"
        validation_path = protocol_dir / f"{base_name}_validation.json"
        _write_json(json_path, protocol.to_dict())
        md_path.write_text(render_protocol_markdown(protocol, validation), encoding="utf-8")
        _write_json(validation_path, validation.to_dict())
        artifacts.append(_artifact(f"{base_name}.json", json_path, f"structured rendering protocol for {chart_type} charts"))
        artifacts.append(_artifact(f"{base_name}.md", md_path, f"human-readable rendering protocol for {chart_type} charts"))
        artifacts.append(_artifact(f"{base_name}_validation.json", validation_path, f"validation report for {chart_type} protocol"))
        protocols.append(protocol.to_dict())
        for issue in validation.issues:
            issues.append(
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "artifact": str(validation_path),
                    "evidence": issue.evidence,
                }
            )
    return _GeneratedArtifacts(artifacts=tuple(artifacts), issues=tuple(issues), metadata={"protocols": protocols})


def render_plan_markdown(
    *,
    case_id: str,
    query: str,
    construction_plan: ChartConstructionPlan,
    plan_validation_report: PlanValidationReport,
) -> str:
    plan = construction_plan.to_dict()
    lines = [
        f"# Plan Round 1: {case_id}",
        "",
        "## Query",
        query.strip(),
        "",
        "## Non-Ambiguity Rules",
        "- Every executable step must declare input, operation, output artifact, purpose, and assertion when applicable.",
        "- If a data semantic is uncertain, mark it as needs_evidence instead of inventing a transformation.",
        "- Do not use vague operations such as compute yearly change without specifying whether source values are raw totals, deltas, or cumulative totals.",
        "",
        "## Layout",
        f"- plan_type: {plan.get('plan_type')}",
        f"- layout_strategy: {plan.get('layout_strategy')}",
        f"- figure_size: {plan.get('figure_size')}",
        "",
        "## Execution Steps",
    ]
    for step in plan.get("execution_steps") or []:
        lines.extend(
            [
                f"- {step.get('step_id')}: {step.get('action')}",
                f"  purpose: {step.get('purpose')}",
                f"  inputs: {step.get('inputs')}",
                f"  outputs: {step.get('outputs')}",
            ]
        )
    lines.extend(["", "## Validation", f"- ok: {plan_validation_report.ok}"])
    for issue in plan_validation_report.issues:
        lines.append(f"- {issue.severity}: {issue.code} - {issue.message}")
    replan_decisions = [
        decision
        for decision in construction_plan.decisions
        if str(decision.category) == "plan_replanning"
    ]
    if replan_decisions:
        lines.extend(["", "## Replan Feedback"])
        for decision in replan_decisions:
            value = decision.value if isinstance(decision.value, dict) else {}
            lines.append(f"- decision_id: {decision.decision_id}")
            lines.append(f"- rationale: {decision.rationale}")
            for item in list(value.get("feedback_items") or []):
                lines.append(f"- must_address: {item}")
    return "\n".join(lines).strip() + "\n"


def render_execution_steps_markdown(construction_plan: ChartConstructionPlan) -> str:
    lines = [
        "# Execution Round 1 Steps",
        "",
        "ExecutorAgent must implement these steps through explicit scripts/artifacts. Plotting code should read prepared artifacts rather than hiding transformations inside plot calls.",
        "",
    ]
    for step in construction_plan.to_dict().get("execution_steps") or []:
        lines.extend(
            [
                f"## {step.get('step_id')}",
                f"- action: {step.get('action')}",
                f"- purpose: {step.get('purpose')}",
                f"- script: {step.get('script')}",
                f"- inputs: {step.get('inputs')}",
                f"- outputs: {step.get('outputs')}",
            ]
        )
        for assertion in step.get("assertions") or []:
            lines.append(f"- assertion: {assertion}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_protocol_markdown(protocol: ChartRenderingProtocol, validation_report) -> str:
    lines = [
        f"# Chart Protocol: {protocol.chart_type}",
        "",
        f"- protocol_id: {protocol.protocol_id}",
        f"- version: {protocol.version}",
        f"- source: {protocol.source}",
        f"- validation_ok: {validation_report.ok}",
        "",
        "## Rendering Rules",
    ]
    lines.extend(f"- {item}" for item in protocol.rendering_rules)
    lines.extend(["", "## Required Artifact Columns"])
    lines.extend(f"- {item}" for item in protocol.required_artifact_columns)
    if protocol.plotting_primitives:
        lines.extend(["", "## Plotting Primitives"])
        for primitive in protocol.plotting_primitives:
            lines.append(f"- {primitive}")
    if protocol.forbidden_shortcuts:
        lines.extend(["", "## Forbidden Shortcuts"])
        lines.extend(f"- {item}" for item in protocol.forbidden_shortcuts)
    return "\n".join(lines).strip() + "\n"


def render_plot_spec_markdown(construction_plan: ChartConstructionPlan, *, protocols: tuple[dict[str, Any], ...] = ()) -> str:
    lines = [
        "# Plot Spec Round 1",
        "",
        "The final plotting code must consume prepared CSV artifacts from this execution round.",
        "",
        "## Visual Layers",
    ]
    for panel in construction_plan.panels:
        lines.append(f"- panel: {panel.panel_id} ({panel.role})")
        for layer in panel.layers:
            lines.append(
                f"- layer: {layer.layer_id}, chart_type={layer.chart_type}, role={layer.role}, data_source={layer.data_source}, x={layer.x}, y={list(layer.y)}, axis={layer.axis}"
            )
    lines.extend(
        [
            "",
            "## Chart Protocols",
            "- chart protocols in execution/round_1/chart_protocols/ define chart-type-specific rendering semantics.",
            "- if a protocol exists for a planned chart type, plotting code must follow the protocol instead of using a generic mark.",
        ]
    )
    for protocol in protocols:
        lines.append(f"- {protocol.get('chart_type')}: {protocol.get('protocol_id')}")
    lines.extend(
        [
            "",
            "## Data Artifact Rules",
            "- import waterfall bars must use step_02_imports_waterfall_render_table.csv for bar geometry.",
            "- step_02_imports_waterfall_values.csv records source values and should not be used as ordinary zero-based bar heights when a render table exists.",
            "- waterfall plotting must call ax.bar or equivalent with bottom=bar_bottom and height=bar_height from the render table.",
            "- waterfall color_role values are increase, decrease, and total; map those exact values to distinct up/down/total colors.",
            "- consumption area must use step_03_consumption_area_values.csv.",
            "- area plotting must obey composition_policy and fill-bottom/fill-top columns in step_03_consumption_area_values.csv.",
            "- if composition_policy is overlap, draw independent translucent fill_between layers; do not use Urban + Rural as a stacked top.",
            "- pie charts must use step_04_pie_values.csv.",
            "- all year-based overlays must use the same x_index or the same raw Year basis consistently.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _write_imports_waterfall_values(execution_dir: Path, table: dict[str, Any]) -> tuple[Path, dict[str, Any] | None]:
    rows = _sort_rows_by_year(table.get("rows") or [])
    out_rows = []
    issue = _require_columns(table, {"Year", "Urban", "Rural"}, artifact="step_02_imports_waterfall_values.csv")
    for index, row in enumerate(rows):
        role = "annual_change"
        if index == 0:
            role = "initial"
        elif index == len(rows) - 1:
            role = "cumulative_total"
        out_rows.append(
            {
                "x_index": index,
                "Year": row.get("Year"),
                "Urban": row.get("Urban"),
                "Rural": row.get("Rural"),
                "Urban_source": row.get("Urban"),
                "Rural_source": row.get("Rural"),
                "Urban_plot": row.get("Urban"),
                "Rural_plot": row.get("Rural"),
                "waterfall_role": role,
                "transform": "direct_use_source_values",
            }
        )
    path = execution_dir / "step_02_imports_waterfall_values.csv"
    _write_csv(path, out_rows)
    return path, issue


def _write_imports_waterfall_render_table(execution_dir: Path, table: dict[str, Any]) -> tuple[Path, dict[str, Any] | None]:
    rows = _sort_rows_by_year(table.get("rows") or [])
    issue = _require_columns(table, {"Year", "Urban", "Rural"}, artifact="step_02_imports_waterfall_render_table.csv")
    series_names = [name for name in ("Urban", "Rural") if name in {str(item) for item in table.get("columns") or []}]
    offsets = _series_offsets(series_names)
    out_rows: list[dict[str, Any]] = []
    bar_width = 0.32
    for series in series_names:
        cumulative = 0.0
        series_positions = [index + offsets.get(series, 0.0) for index, _ in enumerate(rows)]
        for index, row in enumerate(rows):
            value = _number(row.get(series))
            role = "delta"
            if index == 0:
                role = "initial"
                bottom = 0.0
                height = value
                top = value
                cumulative = top
            elif index == len(rows) - 1:
                role = "total"
                bottom = 0.0
                height = value
                top = value
                cumulative = top
            else:
                bottom = cumulative
                height = value
                top = cumulative + value
                cumulative = top
            connector_y_start = top if index < max(0, len(rows) - 2) else ""
            connector_y_end = top if index < max(0, len(rows) - 2) else ""
            connector_x_start = series_positions[index] + bar_width / 2 if index < max(0, len(rows) - 2) else ""
            connector_x_end = series_positions[index + 1] - bar_width / 2 if index < max(0, len(rows) - 2) else ""
            out_rows.append(
                {
                    "Year": row.get("Year"),
                    "series": series,
                    "x_index": index,
                    "x_offset": offsets.get(series, 0.0),
                    "x_position": series_positions[index],
                    "bar_width": bar_width,
                    "source_value": value,
                    "bar_bottom": bottom,
                    "bar_height": height,
                    "bar_top": top,
                    "role": role,
                    "color_role": "total" if role == "total" else "increase" if value >= 0 else "decrease",
                    "connector_x_start": connector_x_start,
                    "connector_x_end": connector_x_end,
                    "connector_y_start": connector_y_start,
                    "connector_y_end": connector_y_end,
                    "transform": "waterfall_protocol_bar_geometry",
                }
            )
    path = execution_dir / "step_02_imports_waterfall_render_table.csv"
    _write_csv(path, out_rows)
    return path, issue


def _write_consumption_area_values(
    execution_dir: Path,
    table: dict[str, Any],
    *,
    construction_plan: ChartConstructionPlan,
) -> tuple[Path, dict[str, Any] | None]:
    rows = _sort_rows_by_year(table.get("rows") or [])
    out_rows = []
    issue = _require_columns(table, {"Year", "Urban", "Rural"}, artifact="step_03_consumption_area_values.csv")
    modifiers = _area_modifiers(construction_plan)
    composition = str(modifiers.get("composition") or "additive_stack")
    scale_policy = modifiers.get("scale_policy") if isinstance(modifiers.get("scale_policy"), dict) else {}
    axis_min = _number(scale_policy.get("min")) if scale_policy.get("type") == "explicit_range" else 0.0
    for index, row in enumerate(rows):
        urban = _number(row.get("Urban"))
        rural = _number(row.get("Rural"))
        urban_bottom = axis_min if composition == "overlap" else 0.0
        urban_top = urban
        rural_bottom = axis_min if composition == "overlap" else urban
        rural_top = rural if composition == "overlap" else urban + rural
        out_rows.append(
            {
                "x_index": index,
                "Year": row.get("Year"),
                "Urban": urban,
                "Rural": rural,
                "Urban_area": urban,
                "Rural_area": rural,
                "Urban_fill_bottom": urban_bottom,
                "Urban_fill_top": urban_top,
                "Rural_fill_bottom": rural_bottom,
                "Rural_fill_top": rural_top,
                "Total": urban + rural,
                "Rural_stack_top": urban + rural,
                "composition_policy": composition,
                "opacity_policy": modifiers.get("opacity") or "",
                "axis_min": scale_policy.get("min") if scale_policy.get("type") == "explicit_range" else "",
                "axis_max": scale_policy.get("max") if scale_policy.get("type") == "explicit_range" else "",
                "transform": "modifier_grounded_area_fill_geometry",
            }
        )
    path = execution_dir / "step_03_consumption_area_values.csv"
    _write_csv(path, out_rows)
    return path, issue


def _write_pie_values(execution_dir: Path, table: dict[str, Any], *, years: list[int]) -> tuple[Path, dict[str, Any] | None]:
    rows = [dict(row) for row in table.get("rows") or []]
    issue = _require_columns(table, {"Year", "Age Group", "Consumption Ratio"}, artifact="step_04_pie_values.csv")
    wanted = set(years)
    filtered = [row for row in rows if not wanted or int(_number(row.get("Year"))) in wanted]
    totals: dict[int, float] = {}
    for row in filtered:
        year = int(_number(row.get("Year")))
        totals[year] = totals.get(year, 0.0) + _number(row.get("Consumption Ratio"))
    out_rows = []
    for row in filtered:
        year = int(_number(row.get("Year")))
        raw = _number(row.get("Consumption Ratio"))
        total = totals.get(year, 0.0)
        out_rows.append(
            {
                "Year": year,
                "Age Group": row.get("Age Group"),
                "Consumption Ratio": raw,
                "Consumption Ratio_raw": raw,
                "Percentage": raw / total * 100 if total else 0.0,
                "Pie_autopct_percent": raw / total * 100 if total else 0.0,
                "Explode": 0.1 if "60" in str(row.get("Age Group") or "") or "older" in str(row.get("Age Group") or "").lower() else 0.0,
                "transform": "filter_year_preserve_raw_and_record_normalized_percent",
            }
        )
    path = execution_dir / "step_04_pie_values.csv"
    _write_csv(path, out_rows)
    return path, issue


def _source_summary_payload(tables: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "tables": [
            {
                "name": name,
                "columns": list(table.get("columns") or []),
                "row_count_loaded": table.get("row_count_loaded"),
                "truncated": table.get("truncated"),
                "read_error": table.get("read_error"),
            }
            for name, table in sorted(tables.items())
        ]
    }


def _layer_source(construction_plan: ChartConstructionPlan, *, role: str) -> str | None:
    for panel in construction_plan.panels:
        for layer in panel.layers:
            if layer.role == role and layer.data_source:
                return layer.data_source
    return None


def _has_chart_type(construction_plan: ChartConstructionPlan, chart_type: str) -> bool:
    return any(layer.chart_type == chart_type for panel in construction_plan.panels for layer in panel.layers)


def _chart_types(construction_plan: ChartConstructionPlan) -> list[str]:
    chart_types = []
    for panel in construction_plan.panels:
        for layer in panel.layers:
            chart_type = str(layer.chart_type or "").strip().lower()
            if chart_type and chart_type not in chart_types:
                chart_types.append(chart_type)
    return chart_types


def _area_modifiers(construction_plan: ChartConstructionPlan) -> dict[str, Any]:
    for panel in construction_plan.panels:
        for layer in panel.layers:
            if layer.chart_type == "area":
                return dict(layer.semantic_modifiers)
    return {}


def _pie_years(construction_plan: ChartConstructionPlan) -> list[int]:
    years = []
    for panel in construction_plan.panels:
        if panel.role == "inset_pie_chart" and panel.anchor.get("value") is not None:
            try:
                years.append(int(panel.anchor["value"]))
            except (TypeError, ValueError):
                pass
    return years


def _match_table_name(tables: dict[str, dict[str, Any]], keyword: str) -> str | None:
    for name in tables:
        if keyword.lower() in name.lower():
            return name
    return None


def _sort_rows_by_year(rows: list[Any]) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows if isinstance(row, dict)]
    return sorted(normalized, key=lambda row: _number(row.get("Year")))


def _require_columns(table: dict[str, Any], required: set[str], *, artifact: str) -> dict[str, Any] | None:
    columns = {str(item) for item in table.get("columns") or []}
    missing = sorted(required - columns)
    if not missing:
        return None
    return {
        "code": "missing_required_columns",
        "message": f"{artifact} cannot be fully validated because columns are missing: {missing}.",
        "severity": "error",
        "artifact": artifact,
        "available_columns": sorted(columns),
    }


def _missing_table_issue(artifact_family: str, table_name: str) -> dict[str, Any]:
    return {
        "code": "missing_source_table",
        "message": f"Cannot generate {artifact_family} because source table `{table_name}` is unavailable.",
        "severity": "error",
        "table": table_name,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _artifact(name: str, path: Path, purpose: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "relative_path": _workspace_relative_path(path),
        "purpose": purpose,
    }


def _workspace_relative_path(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part in {"plan", "execution", "repair"}:
            return str(Path(*parts[index:])).replace("\\", "/")
    return path.name


def _number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _series_offsets(series_names: list[str]) -> dict[str, float]:
    if not series_names:
        return {}
    if len(series_names) == 1:
        return {series_names[0]: 0.0}
    step = 0.36
    start = -step * (len(series_names) - 1) / 2
    return {name: start + index * step for index, name in enumerate(series_names)}


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe.strip("_").lower() or "chart"
