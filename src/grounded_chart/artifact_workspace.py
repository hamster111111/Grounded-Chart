from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from grounded_chart.chart_protocol import ChartProtocolAgent, ChartRenderingProtocol, validate_protocol
from grounded_chart.construction_plan import ChartConstructionPlan, PlanValidationReport
from grounded_chart.source_data import SourceDataExecution, SourceDataPlan


PLAN_AGENT_DIR = "PlanAgent"
EXECUTOR_AGENT_DIR = "ExecutorAgent"
REPAIR_AGENT_DIR = "RepairAgent"
LAYOUT_AGENT_DIR = "LayoutAgent"
FIGURE_READER_AGENT_DIR = "FigureReaderAgent"
PLAN_REVISION_AGENT_DIR = "PlanRevisionAgent"

_WORKSPACE_RELATIVE_PATH_ANCHORS = {
    PLAN_AGENT_DIR,
    EXECUTOR_AGENT_DIR,
    REPAIR_AGENT_DIR,
    LAYOUT_AGENT_DIR,
    FIGURE_READER_AGENT_DIR,
    PLAN_REVISION_AGENT_DIR,
    # Legacy output roots remain readable in old manifests and tests.
    "plan",
    "execution",
    "repair",
}


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


@dataclass(frozen=True)
class ArtifactRequest:
    artifact_id: str
    layer_id: str
    chart_type: str
    artifact_role: str
    source_table: str
    x_column: str | None = None
    series_columns: tuple[str, ...] = ()
    category_column: str | None = None
    value_column: str | None = None
    filter_column: str | None = None
    filter_values: tuple[Any, ...] = ()
    transform_ops: tuple[str, ...] = ()
    semantic_modifiers: dict[str, Any] = field(default_factory=dict)
    output_name: str = ""
    required_for_plotting: bool = True
    contract_tier: str = "hard_fidelity"
    aliases: tuple[str, ...] = ()
    assertions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["series_columns"] = list(self.series_columns)
        payload["filter_values"] = list(self.filter_values)
        payload["transform_ops"] = list(self.transform_ops)
        payload["aliases"] = list(self.aliases)
        payload["assertions"] = list(self.assertions)
        return payload


class ArtifactWorkspaceBuilder:
    """Create human- and machine-readable artifacts for one pipeline round."""

    def __init__(self, *, protocol_agent: ChartProtocolAgent | None = None) -> None:
        self.protocol_agent = protocol_agent

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

        protocol_result = build_chart_protocol_artifacts(
            workspace=workspace,
            construction_plan=construction_plan,
            query=query,
            source_plan=source_plan,
            source_execution=source_execution,
            prepared_artifacts=tuple(artifacts),
            agent=self.protocol_agent,
        )
        artifacts.extend(protocol_result.artifacts)
        issues.extend(protocol_result.issues)
        protocols = tuple(protocol_result.metadata.get("protocols") or ())

        plot_spec = workspace.execution_dir / "plot_spec.md"
        plot_spec.write_text(
            render_plot_spec_markdown(construction_plan, artifacts=tuple(artifacts), protocols=protocols),
            encoding="utf-8",
        )
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
    plan_dir = root / PLAN_AGENT_DIR / round_id
    execution_dir = root / EXECUTOR_AGENT_DIR / round_id
    repair_dir = root / REPAIR_AGENT_DIR / round_id
    for path in (plan_dir, execution_dir):
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

    requests = compile_artifact_requests(construction_plan, tables=tables)
    request_path = workspace.execution_dir / "artifact_requests.json"
    _write_json(request_path, {"requests": [request.to_dict() for request in requests]})
    artifacts.append(_artifact("artifact_requests.json", request_path, "compiled deterministic artifact requests"))

    for request in requests:
        table = tables.get(request.source_table)
        if not table:
            issues.append(_missing_table_issue(request.artifact_role, request.source_table))
            continue
        generated, request_issues = _execute_artifact_request(workspace.execution_dir, request, table)
        artifacts.extend(generated)
        issues.extend(request_issues)

    return _GeneratedArtifacts(artifacts=tuple(artifacts), issues=tuple(issues))


def build_chart_protocol_artifacts(
    *,
    workspace: ArtifactWorkspace,
    construction_plan: ChartConstructionPlan,
    query: str = "",
    source_plan: SourceDataPlan | None = None,
    source_execution: SourceDataExecution | None = None,
    prepared_artifacts: tuple[dict[str, Any], ...] = (),
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
        protocol = protocol_agent.build_protocol(
            chart_type=chart_type,
            context={
                "query": query,
                "construction_plan": construction_plan.to_dict(),
                "source_data_plan": source_plan.to_dict() if source_plan is not None else None,
                "source_data_execution": source_execution.to_dict() if source_execution is not None else None,
                "prepared_artifacts": [dict(item) for item in prepared_artifacts],
                "round_id": workspace.round_id,
            },
        )
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
        "# ExecutorAgent Round Steps",
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
        "## Contract Tiers",
    ]
    for tier, description in protocol.contract_tiers.items():
        lines.append(f"- {tier}: {description}")
    lines.extend(
        [
            "",
            "## Rendering Rules",
        ]
    )
    lines.extend(f"- {item}" for item in protocol.rendering_rules)
    lines.extend(["", "## Required Artifact Columns"])
    lines.extend(f"- {item}" for item in protocol.required_artifact_columns)
    if protocol.plotting_primitives:
        lines.extend(["", "## Plotting Primitives"])
        for primitive in protocol.plotting_primitives:
            lines.append(f"- {primitive}")
    if protocol.visual_channel_contracts:
        lines.extend(["", "## Minimal Visual Channel Contracts"])
        for contract in protocol.visual_channel_contracts:
            lines.append(f"- {contract.to_dict()}")
    if protocol.forbidden_shortcuts:
        lines.extend(["", "## Forbidden Shortcuts"])
        lines.extend(f"- {item}" for item in protocol.forbidden_shortcuts)
    return "\n".join(lines).strip() + "\n"


def render_plot_spec_markdown(
    construction_plan: ChartConstructionPlan,
    *,
    artifacts: tuple[dict[str, Any], ...] = (),
    protocols: tuple[dict[str, Any], ...] = (),
) -> str:
    lines = [
        "# ExecutorAgent Plot Spec",
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
            "- chart protocols in the current ExecutorAgent round's chart_protocols/ directory define chart-type-specific rendering semantics.",
            "- hard_fidelity protocol commitments must be followed for source-grounded data, deterministic geometry, explicit requirements, and semantic channel bindings.",
            "- soft_guidance commitments should improve readability and may feed visual feedback/replanning, but should not override hard fidelity.",
            "- free_design commitments are delegated to ExecutorAgent unless the source request explicitly constrains them.",
        ]
    )
    for protocol in protocols:
        lines.append(f"- {protocol.get('chart_type')}: {protocol.get('protocol_id')}")
    lines.extend(["", "## Prepared Data Artifacts"])
    prepared_csvs = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and str(item.get("name") or "").endswith(".csv")
        and bool(item.get("required_for_plotting"))
    ]
    if not prepared_csvs:
        lines.append("- no required plotting CSV artifacts were generated for this round.")
    for artifact in prepared_csvs:
        schema = artifact.get("schema") if isinstance(artifact.get("schema"), dict) else {}
        columns = list(schema.get("columns") or artifact.get("columns") or [])
        lines.append(
            "- "
            f"name={artifact.get('name')}, role={artifact.get('artifact_role')}, chart_type={artifact.get('chart_type')}, "
            f"layer_id={artifact.get('layer_id')}, contract_tier={artifact.get('contract_tier')}, "
            f"relative_path={artifact.get('relative_path')}, columns={columns}"
        )
    lines.extend(
        [
            "",
            "## Data Artifact Rules",
            "- Prepared artifact metadata defines the executable contract: use artifact_role, chart_type, layer_id, contract_tier, relative_path, and schema.columns rather than fixed filenames or guessed columns.",
            "- If an artifact has role=waterfall_geometry, bars must use bar_bottom and bar_height from that artifact; source-value artifacts are evidence tables, not zero-based bar geometry.",
            "- If an artifact has role=area_fill_geometry, use its *_fill_bottom and *_fill_top columns and composition_policy rather than recomputing hidden stack geometry.",
            "- Geometry roles and visual-channel roles are separate; follow chart_protocols and artifact columns such as fill_color_role, series_color_role, series, and change_role.",
            "- Hard fidelity artifacts are blocking data/geometry inputs; soft guidance may inform readability; free design remains delegated to ExecutorAgent.",
            "- All overlaid layers that share a conceptual x-axis should use a single x-coordinate basis, either artifact x_index/x_position or raw x values consistently.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def compile_artifact_requests(
    construction_plan: ChartConstructionPlan,
    *,
    tables: dict[str, dict[str, Any]],
) -> tuple[ArtifactRequest, ...]:
    requests: list[ArtifactRequest] = []
    pie_groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for panel in construction_plan.panels:
        for layer in panel.layers:
            chart_type = str(layer.chart_type or "").strip().lower()
            if not chart_type:
                continue
            source_table = _resolve_source_table(layer.to_dict(), tables)
            if not source_table:
                continue
            table = tables.get(source_table, {})
            columns = [str(item) for item in list(table.get("columns") or [])]
            x_column = _resolve_x_column(layer.to_dict(), columns)
            if chart_type == "waterfall":
                series_columns = tuple(_resolve_series_columns(layer.to_dict(), table, x_column=x_column))
                if not series_columns:
                    continue
                base = _request_base(layer.layer_id, chart_type)
                requests.append(
                    ArtifactRequest(
                        artifact_id=f"{base}.source_values",
                        layer_id=layer.layer_id,
                        chart_type=chart_type,
                        artifact_role="source_values",
                        source_table=source_table,
                        x_column=x_column,
                        series_columns=series_columns,
                        transform_ops=("sort_by_x", "copy_source_values"),
                        output_name=f"{base}_source_values.csv",
                        required_for_plotting=False,
                        aliases=_legacy_aliases(chart_type, "source_values", layer.to_dict()),
                        assertions=(
                            "row_count equals source row_count",
                            "plotted value columns preserve source values row by row",
                        ),
                    )
                )
                requests.append(
                    ArtifactRequest(
                        artifact_id=f"{base}.waterfall_geometry",
                        layer_id=layer.layer_id,
                        chart_type=chart_type,
                        artifact_role="waterfall_geometry",
                        source_table=source_table,
                        x_column=x_column,
                        series_columns=series_columns,
                        transform_ops=("sort_by_x", "compute_waterfall_geometry"),
                        output_name=f"{base}_waterfall_geometry.csv",
                        aliases=_legacy_aliases(chart_type, "waterfall_geometry", layer.to_dict()),
                        assertions=(
                            "delta bars start at previous cumulative value for the same series",
                            "terminal total bars start from zero",
                            "visual color role is not forced to equal geometry role",
                        ),
                    )
                )
            elif chart_type == "area":
                series_columns = tuple(_resolve_series_columns(layer.to_dict(), table, x_column=x_column))
                if not series_columns:
                    continue
                base = _request_base(layer.layer_id, chart_type)
                requests.append(
                    ArtifactRequest(
                        artifact_id=f"{base}.area_fill_geometry",
                        layer_id=layer.layer_id,
                        chart_type=chart_type,
                        artifact_role="area_fill_geometry",
                        source_table=source_table,
                        x_column=x_column,
                        series_columns=series_columns,
                        transform_ops=("sort_by_x", "compute_area_fill_geometry"),
                        semantic_modifiers=dict(layer.semantic_modifiers),
                        output_name=f"{base}_area_fill_geometry.csv",
                        aliases=_legacy_aliases(chart_type, "area_fill_geometry", layer.to_dict()),
                        assertions=(
                            "source series values are preserved",
                            "composition_policy controls whether fill intervals overlap or accumulate",
                        ),
                    )
                )
            elif chart_type == "pie":
                category_column = _resolve_category_column(layer.to_dict(), table)
                value_column = _resolve_value_column(layer.to_dict(), table, exclude={x_column, category_column})
                filter_column, filter_values = _resolve_filter(layer.to_dict(), panel.to_dict(), columns)
                if not category_column or not value_column:
                    continue
                key = (source_table, category_column, value_column, filter_column or "")
                group = pie_groups.setdefault(
                    key,
                    {
                        "layer_ids": [],
                        "filter_values": [],
                        "layer": layer.to_dict(),
                    },
                )
                group["layer_ids"].append(layer.layer_id)
                for value in filter_values:
                    if value not in group["filter_values"]:
                        group["filter_values"].append(value)
            elif chart_type in {"bar", "line", "scatter", "heatmap", "box"}:
                series_columns = tuple(_resolve_series_columns(layer.to_dict(), table, x_column=x_column))
                if not x_column and not series_columns:
                    continue
                base = _request_base(layer.layer_id, chart_type)
                requests.append(
                    ArtifactRequest(
                        artifact_id=f"{base}.source_values",
                        layer_id=layer.layer_id,
                        chart_type=chart_type,
                        artifact_role="source_values",
                        source_table=source_table,
                        x_column=x_column,
                        series_columns=series_columns,
                        transform_ops=("sort_by_x", "copy_source_values") if x_column else ("copy_source_values",),
                        output_name=f"{base}_source_values.csv",
                        assertions=("source values are preserved row by row",),
                    )
                )

    for (source_table, category_column, value_column, filter_column), group in pie_groups.items():
        layer = group["layer"]
        layer_ids = [str(item) for item in group["layer_ids"]]
        base = _request_base(layer_ids[0] if len(layer_ids) == 1 else ".".join(layer_ids), "pie")
        requests.append(
            ArtifactRequest(
                artifact_id=f"{base}.categorical_values",
                layer_id=",".join(layer_ids),
                chart_type="pie",
                artifact_role="categorical_values",
                source_table=source_table,
                category_column=category_column,
                value_column=value_column,
                filter_column=filter_column or None,
                filter_values=tuple(group["filter_values"]),
                transform_ops=("filter_rows", "normalize_percent_by_group"),
                output_name=f"{base}_categorical_values.csv",
                aliases=_legacy_aliases("pie", "categorical_values", layer),
                assertions=(
                    "category labels are preserved",
                    "raw values are preserved",
                    "normalized percentages are recorded separately from raw values",
                ),
            )
        )

    return tuple(requests)


def _execute_artifact_request(
    execution_dir: Path,
    request: ArtifactRequest,
    table: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    required = _required_columns_for_request(request)
    issue = _require_columns(table, required, artifact=request.output_name) if required else None
    if issue:
        issues.append(issue)

    if request.artifact_role == "waterfall_geometry":
        rows = _waterfall_geometry_rows(table, request)
        purpose = "protocol-grounded waterfall geometry for a planned visual layer"
    elif request.artifact_role == "area_fill_geometry":
        rows = _area_fill_geometry_rows(table, request)
        purpose = "modifier-grounded area fill geometry for a planned visual layer"
    elif request.artifact_role == "categorical_values":
        rows = _categorical_value_rows(table, request)
        purpose = "filtered categorical values with raw and normalized percentages"
    else:
        rows = _source_value_rows(table, request)
        purpose = "source values copied for a planned visual layer"

    artifacts: list[dict[str, Any]] = []
    for output_name in (request.output_name, *request.aliases):
        path = execution_dir / output_name
        _write_csv(path, rows)
        artifact = _artifact(output_name, path, purpose)
        artifact.update(_artifact_request_metadata(request))
        if output_name != request.output_name:
            artifact["alias_for"] = request.output_name
            artifact["legacy_alias"] = True
            artifact["required_for_plotting"] = False
            artifact["contract_tier"] = "free_design"
        artifacts.append(artifact)
    return artifacts, issues


def _source_value_rows(table: dict[str, Any], request: ArtifactRequest) -> list[dict[str, Any]]:
    rows = _sort_rows_by_column(table.get("rows") or [], request.x_column)
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        out = dict(row)
        out["x_index"] = index
        if request.x_column:
            out["x_value"] = row.get(request.x_column)
        for series in request.series_columns:
            out[f"{series}_source"] = row.get(series)
            out[f"{series}_plot"] = row.get(series)
        out["artifact_role"] = request.artifact_role
        out["transform"] = "direct_use_source_values"
        out_rows.append(out)
    return out_rows


def _waterfall_geometry_rows(table: dict[str, Any], request: ArtifactRequest) -> list[dict[str, Any]]:
    rows = _sort_rows_by_column(table.get("rows") or [], request.x_column)
    series_names = list(request.series_columns)
    offsets = _series_offsets(series_names)
    out_rows: list[dict[str, Any]] = []
    bar_width = min(0.8, 0.72 / max(1, len(series_names)))
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
            has_connector = index < max(0, len(rows) - 2)
            out_rows.append(
                {
                    request.x_column or "x": row.get(request.x_column) if request.x_column else index,
                    "series": series,
                    "x_index": index,
                    "x_value": row.get(request.x_column) if request.x_column else index,
                    "x_offset": offsets.get(series, 0.0),
                    "x_position": series_positions[index],
                    "bar_width": bar_width,
                    "source_value": value,
                    "bar_bottom": bottom,
                    "bar_height": height,
                    "bar_top": top,
                    "role": role,
                    "change_role": "terminal_total" if role == "total" else "increase" if value >= 0 else "decrease",
                    "color_role": series,
                    "fill_color_role": series,
                    "series_color_role": series,
                    "connector_x_start": series_positions[index] + bar_width / 2 if has_connector else "",
                    "connector_x_end": series_positions[index + 1] - bar_width / 2 if has_connector else "",
                    "connector_y_start": top if has_connector else "",
                    "connector_y_end": top if has_connector else "",
                    "artifact_role": request.artifact_role,
                    "transform": "compute_waterfall_geometry",
                }
            )
    return out_rows


def _area_fill_geometry_rows(table: dict[str, Any], request: ArtifactRequest) -> list[dict[str, Any]]:
    rows = _sort_rows_by_column(table.get("rows") or [], request.x_column)
    modifiers = request.semantic_modifiers
    composition = str(modifiers.get("composition") or "additive_stack")
    scale_policy = modifiers.get("scale_policy") if isinstance(modifiers.get("scale_policy"), dict) else {}
    axis_min = _number(scale_policy.get("min")) if scale_policy.get("type") == "explicit_range" else 0.0
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        out: dict[str, Any] = dict(row)
        out["x_index"] = index
        if request.x_column:
            out["x_value"] = row.get(request.x_column)
        cumulative = 0.0
        total = 0.0
        for series in request.series_columns:
            value = _number(row.get(series))
            total += value
            if composition == "overlap":
                bottom = axis_min
                top = value
            else:
                bottom = cumulative
                top = cumulative + value
                cumulative = top
            out[f"{series}_area"] = value
            out[f"{series}_fill_bottom"] = bottom
            out[f"{series}_fill_top"] = top
            out[f"{series}_stack_top"] = top
        out["Total"] = total
        out["composition_policy"] = composition
        out["opacity_policy"] = modifiers.get("opacity") or ""
        out["axis_min"] = scale_policy.get("min") if scale_policy.get("type") == "explicit_range" else ""
        out["axis_max"] = scale_policy.get("max") if scale_policy.get("type") == "explicit_range" else ""
        out["series_columns"] = "|".join(request.series_columns)
        out["artifact_role"] = request.artifact_role
        out["transform"] = "compute_area_fill_geometry"
        out_rows.append(out)
    return out_rows


def _categorical_value_rows(table: dict[str, Any], request: ArtifactRequest) -> list[dict[str, Any]]:
    rows = [dict(row) for row in table.get("rows") or [] if isinstance(row, dict)]
    if request.filter_column and request.filter_values:
        wanted = {_comparable_value(value) for value in request.filter_values}
        rows = [row for row in rows if _comparable_value(row.get(request.filter_column)) in wanted]
    group_column = request.filter_column or "__all__"
    totals: dict[Any, float] = {}
    for row in rows:
        group = _comparable_value(row.get(group_column)) if group_column != "__all__" else "__all__"
        totals[group] = totals.get(group, 0.0) + _number(row.get(request.value_column))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        group = _comparable_value(row.get(group_column)) if group_column != "__all__" else "__all__"
        raw = _number(row.get(request.value_column))
        total = totals.get(group, 0.0)
        out = dict(row)
        out["category"] = row.get(request.category_column)
        out["raw_value"] = raw
        if request.value_column:
            out[f"{request.value_column}_raw"] = raw
        out["Percentage"] = raw / total * 100 if total else 0.0
        out["Pie_autopct_percent"] = raw / total * 100 if total else 0.0
        out["artifact_role"] = request.artifact_role
        out["transform"] = "filter_rows_and_normalize_percent_by_group"
        out_rows.append(out)
    return out_rows


def _artifact_request_metadata(request: ArtifactRequest) -> dict[str, Any]:
    return {
        "artifact_id": request.artifact_id,
        "layer_id": request.layer_id,
        "chart_type": request.chart_type,
        "artifact_role": request.artifact_role,
        "source_table": request.source_table,
        "x_column": request.x_column,
        "series_columns": list(request.series_columns),
        "category_column": request.category_column,
        "value_column": request.value_column,
        "filter_column": request.filter_column,
        "filter_values": list(request.filter_values),
        "transform_ops": list(request.transform_ops),
        "required_for_plotting": request.required_for_plotting,
        "contract_tier": request.contract_tier,
        "contract_reason": _artifact_contract_reason(request),
        "assertions": list(request.assertions),
    }


def _artifact_contract_reason(request: ArtifactRequest) -> str:
    if request.contract_tier == "hard_fidelity":
        if request.artifact_role in {"waterfall_geometry", "area_fill_geometry", "categorical_values"}:
            return "Deterministic expected artifact required to preserve source-grounded plotted values or geometry."
        return "Source-grounded evidence artifact; it may support auditing even when not required for plotting."
    if request.contract_tier == "soft_guidance":
        return "Non-blocking guidance artifact used for readability or visual feedback."
    return "Compatibility or executor-owned design-space artifact; not a blocking fidelity contract."


def _required_columns_for_request(request: ArtifactRequest) -> set[str]:
    required = set(request.series_columns)
    for column in (request.x_column, request.category_column, request.value_column, request.filter_column):
        if column:
            required.add(column)
    return required


def _resolve_source_table(layer: dict[str, Any], tables: dict[str, dict[str, Any]]) -> str | None:
    data_source = str(layer.get("data_source") or "").strip()
    if data_source in tables:
        return data_source
    if data_source:
        basename = Path(data_source).name
        if basename in tables:
            return basename
    chart_type = str(layer.get("chart_type") or "").lower()
    role = str(layer.get("role") or "").lower()
    candidates = list(tables.keys())
    if not candidates:
        return None
    for keyword in _source_keywords_for_layer(chart_type, role):
        match = _match_table_name(tables, keyword)
        if match:
            return match
    return candidates[0] if len(candidates) == 1 else None


def _source_keywords_for_layer(chart_type: str, role: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for source in (role, chart_type):
        for token in source.replace("-", "_").split("_"):
            token = token.strip().lower()
            if token and token not in tokens and token not in {"chart", "layer", "main", "primary", "values"}:
                tokens.append(token)
    return tuple(tokens)


def _resolve_x_column(layer: dict[str, Any], columns: list[str]) -> str | None:
    explicit = _column_if_available(layer.get("x"), columns)
    if explicit:
        return explicit
    encoding = layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {}
    explicit = _column_if_available(encoding.get("x"), columns)
    if explicit:
        return explicit
    for preferred in ("Year", "Date", "Time", "Month", "Category"):
        match = _column_if_available(preferred, columns)
        if match:
            return match
    return columns[0] if columns else None


def _resolve_series_columns(layer: dict[str, Any], table: dict[str, Any], *, x_column: str | None) -> list[str]:
    columns = [str(item) for item in list(table.get("columns") or [])]
    explicit: list[str] = []
    for item in list(layer.get("y") or []):
        match = _column_if_available(item, columns)
        if match and match not in explicit:
            explicit.append(match)
    encoding = layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {}
    for key in ("y", "value", "values", "measure"):
        value = encoding.get(key)
        if isinstance(value, (list, tuple)):
            values = value
        else:
            values = [value]
        for item in values:
            match = _column_if_available(item, columns)
            if match and match not in explicit:
                explicit.append(match)
    series = encoding.get("series")
    if isinstance(series, (list, tuple)):
        for item in series:
            match = _column_if_available(item, columns)
            if match and match not in explicit:
                explicit.append(match)
    if explicit:
        return explicit
    return _numeric_measure_columns(table, exclude={x_column})


def _resolve_category_column(layer: dict[str, Any], table: dict[str, Any]) -> str | None:
    columns = [str(item) for item in list(table.get("columns") or [])]
    encoding = layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {}
    for value in (layer.get("x"), encoding.get("labels"), encoding.get("category"), encoding.get("x")):
        match = _column_if_available(value, columns)
        if match:
            return match
    text_columns = _text_columns(table)
    return text_columns[0] if text_columns else (columns[0] if columns else None)


def _resolve_value_column(layer: dict[str, Any], table: dict[str, Any], *, exclude: set[str | None]) -> str | None:
    columns = [str(item) for item in list(table.get("columns") or [])]
    encoding = layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {}
    for value in (*list(layer.get("y") or []), encoding.get("values"), encoding.get("value"), encoding.get("measure")):
        match = _column_if_available(value, columns)
        if match and match not in exclude:
            return match
    measures = _numeric_measure_columns(table, exclude=exclude)
    return measures[0] if measures else None


def _resolve_filter(layer: dict[str, Any], panel: dict[str, Any], columns: list[str]) -> tuple[str | None, tuple[Any, ...]]:
    encoding = layer.get("encoding") if isinstance(layer.get("encoding"), dict) else {}
    filter_spec = encoding.get("filter") if isinstance(encoding.get("filter"), dict) else {}
    if filter_spec:
        column = next(iter(filter_spec.keys()))
        match = _column_if_available(column, columns)
        if match:
            value = filter_spec.get(column)
            values = tuple(value) if isinstance(value, (list, tuple, set)) else (value,)
            return match, values
    anchor = panel.get("anchor") if isinstance(panel.get("anchor"), dict) else {}
    if anchor.get("value") is not None:
        for preferred in ("Year", "Date", "Time", "Month"):
            match = _column_if_available(preferred, columns)
            if match:
                return match, (anchor.get("value"),)
    return None, ()


def _numeric_measure_columns(table: dict[str, Any], *, exclude: set[str | None]) -> list[str]:
    columns = [str(item) for item in list(table.get("columns") or [])]
    excluded = {str(item) for item in exclude if item}
    rows = [dict(row) for row in list(table.get("rows") or []) if isinstance(row, dict)]
    result: list[str] = []
    for column in columns:
        if column in excluded:
            continue
        if _looks_numeric_column(rows, column):
            result.append(column)
    return result


def _text_columns(table: dict[str, Any]) -> list[str]:
    columns = [str(item) for item in list(table.get("columns") or [])]
    rows = [dict(row) for row in list(table.get("rows") or []) if isinstance(row, dict)]
    result = []
    for column in columns:
        if not _looks_numeric_column(rows, column):
            result.append(column)
    return result


def _looks_numeric_column(rows: list[dict[str, Any]], column: str) -> bool:
    values = [row.get(column) for row in rows[:12] if row.get(column) not in (None, "")]
    if not values:
        return False
    numeric = 0
    for value in values:
        try:
            float(value)
            numeric += 1
        except (TypeError, ValueError):
            pass
    return numeric == len(values)


def _column_if_available(value: Any, columns: list[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in columns:
        return text
    lowered = text.lower()
    for column in columns:
        if column.lower() == lowered:
            return column
    return None


def _request_base(layer_id: str, chart_type: str) -> str:
    safe_layer = _safe_filename(layer_id.replace("layer.", ""))
    safe_type = _safe_filename(chart_type)
    if safe_type and safe_type not in safe_layer:
        return f"artifact_{safe_layer}_{safe_type}"
    return f"artifact_{safe_layer}"


def _legacy_aliases(chart_type: str, artifact_role: str, layer: dict[str, Any]) -> tuple[str, ...]:
    """Temporary compatibility aliases for older tests/runners; not used as the canonical contract."""

    layer_id = str(layer.get("layer_id") or "")
    if chart_type == "waterfall" and artifact_role == "source_values" and layer_id == "layer.import_waterfall":
        return ("step_02_imports_waterfall_values.csv",)
    if chart_type == "waterfall" and artifact_role == "waterfall_geometry" and layer_id == "layer.import_waterfall":
        return ("step_02_imports_waterfall_render_table.csv",)
    if chart_type == "area" and artifact_role == "area_fill_geometry" and layer_id == "layer.consumption_area":
        return ("step_03_consumption_area_values.csv",)
    if chart_type == "pie" and artifact_role == "categorical_values":
        return ("step_04_pie_values.csv",)
    return ()


def _sort_rows_by_column(rows: list[Any], column: str | None) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows if isinstance(row, dict)]
    if not column:
        return normalized
    return sorted(normalized, key=lambda row: (_sort_key(row.get(column)), str(row.get(column))))


def _sort_key(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _comparable_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return str(value)


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
    artifact = {
        "name": name,
        "path": str(path),
        "relative_path": _workspace_relative_path(path),
        "purpose": purpose,
    }
    schema = _artifact_schema(path)
    if schema:
        artifact["schema"] = schema
        artifact["columns"] = list(schema.get("columns") or [])
    return artifact


def _artifact_schema(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".csv" or not path.exists() or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = [str(item) for item in list(reader.fieldnames or [])]
            row_count = sum(1 for _ in reader)
        return {"columns": columns, "row_count": row_count}
    except OSError:
        return {}


def _workspace_relative_path(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part in _WORKSPACE_RELATIVE_PATH_ANCHORS:
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
