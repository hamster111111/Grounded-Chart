from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from grounded_chart.llm import LLMClient, LLMCompletionTrace
from grounded_chart.requirements import ChartRequirementPlan
from grounded_chart.schema import ChartIntentPlan, TableSchema


@dataclass(frozen=True)
class ChartCodeGenerationRequest:
    """Inputs needed to generate executable chart code."""

    query: str
    schema: TableSchema
    rows: tuple[dict[str, Any], ...]
    output_filename: str = "figure.png"
    plan: ChartIntentPlan | None = None
    requirement_plan: ChartRequirementPlan | None = None
    case_id: str = "generation"
    generation_mode: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChartCodeGeneration:
    """Generated chart code plus provenance from the generator."""

    code: str
    generator_name: str = "unknown"
    backend_hint: str | None = None
    instruction: str | None = None
    llm_trace: LLMCompletionTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChartCodeGenerator(Protocol):
    def generate(self, request: ChartCodeGenerationRequest) -> ChartCodeGeneration:
        """Generate executable Python plotting code for one request."""


class StaticChartCodeGenerator:
    """Deterministic generator used by tests and smoke runs."""

    def __init__(self, code: str, *, generator_name: str = "static") -> None:
        self.code = code
        self.generator_name = generator_name

    def generate(self, request: ChartCodeGenerationRequest) -> ChartCodeGeneration:
        return ChartCodeGeneration(
            code=self.code,
            generator_name=self.generator_name,
            backend_hint="matplotlib" if "matplotlib" in self.code or "plt." in self.code else None,
            metadata={"case_id": request.case_id},
        )


class LLMChartCodeGenerator:
    """OpenAI-compatible ExecutorAgent for chart creation.

    The generated code must faithfully execute the construction plan, data
    contract, and output contract. It is not allowed to freely reinterpret the
    figure structure after planning.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        generator_name: str = "llm_chart_codegen_v1",
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.generator_name = generator_name
        self.max_tokens = max_tokens

    def generate(self, request: ChartCodeGenerationRequest) -> ChartCodeGeneration:
        result = self.client.complete_json_with_trace(
            system_prompt=_codegen_system_prompt(),
            user_prompt=_codegen_user_prompt(request),
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        payload = result.payload
        code = str(payload.get("code") or "").strip()
        if not code:
            raise ValueError("Code generator returned empty `code`.")
        return ChartCodeGeneration(
            code=_strip_code_fence(code),
            generator_name=str(payload.get("generator_name") or self.generator_name),
            backend_hint=str(payload.get("backend") or "").strip() or None,
            instruction=str(payload.get("instruction") or "").strip() or None,
            llm_trace=result.trace,
            metadata={
                "notes": payload.get("notes"),
                "assumptions": payload.get("assumptions"),
            },
        )


def _codegen_system_prompt() -> str:
    return (
        "You are ExecutorAgent for a grounded chart pipeline. "
        "Your job is strict execution of the provided plan, not open-ended redesign. "
        "Return only a JSON object with keys: code, backend, instruction, notes, assumptions. "
        "When table rows are provided, the code must use the global variable `rows`, a list of dictionaries. "
        "When source_data_plan is provided, the code must read the listed copied files by relative filename and must not fabricate substitute data. "
        "When construction_plan is provided, follow its whole-figure layout, panel, layer, inset, legend, and title decisions. "
        "If construction_plan provides semantic placement_policy/anchor/avoid_occlusion rather than numeric bounds, compute concrete layout coordinates yourself from figure size, axes, legends, titles, data density, and inset count. "
        "Record those concrete layout decisions in a JSON file named computed_layout.json and a Markdown file named layout_decisions.md in the same directory as OUTPUT_PATH. "
        "computed_layout.json should map plan refs such as panel.main or panel.pie_2008 to computed bounds/positions and short reasons. "
        "When context.layout_replanning is provided, this is a PlanAgent replanning round: preserve source-grounded data and explicit requirements, but make concrete rendered changes required by layout_replanning.feedback_bundle. "
        "Do not satisfy replanning feedback by changing only comments, metadata, or artifact round paths. "
        "In layout_replanning mode, do not change source file reads, prepared artifact paths, deterministic data transformations, plotted values, source files, required labels, or required legend categories. "
        "When artifact_workspace is provided, treat its listed plan_dir and execution_dir as the working protocol; the final plot code must read prepared plotting CSV artifacts using each artifact's `relative_path` (for example execution/round_1/artifact_*.csv), not a bare filename. "
        "Do not assume prepared artifacts are in the current directory; they are under the artifact_workspace execution_dir. "
        "The final plot code must not recompute prepared plotted values from raw source CSVs. "
        "Artifact metadata is tiered: hard_fidelity artifacts are binding data/geometry inputs; soft_guidance artifacts are advisory; free_design artifacts are optional compatibility/design-space material. Choose artifacts by artifact_role, chart_type, layer_id, contract_tier, and schema.columns; do not rely on fixed benchmark filenames or guessed column names. "
        "Artifact schemas are binding: use only columns listed in context.artifact_workspace.artifacts[].schema.columns/columns unless your code explicitly creates a documented derived intermediate table. "
        "When artifact_workspace contains chart_protocols, follow hard_fidelity protocol commitments as binding rendering semantics; use soft_guidance for readability, and treat free_design as Executor-owned choices. "
        "When chart_protocols include visual_channel_contracts, treat hard contracts as semantic bindings only: preserve which field controls a channel, but choose palette/line style/spacing yourself unless explicitly specified. "
        "Record actual hard visual-channel choices in visual_channel_decisions.json next to OUTPUT_PATH when you implement fill_color, line_color, marker_shape, hatch, edge_style, alpha, or x_offset from a contract. "
        "Prepared artifacts listed in artifact_workspace are read-only for plotting code; do not overwrite framework-prepared CSV files with to_csv/open/write. "
        "For waterfall charts, if an artifact has artifact_role=waterfall_geometry, use it as the bar geometry table with bottom=bar_bottom and height=bar_height; do not draw source_values artifacts as ordinary zero-based bars. "
        "For waterfall charts, do not assume terminal/total geometry roles must use a distinct fill color; follow chart_protocols.visual_channel_policy and artifact fields such as fill_color_role, series_color_role, series, and change_role. "
        "For area charts, use semantic_modifiers and composition_policy from prepared artifacts. If composition_policy is overlap, draw independent translucent fill_between layers using the artifact's per-series *_fill_bottom and *_fill_top columns; do not silently stack series by addition. "
        "For overlaid layers and twinx axes, use the same x-coordinate basis everywhere; do not plot bars at 0..N while plotting areas/lines at raw years. "
        "When generation_mode is instruction_only, `rows` may be empty; then use only constants, labels, and structures explicitly stated in the request. "
        "The code must save the final figure or interactive chart to the provided global variable `OUTPUT_PATH`. "
        "Prefer matplotlib unless the request explicitly needs Plotly or another backend. "
        "Do not call network APIs, fabricate unstated data, or ignore the schema/source-data constraints. "
        "Do not drop planned panels, layers, insets, axes, legends, or titles unless the plan marks them as unsupported. "
        "For any arithmetic or aggregation over table rows, write Python code that computes from `rows`; do not hard-code derived values. "
        "Keep the code self-contained and deterministic."
    )


def _codegen_user_prompt(request: ChartCodeGenerationRequest) -> str:
    generation_mode = request.generation_mode or _infer_generation_mode(request)
    payload = {
        "case_id": request.case_id,
        "query": request.query,
        "generation_mode": generation_mode,
        "schema": {
            "table_name": request.schema.table_name,
            "columns": dict(request.schema.columns),
        },
        "row_count": len(request.rows),
        "row_sample": [_jsonable(row) for row in request.rows[:12]],
        "output_contract": {
            "rows_variable": "rows",
            "output_path_variable": "OUTPUT_PATH",
            "output_filename": request.output_filename,
        },
        "context": _jsonable(request.context),
        "parsed_plan": _plan_payload(request.plan),
        "requirements": _requirements_payload(request.requirement_plan),
        "implementation_rules": _implementation_rules(generation_mode),
    }
    return (
        "Generate chart code for this request. The verifier will execute the code with "
        "globals `rows` and `OUTPUT_PATH`, then compare actual artifacts against parsed requirements.\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _infer_generation_mode(request: ChartCodeGenerationRequest) -> str:
    if request.rows or request.schema.columns:
        return "table"
    return "instruction_only"


def _implementation_rules(generation_mode: str) -> list[str]:
    common = [
        "Write executable Python code, not pseudocode.",
        "Create all required labels, titles, legends, scales, annotations, and subplot structure when requested.",
        "Treat context.construction_plan as the execution blueprint for figure layout, panels, visual layers, and inferred placement decisions.",
        "Inferred construction_plan decisions may fill missing layout details, but must not contradict explicit requirements.",
        "When layout coordinates are not hard-coded in the plan, compute them in code and save computed_layout.json plus layout_decisions.md next to OUTPUT_PATH.",
        "The computed layout record should include each main panel/inset/global element, its computed coordinates or placement, the source plan refs used, and the reason for the decision.",
        "Implement every explicit construction_plan layer; if a layer cannot be implemented, add a short assumption explaining why.",
        "When multiple layers share an axis or use twinx, define one x variable and reuse it for bars, areas, lines, ticks, and inset anchor calculations.",
        "If using index positions for years, map every data table onto those positions before plotting; if using raw years, use raw years for every overlaid layer.",
        "If context.artifact_workspace lists execution artifacts, read the relevant required_for_plotting CSV files by their artifact relative_path for plotting instead of recomputing equivalent data from source CSVs. This is mandatory, not optional.",
        "Only hard_fidelity artifacts and explicit source requirements are blocking contracts. Soft_guidance should improve readability; free_design is delegated to your judgment.",
        "Before using a prepared CSV column, check context.artifact_workspace artifact schemas. Do not assume long-form columns such as series/value exist when the artifact schema is wide.",
        "If chart protocol files are listed in context.artifact_workspace, treat hard_fidelity items as binding chart-type instructions. In particular, hard_fidelity waterfall geometry requires a render table and explicit bar bottoms.",
        "If chart protocols expose visual_channel_contracts, do not turn them into fixed aesthetics. Use hard contracts only to bind channels to semantic fields, then record your actual mapping in visual_channel_decisions.json.",
        "If context.layout_replanning exists, use context.layout_replanning.previous_code as the baseline, but implement the feedback_bundle through visible rendering changes.",
        "In layout_replanning mode, preserve source-grounded data and deterministic artifacts, but revise layout, visual channels, legend/inset placement, rendering semantics, and readability details when needed to address PlanAgent feedback.",
        "If a replanning candidate only changes comments, metadata, or artifact round paths, it has failed the replanning task.",
        "For waterfall charts with artifact_role=waterfall_geometry, plot from x_position, bar_height, bar_bottom, bar_width, role, change_role, fill_color_role, and series when those columns exist. Use the protocol visual_channel_policy to choose the color field; if fill_color_role is present, prefer it over hard-coded change-direction colors. Do not use source_values artifacts as ordinary bar heights.",
        "When fill_color is bound to a semantic field such as fill_color_role or series, keep the same field value visually consistent across the layer and do not introduce extra fill-color legend categories such as total unless requested.",
        "For area charts with artifact_role=area_fill_geometry, discover per-series fill columns from schema columns ending with _fill_bottom/_fill_top. Respect composition_policy; overlap means independent translucent areas, additive_stack means cumulative stacking.",
        "If an explicit axis range is provided in the prepared area artifact columns axis_min/axis_max, apply it to the bound axis.",
        "Do not overwrite framework-prepared CSV artifacts. If a new intermediate table is needed, write a new filename and explain why in markdown.",
        "Avoid calling tight_layout/constrained_layout after manually adding inset axes unless the code explicitly preserves all axes positions.",
        "Save exactly one final artifact to `OUTPUT_PATH` when possible.",
        "If using matplotlib, call `fig.savefig(OUTPUT_PATH, bbox_inches='tight')` or `plt.savefig(OUTPUT_PATH, bbox_inches='tight')`.",
        "If using Plotly and static export is unavailable, write HTML to `OUTPUT_PATH` with an .html suffix only if necessary.",
    ]
    if generation_mode == "instruction_only":
        return [
            "This is an instruction-only plotting task: `rows` may be empty and schema may have no columns.",
            "Use only constants, labels, layout constraints, and visual structures explicitly stated in the query or context.",
            "Do not invent extra data series or semantic values that the request does not specify.",
            *common,
        ]
    if generation_mode == "source_file_grounded":
        return [
            "This is a source-file-grounded plotting task.",
            "Read each relevant file listed in context.source_data_plan by relative filename from the execution directory.",
            "Use the exact file schemas and previews in context.source_data_plan/source_data_execution.",
            "Do not create random, dummy, sample, placeholder, or synthetic replacement data for listed files.",
            "If a table is wide, use its existing measure columns directly or explicitly melt it before long-form plotting.",
            *common,
        ]
    return [
        "Use only `rows` as the data source.",
        "Compute filters, grouping, sorting, aggregations, and derived plotted values from `rows` in code.",
        *common,
    ]


def _plan_payload(plan: ChartIntentPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "chart_type": plan.chart_type,
        "dimensions": list(plan.dimensions),
        "measure": {"column": plan.measure.column, "agg": plan.measure.agg},
        "filters": [
            {"column": item.column, "op": item.op, "value": _jsonable(item.value)}
            for item in plan.filters
        ],
        "sort": {"by": plan.sort.by, "direction": plan.sort.direction} if plan.sort is not None else None,
        "limit": plan.limit,
        "confidence": plan.confidence,
        "raw_query": plan.raw_query,
    }


def _requirements_payload(requirement_plan: ChartRequirementPlan | None) -> list[dict[str, Any]]:
    if requirement_plan is None:
        return []
    return [
        {
            "requirement_id": requirement.requirement_id,
            "scope": requirement.scope,
            "type": requirement.type,
            "name": requirement.name,
            "value": _jsonable(requirement.value),
            "source_span": requirement.source_span,
            "status": requirement.status,
            "confidence": requirement.confidence,
            "depends_on": list(requirement.depends_on),
            "priority": requirement.priority,
            "panel_id": requirement.panel_id,
            "assumption": requirement.assumption,
            "severity": requirement.severity,
            "match_policy": requirement.match_policy,
        }
        for requirement in requirement_plan.requirements
    ]


def _strip_code_fence(text: str) -> str:
    stripped = str(text or "").strip()
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)
