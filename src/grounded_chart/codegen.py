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
    """OpenAI-compatible LLM code generator for chart creation.

    The generated code is constrained to use the in-memory `rows` variable and
    save a final artifact to `OUTPUT_PATH`. This keeps downstream tracing,
    verification, repair, and rendering tied to the same data contract.
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
        "You generate executable Python chart code for a grounded chart pipeline. "
        "Return only a JSON object with keys: code, backend, instruction, notes, assumptions. "
        "When table rows are provided, the code must use the global variable `rows`, a list of dictionaries. "
        "When generation_mode is instruction_only, `rows` may be empty; then use only constants, labels, and structures explicitly stated in the request. "
        "The code must save the final figure or interactive chart to the provided global variable `OUTPUT_PATH`. "
        "Prefer matplotlib unless the request explicitly needs Plotly or another backend. "
        "Do not read external files, call network APIs, fabricate unstated data, or ignore the schema. "
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
