from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from grounded_chart.codegen import ChartCodeGeneration, ChartCodeGenerationRequest, ChartCodeGenerator
from grounded_chart.evidence import build_requirement_plan
from grounded_chart.llm import LLMCompletionTrace, LLMUsage
from grounded_chart.patch_ops import apply_patch_operations
from grounded_chart.pipeline import GroundedChartPipeline
from grounded_chart.rendering import ChartImageRenderer, ChartRenderResult
from grounded_chart.schema import FigureRequirementSpec, ParsedRequirementBundle, PipelineResult, TableSchema


@dataclass(frozen=True)
class ChartGenerationPipelineResult:
    """End-to-end result for instruction-to-image generation."""

    case_id: str
    query: str
    output_dir: Path
    code_generation: ChartCodeGeneration
    pipeline_result: PipelineResult
    render_result: ChartRenderResult
    initial_code_path: Path
    final_code_path: Path
    report_path: Path
    manifest_path: Path
    case_report: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_path(self) -> Path | None:
        return self.render_result.image_path

    @property
    def final_code(self) -> str:
        return self.final_code_path.read_text(encoding="utf-8")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "output_dir": str(self.output_dir),
            "image_path": str(self.image_path) if self.image_path is not None else None,
            "initial_code_path": str(self.initial_code_path),
            "final_code_path": str(self.final_code_path),
            "report_path": str(self.report_path),
            "generator": {
                "name": self.code_generation.generator_name,
                "backend_hint": self.code_generation.backend_hint,
                "instruction": self.code_generation.instruction,
                "metadata": _jsonable(self.code_generation.metadata),
                "llm_trace": _llm_trace_to_dict(self.code_generation.llm_trace),
            },
            "render": _render_result_to_dict(self.render_result),
            "verification": {
                "ok": self.pipeline_result.report.ok,
                "error_codes": list(self.pipeline_result.report.error_codes),
                "repair_loop_status": self.pipeline_result.repair_loop_status,
                "repair_loop_reason": self.pipeline_result.repair_loop_reason,
                "repair_attempt_count": len(self.pipeline_result.repair_attempts),
                "execution_exception_type": self.pipeline_result.execution_exception_type,
                "execution_exception_message": self.pipeline_result.execution_exception_message,
                "parse_source": self.pipeline_result.parse_source,
                "final_code_source": self.metadata.get("final_code_source"),
                "final_code_verified": self.metadata.get("final_code_verified"),
            },
            "metadata": _jsonable(self.metadata),
        }

class ChartRenderStageError(RuntimeError):
    """Wrap final artifact export failures so the repairer can handle them."""


class ChartGenerationPipeline:
    """Instruction -> code -> trace/verify/repair -> rendered image."""

    def __init__(
        self,
        *,
        code_generator: ChartCodeGenerator,
        verifier_pipeline,
        renderer: ChartImageRenderer | None = None,
    ) -> None:
        self.code_generator = code_generator
        self.verifier_pipeline = verifier_pipeline
        self.renderer = renderer or ChartImageRenderer()

    def run(
        self,
        *,
        query: str,
        schema: TableSchema,
        rows: Iterable[dict[str, Any]],
        output_dir: str | Path,
        case_id: str = "generation",
        expected_figure: FigureRequirementSpec | None = None,
        verification_mode: str = "full",
        parsed_requirements: ParsedRequirementBundle | None = None,
        parse_source: str | None = None,
        output_filename: str | None = None,
        generation_mode: str | None = None,
        generation_context: dict[str, Any] | None = None,
    ) -> ChartGenerationPipelineResult:
        row_tuple = tuple(dict(row) for row in rows)
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        image_name = output_filename or f"{_safe_name(case_id)}.png"
        effective_parse_source = parse_source or ("oracle" if parsed_requirements is not None else "predicted")
        parsed = parsed_requirements or _parse_requirements(self.verifier_pipeline.parser, query, schema)

        request = ChartCodeGenerationRequest(
            query=query,
            schema=schema,
            rows=row_tuple,
            output_filename=image_name,
            plan=parsed.plan,
            requirement_plan=parsed.requirement_plan,
            case_id=case_id,
            generation_mode=generation_mode or _infer_generation_mode(schema, row_tuple),
            context=dict(generation_context or {}),
        )
        code_generation = self.code_generator.generate(request)
        initial_code_path = output_root / "generated_initial.py"
        initial_code_path.write_text(code_generation.code, encoding="utf-8")

        verify_data = verification_mode in {"full", "figure_and_data"}
        pipeline_result = self._run_verification(
            code=code_generation.code,
            query=query,
            schema=schema,
            rows=row_tuple,
            output_root=output_root,
            image_name=image_name,
            file_path=initial_code_path,
            expected_figure=expected_figure,
            verify_data=verify_data,
            parsed=parsed,
            parse_source=effective_parse_source,
            allow_repair=True,
        )
        final_code, pipeline_result, final_contract = self._resolve_final_code(
            initial_code=code_generation.code,
            initial_result=pipeline_result,
            query=query,
            schema=schema,
            rows=row_tuple,
            output_root=output_root,
            image_name=image_name,
            expected_figure=expected_figure,
            verify_data=verify_data,
            parsed=parsed,
            parse_source=effective_parse_source,
        )

        final_code_path = output_root / "generated_final.py"
        final_code_path.write_text(final_code, encoding="utf-8")
        render_result = self.renderer.render(
            final_code,
            rows=row_tuple,
            output_dir=output_root,
            output_filename=image_name,
            file_path=final_code_path,
        )
        final_code, pipeline_result, render_result, render_contract = self._resolve_render_failure(
            final_code=final_code,
            pipeline_result=pipeline_result,
            render_result=render_result,
            query=query,
            schema=schema,
            rows=row_tuple,
            output_root=output_root,
            image_name=image_name,
            final_code_path=final_code_path,
            expected_figure=expected_figure,
            verify_data=verify_data,
            parsed=parsed,
            parse_source=effective_parse_source,
        )
        if render_contract.get("render_repair_accepted"):
            final_code_path.write_text(final_code, encoding="utf-8")
        final_contract = {**final_contract, **render_contract}

        from grounded_chart_adapters.base import AdapterRunResult, ChartCase
        from grounded_chart_adapters.reporting import case_report_from_result

        chart_case = ChartCase(
            case_id=case_id,
            query=query,
            schema=schema,
            rows=row_tuple,
            generated_code=final_code,
            figure_requirements=expected_figure,
            verification_mode=verification_mode,  # type: ignore[arg-type]
            parsed_requirements=parsed,
            parse_source=effective_parse_source,  # type: ignore[arg-type]
            metadata={
                "generation_pipeline": True,
                "generator": code_generation.generator_name,
                "image_path": str(render_result.image_path) if render_result.image_path is not None else None,
                "initial_code_path": str(initial_code_path),
                "final_code_path": str(final_code_path),
                "generation_mode": request.generation_mode,
                "generation_context": _jsonable(request.context),
                **final_contract,
            },
        )
        case_report = case_report_from_result(
            AdapterRunResult(
                case=chart_case,
                pipeline_result=pipeline_result,
                metadata={
                    "initial_code_path": str(initial_code_path),
                    "final_code_path": str(final_code_path),
                    "render": _render_result_to_dict(render_result),
                    "final_contract": final_contract,
                },
            )
        ).to_dict()
        report_path = output_root / "generation_report.json"
        report_path.write_text(json.dumps(case_report, ensure_ascii=False, indent=2), encoding="utf-8")

        result = ChartGenerationPipelineResult(
            case_id=case_id,
            query=query,
            output_dir=output_root,
            code_generation=code_generation,
            pipeline_result=pipeline_result,
            render_result=render_result,
            initial_code_path=initial_code_path,
            final_code_path=final_code_path,
            report_path=report_path,
            manifest_path=output_root / "generation_manifest.json",
            case_report=case_report,
            metadata={"verification_mode": verification_mode, "generation_mode": request.generation_mode, **final_contract},
        )
        result.manifest_path.write_text(json.dumps(result.to_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _run_verification(
        self,
        *,
        code: str,
        query: str,
        schema: TableSchema,
        rows: tuple[dict[str, Any], ...],
        output_root: Path,
        image_name: str,
        file_path: Path,
        expected_figure: FigureRequirementSpec | None,
        verify_data: bool,
        parsed: ParsedRequirementBundle,
        parse_source: str,
        allow_repair: bool,
    ) -> PipelineResult:
        pipeline = self.verifier_pipeline if allow_repair else self._verification_only_pipeline()
        trace_output_path = output_root / image_name
        globals_dict = {
            "rows": [dict(row) for row in rows],
            "OUTPUT_PATH": str(trace_output_path),
        }
        try:
            run_trace = pipeline.trace_runner.run_code_with_figure(
                code,
                globals_dict=globals_dict,
                execution_dir=output_root,
                file_path=file_path,
            )
            return pipeline.run(
                query=query,
                schema=schema,
                rows=rows,
                actual_trace=run_trace.plot_trace,
                generated_code=code,
                expected_figure=expected_figure,
                actual_figure=run_trace.figure_trace,
                verify_data=verify_data,
                execution_dir=str(output_root),
                file_path=str(file_path),
                parsed_requirements=parsed,
                parse_source=parse_source,
            )
        except Exception as exc:
            return pipeline.run_with_execution_error(
                query=query,
                schema=schema,
                rows=rows,
                generated_code=code,
                execution_exception=exc,
                expected_figure=expected_figure,
                verify_data=verify_data,
                execution_dir=str(output_root),
                file_path=str(file_path),
                parsed_requirements=parsed,
                parse_source=parse_source,
            )

    def _resolve_render_failure(
        self,
        *,
        final_code: str,
        pipeline_result: PipelineResult,
        render_result: ChartRenderResult,
        query: str,
        schema: TableSchema,
        rows: tuple[dict[str, Any], ...],
        output_root: Path,
        image_name: str,
        final_code_path: Path,
        expected_figure: FigureRequirementSpec | None,
        verify_data: bool,
        parsed: ParsedRequirementBundle,
        parse_source: str,
    ) -> tuple[str, PipelineResult, ChartRenderResult, dict[str, Any]]:
        if render_result.ok:
            return final_code, pipeline_result, render_result, {
                "render_repair_attempted": False,
                "render_repair_accepted": False,
            }

        exception_type = render_result.exception_type or "RenderError"
        exception_message = render_result.exception_message or "No visible chart artifact was produced."
        contract: dict[str, Any] = {
            "render_repair_attempted": False,
            "render_repair_accepted": False,
            "render_repair_reason": "render_failed",
            "render_repair_exception_type": exception_type,
            "render_repair_exception_message": exception_message,
        }
        if getattr(self.verifier_pipeline, "repairer", None) is None:
            contract["render_repair_reason"] = "no_repairer"
            return final_code, pipeline_result, render_result, contract

        contract["render_repair_attempted"] = True
        repair_exception = ChartRenderStageError(
            f"Render-stage artifact export failed after verification: {exception_type}: {exception_message}"
        )
        render_error_result = self._render_repair_pipeline().run_with_execution_error(
            query=query,
            schema=schema,
            rows=rows,
            generated_code=final_code,
            execution_exception=repair_exception,
            expected_figure=expected_figure,
            verify_data=verify_data,
            execution_dir=str(output_root),
            file_path=str(final_code_path),
            case_metadata={
                "repair_stage": "render",
                "render": _render_result_to_dict(render_result),
            },
            parsed_requirements=parsed,
            parse_source=parse_source,
            expected_trace_override=pipeline_result.expected_trace,
        )
        repair = render_error_result.repair
        llm_candidate_code, llm_candidate_contract = _repair_candidate_code(final_code, repair)
        contract.update(
            {
                "render_repair_error_codes": list(render_error_result.report.error_codes),
                "render_repair_strategy": repair.strategy if repair else None,
                "render_repair_plan_scope": render_error_result.repair_plan.scope if render_error_result.repair_plan else None,
                **llm_candidate_contract,
            }
        )

        candidate_specs: list[tuple[str, str, Path, dict[str, Any]]] = []
        if llm_candidate_code and llm_candidate_code.strip() != final_code.strip():
            candidate_specs.append(
                (
                    "repairer",
                    llm_candidate_code,
                    output_root / "generated_render_repair_candidate.py",
                    llm_candidate_contract,
                )
            )
        safety_candidate_code, safety_candidate_contract = _render_export_safety_candidate_code(final_code)
        if safety_candidate_code and safety_candidate_code.strip() != final_code.strip():
            if not any(safety_candidate_code.strip() == existing_code.strip() for _, existing_code, _, _ in candidate_specs):
                candidate_specs.append(
                    (
                        "export_safety",
                        safety_candidate_code,
                        output_root / "generated_render_safety_candidate.py",
                        safety_candidate_contract,
                    )
                )
        if not candidate_specs:
            contract["render_repair_reason"] = (
                llm_candidate_contract.get("render_repair_candidate_rejected_reason")
                or safety_candidate_contract.get("render_repair_candidate_rejected_reason")
                or "no_candidate_code"
            )
            return final_code, pipeline_result, render_result, contract

        original_score = _verification_score(pipeline_result)
        last_contract: dict[str, Any] = {}
        for candidate_label, candidate_code, candidate_path, candidate_source_contract in candidate_specs:
            candidate_path.write_text(candidate_code, encoding="utf-8")
            candidate_result = self._run_verification(
                code=candidate_code,
                query=query,
                schema=schema,
                rows=rows,
                output_root=output_root,
                image_name=image_name,
                file_path=candidate_path,
                expected_figure=expected_figure,
                verify_data=verify_data,
                parsed=parsed,
                parse_source=parse_source,
                allow_repair=False,
            )
            candidate_result = replace(
                candidate_result,
                repair_plan=render_error_result.repair_plan,
                repair=render_error_result.repair,
                repaired_code=candidate_code,
            )
            candidate_render = self.renderer.render(
                candidate_code,
                rows=rows,
                output_dir=output_root,
                output_filename=image_name,
                file_path=candidate_path,
            )
            candidate_score = _verification_score(candidate_result)
            verification_not_regressed = candidate_score <= original_score
            accepted = candidate_render.ok and verification_not_regressed
            last_contract = {
                **candidate_source_contract,
                "render_repair_candidate_label": candidate_label,
                "render_repair_candidate_path": str(candidate_path),
                "render_repair_candidate_render": _render_result_to_dict(candidate_render),
                "render_repair_initial_verification_score": original_score,
                "render_repair_candidate_verification_score": candidate_score,
                "render_repair_verification_not_regressed": verification_not_regressed,
                "render_repair_accepted": accepted,
                "render_repair_reason": (
                    "accepted"
                    if accepted
                    else "candidate_render_failed"
                    if not candidate_render.ok
                    else "candidate_verification_regressed"
                ),
            }
            contract.update(last_contract)
            if accepted:
                contract.update(
                    {
                        "final_code_source": "render_repair_candidate" if candidate_label == "repairer" else f"render_{candidate_label}_candidate",
                        "final_code_verified": True,
                        "final_verification_score": candidate_score,
                    }
                )
                return candidate_code, candidate_result, candidate_render, contract
        contract.update(last_contract)
        return final_code, pipeline_result, render_result, contract

    def _render_repair_pipeline(self) -> GroundedChartPipeline:
        return GroundedChartPipeline(
            parser=self.verifier_pipeline.parser,
            executor=self.verifier_pipeline.executor,
            verifier=self.verifier_pipeline.verifier,
            repairer=self.verifier_pipeline.repairer,
            trace_runner=self.verifier_pipeline.trace_runner,
            enable_bounded_repair_loop=False,
            max_repair_rounds=1,
            repair_policy_mode=getattr(self.verifier_pipeline, "repair_policy_mode", "strict"),
            repair_planner=getattr(self.verifier_pipeline, "repair_planner", None),
        )

    def _verification_only_pipeline(self) -> GroundedChartPipeline:
        return GroundedChartPipeline(
            parser=self.verifier_pipeline.parser,
            executor=self.verifier_pipeline.executor,
            verifier=self.verifier_pipeline.verifier,
            repairer=None,
            trace_runner=self.verifier_pipeline.trace_runner,
            enable_bounded_repair_loop=False,
            repair_policy_mode=getattr(self.verifier_pipeline, "repair_policy_mode", "strict"),
        )

    def _resolve_final_code(
        self,
        *,
        initial_code: str,
        initial_result: PipelineResult,
        query: str,
        schema: TableSchema,
        rows: tuple[dict[str, Any], ...],
        output_root: Path,
        image_name: str,
        expected_figure: FigureRequirementSpec | None,
        verify_data: bool,
        parsed: ParsedRequirementBundle,
        parse_source: str,
    ) -> tuple[str, PipelineResult, dict[str, Any]]:
        if initial_result.repaired_code:
            return initial_result.repaired_code, initial_result, {
                "final_code_source": "bounded_repair_loop",
                "final_code_verified": True,
                "final_verification_score": _verification_score(initial_result),
            }

        candidate_code = initial_result.repair.repaired_code if initial_result.repair else None
        if not candidate_code or candidate_code.strip() == initial_code.strip():
            return initial_code, initial_result, {
                "final_code_source": "initial_code",
                "final_code_verified": True,
                "one_shot_repair_candidate": False,
                "final_verification_score": _verification_score(initial_result),
            }

        candidate_path = output_root / "generated_repair_candidate.py"
        candidate_path.write_text(candidate_code, encoding="utf-8")
        candidate_result = self._run_verification(
            code=candidate_code,
            query=query,
            schema=schema,
            rows=rows,
            output_root=output_root,
            image_name=image_name,
            file_path=candidate_path,
            expected_figure=expected_figure,
            verify_data=verify_data,
            parsed=parsed,
            parse_source=parse_source,
            allow_repair=False,
        )
        candidate_result = replace(
            candidate_result,
            repair_plan=initial_result.repair_plan,
            repair=initial_result.repair,
            repaired_code=candidate_code,
        )
        original_score = _verification_score(initial_result)
        candidate_score = _verification_score(candidate_result)
        accepted = candidate_result.report.ok or candidate_score < original_score
        contract = {
            "final_code_source": "one_shot_repair_candidate" if accepted else "initial_code",
            "final_code_verified": True,
            "one_shot_repair_candidate": True,
            "one_shot_repair_accepted": accepted,
            "one_shot_repair_candidate_path": str(candidate_path),
            "initial_verification_score": original_score,
            "candidate_verification_score": candidate_score,
            "final_verification_score": candidate_score if accepted else original_score,
        }
        if accepted:
            return candidate_code, candidate_result, contract
        return initial_code, initial_result, contract



def _repair_candidate_code(final_code: str, repair) -> tuple[str | None, dict[str, Any]]:
    if repair is None:
        return None, {"render_repair_candidate_source": None, "render_repair_candidate_rejected_reason": "no_repair_patch"}
    if repair.repaired_code and repair.repaired_code.strip():
        return repair.repaired_code, {
            "render_repair_candidate_source": "repaired_code",
            "render_repair_patch_ops_count": len(repair.patch_ops),
        }
    if repair.patch_ops:
        patch_result = apply_patch_operations(
            final_code,
            repair.patch_ops,
            max_operations=4,
            max_changed_lines=25,
        )
        contract = {
            "render_repair_candidate_source": "patch_ops",
            "render_repair_patch_ops_count": len(repair.patch_ops),
            "render_repair_patch_applied": patch_result.applied,
            "render_repair_patch_rejected_reason": patch_result.rejected_reason,
        }
        if patch_result.applied:
            return patch_result.code, contract
        contract["render_repair_candidate_rejected_reason"] = patch_result.rejected_reason or "patch_ops_not_applied"
        return None, contract
    return None, {
        "render_repair_candidate_source": None,
        "render_repair_patch_ops_count": 0,
        "render_repair_candidate_rejected_reason": "no_candidate_code",
    }


def _render_export_safety_candidate_code(final_code: str) -> tuple[str | None, dict[str, Any]]:
    patterns = (
        re.compile(r",\s*bbox_inches\s*=\s*(['\"])tight\1"),
        re.compile(r"bbox_inches\s*=\s*(['\"])tight\1\s*,\s*"),
    )
    candidate = final_code
    replacements = 0
    for pattern in patterns:
        candidate, count = pattern.subn("", candidate)
        replacements += count
    if replacements <= 0 or candidate == final_code:
        return None, {
            "render_repair_export_safety_attempted": False,
            "render_repair_candidate_rejected_reason": "no_export_safety_patch",
        }
    return candidate, {
        "render_repair_candidate_source": "export_safety_patch",
        "render_repair_export_safety_attempted": True,
        "render_repair_export_safety_patch": "remove_tight_bbox_savefig",
        "render_repair_export_safety_replacements": replacements,
    }


def _infer_generation_mode(schema: TableSchema, rows: tuple[dict[str, Any], ...]) -> str:
    if rows or schema.columns:
        return "table"
    return "instruction_only"

def _parse_requirements(parser, query: str, schema: TableSchema) -> ParsedRequirementBundle:
    parse_requirements = getattr(parser, "parse_requirements", None)
    if callable(parse_requirements):
        return parse_requirements(query, schema)
    plan = parser.parse(query, schema)
    return ParsedRequirementBundle(plan=plan, requirement_plan=build_requirement_plan(plan))


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe.strip("_") or "generation"


def _verification_score(result: PipelineResult) -> tuple[int, int, int]:
    execution_penalty = 1 if _has_execution_failure(result) else 0
    hard_errors = sum(1 for error in result.report.errors if error.severity == "error")
    total_errors = len(result.report.errors)
    return (execution_penalty, hard_errors, total_errors)


def _has_execution_failure(result: PipelineResult) -> bool:
    if result.execution_exception_type:
        return True
    return any(error.code == "execution_error" for error in result.report.errors)


def _render_result_to_dict(render: ChartRenderResult) -> dict[str, Any]:
    return {
        "ok": render.ok,
        "image_path": str(render.image_path) if render.image_path is not None else None,
        "artifact_paths": [str(path) for path in render.artifact_paths],
        "backend": render.backend,
        "stdout": render.stdout,
        "stderr": render.stderr,
        "exception_type": render.exception_type,
        "exception_message": render.exception_message,
        "metadata": _jsonable(render.metadata),
    }


def _llm_trace_to_dict(trace: LLMCompletionTrace | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    return {
        "provider": trace.provider,
        "model": trace.model,
        "base_url": trace.base_url,
        "temperature": trace.temperature,
        "max_tokens": trace.max_tokens,
        "usage": _usage_to_dict(trace.usage),
        "parsed_json": _jsonable(trace.parsed_json),
        "raw_text_preview": str(trace.raw_text or "")[:1200],
    }


def _usage_to_dict(usage: LLMUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "raw": _jsonable(usage.raw),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
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










