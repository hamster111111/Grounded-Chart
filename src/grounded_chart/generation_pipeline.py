from __future__ import annotations

import json
import math
import re
import shutil
import html
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from grounded_chart.artifact_workspace import ArtifactWorkspaceBuilder, LAYOUT_AGENT_DIR
from grounded_chart.chart_protocol import ChartProtocolAgent
from grounded_chart.codegen import ChartCodeGeneration, ChartCodeGenerationRequest, ChartCodeGenerator
from grounded_chart.construction_plan import HeuristicChartConstructionPlanner, PlanDecision, validate_construction_plan
from grounded_chart.evidence import build_requirement_plan
from grounded_chart.executor_agent import validate_executor_fidelity
from grounded_chart.figure_reader import (
    FigureAudit,
    FigureReaderAgent,
    figure_audit_plan_feedback,
    write_figure_audit_artifact,
)
from grounded_chart.layout_critic import LayoutCriticAgent, LayoutCritique
from grounded_chart.llm import LLMCompletionTrace, LLMUsage
from grounded_chart.patch_ops import apply_patch_operations
from grounded_chart.pipeline import GroundedChartPipeline
from grounded_chart.plan_agent import ChartPlanAgent, HeuristicPlanAgent, PlanAgentRequest, PlanAgentResult
from grounded_chart.plan_feedback import plan_updates_from_feedback
from grounded_chart.rendering import ChartImageRenderer, ChartRenderResult
from grounded_chart.schema import FigureRequirementSpec, ParsedRequirementBundle, PipelineResult, TableSchema
from grounded_chart.source_data import SourceDataExecutor, SourceDataPlanner


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
        layout_critic: LayoutCriticAgent | None = None,
        figure_reader: FigureReaderAgent | None = None,
        plan_agent: ChartPlanAgent | None = None,
        protocol_agent: ChartProtocolAgent | None = None,
        plan_reviser: Any | None = None,
        enable_layout_replanning: bool = False,
        layout_replan_rounds: int = 1,
        layout_replan_acceptance: str = "candidate_only",
    ) -> None:
        self.code_generator = code_generator
        self.verifier_pipeline = verifier_pipeline
        self.renderer = renderer or ChartImageRenderer()
        self.layout_critic = layout_critic
        self.figure_reader = figure_reader
        self.plan_agent = plan_agent or HeuristicPlanAgent(HeuristicChartConstructionPlanner())
        self.protocol_agent = protocol_agent
        self.plan_reviser = plan_reviser
        self.enable_layout_replanning = enable_layout_replanning
        self.layout_replan_rounds = layout_replan_rounds
        self.layout_replan_acceptance = layout_replan_acceptance

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
        source_workspace: str | Path | None = None,
        enable_layout_replanning: bool | None = None,
        layout_replan_rounds: int | None = None,
        layout_replan_acceptance: str | None = None,
    ) -> ChartGenerationPipelineResult:
        row_tuple = tuple(dict(row) for row in rows)
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        image_name = output_filename or f"{_safe_name(case_id)}.png"
        layout_replanning_enabled = self.enable_layout_replanning if enable_layout_replanning is None else enable_layout_replanning
        resolved_layout_rounds = self.layout_replan_rounds if layout_replan_rounds is None else layout_replan_rounds
        resolved_layout_acceptance = self.layout_replan_acceptance if layout_replan_acceptance is None else layout_replan_acceptance
        primary_render_image_name = _round_image_name(1, image_name) if layout_replanning_enabled else image_name
        effective_parse_source = parse_source or ("oracle" if parsed_requirements is not None else "predicted")
        parsed = parsed_requirements or _parse_requirements(self.verifier_pipeline.parser, query, schema)
        source_plan = SourceDataPlanner().build_plan(
            workspace=source_workspace,
            instruction=query,
        )
        source_execution = SourceDataExecutor().execute(source_plan) if source_plan.has_files else None
        copied_source_files = _copy_source_files_for_execution(source_plan, output_root)
        scaffold_plan = HeuristicChartConstructionPlanner().build_plan(
            query=query,
            requirement_plan=parsed.requirement_plan,
            source_data_plan=source_plan,
            context=generation_context,
        )
        plan_agent_result = self._build_plan_with_agent(
            PlanAgentRequest(
                query=query,
                case_id=case_id,
                output_root=output_root,
                requirement_plan=parsed.requirement_plan,
                source_data_plan=source_plan,
                source_data_execution=source_execution,
                context=dict(generation_context or {}),
                scaffold_plan=scaffold_plan,
                round_index=1,
            ),
            fallback_plan=scaffold_plan,
        )
        construction_plan = plan_agent_result.plan
        plan_validation_report = validate_construction_plan(
            construction_plan,
            query=query,
            source_data_plan=source_plan,
        )
        artifact_workspace_report = ArtifactWorkspaceBuilder(protocol_agent=self.protocol_agent).build(
            output_root=output_root,
            case_id=case_id,
            query=query,
            construction_plan=construction_plan,
            plan_validation_report=plan_validation_report,
            source_plan=source_plan,
            source_execution=source_execution,
            round_id="round_1",
        )
        effective_context = {
            **dict(generation_context or {}),
            "pipeline_architecture": {
                "stages": ["plan", "execution", "repair"],
                "current_stage": "plan_to_execution_codegen",
                "repair_stage": "only_after_generation_and_verification",
            },
            "construction_plan": construction_plan.to_dict(),
            "artifact_workspace": _artifact_workspace_context(artifact_workspace_report),
            "plan_agent": _plan_agent_context(plan_agent_result),
        }
        if source_plan.has_files or source_workspace is not None:
            effective_context["source_data_plan"] = source_plan.to_dict()
            effective_context["source_data_runtime"] = {
                "execution_dir": str(output_root),
                "copied_files": copied_source_files,
                "read_rule": "Read copied source files by relative filename from the execution directory.",
            }
        if source_execution is not None:
            effective_context["source_data_execution"] = source_execution.to_dict()
        effective_generation_mode = _resolve_generation_mode(
            generation_mode,
            schema,
            row_tuple,
            has_source_files=source_plan.has_files,
        )

        request = ChartCodeGenerationRequest(
            query=query,
            schema=schema,
            rows=row_tuple,
            output_filename=primary_render_image_name,
            plan=parsed.plan,
            requirement_plan=parsed.requirement_plan,
            case_id=case_id,
            generation_mode=effective_generation_mode,
            context=effective_context,
        )
        code_generation = self.code_generator.generate(request)
        executor_fidelity_report = validate_executor_fidelity(
            code_generation.code,
            context=request.context,
        )
        initial_code_path = output_root / "generated_initial.py"
        initial_code_path.write_text(code_generation.code, encoding="utf-8")
        plan_validation_path = output_root / "construction_plan_validation_report.json"
        plan_validation_path.write_text(
            json.dumps(plan_validation_report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        executor_fidelity_path = output_root / "executor_fidelity_report.json"
        executor_fidelity_path.write_text(
            json.dumps(executor_fidelity_report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        verify_data = verification_mode in {"full", "figure_and_data"}
        pipeline_result = self._run_verification(
            code=code_generation.code,
            query=query,
            schema=schema,
            rows=row_tuple,
            output_root=output_root,
            image_name=primary_render_image_name,
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
            image_name=primary_render_image_name,
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
            output_filename=primary_render_image_name,
            file_path=final_code_path,
            globals_dict=_render_globals(artifact_workspace_report),
        )
        final_code, pipeline_result, render_result, render_contract = self._resolve_render_failure(
            final_code=final_code,
            pipeline_result=pipeline_result,
            render_result=render_result,
            query=query,
            schema=schema,
            rows=row_tuple,
            output_root=output_root,
            image_name=primary_render_image_name,
            final_code_path=final_code_path,
            expected_figure=expected_figure,
            verify_data=verify_data,
            parsed=parsed,
            parse_source=effective_parse_source,
            artifact_workspace_report=artifact_workspace_report,
        )
        if render_contract.get("render_repair_accepted"):
            final_code_path.write_text(final_code, encoding="utf-8")
        baseline_layout_records = _snapshot_executor_layout_records(
            output_root=output_root,
            artifact_workspace_report=artifact_workspace_report,
            round_id="round_1",
        )
        final_contract = {**final_contract, **render_contract}
        final_construction_plan = construction_plan
        final_plan_validation_report = plan_validation_report
        final_plan_validation_path = plan_validation_path
        final_artifact_workspace_report = artifact_workspace_report
        final_executor_fidelity_report = executor_fidelity_report
        final_executor_fidelity_path = executor_fidelity_path
        final_plan_agent_result = plan_agent_result
        final_layout_records = baseline_layout_records

        if layout_replanning_enabled:
            (
                final_code,
                pipeline_result,
                render_result,
                final_construction_plan,
                final_plan_validation_report,
                final_plan_validation_path,
                final_artifact_workspace_report,
                final_executor_fidelity_report,
                final_executor_fidelity_path,
                final_plan_agent_result,
                layout_contract,
            ) = self._resolve_layout_replanning(
                final_code=final_code,
                pipeline_result=pipeline_result,
                render_result=render_result,
                construction_plan=final_construction_plan,
                plan_validation_report=final_plan_validation_report,
                plan_validation_path=final_plan_validation_path,
                artifact_workspace_report=final_artifact_workspace_report,
                executor_fidelity_report=final_executor_fidelity_report,
                executor_fidelity_path=final_executor_fidelity_path,
                plan_agent_result=final_plan_agent_result,
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
                generation_mode=effective_generation_mode,
                generation_context=generation_context,
                source_plan=source_plan,
                source_execution=source_execution,
                copied_source_files=copied_source_files,
                source_workspace=source_workspace,
                case_id=case_id,
                max_rounds=resolved_layout_rounds,
                acceptance_policy=resolved_layout_acceptance,
                baseline_layout_records=baseline_layout_records,
            )
            final_contract = {**final_contract, **layout_contract}
            final_layout_records = final_contract.get("executor_layout_records_final") or baseline_layout_records

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
                "source_workspace": str(Path(source_workspace).resolve()) if source_workspace is not None else None,
                "construction_plan": final_construction_plan.to_dict(),
                "plan_agent": _plan_agent_result_to_dict(final_plan_agent_result),
                "construction_plan_validation_report_path": str(final_plan_validation_path),
                "construction_plan_validation": final_plan_validation_report.to_dict(),
                "artifact_workspace": final_artifact_workspace_report.to_dict(),
                "executor_fidelity_report_path": str(final_executor_fidelity_path),
                "executor_fidelity": final_executor_fidelity_report.to_dict(),
                "executor_layout_records": final_layout_records,
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
            metadata={
                "verification_mode": verification_mode,
                "generation_mode": request.generation_mode,
                "source_workspace": str(Path(source_workspace).resolve()) if source_workspace is not None else None,
                "source_data_files": [item.name for item in source_plan.files],
                "construction_plan_type": final_construction_plan.plan_type,
                "plan_agent": _plan_agent_result_to_dict(final_plan_agent_result),
                "plan_valid_ok": final_plan_validation_report.ok,
                "construction_plan_validation_report_path": str(final_plan_validation_path),
                "artifact_workspace_ok": final_artifact_workspace_report.ok,
                "artifact_workspace_manifest_path": str(output_root / "artifact_workspace_manifest.json"),
                "artifact_execution_dir": final_artifact_workspace_report.execution_dir,
                "layout_strategy": final_construction_plan.layout_strategy,
                "executor_fidelity_ok": final_executor_fidelity_report.ok,
                "executor_fidelity_report_path": str(final_executor_fidelity_path),
                "executor_layout_records": final_layout_records,
                **final_contract,
            },
        )
        result.manifest_path.write_text(json.dumps(result.to_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
        _write_round_image_gallery(
            output_root=output_root,
            case_id=case_id,
            query=query,
            round_images=result.metadata.get("layout_replanning_round_images"),
            final_image_path=result.image_path,
        )
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

    def _resolve_layout_replanning(
        self,
        *,
        final_code: str,
        pipeline_result: PipelineResult,
        render_result: ChartRenderResult,
        construction_plan,
        plan_validation_report,
        plan_validation_path: Path,
        artifact_workspace_report,
        executor_fidelity_report,
        executor_fidelity_path: Path,
        plan_agent_result: PlanAgentResult,
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
        generation_mode: str,
        generation_context: dict[str, Any] | None,
        source_plan,
        source_execution,
        copied_source_files: list[dict[str, Any]],
        source_workspace: str | Path | None,
        case_id: str,
        max_rounds: int,
        acceptance_policy: str,
        baseline_layout_records: dict[str, str | None] | None = None,
    ):
        contract: dict[str, Any] = {
            "layout_replanning_attempted": False,
            "layout_replanning_accepted": False,
            "layout_replanning_rounds": 0,
            "layout_replanning_reason": None,
            "layout_replanning_acceptance_policy": acceptance_policy,
            "layout_replanning_round_images": [
                {
                    "round": 1,
                    "label": "baseline",
                    "image_path": str(render_result.image_path) if render_result.image_path is not None else None,
                    "code_path": str(final_code_path),
                    "promoted_to_final": True,
                }
            ],
            "layout_replanning_trace": [],
            "figure_reader_enabled": self.figure_reader is not None,
            "executor_layout_records_by_round": {
                "round_1": dict(baseline_layout_records or {}),
            },
            "executor_layout_records_final": dict(baseline_layout_records or {}),
        }
        if max_rounds <= 0:
            contract["layout_replanning_reason"] = "disabled_zero_rounds"
            return (
                final_code,
                pipeline_result,
                render_result,
                construction_plan,
                plan_validation_report,
                plan_validation_path,
                artifact_workspace_report,
                executor_fidelity_report,
                executor_fidelity_path,
                plan_agent_result,
                contract,
            )
        if self.layout_critic is None:
            contract["layout_replanning_reason"] = "no_layout_critic"
            return (
                final_code,
                pipeline_result,
                render_result,
                construction_plan,
                plan_validation_report,
                plan_validation_path,
                artifact_workspace_report,
                executor_fidelity_report,
                executor_fidelity_path,
                plan_agent_result,
                contract,
            )
        if not render_result.ok:
            contract["layout_replanning_reason"] = "render_not_ok"
            return (
                final_code,
                pipeline_result,
                render_result,
                construction_plan,
                plan_validation_report,
                plan_validation_path,
                artifact_workspace_report,
                executor_fidelity_report,
                executor_fidelity_path,
                plan_agent_result,
                contract,
            )

        current_code = final_code
        current_result = pipeline_result
        current_render = render_result
        current_plan = construction_plan
        current_plan_validation = plan_validation_report
        current_plan_validation_path = plan_validation_path
        current_artifact_report = artifact_workspace_report
        current_context = _generation_context_for_plan(
            generation_context=generation_context,
            output_root=output_root,
            construction_plan=current_plan,
            artifact_workspace_report=current_artifact_report,
            source_plan=source_plan,
            source_execution=source_execution,
            copied_source_files=copied_source_files,
            source_workspace=source_workspace,
            stage="layout_replanning_current",
        )
        current_executor_report = executor_fidelity_report
        current_executor_path = executor_fidelity_path
        current_plan_agent_result = plan_agent_result

        for round_index in range(1, max_rounds + 1):
            contract["layout_replanning_attempted"] = True
            critique = self.layout_critic.critique(
                query=query,
                construction_plan=current_plan.to_dict(),
                generated_code=current_code,
                render_result=current_render,
                actual_figure=current_result.actual_figure,
                generation_context=generation_context,
            )
            audit = self._audit_current_figure(
                query=query,
                construction_plan=current_plan.to_dict(),
                generated_code=current_code,
                render_result=current_render,
                actual_figure=current_result.actual_figure,
                artifact_workspace_report=current_artifact_report,
                generation_context=generation_context,
                source_data_plan=source_plan,
            )
            combined_critique = _combine_layout_critique_with_figure_audit(critique, audit)
            feedback_bundle = _layout_replan_feedback_bundle(
                round_index=round_index,
                critique=critique,
                combined_critique=combined_critique,
                audit=audit,
            )
            artifact_paths = _write_replan_feedback_artifacts(
                output_root=output_root,
                round_index=round_index,
                critique=critique,
                combined_critique=combined_critique,
                feedback_bundle=feedback_bundle,
            )
            if audit is not None:
                artifact_paths["figure_audit_path"] = write_figure_audit_artifact(
                    output_root=output_root,
                    round_index=round_index,
                    audit=audit,
                )
            round_record: dict[str, Any] = {
                "round_index": round_index,
                "critique": critique.to_dict(),
                "combined_critique": combined_critique.to_dict(),
                "figure_audit": audit.to_dict() if audit is not None else None,
                "replan": {
                    "mode": "feedback_to_plan_agent",
                    "feedback_count": len(feedback_bundle["feedback_items"]),
                    "failed_contracts": list(feedback_bundle["failed_contracts"]),
                    "rationale": feedback_bundle["summary"],
                    "feedback_issue_ids": [
                        str(item.get("issue_id") or "")
                        for item in feedback_bundle["feedback_items"]
                        if isinstance(item, dict)
                    ],
                },
                "artifacts": artifact_paths,
            }
            if combined_critique.ok:
                round_record["decision"] = "stop_layout_ok"
                contract["layout_replanning_trace"].append(round_record)
                contract["layout_replanning_reason"] = "layout_critic_ok"
                break
            if not feedback_bundle["feedback_items"] and not feedback_bundle["summary"]:
                round_record["decision"] = "stop_no_revision"
                contract["layout_replanning_trace"].append(round_record)
                contract["layout_replanning_reason"] = "no_replanning_feedback"
                break

            revised_round_id = f"round_{round_index + 1}"
            replan_generation_context = _replan_generation_context(
                generation_context=generation_context,
                feedback_bundle=feedback_bundle,
                previous_construction_plan=current_plan.to_dict(),
            )
            scaffold_revised_plan = HeuristicChartConstructionPlanner().build_plan(
                query=query,
                requirement_plan=parsed.requirement_plan,
                source_data_plan=source_plan,
                context=replan_generation_context,
            )
            revised_plan_agent_result = self._build_plan_with_agent(
                PlanAgentRequest(
                    query=query,
                    case_id=case_id,
                    output_root=output_root,
                    requirement_plan=parsed.requirement_plan,
                    source_data_plan=source_plan,
                    source_data_execution=source_execution,
                    context=replan_generation_context,
                    scaffold_plan=scaffold_revised_plan,
                    previous_plan=current_plan,
                    feedback_bundle=feedback_bundle,
                    round_index=round_index + 1,
                ),
                fallback_plan=scaffold_revised_plan,
            )
            revised_plan = revised_plan_agent_result.plan
            revised_plan = _attach_replan_feedback_to_plan(
                revised_plan,
                feedback_bundle=feedback_bundle,
                round_index=round_index,
            )
            revised_plan_validation = validate_construction_plan(
                revised_plan,
                query=query,
                source_data_plan=source_plan,
            )
            revised_artifact_report = ArtifactWorkspaceBuilder(protocol_agent=self.protocol_agent).build(
                output_root=output_root,
                case_id=case_id,
                query=query,
                construction_plan=revised_plan,
                plan_validation_report=revised_plan_validation,
                source_plan=source_plan,
                source_execution=source_execution,
                round_id=revised_round_id,
            )
            revised_context = _generation_context_for_plan(
                generation_context=replan_generation_context,
                output_root=output_root,
                construction_plan=revised_plan,
                artifact_workspace_report=revised_artifact_report,
                source_plan=source_plan,
                source_execution=source_execution,
                copied_source_files=copied_source_files,
                source_workspace=source_workspace,
                stage="layout_replanning_codegen",
            )
            revised_context.pop("plan_replanning", None)
            revised_context["plan_agent"] = _plan_agent_context(revised_plan_agent_result)
            revised_context["layout_replanning"] = {
                "round_index": round_index,
                "acceptance_policy": acceptance_policy,
                "replan_mode": "feedback_to_plan_agent",
                "allowed_edit_scope": "bounded_visual_presentation_and_encoding",
                "preserve_non_visual_behavior": True,
                "rules": [
                    "Use the previous generated code as the baseline.",
                    "Treat feedback_bundle.feedback_items as mandatory PlanAgent replan evidence. Do not merely restate the feedback as comments or notes.",
                    "Make concrete plotting-code changes when needed to address the feedback, while preserving source-grounded data and explicit requirements.",
                    "Allowed changes include figure size, axes/panel arrangement, inset positions, legend/title placement, margins, spacing, visual channel styling, color/alpha/hatch/line style mappings, readability annotations, and chart-specific rendering details that clarify the planned visual semantics.",
                    "Do not change data loading, prepared artifact paths, deterministic data transformations, plotted values, source files, required labels, or required legend categories.",
                    "If visual audit feedback conflicts with deterministic artifacts or explicit requirements, keep the deterministic artifact/explicit requirement and explain the limitation.",
                    "If a requested visual update cannot be implemented locally, keep the previous non-visual code unchanged and explain the limitation.",
                    "The candidate is considered ineffective if it only changes comments, artifact round paths, or metadata without changing the rendered visual outcome.",
                ],
                "feedback_bundle": feedback_bundle,
                "figure_audit_feedback": figure_audit_plan_feedback(audit),
                "previous_code": _code_preservation_excerpt(current_code),
            }
            revised_request = ChartCodeGenerationRequest(
                query=query,
                schema=schema,
                rows=rows,
                output_filename=_round_image_name(round_index + 1, image_name),
                plan=parsed.plan,
                requirement_plan=parsed.requirement_plan,
                case_id=case_id,
                generation_mode=generation_mode,
                context=revised_context,
            )
            revised_codegen = self.code_generator.generate(revised_request)
            revised_code_path = output_root / f"generated_layout_replan_round_{round_index}.py"
            revised_code_path.write_text(revised_codegen.code, encoding="utf-8")
            revised_executor_report = validate_executor_fidelity(
                revised_codegen.code,
                context=revised_request.context,
            )
            revised_executor_path = output_root / f"executor_fidelity_report_layout_round_{round_index}.json"
            revised_executor_path.write_text(
                json.dumps(revised_executor_report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            revised_result = self._run_verification(
                code=revised_codegen.code,
                query=query,
                schema=schema,
                rows=rows,
                output_root=output_root,
                image_name=_round_image_name(round_index + 1, image_name),
                file_path=revised_code_path,
                expected_figure=expected_figure,
                verify_data=verify_data,
                parsed=parsed,
                parse_source=parse_source,
                allow_repair=False,
            )
            revised_render = self.renderer.render(
                revised_codegen.code,
                rows=rows,
                output_dir=output_root,
                output_filename=_round_image_name(round_index + 1, image_name),
                file_path=revised_code_path,
                globals_dict=_render_globals(revised_artifact_report),
            )
            revised_layout_records = _snapshot_executor_layout_records(
                output_root=output_root,
                artifact_workspace_report=revised_artifact_report,
                round_id=revised_round_id,
            )
            current_score = _verification_score(current_result)
            revised_score = _verification_score(revised_result)
            internal_gate_accepted = revised_render.ok and revised_score <= current_score
            accepted = _accept_layout_replan_candidate(
                acceptance_policy,
                internal_gate_accepted=internal_gate_accepted,
            )
            round_record.update(
                {
                    "decision": "accepted" if accepted else "candidate_only" if acceptance_policy == "candidate_only" else "rejected",
                    "revised_code_path": str(revised_code_path),
                    "candidate_code_changed": revised_codegen.code.strip() != current_code.strip(),
                    "revised_executor_fidelity_report_path": str(revised_executor_path),
                    "revised_executor_fidelity_ok": revised_executor_report.ok,
                    "revised_plan_round_id": revised_round_id,
                    "revised_plan": revised_plan.to_dict(),
                    "revised_plan_agent": _plan_agent_result_to_dict(revised_plan_agent_result),
                    "revised_artifact_workspace": revised_artifact_report.to_dict(),
                    "revised_render": _render_result_to_dict(revised_render),
                    "revised_executor_layout_records": revised_layout_records,
                    "current_verification_score": current_score,
                    "revised_verification_score": revised_score,
                    "verification_not_regressed": revised_score <= current_score,
                    "internal_gate_accepted": internal_gate_accepted,
                    "acceptance_policy": acceptance_policy,
                    "codegen": {
                        "generator_name": revised_codegen.generator_name,
                        "backend_hint": revised_codegen.backend_hint,
                        "instruction": revised_codegen.instruction,
                        "metadata": _jsonable(revised_codegen.metadata),
                        "llm_trace": _llm_trace_to_dict(revised_codegen.llm_trace),
                    },
                }
            )
            contract["layout_replanning_trace"].append(round_record)
            contract["layout_replanning_rounds"] = round_index
            contract["executor_layout_records_by_round"][revised_round_id] = revised_layout_records
            contract["layout_replanning_round_images"].append(
                {
                    "round": round_index + 1,
                    "label": f"layout_replan_candidate_{round_index}",
                    "image_path": str(revised_render.image_path) if revised_render.image_path is not None else None,
                    "code_path": str(revised_code_path),
                    "promoted_to_final": accepted,
                }
            )
            if not accepted:
                if acceptance_policy == "candidate_only":
                    contract["layout_replanning_reason"] = "candidate_only_not_promoted"
                else:
                    contract["layout_replanning_reason"] = (
                        "candidate_render_failed" if not revised_render.ok else "candidate_verification_regressed"
                    )
                break

            current_code = revised_codegen.code
            current_result = revised_result
            current_render = revised_render
            current_plan = revised_plan
            current_plan_validation = revised_plan_validation
            current_plan_validation_path = output_root / f"construction_plan_validation_report_layout_round_{round_index}.json"
            current_plan_validation_path.write_text(
                json.dumps(current_plan_validation.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            current_artifact_report = revised_artifact_report
            current_executor_report = revised_executor_report
            current_executor_path = revised_executor_path
            current_plan_agent_result = revised_plan_agent_result
            contract["executor_layout_records_final"] = revised_layout_records
            final_code_path.write_text(current_code, encoding="utf-8")
            contract.update(
                {
                    "layout_replanning_accepted": True,
                    "layout_replanning_reason": "accepted",
                    "final_code_source": f"layout_replan_round_{round_index}",
                    "final_code_verified": True,
                    "final_verification_score": revised_score,
                }
            )

        return (
            current_code,
            current_result,
            current_render,
            current_plan,
            current_plan_validation,
            current_plan_validation_path,
            current_artifact_report,
            current_executor_report,
            current_executor_path,
            current_plan_agent_result,
            contract,
        )

    def _audit_current_figure(
        self,
        *,
        query: str,
        construction_plan: dict[str, Any],
        generated_code: str,
        render_result: ChartRenderResult,
        actual_figure: Any | None,
        artifact_workspace_report,
        generation_context: dict[str, Any] | None,
        source_data_plan,
    ) -> FigureAudit | None:
        if self.figure_reader is None:
            return None
        return self.figure_reader.audit(
            query=query,
            construction_plan=construction_plan,
            generated_code=generated_code,
            render_result=render_result,
            actual_figure=actual_figure,
            artifact_workspace_report=artifact_workspace_report,
            generation_context=generation_context,
            source_data_plan=source_data_plan,
        )

    def _build_plan_with_agent(
        self,
        request: PlanAgentRequest,
        *,
        fallback_plan,
    ) -> PlanAgentResult:
        try:
            return self.plan_agent.build_plan(request)
        except Exception as exc:
            return PlanAgentResult(
                plan=fallback_plan,
                agent_name="heuristic_fallback_after_plan_agent_error",
                rationale=f"{type(exc).__name__}: {str(exc)[:240]}",
                metadata={
                    "fallback": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "round_index": request.round_index,
                },
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
        artifact_workspace_report,
    ) -> tuple[str, PipelineResult, ChartRenderResult, dict[str, Any]]:
        if render_result.ok:
            return final_code, pipeline_result, render_result, {
                "render_repair_attempted": False,
                "render_repair_accepted": False,
            }

        exception_type = render_result.exception_type or "RenderError"
        exception_message = render_result.exception_message or "No visible chart artifact was produced."
        exception_traceback = _render_exception_traceback(render_result)
        contract: dict[str, Any] = {
            "render_repair_attempted": False,
            "render_repair_accepted": False,
            "render_repair_reason": "render_failed",
            "render_repair_exception_type": exception_type,
            "render_repair_exception_message": exception_message,
            "render_repair_exception_traceback": exception_traceback,
        }
        if getattr(self.verifier_pipeline, "repairer", None) is None:
            contract["render_repair_reason"] = "no_repairer"
            return final_code, pipeline_result, render_result, contract

        contract["render_repair_attempted"] = True
        repair_exception_detail = f"{exception_type}: {exception_message}"
        if exception_traceback:
            repair_exception_detail = f"{repair_exception_detail}\nTraceback:\n{exception_traceback}"
        repair_exception = ChartRenderStageError(
            f"Render-stage artifact export failed after verification: {repair_exception_detail}"
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
                globals_dict=_render_globals(artifact_workspace_report),
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



def _combine_layout_critique_with_figure_audit(
    critique: LayoutCritique,
    audit: FigureAudit | None,
) -> LayoutCritique:
    if audit is None or audit.ok:
        return critique
    audit_contracts = _audit_failed_contracts(audit)
    plan_feedback = (*critique.normalized_plan_feedback(), *audit.normalized_plan_feedback())
    audit_updates = plan_updates_from_feedback(audit.normalized_plan_feedback())
    if not audit_contracts and not audit_updates:
        return critique
    diagnosis_parts = [part for part in (critique.diagnosis, _audit_diagnosis(audit)) if part]
    metadata = {
        **dict(critique.metadata),
        "figure_audit_used_for_replanning": True,
        "figure_audit_summary": audit.summary,
        "figure_audit_confidence": audit.confidence,
    }
    return LayoutCritique(
        ok=False,
        failed_contracts=tuple(dict.fromkeys((*critique.failed_contracts, *audit_contracts))),
        diagnosis=" | ".join(diagnosis_parts),
        recommended_plan_updates=tuple((*critique.recommended_plan_updates, *audit_updates)),
        plan_feedback=plan_feedback,
        confidence=max(critique.confidence, audit.confidence),
        llm_trace=critique.llm_trace,
        metadata=metadata,
    )


def _audit_failed_contracts(audit: FigureAudit) -> tuple[str, ...]:
    contracts: list[str] = []
    if audit.readability_issues:
        contracts.append("visual.readability")
    if audit.encoding_confusions:
        contracts.append("visual.encoding_confusion")
    if audit.data_semantic_warnings:
        contracts.append("visual.data_semantic_warning")
    if audit.suspicious_artifacts:
        contracts.append("visual.suspicious_artifact")
    if audit.unclear_regions:
        contracts.append("visual.unclear_region")
    return tuple(contracts)


def _audit_diagnosis(audit: FigureAudit) -> str:
    issue_types: list[str] = []
    for group in (
        audit.readability_issues,
        audit.encoding_confusions,
        audit.data_semantic_warnings,
        audit.suspicious_artifacts,
        audit.unclear_regions,
    ):
        for item in group:
            issue_type = str(item.get("issue_type") or "").strip()
            if issue_type:
                issue_types.append(issue_type)
    prefix = audit.summary.strip() if audit.summary else "FigureReaderAgent reported visual-semantic issues."
    if not issue_types:
        return prefix
    return f"{prefix} Issue types: {', '.join(dict.fromkeys(issue_types))}."


def _layout_replan_feedback_bundle(
    *,
    round_index: int,
    critique: LayoutCritique,
    combined_critique: LayoutCritique,
    audit: FigureAudit | None,
) -> dict[str, Any]:
    feedback_items = [_jsonable(item) for item in combined_critique.normalized_plan_feedback()]
    return {
        "round_index": round_index,
        "mode": "feedback_to_plan_agent",
        "summary": combined_critique.diagnosis,
        "failed_contracts": list(combined_critique.failed_contracts),
        "feedback_items": feedback_items,
        "layout_agent": {
            "ok": critique.ok,
            "diagnosis": critique.diagnosis,
            "failed_contracts": list(critique.failed_contracts),
            "feedback_items": [_jsonable(item) for item in critique.normalized_plan_feedback()],
            "confidence": critique.confidence,
        },
        "figure_reader": None
        if audit is None
        else {
            "ok": audit.ok,
            "summary": audit.summary,
            "feedback_items": [_jsonable(item) for item in audit.normalized_plan_feedback()],
            "confidence": audit.confidence,
        },
        "plan_agent_instruction": (
            "Replan the whole construction plan from the original request and current visual feedback. "
            "Do not apply shallow field patches. Convert each feedback item into concrete layout, composition, "
            "visual-channel, or rendering commitments that ExecutorAgent can implement."
        ),
    }


def _write_replan_feedback_artifacts(
    *,
    output_root: str | Path,
    round_index: int,
    critique: LayoutCritique,
    combined_critique: LayoutCritique,
    feedback_bundle: dict[str, Any],
) -> dict[str, str]:
    agent_dir = Path(output_root).resolve() / LAYOUT_AGENT_DIR / f"round_{round_index}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    critique_path = agent_dir / "layout_critique.json"
    combined_path = agent_dir / "combined_replanning_critique.json"
    feedback_path = agent_dir / "feedback_bundle.json"
    trace_path = agent_dir / "replan_trace.json"
    critique_path.write_text(json.dumps(critique.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    combined_path.write_text(json.dumps(combined_critique.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    feedback_path.write_text(json.dumps(feedback_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    trace_path.write_text(
        json.dumps(
            {
                "mode": "feedback_to_plan_agent",
                "status": "feedback_collected",
                "feedback_count": len(feedback_bundle.get("feedback_items") or []),
                "failed_contracts": list(feedback_bundle.get("failed_contracts") or []),
                "note": "PlanRevision patching is bypassed; feedback is sent to PlanAgent through generation context.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "agent_dir": str(agent_dir),
        "layout_critique_path": str(critique_path),
        "combined_critique_path": str(combined_path),
        "feedback_bundle_path": str(feedback_path),
        "replan_trace_path": str(trace_path),
    }


def _replan_generation_context(
    *,
    generation_context: dict[str, Any] | None,
    feedback_bundle: dict[str, Any],
    previous_construction_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        **dict(generation_context or {}),
        "plan_replanning": {
            "mode": "feedback_to_plan_agent",
            "feedback_bundle": feedback_bundle,
            "previous_construction_plan": previous_construction_plan,
            "required_output": {
                "revised_plan": "A complete construction plan, not a shallow patch.",
                "feedback_resolution": "For every feedback item, state how the revised plan addresses, defers, or rejects it.",
            },
            "rules": [
                "Use the original natural-language request as the source of truth.",
                "Use visual feedback as replan evidence, not as direct code edits.",
                "Do not contradict deterministic artifacts, source files, or explicit chart requirements.",
                "Avoid vague notes. Produce executable commitments for layout, visual channels, legends, insets, and rendering semantics when applicable.",
            ],
        },
    }


def _attach_replan_feedback_to_plan(
    plan,
    *,
    feedback_bundle: dict[str, Any],
    round_index: int,
):
    feedback_items = list(feedback_bundle.get("feedback_items") or [])
    feedback_summaries = []
    for item in feedback_items:
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get("issue_id") or "feedback")
        issue_type = str(item.get("issue_type") or "issue")
        evidence = str(item.get("evidence") or "").strip()
        action = item.get("suggested_plan_action") if isinstance(item.get("suggested_plan_action"), dict) else {}
        proposal = str(action.get("proposal") or "").strip()
        summary = f"{issue_id} [{issue_type}]"
        if evidence:
            summary += f": {evidence}"
        if proposal and proposal != evidence:
            summary += f" Suggested plan action: {proposal}"
        feedback_summaries.append(summary)
    if not feedback_summaries and not feedback_bundle.get("summary"):
        return plan
    decision = PlanDecision(
        decision_id=f"plan_agent.replan.{round_index}",
        category="plan_replanning",
        value={
            "mode": "feedback_to_plan_agent",
            "feedback_count": len(feedback_summaries),
            "failed_contracts": list(feedback_bundle.get("failed_contracts") or []),
            "feedback_items": feedback_summaries,
        },
        status="inferred",
        rationale=str(feedback_bundle.get("summary") or "Visual feedback requires PlanAgent replanning."),
    )
    constraints = tuple(
        dict.fromkeys(
            (
                *plan.constraints,
                "Round replan feedback is mandatory execution evidence for ExecutorAgent; do not satisfy it by comments only.",
                "For every feedback item, change layout, visual channels, legend/inset placement, or rendering semantics when needed while preserving source-grounded data.",
                *[f"Replan feedback: {item}" for item in feedback_summaries],
            )
        )
    )
    return replace(plan, decisions=(*plan.decisions, decision), constraints=constraints)


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


def _accept_layout_replan_candidate(
    acceptance_policy: str,
    *,
    internal_gate_accepted: bool,
) -> bool:
    normalized = str(acceptance_policy or "candidate_only").strip().lower()
    if normalized in {"candidate_only", "none", "off"}:
        return False
    if normalized in {"internal_verifier", "internal", "auto"}:
        return internal_gate_accepted
    raise ValueError(
        f"Unsupported layout replanning acceptance policy {acceptance_policy!r}; "
        "use candidate_only or internal_verifier."
    )


def _round_image_name(round_number: int, base_image_name: str) -> str:
    suffix = Path(str(base_image_name or "figure.png")).suffix or ".png"
    return f"round{int(round_number)}{suffix}"


def _write_round_image_gallery(
    *,
    output_root: Path,
    case_id: str,
    query: str,
    round_images: Any,
    final_image_path: Path | None,
) -> None:
    if not isinstance(round_images, list) or not round_images:
        return
    cards = []
    for item in round_images:
        if not isinstance(item, dict):
            continue
        raw_image_path = str(item.get("image_path") or "").strip()
        if not raw_image_path:
            continue
        image_path = Path(raw_image_path)
        try:
            rel_image = image_path.resolve().relative_to(output_root.resolve()).as_posix()
        except ValueError:
            rel_image = image_path.as_posix()
        promoted = bool(item.get("promoted_to_final"))
        final_marker = final_image_path is not None and image_path.resolve() == final_image_path.resolve()
        badge = "final" if final_marker else "candidate"
        if promoted and not final_marker:
            badge = "promoted"
        cards.append(
            f"""
      <article class="round-card">
        <div class="round-meta">
          <strong>{html.escape('round' + str(item.get('round', '?')))}</strong>
          <span>{html.escape(str(item.get('label') or ''))}</span>
          <em>{html.escape(badge)}</em>
        </div>
        <a href="{html.escape(rel_image, quote=True)}"><img src="{html.escape(rel_image, quote=True)}" alt="{html.escape(str(item.get('label') or 'round image'), quote=True)}"></a>
        <code>{html.escape(rel_image)}</code>
      </article>
""".rstrip()
        )
    if not cards:
        return
    gallery_path = output_root / "round_images.html"
    gallery_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Round Images - {html.escape(case_id)}</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --ink: #1e1c18;
      --muted: #6f695f;
      --line: #d8cfbe;
      --card: #fffaf0;
      --accent: #1e5b4f;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #efe2c3, transparent 32rem), var(--bg);
      color: var(--ink);
      font-family: Georgia, 'Times New Roman', serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1180px, calc(100vw - 48px));
      margin: 32px auto 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(30px, 5vw, 56px);
      letter-spacing: -0.04em;
    }}
    .query {{
      max-width: 920px;
      color: var(--muted);
      font-family: Consolas, 'Courier New', monospace;
      white-space: pre-wrap;
      border-left: 4px solid var(--accent);
      padding-left: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
      margin-top: 28px;
    }}
    .round-card {{
      background: color-mix(in srgb, var(--card) 94%, white);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 18px 46px rgba(57, 47, 29, 0.10);
    }}
    .round-meta {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 12px;
      font-family: Consolas, 'Courier New', monospace;
    }}
    .round-meta span {{
      color: var(--muted);
      flex: 1;
    }}
    .round-meta em {{
      font-style: normal;
      color: white;
      background: var(--accent);
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 620px;
      object-fit: contain;
      background: white;
      border-radius: 12px;
      border: 1px solid var(--line);
    }}
    code {{
      display: block;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(case_id)}</h1>
    <div class="query">{html.escape(query)}</div>
    <section class="grid">
{chr(10).join(cards)}
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def _snapshot_executor_layout_records(
    *,
    output_root: Path,
    artifact_workspace_report,
    round_id: str,
) -> dict[str, str | None]:
    """Copy executor-emitted layout evidence into the round execution folder."""

    execution_dir = Path(str(getattr(artifact_workspace_report, "execution_dir", "") or "")).resolve()
    execution_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, str | None] = {
        "round_id": round_id,
        "computed_layout_json": None,
        "layout_decisions_md": None,
    }
    for filename, key in (
        ("computed_layout.json", "computed_layout_json"),
        ("layout_decisions.md", "layout_decisions_md"),
    ):
        source = (Path(output_root).resolve() / filename).resolve()
        if not source.exists() or not source.is_file():
            continue
        target = execution_dir / filename
        try:
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            records[key] = str(target)
        except OSError:
            records[key] = None
    return records


def _render_globals(artifact_workspace_report) -> dict[str, Any]:
    if artifact_workspace_report is None:
        return {}
    to_dict = getattr(artifact_workspace_report, "to_dict", None)
    if callable(to_dict):
        return {"artifact_workspace": to_dict()}
    if isinstance(artifact_workspace_report, dict):
        return {"artifact_workspace": dict(artifact_workspace_report)}
    return {}


def _code_preservation_excerpt(code: str, *, max_chars: int = 18000) -> str:
    text = str(code or "")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n# ... [middle truncated for layout-only replanning prompt] ...\n\n{tail}"


def _infer_generation_mode(schema: TableSchema, rows: tuple[dict[str, Any], ...], *, has_source_files: bool = False) -> str:
    if has_source_files:
        return "source_file_grounded"
    if rows or schema.columns:
        return "table"
    return "instruction_only"


def _resolve_generation_mode(
    requested: str | None,
    schema: TableSchema,
    rows: tuple[dict[str, Any], ...],
    *,
    has_source_files: bool,
) -> str:
    normalized = str(requested or "").strip()
    if has_source_files and normalized in {"", "instruction_only"}:
        return "source_file_grounded"
    return normalized or _infer_generation_mode(schema, rows, has_source_files=has_source_files)


def _copy_source_files_for_execution(source_plan, output_root: Path) -> list[dict[str, Any]]:
    copied = []
    for item in getattr(source_plan, "files", ()) or ():
        source = Path(item.path)
        target = output_root / item.name
        record = {
            "name": item.name,
            "source_path": str(source),
            "runtime_path": str(target),
            "copied": False,
            "error": None,
        }
        try:
            if source.exists() and source.is_file():
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                record["copied"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        copied.append(record)
    return copied


def _generation_context_for_plan(
    *,
    generation_context: dict[str, Any] | None,
    output_root: Path,
    construction_plan,
    artifact_workspace_report,
    source_plan,
    source_execution,
    copied_source_files: list[dict[str, Any]],
    source_workspace: str | Path | None,
    stage: str,
) -> dict[str, Any]:
    context = {
        **dict(generation_context or {}),
        "pipeline_architecture": {
            "stages": ["plan", "execution", "repair"],
            "current_stage": stage,
            "repair_stage": "only_after_generation_and_verification",
        },
        "construction_plan": construction_plan.to_dict(),
        "artifact_workspace": _artifact_workspace_context(artifact_workspace_report),
    }
    if getattr(source_plan, "has_files", False) or source_workspace is not None:
        context["source_data_plan"] = source_plan.to_dict()
        context["source_data_runtime"] = {
            "execution_dir": str(output_root),
            "copied_files": copied_source_files,
            "read_rule": "Read copied source files by relative filename from the execution directory.",
        }
    if source_execution is not None:
        context["source_data_execution"] = source_execution.to_dict()
    return context


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


def _render_exception_traceback(render: ChartRenderResult) -> str | None:
    metadata = render.metadata if isinstance(render.metadata, dict) else {}
    value = metadata.get("traceback")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _artifact_workspace_context(report: Any) -> dict[str, Any]:
    """Compact artifact workspace payload for ExecutorAgent prompts.

    Full manifests remain on disk and in reports. Codegen only needs paths,
    schemas, contract tiers, artifact roles, and compact protocol commitments.
    Keeping LLM traces and full protocol payloads here creates large prompt
    spikes during layout replanning without improving executable fidelity.
    """

    if report is None:
        return {}
    payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    artifacts = [
        _artifact_context(item)
        for item in list(payload.get("artifacts") or [])
        if isinstance(item, dict)
    ]
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "ok": payload.get("ok"),
        "root": payload.get("root"),
        "plan_dir": payload.get("plan_dir"),
        "execution_dir": payload.get("execution_dir"),
        "repair_dir": payload.get("repair_dir"),
        "artifacts": artifacts,
        "issues": [
            _compact_issue(item)
            for item in list(payload.get("issues") or [])[:12]
            if isinstance(item, dict)
        ],
        "metadata": {
            "round_id": metadata.get("round_id"),
            "case_id": metadata.get("case_id"),
            "source_files": list(metadata.get("source_files") or []),
            "chart_protocols": [
                _protocol_context(item)
                for item in list(metadata.get("chart_protocols") or [])
                if isinstance(item, dict)
            ],
        },
    }


def _artifact_context(item: dict[str, Any]) -> dict[str, Any]:
    schema = item.get("schema") if isinstance(item.get("schema"), dict) else {}
    columns = list(schema.get("columns") or item.get("columns") or [])
    return {
        key: _jsonable(value)
        for key, value in {
            "name": item.get("name"),
            "relative_path": item.get("relative_path"),
            "description": item.get("description"),
            "artifact_role": item.get("artifact_role"),
            "chart_type": item.get("chart_type"),
            "layer_id": item.get("layer_id"),
            "contract_tier": item.get("contract_tier"),
            "required_for_plotting": item.get("required_for_plotting"),
            "legacy_alias": item.get("legacy_alias"),
            "alias_for": item.get("alias_for"),
            "schema": {"columns": columns} if columns else None,
        }.items()
        if value not in (None, "", [], {})
    }


def _protocol_context(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in {
            "chart_type": item.get("chart_type"),
            "protocol_id": item.get("protocol_id"),
            "source": item.get("source"),
            "contract_tiers": item.get("contract_tiers"),
            "required_artifact_columns": item.get("required_artifact_columns"),
            "visual_channel_policy": item.get("visual_channel_policy"),
            "visual_channel_contracts": item.get("visual_channel_contracts"),
            "hard_fidelity": _list_prefix(item.get("hard_fidelity"), max_items=8),
            "soft_guidance": _list_prefix(item.get("soft_guidance"), max_items=6),
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_issue(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(item.get(key))
        for key in ("code", "message", "severity", "artifact", "plan_ref")
        if item.get(key) not in (None, "", [], {})
    }


def _list_prefix(value: Any, *, max_items: int) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_jsonable(item) for item in list(value)[:max_items]]


def _plan_agent_context(result: PlanAgentResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    trace = result.llm_trace
    payload = {
        "agent_name": result.agent_name,
        "feedback_resolution": [dict(item) for item in result.feedback_resolution],
        "rationale": result.rationale,
        "llm_trace": {
            "provider": trace.provider,
            "model": trace.model,
            "usage": _usage_to_dict(trace.usage),
            "max_tokens": trace.max_tokens,
        }
        if trace is not None
        else None,
        "metadata": _jsonable(result.metadata),
    }
    if result.plan_brief:
        payload["plan_brief"] = _jsonable(result.plan_brief)
    return payload


def _plan_agent_result_to_dict(result: PlanAgentResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = {
        "agent_name": result.agent_name,
        "feedback_resolution": [dict(item) for item in result.feedback_resolution],
        "rationale": result.rationale,
        "llm_trace": _llm_trace_to_dict(result.llm_trace),
        "metadata": _jsonable(result.metadata),
    }
    if result.plan_brief:
        payload["plan_brief"] = _jsonable(result.plan_brief)
    return payload


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










