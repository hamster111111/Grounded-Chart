from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from grounded_chart.schema import TableSchema


@dataclass(frozen=True)
class MatplotBenchRecord:
    """Native MatPlotBench instruction record.

    MatPlotBench is an instruction-to-figure benchmark with reference images.
    It is not directly equivalent to GroundedChart's table-to-plotted-data
    `ChartCase`, so this adapter keeps it as a native manifest first.
    """

    case_id: str
    simple_instruction: str
    expert_instruction: str
    ground_truth_path: Path | None = None
    data_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def preferred_instruction(self, *, prefer_expert: bool = True) -> str:
        if prefer_expert and self.expert_instruction:
            return self.expert_instruction
        return self.simple_instruction or self.expert_instruction

    @property
    def has_external_data(self) -> bool:
        return self.data_dir is not None and self.data_dir.exists()

    @property
    def has_ground_truth_image(self) -> bool:
        return self.ground_truth_path is not None and self.ground_truth_path.exists()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "simple_instruction": self.simple_instruction,
            "expert_instruction": self.expert_instruction,
            "ground_truth_path": str(self.ground_truth_path) if self.ground_truth_path else None,
            "data_dir": str(self.data_dir) if self.data_dir else None,
            "has_external_data": self.has_external_data,
            "has_ground_truth_image": self.has_ground_truth_image,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MatplotBenchWorkspaceRecord:
    """MatPlotBench instruction record joined with one workspace run."""

    native_record: MatplotBenchRecord
    workspace_dir: Path
    code_paths: tuple[Path, ...] = ()
    selected_code_path: Path | None = None
    selected_code_stage: str | None = None
    selected_code_iteration: int | None = None
    log_paths: tuple[Path, ...] = ()
    output_image_paths: tuple[Path, ...] = ()
    score: float | int | None = None
    eval_raw: str | None = None
    eval_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def case_id(self) -> str:
        return self.native_record.case_id

    def preferred_instruction(self, *, prefer_expert: bool = True) -> str:
        return self.native_record.preferred_instruction(prefer_expert=prefer_expert)

    @property
    def has_workspace(self) -> bool:
        return self.workspace_dir.exists()

    @property
    def has_selected_code(self) -> bool:
        return self.selected_code_path is not None and self.selected_code_path.exists()

    @property
    def has_output_image(self) -> bool:
        return any(path.exists() for path in self.output_image_paths)

    def read_selected_code(self) -> str:
        if self.selected_code_path is None:
            raise FileNotFoundError(f"No selected code for MatPlotBench case {self.case_id}")
        return self.selected_code_path.read_text(encoding="utf-8")

    def selected_log_path(self) -> Path | None:
        if self.selected_code_path is None:
            return None
        candidate = self.selected_code_path.with_suffix(self.selected_code_path.suffix + ".log")
        return candidate if candidate.exists() else None

    def to_dict(self, include_instruction: bool = False, include_eval_raw: bool = False) -> dict[str, Any]:
        result = {
            "case_id": self.case_id,
            "score": self.score,
            "eval_error": self.eval_error,
            "workspace_dir": str(self.workspace_dir),
            "has_workspace": self.has_workspace,
            "has_selected_code": self.has_selected_code,
            "has_output_image": self.has_output_image,
            "selected_code_path": str(self.selected_code_path) if self.selected_code_path else None,
            "selected_code_stage": self.selected_code_stage,
            "selected_code_iteration": self.selected_code_iteration,
            "selected_log_path": str(self.selected_log_path()) if self.selected_log_path() else None,
            "code_paths": [str(path) for path in self.code_paths],
            "log_paths": [str(path) for path in self.log_paths],
            "output_image_paths": [str(path) for path in self.output_image_paths],
            "ground_truth_path": str(self.native_record.ground_truth_path) if self.native_record.ground_truth_path else None,
            "data_dir": str(self.native_record.data_dir) if self.native_record.data_dir else None,
            "has_external_data": self.native_record.has_external_data,
            "metadata": dict(self.metadata),
        }
        if include_instruction:
            result["simple_instruction"] = self.native_record.simple_instruction
            result["expert_instruction"] = self.native_record.expert_instruction
        if include_eval_raw:
            result["eval_raw"] = self.eval_raw
        return result


class MatplotBenchInstructionAdapter:
    """Load MatPlotBench records from the MatPlotAgent benchmark layout."""

    def __init__(self, benchmark_root: str | Path, instruction_file: str | Path | None = None) -> None:
        self.benchmark_root = Path(benchmark_root)
        self.instruction_path = Path(instruction_file) if instruction_file else self.benchmark_root / "benchmark_instructions.json"

    def iter_records(self) -> Iterable[MatplotBenchRecord]:
        records = self._load_records()
        for raw in records:
            case_id = str(raw["id"])
            data_dir = self.benchmark_root / "data" / case_id
            ground_truth_path = self.benchmark_root / "ground_truth" / f"example_{case_id}.png"
            yield MatplotBenchRecord(
                case_id=case_id,
                simple_instruction=raw["simple_instruction"],
                expert_instruction=raw["expert_instruction"],
                ground_truth_path=ground_truth_path,
                data_dir=data_dir if data_dir.exists() else None,
                metadata={
                    "benchmark": "MatPlotBench",
                    "source_instruction_path": str(self.instruction_path),
                    "native_id": raw["id"],
                },
            )

    def count(self) -> int:
        return len(self._load_records())

    def get_record(self, case_id: str | int) -> MatplotBenchRecord:
        normalized_id = str(case_id)
        for record in self.iter_records():
            if record.case_id == normalized_id:
                return record
        raise KeyError(f"MatPlotBench case not found: {case_id}")

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.instruction_path.exists():
            raise FileNotFoundError(f"MatPlotBench instruction file not found: {self.instruction_path}")
        records = json.loads(self.instruction_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("MatPlotBench instruction file must contain a JSON list.")
        for index, record in enumerate(records):
            missing = {"id", "simple_instruction", "expert_instruction"} - set(record)
            if missing:
                raise ValueError(f"MatPlotBench record at index {index} is missing fields: {sorted(missing)}")
        return records


class MatplotBenchWorkspaceAdapter:
    """Join MatPlotBench instructions with generated-code workspace outputs."""

    def __init__(
        self,
        benchmark_root: str | Path,
        workspace_root: str | Path,
        eval_results_path: str | Path | None = None,
        instruction_file: str | Path | None = None,
    ) -> None:
        self.benchmark_root = Path(benchmark_root)
        self.workspace_root = Path(workspace_root)
        self.eval_results_path = Path(eval_results_path) if eval_results_path else self.workspace_root / "eval_results.json"
        self.instruction_adapter = MatplotBenchInstructionAdapter(self.benchmark_root, instruction_file=instruction_file)

    def iter_records(self) -> Iterable[MatplotBenchWorkspaceRecord]:
        eval_details = self._load_eval_details()
        for native_record in self.instruction_adapter.iter_records():
            workspace_dir = self.workspace_root / f"example_{native_record.case_id}"
            code_paths = self._code_paths(workspace_dir)
            selected_code_path, stage, iteration = self._select_code_path(code_paths)
            detail = eval_details.get(native_record.case_id, {})
            yield MatplotBenchWorkspaceRecord(
                native_record=native_record,
                workspace_dir=workspace_dir,
                code_paths=code_paths,
                selected_code_path=selected_code_path,
                selected_code_stage=stage,
                selected_code_iteration=iteration,
                log_paths=self._log_paths(workspace_dir),
                output_image_paths=self._output_image_paths(workspace_dir),
                score=detail.get("score"),
                eval_raw=detail.get("raw"),
                eval_error=detail.get("error"),
                metadata={
                    "benchmark": "MatPlotBench",
                    "source_instruction_path": str(self.instruction_adapter.instruction_path),
                    "source_workspace_root": str(self.workspace_root),
                    "source_eval_results_path": str(self.eval_results_path),
                },
            )

    def select_candidates(
        self,
        score_lte: float | int | None = None,
        score_gte: float | int | None = None,
        require_code: bool = True,
        limit: int | None = None,
    ) -> tuple[MatplotBenchWorkspaceRecord, ...]:
        records = list(self.iter_records())
        if require_code:
            records = [record for record in records if record.has_selected_code]
        if score_lte is not None:
            records = [record for record in records if record.score is not None and record.score <= score_lte]
        if score_gte is not None:
            records = [record for record in records if record.score is not None and record.score >= score_gte]
        records.sort(key=lambda record: (float(record.score) if record.score is not None else float("inf"), int(record.case_id)))
        if limit is not None:
            records = records[:limit]
        return tuple(records)

    def write_candidate_manifest(
        self,
        path: str | Path,
        score_lte: float | int | None = None,
        score_gte: float | int | None = None,
        require_code: bool = True,
        limit: int | None = None,
        include_instruction: bool = True,
        include_eval_raw: bool = False,
    ) -> tuple[MatplotBenchWorkspaceRecord, ...]:
        candidates = self.select_candidates(
            score_lte=score_lte,
            score_gte=score_gte,
            require_code=require_code,
            limit=limit,
        )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [candidate.to_dict(include_instruction=include_instruction, include_eval_raw=include_eval_raw) for candidate in candidates]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return candidates

    def summary(self) -> dict[str, Any]:
        records = list(self.iter_records())
        scored = [record for record in records if record.score is not None]
        with_code = sum(1 for record in records if record.has_selected_code)
        with_image = sum(1 for record in records if record.has_output_image)
        low_score = sum(1 for record in scored if float(record.score) < 50)
        return {
            "total_records": len(records),
            "scored_records": len(scored),
            "records_with_selected_code": with_code,
            "records_with_output_image": with_image,
            "low_score_lt_50": low_score,
            "eval_summary": self._load_eval_summary(),
        }

    def _load_eval_results(self) -> dict[str, Any]:
        if not self.eval_results_path.exists():
            return {}
        return json.loads(self.eval_results_path.read_text(encoding="utf-8"))

    def _load_eval_details(self) -> dict[str, dict[str, Any]]:
        data = self._load_eval_results()
        details = data.get("details", {})
        return {str(key): value for key, value in details.items()}

    def _load_eval_summary(self) -> dict[str, Any]:
        data = self._load_eval_results()
        summary = data.get("summary", {})
        return dict(summary) if isinstance(summary, dict) else {}

    def _code_paths(self, workspace_dir: Path) -> tuple[Path, ...]:
        if not workspace_dir.exists():
            return ()
        return tuple(sorted(workspace_dir.glob("code_action_*.py"), key=lambda path: path.name))

    def _log_paths(self, workspace_dir: Path) -> tuple[Path, ...]:
        if not workspace_dir.exists():
            return ()
        return tuple(sorted(workspace_dir.glob("*.py.log"), key=lambda path: path.name))

    def _output_image_paths(self, workspace_dir: Path) -> tuple[Path, ...]:
        if not workspace_dir.exists():
            return ()
        preferred = [workspace_dir / "novice_final.png", workspace_dir / "novice.png"]
        existing_preferred = [path for path in preferred if path.exists()]
        other_images = sorted(
            [path for path in workspace_dir.glob("*.png") if path not in existing_preferred],
            key=lambda path: path.name,
        )
        return tuple(existing_preferred + other_images)

    def _select_code_path(self, code_paths: tuple[Path, ...]) -> tuple[Path | None, str | None, int | None]:
        if not code_paths:
            return None, None, None
        selected = max(code_paths, key=self._code_rank)
        stage, iteration = self._parse_code_name(selected)
        return selected, stage, iteration

    def _code_rank(self, path: Path) -> tuple[int, int, str]:
        stage, iteration = self._parse_code_name(path)
        stage_rank = {"initial": 0, "vis_refined": 1}.get(stage or "", -1)
        return stage_rank, iteration if iteration is not None else -1, path.name

    def _parse_code_name(self, path: Path) -> tuple[str | None, int | None]:
        match = re.search(r"code_action_.*_(initial|vis_refined)_(\d+)\.py$", path.name)
        if not match:
            return None, None
        return match.group(1), int(match.group(2))

@dataclass(frozen=True)
class MatplotBenchGenerationCase:
    """MatPlotBench case shape consumed by ChartGenerationPipeline."""

    case_id: str
    query: str
    schema: TableSchema
    rows: tuple[dict[str, Any], ...] = ()
    generation_mode: str = "instruction_only"
    ground_truth_path: Path | None = None
    data_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def generation_context(self) -> dict[str, Any]:
        return {
            "benchmark": "MatPlotBench",
            "ground_truth_path": str(self.ground_truth_path) if self.ground_truth_path else None,
            "data_dir": str(self.data_dir) if self.data_dir else None,
            **dict(self.metadata),
        }


class MatplotBenchGenerationAdapter:
    """Load MatPlotBench-like manifests for instruction-to-image generation runs.

    This adapter deliberately keeps native instruction-only tasks separate from
    table-to-chart tasks. If a record has no rows and no schema columns, the
    generation mode is `instruction_only`; codegen should then rely only on
    explicitly stated constants and visual structure in the prompt/context.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        prefer_expert: bool = True,
        case_ids: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.prefer_expert = prefer_expert
        self.case_ids = tuple(str(case_id) for case_id in case_ids) if case_ids is not None else ()
        self.limit = limit

    def iter_cases(self) -> Iterable[MatplotBenchGenerationCase]:
        selected = set(self.case_ids)
        yielded = 0
        for raw in self._load_records():
            case = self._case_from_record(raw)
            if selected and case.case_id not in selected and str(case.metadata.get("native_id", "")) not in selected:
                continue
            yield case
            yielded += 1
            if self.limit is not None and yielded >= self.limit:
                break

    def count(self) -> int:
        return sum(1 for _ in self.iter_cases())

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"MatPlotBench generation manifest not found: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("cases", "records", "data"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError("MatPlotBench generation manifest must contain a JSON list or an object with cases/records/data.")
        return [dict(item) for item in payload]

    def _case_from_record(self, raw: dict[str, Any]) -> MatplotBenchGenerationCase:
        case_id = str(raw.get("case_id") or raw.get("id") or raw.get("native_id") or "matplotbench-case")
        query = _matplotbench_query(raw, prefer_expert=self.prefer_expert)
        schema = _matplotbench_schema(raw.get("schema"))
        rows = tuple(dict(row) for row in (raw.get("rows") or ()))
        generation_mode = str(raw.get("generation_mode") or _matplotbench_generation_mode(schema, rows))
        ground_truth_path = _optional_path(raw.get("ground_truth_path"))
        data_dir = _optional_path(raw.get("data_dir") or raw.get("workspace_dir"))
        metadata = {
            "source_manifest": str(self.manifest_path),
            "native_id": raw.get("native_id", raw.get("id")),
            "score": raw.get("score"),
            "failure_source": raw.get("failure_source"),
            "failure_definition": raw.get("failure_definition"),
            "selected_code_path": raw.get("selected_code_path") or raw.get("generated_code_path"),
            "selected_log_path": raw.get("selected_log_path"),
            "output_image_paths": raw.get("output_image_paths", ()),
            "simple_instruction": raw.get("simple_instruction"),
            "expert_instruction": raw.get("expert_instruction"),
            **dict(raw.get("metadata") or {}),
        }
        return MatplotBenchGenerationCase(
            case_id=case_id,
            query=query,
            schema=schema,
            rows=rows,
            generation_mode=generation_mode,
            ground_truth_path=ground_truth_path,
            data_dir=data_dir,
            metadata={key: value for key, value in metadata.items() if value not in (None, "", [])},
        )


def _matplotbench_query(raw: dict[str, Any], *, prefer_expert: bool) -> str:
    candidates = (
        raw.get("query"),
        raw.get("expert_instruction") if prefer_expert else raw.get("simple_instruction"),
        raw.get("simple_instruction") if prefer_expert else raw.get("expert_instruction"),
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    raise ValueError("MatPlotBench generation record is missing query/simple_instruction/expert_instruction.")


def _matplotbench_schema(raw_schema: Any) -> TableSchema:
    if isinstance(raw_schema, dict):
        columns = raw_schema.get("columns", raw_schema)
        table_name = str(raw_schema.get("table_name", "table"))
        if isinstance(columns, dict):
            return TableSchema(columns={str(key): str(value) for key, value in columns.items()}, table_name=table_name)
    return TableSchema(columns={})


def _matplotbench_generation_mode(schema: TableSchema, rows: tuple[dict[str, Any], ...]) -> str:
    if rows or schema.columns:
        return "table"
    return "instruction_only"


def _optional_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text) if text else None

@dataclass(frozen=True)
class MatplotBenchEvalExportRecord:
    case_id: str
    native_id: int
    workspace_dir: Path
    image_path: Path | None
    code_path: Path | None
    manifest_path: Path | None = None
    report_path: Path | None = None
    skipped_reason: str | None = None

    @property
    def exported(self) -> bool:
        return self.skipped_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "native_id": self.native_id,
            "workspace_dir": str(self.workspace_dir),
            "image_path": str(self.image_path) if self.image_path else None,
            "code_path": str(self.code_path) if self.code_path else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "report_path": str(self.report_path) if self.report_path else None,
            "skipped_reason": self.skipped_reason,
            "exported": self.exported,
        }


class MatplotBenchEvalWorkspaceExporter:
    """Export generation outputs into the workspace layout expected by MatPlotBench eval_qwen.py."""

    def __init__(
        self,
        generation_summary_path: str | Path,
        workspace_dir: str | Path,
        *,
        image_filename: str = "novice_final.png",
        code_filename: str = "code_action_groundedchart_initial_0.py",
        overwrite: bool = True,
    ) -> None:
        self.generation_summary_path = Path(generation_summary_path)
        self.workspace_dir = Path(workspace_dir)
        self.image_filename = image_filename
        self.code_filename = code_filename
        self.overwrite = overwrite

    def export(self) -> dict[str, Any]:
        summary = self._load_summary()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        records = [self._export_case(dict(case)) for case in summary.get("cases", [])]
        exported_ids = sorted(record.native_id for record in records if record.exported)
        manifest = {
            "source_summary_path": str(self.generation_summary_path),
            "workspace_dir": str(self.workspace_dir),
            "total_cases": len(records),
            "exported_cases": sum(1 for record in records if record.exported),
            "skipped_cases": sum(1 for record in records if not record.exported),
            "native_ids": exported_ids,
            "records": [record.to_dict() for record in records],
            "eval_qwen_commands_by_id": [
                f"python evaluation/eval_qwen.py --workspace {self.workspace_dir} --start_id {native_id} --end_id {native_id} --workers 1"
                for native_id in exported_ids
            ],
        }
        manifest_path = self.workspace_dir / "groundedchart_eval_export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def _load_summary(self) -> dict[str, Any]:
        if not self.generation_summary_path.exists():
            raise FileNotFoundError(f"Generation summary not found: {self.generation_summary_path}")
        payload = json.loads(self.generation_summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            raise ValueError("Generation summary must be an object with a `cases` list.")
        return payload

    def _export_case(self, case: dict[str, Any]) -> MatplotBenchEvalExportRecord:
        case_id = str(case.get("case_id") or "")
        native_id = _native_id_from_generation_case(case)
        if native_id is None:
            return MatplotBenchEvalExportRecord(
                case_id=case_id,
                native_id=-1,
                workspace_dir=self.workspace_dir,
                image_path=None,
                code_path=None,
                skipped_reason="missing_native_id",
            )
        case_workspace = self.workspace_dir / f"example_{native_id}"
        case_workspace.mkdir(parents=True, exist_ok=True)

        image_source = _optional_path(case.get("image_path"))
        code_source = _optional_path(case.get("final_code_path"))
        if image_source is None or not image_source.exists():
            return MatplotBenchEvalExportRecord(
                case_id=case_id,
                native_id=native_id,
                workspace_dir=case_workspace,
                image_path=None,
                code_path=None,
                skipped_reason="missing_image_path",
            )
        if code_source is None or not code_source.exists():
            return MatplotBenchEvalExportRecord(
                case_id=case_id,
                native_id=native_id,
                workspace_dir=case_workspace,
                image_path=None,
                code_path=None,
                skipped_reason="missing_final_code_path",
            )

        target_image = case_workspace / self.image_filename
        target_code = case_workspace / self.code_filename
        _copy_file(image_source, target_image, overwrite=self.overwrite)
        _copy_file(code_source, target_code, overwrite=self.overwrite)

        copied_manifest = _copy_optional_generation_artifact(case.get("manifest_path"), case_workspace / "groundedchart_generation_manifest.json", overwrite=self.overwrite)
        copied_report = _copy_optional_generation_artifact(case.get("report_path"), case_workspace / "groundedchart_generation_report.json", overwrite=self.overwrite)
        (case_workspace / "workflow.log").write_text(
            "GroundedChart export for MatPlotBench evaluation\n"
            f"case_id: {case_id}\n"
            f"native_id: {native_id}\n"
            f"source_image: {image_source}\n"
            f"source_code: {code_source}\n",
            encoding="utf-8",
        )
        return MatplotBenchEvalExportRecord(
            case_id=case_id,
            native_id=native_id,
            workspace_dir=case_workspace,
            image_path=target_image,
            code_path=target_code,
            manifest_path=copied_manifest,
            report_path=copied_report,
        )


def _native_id_from_generation_case(case: dict[str, Any]) -> int | None:
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    candidates = (metadata.get("native_id"), case.get("native_id"), case.get("id"))
    for candidate in candidates:
        try:
            if candidate is not None and str(candidate).strip() != "":
                return int(candidate)
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?:^|[-_])(?:failed[-_])?(\d+)$", str(case.get("case_id") or ""))
    if match:
        return int(match.group(1))
    return None


def _copy_optional_generation_artifact(source: Any, target: Path, *, overwrite: bool) -> Path | None:
    source_path = _optional_path(source)
    if source_path is None or not source_path.exists():
        return None
    _copy_file(source_path, target, overwrite=overwrite)
    return target


def _copy_file(source: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
