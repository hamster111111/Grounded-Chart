from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DATA_FILE_SUFFIXES = frozenset({".csv", ".tsv", ".json", ".xlsx", ".xls"})


@dataclass(frozen=True)
class SourceFileSummary:
    name: str
    path: str
    suffix: str
    size_bytes: int | None = None
    columns: tuple[str, ...] = ()
    preview_rows: tuple[dict[str, Any], ...] = ()
    row_count_preview: int | None = None
    read_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        payload["preview_rows"] = [dict(row) for row in self.preview_rows]
        return payload


@dataclass(frozen=True)
class SourceDataPlan:
    workspace: str
    files: tuple[SourceFileSummary, ...] = ()
    mentioned_files: tuple[str, ...] = ()
    missing_mentioned_files: tuple[str, ...] = ()
    schema_constraints: tuple[dict[str, Any], ...] = ()
    global_constraints: tuple[str, ...] = ()

    @property
    def has_files(self) -> bool:
        return bool(self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "files": [item.to_dict() for item in self.files],
            "mentioned_files": list(self.mentioned_files),
            "missing_mentioned_files": list(self.missing_mentioned_files),
            "schema_constraints": [dict(item) for item in self.schema_constraints],
            "global_constraints": list(self.global_constraints),
        }


@dataclass(frozen=True)
class SourceDataExecution:
    plan: SourceDataPlan
    loaded_tables: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "loaded_tables": [dict(item) for item in self.loaded_tables],
        }


class SourceDataPlanner:
    def build_plan(
        self,
        *,
        workspace: str | Path | None,
        instruction: str = "",
        max_preview_rows: int = 5,
        max_files: int = 12,
    ) -> SourceDataPlan:
        root = Path(workspace).resolve() if workspace is not None else None
        files: list[SourceFileSummary] = []
        if root is not None and root.exists() and root.is_dir():
            for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
                if not path.is_file() or path.suffix.lower() not in DATA_FILE_SUFFIXES:
                    continue
                files.append(summarize_source_file(path, max_preview_rows=max_preview_rows))
                if len(files) >= max_files:
                    break
        mentioned = mentioned_available_files(instruction, files)
        available_names = {item.name for item in files}
        constraints = [schema_constraint_for_file(item) for item in files]
        return SourceDataPlan(
            workspace=str(root) if root is not None else "",
            files=tuple(files),
            mentioned_files=tuple(mentioned),
            missing_mentioned_files=tuple(name for name in mentioned if name not in available_names),
            schema_constraints=tuple(constraints),
            global_constraints=(
                "Use listed source files as the data source when they are relevant to the instruction.",
                "Treat listed columns as exact unless code explicitly creates derived columns.",
                "Do not fabricate random/sample data for a listed source file.",
                "For wide tables with Year plus measure columns, use direct columns or melt before long-form plotting.",
            ),
        )


class SourceDataExecutor:
    def execute(self, plan: SourceDataPlan, *, max_rows_per_table: int = 200) -> SourceDataExecution:
        loaded = []
        for file_summary in plan.files:
            loaded.append(load_source_table(file_summary, max_rows=max_rows_per_table))
        return SourceDataExecution(plan=plan, loaded_tables=tuple(loaded))


def summarize_source_file(path: str | Path, *, max_preview_rows: int = 5) -> SourceFileSummary:
    source = Path(path)
    suffix = source.suffix.lower()
    columns: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    read_error = None
    try:
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                columns = [str(item) for item in list(reader.fieldnames or [])]
                for index, row in enumerate(reader):
                    if index >= max_preview_rows:
                        break
                    preview_rows.append({str(key): short_cell(value) for key, value in dict(row).items()})
        elif suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            metadata["json_type"] = type(payload).__name__
            if isinstance(payload, list):
                preview_rows = compact_json_rows(payload[:max_preview_rows])
                if payload and isinstance(payload[0], dict):
                    columns = [str(key) for key in payload[0].keys()]
            elif isinstance(payload, dict):
                columns = [str(key) for key in list(payload.keys())[:24]]
                preview_rows = [{"key": str(key), "value": short_cell(value)} for key, value in list(payload.items())[:max_preview_rows]]
        else:
            read_error = "preview_not_supported_for_excel"
    except Exception as exc:
        read_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    return SourceFileSummary(
        name=source.name,
        path=str(source),
        suffix=suffix,
        size_bytes=source.stat().st_size if source.exists() else None,
        columns=tuple(columns),
        preview_rows=tuple(preview_rows),
        row_count_preview=len(preview_rows),
        read_error=read_error,
        metadata=metadata,
    )


def load_source_table(file_summary: SourceFileSummary, *, max_rows: int) -> dict[str, Any]:
    path = Path(file_summary.path)
    result: dict[str, Any] = {
        "name": file_summary.name,
        "path": file_summary.path,
        "suffix": file_summary.suffix,
        "columns": list(file_summary.columns),
        "rows": [],
        "row_count_loaded": 0,
        "truncated": False,
        "read_error": None,
    }
    try:
        if file_summary.suffix in {".csv", ".tsv"}:
            delimiter = "\t" if file_summary.suffix == ".tsv" else ","
            rows = []
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                result["columns"] = [str(item) for item in list(reader.fieldnames or [])]
                for index, row in enumerate(reader):
                    if index >= max_rows:
                        result["truncated"] = True
                        break
                    rows.append({str(key): parse_scalar(value) for key, value in dict(row).items()})
            result["rows"] = rows
            result["row_count_loaded"] = len(rows)
        elif file_summary.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                result["rows"] = [jsonable(item) for item in payload[:max_rows]]
                result["row_count_loaded"] = len(result["rows"])
                result["truncated"] = len(payload) > max_rows
            else:
                result["rows"] = [jsonable(payload)]
                result["row_count_loaded"] = 1
        else:
            result["read_error"] = "load_not_supported_for_excel"
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return result


def mentioned_available_files(instruction: str, files: Iterable[SourceFileSummary]) -> list[str]:
    instruction_lower = str(instruction or "").lower()
    mentioned = []
    for item in files:
        if item.name.lower() in instruction_lower:
            mentioned.append(item.name)
    return sorted(set(mentioned))


def schema_constraint_for_file(file_summary: SourceFileSummary) -> dict[str, Any]:
    columns = list(file_summary.columns)
    return {
        "name": file_summary.name,
        "columns": columns,
        "schema_type": infer_table_schema_type(columns),
        "usage_constraints": schema_usage_constraints(name=file_summary.name, columns=columns),
    }


def infer_table_schema_type(columns: list[str]) -> str:
    normalized = {column.strip().lower() for column in columns}
    if "category" in normalized and "value" in normalized:
        return "long_category_value_table"
    if "year" in normalized and len(columns) >= 3:
        return "wide_year_measure_table"
    return "unknown_table"


def schema_usage_constraints(*, name: str, columns: list[str]) -> list[str]:
    constraints = [f"{name}: exact columns are {columns}."]
    normalized = {column.strip().lower() for column in columns}
    if "year" in normalized and {"urban", "rural"}.issubset(normalized):
        constraints.append(f"{name}: wide table; use Urban/Rural directly or melt before Category/Value logic.")
        constraints.append(f"{name}: do not pivot on Category/Value unless those columns were created first.")
    if "age group" in normalized and "consumption ratio" in normalized:
        constraints.append(f"{name}: preserve Age Group and Consumption Ratio semantics.")
    return constraints


def compact_json_rows(values: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in values:
        if isinstance(item, dict):
            rows.append({str(key): short_cell(value) for key, value in item.items()})
        else:
            rows.append({"value": short_cell(item)})
    return rows


def parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def short_cell(value: Any) -> Any:
    text = "" if value is None else str(value)
    if len(text) <= 80:
        return text
    return text[:77] + "..."


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return str(value)
