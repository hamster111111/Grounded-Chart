from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CodeStructureArtifact:
    """Static code evidence for visual structures that runtime artists blur."""

    artifact_id: str
    artifact_type: str
    value: dict[str, Any]
    locator: dict[str, Any] = field(default_factory=dict)
    source: str = "code_structure_ast_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_code_structure_artifacts(code: str) -> tuple[dict[str, Any], ...]:
    """Extract conservative chart-structure evidence from Python plotting code.

    This extractor is intentionally evidence-producing rather than verdict-making.
    Runtime figure artifacts remain primary; these static artifacts only close gaps
    where Matplotlib exposes a generic artist family, such as stacked/grouped bars.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    visitor = _CodeStructureVisitor(code)
    visitor.visit(tree)
    return tuple(artifact.to_dict() for artifact in visitor.finalize())


@dataclass(frozen=True)
class _BarCall:
    node: ast.Call
    call_name: str
    axis_index: int | None
    position_expr: ast.AST | None


class _CodeStructureVisitor(ast.NodeVisitor):
    def __init__(self, code: str) -> None:
        self.code = code
        self.lines = code.splitlines()
        self.artifacts: list[CodeStructureArtifact] = []
        self.bar_calls: list[_BarCall] = []
        self.axis_names: dict[str, int] = {}
        self.axes_array_names: dict[str, tuple[int, int]] = {}
        self.name_values: dict[str, Any] = {}
        self.cumulative_updates: dict[str, list[ast.AST]] = {}
        self._seen: set[tuple[str, int | None, int]] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._capture_literal_assignment(target, node.value)
            self._capture_axes_assignment(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._capture_literal_assignment(node.target, node.value)
            self._capture_axes_assignment(node.target, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        name = _simple_name(node.target)
        if name is not None:
            self.cumulative_updates.setdefault(name, []).append(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        short_name = call_name.rsplit(".", 1)[-1]
        axis_index = self._axis_index_for_call(node.func)

        if short_name in {"bar", "barh"}:
            position_expr = node.args[0] if node.args else None
            self.bar_calls.append(_BarCall(node=node, call_name=short_name, axis_index=axis_index, position_expr=position_expr))
            stack_keyword = "left" if short_name == "barh" else "bottom"
            stack_value = _keyword_value(node, stack_keyword)
            if stack_value is not None and self._is_stack_reference(stack_value):
                evidence = [self._evidence(node)]
                name = _simple_name(stack_value)
                if name is not None:
                    evidence.extend(self._evidence(update) for update in self.cumulative_updates.get(name, ()))
                self._add_artifact(
                    "stacked_bar",
                    node,
                    axis_index=axis_index,
                    value={
                        "structure": "stacked_bar",
                        "call": short_name,
                        "stack_parameter": stack_keyword,
                        "evidence": evidence,
                    },
                )

        if short_name == "pie":
            explode_value = _keyword_value(node, "explode")
            if explode_value is not None:
                nonzero = self._has_nonzero_value(explode_value)
                if nonzero is not False:
                    self._add_artifact(
                        "exploded_pie",
                        node,
                        axis_index=axis_index,
                        value={
                            "structure": "exploded_pie",
                            "explode_parameter": True,
                            "nonzero_known": nonzero,
                            "evidence": [self._evidence(node)],
                        },
                    )

        if self._is_connector_call(node, call_name, short_name):
            self._add_artifact(
                "connector",
                node,
                axis_index=axis_index,
                value={
                    "structure": "connector",
                    "call": call_name,
                    "evidence": [self._evidence(node)],
                },
            )

        if self._is_subplots_call(node):
            rows, cols = _subplots_shape(node)
            self._add_artifact(
                "subplot_layout",
                node,
                value={
                    "structure": "subplot_layout",
                    "rows": rows,
                    "cols": cols,
                    "axes_count": rows * cols if rows is not None and cols is not None else None,
                    "orientation": _orientation(rows, cols),
                    "evidence": [self._evidence(node)],
                },
                locator={"scope": "figure"},
            )

        self.generic_visit(node)

    def finalize(self) -> list[CodeStructureArtifact]:
        self._add_stacked_bar_artifacts_from_updates()
        self._add_grouped_bar_artifacts()
        return self.artifacts

    def _capture_literal_assignment(self, target: ast.AST, value: ast.AST) -> None:
        name = _simple_name(target)
        if name is None:
            return
        literal = _literal_value(value)
        if literal is not _UNKNOWN:
            self.name_values[name] = literal

    def _capture_axes_assignment(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Tuple) and len(target.elts) >= 2 and self._is_subplots_call(value):
            rows, cols = _subplots_shape(value)
            axis_target = target.elts[1]
            self._register_axis_target(axis_target, rows=rows, cols=cols)
            return
        if isinstance(value, ast.Subscript):
            axis_index = self._axis_index_from_subscript(value)
            if axis_index is not None:
                for name in _flatten_target_names(target):
                    self.axis_names[name] = axis_index
            return
        value_name = _simple_name(value)
        if value_name in self.axes_array_names and isinstance(target, (ast.Tuple, ast.List)):
            for index, name in enumerate(_flatten_target_names(target)):
                self.axis_names[name] = index

    def _register_axis_target(self, target: ast.AST, *, rows: int | None, cols: int | None) -> None:
        names = _flatten_target_names(target)
        if isinstance(target, ast.Name):
            if (rows or 1) * (cols or 1) <= 1:
                self.axis_names[target.id] = 0
            elif rows is not None and cols is not None:
                self.axes_array_names[target.id] = (rows, cols)
            elif names:
                self.axes_array_names[target.id] = (1, max(1, len(names)))
            return
        for index, name in enumerate(names):
            self.axis_names[name] = index

    def _axis_index_for_call(self, func: ast.AST) -> int | None:
        if not isinstance(func, ast.Attribute):
            return None
        return self._axis_index_from_expr(func.value)

    def _axis_index_from_expr(self, node: ast.AST) -> int | None:
        name = _simple_name(node)
        if name in self.axis_names:
            return self.axis_names[name]
        if isinstance(node, ast.Subscript):
            return self._axis_index_from_subscript(node)
        return None

    def _axis_index_from_subscript(self, node: ast.Subscript) -> int | None:
        base_name = _simple_name(node.value)
        if base_name not in self.axes_array_names:
            return None
        rows, cols = self.axes_array_names[base_name]
        index = _literal_index(node.slice)
        if isinstance(index, int):
            return index
        if isinstance(index, tuple) and len(index) == 2 and all(isinstance(item, int) for item in index):
            row, col = index
            return row * cols + col
        return None

    def _is_stack_reference(self, node: ast.AST) -> bool:
        nonzero = self._has_nonzero_value(node)
        if nonzero is False:
            return False
        name = _simple_name(node)
        if name is not None and self.cumulative_updates.get(name):
            return True
        return nonzero is True or name is not None or isinstance(node, (ast.BinOp, ast.Call, ast.Subscript))

    def _has_nonzero_value(self, node: ast.AST) -> bool | None:
        value = _literal_value(node)
        if value is _UNKNOWN:
            name = _simple_name(node)
            if name is not None and name in self.name_values:
                value = self.name_values[name]
            else:
                return None
        return _contains_nonzero_number(value)

    def _is_connector_call(self, node: ast.Call, call_name: str, short_name: str) -> bool:
        if short_name in {"ConnectionPatch", "FancyArrowPatch"}:
            return True
        if short_name == "annotate" and _keyword_value(node, "arrowprops") is not None:
            return True
        if short_name == "Line2D" and _keyword_contains_text(node, "transform", "transFigure"):
            return True
        if short_name == "append" and isinstance(node.func, ast.Attribute):
            target_name = _attribute_name(node.func.value)
            if target_name.endswith(".lines") or target_name.endswith(".patches"):
                return True
        return False

    def _is_subplots_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        return _call_name(node.func).endswith("subplots")

    def _add_stacked_bar_artifacts_from_updates(self) -> None:
        for call in self.bar_calls:
            stack_keyword = "left" if call.call_name == "barh" else "bottom"
            stack_value = _keyword_value(call.node, stack_keyword)
            name = _simple_name(stack_value)
            if name is None or not self.cumulative_updates.get(name):
                continue
            evidence = [self._evidence(call.node)]
            evidence.extend(self._evidence(update) for update in self.cumulative_updates[name])
            self._add_artifact(
                "stacked_bar",
                call.node,
                axis_index=call.axis_index,
                value={
                    "structure": "stacked_bar",
                    "call": call.call_name,
                    "stack_parameter": stack_keyword,
                    "cumulative_update": name,
                    "evidence": evidence,
                },
            )
    def _add_grouped_bar_artifacts(self) -> None:
        calls_by_axis: dict[int | None, list[_BarCall]] = {}
        for call in self.bar_calls:
            calls_by_axis.setdefault(call.axis_index, []).append(call)
        for axis_index, calls in calls_by_axis.items():
            if len(calls) < 2:
                continue
            offset_calls = [call for call in calls if call.position_expr is not None and _expr_has_offset(call.position_expr)]
            if not offset_calls:
                continue
            evidence = [self._evidence(call.node) for call in offset_calls[:4]]
            self._add_artifact(
                "grouped_bar",
                offset_calls[0].node,
                axis_index=axis_index,
                value={
                    "structure": "grouped_bar",
                    "bar_call_count": len(calls),
                    "offset_call_count": len(offset_calls),
                    "evidence": evidence,
                },
            )

    def _add_artifact(
        self,
        structure: str,
        node: ast.AST,
        *,
        axis_index: int | None = None,
        value: dict[str, Any],
        locator: dict[str, Any] | None = None,
    ) -> None:
        line = int(getattr(node, "lineno", 0) or 0)
        key = (structure, axis_index, line)
        if key in self._seen:
            return
        self._seen.add(key)
        resolved_locator = dict(locator or {})
        if axis_index is not None:
            resolved_locator.setdefault("axis_index", axis_index)
            resolved_locator.setdefault("panel_id", f"panel_{axis_index}")
        artifact_id = f"actual.code_structure.{structure}.{len(self.artifacts)}"
        self.artifacts.append(
            CodeStructureArtifact(
                artifact_id=artifact_id,
                artifact_type="code_structure",
                value=value,
                locator=resolved_locator,
            )
        )

    def _evidence(self, node: ast.AST) -> dict[str, Any]:
        line = int(getattr(node, "lineno", 0) or 0)
        span = ast.get_source_segment(self.code, node)
        if span is None and 1 <= line <= len(self.lines):
            span = self.lines[line - 1].strip()
        return {"line": line, "span": span or ""}


_UNKNOWN = object()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _simple_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _flatten_target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in node.elts:
            names.extend(_flatten_target_names(item))
        return names
    return []


def _keyword_value(node: ast.Call, keyword: str) -> ast.AST | None:
    for item in node.keywords:
        if item.arg == keyword:
            return item.value
    return None


def _keyword_contains_text(node: ast.Call, keyword: str, text: str) -> bool:
    value = _keyword_value(node, keyword)
    if value is None:
        return False
    return text in ast.dump(value, include_attributes=False)


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return _UNKNOWN


def _contains_nonzero_number(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, (list, tuple)):
        values = [_contains_nonzero_number(item) for item in value]
        if any(item is True for item in values):
            return True
        if values and all(item is False for item in values):
            return False
        return None
    return None


def _literal_index(node: ast.AST) -> int | tuple[int, ...] | None:
    value = _literal_value(node)
    if isinstance(value, int):
        return value
    if isinstance(value, tuple) and all(isinstance(item, int) for item in value):
        return value
    if isinstance(node, ast.Tuple):
        items = tuple(_literal_index(item) for item in node.elts)
        if all(isinstance(item, int) for item in items):
            return items  # type: ignore[return-value]
    return None


def _subplots_shape(node: ast.Call) -> tuple[int | None, int | None]:
    rows = _int_arg(node, 0, "nrows")
    cols = _int_arg(node, 1, "ncols")
    return rows or 1, cols or 1


def _int_arg(node: ast.Call, position: int, keyword: str) -> int | None:
    if len(node.args) > position:
        value = _literal_value(node.args[position])
        if isinstance(value, int):
            return value
    for item in node.keywords:
        if item.arg == keyword:
            value = _literal_value(item.value)
            if isinstance(value, int):
                return value
    return None


def _orientation(rows: int | None, cols: int | None) -> str | None:
    if rows == 1 and cols and cols > 1:
        return "side_by_side"
    if cols == 1 and rows and rows > 1:
        return "stacked_vertical"
    if rows and cols and rows > 1 and cols > 1:
        return "grid"
    return None


def _expr_has_offset(node: ast.AST) -> bool:
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return True
        if isinstance(node.op, ast.Mult):
            return _expr_has_offset(node.left) or _expr_has_offset(node.right)
    if isinstance(node, ast.UnaryOp):
        return _expr_has_offset(node.operand)
    if isinstance(node, ast.Call):
        return any(_expr_has_offset(arg) for arg in node.args)
    if isinstance(node, ast.ListComp):
        return _expr_has_offset(node.elt)
    if isinstance(node, (ast.List, ast.Tuple)):
        return any(_expr_has_offset(item) for item in node.elts)
    if isinstance(node, ast.Subscript):
        return _expr_has_offset(node.value)
    return False




