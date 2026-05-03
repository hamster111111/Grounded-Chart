from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Literal

PatchAnchorKind = Literal["method_call", "function_call", "text"]
PatchOpName = Literal["replace_call_arg", "replace_keyword_arg", "remove_keyword_arg", "insert_after_anchor", "replace_text"]


@dataclass(frozen=True)
class PatchAnchor:
    kind: PatchAnchorKind
    name: str | None = None
    text: str | None = None
    occurrence: int = 1


@dataclass(frozen=True)
class PatchOperation:
    op: PatchOpName
    anchor: PatchAnchor
    arg_index: int | None = None
    keyword: str | None = None
    new_value: Any = None
    description: str | None = None


@dataclass(frozen=True)
class PatchApplyResult:
    code: str
    applied: bool
    applied_ops: tuple[PatchOperation, ...] = ()
    rejected_reason: str | None = None


def parse_patch_operations(value: object) -> tuple[PatchOperation, ...]:
    if not isinstance(value, list):
        return ()
    operations: list[PatchOperation] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        op_name = str(item.get("op") or "").strip().lower()
        if op_name not in {
            "replace_call_arg",
            "replace_keyword_arg",
            "remove_keyword_arg",
            "insert_after_anchor",
            "replace_text",
        }:
            continue
        anchor_payload = item.get("anchor")
        if not isinstance(anchor_payload, dict):
            continue
        anchor_kind = str(anchor_payload.get("kind") or "").strip().lower()
        if anchor_kind not in {"method_call", "function_call", "text"}:
            continue
        occurrence = _coerce_positive_int(anchor_payload.get("occurrence"), fallback=1)
        anchor = PatchAnchor(
            kind=anchor_kind,
            name=_normalize_optional_string(anchor_payload.get("name")),
            text=_normalize_optional_string(anchor_payload.get("text")),
            occurrence=occurrence,
        )
        operations.append(
            PatchOperation(
                op=op_name,
                anchor=anchor,
                arg_index=_coerce_non_negative_int(item.get("arg_index")),
                keyword=_normalize_optional_string(item.get("keyword")),
                new_value=item.get("new_value"),
                description=_normalize_optional_string(item.get("description")),
            )
        )
    return tuple(operations)


def apply_patch_operations(
    code: str,
    operations: tuple[PatchOperation, ...],
    *,
    max_operations: int = 2,
    max_changed_lines: int = 15,
) -> PatchApplyResult:
    if not operations:
        return PatchApplyResult(code=code, applied=False, rejected_reason="No structured patch operations were provided.")
    if len(operations) > max(1, int(max_operations)):
        return PatchApplyResult(
            code=code,
            applied=False,
            rejected_reason=f"Structured patch rejected because {len(operations)} operations exceeded the budget of {max_operations}.",
        )

    updated = code
    applied_ops: list[PatchOperation] = []
    for operation in operations:
        candidate, did_apply = _apply_operation(updated, operation)
        if not did_apply or candidate == updated:
            return PatchApplyResult(
                code=code,
                applied=False,
                applied_ops=tuple(applied_ops),
                rejected_reason=f"Structured patch op '{operation.op}' could not be applied safely.",
            )
        updated = candidate
        applied_ops.append(operation)

    changed_lines = _changed_line_count(code, updated)
    if changed_lines > max(1, int(max_changed_lines)):
        return PatchApplyResult(
            code=code,
            applied=False,
            rejected_reason=(
                f"Structured patch rejected because it would change {changed_lines} lines, "
                f"exceeding the budget of {max_changed_lines}."
            ),
        )
    return PatchApplyResult(code=updated, applied=True, applied_ops=tuple(applied_ops))


def _apply_operation(code: str, operation: PatchOperation) -> tuple[str, bool]:
    if operation.op == "replace_call_arg":
        if operation.arg_index is None:
            return code, False
        return _replace_call_argument(code, operation.anchor, operation.arg_index, operation.new_value)
    if operation.op == "replace_keyword_arg":
        if not operation.keyword:
            return code, False
        return _replace_keyword_argument(code, operation.anchor, operation.keyword, operation.new_value)
    if operation.op == "remove_keyword_arg":
        if not operation.keyword:
            return code, False
        return _remove_keyword_argument(code, operation.anchor, operation.keyword)
    if operation.op == "insert_after_anchor":
        return _insert_after_anchor(code, operation.anchor, operation.new_value)
    if operation.op == "replace_text":
        return _replace_text(code, operation.anchor, operation.new_value)
    return code, False


def _replace_call_argument(code: str, anchor: PatchAnchor, arg_index: int, new_value: Any) -> tuple[str, bool]:
    call_span = _find_call_span(code, anchor)
    if call_span is None:
        return code, False
    args_start, args_end = call_span
    arguments = _split_top_level_arguments(code[args_start:args_end])
    if arg_index < 0 or arg_index >= len(arguments):
        return code, False
    arguments[arg_index] = _python_source(new_value)
    new_args = ", ".join(argument.strip() for argument in arguments if argument.strip())
    return code[:args_start] + new_args + code[args_end:], True


def _replace_keyword_argument(code: str, anchor: PatchAnchor, keyword: str, new_value: Any) -> tuple[str, bool]:
    call_span = _find_call_span(code, anchor)
    if call_span is None:
        return code, False
    args_start, args_end = call_span
    args_text = code[args_start:args_end]
    for start, end, argument in _split_top_level_argument_spans(args_text):
        if re.match(rf"^\s*{re.escape(keyword)}\s*=", argument):
            replacement = f"{keyword}={_python_source(new_value)}"
            return code[: args_start + start] + replacement + code[args_start + end :], True
    if keyword not in {
        "label",
        "projection",
        "reverse",
        "subplot_kw",
        "title",
        "title_text",
        "annotations",
        "xaxis_title",
        "yaxis_title",
        "zaxis_title",
        "scene_zaxis_title",
    }:
        return code, False
    insertion = f"{keyword}={_python_source(new_value)}"
    if args_text.strip():
        insertion = _keyword_insertion_prefix(args_text) + insertion
    return code[:args_end] + insertion + code[args_end:], True


def _remove_keyword_argument(code: str, anchor: PatchAnchor, keyword: str) -> tuple[str, bool]:
    call_span = _find_call_span(code, anchor)
    if call_span is None:
        return code, False
    args_start, args_end = call_span
    args_text = code[args_start:args_end]
    spans = _split_top_level_argument_spans(args_text)
    for index, (start, end, argument) in enumerate(spans):
        if not re.match(rf"^\s*{re.escape(keyword)}\s*=", argument):
            continue
        remove_start = start
        remove_end = end
        if index + 1 < len(spans):
            next_start = spans[index + 1][0]
            remove_end = next_start
        elif index > 0:
            previous_end = spans[index - 1][1]
            remove_start = previous_end
        return code[: args_start + remove_start] + code[args_start + remove_end :], True
    return code, False


def _insert_after_anchor(code: str, anchor: PatchAnchor, new_value: Any) -> tuple[str, bool]:
    anchor_text = _anchor_text(anchor)
    if anchor_text:
        span = _find_text_span(code, anchor_text, anchor.occurrence)
    else:
        span = _find_call_full_span(code, anchor)
    if span is None:
        return code, False
    insertion = str(new_value or "")
    if not insertion:
        return code, False
    insert_at = span[1]
    prefix = "" if code[:insert_at].endswith("\n") else "\n"
    suffix = "" if insertion.endswith("\n") or code[insert_at:].startswith("\n") else "\n"
    return code[:insert_at] + prefix + insertion + suffix + code[insert_at:], True


def _replace_text(code: str, anchor: PatchAnchor, new_value: Any) -> tuple[str, bool]:
    anchor_text = _anchor_text(anchor)
    if not anchor_text:
        return code, False
    span = _find_text_span(code, anchor_text, anchor.occurrence)
    if span is None:
        return code, False
    replacement = str(new_value or "")
    return code[: span[0]] + replacement + code[span[1] :], True


def _find_call_full_span(code: str, anchor: PatchAnchor) -> tuple[int, int] | None:
    pattern = _call_anchor_pattern(anchor)
    if pattern is None:
        return None
    matches = list(pattern.finditer(code))
    occurrence_index = anchor.occurrence - 1
    if occurrence_index < 0 or occurrence_index >= len(matches):
        return None
    match = matches[occurrence_index]
    open_paren = code.find("(", match.start())
    if open_paren == -1:
        return None
    close_paren = _find_matching_paren(code, open_paren)
    if close_paren is None:
        return None
    return match.start(), close_paren + 1

def _find_call_span(code: str, anchor: PatchAnchor) -> tuple[int, int] | None:
    pattern = _call_anchor_pattern(anchor)
    if pattern is None:
        return None
    matches = list(pattern.finditer(code))
    occurrence_index = anchor.occurrence - 1
    if occurrence_index < 0 or occurrence_index >= len(matches):
        return None
    match = matches[occurrence_index]
    open_paren = code.find("(", match.start())
    if open_paren == -1:
        return None
    close_paren = _find_matching_paren(code, open_paren)
    if close_paren is None:
        return None
    return open_paren + 1, close_paren


def _call_anchor_pattern(anchor: PatchAnchor):
    name = str(anchor.name or "").strip()
    if not name:
        return None
    if anchor.kind == "method_call":
        return re.compile(rf"\.\s*{re.escape(name)}\s*\(")
    if anchor.kind == "function_call":
        return re.compile(rf"\b{re.escape(name)}\s*\(")
    return None


def _find_matching_paren(text: str, open_index: int) -> int | None:
    depth = 0
    in_string = False
    quote_char = ""
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                in_string = False
                quote_char = ""
            continue
        if char in {"'", '"'}:
            in_string = True
            quote_char = char
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_top_level_argument_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_string = False
    quote_char = ""
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                in_string = False
                quote_char = ""
            continue
        if char in {"'", '"'}:
            in_string = True
            quote_char = char
            continue
        if char == "(":
            paren_depth += 1
            continue
        if char == ")":
            paren_depth = max(0, paren_depth - 1)
            continue
        if char == "[":
            bracket_depth += 1
            continue
        if char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            continue
        if char == "{":
            brace_depth += 1
            continue
        if char == "}":
            brace_depth = max(0, brace_depth - 1)
            continue
        if char == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            raw = text[start:index]
            stripped_start = start + len(raw) - len(raw.lstrip())
            stripped_end = index - (len(raw) - len(raw.rstrip()))
            if stripped_start < stripped_end:
                spans.append((stripped_start, stripped_end, text[stripped_start:stripped_end]))
            start = index + 1
    raw = text[start:]
    stripped_start = start + len(raw) - len(raw.lstrip())
    stripped_end = len(text) - (len(raw) - len(raw.rstrip()))
    if stripped_start < stripped_end:
        spans.append((stripped_start, stripped_end, text[stripped_start:stripped_end]))
    return spans

def _split_top_level_arguments(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    arguments: list[str] = []
    current: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_string = False
    quote_char = ""
    escaped = False
    for char in text:
        if in_string:
            current.append(char)
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                in_string = False
                quote_char = ""
            continue
        if char in {"'", '"'}:
            in_string = True
            quote_char = char
            current.append(char)
            continue
        if char == "(":
            paren_depth += 1
            current.append(char)
            continue
        if char == ")":
            paren_depth = max(0, paren_depth - 1)
            current.append(char)
            continue
        if char == "[":
            bracket_depth += 1
            current.append(char)
            continue
        if char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            current.append(char)
            continue
        if char == "{":
            brace_depth += 1
            current.append(char)
            continue
        if char == "}":
            brace_depth = max(0, brace_depth - 1)
            current.append(char)
            continue
        if char == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            arguments.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        arguments.append(tail)
    return arguments


def _keyword_insertion_prefix(args_text: str) -> str:
    if "\n" in args_text:
        indent = _last_argument_indent(args_text)
        return ",\n" + indent
    return ", "


def _last_argument_indent(args_text: str) -> str:
    lines = args_text.splitlines()
    for line in reversed(lines):
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return ""

def _python_source(value: Any) -> str:
    return repr(value)


def _anchor_text(anchor: PatchAnchor) -> str:
    if anchor.text:
        return anchor.text
    if anchor.kind == "text" and anchor.name:
        return anchor.name
    return ""


def _find_text_span(text: str, needle: str, occurrence: int) -> tuple[int, int] | None:
    if not needle:
        return None
    start = 0
    for _ in range(max(1, occurrence)):
        index = text.find(needle, start)
        if index == -1:
            return None
        start = index + len(needle)
    return index, index + len(needle)


def _changed_line_count(original: str, updated: str) -> int:
    original_lines = original.splitlines()
    updated_lines = updated.splitlines()
    changed = 0
    matcher = difflib.SequenceMatcher(a=original_lines, b=updated_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed


def _normalize_optional_string(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _coerce_positive_int(value: object, *, fallback: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return fallback
    return normalized if normalized > 0 else fallback


def _coerce_non_negative_int(value: object) -> int | None:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized >= 0 else None
