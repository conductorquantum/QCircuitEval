"""SSA token splitting and typed argument parsing for adaptive QIR."""

from __future__ import annotations

import re
import struct
from typing import Any

from qceval.frameworks.cudaq.qir.models import (
    QirParseError,
    _BitPredicate,
    _QubitRef,
    _State,
)

_VALUE = r"%[-a-zA-Z$._0-9]+"

_QUBIT_POINTER = re.compile(r"(?:%Qubit\*|i8\*)\s+inttoptr\s+\(i64\s+(-?[0-9]+)\s+to\s+(?:%Qubit\*|i8\*)\)")

_RESULT_POINTER = re.compile(r"%Result\*\s+inttoptr\s+\(i64\s+(-?[0-9]+)\s+to\s+%Result\*\)")

_SSA_REFERENCE = re.compile(_VALUE)

_REFERENCE = re.compile(r"[%@][-a-zA-Z$._0-9]+")


def _resolve_value(name: str, state: _State, expected: type[Any] | None = None) -> Any:
    if name not in state.values:
        raise QirParseError(f"QIR SSA value {name!r} is unresolved")
    value = state.values[name]
    if expected is not None and not isinstance(value, expected):
        raise QirParseError(f"QIR SSA value {name!r} has an unexpected type")
    return value


def _integer_token(token: str, state: _State) -> int:
    token = token.strip()
    if token in state.values:
        value = state.values[token]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        raise QirParseError(f"QIR integer SSA value {token!r} is not concrete")
    try:
        return int(token)
    except ValueError as exc:
        raise QirParseError(f"QIR integer token {token!r} is not concrete") from exc


def _scalar_token(token: str, state: _State) -> int | _BitPredicate:
    token = token.strip()
    if token in {"true", "false"}:
        return int(token == "true")
    if token in state.values and isinstance(state.values[token], _BitPredicate):
        return state.values[token]
    return _integer_token(token, state)


def _typed_integer(argument: str, state: _State) -> int:
    match = re.search(r"i(?:1|8|16|32|64)\s+(-?[0-9]+|%[-a-zA-Z$._0-9]+)", argument)
    if match is None:
        raise QirParseError(f"QIR typed integer is malformed: {argument[:80]}")
    return _integer_token(match.group(1), state)


def _floating_token(argument: str) -> float:
    match = re.search(r"double\s+([^\s,)]+)", argument)
    if match is None:
        raise QirParseError(f"QIR floating-point argument is malformed: {argument[:80]}")
    token = match.group(1)
    if token.startswith("0x") and len(token) == 18:
        return struct.unpack(">d", bytes.fromhex(token[2:]))[0]
    try:
        return float(token)
    except ValueError as exc:
        raise QirParseError(f"QIR floating-point token {token!r} is unsupported") from exc


def _floating_scalar(token: str, state: _State) -> float:
    token = token.strip().removeprefix("double ").strip()
    if token in state.values:
        value = state.values[token]
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        raise QirParseError(f"QIR floating-point SSA value {token!r} is not concrete")
    return _floating_token(f"double {token}")


def _qubit_pointer(argument: str, state: _State | None = None) -> int:
    if re.search(r"(?:%Qubit\*|i8\*)\s+null(?:[,)]|$)", argument):
        return 0
    match = _QUBIT_POINTER.search(argument)
    if match is not None:
        value = int(match.group(1))
    else:
        reference = _last_ssa(argument)
        if state is None:
            raise QirParseError(f"QIR qubit pointer is not static: {argument[:100]}")
        value = _resolve_value(reference, state)
        if isinstance(value, _QubitRef):
            value = value.wire
        if not isinstance(value, int):
            raise QirParseError(f"QIR qubit pointer is not concrete: {argument[:100]}")
    if value < 0:
        raise QirParseError("QIR qubit pointer is negative")
    return value


def _i8_wire(argument: str, state: _State) -> int:
    return _qubit_pointer(argument, state)


def _result_pointer(argument: str) -> int:
    if re.search(r"%Result\*\s+null(?:[,)]|$)", argument):
        return 0
    match = _RESULT_POINTER.search(argument)
    if match is None:
        raise QirParseError(f"QIR result pointer is not static: {argument[:100]}")
    value = int(match.group(1))
    if value < 0:
        raise QirParseError("QIR result pointer is negative")
    return value


def _first_ssa(text: str) -> str:
    match = _SSA_REFERENCE.search(text)
    if match is None:
        raise QirParseError(f"QIR expression has no SSA reference: {text[:100]}")
    return match.group(0)


def _first_resolved_ssa(text: str, state: _State) -> str:
    for name in _SSA_REFERENCE.findall(text):
        if name in state.values:
            return name
    raise QirParseError(f"QIR expression has no resolved SSA reference: {text[:100]}")


def _last_ssa(text: str) -> str:
    matches = _SSA_REFERENCE.findall(text)
    if not matches:
        raise QirParseError(f"QIR expression has no SSA reference: {text[:100]}")
    return matches[-1]


def _call_arguments(text: str, marker: str) -> list[str]:
    start = text.find(marker)
    if start < 0:
        raise QirParseError(f"QIR call marker {marker!r} is missing")
    open_index = text.find("(", start + len(marker))
    return _split_arguments(_balanced_contents(text, open_index))


def _balanced_contents(text: str, open_index: int) -> str:
    if open_index < 0 or open_index >= len(text) or text[open_index] != "(":
        raise QirParseError("QIR call has no opening parenthesis")
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index]
    raise QirParseError("QIR call has unbalanced parentheses")


def _split_arguments(text: str) -> list[str]:
    arguments: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(text):
        if character in depths:
            depths[character] += 1
        elif character in pairs:
            depths[pairs[character]] -= 1
        elif character == "," and all(value == 0 for value in depths.values()):
            arguments.append(text[start:index].strip())
            start = index + 1
    arguments.append(text[start:].strip())
    return arguments


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0]
