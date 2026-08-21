"""SSA assignment evaluation, classical memory, and phi resolution."""

from __future__ import annotations

import re
from typing import Any

from qceval.frameworks.cudaq.qir.gates import (
    _CUSTOM_UNITARY,
    _GENERALIZED,
    _custom_unitary_operation,
    _generalized_operation,
    _qis_operation,
)
from qceval.frameworks.cudaq.qir.models import (
    QirParseError,
    _BitPredicate,
    _ComplexArray,
    _Memory,
    _MemoryRef,
    _QubitArray,
    _QubitArraySlot,
    _QubitRef,
    _State,
)
from qceval.frameworks.cudaq.qir.tokens import (
    _REFERENCE,
    _SSA_REFERENCE,
    _VALUE,
    _call_arguments,
    _first_resolved_ssa,
    _floating_scalar,
    _floating_token,
    _integer_token,
    _last_ssa,
    _qubit_pointer,
    _resolve_value,
    _result_pointer,
    _scalar_token,
    _split_arguments,
    _typed_integer,
)
from qceval.semantics.ir import Operation

_ASSIGNMENT = re.compile(rf"^\s*({_VALUE})\s*=\s*(.+)$")


def _global_constants(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    scalar_pattern = re.compile(
        r"^(@[-a-zA-Z$._0-9]+)\s+=\s+(?:private|internal)\s+constant\s+\[[0-9]+\s+x\s+(i64|double)\]\s+\[(.*)\]$"
    )
    complex_pattern = re.compile(
        r"^(@[-a-zA-Z$._0-9]+)\s+=\s+(?:private|internal)\s+constant\s+"
        r"\[[0-9]+\s+x\s+\{\s*double\s*,\s*double\s*\}\]\s+\[(.*)\]$"
    )
    for raw in text.splitlines():
        match = scalar_pattern.match(raw.strip())
        if match is not None:
            item_type = match.group(2)
            items = _split_arguments(match.group(3))
            parsed = [
                _integer_token(item.removeprefix("i64 ").strip(), _State())
                if item_type == "i64"
                else _floating_token(item)
                for item in items
            ]
            values[match.group(1)] = _Memory(parsed)
            continue
        complex_match = complex_pattern.match(raw.strip())
        if complex_match is not None:
            values[complex_match.group(1)] = _ComplexArray(_complex_elements(complex_match.group(2)))
    return values


def _complex_elements(text: str) -> tuple[complex, ...]:
    elements: list[complex] = []
    for item in _split_arguments(text):
        item = item.strip().removeprefix("{ double, double }").strip()
        if item == "zeroinitializer":
            elements.append(0j)
            continue
        parts = _split_arguments(_balanced_braces(item))
        if len(parts) != 2:
            raise QirParseError("QIR complex constant element is malformed")
        real = _floating_token(parts[0])
        imaginary = _floating_token(parts[1])
        elements.append(complex(real, imaginary))
    return tuple(elements)


def _balanced_braces(text: str) -> str:
    if not text.startswith("{") or not text.endswith("}"):
        raise QirParseError(f"QIR constant element has unbalanced braces: {text[:80]}")
    return text[1:-1]


def _evaluate_instruction(line: str, state: _State) -> Operation | None:
    assignment = _ASSIGNMENT.match(line)
    if assignment is not None:
        _evaluate_assignment(assignment.group(1), assignment.group(2), state)
        return None
    if _GENERALIZED in line:
        return _generalized_operation(line, state)
    if _CUSTOM_UNITARY in line:
        return _custom_unitary_operation(line, state)
    if "@__quantum__qis__" in line:
        return _qis_operation(line, state)
    if line.startswith("store "):
        _evaluate_store(line, state)
        return None
    if line.startswith("call void @__quantum__rt__") or line.startswith("call void @free("):
        return None
    if line.startswith("call void @__quantum__qis__"):
        raise QirParseError(f"unsupported QIS call: {line[:160]}")
    raise QirParseError(f"unsupported adaptive-QIR instruction: {line[:160]}")


def _evaluate_assignment(target: str, expression: str, state: _State) -> None:
    handlers = (_evaluate_runtime_assignment, _evaluate_memory_assignment, _evaluate_numeric_assignment)
    if any(handler(target, expression, state) for handler in handlers):
        return
    raise QirParseError(f"unsupported adaptive-QIR assignment: {expression[:160]}")


def _evaluate_runtime_assignment(target: str, expression: str, state: _State) -> bool:
    if "@__quantum__qis__read_result__body" in expression:
        state.values[target] = _BitPredicate(_result_pointer(expression))
        return True
    if "@__quantum__rt__array_create_1d" in expression:
        arguments = _call_arguments(expression, "@__quantum__rt__array_create_1d")
        size = _typed_integer(arguments[-1], state)
        state.values[target] = _QubitArray([None] * size)
        return True
    if "@__quantum__rt__array_get_element_ptr_1d" in expression:
        arguments = _call_arguments(expression, "@__quantum__rt__array_get_element_ptr_1d")
        array = _resolve_value(_last_ssa(arguments[0]), state, _QubitArray)
        state.values[target] = _QubitArraySlot(array, _typed_integer(arguments[1], state))
        return True
    if "@__quantum__rt__array_concatenate" in expression:
        arguments = _call_arguments(expression, "@__quantum__rt__array_concatenate")
        left = _resolve_value(_last_ssa(arguments[0]), state, _QubitArray)
        right = _resolve_value(_last_ssa(arguments[1]), state, _QubitArray)
        state.values[target] = _QubitArray([*left.values, *right.values])
        return True
    if "@__quantum__rt__array_get_size_1d" in expression:
        arguments = _call_arguments(expression, "@__quantum__rt__array_get_size_1d")
        array = _resolve_value(_last_ssa(arguments[0]), state, _QubitArray)
        state.values[target] = len(array.values)
        return True
    return False


def _evaluate_memory_assignment(target: str, expression: str, state: _State) -> bool:
    if expression.startswith("alloca "):
        count_match = re.search(r"\[([0-9]+)\s+x\s+", expression)
        count = 1 if count_match is None else int(count_match.group(1))
        state.values[target] = _Memory([None] * count)
        return True
    if expression.startswith("bitcast "):
        source = _last_ssa(expression)
        value = _resolve_value(source, state)
        state.values[target] = _MemoryRef(value, 0) if isinstance(value, _Memory) else value
        return True
    if expression.startswith("getelementptr "):
        _evaluate_getelementptr(target, expression, state)
        return True
    if expression.startswith("load "):
        reference = _resolve_value(_last_ssa(expression), state, _MemoryRef)
        value = reference.memory.values[reference.index]
        if value is None:
            raise QirParseError("QIR loads an uninitialized classical value")
        state.values[target] = value
        return True
    return False


def _evaluate_numeric_assignment(target: str, expression: str, state: _State) -> bool:
    handlers = (_evaluate_phi_or_math_call, _evaluate_numeric_operation, _evaluate_numeric_cast)
    return any(handler(target, expression, state) for handler in handlers)


def _evaluate_phi_or_math_call(target: str, expression: str, state: _State) -> bool:
    if expression.startswith("phi "):
        _evaluate_phi(target, expression, state)
        return True
    if expression.startswith("inttoptr ") and expression.endswith("to %Qubit*"):
        source = re.search(r"inttoptr\s+i64\s+([^ ]+)\s+to\s+%Qubit\*", expression)
        if source is None:
            raise QirParseError("QIR dynamic qubit pointer is malformed")
        state.values[target] = _QubitRef(_integer_token(source.group(1), state))
        return True
    if expression.startswith("call double @__mlir_math_fpowi_f64_i64"):
        arguments = _call_arguments(expression, "@__mlir_math_fpowi_f64_i64")
        base = _floating_scalar(arguments[0], state)
        exponent = _typed_integer(arguments[1], state)
        state.values[target] = base**exponent
        return True
    return False


def _evaluate_numeric_operation(target: str, expression: str, state: _State) -> bool:
    if expression.startswith("icmp "):
        _evaluate_comparison(target, expression, state)
        return True
    if re.match(r"^(?:add|sub|mul|sdiv|udiv|srem|urem)\s+", expression):
        _evaluate_arithmetic(target, expression, state)
        return True
    if re.match(r"^(?:fadd|fsub|fmul|fdiv|frem)\s+", expression):
        _evaluate_float_arithmetic(target, expression, state)
        return True
    if re.match(r"^(?:and|or|xor)\s+i1\s+", expression):
        _evaluate_boolean_arithmetic(target, expression, state)
        return True
    return False


def _evaluate_numeric_cast(target: str, expression: str, state: _State) -> bool:
    if re.match(r"^(?:zext|sext|trunc)\s+", expression):
        source = _first_resolved_ssa(expression, state)
        state.values[target] = _resolve_value(source, state)
        return True
    if re.match(r"^(?:sitofp|uitofp|fptosi|fptoui|fpext|fptrunc)\s+", expression):
        source = _first_resolved_ssa(expression, state)
        value = _resolve_value(source, state)
        state.values[target] = (
            float(value) if expression.startswith(("sitofp", "uitofp", "fpext", "fptrunc")) else int(value)
        )
        return True
    if expression.startswith("extractvalue "):
        _evaluate_extractvalue(target, expression, state)
        return True
    return False


def _evaluate_phi(target: str, expression: str, state: _State) -> None:
    if state.predecessor is None:
        raise QirParseError("QIR phi node has no predecessor block")
    incoming = re.findall(r"\[\s*([^,\]]+)\s*,\s*%?([-a-zA-Z$._0-9]+)\s*\]", expression)
    selected = next((value.strip() for value, label in incoming if label == state.predecessor), None)
    if selected is None:
        raise QirParseError(f"QIR phi node has no value for predecessor {state.predecessor!r}")
    if expression.startswith("phi double "):
        state.values[target] = _floating_scalar(selected, state)
    else:
        state.values[target] = _scalar_token(selected, state)


def _evaluate_extractvalue(target: str, expression: str, state: _State) -> None:
    source = _first_resolved_ssa(expression, state)
    value = _resolve_value(source, state)
    index_match = re.search(r",\s*([0-9]+)\s*$", expression)
    if not isinstance(value, tuple) or index_match is None:
        raise QirParseError("QIR extractvalue does not reference a bounded tuple")
    index = int(index_match.group(1))
    if not 0 <= index < len(value):
        raise QirParseError("QIR extractvalue index is outside its tuple")
    state.values[target] = value[index]


def _evaluate_getelementptr(target: str, expression: str, state: _State) -> None:
    references = _REFERENCE.findall(expression)
    if not references:
        raise QirParseError("QIR getelementptr has no base pointer")
    base = _resolve_value(references[0], state)
    integers = re.findall(r"i(?:32|64)\s+(-?[0-9]+|%[-a-zA-Z$._0-9]+)", expression)
    index = _integer_token(integers[-1], state) if integers else 0
    if isinstance(base, _Memory):
        state.values[target] = _MemoryRef(base, index)
        return
    if isinstance(base, _MemoryRef):
        state.values[target] = _MemoryRef(base.memory, base.index + index)
        return
    raise QirParseError(f"QIR getelementptr base is not bounded classical memory: {expression[:120]}")


def _evaluate_store(line: str, state: _State) -> None:
    if line.startswith("store %Qubit*"):
        references = _SSA_REFERENCE.findall(line)
        if not references:
            raise QirParseError("QIR qubit-array store has no destination")
        slot = _resolve_value(references[-1], state, _QubitArraySlot)
        slot.array.values[slot.index] = _qubit_pointer(line, state)
        return
    struct = re.match(
        r"store\s+\{\s*i64\s*,\s*i64\s*\}\s+\{\s*i64\s+([^,]+),\s*i64\s+([^}]+)\s*\},.*(%[-a-zA-Z$._0-9]+)",
        line,
    )
    if struct is not None:
        struct_value = (_integer_token(struct.group(1).strip(), state), _integer_token(struct.group(2).strip(), state))
        _store_memory_value(struct.group(3), struct_value, state)
        return
    floating = re.match(r"store\s+double\s+([^,]+),\s+[^%]*(%[-a-zA-Z$._0-9]+)", line)
    if floating is not None:
        _store_memory_value(floating.group(2), _floating_scalar(floating.group(1), state), state)
        return
    match = re.match(r"store\s+i(?:1|8|16|32|64)\s+([^,]+),\s+[^%]*(%[-a-zA-Z$._0-9]+)", line)
    if match is None:
        raise QirParseError(f"unsupported QIR store: {line[:160]}")
    scalar_value = _scalar_token(match.group(1).strip(), state)
    _store_memory_value(match.group(2), scalar_value, state)


def _store_memory_value(pointer: str, value: Any, state: _State) -> None:
    reference = _resolve_value(pointer, state, _MemoryRef)
    if not 0 <= reference.index < len(reference.memory.values):
        raise QirParseError("QIR classical store index is outside bounded memory")
    reference.memory.values[reference.index] = value


def _evaluate_comparison(target: str, expression: str, state: _State) -> None:
    match = re.match(r"icmp\s+(eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge)\s+i[0-9]+\s+([^,]+),\s+(.+)", expression)
    if match is None:
        raise QirParseError(f"unsupported QIR comparison: {expression[:160]}")
    left = _scalar_token(match.group(2).strip(), state)
    right = _scalar_token(match.group(3).strip(), state)
    predicate = _predicate_comparison(match.group(1), left, right)
    if predicate is not None:
        state.values[target] = predicate
        return
    if not isinstance(left, int) or not isinstance(right, int):
        raise QirParseError("QIR comparison operands are not concrete integers")
    operators = {
        "eq": left == right,
        "ne": left != right,
        "slt": left < right,
        "ult": left < right,
        "sle": left <= right,
        "ule": left <= right,
        "sgt": left > right,
        "ugt": left > right,
        "sge": left >= right,
        "uge": left >= right,
    }
    state.values[target] = operators[match.group(1)]


def _predicate_comparison(operator: str, left: Any, right: Any) -> _BitPredicate | None:
    if isinstance(right, _BitPredicate) and isinstance(left, int):
        left, right = right, left
    if not isinstance(left, _BitPredicate) or not isinstance(right, int) or right not in {0, 1}:
        return None
    if operator not in {"eq", "ne"}:
        raise QirParseError("QIR measurement predicates only support equality comparisons")
    inverted = left.inverted ^ (right == 0) ^ (operator == "ne")
    return _BitPredicate(left.bit, inverted)


def _evaluate_arithmetic(target: str, expression: str, state: _State) -> None:
    match = re.match(r"(add|sub|mul|sdiv|udiv|srem|urem)\s+i[0-9]+\s+([^,]+),\s+(.+)", expression)
    if match is None:
        raise QirParseError(f"unsupported QIR arithmetic: {expression[:160]}")
    left = _integer_token(match.group(2).strip(), state)
    right = _integer_token(match.group(3).strip(), state)
    if match.group(1) in {"sdiv", "udiv", "srem", "urem"} and right == 0:
        raise QirParseError("QIR classical arithmetic divides by zero")

    def _sdiv() -> int:
        # LLVM sdiv truncates toward zero; Python // floors.
        quotient = abs(left) // abs(right)
        return -quotient if (left < 0) != (right < 0) else quotient

    functions = {
        "add": lambda: left + right,
        "sub": lambda: left - right,
        "mul": lambda: left * right,
        "sdiv": _sdiv,
        "udiv": lambda: left // right,
        # LLVM srem takes the sign of the dividend: left == sdiv * right + srem.
        "srem": lambda: left - _sdiv() * right,
        "urem": lambda: left % right,
    }
    state.values[target] = functions[match.group(1)]()


def _evaluate_float_arithmetic(target: str, expression: str, state: _State) -> None:
    match = re.match(r"(fadd|fsub|fmul|fdiv|frem)\s+double\s+([^,]+),\s+(.+)", expression)
    if match is None:
        raise QirParseError(f"unsupported QIR floating-point arithmetic: {expression[:160]}")
    left = _floating_scalar(match.group(2), state)
    right = _floating_scalar(match.group(3), state)
    if match.group(1) in {"fdiv", "frem"} and right == 0:
        raise QirParseError("QIR floating-point arithmetic divides by zero")
    functions = {
        "fadd": lambda: left + right,
        "fsub": lambda: left - right,
        "fmul": lambda: left * right,
        "fdiv": lambda: left / right,
        "frem": lambda: left % right,
    }
    state.values[target] = functions[match.group(1)]()


def _evaluate_boolean_arithmetic(target: str, expression: str, state: _State) -> None:
    match = re.match(r"(and|or|xor)\s+i1\s+([^,]+),\s+(.+)", expression)
    if match is None:
        raise QirParseError(f"unsupported QIR boolean arithmetic: {expression[:160]}")
    left = _scalar_token(match.group(2), state)
    right = _scalar_token(match.group(3), state)
    if not isinstance(left, bool | int) or not isinstance(right, bool | int):
        raise QirParseError("QIR boolean arithmetic operands are not concrete")
    left_bool, right_bool = bool(left), bool(right)
    functions = {
        "and": lambda: left_bool and right_bool,
        "or": lambda: left_bool or right_bool,
        "xor": lambda: left_bool != right_bool,
    }
    state.values[target] = functions[match.group(1)]()
