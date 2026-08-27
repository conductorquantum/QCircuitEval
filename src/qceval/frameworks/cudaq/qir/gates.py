"""QIS gate and measurement lowering from adaptive QIR call sites."""

from __future__ import annotations

import re

import numpy as np

from qceval.frameworks.cudaq.qir.models import QirParseError, _ComplexArray, _QubitArray, _State
from qceval.frameworks.cudaq.qir.tokens import (
    _REFERENCE,
    _SSA_REFERENCE,
    _balanced_contents,
    _floating_scalar,
    _i8_wire,
    _last_ssa,
    _qubit_pointer,
    _resolve_value,
    _result_pointer,
    _split_arguments,
    _typed_integer,
)
from qceval.semantics.ir import Control, Operation, OperationKind
from qceval.semantics.lowering.utils import bounded_matrix_semantic_data, normalize_parameter

_QIS_SYMBOL = re.compile(r"@__quantum__qis__(?P<name>[a-zA-Z0-9_]+)__(?P<variant>body|ctl|adj)")

_QIS_CALL = re.compile(r"@__quantum__qis__(?P<name>[a-zA-Z0-9_]+)__(?P<variant>body|ctl|adj)\(")

_GENERALIZED = "@generalizedInvokeWithRotationsControlsTargets("

_CUSTOM_UNITARY = "@__quantum__qis__custom_unitary("


def _qis_operation(line: str, state: _State) -> Operation | None:
    match = _QIS_CALL.search(line)
    if match is None:
        raise QirParseError(f"unsupported QIS symbol: {line[:160]}")
    name = match.group("name").lower()
    variant = match.group("variant")
    arguments = _split_arguments(_balanced_contents(line, match.end() - 1))
    if name == "read_result":
        return None
    if name == "mz":
        if len(arguments) != 2:
            raise QirParseError("QIR measurement has an invalid signature")
        return Operation(
            OperationKind.MEASUREMENT,
            "mz",
            (_qubit_pointer(arguments[0], state),),
            (_result_pointer(arguments[1]),),
            semantic_data=(("basis", "z"),),
            source_location="qir",
        )
    if name == "reset":
        return Operation(OperationKind.RESET, "reset", (_qubit_pointer(arguments[-1], state),), source_location="qir")
    angles = [
        _floating_scalar(item.strip().removeprefix("double ").strip(), state)
        for item in arguments
        if item.strip().startswith("double ")
    ]
    control_arrays = [
        _resolve_value(_last_ssa(item), state, _QubitArray)
        for item in arguments
        if item.strip().startswith("%Array* ") and _SSA_REFERENCE.search(item)
    ]
    controls = [wire for array in control_arrays for wire in _complete_array(array)]
    wires = [_qubit_pointer(item, state) for item in arguments if "%Qubit*" in item]
    return _gate_operation(name, variant, angles, wires, controls)


def _custom_unitary_operation(line: str, state: _State) -> Operation:
    """Lower a registered custom unitary applied through the QIR runtime.

    The call carries a module-level complex row-major matrix constant, a
    control qubit array, and a target qubit array. CUDA-Q declares the first
    target as the most significant subsystem, so the payload is re-encoded
    into the IR's little-endian dense-gate convention.
    """
    arguments = _split_arguments(_balanced_contents(line, line.index(_CUSTOM_UNITARY) + len(_CUSTOM_UNITARY) - 1))
    if len(arguments) != 4:
        raise QirParseError("QIR custom unitary call has an unexpected signature")
    constant_names = [name for name in _REFERENCE.findall(arguments[0]) if name.startswith("@")]
    if not constant_names:
        raise QirParseError("QIR custom unitary has no matrix constant")
    matrix_values = _resolve_value(constant_names[0], state)
    if not isinstance(matrix_values, _ComplexArray):
        raise QirParseError("QIR custom unitary matrix constant is not a bounded complex array")
    dimension = int(round(len(matrix_values.values) ** 0.5))
    if dimension * dimension != len(matrix_values.values) or dimension & (dimension - 1):
        raise QirParseError("QIR custom unitary matrix has a non-square power-of-two dimension")
    controls = _qubit_array_wires(arguments[1], state)
    targets = _qubit_array_wires(arguments[2], state)
    if 1 << len(targets) != dimension:
        raise QirParseError("QIR custom unitary target count disagrees with its matrix dimension")
    matrix = np.asarray(matrix_values.values, dtype=np.complex128).reshape(dimension, dimension)
    # Reverse subsystem order: CUDA-Q registers matrices with the first target
    # most significant; the dense payload convention is little endian.
    wire_count = len(targets)
    permutation = [int(f"{index:0{wire_count}b}"[::-1], 2) for index in range(dimension)] if wire_count > 1 else None
    if permutation is not None:
        matrix = matrix[np.ix_(permutation, permutation)]
    return Operation(
        OperationKind.GATE,
        "dense_unitary",
        tuple(targets),
        controls=tuple(Control(wire) for wire in controls),
        semantic_data=bounded_matrix_semantic_data(matrix, wire_order="little_endian"),
        source_location="qir",
    )


def _generalized_operation(line: str, state: _State) -> Operation:
    arguments = _split_arguments(_balanced_contents(line, line.index(_GENERALIZED) + len(_GENERALIZED) - 1))
    if len(arguments) < 5:
        raise QirParseError("generalized QIR gate call is truncated")
    rotation_count = _typed_integer(arguments[0], state)
    array_count = _typed_integer(arguments[1], state)
    control_count = _typed_integer(arguments[2], state)
    target_count = _typed_integer(arguments[3], state)
    marker = _QIS_SYMBOL.search(arguments[4])
    if marker is None:
        raise QirParseError("generalized QIR gate call has no QIS symbol")
    name = marker.group("name").lower()
    variant = marker.group("variant")
    cursor = 5
    angles = [
        _floating_scalar(item.strip().removeprefix("double ").strip(), state)
        for item in arguments[cursor : cursor + rotation_count]
    ]
    cursor += rotation_count
    controls: list[int] = []
    for _ in range(array_count):
        if cursor + 1 >= len(arguments):
            raise QirParseError("generalized QIR control array is truncated")
        expected_size = _typed_integer(arguments[cursor], state)
        pointer_name = _last_ssa(arguments[cursor + 1])
        array = _resolve_value(pointer_name, state, _QubitArray)
        values = _complete_array(array)
        if len(values) != expected_size:
            raise QirParseError("generalized QIR control-array size disagrees with its payload")
        controls.extend(values)
        cursor += 2
    controls.extend(_i8_wire(item, state) for item in arguments[cursor : cursor + control_count])
    cursor += control_count
    targets = [_i8_wire(item, state) for item in arguments[cursor : cursor + target_count]]
    cursor += target_count
    if cursor != len(arguments):
        raise QirParseError("generalized QIR gate call has unexpected trailing arguments")
    return _gate_operation(name, variant, angles, targets, controls)


def _gate_operation(
    name: str,
    variant: str,
    angles: list[float],
    wires: list[int],
    controls: list[int],
) -> Operation:
    name, wires, controls = _normalize_controlled_gate(name, wires, controls)
    name = _normalize_adjoint_gate(name, variant)
    _validate_gate_shape(name, wires)
    parameters = tuple(normalize_parameter(value) for value in angles)
    return Operation(
        OperationKind.GATE,
        name,
        tuple(wires),
        parameters=parameters,
        controls=tuple(Control(wire) for wire in controls),
        source_location="qir",
    )


def _normalize_controlled_gate(
    name: str,
    wires: list[int],
    controls: list[int],
) -> tuple[str, list[int], list[int]]:
    aliases = {"cnot": "x", "cx": "x", "cy": "y", "cz": "z", "ch": "h"}
    if name in aliases and not controls:
        if len(wires) != 2:
            raise QirParseError(f"QIR controlled gate {name!r} requires two wires")
        return aliases[name], [wires[1]], [wires[0]]
    if name in {"ccx", "ccnot"} and not controls:
        if len(wires) != 3:
            raise QirParseError("QIR ccx requires three wires")
        return "x", wires[2:], wires[:2]
    return name, wires, controls


def _normalize_adjoint_gate(name: str, variant: str) -> str:
    if variant != "adj":
        return name
    adjoints = {"s": "sdg", "t": "tdg", "sdg": "s", "tdg": "t"}
    if name in adjoints:
        return adjoints[name]
    if name not in {"x", "y", "z", "h", "swap"}:
        raise QirParseError(f"QIR adjoint gate {name!r} is not normalized")
    return name


def _validate_gate_shape(name: str, wires: list[int]) -> None:
    if name == "swap":
        if len(wires) != 2:
            raise QirParseError("QIR swap requires two target wires")
    elif len(wires) != 1:
        raise QirParseError(f"QIR gate {name!r} requires exactly one target wire")


def _qubit_array_wires(argument: str, state: _State) -> list[int]:
    """Resolve one ``%Array*`` argument to absolute wires; ``null`` is empty."""
    if re.search(r"%Array\*\s+null(?:[,)]|$)", argument.strip()):
        return []
    return _complete_array(_resolve_value(_last_ssa(argument), state, _QubitArray))


def _complete_array(array: _QubitArray) -> list[int]:
    if any(value is None for value in array.values):
        raise QirParseError("QIR control array contains an uninitialized qubit")
    return [int(value) for value in array.values if value is not None]
