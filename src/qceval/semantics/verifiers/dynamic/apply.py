"""Exact gate and unitary application on pure statevectors."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from qceval.semantics.ir import Operation, ParameterKind
from qceval.semantics.verifiers.dynamic.payload import _has_dense_payload, _semantic_matrix
from qceval.semantics.verifiers.dynamic.simulator import DynamicSimulationError
from qceval.semantics.verifiers.result import SemanticStatus


def _apply_operation(state: np.ndarray, operation: Operation, num_qubits: int) -> np.ndarray:
    del num_qubits
    name = operation.name.lower()
    if name == "dense_unitary":
        return _apply_dense_operation(state, operation)
    if operation.power is not None:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "gate_modifier_unsupported")
    controls = tuple((item.wire, item.value) for item in operation.controls)
    if name == "qft":
        return _apply_qft(state, operation, controls)
    wires = operation.quantum_wires
    if name in {"cx", "cnot", "cz", "cy", "ch"} and len(wires) == 2 and not controls:
        # Some adapters keep a controlled name's control wire inline instead of
        # externalizing it; the first listed wire is the control.
        controls = ((wires[0], 1),)
        wires = wires[1:]
    if name in {"cswap", "fredkin"} and len(wires) == 3 and not controls:
        controls = ((wires[0], 1),)
        wires = wires[1:]
    if name in {"swap", "cswap", "fredkin"} and len(wires) == 2:
        # SWAP is self-inverse, so the inverse modifier needs no handling.
        return _apply_swap(state, *wires, controls=controls)
    return _apply_named_or_dense(state, operation, name, wires, controls)


def _apply_named_or_dense(
    state: np.ndarray,
    operation: Operation,
    name: str,
    wires: tuple[int, ...],
    controls: tuple[tuple[int, int], ...],
) -> np.ndarray:
    # An exact dense payload is authoritative provider data: adapters may pair
    # it with an approximate or aliased name, so the payload must win.
    if _has_dense_payload(operation):
        return _apply_dense_operation(state, operation)
    gate_matrix = _named_gate_matrix(name, operation)
    if gate_matrix is None and operation.definition:
        if operation.inverse:
            raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "gate_modifier_unsupported")
        return _apply_definition(state, operation)
    if gate_matrix is None or len(wires) != 1:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, f"dynamic_gate_unsupported:{name}")
    if operation.inverse:
        gate_matrix = gate_matrix.conjugate().T
    return _apply_single(state, gate_matrix, wires[0], controls)


def _apply_dense_operation(state: np.ndarray, operation: Operation) -> np.ndarray:
    matrix = _semantic_matrix(operation)
    if operation.inverse:
        matrix = matrix.conjugate().T
    controls = tuple((item.wire, item.value) for item in operation.controls)
    first_target_msb = dict(operation.semantic_data).get("matrix_wire_order", "big_endian") == "big_endian"
    return _apply_dense(
        state,
        matrix,
        operation.quantum_wires,
        controls,
        first_target_msb=first_target_msb,
    )


def _apply_qft(
    state: np.ndarray,
    operation: Operation,
    controls: tuple[tuple[int, int], ...],
) -> np.ndarray:
    dimension = 2 ** len(operation.quantum_wires)
    indices = np.arange(dimension)
    matrix = np.exp(2j * np.pi * np.outer(indices, indices) / dimension) / math.sqrt(dimension)
    if operation.inverse:
        matrix = matrix.conjugate().T
    return _apply_dense(state, matrix, operation.quantum_wires, controls)


def _named_gate_matrix(name: str, operation: Operation) -> np.ndarray | None:
    matrices: dict[str, np.ndarray] = {
        "id": _I,
        "i": _I,
        "x": _X,
        "y": _Y,
        "z": _Z,
        "h": _H,
        "s": _S,
        "sdg": _S.conjugate().T,
        "t": _T,
        "tdg": _T.conjugate().T,
        "sx": _SX,
        "sxdg": _SX.conjugate().T,
        "cx": _X,
        "cnot": _X,
        "ccx": _X,
        "cz": _Z,
    }
    gate_matrix = matrices.get(name)
    if name in {"rx", "ry", "rz", "p", "phase", "r1", "cp", "cphase", "mcp", "mcphase"}:
        rotation_name = "p" if name in {"cp", "cphase", "mcp", "mcphase"} else name
        gate_matrix = _rotation(rotation_name, _numeric_parameter(operation, 0))
    if name in {"xpow", "ypow", "zpow", "hpow"}:
        gate_matrix = _pauli_power(name, _numeric_parameter(operation, 0))
    if name in {"u", "u3"}:
        gate_matrix = _u_matrix(
            _numeric_parameter(operation, 0),
            _numeric_parameter(operation, 1),
            _numeric_parameter(operation, 2),
        )
    return gate_matrix


def _apply_definition(state: np.ndarray, operation: Operation) -> np.ndarray:
    value = state
    wire_map = (*[item.wire for item in operation.controls], *operation.quantum_wires)
    for child in operation.definition:
        value = _apply_operation(value, _remap_operation(child, wire_map), 0)
    return value


def _numeric_parameter(operation: Operation, index: int) -> float:
    try:
        parameter = operation.parameters[index]
    except IndexError as exc:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "missing_numeric_parameter") from exc
    if parameter.kind is not ParameterKind.NUMBER:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "symbolic_parameter_unbound")
    try:
        value = float(parameter.value)
    except ValueError as exc:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "invalid_numeric_parameter") from exc
    if not math.isfinite(value):
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "invalid_numeric_parameter")
    return value


def _rotation(name: str, angle: float) -> np.ndarray:
    if name == "rx":
        return math.cos(angle / 2) * _I - 1j * math.sin(angle / 2) * _X
    if name == "ry":
        return math.cos(angle / 2) * _I - 1j * math.sin(angle / 2) * _Y
    if name == "rz":
        return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
    return np.diag([1.0, np.exp(1j * angle)])


def _u_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    return np.asarray(
        [
            [cosine, -np.exp(1j * lam) * sine],
            [np.exp(1j * phi) * sine, np.exp(1j * (phi + lam)) * cosine],
        ],
        dtype=np.complex128,
    )


def _pauli_power(name: str, exponent: float) -> np.ndarray:
    phase = np.exp(1j * np.pi * exponent)
    if name == "zpow":
        return np.diag([1.0, phase])
    operator = {"xpow": _X, "ypow": _Y, "hpow": _H}[name]
    positive = (_I + operator) / 2
    negative = (_I - operator) / 2
    return positive + phase * negative


def _remap_operation(operation: Operation, wire_map: tuple[int, ...]) -> Operation:
    """Map a gate-definition operation from local to parent program wires."""

    try:
        quantum_wires = tuple(wire_map[wire] for wire in operation.quantum_wires)
        controls = tuple(replace(control, wire=wire_map[control.wire]) for control in operation.controls)
    except IndexError as exc:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "gate_definition_wire_out_of_range") from exc
    return replace(operation, quantum_wires=quantum_wires, controls=controls)


def _apply_single(
    state: np.ndarray,
    matrix: np.ndarray,
    target: int,
    controls: tuple[tuple[int, int], ...],
) -> np.ndarray:
    output = state.copy()
    stride = 1 << target
    for base in range(state.size):
        if base & stride or any(((base >> wire) & 1) != value for wire, value in controls):
            continue
        one = base | stride
        output[base] = matrix[0, 0] * state[base] + matrix[0, 1] * state[one]
        output[one] = matrix[1, 0] * state[base] + matrix[1, 1] * state[one]
    return output


def _apply_swap(
    state: np.ndarray,
    first: int,
    second: int,
    controls: tuple[tuple[int, int], ...] = (),
) -> np.ndarray:
    output = np.zeros_like(state)
    for basis, amplitude in enumerate(state):
        first_value = (basis >> first) & 1
        second_value = (basis >> second) & 1
        target = basis
        controlled = all(((basis >> wire) & 1) == value for wire, value in controls)
        if controlled and first_value != second_value:
            target ^= (1 << first) | (1 << second)
        output[target] = amplitude
    return output


def _apply_dense(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: tuple[int, ...],
    controls: tuple[tuple[int, int], ...],
    *,
    first_target_msb: bool = True,
) -> np.ndarray:
    local_dimension = 2 ** len(targets)
    if matrix.shape != (local_dimension, local_dimension):
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "dense_gate_dimension_mismatch")
    output = state.copy()
    target_mask = sum(1 << wire for wire in targets)
    for base in range(state.size):
        if base & target_mask or any(((base >> wire) & 1) != value for wire, value in controls):
            continue
        indices = np.asarray(
            [
                base
                | sum(
                    ((local >> (len(targets) - index - 1) if first_target_msb else local >> index) & 1) << wire
                    for index, wire in enumerate(targets)
                )
                for local in range(local_dimension)
            ]
        )
        output[indices] = matrix @ state[indices]
    return output


_I = np.eye(2, dtype=np.complex128)
_X = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
_H = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
_S = np.diag([1, 1j]).astype(np.complex128)
_T = np.diag([1, np.exp(0.25j * math.pi)]).astype(np.complex128)
_SX = np.asarray([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) / 2
