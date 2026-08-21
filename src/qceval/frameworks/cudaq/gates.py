"""CUDA-Q AST gate-call lowering helpers."""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from qceval.evals.ir.core import _to_little_endian_matrix
from qceval.frameworks.cudaq.values import (
    _CUDAQ_MATRICES,
    _attr_or_name,
    _const_float,
    _const_int,
    _rotation_matrix,
)

_CUDAQ_MEASUREMENT_CALLS = frozenset({"mz", "mx", "my", "measure"})
_CUDAQ_IGNORED_CALLS = frozenset(
    {
        "qvector",
        "qreg",
        "qalloc",
        "qubit",
        "kernel",
        "make_kernel",
        "register_operation",
        "float",
        "float32",
        "float64",
    }
)
_CUDAQ_ROTATIONS = frozenset({"rx", "ry", "rz", "r1"})


def _cudaq_gate_from_call(
    call: ast.Call,
    *,
    constants: dict[str, Any] | None = None,
    registered: dict[str, np.ndarray] | None = None,
    register_name: str | None = None,
    num_qubits: int | None = None,
) -> tuple[str, np.ndarray, list[int], list[int]] | None:
    name = _attr_or_name(call.func)
    if name in _CUDAQ_MEASUREMENT_CALLS or name in _CUDAQ_IGNORED_CALLS:
        return None
    if name == "ctrl" and isinstance(call.func, ast.Attribute):
        return _cudaq_controlled_gate(
            call,
            constants=constants,
            registered=registered,
            register_name=register_name,
            num_qubits=num_qubits,
        )
    if name == "adj" and isinstance(call.func, ast.Attribute):
        return _cudaq_adjoint_gate(
            call,
            constants=constants,
            registered=registered,
            register_name=register_name,
            num_qubits=num_qubits,
        )
    base_name, matrix = _cudaq_base_gate_matrix(call.func, registered=registered)
    if matrix is not None:
        wires = _call_wires(call, register_name=register_name, num_qubits=num_qubits)
        if not wires:
            raise NotImplementedError(f"CUDA-Q gate {base_name!r} has no resolved wires")
        return base_name, matrix, wires, []
    if name in _CUDAQ_ROTATIONS:
        return _cudaq_rotation_gate(
            call,
            name,
            constants,
            register_name=register_name,
            num_qubits=num_qubits,
        )
    raise NotImplementedError(f"unsupported CUDA-Q gate call: {name or ast.dump(call.func)}")


def _cudaq_controlled_gate(
    call: ast.Call,
    *,
    constants: dict[str, Any] | None = None,
    registered: dict[str, np.ndarray] | None = None,
    register_name: str | None = None,
    num_qubits: int | None = None,
) -> tuple[str, np.ndarray, list[int], list[int]]:
    assert isinstance(call.func, ast.Attribute)
    base = call.func.value
    adjoint = False
    if isinstance(base, ast.Attribute) and base.attr == "adj":
        adjoint = True
        base = base.value
    base_name, matrix = _cudaq_base_gate_matrix(base, registered=registered)
    if matrix is not None:
        if adjoint:
            matrix = matrix.conj().T
        wires = _call_wires(call, register_name=register_name, num_qubits=num_qubits)
        target_count = _matrix_qubit_count(matrix)
        if len(wires) <= target_count:
            raise NotImplementedError(f"CUDA-Q controlled {base_name!r} call has no control wires")
        return (
            base_name,
            matrix,
            wires[-target_count:],
            wires[:-target_count],
        )
    rotation_name = _attr_or_name(base)
    if rotation_name in _CUDAQ_ROTATIONS:
        angle = _const_float(call.args[0], constants=constants) if call.args else None
        if angle is None:
            raise NotImplementedError(f"CUDA-Q controlled gate {rotation_name!r} has non-constant angle")
        wires = _call_wires(
            call.args[1:],
            register_name=register_name,
            num_qubits=num_qubits,
        )
        if len(wires) < 2:
            raise NotImplementedError(f"CUDA-Q controlled {rotation_name!r} call has no control wires")
        matrix = _rotation_matrix(rotation_name, angle)
        if adjoint:
            matrix = matrix.conj().T
        return rotation_name, matrix, wires[-1:], wires[:-1]
    raise NotImplementedError(f"unsupported CUDA-Q controlled gate: {rotation_name or base_name!r}")


def _cudaq_adjoint_gate(
    call: ast.Call,
    *,
    constants: dict[str, Any] | None = None,
    registered: dict[str, np.ndarray] | None = None,
    register_name: str | None = None,
    num_qubits: int | None = None,
) -> tuple[str, np.ndarray, list[int], list[int]]:
    assert isinstance(call.func, ast.Attribute)
    base = call.func.value
    base_name, matrix = _cudaq_base_gate_matrix(base, registered=registered)
    if matrix is not None:
        wires = _call_wires(call, register_name=register_name, num_qubits=num_qubits)
        if not wires:
            raise NotImplementedError(f"CUDA-Q adjoint gate {base_name!r} has no resolved wires")
        return base_name, matrix.conj().T, wires, []
    rotation_name = _attr_or_name(base)
    if rotation_name in _CUDAQ_ROTATIONS:
        angle = _const_float(call.args[0], constants=constants) if call.args else None
        if angle is None:
            raise NotImplementedError(f"CUDA-Q adjoint gate {rotation_name!r} has non-constant angle")
        wires = _call_wires(
            call.args[1:],
            register_name=register_name,
            num_qubits=num_qubits,
        )
        if not wires:
            raise NotImplementedError(f"CUDA-Q adjoint gate {rotation_name!r} has no resolved wires")
        return (
            rotation_name,
            _rotation_matrix(rotation_name, angle).conj().T,
            wires,
            [],
        )
    raise NotImplementedError(f"unsupported CUDA-Q adjoint gate: {rotation_name or base_name!r}")


def _cudaq_rotation_gate(
    call: ast.Call,
    name: str,
    constants: dict[str, Any] | None,
    *,
    register_name: str | None = None,
    num_qubits: int | None = None,
) -> tuple[str, np.ndarray, list[int], list[int]]:
    angle = _const_float(call.args[0], constants=constants) if call.args else None
    if angle is None:
        raise NotImplementedError(f"CUDA-Q gate {name!r} has non-constant angle")
    wires = _call_wires(
        call.args[1:],
        register_name=register_name,
        num_qubits=num_qubits,
    )
    if not wires:
        raise NotImplementedError(f"CUDA-Q gate {name!r} has no resolved wires")
    return name, _rotation_matrix(name, angle), wires, []


def _cudaq_base_gate_matrix(
    node: ast.AST,
    registered: dict[str, np.ndarray] | None = None,
) -> tuple[str, np.ndarray | None]:
    if isinstance(node, ast.Attribute) and node.attr == "adj":
        base_name, matrix = _cudaq_base_gate_matrix(node.value, registered=registered)
        return (
            base_name,
            None if matrix is None else matrix.conj().T,
        )
    base_name = _attr_or_name(node)
    matrix = _CUDAQ_MATRICES.get(base_name)
    if matrix is None and registered and base_name in registered:
        # Registered custom operations are declared with the first qubit
        # argument as the most significant subsystem, like the intrinsic
        # multi-qubit gates.
        value = registered[base_name]
        matrix = _to_little_endian_matrix(value, int(round(math.log2(value.shape[0]))))
    return base_name, matrix


def _call_wires(
    node_or_args: ast.Call | Sequence[ast.AST],
    *,
    register_name: str | None = None,
    num_qubits: int | None = None,
) -> list[int]:
    args = node_or_args.args if isinstance(node_or_args, ast.Call) else node_or_args
    wires: list[int] = []
    for argument in args:
        if (
            register_name is not None
            and num_qubits is not None
            and isinstance(argument, ast.Name)
            and argument.id == register_name
        ):
            wires.extend(range(num_qubits))
            continue
        wires.extend(_subscript_indices(argument))
    return wires


def _subscript_indices(node: ast.AST) -> list[int]:
    if isinstance(node, ast.Subscript):
        value = _const_int(node.slice)
        return [value] if value is not None else []
    if isinstance(node, ast.List | ast.Tuple):
        out: list[int] = []
        for element in node.elts:
            out.extend(_subscript_indices(element))
        return out
    return []


def _matrix_qubit_count(matrix: np.ndarray) -> int:
    return int(round(math.log2(matrix.shape[0])))
