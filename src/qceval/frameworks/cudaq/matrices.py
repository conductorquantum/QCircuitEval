"""CUDA-Q intrinsic gate matrices and register_operation evaluation."""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np

from qceval.evals.ir.core import (
    _to_little_endian_matrix,
)
from qceval.frameworks.cudaq.constfold import _attr_or_name

_MAX_REGISTERED_DIMENSION = 64


def _cudaq_registered_matrices(tree: ast.Module) -> dict[str, np.ndarray]:
    """Statically evaluate module-level ``cudaq.register_operation`` matrices.

    Only a bounded numeric expression grammar is admitted (numpy arrays,
    scalar arithmetic, and ``scipy.linalg.expm``); anything else leaves the
    operation unregistered so its use stays typed-unsupported.
    """
    environment: dict[str, Any] = {}
    registered: dict[str, np.ndarray] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            value = _static_numeric_value(statement.value, environment)
            if value is not None:
                environment[statement.targets[0].id] = value
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)):
            continue
        call = statement.value
        if _attr_or_name(call.func) != "register_operation" or len(call.args) != 2:
            continue
        name = call.args[0]
        matrix = _static_numeric_value(call.args[1], environment)
        if (
            isinstance(name, ast.Constant)
            and isinstance(name.value, str)
            and isinstance(matrix, np.ndarray)
            and matrix.ndim == 2
            and matrix.shape[0] == matrix.shape[1]
            and 2 <= matrix.shape[0] <= _MAX_REGISTERED_DIMENSION
            and not matrix.shape[0] & (matrix.shape[0] - 1)
        ):
            registered[name.value] = np.asarray(matrix, dtype=complex)
    return registered


def _static_numeric_value(node: ast.AST, environment: dict[str, Any]) -> Any | None:
    """Evaluate a bounded numeric expression without executing candidate code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float | complex):
        return node.value
    if isinstance(node, ast.Name):
        return environment.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _static_numeric_value(node.operand, environment)
        return None if value is None else -value
    if isinstance(node, ast.List | ast.Tuple):
        items = [_static_numeric_value(element, environment) for element in node.elts]
        return None if any(item is None for item in items) else items
    if isinstance(node, ast.BinOp):
        return _static_numeric_binop(node, environment)
    if isinstance(node, ast.Call):
        return _static_numeric_call(node, environment)
    return None


def _static_numeric_binop(node: ast.BinOp, environment: dict[str, Any]) -> Any | None:
    left = _static_numeric_value(node.left, environment)
    right = _static_numeric_value(node.right, environment)
    if left is None or right is None or isinstance(left, list) or isinstance(right, list):
        return None
    try:
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.MatMult):
            return left @ right
        if isinstance(node.op, ast.Pow):
            return left**right
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _static_numeric_call(node: ast.Call, environment: dict[str, Any]) -> Any | None:
    name = _attr_or_name(node.func)
    if len(node.args) != 1:
        return None
    argument = _static_numeric_value(node.args[0], environment)
    if argument is None:
        return None
    try:
        if name in {"array", "asarray"}:
            value = np.asarray(argument, dtype=complex)
            return value if value.size <= _MAX_REGISTERED_DIMENSION**2 else None
        if name == "expm" and isinstance(argument, np.ndarray):
            from scipy.linalg import expm

            return expm(argument)
    except (TypeError, ValueError):
        return None
    return None


def _rotation_matrix(name: str, angle: float) -> np.ndarray:
    """Return the CUDA-Q intrinsic rotation matrix for ``name``."""
    if name == "rx":
        cosine = math.cos(angle / 2)
        sine = -1j * math.sin(angle / 2)
        return np.asarray(
            [[cosine, sine], [sine, cosine]],
            dtype=complex,
        )
    if name == "ry":
        cosine = math.cos(angle / 2)
        sine = math.sin(angle / 2)
        return np.asarray(
            [[cosine, -sine], [sine, cosine]],
            dtype=complex,
        )
    if name == "rz":
        return np.asarray(
            [
                [np.exp(-1j * angle / 2), 0.0],
                [0.0, np.exp(1j * angle / 2)],
            ],
            dtype=complex,
        )
    if name == "r1":
        return np.asarray(
            [[1.0, 0.0], [0.0, np.exp(1j * angle)]],
            dtype=complex,
        )
    raise ValueError(name)


_SQRT2_INV = 1 / math.sqrt(2)
_CUDAQ_MATRICES: dict[str, np.ndarray] = {
    "h": np.asarray(
        [
            [_SQRT2_INV, _SQRT2_INV],
            [_SQRT2_INV, -_SQRT2_INV],
        ],
        dtype=complex,
    ),
    "x": np.asarray([[0, 1], [1, 0]], dtype=complex),
    "y": np.asarray([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.asarray([[1, 0], [0, -1]], dtype=complex),
    "s": np.asarray([[1, 0], [0, 1j]], dtype=complex),
    "t": np.asarray(
        [[1, 0], [0, np.exp(1j * math.pi / 4)]],
        dtype=complex,
    ),
    "swap": _to_little_endian_matrix(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        ),
        2,
    ),
    # Two-qubit named gates in CUDA-Q call order (control, target); the
    # diagonal CZ is basis-order independent.
    "cz": _to_little_endian_matrix(
        np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex),
        2,
    ),
    "cx": _to_little_endian_matrix(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        ),
        2,
    ),
    "cy": _to_little_endian_matrix(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
            ],
            dtype=complex,
        ),
        2,
    ),
}
