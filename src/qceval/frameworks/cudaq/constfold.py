"""CUDA-Q AST constant-folding helpers."""

from __future__ import annotations

import ast
import math
import operator as operator_module
from collections.abc import Sequence
from typing import Any


def _const_int(node: ast.AST) -> int | None:
    return _const_int_expr(node)


def _const_int_expr(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _const_int_expr(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.BinOp):
        left = _const_int_expr(node.left)
        right = _const_int_expr(node.right)
        if left is None or right is None:
            return None
        return _apply_int_operator(node.op, left, right)
    return None


def _apply_int_operator(
    operator: ast.operator,
    left: int,
    right: int,
) -> int | None:
    if isinstance(operator, ast.Add):
        return left + right
    if isinstance(operator, ast.Sub):
        return left - right
    if isinstance(operator, ast.Mult):
        return left * right
    if isinstance(operator, ast.Pow):
        return left**right
    if isinstance(operator, ast.FloorDiv) and right != 0:
        return left // right
    return None


def _const_float(
    node: ast.AST,
    *,
    constants: dict[str, Any] | None = None,
) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.Name) and constants is not None and node.id in constants:
        value = constants[node.id]
        return float(value) if isinstance(value, int | float) else None
    if isinstance(node, ast.Call):
        return _const_call_float(node, constants=constants)
    if isinstance(node, ast.Subscript) and constants is not None:
        return _constant_subscript_float(node, constants)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _const_float(node.operand, constants=constants)
        return None if value is None else -value
    if isinstance(node, ast.IfExp):
        return _const_ifexp_float(node, constants=constants)
    if isinstance(node, ast.BinOp):
        left = _const_float(node.left, constants=constants)
        right = _const_float(node.right, constants=constants)
        if left is None or right is None:
            return None
        return _apply_float_operator(node.op, left, right)
    return _pi_constant(node)


def _const_ifexp_float(
    node: ast.IfExp,
    *,
    constants: dict[str, Any] | None = None,
) -> float | None:
    test = _const_bool(node.test, constants=constants)
    if test is None:
        return None
    return _const_float(node.body if test else node.orelse, constants=constants)


def _const_call_float(
    node: ast.Call,
    *,
    constants: dict[str, Any] | None = None,
) -> float | None:
    """Statically evaluate ``float(...)`` and whitelisted math calls."""
    name = _attr_or_name(node.func)
    if name in {"float", "float32", "float64"} and len(node.args) == 1:
        return _const_float(node.args[0], constants=constants)
    if name in _CONSTANT_MATH_CALLS and len(node.args) == 1:
        argument = _const_float(node.args[0], constants=constants)
        if argument is None:
            return None
        try:
            return float(_CONSTANT_MATH_CALLS[name](argument))
        except ValueError:
            return None
    return None


def _const_bool(
    node: ast.AST,
    *,
    constants: dict[str, Any] | None = None,
) -> bool | None:
    """Statically evaluate a comparison over bound numeric constants."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    operator = node.ops[0]
    comparator = node.comparators[0]
    none_result = _bound_none_comparison(node.left, comparator, operator, constants)
    if none_result is not None:
        return none_result
    left = _const_float(node.left, constants=constants)
    right = _const_float(comparator, constants=constants)
    if left is None or right is None:
        return None
    comparator_function = _FLOAT_COMPARISONS.get(type(operator))
    return None if comparator_function is None else comparator_function(left, right)


def _bound_none_comparison(
    left: ast.expr,
    right: ast.expr,
    operator: ast.cmpop,
    constants: dict[str, Any] | None,
) -> bool | None:
    if not (
        isinstance(operator, ast.Is | ast.IsNot)
        and isinstance(left, ast.Name)
        and isinstance(right, ast.Constant)
        and right.value is None
        and constants is not None
        and left.id in constants
    ):
        return None
    is_none = constants[left.id] is None
    return is_none if isinstance(operator, ast.Is) else not is_none


_CONSTANT_MATH_CALLS = {
    "sqrt": math.sqrt,
    "acos": math.acos,
    "arccos": math.acos,
    "asin": math.asin,
    "arcsin": math.asin,
    "atan": math.atan,
    "arctan": math.atan,
    "cos": math.cos,
    "sin": math.sin,
}

_FLOAT_COMPARISONS = {
    ast.Eq: operator_module.eq,
    ast.NotEq: operator_module.ne,
    ast.Lt: operator_module.lt,
    ast.LtE: operator_module.le,
    ast.Gt: operator_module.gt,
    ast.GtE: operator_module.ge,
}


def _pi_constant(node: ast.AST) -> float | None:
    if isinstance(node, ast.Attribute) and node.attr == "pi":
        return math.pi
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    return None


def _constant_subscript_float(
    node: ast.Subscript,
    constants: dict[str, Any],
) -> float | None:
    base = constants.get(_attr_or_name(node.value))
    index = _const_int_expr(node.slice)
    if not isinstance(base, Sequence) or index is None or not 0 <= index < len(base):
        return None
    value = base[index]
    return float(value) if isinstance(value, int | float) else None


def _apply_float_operator(
    operator: ast.operator,
    left: float,
    right: float,
) -> float | None:
    if isinstance(operator, ast.Add):
        return left + right
    if isinstance(operator, ast.Sub):
        return left - right
    if isinstance(operator, ast.Mult):
        return left * right
    if isinstance(operator, ast.Div) and right != 0:
        return left / right
    if isinstance(operator, ast.Pow):
        return left**right
    return None


def _const_constant(
    node: ast.AST,
    *,
    constants: dict[str, Any] | None = None,
) -> Any | None:
    if isinstance(node, ast.List | ast.Tuple):
        values = [_const_float(element, constants=constants) for element in node.elts]
        if all(value is not None for value in values):
            return tuple(float(value) for value in values if value is not None)
    return _const_float(node, constants=constants)


def _cudaq_constant_bindings(
    code: str,
    entry_point: str,
    kernel: ast.FunctionDef,
    call_args: Sequence[Any] = (),
    *,
    include_kernel_assignments: bool = True,
) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    tree = ast.parse(code)
    entry = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == entry_point),
        None,
    )
    if entry is not None:
        defaults_start = len(entry.args.args) - len(entry.args.defaults)
        for index, argument in enumerate(entry.args.args):
            if index >= defaults_start:
                value = _const_float(
                    entry.args.defaults[index - defaults_start],
                    constants=constants,
                )
                if value is not None:
                    constants[argument.arg] = value
        for argument, value in zip(entry.args.args, call_args, strict=False):
            bound = _bindable_argument(value)
            if bound is not None:
                constants[argument.arg] = bound
        if entry is not kernel:
            _collect_constant_assignments(entry.body, constants)
    if include_kernel_assignments:
        _collect_constant_assignments(kernel.body, constants)
    return constants


def _bindable_argument(value: Any) -> Any | None:
    """Return a numeric binding for one runtime entry-point argument."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        items = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int | float):
                return None
            items.append(float(item))
        return tuple(items)
    return None


def _collect_constant_assignments(
    statements: Sequence[ast.stmt],
    constants: dict[str, Any],
) -> None:
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            value = _const_constant(
                statement.value,
                constants=constants,
            )
            if value is not None:
                constants[target.id] = value
            continue
        if isinstance(target, ast.Tuple):
            names = [item for item in target.elts if isinstance(item, ast.Name)]
            if len(names) != len(target.elts):
                continue
            values = _const_sequence(statement.value, constants=constants)
            if values is not None and len(values) == len(names):
                for name, value in zip(names, values, strict=True):
                    constants[name.id] = value


def _const_sequence(
    node: ast.AST,
    *,
    constants: dict[str, Any] | None = None,
) -> tuple[float, ...] | None:
    """Statically evaluate a numeric sequence expression."""
    value = _const_constant(node, constants=constants)
    if isinstance(value, tuple):
        return value
    if (
        isinstance(node, ast.ListComp)
        and len(node.generators) == 1
        and not node.generators[0].ifs
        and isinstance(node.generators[0].target, ast.Name)
    ):
        generator = node.generators[0]
        target = generator.target
        assert isinstance(target, ast.Name)
        source = _const_constant(generator.iter, constants=constants)
        if not isinstance(source, tuple):
            source_name = constants.get(_attr_or_name(generator.iter)) if constants else None
            source = source_name if isinstance(source_name, tuple) else None
        if source is None:
            return None
        values = []
        for item in source:
            local = dict(constants or {})
            local[target.id] = item
            element = _const_float(node.elt, constants=local)
            if element is None:
                return None
            values.append(element)
        return tuple(values)
    return None


def _attr_or_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""
