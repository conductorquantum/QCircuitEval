"""Portable AST-to-matrix lowering for non-Qiskit symbolic proofs."""

from __future__ import annotations

import ast

import sympy as sp

from qceval.frameworks.qiskit.symbolic.proof import (
    _PARAMETERS,
    _Budget,
    _Inconclusive,
    _Refuted,
)
from qceval.frameworks.qiskit.symbolic.validation import _call_name
from qceval.semantics.verifiers.symbolic_literals import LiteralKind, certify_float


def _portable_matrix(
    code: str,
    entry_point: str,
    symbols: dict[str, sp.Symbol],
    budget: _Budget,
) -> tuple[sp.Matrix, tuple[str, ...], float, tuple[tuple[str, sp.Expr | None], ...]]:
    """Lower a portable RZ/SX family source into a symbolic matrix.

    Args:
        code: Candidate Python source.
        entry_point: Required top-level function name.
        symbols: Canonical parameter symbols.
        budget: Expression-node budget tracker.

    Returns:
        Matrix, observed gate names, and certified literal error.
    """
    function = _portable_function(code, entry_point)
    assignments = _portable_assignments(function)
    _validate_parameter_assignments(assignments)
    matrix = sp.eye(2)
    gates: list[str] = []
    steps: list[tuple[str, sp.Expr | None]] = []
    literal_error = 0.0
    # A gate constructor stored in a variable is a definition, not an
    # application: skip it here and count each application site instead, so
    # the matrix reflects exactly the applied gate sequence.
    definitions = {value for value in assignments.values() if _is_gate_constructor(value)}
    nested = {node.name for node in ast.walk(function) if isinstance(node, ast.FunctionDef) and node is not function}
    plumbing = nested | {
        "Circuit",
        "LineQubit",
        "append",
        "on",
        "range",
        "device",
        "qnode",
        "probs",
        "state",
        "sample",
        "qvector",
        "qubit",
        "kernel",
        "float",
        entry_point,
        "circuit",
    }
    quantum_forbidden = {
        "RX",
        "RY",
        "U",
        "U3",
        "X",
        "Y",
        "Z",
        "rx",
        "ry",
        "u",
        "u3",
        "x",
        "y",
        "z",
    }
    for call in sorted(
        (node for node in ast.walk(function) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    ):
        if call in definitions:
            continue
        raw = _call_name(call.func) or ""
        step = _portable_rotation_step(call, raw, assignments, symbols, budget)
        if step is None:
            step = _portable_sx_step(call, raw, assignments)
        if step is None:
            _validate_portable_plumbing(call, raw, plumbing, quantum_forbidden)
            continue
        operation, gate, error, angle = step
        gates.append(gate)
        steps.append((gate, angle))
        literal_error += error
        matrix = operation * matrix
        budget.check(matrix)
    if not {"rz", "sx"}.issubset(gates):
        raise _Refuted("symbolic_required_gate_family_missing")
    return matrix, tuple(gates), literal_error, tuple(steps)


def _portable_function(code: str, entry_point: str) -> ast.FunctionDef:
    if len(code.encode("utf-8")) > 100_000:
        raise _Inconclusive("symbolic_source_size_limit")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise _Inconclusive("symbolic_source_syntax_error") from exc
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_point),
        None,
    )
    if function is None:
        raise _Inconclusive("symbolic_entry_point_missing")
    forbidden = (ast.For, ast.While, ast.Try, ast.With, ast.Match, ast.Lambda, ast.comprehension)
    invalid_control = any(isinstance(node, forbidden) for node in ast.walk(function)) or any(
        isinstance(node, ast.If) and not _is_none_default_if(node) for node in ast.walk(function)
    )
    if invalid_control:
        raise _Inconclusive("symbolic_control_unsupported")
    return function


def _portable_rotation_step(
    call: ast.Call,
    raw: str,
    assignments: dict[str, ast.expr],
    symbols: dict[str, sp.Symbol],
    budget: _Budget,
) -> tuple[sp.Matrix, str, float, sp.Expr | None] | None:
    normalized = raw.lower()
    constructor: ast.Call | None = call if normalized in {"rz", "rx"} else None
    if constructor is None:
        resolved = _applied_gate_constructor(call, assignments)
        if resolved is None:
            return None
        constructor = resolved
        normalized = (_call_name(constructor.func) or "").lower()
        if normalized not in {"rz", "rx"}:
            return None
    if not constructor.args:
        raise _Inconclusive(f"symbolic_{normalized}_arity")
    angle, error = _portable_angle(constructor.args[0], assignments, symbols, budget)
    if normalized == "rz":
        return sp.diag(sp.exp(-sp.I * angle / 2), sp.exp(sp.I * angle / 2)), "rz", error, angle
    if sp.simplify(angle - sp.pi / 2) != 0:
        raise _Refuted("symbolic_forbidden_gate_family:rx")
    return _sx_matrix(), "sx", error, None


def _applied_gate_constructor(call: ast.Call, assignments: dict[str, ast.expr]) -> ast.Call | None:
    """Resolve a stored-gate application site to its constructor call.

    Recognizes ``gate.on(q)`` and direct ``gate(q)`` applications where
    ``gate`` is a variable bound to a gate-constructor call.
    """
    receiver: ast.expr | None = None
    if isinstance(call.func, ast.Attribute) and call.func.attr == "on":
        receiver = call.func.value
    elif isinstance(call.func, ast.Name):
        receiver = call.func
    if receiver is None:
        return None
    resolved = _resolved_expression(receiver, assignments)
    return resolved if isinstance(resolved, ast.Call) and _is_gate_constructor(resolved) else None


def _resolved_expression(node: ast.expr, assignments: dict[str, ast.expr]) -> ast.expr:
    seen: set[str] = set()
    while isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        seen.add(node.id)
        node = assignments[node.id]
    return node


def _is_gate_constructor(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and (_call_name(node.func) or "") in {"rz", "rx", "XPowGate", "SX"}


def _portable_sx_step(
    call: ast.Call,
    raw: str,
    assignments: dict[str, ast.expr],
) -> tuple[sp.Matrix, str, float, sp.Expr | None] | None:
    if raw in {"SX", "sx"}:
        return _sx_matrix(), "sx", 0.0, None
    if raw == "XPowGate":
        return _xpow_step(call)
    stored = _applied_gate_constructor(call, assignments)
    if stored is not None and (_call_name(stored.func) or "") == "XPowGate":
        return _xpow_step(stored)
    if stored is not None and (_call_name(stored.func) or "") == "SX":
        return _sx_matrix(), "sx", 0.0, None
    assigned = raw in assignments and _is_sx_power(assignments[raw], assignments)
    inline = not raw and _is_sx_power(call.func, assignments)
    # cirq's idiomatic application is (cirq.X ** 0.5).on(q); the receiver of
    # the .on call carries the gate, which the plain plumbing rule would skip.
    on_receiver = (
        isinstance(call.func, ast.Attribute) and call.func.attr == "on" and _is_sx_power(call.func.value, assignments)
    )
    return (_sx_matrix(), "sx", 0.0, None) if assigned or inline or on_receiver else None


def _xpow_step(call: ast.Call) -> tuple[sp.Matrix, str, float, sp.Expr | None]:
    exponent = next((item.value for item in call.keywords if item.arg == "exponent"), None)
    exponent = exponent if exponent is not None else (call.args[0] if call.args else None)
    if exponent is None or not _is_exact_half(exponent):
        raise _Refuted("symbolic_forbidden_gate_family:xpow")
    return _sx_matrix(), "sx", 0.0, None


def _validate_portable_plumbing(
    call: ast.Call,
    raw: str,
    plumbing: set[str],
    quantum_forbidden: set[str],
) -> None:
    nested_factory = (
        not raw and isinstance(call.func, ast.Call) and (_call_name(call.func.func) or "") in {"rz", "rx", "XPowGate"}
    )
    if nested_factory or raw in plumbing:
        return
    if raw in quantum_forbidden:
        raise _Refuted(f"symbolic_forbidden_gate_family:{raw.lower()}")
    raise _Inconclusive(f"symbolic_call_unsupported:{raw or type(call.func).__name__}")


def _sx_matrix() -> sp.Matrix:
    return sp.Matrix([[1 + sp.I, 1 - sp.I], [1 - sp.I, 1 + sp.I]]) / 2


def _portable_assignments(function: ast.FunctionDef) -> dict[str, ast.expr]:
    values: dict[str, ast.expr] = {}
    parents = {child: parent for parent in ast.walk(function) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.If) and _is_none_default_if(parent):
            continue
        values[node.targets[0].id] = node.value
    return values


def _is_none_default_if(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _is_exact_half(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int | float) and node.value == 0.5


def _is_sx_power(node: ast.expr, assignments: dict[str, ast.expr]) -> bool:
    seen: set[str] = set()
    while isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        seen.add(node.id)
        node = assignments[node.id]
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and (
            isinstance(node.left, ast.Name)
            and node.left.id == "X"
            or isinstance(node.left, ast.Attribute)
            and node.left.attr == "X"
        )
        and _is_exact_half(node.right)
    )


def _validate_parameter_assignments(assignments: dict[str, ast.expr]) -> None:
    for name in _PARAMETERS:
        value = assignments.get(name)
        if value is None:
            continue
        if not (
            isinstance(value, ast.IfExp)
            and isinstance(value.orelse, ast.Name)
            and value.orelse.id == name
            and isinstance(value.test, ast.Compare)
            and isinstance(value.test.left, ast.Name)
            and value.test.left.id == name
            and len(value.test.ops) == 1
            and isinstance(value.test.ops[0], ast.Is)
            and len(value.test.comparators) == 1
            and isinstance(value.test.comparators[0], ast.Constant)
            and value.test.comparators[0].value is None
        ):
            raise _Refuted(f"symbolic_parameter_reassigned:{name}")


def _portable_angle(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    symbols: dict[str, sp.Symbol],
    budget: _Budget,
) -> tuple[sp.Expr, float]:
    expression, error = _portable_expression(node, assignments, symbols, set())
    budget.check(expression)
    return sp.expand(expression), error


def _portable_expression(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    symbols: dict[str, sp.Symbol],
    seen: set[str],
) -> tuple[sp.Expr, float]:
    if isinstance(node, ast.Name):
        return _portable_name(node, assignments, symbols, seen)
    if isinstance(node, ast.Attribute) and node.attr == "pi":
        return sp.pi, 0.0
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return _portable_constant(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value, error = _portable_expression(node.operand, assignments, symbols, seen)
        return (-value if isinstance(node.op, ast.USub) else value), error
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        return _portable_binary(node, assignments, symbols, seen)
    if isinstance(node, ast.Call) and _call_name(node.func) == "float" and len(node.args) == 1:
        return _portable_expression(node.args[0], assignments, symbols, seen)
    if isinstance(node, ast.IfExp):
        branch = _non_none_guard_branch(node)
        if branch is not None:
            return _portable_expression(branch, assignments, symbols, seen)
    raise _Inconclusive(f"symbolic_expression_unsupported:{type(node).__name__}")


def _non_none_guard_branch(node: ast.IfExp) -> ast.expr | None:
    """Return the branch selected for a symbolic non-None parameter."""
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id in _PARAMETERS
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is | ast.IsNot)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    ):
        return None
    return node.orelse if isinstance(test.ops[0], ast.Is) else node.body


def _portable_name(
    node: ast.Name,
    assignments: dict[str, ast.expr],
    symbols: dict[str, sp.Symbol],
    seen: set[str],
) -> tuple[sp.Expr, float]:
    if node.id in symbols:
        return symbols[node.id], 0.0
    if node.id == "pi":
        return sp.pi, 0.0
    if node.id not in assignments or node.id in seen:
        raise _Inconclusive(f"symbolic_parameter_unknown:{node.id}")
    value = assignments[node.id]
    if isinstance(value, ast.Call) and _call_name(value.func) == "float" and value.args:
        value = value.args[0]
    return _portable_expression(value, assignments, symbols, {*seen, node.id})


def _portable_constant(raw: int | float) -> tuple[sp.Expr, float]:
    if isinstance(raw, int):
        return sp.Integer(raw), 0.0
    certification = certify_float(float(raw))
    if certification.kind is LiteralKind.UNMATCHED:
        raise _Inconclusive("symbolic_unmatched_numeric_literal")
    value = sp.Rational(certification.numerator, certification.denominator)
    if certification.kind is LiteralKind.PI_MULTIPLE:
        value *= sp.pi
    return value, certification.absolute_error or 0.0


def _portable_binary(
    node: ast.BinOp,
    assignments: dict[str, ast.expr],
    symbols: dict[str, sp.Symbol],
    seen: set[str],
) -> tuple[sp.Expr, float]:
    left, left_error = _portable_expression(node.left, assignments, symbols, seen)
    right, right_error = _portable_expression(node.right, assignments, symbols, seen)
    operations = {
        ast.Add: lambda: left + right,
        ast.Sub: lambda: left - right,
        ast.Mult: lambda: left * right,
        ast.Div: lambda: left / right,
    }
    value = operations[type(node.op)]()
    return value, left_error + right_error
