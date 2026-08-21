"""Symbolic projective proof orchestration for RZ/SX families."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import sympy as sp

from qceval.evals.sandbox import execute_code_with_args
from qceval.frameworks.qiskit.symbolic.validation import _validate_source
from qceval.semantics.verifiers.symbolic_literals import LiteralKind, certify_float

_PARAMETERS = ("theta", "phi", "lam")
_SymbolicStep = tuple[str, sp.Expr | None]


@dataclass
class _Budget:
    max_expression_nodes: int
    max_expanded_nodes: int
    peak_expression_nodes: int = 0

    def check(self, expression: sp.Basic) -> None:
        nodes = sum(1 for _ in sp.preorder_traversal(expression))
        self.peak_expression_nodes = max(self.peak_expression_nodes, nodes)
        if nodes > self.max_expression_nodes:
            raise _Inconclusive("symbolic_expression_node_limit")

    def check_expanded(self, expression: sp.Basic) -> None:
        nodes = sum(1 for _ in sp.preorder_traversal(expression))
        self.peak_expression_nodes = max(self.peak_expression_nodes, nodes)
        if nodes > self.max_expanded_nodes:
            raise _Inconclusive("symbolic_expansion_node_limit")


class _Inconclusive(RuntimeError):
    pass


class _Refuted(RuntimeError):
    pass


def _prove(payload: dict[str, Any]) -> dict[str, Any]:
    """Prove or refute one symbolic worker request payload.

    Args:
        payload: Worker request fields including code and budgets.

    Returns:
        Bounded JSON-serializable proof response.
    """
    code = str(payload["code"])
    entry_point = str(payload["entry_point"])
    tolerance = float(payload["tolerance"])
    budget = _Budget(int(payload["max_expression_nodes"]), int(payload["max_expanded_nodes"]))
    framework = str(payload.get("framework", "qiskit"))
    try:
        _install_resource_limits(int(payload["cpu_seconds"]))
    except (OSError, ValueError):
        return _answer("inconclusive", "symbolic_resource_limit_unavailable", budget)
    try:
        return _execute_proof(code, entry_point, framework, tolerance, budget)
    except _Inconclusive as exc:
        return _answer("inconclusive", str(exc), budget)
    except _Refuted as exc:
        return _answer("refuted", str(exc), budget)
    except Exception as exc:  # noqa: BLE001 - worker failures are bounded result data.
        return _answer("inconclusive", f"symbolic_worker_exception:{type(exc).__name__}", budget)


def _execute_proof(
    code: str,
    entry_point: str,
    framework: str,
    tolerance: float,
    budget: _Budget,
) -> dict[str, Any]:
    from qceval.frameworks.qiskit.symbolic.portable import _portable_matrix

    symbols = {name: sp.Symbol(name, real=True) for name in _PARAMETERS}
    if framework == "qiskit":
        matrix, gates, literal_error, steps = _qiskit_matrix(code, entry_point, symbols, budget)
    else:
        matrix, gates, literal_error, steps = _portable_matrix(code, entry_point, symbols, budget)
    if math.isinf(literal_error):
        # Uncertified literals are used exactly as written: they can support a
        # decisive numeric refutation but never a proof, so skip the symbolic
        # simplification entirely (it is slow and cannot change the outcome).
        if _projective_counterexample(matrix, symbols, tolerance):
            return _answer("refuted", "symbolic_projective_counterexample", budget, gates, 0.0)
        return _answer("inconclusive", "symbolic_unmatched_numeric_literal", budget, gates)
    certified_error = literal_error / 2
    if _standard_u_decomposition(steps, symbols) and certified_error <= tolerance:
        return _answer(
            "proved",
            "symbolic_projective_identity",
            budget,
            gates,
            certified_error,
            ("0", "0", "0"),
        )
    outcome, reason, residuals = _projective(matrix, symbols, tolerance, budget)
    if outcome == "proved" and certified_error > tolerance:
        outcome, reason = "inconclusive", "symbolic_literal_error_exceeds_tolerance"
    return _answer(outcome, reason, budget, gates, certified_error, residuals)


def _qiskit_matrix(
    code: str,
    entry_point: str,
    symbols: dict[str, sp.Symbol],
    budget: _Budget,
) -> tuple[sp.Matrix, tuple[str, ...], float, tuple[_SymbolicStep, ...]]:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    source_issue = _validate_source(code, entry_point)
    if source_issue is not None:
        raise _Inconclusive(source_issue)
    circuit = execute_code_with_args(code, entry_point, *(Parameter(name) for name in _PARAMETERS))
    if not isinstance(circuit, QuantumCircuit):
        raise _Inconclusive("symbolic_result_type_unsupported")
    if circuit.num_qubits != 1:
        raise _Inconclusive("symbolic_register_width_unsupported")
    return _matrix(circuit, symbols, budget)


def _matrix(
    circuit: Any,
    symbols: dict[str, sp.Symbol],
    budget: _Budget,
) -> tuple[sp.Matrix, tuple[str, ...], float, tuple[_SymbolicStep, ...]]:
    matrix = sp.eye(2)
    gates = []
    steps: list[_SymbolicStep] = []
    literal_error = 0.0
    for instruction in circuit.data:
        name = str(instruction.operation.name).lower()
        if name == "barrier":
            continue
        gates.append(name)
        if name == "rz":
            if len(instruction.operation.params) != 1:
                raise _Inconclusive("symbolic_rz_arity")
            angle, error = _angle(instruction.operation.params[0], symbols, budget)
            literal_error += error
            operation = sp.diag(sp.exp(-sp.I * angle / 2), sp.exp(sp.I * angle / 2))
            steps.append(("rz", angle))
        elif name == "sx":
            operation = sp.Matrix([[1 + sp.I, 1 - sp.I], [1 - sp.I, 1 + sp.I]]) / 2
            steps.append(("sx", None))
        elif name == "sxdg":
            raise _Refuted("symbolic_forbidden_gate_family:sxdg")
        else:
            raise _Inconclusive(f"symbolic_gate_unsupported:{name}")
        matrix = operation * matrix
        budget.check(matrix)
    if not {"rz", "sx"}.issubset(gates):
        return matrix, tuple(gates), literal_error, tuple(steps)
    return matrix, tuple(gates), literal_error, tuple(steps)


def _standard_u_decomposition(
    steps: tuple[_SymbolicStep, ...],
    symbols: dict[str, sp.Symbol],
) -> bool:
    """Recognize the exact five-gate U decomposition without heavy expansion."""
    if tuple(name for name, _ in steps) != ("rz", "sx", "rz", "sx", "rz"):
        return False
    angles = (steps[0][1], steps[2][1], steps[4][1])
    if any(angle is None for angle in angles):
        return False
    theta, phi, lam = (symbols[name] for name in _PARAMETERS)
    expected = (lam, theta + sp.pi, phi + sp.pi)
    return all(_equal_mod_two_pi(actual, target) for actual, target in zip(angles, expected, strict=True))


def _equal_mod_two_pi(actual: sp.Expr | None, expected: sp.Expr) -> bool:
    if actual is None:
        return False
    turns = sp.simplify(sp.expand(actual - expected) / (2 * sp.pi))
    return not turns.free_symbols and turns.is_integer is True


def _angle(value: Any, symbols: dict[str, sp.Symbol], budget: _Budget) -> tuple[sp.Expr, float]:
    raw = value.sympify() if hasattr(value, "sympify") else sp.sympify(value)
    replacements = {}
    for symbol in raw.free_symbols:
        name = str(symbol)
        if name not in symbols:
            raise _Inconclusive(f"symbolic_parameter_unknown:{name}")
        replacements[symbol] = symbols[name]
    expression = raw.xreplace(replacements)
    error = 0.0
    for number in tuple(expression.atoms(sp.Float)):
        certification = certify_float(float(number))
        if certification.kind is LiteralKind.UNMATCHED:
            # Uncertified literals cannot support a proof, but keeping them as
            # exact rationals still allows a decisive numeric refutation
            # (e.g. hardcoded angles); the infinite certified error blocks any
            # "proved" outcome from being reported.
            error = math.inf
            continue
        replacement = sp.Rational(certification.numerator, certification.denominator)
        if certification.kind is LiteralKind.PI_MULTIPLE:
            replacement *= sp.pi
        expression = expression.xreplace({number: replacement})
        error += certification.absolute_error or 0.0
    budget.check(expression)
    return sp.expand(expression), error


def _projective_residuals(matrix: sp.Matrix, symbols: dict[str, sp.Symbol]) -> tuple[sp.Expr, ...]:
    theta, phi, lam = (symbols[name] for name in _PARAMETERS)
    target = sp.Matrix(
        [
            [sp.cos(theta / 2), -sp.exp(sp.I * lam) * sp.sin(theta / 2)],
            [sp.exp(sp.I * phi) * sp.sin(theta / 2), sp.exp(sp.I * (phi + lam)) * sp.cos(theta / 2)],
        ]
    )
    relative = target.conjugate().T * matrix
    return (relative[0, 1], relative[1, 0], relative[0, 0] - relative[1, 1])


def _projective_counterexample(
    matrix: sp.Matrix,
    symbols: dict[str, sp.Symbol],
    tolerance: float,
) -> bool:
    return _numeric_counterexample(_projective_residuals(matrix, symbols), tolerance)


def _projective(
    matrix: sp.Matrix,
    symbols: dict[str, sp.Symbol],
    tolerance: float,
    budget: _Budget,
) -> tuple[str, str, tuple[str, ...]]:
    residuals = _projective_residuals(matrix, symbols)
    if _numeric_counterexample(residuals, tolerance):
        rendered = tuple(str(expression)[:300] for expression in residuals)
        return "refuted", "symbolic_projective_counterexample", rendered
    simplified = tuple(_simplify(expression, budget) for expression in residuals)
    rendered = tuple(str(expression)[:300] for expression in simplified)
    if all(expression == 0 for expression in simplified):
        return "proved", "symbolic_projective_identity", rendered
    return "inconclusive", "symbolic_identity_unresolved", rendered


def _simplify(expression: sp.Expr, budget: _Budget) -> sp.Expr:
    budget.check(expression)
    expanded = sp.expand_trig(expression)
    budget.check_expanded(expanded)
    expanded = sp.expand_complex(expanded)
    budget.check_expanded(expanded)
    value = sp.simplify(sp.trigsimp(expanded))
    budget.check(value)
    return value


def _numeric_counterexample(residuals: tuple[sp.Expr, ...], tolerance: float) -> bool:
    points = ((0.0, 0.0, 0.0), (0.37, -0.91, 1.23), (-1.1, 2.2, -0.4), (1.7, 0.2, -2.1))
    symbols = tuple(sp.Symbol(name, real=True) for name in _PARAMETERS)
    for point in points:
        substitutions = dict(zip(symbols, point, strict=True))
        for residual in residuals:
            try:
                value = complex(residual.evalf(30, subs=substitutions))
            except (TypeError, ValueError):
                continue
            if abs(value) > max(1e-8, tolerance * 8):
                return True
    return False


def _answer(
    outcome: str,
    reason: str,
    budget: _Budget,
    gates: tuple[str, ...] = (),
    certified_error: float | None = None,
    residuals: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "reason": reason,
        "certified_error_bound": certified_error,
        "gate_families": list(gates),
        "peak_expression_nodes": budget.peak_expression_nodes,
        "residuals": list(residuals),
    }


def _install_resource_limits(cpu_seconds: int) -> None:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    consumed_cpu = usage.ru_utime + usage.ru_stime
    cpu_limit = max(1, math.ceil(consumed_cpu + cpu_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
