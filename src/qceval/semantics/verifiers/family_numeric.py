"""Numeric fallback verification for structured parameter-family contracts.

The structured-rotation and bounded-symbolic source engines prove parameter
families syntactically. Their grammar is deliberately narrow, so behaviorally
exact candidates written in an unanticipated spelling come out inconclusive.
This module verifies such candidates numerically: the lowered Program IR is
compared against the contract's independently derived target family at every
executed diagnostic and probe point, within the contract tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qceval.semantics.contracts import Contract, contract_hash
from qceval.semantics.ir import Program, program_hash
from qceval.semantics.targets import load_contract_target_document
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.program_materializer import ProgramIRMaterializer
from qceval.semantics.verifiers.result import (
    SemanticStatus,
    VerifierResult,
    make_evidence,
    make_verifier_result,
)

FAMILY_NUMERIC_VERSION = "1.0.0"

FAMILY_NUMERIC_PASS = "family_numeric_identity"
FAMILY_NUMERIC_MISMATCH = "family_numeric_mismatch"

_ANALYTIC_MATRIX_FORMAT = "analytic_parameterized_matrix_v1"


def verify_analytic_family_unitary(
    contract: Contract,
    program: Program,
    arguments: tuple[Any, ...],
) -> VerifierResult | None:
    """Compare a lowered candidate against an analytic parameterized matrix.

    Args:
        contract: Behavior contract whose target is an analytic matrix family.
        program: Lowered framework-neutral candidate program.
        arguments: Concrete real parameter values bound for this execution.

    Returns:
        Decisive numeric result, or ``None`` when the target family is not an
        analytic parameterized matrix or the candidate cannot be materialized.
    """
    target = _analytic_family_unitary(contract, arguments)
    if target is None:
        return None
    context = VerificationContext(
        contract=contract,
        contract_hash=contract_hash(contract),
        target_hash=contract.target.sha256,
        input_hash=program_hash(program),
        program=program,
        arguments=arguments,
    )
    try:
        actual = ProgramIRMaterializer().array(context, "unitary").value
    except Exception:  # noqa: BLE001 - unsupported IR falls back to the source verdict.
        return None
    if actual.shape != target.shape:
        return _numeric_result(context, SemanticStatus.SEMANTIC_FAIL, FAMILY_NUMERIC_MISMATCH, 1.0)
    error = _global_phase_operator_distance(actual, target)
    tolerance = contract.approximation.tolerance
    if error <= max(tolerance, 1e-9):
        return _numeric_result(context, SemanticStatus.VERIFIED_PASS, FAMILY_NUMERIC_PASS, error)
    return _numeric_result(context, SemanticStatus.SEMANTIC_FAIL, FAMILY_NUMERIC_MISMATCH, error)


def _numeric_result(
    context: VerificationContext,
    status: SemanticStatus,
    reason: str,
    error: float,
) -> VerifierResult:
    evidence = make_evidence(
        "family_numeric_fallback",
        FAMILY_NUMERIC_VERSION,
        reason,
        input_hash=context.input_hash,
        target_hash=context.target_hash,
        metric="operator_norm",
        value=float(min(max(error, 0.0), 2.0)),
        tolerance=context.contract.approximation.tolerance,
        uncertainty=context.contract.approximation.uncertainty,
        cases_checked=1,
        preconditions=(f"arguments={tuple(context.arguments)!r}",),
    )
    return make_verifier_result(
        status,
        reason,
        contract_hash=context.contract_hash,
        target_hash=context.target_hash,
        verifier_version=FAMILY_NUMERIC_VERSION,
        evidence=(evidence,),
    )


def _analytic_family_unitary(contract: Contract, arguments: tuple[Any, ...]) -> np.ndarray | None:
    """Evaluate the packaged analytic matrix family at concrete arguments."""
    try:
        document = load_contract_target_document(contract)
    except Exception:  # noqa: BLE001 - a missing artifact keeps the source verdict.
        return None
    if not isinstance(document, dict) or document.get("format") != _ANALYTIC_MATRIX_FORMAT:
        return None
    matrix = document.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        return None
    names = [argument.name for argument in contract.signature.arguments]
    if len(names) != len(arguments):
        return None
    try:
        values = {name: float(value) for name, value in zip(names, arguments, strict=True)}
    except (TypeError, ValueError):
        return None
    return _evaluate_symbolic_matrix(matrix, values)


def _evaluate_symbolic_matrix(matrix: list[Any], values: dict[str, float]) -> np.ndarray | None:
    import sympy as sp

    symbols = {name: sp.Symbol(name, real=True) for name in values}
    local_names: dict[str, Any] = {"i": sp.I, "pi": sp.pi, **symbols}
    rows: list[list[complex]] = []
    for raw_row in matrix:
        if not isinstance(raw_row, list):
            return None
        row: list[complex] = []
        for raw_entry in raw_row:
            try:
                expression = sp.sympify(str(raw_entry), locals=local_names)
                bound = expression.subs({symbols[name]: values[name] for name in values})
                row.append(complex(bound.evalf(30)))
            except Exception:  # noqa: BLE001 - unparsable targets keep the source verdict.
                return None
        rows.append(row)
    return np.asarray(rows, dtype=np.complex128)


def _global_phase_operator_distance(actual: np.ndarray, target: np.ndarray) -> float:
    """Return the operator-norm distance minimized over a global phase."""
    overlap = complex(np.trace(target.conjugate().T @ actual))
    phase = 1.0 if abs(overlap) < 1e-14 else overlap / abs(overlap)
    return float(np.linalg.norm(actual - phase * target, ord=2))
