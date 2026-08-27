"""Numeric metrics and decisive results for exact array engines."""

from __future__ import annotations

import math

import numpy as np

from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.exact.engines import EXACT_ENGINE_VERSION, ExactEngineSpec
from qceval.semantics.verifiers.result import (
    RESULT_SCHEMA_VERSION,
    EvidenceRecord,
    SemanticStatus,
    VerifierResult,
)

# Infidelities below this bound are floating-point rounding, not physics:
# sqrt() would otherwise amplify an O(eps) overlap deficit to ~1.5e-8, above
# the 1e-9 contract tolerance, spuriously failing machine-exact candidates.
_STATE_INFIDELITY_FLOOR = 1e-14


def _array_error(representation: str, actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    if not np.all(np.isfinite(actual)):
        return math.inf, math.inf
    if representation == "statevector" and actual.ndim == 1 and target.ndim == 2:
        return _state_set_error(actual, target)
    if actual.shape != target.shape:
        return math.inf, math.inf
    if representation == "statevector":
        return _state_error(actual, target)
    if representation in {"unitary", "isometry"}:
        return _operator_error(actual, target)
    if representation == "choi":
        # A trace-preserving channel's Choi trace equals its input dimension.
        # The pinned target is trusted, so its trace is that dimension for any
        # channel geometry; sqrt(dim) would assume a square channel.
        sanity = abs(complex(np.trace(actual) - np.trace(target)))
        scale = max(1.0, float(np.linalg.norm(target)))
        return float(np.linalg.norm(actual - target) / scale), sanity
    raise NotImplementedError(f"unknown exact representation {representation!r}")


def _state_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    actual_norm = float(np.linalg.norm(actual))
    target_norm = float(np.linalg.norm(target))
    if actual_norm == 0 or target_norm == 0:
        return math.inf, math.inf
    overlap = abs(np.vdot(target / target_norm, actual / actual_norm))
    infidelity = max(0.0, 1.0 - min(1.0, float(overlap) ** 2))
    error = 0.0 if infidelity < _STATE_INFIDELITY_FLOOR else math.sqrt(infidelity)
    return error, abs(actual_norm - 1.0)


def _state_set_error(actual: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Return the minimum state error over an enumerated valid-state set.

    Membership targets replace ball-around-exact approximation budgets: the
    candidate must be exactly one of the enumerated valid states (for example
    every symmetric Trotter factor ordering) instead of merely close to the
    ideal evolution, which a wrong Hamiltonian could also satisfy.
    """
    if targets.size == 0 or targets.shape[1] != actual.shape[0]:
        return math.inf, math.inf
    best_error = math.inf
    best_sanity = math.inf
    for row in targets:
        error, sanity = _state_error(actual, row)
        if error < best_error:
            best_error, best_sanity = error, sanity
    return best_error, best_sanity


def _operator_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    overlap = np.vdot(target, actual)
    phase = 1.0 + 0j if abs(overlap) <= np.finfo(float).tiny else overlap / abs(overlap)
    residual = actual * np.conjugate(phase) - target
    gram = actual.conjugate().T @ actual
    sanity = float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord=2))
    return float(np.linalg.norm(residual, ord=2)), sanity


def _numeric_result(
    context: VerificationContext,
    spec: ExactEngineSpec,
    error: float,
    sanity: float,
    cases: int,
    elapsed: float,
) -> VerifierResult:
    tolerance = context.contract.approximation.tolerance
    uncertainty = context.contract.approximation.uncertainty
    if not math.isfinite(error) or not math.isfinite(sanity):
        status, reason = SemanticStatus.SEMANTIC_FAIL, "malformed_semantic_object"
    elif sanity > max(tolerance, uncertainty):
        status, reason = SemanticStatus.SEMANTIC_FAIL, "semantic_sanity_check_failed"
    elif error <= max(0.0, tolerance - uncertainty):
        status, reason = SemanticStatus.VERIFIED_PASS, "metric_within_pass_bound"
    elif error > tolerance + uncertainty:
        status, reason = SemanticStatus.SEMANTIC_FAIL, "metric_exceeds_fail_bound"
    else:
        status, reason = SemanticStatus.EXECUTION_ERROR, "metric_in_uncertainty_band"
    evidence = EvidenceRecord(
        spec.name,
        EXACT_ENGINE_VERSION,
        reason,
        context.input_hash,
        context.target_hash,
        metric=spec.metric,
        value=error if math.isfinite(error) else None,
        tolerance=tolerance,
        uncertainty=uncertainty,
        cases_checked=cases,
        elapsed_seconds=elapsed,
        preconditions=(f"sanity_residual={sanity:.17g}",),
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        reason,
        context.contract_hash,
        context.target_hash,
        EXACT_ENGINE_VERSION,
        (evidence,),
    )


def _candidate_semantic_failure(
    context: VerificationContext,
    spec: ExactEngineSpec,
    reason: str,
    elapsed: float,
) -> VerifierResult:
    evidence = EvidenceRecord(
        spec.name,
        EXACT_ENGINE_VERSION,
        reason,
        context.input_hash,
        context.target_hash,
        metric=spec.metric,
        elapsed_seconds=elapsed,
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        SemanticStatus.SEMANTIC_FAIL,
        reason,
        context.contract_hash,
        context.target_hash,
        EXACT_ENGINE_VERSION,
        (evidence,),
    )
