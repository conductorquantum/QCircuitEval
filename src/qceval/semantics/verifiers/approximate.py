"""Certified numerical and algorithmic approximation verification."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

from qceval.semantics.contracts import BehaviorKind
from qceval.semantics.verifiers.base import CostEstimate, EngineDescriptor, VerificationContext
from qceval.semantics.verifiers.result import (
    RESULT_SCHEMA_VERSION,
    EvidenceRecord,
    SemanticStatus,
    VerifierResult,
)

APPROXIMATION_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class CertifiedMetric:
    """One metric estimate with composable nonnegative error bounds."""

    metric: str
    value: float
    numerical_error: float
    algorithmic_error: float
    cases_checked: int


class CertifiedMetricProvider(Protocol):
    """Materialize a contract-selected metric with a certificate."""

    def metric(self, context: VerificationContext) -> CertifiedMetric:
        """Compute a certified metric estimate.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Central estimate and component error bounds.
        """
        ...


class CertifiedApproximationEngine:
    """Decide approximate contracts only from certified metric intervals."""

    def __init__(self, provider: CertifiedMetricProvider) -> None:
        """Initialize a certified approximation engine.

        Args:
            provider: Contract-specific certified metric implementation.
        """
        self._provider = provider

    def descriptor(self) -> EngineDescriptor:
        """Return immutable capability metadata.

        Returns:
            Approximation engine identity and capabilities.
        """
        return EngineDescriptor(
            "approximation_certified",
            APPROXIMATION_ENGINE_VERSION,
            tuple(kind.value for kind in BehaviorKind),
            ("certified_bound", "error_composition", "framework_neutral_ir"),
        )

    def estimate(self, context: VerificationContext) -> CostEstimate:
        """Estimate the contract-bounded metric computation.

        Args:
            context: Verification context.

        Returns:
            Conservative contract-limited cost.
        """
        limits = context.contract.limits
        return CostEstimate(
            context.program.num_qubits,
            limits.max_dimension,
            limits.max_cases,
            limits.memory_mib,
            limits.wall_seconds,
        )

    def verify(self, context: VerificationContext) -> VerifierResult:
        """Compare a certified metric interval with the contract bound.

        Args:
            context: Verification context.

        Returns:
            Decisive or execution-error result.
        """
        started = time.perf_counter()
        estimate = self._provider.metric(context)
        if estimate.metric != context.contract.approximation.metric:
            return _result(
                context,
                estimate,
                SemanticStatus.EXECUTION_ERROR,
                "approximation_metric_unsupported",
                time.perf_counter() - started,
            )
        if not _valid(estimate):
            return _result(
                context,
                estimate,
                SemanticStatus.EXECUTION_ERROR,
                "approximation_certificate_malformed",
                time.perf_counter() - started,
            )
        certificate = estimate.numerical_error + estimate.algorithmic_error
        if certificate > context.contract.approximation.error_budget:
            return _result(
                context,
                estimate,
                SemanticStatus.EXECUTION_ERROR,
                "approximation_error_budget_unproven",
                time.perf_counter() - started,
            )
        lower = max(0.0, estimate.value - certificate)
        upper = estimate.value + certificate
        tolerance = context.contract.approximation.tolerance
        uncertainty = context.contract.approximation.uncertainty
        if upper <= max(0.0, tolerance - uncertainty):
            status, reason = SemanticStatus.VERIFIED_PASS, "approximation_upper_bound_passes"
        elif lower > tolerance + uncertainty:
            status, reason = SemanticStatus.SEMANTIC_FAIL, "approximation_lower_bound_fails"
        else:
            status, reason = SemanticStatus.EXECUTION_ERROR, "approximation_interval_overlaps_boundary"
        return _result(context, estimate, status, reason, time.perf_counter() - started)


def _valid(value: CertifiedMetric) -> bool:
    numbers = (value.value, value.numerical_error, value.algorithmic_error)
    return all(math.isfinite(item) and item >= 0 for item in numbers) and value.cases_checked >= 0


def _result(
    context: VerificationContext,
    estimate: CertifiedMetric,
    status: SemanticStatus,
    reason: str,
    elapsed: float,
) -> VerifierResult:
    certificate = estimate.numerical_error + estimate.algorithmic_error
    evidence_value = estimate.value if math.isfinite(estimate.value) and estimate.value >= 0 else None
    evidence = EvidenceRecord(
        "approximation_certified",
        APPROXIMATION_ENGINE_VERSION,
        reason,
        context.input_hash,
        context.target_hash,
        metric=estimate.metric,
        value=evidence_value,
        tolerance=context.contract.approximation.tolerance,
        uncertainty=context.contract.approximation.uncertainty,
        cases_checked=max(0, estimate.cases_checked),
        elapsed_seconds=elapsed,
        preconditions=(
            f"numerical_error={estimate.numerical_error:.17g}",
            f"algorithmic_error={estimate.algorithmic_error:.17g}",
            f"certified_lower={max(0.0, estimate.value - certificate):.17g}",
            f"certified_upper={estimate.value + certificate:.17g}",
        ),
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        reason,
        context.contract_hash,
        context.target_hash,
        APPROXIMATION_ENGINE_VERSION,
        (evidence,),
    )
