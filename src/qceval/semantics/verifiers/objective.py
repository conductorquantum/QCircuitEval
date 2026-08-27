"""Objective-value verification separated from circuit-family validity."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from qceval.semantics.targets import load_contract_target_document
from qceval.semantics.verifiers.base import CostEstimate, EngineDescriptor, VerificationContext
from qceval.semantics.verifiers.result import (
    RESULT_SCHEMA_VERSION,
    EvidenceRecord,
    SemanticStatus,
    VerifierResult,
)

OBJECTIVE_ENGINE_VERSION = "1.0.0"


class ObjectiveDirection(StrEnum):
    """Optimization direction used to define a nonnegative gap."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class ObjectiveTarget:
    """Independent optimum/bound, success gap, and search policy."""

    optimum: float
    max_gap: float
    direction: ObjectiveDirection
    optimization_required: bool
    max_evaluations: int | None


@dataclass(frozen=True)
class ObjectiveObservation:
    """Candidate family checks and certified objective observation."""

    api_valid: bool
    family_valid: bool
    value: float
    numerical_error: float
    evaluations: int


class ObjectiveMaterializer(Protocol):
    """Evaluate candidate API, family, and objective value."""

    def objective(self, context: VerificationContext) -> ObjectiveObservation:
        """Materialize a candidate objective observation.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Candidate family checks and objective estimate.
        """
        ...


class ObjectiveTargetProvider(Protocol):
    """Load an independently specified objective target."""

    def objective_target(self, context: VerificationContext) -> ObjectiveTarget:
        """Load one objective target.

        Args:
            context: Contract identifying the target.

        Returns:
            Independent objective bound and policy.
        """
        ...


class ObjectiveTargetUnavailable(ValueError):
    """Stable signal that an objective target is unresolved."""


class PackagedObjectiveTargetProvider:
    """Load packaged objective targets without filling specification gaps."""

    def objective_target(self, context: VerificationContext) -> ObjectiveTarget:
        """Load a hash-verified resolved objective target.

        Args:
            context: Contract identifying the target artifact.

        Returns:
            Resolved target.

        Raises:
            ObjectiveTargetUnavailable: If the target is intentionally blocked.
            ValueError: If a purported resolved target is malformed.
        """
        target = load_contract_target_document(context.contract).get("target")
        if not isinstance(target, dict):
            raise ValueError("objective target must be an object")
        if target.get("type") == "unresolved_vqe_objective":
            raise ObjectiveTargetUnavailable("objective_target_unresolved")
        required = {"optimum", "max_gap", "direction", "optimization_required", "max_evaluations"}
        if set(target) != required:
            raise ValueError("resolved objective target fields differ")
        return ObjectiveTarget(
            float(target["optimum"]),
            float(target["max_gap"]),
            ObjectiveDirection(target["direction"]),
            bool(target["optimization_required"]),
            None if target["max_evaluations"] is None else int(target["max_evaluations"]),
        )


class ObjectiveEngine:
    """Verify family validity, objective gap, and required search budget."""

    def __init__(self, materializer: ObjectiveMaterializer, targets: ObjectiveTargetProvider) -> None:
        """Initialize an objective engine.

        Args:
            materializer: Candidate objective materializer.
            targets: Independent objective target provider.
        """
        self._materializer = materializer
        self._targets = targets

    def descriptor(self) -> EngineDescriptor:
        """Return immutable capability metadata.

        Returns:
            Objective engine identity and capabilities.
        """
        return EngineDescriptor(
            "objective_exact",
            OBJECTIVE_ENGINE_VERSION,
            ("objective",),
            ("objective", "family_validation", "optimization_budget", "framework_neutral_ir"),
        )

    def estimate(self, context: VerificationContext) -> CostEstimate:
        """Estimate contract-bounded objective evaluation.

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
        """Evaluate objective semantics without conflating optimizer labels.

        Args:
            context: Verification context.

        Returns:
            Decisive or operational objective result.
        """
        started = time.perf_counter()
        try:
            target = self._targets.objective_target(context)
        except ObjectiveTargetUnavailable as exc:
            return _result(
                context,
                SemanticStatus.EXECUTION_ERROR,
                str(exc),
                None,
                0,
                time.perf_counter() - started,
            )
        observation = self._materializer.objective(context)
        if not observation.api_valid:
            return _result(
                context, SemanticStatus.SEMANTIC_FAIL, "objective_api_invalid", None, 0, time.perf_counter() - started
            )
        if not observation.family_valid:
            return _result(
                context,
                SemanticStatus.SEMANTIC_FAIL,
                "objective_family_invalid",
                None,
                observation.evaluations,
                time.perf_counter() - started,
            )
        if not _valid(target, observation):
            return _result(
                context,
                SemanticStatus.EXECUTION_ERROR,
                "objective_certificate_malformed",
                None,
                max(0, observation.evaluations),
                time.perf_counter() - started,
            )
        if target.optimization_required and (
            target.max_evaluations is None or observation.evaluations > target.max_evaluations
        ):
            return _result(
                context,
                SemanticStatus.SEMANTIC_FAIL,
                "objective_optimization_budget_exceeded",
                None,
                observation.evaluations,
                time.perf_counter() - started,
            )
        gap = _gap(target, observation.value)
        lower = max(0.0, gap - observation.numerical_error)
        upper = gap + observation.numerical_error
        allowed = target.max_gap + context.contract.approximation.tolerance
        uncertainty = context.contract.approximation.uncertainty
        if upper <= max(0.0, allowed - uncertainty):
            status, reason = SemanticStatus.VERIFIED_PASS, "objective_gap_within_bound"
        elif lower > allowed + uncertainty:
            status, reason = SemanticStatus.SEMANTIC_FAIL, "objective_gap_exceeds_bound"
        else:
            status, reason = SemanticStatus.EXECUTION_ERROR, "objective_gap_in_uncertainty_band"
        return _result(context, status, reason, gap, observation.evaluations, time.perf_counter() - started)


def _gap(target: ObjectiveTarget, value: float) -> float:
    raw = value - target.optimum if target.direction is ObjectiveDirection.MINIMIZE else target.optimum - value
    return max(0.0, raw)


def _valid(target: ObjectiveTarget, observation: ObjectiveObservation) -> bool:
    values = (target.optimum, target.max_gap, observation.value, observation.numerical_error)
    return (
        all(math.isfinite(value) for value in values)
        and target.max_gap >= 0
        and observation.numerical_error >= 0
        and observation.evaluations >= 0
        and (target.max_evaluations is None or target.max_evaluations >= 0)
    )


def _result(
    context: VerificationContext,
    status: SemanticStatus,
    reason: str,
    gap: float | None,
    evaluations: int,
    elapsed: float,
) -> VerifierResult:
    evidence = EvidenceRecord(
        "objective_exact",
        OBJECTIVE_ENGINE_VERSION,
        reason,
        context.input_hash,
        context.target_hash,
        metric="objective_gap" if gap is not None else None,
        value=gap,
        tolerance=context.contract.approximation.tolerance if gap is not None else None,
        uncertainty=context.contract.approximation.uncertainty if gap is not None else None,
        cases_checked=max(0, evaluations),
        elapsed_seconds=elapsed,
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        reason,
        context.contract_hash,
        context.target_hash,
        OBJECTIVE_ENGINE_VERSION,
        (evidence,),
    )
