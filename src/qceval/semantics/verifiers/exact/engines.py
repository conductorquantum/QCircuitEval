"""CPU exact state, operator, isometry, channel, and classical engines."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from qceval.semantics.verifiers.base import CostEstimate, EngineDescriptor, VerificationContext
from qceval.semantics.verifiers.materialize import Materializer, TargetProvider
from qceval.semantics.verifiers.result import VerifierResult

EXACT_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class ExactEngineSpec:
    """Configuration shared by one exact semantic engine."""

    name: str
    kind: str
    representation: str
    metric: str
    capabilities: tuple[str, ...]


class ExactArrayEngine:
    """Validate and compare one dense exact semantic representation."""

    def __init__(self, spec: ExactEngineSpec, materializer: Materializer, targets: TargetProvider) -> None:
        """Initialize an exact engine.

        Args:
            spec: Engine kind, metric, and capabilities.
            materializer: Candidate Program IR materializer.
            targets: Independent target provider.
        """
        self._spec = spec
        self._materializer = materializer
        self._targets = targets

    def descriptor(self) -> EngineDescriptor:
        """Return immutable capability metadata.

        Returns:
            Engine identity and supported capabilities.
        """
        return EngineDescriptor(self._spec.name, EXACT_ENGINE_VERSION, (self._spec.kind,), self._spec.capabilities)

    def estimate(self, context: VerificationContext) -> CostEstimate:
        """Estimate dense allocation from contracted dimension.

        Args:
            context: Verification context.

        Returns:
            Deterministic conservative cost.
        """
        dimension = context.contract.limits.max_dimension
        memory = max(1, math.ceil((dimension * dimension * 16 * 3) / (1024 * 1024)))
        return CostEstimate(context.program.num_qubits, dimension, 1, memory, min(10.0, dimension**3 / 1e8))

    def verify(self, context: VerificationContext) -> VerifierResult:
        """Materialize, sanity-check, and compare exact semantics.

        Args:
            context: Verification context.

        Returns:
            Decisive or numerical-gray-zone result.
        """
        from qceval.semantics.verifiers.exact.metrics import (
            _array_error,
            _candidate_semantic_failure,
            _numeric_result,
        )
        from qceval.semantics.verifiers.materialize import CandidateSemanticError

        started = time.perf_counter()
        try:
            actual = self._materializer.array(context, self._spec.representation)
        except CandidateSemanticError as exc:
            return _candidate_semantic_failure(context, self._spec, exc.reason, time.perf_counter() - started)
        target = self._targets.array(context, self._spec.representation)
        error, sanity = _array_error(self._spec.representation, actual.value, target.value)
        return _numeric_result(
            context,
            self._spec,
            error,
            sanity,
            max(actual.cases, target.cases),
            time.perf_counter() - started,
        )


class ClassicalIOEngine:
    """Exhaustively compare a finite deterministic classical relation."""

    def __init__(self, materializer: Materializer, targets: TargetProvider) -> None:
        """Initialize a classical-I/O engine."""
        self._materializer = materializer
        self._targets = targets

    def descriptor(self) -> EngineDescriptor:
        """Return immutable capability metadata.

        Returns:
            Engine identity and supported capabilities.
        """
        return EngineDescriptor(
            "classical_io_exhaustive",
            EXACT_ENGINE_VERSION,
            ("classical_io",),
            ("classical_io", "framework_neutral_ir"),
        )

    def estimate(self, context: VerificationContext) -> CostEstimate:
        """Estimate complete contracted case enumeration.

        Args:
            context: Verification context.

        Returns:
            Conservative enumeration cost.
        """
        return CostEstimate(context.program.num_qubits, 1, context.contract.limits.max_cases, 1, 0.1)

    def verify(self, context: VerificationContext) -> VerifierResult:
        """Compare every deterministic input/output row.

        Args:
            context: Verification context.

        Returns:
            Decisive result with exhaustive evidence.
        """
        from qceval.semantics.verifiers.exact.metrics import _numeric_result

        started = time.perf_counter()
        actual = dict(self._materializer.classical_table(context).rows)
        target = dict(self._targets.classical_table(context).rows)
        cases = len(target)
        error = 0.0 if actual == target else 1.0
        spec = ExactEngineSpec(
            "classical_io_exhaustive",
            "classical_io",
            "classical_table",
            "max_case_error",
            ("classical_io", "framework_neutral_ir"),
        )
        return _numeric_result(context, spec, error, 0.0, cases, time.perf_counter() - started)


def state_engine(materializer: Materializer, targets: TargetProvider) -> ExactArrayEngine:
    """Build the exact pure-state engine.

    Args:
        materializer: Candidate semantic materializer.
        targets: Independent target provider.

    Returns:
        Configured exact state engine.
    """
    return ExactArrayEngine(
        ExactEngineSpec(
            "state_exact",
            "state",
            "statevector",
            "trace_distance",
            ("state", "static", "pure_state", "terminal_measurement_removal", "framework_neutral_ir"),
        ),
        materializer,
        targets,
    )


def unitary_engine(materializer: Materializer, targets: TargetProvider) -> ExactArrayEngine:
    """Build the exact total-unitary engine.

    Args:
        materializer: Candidate semantic materializer.
        targets: Independent target provider.

    Returns:
        Configured exact unitary engine.
    """
    return ExactArrayEngine(
        ExactEngineSpec(
            "unitary_exact",
            "total_unitary",
            "unitary",
            "operator_norm",
            ("static", "total_unitary", "global_phase", "framework_neutral_ir"),
        ),
        materializer,
        targets,
    )


def isometry_engine(materializer: Materializer, targets: TargetProvider) -> ExactArrayEngine:
    """Build the exact logical-isometry engine.

    Args:
        materializer: Candidate semantic materializer.
        targets: Independent target provider.

    Returns:
        Configured exact isometry engine.
    """
    return ExactArrayEngine(
        ExactEngineSpec(
            "isometry_exact",
            "isometry",
            "isometry",
            "operator_norm",
            (
                "isometry",
                "static",
                "logical_subspace",
                "ancilla_restoration",
                "global_phase",
                "framework_neutral_ir",
            ),
        ),
        materializer,
        targets,
    )


def channel_engine(materializer: Materializer, targets: TargetProvider) -> ExactArrayEngine:
    """Build the exact Choi-equality channel engine.

    Args:
        materializer: Candidate semantic materializer.
        targets: Independent target provider.

    Returns:
        Configured exact channel engine.
    """
    return ExactArrayEngine(
        ExactEngineSpec(
            "channel_exact",
            "channel",
            "choi",
            "normalized_choi_frobenius",
            ("channel", "framework_neutral_ir"),
        ),
        materializer,
        targets,
    )
