"""Verifier engine interface and deterministic cost estimates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from qceval.semantics.contracts import Contract
from qceval.semantics.ir import Program
from qceval.semantics.verifiers.result import VerifierResult


@dataclass(frozen=True)
class CostEstimate:
    """Deterministic preflight resource estimate."""

    qubits: int
    dimension: int
    cases: int
    memory_mib: int
    wall_seconds: float
    branches: int = 1


@dataclass(frozen=True)
class EngineDescriptor:
    """Stable engine capabilities and version."""

    name: str
    version: str
    kinds: tuple[str, ...]
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class VerificationContext:
    """Framework-neutral inputs supplied to one engine."""

    contract: Contract
    contract_hash: str
    target_hash: str
    input_hash: str
    program: Program
    arguments: tuple[object, ...] = ()
    execution_metadata: Mapping[str, object] | None = None
    execution_probabilities: tuple[float, ...] | None = None


class VerifierEngine(Protocol):
    """Semantic engine contract used by the router."""

    def descriptor(self) -> EngineDescriptor:
        """Return immutable capability metadata."""
        ...

    def estimate(self, context: VerificationContext) -> CostEstimate:
        """Return deterministic cost before materialization.

        Args:
            context: Framework-neutral verification inputs.

        Returns:
            Conservative resource estimate.
        """
        ...

    def verify(self, context: VerificationContext) -> VerifierResult:
        """Verify one framework-neutral context.

        Args:
            context: Framework-neutral verification inputs.

        Returns:
            One fail-closed semantic result.
        """
        ...
