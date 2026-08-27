"""Framework-neutral materialization interfaces for exact engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qceval.semantics.verifiers.base import VerificationContext


class CandidateSemanticError(ValueError):
    """A decisive malformed or invalid candidate semantic object."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class ArrayMaterialization:
    """Dense complex semantic object with sanity metadata."""

    value: np.ndarray
    representation: str
    cases: int = 1


@dataclass(frozen=True)
class ClassicalTableMaterialization:
    """Complete finite deterministic classical input/output relation."""

    rows: tuple[tuple[str, str], ...]


class Materializer(Protocol):
    """Produce candidate semantics from Program IR behind one narrow seam."""

    def array(self, context: VerificationContext, representation: str) -> ArrayMaterialization:
        """Materialize a state, operator, isometry, or Choi matrix.

        Args:
            context: Task contract, hashes, and candidate Program IR.
            representation: Requested dense semantic representation.

        Returns:
            Candidate dense semantic object.
        """
        ...

    def classical_table(self, context: VerificationContext) -> ClassicalTableMaterialization:
        """Materialize a complete finite classical relation.

        Args:
            context: Task contract, hashes, and candidate Program IR.

        Returns:
            Candidate deterministic relation.
        """
        ...


class TargetProvider(Protocol):
    """Load independently generated target semantics."""

    def array(self, context: VerificationContext, representation: str) -> ArrayMaterialization:
        """Load a target dense object.

        Args:
            context: Task contract, hashes, and candidate Program IR.
            representation: Requested dense semantic representation.

        Returns:
            Independently generated target object.
        """
        ...

    def classical_table(self, context: VerificationContext) -> ClassicalTableMaterialization:
        """Load a target finite classical relation.

        Args:
            context: Task contract, hashes, and candidate Program IR.

        Returns:
            Independently generated target relation.
        """
        ...
