"""Internal evaluator data models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GradeContext:
    """Inputs passed from framework execution to generic graders.

    Attributes:
        probabilities: Candidate probability vector.
        canonical_probabilities: Reference probability vector, if required.
        candidate_unitary: Candidate unitary matrix, if available.
        canonical_unitary: Reference unitary matrix, if required.
        target_unitary: Analytic target unitary matrix, if specified.
        metadata: Framework execution metadata used by structural graders.
        code: Candidate source code used by structural graders.
    """

    probabilities: Sequence[float]
    canonical_probabilities: Sequence[float] | None = None
    candidate_unitary: Any | None = None
    canonical_unitary: Any | None = None
    target_unitary: Any | None = None
    metadata: Mapping[str, Any] | None = None
    code: str | None = None


@dataclass
class ExecutionResult:
    """Framework execution output normalized for grading.

    Attributes:
        probabilities: Candidate probability vector in integer bitstring order.
        metadata: Framework-specific metadata about circuit structure and
            probability extraction.
        unitary: Candidate unitary matrix when available.
        circuit: Native framework circuit or tape object when available.
        statevector: Candidate final statevector when available.
    """

    probabilities: list[float]
    metadata: dict[str, Any]
    unitary: Any | None = None
    circuit: Any | None = None
    statevector: Any | None = None
