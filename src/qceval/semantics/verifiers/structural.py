"""Hard requirement, lifecycle, and diagnostic separation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from qceval.semantics.contracts import Contract, RequirementSpec
from qceval.semantics.verifiers.result import VerifierResult


class RequirementClass(StrEnum):
    """Whether one requirement can affect an authoritative verdict."""

    HARD_API = "hard_api"
    LIFECYCLE = "lifecycle"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class ClassifiedRequirement:
    """One contract requirement and its authority class."""

    requirement: RequirementSpec
    authority: RequirementClass


_LIFECYCLE_KINDS = frozenset({"measurement_exclusion", "ancilla_policy", "register_mapping"})
_DIAGNOSTIC_KINDS = frozenset({"prompt_hashes", "legacy_comparison", "gate_counts", "source_similarity"})


def classify_requirements(contract: Contract) -> tuple[ClassifiedRequirement, ...]:
    """Classify every explicit contract requirement.

    Args:
        contract: Task semantic contract.

    Returns:
        Stable requirement classifications.
    """
    result = []
    for requirement in contract.requirements:
        if requirement.kind in _DIAGNOSTIC_KINDS:
            authority = RequirementClass.DIAGNOSTIC
        elif requirement.kind in _LIFECYCLE_KINDS:
            authority = RequirementClass.LIFECYCLE
        else:
            authority = RequirementClass.HARD_API
        result.append(ClassifiedRequirement(requirement, authority))
    return tuple(result)


def attach_diagnostics(result: VerifierResult, diagnostics: dict[str, str]) -> VerifierResult:
    """Attach bounded diagnostics without changing status or reason.

    Args:
        result: Authoritative semantic result.
        diagnostics: Non-authoritative stable key/value observations.

    Returns:
        Result with sorted diagnostics and unchanged verdict fields.
    """
    return replace(result, diagnostics=tuple(sorted(diagnostics.items())))
