"""Versioned fail-closed semantic verifier results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

RESULT_SCHEMA_VERSION = "1"


class SemanticStatus(StrEnum):
    """Authoritative, decisive semantic and operational outcomes."""

    VERIFIED_PASS = "verified_pass"
    SEMANTIC_FAIL = "semantic_fail"
    EXECUTION_ERROR = "execution_error"
    RESOURCE_LIMIT = "resource_limit"


@dataclass(frozen=True)
class EvidenceRecord:
    """Bounded evidence emitted by one verifier engine."""

    engine: str
    engine_version: str
    reason: str
    input_hash: str
    target_hash: str
    metric: str | None = None
    value: float | None = None
    tolerance: float | None = None
    uncertainty: float | None = None
    cases_checked: int = 0
    elapsed_seconds: float = 0.0
    peak_rss_mib: float | None = None
    preconditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("engine", "engine_version", "reason", "input_hash", "target_hash"):
            if not getattr(self, name):
                raise ValueError(f"evidence {name} must not be empty")
        for name in ("value", "tolerance", "uncertainty", "elapsed_seconds", "peak_rss_mib"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"evidence {name} must be finite and non-negative")
        if self.cases_checked < 0:
            raise ValueError("evidence cases_checked must be non-negative")


@dataclass(frozen=True)
class VerifierResult:
    """One versioned authoritative status plus bounded evidence."""

    schema_version: str
    status: SemanticStatus
    reason: str
    contract_hash: str
    target_hash: str
    verifier_version: str
    evidence: tuple[EvidenceRecord, ...] = ()
    diagnostics: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported result schema {self.schema_version!r}")
        if not self.reason or not self.contract_hash or not self.target_hash or not self.verifier_version:
            raise ValueError("result reason and version hashes must not be empty")

    @property
    def passed(self) -> bool:
        """Return the compatibility pass projection."""
        return self.status is SemanticStatus.VERIFIED_PASS


def make_verifier_result(
    status: SemanticStatus,
    reason: str,
    *,
    contract_hash: str,
    target_hash: str,
    verifier_version: str,
    evidence: tuple[EvidenceRecord, ...] = (),
    diagnostics: tuple[tuple[str, str], ...] = (),
) -> VerifierResult:
    """Build one versioned verifier result.

    Args:
        status: Authoritative semantic status.
        reason: Stable reason code.
        contract_hash: Contract identity hash.
        target_hash: Target artifact hash.
        verifier_version: Engine or portfolio version identity.
        evidence: Bounded evidence records.
        diagnostics: Optional non-authoritative diagnostics.

    Returns:
        Validated verifier result.
    """
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        reason,
        contract_hash,
        target_hash,
        verifier_version,
        evidence,
        diagnostics,
    )


def make_evidence(
    engine: str,
    engine_version: str,
    reason: str,
    *,
    input_hash: str,
    target_hash: str,
    metric: str | None = None,
    value: float | None = None,
    tolerance: float | None = None,
    uncertainty: float | None = None,
    cases_checked: int = 0,
    elapsed_seconds: float = 0.0,
    peak_rss_mib: float | None = None,
    preconditions: tuple[str, ...] = (),
) -> EvidenceRecord:
    """Build one bounded evidence record.

    Args:
        engine: Engine identity.
        engine_version: Engine version identity.
        reason: Stable reason code.
        input_hash: Candidate or program input hash.
        target_hash: Target artifact hash.
        metric: Optional metric name.
        value: Optional metric value.
        tolerance: Optional tolerance.
        uncertainty: Optional uncertainty.
        cases_checked: Number of checked cases.
        elapsed_seconds: Elapsed runtime.
        peak_rss_mib: Optional peak RSS.
        preconditions: Optional precondition tags.

    Returns:
        Validated evidence record.
    """
    return EvidenceRecord(
        engine,
        engine_version,
        reason,
        input_hash,
        target_hash,
        metric=metric,
        value=value,
        tolerance=tolerance,
        uncertainty=uncertainty,
        cases_checked=cases_checked,
        elapsed_seconds=elapsed_seconds,
        peak_rss_mib=peak_rss_mib,
        preconditions=preconditions,
    )
