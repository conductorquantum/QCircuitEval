"""Typed framework-lowering interface and capability results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from qceval.semantics.contracts import Contract
from qceval.semantics.ir import Program


class LoweringStatus(StrEnum):
    """Non-verdict outcomes of framework inspection."""

    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    EXECUTION_ERROR = "execution_error"
    RESOURCE_LIMIT = "resource_limit"


@dataclass(frozen=True)
class CapabilitySet:
    """Stable adapter feature claims."""

    features: tuple[str, ...]


@dataclass(frozen=True)
class FrameworkFingerprint:
    """Framework and backend versions relevant to lowering."""

    framework: str
    version: str
    backend: str | None = None


@dataclass(frozen=True)
class SourceMetadata:
    """Candidate source diagnostics supplied to a lowering adapter."""

    framework: str
    source_hash: str | None = None
    backend: str | None = None


@dataclass(frozen=True)
class LoweringError:
    """Stable capability, inspection, or resource failure."""

    reason: str
    node_kind: str | None = None
    source_location: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class LoweringResult:
    """Exactly one successful Program IR or typed failure."""

    status: LoweringStatus
    program: Program | None = None
    error: LoweringError | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        success = self.status is LoweringStatus.SUCCESS
        if success != (self.program is not None) or success == (self.error is not None):
            raise ValueError("lowering result must contain exactly one program or error")


class LoweringAdapter(Protocol):
    """Framework adapter contract consumed by the lowering registry."""

    def lower(
        self,
        returned_value: Any,
        source_metadata: SourceMetadata,
        contract: Contract | None,
    ) -> LoweringResult:
        """Lower one native returned value.

        Args:
            returned_value: Native framework object returned by the candidate.
            source_metadata: Framework, source, and backend diagnostics.
            contract: Optional semantic contract and resource limits.

        Returns:
            Exactly one normalized Program IR or typed failure.
        """
        ...

    def capabilities(self) -> CapabilitySet:
        """Return stable feature claims."""
        ...

    def framework_fingerprint(self) -> FrameworkFingerprint:
        """Return relevant framework versions."""
        ...
