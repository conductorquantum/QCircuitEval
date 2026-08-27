"""Framework adapters for semantic Program IR."""

from qceval.semantics.lowering.base import (
    CapabilitySet,
    FrameworkFingerprint,
    LoweringError,
    LoweringResult,
    LoweringStatus,
    SourceMetadata,
)
from qceval.semantics.lowering.registry import LoweringRegistry, default_lowering_registry

__all__ = [
    "CapabilitySet",
    "FrameworkFingerprint",
    "LoweringError",
    "LoweringRegistry",
    "LoweringResult",
    "LoweringStatus",
    "SourceMetadata",
    "default_lowering_registry",
]
