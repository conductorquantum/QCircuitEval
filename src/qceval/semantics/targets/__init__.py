"""Versioned, reproducible semantic target artifacts.

Target generators encode prompt-derived mathematical objects.  They are kept
separate from verifier implementations so that a verifier cannot silently
define its own truth data.
"""

from qceval.semantics.targets.load import (
    canonical_target_bytes,
    load_contract_target_document,
    load_packaged_target_manifest,
)
from qceval.semantics.targets.schema import (
    TARGET_SCHEMA_VERSION,
    TargetDerivation,
    TargetManifest,
    TargetValidationError,
    TargetVerification,
    parse_target_manifest_json,
)
from qceval.semantics.targets.verify import (
    PILOT_TASK_IDS,
    generated_target_bytes,
    verify_all_core_targets,
    verify_all_pilot_targets,
    verify_packaged_target,
)

__all__ = [
    "PILOT_TASK_IDS",
    "TARGET_SCHEMA_VERSION",
    "TargetDerivation",
    "TargetManifest",
    "TargetValidationError",
    "TargetVerification",
    "canonical_target_bytes",
    "generated_target_bytes",
    "load_contract_target_document",
    "load_packaged_target_manifest",
    "parse_target_manifest_json",
    "verify_all_core_targets",
    "verify_all_pilot_targets",
    "verify_packaged_target",
]
