"""Reproduction and hash verification for packaged semantic targets."""

from __future__ import annotations

import hashlib

from qceval.assets._resources import target_resource
from qceval.semantics._target_generators import pilot_target_generator
from qceval.semantics.targets.load import (
    _target_document,
    canonical_target_bytes,
    load_packaged_target_manifest,
)
from qceval.semantics.targets.schema import TargetValidationError, TargetVerification, _parse_json

PILOT_TASK_IDS = ("02", "27", "28", "42")


def generated_target_bytes(task_id: str) -> bytes:
    """Generate canonical target bytes for a core task.

    Args:
        task_id: Core task id.

    Returns:
        Deterministic artifact bytes.
    """
    normalized = str(task_id).zfill(2)
    generator = pilot_target_generator(normalized)
    if generator is not None:
        return canonical_target_bytes(generator())
    from qceval.semantics.core_audit import generated_target_payload

    return canonical_target_bytes(generated_target_payload(normalized))


def verify_packaged_target(task_id: str) -> TargetVerification:
    """Hash-check one packaged core target document against its manifest.

    Args:
        task_id: Core task id.

    Returns:
        Reproduction and provenance summary.
    """
    manifest = load_packaged_target_manifest(task_id)
    artifact_payload = _parse_json(target_resource("core", manifest.artifact).read_bytes())
    document = _target_document(artifact_payload, manifest.task_id)
    packaged = canonical_target_bytes(document)
    digest = hashlib.sha256(packaged).hexdigest()
    if digest != manifest.artifact_sha256:
        raise TargetValidationError("$.artifact_sha256", "does not match packaged artifact")
    generator = pilot_target_generator(manifest.task_id)
    if generator is not None and packaged != canonical_target_bytes(generator()):
        raise TargetValidationError("$.artifact", "packaged bytes differ from deterministic generator")
    return TargetVerification(
        task_id=manifest.task_id,
        target_id=manifest.target_id,
        artifact_sha256=digest,
        byte_count=len(packaged),
        derivation_count=len(manifest.derivations),
    )


def verify_all_pilot_targets() -> tuple[TargetVerification, ...]:
    """Reproduce all packaged pilot targets in stable task order.

    Returns:
        Stable tuple of reproduction summaries.
    """
    return tuple(verify_packaged_target(task_id) for task_id in PILOT_TASK_IDS)


def verify_all_core_targets() -> tuple[TargetVerification, ...]:
    """Reproduce all packaged core targets in stable task order.

    Returns:
        Stable tuple of reproduction summaries.
    """
    from qceval.semantics.contracts import ContractRegistry

    registry = ContractRegistry.from_package("core")
    return tuple(verify_packaged_target(contract.task_id) for contract in registry)
