"""Loading and canonical serialization of packaged semantic targets.

Supports both per-task manifests and suite-level grouped manifests. Artifact
hashes always cover the selected task document only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from qceval.assets._resources import read_bytes, target_resource
from qceval.semantics.targets.schema import (
    TargetManifest,
    TargetValidationError,
    _parse_json,
    parse_target_manifest_json,
)


def canonical_target_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a generated target as stable finite JSON plus newline.

    Args:
        payload: JSON-compatible target object.

    Returns:
        Canonical UTF-8 artifact bytes.
    """
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def load_contract_target_document(contract: Any) -> dict[str, Any]:
    """Load and hash-verify the target document selected by a contract.

    Both per-task manifests and suite-level grouped manifests are supported.
    Hashes always cover the canonical bytes of the selected task document,
    never unrelated targets in the same package artifact.

    Args:
        contract: Validated semantic contract selecting a packaged target.

    Returns:
        The selected task's complete target document.
    """

    manifest_parts = contract.target.manifest.split("/")
    manifest_payload = _parse_json(read_bytes(*manifest_parts))
    if not isinstance(manifest_payload, dict):
        raise ValueError("target manifest is malformed")
    manifest = _manifest_entry(manifest_payload, contract.task_id)
    _validate_contract_manifest_identity(contract, manifest)

    artifact = manifest.get("artifact")
    if not isinstance(artifact, str) or PurePosixPath(artifact).name != artifact:
        raise ValueError("target artifact name is invalid")
    artifact_payload = _parse_json(read_bytes(*manifest_parts[:-1], artifact))
    document = _target_document(artifact_payload, contract.task_id)
    digest = hashlib.sha256(canonical_target_bytes(document)).hexdigest()
    if digest != contract.target.sha256 or digest != manifest.get("artifact_sha256"):
        raise ValueError("target artifact hash mismatch")
    return document


def load_packaged_target_manifest(task_id: str) -> TargetManifest:
    """Load one packaged core target manifest from the suite-level asset.

    Args:
        task_id: Core task id.

    Returns:
        Strictly parsed target manifest.
    """
    normalized = str(task_id).zfill(2)
    payload = _parse_json(target_resource("core", "manifest.json").read_bytes())
    if not isinstance(payload, dict):
        raise TargetValidationError("$", "suite target manifest is malformed")
    entry = _manifest_entry(payload, normalized)
    manifest = parse_target_manifest_json(json.dumps(entry, separators=(",", ":"), ensure_ascii=False))
    if manifest.suite != "core" or manifest.task_id != normalized:
        raise TargetValidationError("$", "manifest registry key does not match resource path")
    return manifest


def _manifest_entry(payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    tasks = payload.get("tasks")
    if tasks is None:
        return payload
    if not isinstance(tasks, dict) or not isinstance(tasks.get(task_id), dict):
        raise ValueError(f"target manifest is missing task {task_id!r}")
    return tasks[task_id]


def _target_document(payload: Any, task_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("target artifact is malformed")
    tasks = payload.get("tasks")
    if tasks is None:
        return payload
    if not isinstance(tasks, dict) or not isinstance(tasks.get(task_id), dict):
        raise ValueError(f"target artifact is missing task {task_id!r}")
    return tasks[task_id]


def _validate_contract_manifest_identity(contract: Any, manifest: dict[str, Any]) -> None:
    expected = {
        "suite": contract.suite,
        "task_id": contract.task_id,
        "target_id": contract.target.target_id,
        "target_version": contract.target.version,
        "artifact_sha256": contract.target.sha256,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("target manifest identity mismatch")
    derivations = manifest.get("derivations")
    if not isinstance(derivations, list) or len(derivations) != contract.target.independent_derivations:
        raise ValueError("target derivation count mismatch")
