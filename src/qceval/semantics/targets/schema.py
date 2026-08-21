"""Schema types and strict validators for semantic target manifests.

This module owns immutable target dataclasses, path-addressed validation
errors, and JSON parsing for individual manifest entries.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

TARGET_SCHEMA_VERSION = "1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class TargetValidationError(ValueError):
    """A stable, path-addressed target validation failure."""

    def __init__(self, path: str, reason: str) -> None:
        """Initialize a validation error.

        Args:
            path: Dotted JSON path at which validation failed.
            reason: Stable human-readable failure reason.
        """
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class TargetDerivation:
    """One target derivation independent of a canonical implementation."""

    derivation_id: str
    method: str
    narrative: str
    evidence: str


@dataclass(frozen=True)
class TargetManifest:
    """Strict provenance and reproduction metadata for one target."""

    schema_version: str
    suite: str
    task_id: str
    target_id: str
    target_version: str
    artifact: str
    artifact_sha256: str
    artifact_format: str
    dimensions: tuple[int, ...]
    source: str
    normalization: str
    generator_command: str
    derivations: tuple[TargetDerivation, ...]
    invariants: tuple[str, ...]
    applicable_frameworks: tuple[str, ...]
    known_ambiguities: tuple[str, ...]
    resolution_adrs: tuple[str, ...]


@dataclass(frozen=True)
class TargetVerification:
    """Reproduction result for one packaged target."""

    task_id: str
    target_id: str
    artifact_sha256: str
    byte_count: int
    derivation_count: int


def parse_target_manifest_json(payload: str | bytes) -> TargetManifest:
    """Parse and strictly validate a target manifest.

    Args:
        payload: UTF-8 target manifest JSON.

    Returns:
        Immutable target manifest.

    Raises:
        TargetValidationError: If the manifest is malformed.
    """
    raw = _parse_json(payload)
    obj = _object(
        raw,
        "$",
        {
            "schema_version",
            "suite",
            "task_id",
            "target_id",
            "target_version",
            "artifact",
            "artifact_sha256",
            "artifact_format",
            "dimensions",
            "source",
            "normalization",
            "generator_command",
            "derivations",
            "invariants",
            "applicable_frameworks",
            "known_ambiguities",
            "resolution_adrs",
        },
    )
    schema_version = _schema_version(obj["schema_version"])
    version = _semantic_version(obj["target_version"])
    digest = _sha256(obj["artifact_sha256"])
    artifact = _artifact_name(obj["artifact"])
    dimensions = _dimensions(obj["dimensions"])
    derivations = _derivations(obj["derivations"])
    frameworks = _frameworks(obj["applicable_frameworks"])
    return TargetManifest(
        schema_version=schema_version,
        suite=_string(obj["suite"], "$.suite"),
        task_id=_string(obj["task_id"], "$.task_id"),
        target_id=_string(obj["target_id"], "$.target_id"),
        target_version=version,
        artifact=artifact,
        artifact_sha256=digest,
        artifact_format=_string(obj["artifact_format"], "$.artifact_format"),
        dimensions=dimensions,
        source=_string(obj["source"], "$.source"),
        normalization=_string(obj["normalization"], "$.normalization"),
        generator_command=_string(obj["generator_command"], "$.generator_command"),
        derivations=derivations,
        invariants=_strings(obj["invariants"], "$.invariants"),
        applicable_frameworks=frameworks,
        known_ambiguities=_strings(obj["known_ambiguities"], "$.known_ambiguities"),
        resolution_adrs=_strings(obj["resolution_adrs"], "$.resolution_adrs"),
    )


def _parse_json(payload: str | bytes) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TargetValidationError("$", f"invalid JSON: {exc}") from exc


def _schema_version(value: Any) -> str:
    version = _string(value, "$.schema_version")
    if version != TARGET_SCHEMA_VERSION:
        raise TargetValidationError("$.schema_version", f"unsupported version {version!r}")
    return version


def _semantic_version(value: Any) -> str:
    version = _string(value, "$.target_version")
    if not _SEMVER.fullmatch(version):
        raise TargetValidationError("$.target_version", "must be semantic version x.y.z")
    return version


def _sha256(value: Any) -> str:
    digest = _string(value, "$.artifact_sha256")
    if not _SHA256.fullmatch(digest):
        raise TargetValidationError("$.artifact_sha256", "must be lowercase SHA-256")
    return digest


def _artifact_name(value: Any) -> str:
    artifact = _string(value, "$.artifact")
    if PurePosixPath(artifact).name != artifact or artifact in {".", ".."}:
        raise TargetValidationError("$.artifact", "must be a local filename")
    return artifact


def _dimensions(value: Any) -> tuple[int, ...]:
    dimensions = tuple(
        _positive_int(item, f"$.dimensions[{index}]") for index, item in enumerate(_array(value, "$.dimensions"))
    )
    if not dimensions:
        raise TargetValidationError("$.dimensions", "must not be empty")
    return dimensions


def _derivations(value: Any) -> tuple[TargetDerivation, ...]:
    derivations = tuple(
        _parse_derivation(item, f"$.derivations[{index}]") for index, item in enumerate(_array(value, "$.derivations"))
    )
    if not derivations:
        raise TargetValidationError("$.derivations", "must not be empty")
    if len({item.derivation_id for item in derivations}) != len(derivations):
        raise TargetValidationError("$.derivations", "derivation ids must be unique")
    return derivations


def _frameworks(value: Any) -> tuple[str, ...]:
    frameworks = _strings(value, "$.applicable_frameworks")
    if set(frameworks) != {"qiskit", "cirq", "pennylane", "cudaq"}:
        raise TargetValidationError("$.applicable_frameworks", "must cover all four core frameworks")
    return frameworks


def _parse_derivation(value: Any, path: str) -> TargetDerivation:
    raw = _object(value, path, {"id", "method", "narrative", "evidence"})
    return TargetDerivation(
        derivation_id=_string(raw["id"], f"{path}.id"),
        method=_string(raw["method"], f"{path}.method"),
        narrative=_string(raw["narrative"], f"{path}.narrative"),
        evidence=_string(raw["evidence"], f"{path}.evidence"),
    )


def _object(value: Any, path: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TargetValidationError(path, "must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise TargetValidationError(path, f"field mismatch; missing={missing}, unknown={unknown}")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TargetValidationError(path, "must be an array")
    return value


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TargetValidationError(path, "must be a non-empty string")
    return value


def _strings(value: Any, path: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(_array(value, path)))


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TargetValidationError(path, "must be a positive integer")
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TargetValidationError("$", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise TargetValidationError("$", f"non-finite JSON constant {value!r}")
