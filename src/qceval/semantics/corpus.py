"""Strict calibration-corpus manifests and content verification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CORPUS_SCHEMA_VERSION = "1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = frozenset({"verified_pass", "semantic_fail", "execution_error", "resource_limit"})
_CATEGORIES = frozenset({"valid", "mutant", "boundary"})


class CorpusValidationError(ValueError):
    """A stable path-addressed corpus validation failure."""

    def __init__(self, path: str, reason: str) -> None:
        """Initialize a corpus validation failure.

        Args:
            path: JSON manifest path.
            reason: Stable failure reason.
        """
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class CorpusFixture:
    """One independently labeled candidate fixture."""

    fixture_id: str
    task_id: str
    framework: str
    category: str
    path: str
    content_sha256: str
    contract_version: str
    target_sha256: str
    expected_status: str
    mutation_class: str | None
    release_blocking: bool
    author: str
    provenance: str
    review_status: str


@dataclass(frozen=True)
class CorpusManifest:
    """Versioned ordered behavior-grader corpus manifest."""

    schema_version: str
    corpus_version: str
    fixtures: tuple[CorpusFixture, ...]


def load_corpus_manifest(path: Path) -> CorpusManifest:
    """Load and strictly validate a corpus manifest.

    Args:
        path: Manifest JSON path.

    Returns:
        Immutable manifest.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise CorpusValidationError("$", f"invalid JSON: {exc}") from exc
    raw = _object(payload, "$", {"schema_version", "corpus_version", "fixtures"})
    version = _string(raw["schema_version"], "$.schema_version")
    if version != CORPUS_SCHEMA_VERSION:
        raise CorpusValidationError("$.schema_version", f"unsupported version {version!r}")
    fixtures = tuple(
        _parse_fixture(value, f"$.fixtures[{index}]")
        for index, value in enumerate(_array(raw["fixtures"], "$.fixtures"))
    )
    ids = [item.fixture_id for item in fixtures]
    if len(set(ids)) != len(ids):
        raise CorpusValidationError("$.fixtures", "fixture ids must be unique")
    return CorpusManifest(version, _string(raw["corpus_version"], "$.corpus_version"), fixtures)


def verify_corpus_files(manifest_path: Path, manifest: CorpusManifest | None = None) -> None:
    """Verify every fixture path and content hash.

    Args:
        manifest_path: Manifest location used as the relative-path root.
        manifest: Already parsed manifest, or ``None`` to load it.

    Raises:
        CorpusValidationError: If a fixture is missing or has different bytes.
    """
    manifest = manifest or load_corpus_manifest(manifest_path)
    for index, fixture in enumerate(manifest.fixtures):
        path = manifest_path.parent / fixture.path
        if not path.is_file():
            raise CorpusValidationError(f"$.fixtures[{index}].path", "fixture file is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != fixture.content_sha256:
            raise CorpusValidationError(f"$.fixtures[{index}].content_sha256", "fixture hash mismatch")


def _parse_fixture(value: Any, path: str) -> CorpusFixture:
    raw = _object(
        value,
        path,
        {
            "id",
            "task_id",
            "framework",
            "category",
            "path",
            "content_sha256",
            "contract_version",
            "target_sha256",
            "expected_status",
            "mutation_class",
            "release_blocking",
            "author",
            "provenance",
            "review_status",
        },
    )
    category = _choice(raw["category"], f"{path}.category", _CATEGORIES)
    expected = _choice(raw["expected_status"], f"{path}.expected_status", _STATUSES)
    if (category == "valid") != (expected == "verified_pass"):
        raise CorpusValidationError(path, "only valid fixtures may expect verified_pass")
    mutation = raw["mutation_class"]
    if mutation is not None:
        mutation = _string(mutation, f"{path}.mutation_class")
    if category == "mutant" and mutation is None:
        raise CorpusValidationError(f"{path}.mutation_class", "mutants require a mutation class")
    fixture_path = _string(raw["path"], f"{path}.path")
    pure = PurePosixPath(fixture_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise CorpusValidationError(f"{path}.path", "must be a safe relative path")
    content_hash = _digest(raw["content_sha256"], f"{path}.content_sha256")
    target_hash = _digest(raw["target_sha256"], f"{path}.target_sha256")
    release_blocking = raw["release_blocking"]
    if not isinstance(release_blocking, bool):
        raise CorpusValidationError(f"{path}.release_blocking", "must be boolean")
    return CorpusFixture(
        fixture_id=_string(raw["id"], f"{path}.id"),
        task_id=_string(raw["task_id"], f"{path}.task_id"),
        framework=_string(raw["framework"], f"{path}.framework"),
        category=category,
        path=fixture_path,
        content_sha256=content_hash,
        contract_version=_string(raw["contract_version"], f"{path}.contract_version"),
        target_sha256=target_hash,
        expected_status=expected,
        mutation_class=mutation,
        release_blocking=release_blocking,
        author=_string(raw["author"], f"{path}.author"),
        provenance=_string(raw["provenance"], f"{path}.provenance"),
        review_status=_choice(raw["review_status"], f"{path}.review_status", {"provisional", "reviewed"}),
    )


def _object(value: Any, path: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CorpusValidationError(path, "must be an object")
    if set(value) != keys:
        raise CorpusValidationError(
            path, f"field mismatch; missing={sorted(keys - set(value))}, unknown={sorted(set(value) - keys)}"
        )
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise CorpusValidationError(path, "must be an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CorpusValidationError(path, "must be a non-empty string")
    return value


def _choice(value: Any, path: str, allowed: set[str] | frozenset[str]) -> str:
    result = _string(value, path)
    if result not in allowed:
        raise CorpusValidationError(path, f"unsupported value {result!r}")
    return result


def _digest(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _SHA256.fullmatch(result):
        raise CorpusValidationError(path, "must be lowercase SHA-256")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusValidationError("$", f"duplicate JSON key {key!r}")
        result[key] = value
    return result
