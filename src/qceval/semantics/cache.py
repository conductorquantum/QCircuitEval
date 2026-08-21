"""Atomic content-addressed cache for bounded semantic artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = "1"
MAX_CACHE_ENTRY_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class SemanticCacheKey:
    """Every semantic and environment dependency of a cache entry."""

    candidate_hash: str
    contract_hash: str
    target_hash: str
    ir_version: str
    verifier_version: str
    framework: str
    framework_version: str
    backend: str | None
    precision: str
    parameters_hash: str
    observation_hash: str
    limits_hash: str

    @property
    def digest(self) -> str:
        """Return canonical SHA-256 cache identity.

        Returns:
            Lowercase hexadecimal digest.
        """
        return hashlib.sha256(_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True)
class CacheLookup:
    """Cache hit/miss result with fresh operational provenance."""

    hit: bool
    reason: str
    payload: Mapping[str, Any] | None


class ContentAddressedCache:
    """Store checksummed JSON entries with atomic replacement."""

    def __init__(self, root: Path) -> None:
        """Initialize a private cache root.

        Args:
            root: Cache directory.
        """
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def get(self, key: SemanticCacheKey) -> CacheLookup:
        """Read and validate one cache entry.

        Args:
            key: Complete semantic cache identity.

        Returns:
            Hit payload or a corruption-safe miss.
        """
        path = self.path_for(key)
        if not path.exists():
            return CacheLookup(False, "cache_miss", None)
        try:
            if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("cache entry is not a regular file")
            if stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise ValueError("cache entry permissions are unsafe")
            if path.stat().st_size > MAX_CACHE_ENTRY_BYTES:
                raise ValueError("cache entry exceeds size limit")
            value = json.loads(path.read_text(encoding="utf-8"))
            payload = _validate_entry(value, key)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            path.unlink(missing_ok=True)
            return CacheLookup(False, "cache_corrupt", None)
        return CacheLookup(True, "cache_hit", payload)

    def put(self, key: SemanticCacheKey, payload: Mapping[str, Any]) -> Path:
        """Atomically write one checksummed JSON entry.

        Args:
            key: Complete semantic cache identity.
            payload: Bounded JSON-compatible semantic artifact.

        Returns:
            Final cache path.
        """
        payload_value = json.loads(_json_bytes(payload))
        payload_bytes = _json_bytes(payload_value)
        entry = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": asdict(key),
            "key_sha256": key.digest,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "payload": payload_value,
        }
        data = _json_bytes(entry)
        if len(data) > MAX_CACHE_ENTRY_BYTES:
            raise ValueError("cache entry exceeds size limit")
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    def path_for(self, key: SemanticCacheKey) -> Path:
        """Return the sharded path for one key.

        Args:
            key: Complete semantic cache identity.

        Returns:
            Cache entry path.
        """
        digest = key.digest
        return self.root / digest[:2] / f"{digest[2:]}.json"


def _validate_entry(value: Any, key: SemanticCacheKey) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "key",
        "key_sha256",
        "payload_sha256",
        "payload",
    }:
        raise ValueError("cache entry fields differ")
    if value["schema_version"] != CACHE_SCHEMA_VERSION or value["key"] != asdict(key):
        raise ValueError("cache entry identity mismatch")
    if value["key_sha256"] != key.digest or not isinstance(value["payload"], dict):
        raise ValueError("cache entry digest mismatch")
    payload = value["payload"]
    if value["payload_sha256"] != hashlib.sha256(_json_bytes(payload)).hexdigest():
        raise ValueError("cache payload checksum mismatch")
    return payload


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
