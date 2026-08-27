"""Persistent cache for provider responses.

The cache stores provider outputs by provider name, model, framework, task,
prompt hash, and non-secret provider settings.  Cache entries are written
atomically so interrupted runs do not leave partially written JSON files.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qceval.models import Framework, ProviderRequest, ProviderResponse, TokenUsage
from qceval.serialization import to_jsonable


@dataclass(frozen=True)
class CacheKey:
    """Stable provider-response cache identity.

    Attributes:
        provider: Provider implementation name.
        model: Model identifier, if any.
        framework: Framework targeted by the request.
        task_id: Zero-padded task identifier.
        sample_index: Repeated-sample index for Pass@K runs.
        attempt_index: Feedback-repair attempt index.
        prompt_sha: SHA-256 digest of prompt text.
        messages_sha: SHA-256 digest of chat messages when present.
        settings_sha: SHA-256 digest of non-secret provider settings.
    """

    provider: str
    model: str | None
    framework: Framework
    task_id: str
    sample_index: int
    attempt_index: int
    prompt_sha: str
    messages_sha: str | None
    settings_sha: str

    def filename(self) -> str:
        """Return filesystem-safe cache filename for this key."""
        provider = _safe_part(self.provider)
        model = _safe_part(self.model or "none")
        framework = _safe_part(self.framework)
        task_id = _safe_part(self.task_id)
        return (
            f"{provider}-{model}-{framework}-{task_id}-"
            f"s{self.sample_index}-a{self.attempt_index}-"
            f"{self.prompt_sha[:16]}-{_short_hash(self.messages_sha)}-{self.settings_sha[:16]}.json"
        )


class ResponseCache:
    """File-backed provider response cache.

    Args:
        root: Cache root directory.  Response JSON files are stored under the
            ``responses`` child directory, which is created on construction.

    Attributes:
        root: Expanded cache root path.
        responses_dir: Directory containing cached response JSON files.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        self.responses_dir = self.root / "responses"
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, request: ProviderRequest, *, provider: str, settings: Mapping[str, Any]) -> CacheKey:
        """Build cache key for a provider request.

        Secret-like settings are deliberately omitted so API keys and tokens do
        not leak into cache filenames or persisted metadata.

        Args:
            request: Provider request being cached.
            provider: Provider implementation name.
            settings: Provider configuration used for the request.

        Returns:
            Cache key derived from request identity and non-secret settings.
        """
        safe_settings = {key: value for key, value in settings.items() if not _is_secret_setting(key)}
        return CacheKey(
            provider=provider,
            model=request.model,
            framework=request.framework,
            task_id=request.task_id,
            sample_index=request.sample_index,
            attempt_index=request.attempt_index,
            prompt_sha=_sha256_text(request.prompt),
            messages_sha=None if not request.messages else _sha256_json({"messages": request.messages}),
            settings_sha=_sha256_json({"model": request.model, **safe_settings}),
        )

    def get(self, key: CacheKey) -> ProviderResponse | None:
        """Return cached response for ``key`` when present and valid.

        Corrupt or incompatible cache files are treated as misses.  This keeps
        resume behavior robust across interrupted writes and schema changes.

        Args:
            key: Cache identity to read.

        Returns:
            Cached provider response, or ``None`` on miss or invalid payload.
        """
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _response_from_dict(payload["response"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def put(self, key: CacheKey, response: ProviderResponse) -> None:
        """Persist provider response for ``key``.

        Args:
            key: Cache identity to write.
            response: Provider response to serialize.  Raw provider payloads are
                stripped before writing so cache entries stay small and stable.
        """
        if not response.ok:
            return
        payload = {"cache_key": to_jsonable(key), "response": _cacheable_response(response)}
        _atomic_write_json(self._path(key), payload)

    def _path(self, key: CacheKey) -> Path:
        return self.responses_dir / key.filename()


def _cacheable_response(response: ProviderResponse) -> dict[str, Any]:
    payload = response.to_dict()
    payload["raw_response"] = None
    return payload


def _response_from_dict(payload: Mapping[str, Any]) -> ProviderResponse:
    usage_payload = payload.get("usage")
    usage = None if usage_payload is None else TokenUsage(**usage_payload)
    return ProviderResponse(
        code=None if payload.get("code") is None else str(payload.get("code")),
        model=None if payload.get("model") is None else str(payload.get("model")),
        metadata=payload.get("metadata") or {},
        usage=usage,
        raw_response=payload.get("raw_response"),
        error=None if payload.get("error") is None else str(payload.get("error")),
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            # Flush the temporary file before replace so cache hits never read
            # a half-written JSON document after an interrupted process.
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


_SECRET_FRAGMENTS = ("key", "token", "secret", "password", "auth", "credential", "bearer")


def _is_secret_setting(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in _SECRET_FRAGMENTS)


def _safe_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    while ".." in safe:
        safe = safe.replace("..", ".")
    return safe.strip(".") or "none"


def _short_hash(value: str | None) -> str:
    return "nomsg" if value is None else value[:16]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
