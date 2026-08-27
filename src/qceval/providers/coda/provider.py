"""Coda Build agent provider implementation.

Implements :class:`CodaProvider`, which sends benchmark prompts to
Conductor Quantum's Coda agents endpoint over HTTPS, parses the
resulting SSE stream, and extracts generated source code.  The provider
supports configurable retry with exponential backoff, ``Retry-After``
header support, and both ``build`` and ``learn`` agent modes.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from qceval.core.prompt_safety import assert_provider_messages_exclude_oracle
from qceval.models import ProviderMessage, ProviderRequest, ProviderResponse
from qceval.providers.coda.events import CodaEventStream, extract_coda_generated_code, parse_coda_events

DEFAULT_CODA_AGENTS_URL = "https://api.conductorquantum.com/v0/coda/agents"
DEFAULT_CODA_TIMEOUT_SECONDS = 900.0
_LOGGER = logging.getLogger(__name__)
_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_ERROR_PATTERNS = (
    "ValidationException",
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
    "ModelStreamErrorException",
)


@dataclass(frozen=True)
class _CodaHttpResponse:
    """Raw HTTP response from the Coda agents endpoint.

    Attributes:
        lines: Response body lines (text or bytes) for SSE parsing.
        headers: Lower-cased response headers.
    """

    lines: tuple[str | bytes, ...]
    headers: Mapping[str, str]


@dataclass(frozen=True)
class _GenerateAttempt:
    """Result of one Coda generation attempt."""

    response: ProviderResponse | None
    retry_error: str | Exception | None = None
    backoff_error: Exception | None = None


class CodaProvider:
    """Generate code through Coda's Build agent API.

    Args:
        api_key: Coda API key. Missing keys produce provider-failed responses.
        agents_url: Full Coda agents endpoint URL.
        timeout: HTTP request timeout in seconds.
        mode: Coda agent mode, either ``"build"`` or ``"learn"``.
        fast: Whether to request Coda's fast mode.
        prefer_structured_response: Whether structured response code fields
            are preferred when they define the task entry point.
        max_retries: Maximum retry attempts for transient HTTP errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
        retry_max_delay: Maximum delay cap in seconds.

    Attributes:
        name: Stable provider name used in output and cache keys.
        api_key: API key used for bearer authentication.
        agents_url: Full Coda agents endpoint URL.
        timeout: HTTP request timeout in seconds.
        mode: Coda agent mode sent in request bodies.
        fast: Whether fast mode is sent in request bodies.
        prefer_structured_response: Whether structured response code fields are
            preferred during extraction.

    Raises:
        ValueError: If retry settings or mode are invalid.
    """

    name = "coda"

    def __init__(
        self,
        *,
        api_key: str | None,
        agents_url: str = DEFAULT_CODA_AGENTS_URL,
        timeout: float = DEFAULT_CODA_TIMEOUT_SECONDS,
        mode: str = "build",
        fast: bool = False,
        prefer_structured_response: bool = False,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay must be >= 0")
        if retry_max_delay <= 0:
            raise ValueError("retry_max_delay must be > 0")
        if mode not in {"build", "learn"}:
            raise ValueError("coda mode must be build or learn")
        self.api_key = api_key
        self.agents_url = agents_url
        self.timeout = timeout
        self.mode = mode
        self.fast = fast
        self.prefer_structured_response = prefer_structured_response
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate code for one benchmark request.

        Args:
            request: Prompt and task metadata.

        Returns:
            Provider response containing extracted source code or an error
            message.
        """
        if not self.api_key:
            return ProviderResponse(code=None, model=request.model, error="coda api key is not configured")
        payload, error = self._validated_payload(request)
        if error is not None:
            return error
        return self._generate_with_retries(request, payload)

    def _validated_payload(self, request: ProviderRequest) -> tuple[dict[str, Any], ProviderResponse | None]:
        try:
            return self._payload(request), None
        except ValueError as exc:
            return {}, ProviderResponse(code=None, model=request.model, error=str(exc))

    def _generate_with_retries(self, request: ProviderRequest, payload: dict[str, Any]) -> ProviderResponse:
        last_error: str | Exception | None = None
        for attempt in range(self.max_retries + 1):
            result = self._generate_attempt(request, payload, attempt)
            if result.response is not None:
                return result.response
            last_error = result.retry_error
            self._backoff(attempt, result.backoff_error)
        return ProviderResponse(
            code=None,
            model=request.model,
            error=f"exhausted {self.max_retries} retries: {last_error}",
        )

    def _generate_attempt(self, request: ProviderRequest, payload: dict[str, Any], attempt: int) -> _GenerateAttempt:
        try:
            raw = self._post(payload)
            response = self._response_from_raw(raw, request)
            return self._response_attempt(response, attempt)
        except urllib.error.HTTPError as exc:
            return self._http_error_attempt(exc, request, attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return self._transport_error_attempt(exc, request, attempt)
        except Exception as exc:
            _LOGGER.exception("Unexpected Coda generation failure.")
            return _GenerateAttempt(
                response=ProviderResponse(code=None, model=request.model, error=f"{type(exc).__name__}: {exc}")
            )

    def _response_attempt(self, response: ProviderResponse, attempt: int) -> _GenerateAttempt:
        if self._should_retry_response(response, attempt):
            return _GenerateAttempt(response=None, retry_error=response.error)
        return _GenerateAttempt(response=response)

    def _http_error_attempt(
        self, exc: urllib.error.HTTPError, request: ProviderRequest, attempt: int
    ) -> _GenerateAttempt:
        if exc.code in _RETRYABLE_HTTP_STATUS_CODES and attempt < self.max_retries:
            return _GenerateAttempt(response=None, retry_error=exc, backoff_error=exc)
        return _GenerateAttempt(response=ProviderResponse(code=None, model=request.model, error=self._http_error(exc)))

    def _transport_error_attempt(
        self, exc: urllib.error.URLError | TimeoutError | OSError, request: ProviderRequest, attempt: int
    ) -> _GenerateAttempt:
        if attempt < self.max_retries:
            return _GenerateAttempt(response=None, retry_error=exc)
        error = f"{type(exc).__name__}: {exc}"
        return _GenerateAttempt(response=ProviderResponse(code=None, model=request.model, error=error))

    def _should_retry_response(self, response: ProviderResponse, attempt: int) -> bool:
        return (
            response.code is None
            and response.error is not None
            and self._is_retryable_error(response.error)
            and attempt < self.max_retries
        )

    def generate_many(self, requests: Sequence[ProviderRequest]) -> list[ProviderResponse]:
        """Generate responses for ordered requests using worker threads.

        Args:
            requests: Ordered provider requests.

        Returns:
            Ordered provider responses with the same length as ``requests``.
        """
        with ThreadPoolExecutor(max_workers=len(requests) or 1) as pool:
            return list(pool.map(self.generate, requests))

    def _payload(self, request: ProviderRequest) -> dict[str, Any]:
        return {"messages": self._messages(request), "mode": self.mode, "fast": self.fast}

    def _messages(self, request: ProviderRequest) -> list[dict[str, str]]:
        """Build the message list for a Coda request payload.

        Rejects ``system`` messages (the Coda API only supports ``user``
        and ``assistant``), strips trailing empty assistant turns, and
        falls back to the raw prompt when no messages remain.
        """
        source = request.messages or (ProviderMessage(role="user", content=request.prompt),)
        messages: list[dict[str, str]] = []
        for message in source:
            if message.role not in {"user", "assistant"}:
                raise ValueError(f"coda only supports user and assistant messages; got {message.role}")
            messages.append({"role": message.role, "content": message.content})
        # Trailing empty assistant turns are artifacts of multi-turn
        # feedback that would confuse the Coda API.
        while messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"].strip():
            messages.pop()
        messages = messages or [{"role": "user", "content": request.prompt}]
        # Live providers must never receive contracts, targets, or other oracles.
        assert_provider_messages_exclude_oracle(messages)
        return messages

    def _post(self, payload: dict[str, Any]) -> _CodaHttpResponse:
        data = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self.agents_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
            return _CodaHttpResponse(lines=tuple(response.readlines()), headers=headers)

    def _response_from_raw(self, raw: _CodaHttpResponse, request: ProviderRequest) -> ProviderResponse:
        """Parse a raw HTTP response into a ``ProviderResponse``.

        Terminal errors and cancellation events are detected before code
        extraction so diagnostics are preserved even when no code is available.
        """
        stream = parse_coda_events(raw.lines)
        raw_response = self._raw_response(stream, raw.headers)
        metadata = self._metadata(stream, raw.headers, extracted_source=None)
        if stream.terminal_error is not None:
            return ProviderResponse(
                code=None,
                model=request.model,
                metadata=metadata,
                raw_response=raw_response,
                error=f"coda event error: {stream.terminal_error}",
            )
        if stream.cancelled:
            return ProviderResponse(
                code=None,
                model=request.model,
                metadata=metadata,
                raw_response=raw_response,
                error="coda event cancelled",
            )
        extraction = extract_coda_generated_code(
            stream,
            entry_point=request.entry_point,
            framework=request.framework,
            prefer_structured_response=self.prefer_structured_response,
        )
        metadata = self._metadata(stream, raw.headers, extracted_source=extraction.source)
        if extraction.code is None:
            return ProviderResponse(
                code=None,
                model=request.model,
                metadata=metadata,
                raw_response=raw_response,
                error="coda returned no extractable code or assistant text",
            )
        return ProviderResponse(
            code=extraction.code,
            model=request.model,
            metadata=metadata,
            usage=None,
            raw_response=raw_response,
        )

    def _metadata(
        self,
        stream: CodaEventStream,
        headers: Mapping[str, str],
        *,
        extracted_source: str | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": self.name,
            "mode": self.mode,
            "fast": self.fast,
            "event_count": len(stream.events),
            "event_types": dict(stream.event_types),
            "extracted_source": extracted_source,
        }
        metadata.update(_response_header_metadata(headers))
        return metadata

    @staticmethod
    def _raw_response(stream: CodaEventStream, headers: Mapping[str, str]) -> dict[str, Any]:
        return {"events": list(stream.compact_events), "headers": _response_header_metadata(headers)}

    def _backoff(self, attempt: int, exc: Exception | None = None) -> None:
        """Sleep before retrying, respecting ``Retry-After`` when present."""
        retry_after = self._parse_retry_after(exc) if isinstance(exc, urllib.error.HTTPError) else None
        if retry_after is not None:
            delay = min(retry_after, self.retry_max_delay)
        else:
            delay = min(self.retry_base_delay * (2**attempt), self.retry_max_delay)
        # 25 % jitter avoids thundering-herd retries.
        jitter = random.uniform(0, delay * 0.25)
        time.sleep(delay + jitter)

    def _is_retryable_error(self, error: str) -> bool:
        """Check whether a response error string indicates a transient failure.

        Matches known Bedrock/LLM exception patterns that are safe to retry
        because the request itself is valid but the upstream service had a
        transient issue.
        """
        return any(pattern in error for pattern in _RETRYABLE_ERROR_PATTERNS)

    @staticmethod
    def _parse_retry_after(exc: urllib.error.HTTPError) -> float | None:
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> str:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return f"coda HTTP {exc.code}: {body}"


def _response_header_metadata(headers: Mapping[str, str]) -> dict[str, str]:
    """Extract Coda-specific ``x-run-id`` and ``x-thread-id`` response headers.

    Header names are converted to snake_case metadata keys (e.g.
    ``x-run-id`` becomes ``run_id``).
    """
    metadata: dict[str, str] = {}
    for key in ("x-run-id", "x-thread-id"):
        value = headers.get(key)
        if value is not None:
            metadata[key.removeprefix("x-").replace("-", "_")] = value
    return metadata
