"""OpenRouter chat-completions provider for live benchmark runs."""

from __future__ import annotations

import errno
import http.client
import json
import math
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from qceval.core.code import extract_code_from_text
from qceval.core.prompt_safety import OracleLeakError, assert_provider_messages_exclude_oracle
from qceval.models import ProviderRequest, ProviderResponse, TokenUsage
from qceval.production.campaign import configuration_id as expected_configuration_id
from qceval.production.campaign import configuration_identity_sha256

DEFAULT_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OUTPUT_TOKEN_PARAMETERS = frozenset({"max_tokens", "max_completion_tokens"})
OUTPUT_LIMIT_SOURCES = frozenset({"author_native", "benchmark_floor"})
ENDPOINT_CAP_STATUSES = frozenset({"catalog_numeric", "undisclosed_first_party_exception"})


class OpenRouterRouteVerificationError(RuntimeError):
    """A successful response did not prove the frozen singular route."""


class OpenRouterAPIResponseError(RuntimeError):
    """OpenRouter returned a structured API error in a successful HTTP response."""

    def __init__(self, *, code: int | None, message: str) -> None:
        self.code = code
        self.message = message
        label = "unknown" if code is None else str(code)
        super().__init__(f"OpenRouter API error {label}: {message}")


class OpenRouterProvider:
    """Generate candidate code through OpenRouter's chat-completions API.

    The provider sends each benchmark prompt as one user message and extracts
    Python source from the returned text.  HTTP and transport failures are
    encoded in :class:`qceval.models.ProviderResponse` instead of escaping to
    the runner.

    Args:
        api_key: OpenRouter API key.  Missing keys produce provider-failed
            responses.
        base_url: Chat-completions endpoint URL.
        timeout: HTTP request timeout in seconds.
        temperature: Sampling temperature sent to the model.
        reasoning_effort: Optional OpenRouter reasoning effort sent to models
            that expose configurable reasoning.
        reasoning_enabled: Optional enable switch for models that expose
            reasoning without configurable effort levels.
        output_limit_source: Evidence source for a pinned output ceiling;
            either an author-native limit or the benchmark floor.
        max_retries: Maximum retry attempts for transient HTTP errors. Set to
            0 to disable retry.
        retry_base_delay: Base delay in seconds for exponential backoff.
        retry_max_delay: Maximum delay cap in seconds.

    Attributes:
        name: Stable provider name used in output and cache keys.
        api_key: API key used for bearer authentication.
        base_url: Chat-completions endpoint URL.
        timeout: HTTP request timeout in seconds.
        temperature: Sampling temperature sent to the model.
    """

    name = "openrouter"

    def __init__(  # noqa: C901 - validates mutually exclusive routing controls
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_CHAT_COMPLETIONS_URL,
        timeout: float = 120.0,
        temperature: float | None = 0.2,
        reasoning_effort: str | None = None,
        reasoning_enabled: bool | None = None,
        endpoint_tag: str | None = None,
        max_output_tokens: int | None = None,
        output_limit_source: str | None = None,
        endpoint_cap_status: str | None = None,
        output_token_parameter: str | None = None,
        route_revision: str | None = None,
        configuration_id: str | None = None,
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
        pin_values = (
            endpoint_tag,
            max_output_tokens,
            output_limit_source,
            endpoint_cap_status,
            output_token_parameter,
            route_revision,
        )
        if any(value is not None for value in pin_values) and not all(value is not None for value in pin_values):
            raise ValueError(
                "endpoint_tag, max_output_tokens, output_limit_source, endpoint_cap_status, "
                "output_token_parameter and route_revision must be configured together"
            )
        if configuration_id is not None and endpoint_tag is None:
            raise ValueError("configuration_id requires a complete pinned endpoint route")
        if max_output_tokens is not None and (isinstance(max_output_tokens, bool) or max_output_tokens < 1):
            raise ValueError("max_output_tokens must be a positive integer")
        if output_token_parameter is not None and output_token_parameter not in OUTPUT_TOKEN_PARAMETERS:
            raise ValueError(f"unsupported output token parameter: {output_token_parameter}")
        if output_limit_source is not None and output_limit_source not in OUTPUT_LIMIT_SOURCES:
            raise ValueError(f"unsupported output limit source: {output_limit_source}")
        if endpoint_cap_status is not None and endpoint_cap_status not in ENDPOINT_CAP_STATUSES:
            raise ValueError(f"unsupported endpoint cap status: {endpoint_cap_status}")
        if endpoint_cap_status == "undisclosed_first_party_exception" and (
            endpoint_tag != "xai" or output_limit_source != "benchmark_floor" or max_output_tokens != 128_000
        ):
            raise ValueError("undisclosed endpoint-cap exception requires xai, benchmark_floor, and 128000 tokens")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.reasoning_enabled = reasoning_enabled
        self.endpoint_tag = endpoint_tag
        self.max_output_tokens = max_output_tokens
        self.output_limit_source = output_limit_source
        self.endpoint_cap_status = endpoint_cap_status
        self.output_token_parameter = output_token_parameter
        self.route_revision = route_revision
        self.configuration_id = configuration_id
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    # Frozen production retry taxonomy. Other HTTP statuses remain durable
    # infrastructure/configuration evidence but are never retried automatically.
    _RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})
    # Transient transport failures: connection-level errors plus truncated or
    # invalid response bodies from a completed request. Retrying re-sends the
    # same request and never counts as an extra benchmark sample or attempt.
    _RETRYABLE_TRANSPORT_ERRORS = (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        http.client.IncompleteRead,
        json.JSONDecodeError,
        UnicodeDecodeError,
    )

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate candidate code for one benchmark request.

        Args:
            request: Prompt and model selection.

        Returns:
            Provider response containing extracted source code or an error
            message.  The response includes token usage when OpenRouter reports
            it.
        """
        if not self.api_key:
            return ProviderResponse(code=None, model=request.model, error="openrouter api key is not configured")
        if not request.model:
            return ProviderResponse(code=None, model=request.model, error="openrouter model is not configured")
        if self.endpoint_cap_status == "undisclosed_first_party_exception" and request.model != "x-ai/grok-4.6":
            return self._failure(
                request,
                "undisclosed endpoint-cap exception is restricted to x-ai/grok-4.6",
                transport_retries=0,
            )
        try:
            payload = self._payload(request)
        except OracleLeakError as exc:
            # Oracle-isolation violations fail only this attempt as a typed
            # harness error instead of crashing the whole run.
            return ProviderResponse(
                code=None,
                model=request.model,
                metadata={"harness_error": "oracle_isolation"},
                error=f"harness safety violation: {exc}",
            )
        return self._generate_with_retries(payload, request)

    def _generate_with_retries(  # noqa: C901 - bounded transport retry state machine
        self, payload: dict[str, Any], request: ProviderRequest
    ) -> ProviderResponse:
        """Post one request with bounded retries for transient transport failures."""
        last_error: Exception | None = None
        route_verification_attempts: list[dict[str, Any]] = []
        provider_error_attempts: list[dict[str, Any]] = []
        attempt_history: list[dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            raw: dict[str, Any] | None = None
            started_at = self._timestamp()
            try:
                raw = self._post_json(payload)
                response = self._response_from_raw(
                    raw,
                    request,
                    transport_retries=attempt,
                    route_verification_attempts=route_verification_attempts,
                    provider_error_attempts=provider_error_attempts,
                )
                attempt_history.append(
                    self._attempt_evidence(
                        attempt=attempt,
                        started_at=started_at,
                        status=(
                            "nonretryable_infrastructure"
                            if response.metadata.get("infrastructure_error") is True
                            else "accepted_model_outcome"
                        ),
                        transient=False,
                        raw=raw,
                        response=response,
                    )
                )
                classification = str(
                    response.metadata.get("failure_classification")
                    or (
                        "endpoint_capacity_contract"
                        if response.metadata.get("endpoint_capacity_error") is not None
                        else "accepted_model_outcome"
                    )
                )
                return self._finalize_attempts(
                    response,
                    attempt_history,
                    retryable=False,
                    exhausted=False,
                    classification=classification,
                )
            except urllib.error.HTTPError as exc:
                error = self._http_error(exc)
                transient = exc.code in self._RETRYABLE_CODES
                attempt_history.append(
                    self._attempt_evidence(
                        attempt=attempt,
                        started_at=started_at,
                        status="transient_infrastructure" if transient else "permanent_infrastructure",
                        transient=transient,
                        error=error,
                        http_status=exc.code,
                    )
                )
                if not transient or attempt >= self.max_retries:
                    return self._failure(
                        request,
                        error,
                        transport_retries=attempt,
                        route_verification_attempts=route_verification_attempts,
                        provider_error_attempts=provider_error_attempts,
                        attempt_history=attempt_history,
                        retryable_infrastructure=transient,
                        retry_exhausted=transient and attempt >= self.max_retries,
                        failure_classification=("transient_http_exhausted" if transient else "permanent_http_error"),
                    )
                last_error = exc
                attempt_history[-1]["backoff_seconds"] = self._backoff(attempt, exc)
            except OpenRouterAPIResponseError as exc:
                provider_error_attempts.append(self._provider_error_evidence(raw, exc, attempt=attempt))
                transient = self._is_transient_provider_error(exc)
                attempt_history.append(
                    self._attempt_evidence(
                        attempt=attempt,
                        started_at=started_at,
                        status="transient_infrastructure" if transient else "permanent_infrastructure",
                        transient=transient,
                        error=str(exc),
                        http_status=exc.code,
                        raw=raw,
                    )
                )
                if not transient or attempt >= self.max_retries:
                    return self._failure(
                        request,
                        str(exc),
                        transport_retries=attempt,
                        provider_error_code=exc.code,
                        provider_error_message=exc.message,
                        provider_error_attempts=provider_error_attempts,
                        route_verification_attempts=route_verification_attempts,
                        raw_response=raw,
                        attempt_history=attempt_history,
                        retryable_infrastructure=transient,
                        retry_exhausted=transient and attempt >= self.max_retries,
                        failure_classification=(
                            "transient_provider_exhausted" if transient else "permanent_provider_error"
                        ),
                    )
                last_error = exc
                attempt_history[-1]["backoff_seconds"] = self._backoff(attempt)
            except OpenRouterRouteVerificationError as exc:
                route_verification_attempts.append(self._route_verification_evidence(raw, exc, attempt=attempt))
                attempt_history.append(
                    self._attempt_evidence(
                        attempt=attempt,
                        started_at=started_at,
                        status="provenance_failure",
                        transient=False,
                        error=str(exc),
                        raw=raw,
                    )
                )
                return self._failure(
                    request,
                    f"route verification failed: {exc}",
                    transport_retries=attempt,
                    route_verification_error=str(exc),
                    route_verification_attempts=route_verification_attempts,
                    provider_error_attempts=provider_error_attempts,
                    raw_response=raw,
                    attempt_history=attempt_history,
                    retryable_infrastructure=False,
                    retry_exhausted=False,
                    failure_classification="route_provenance_failure",
                )
            except self._RETRYABLE_TRANSPORT_ERRORS as exc:
                transient = self._is_transient_transport_error(exc)
                error = f"{type(exc).__name__}: {exc}"
                attempt_history.append(
                    self._attempt_evidence(
                        attempt=attempt,
                        started_at=started_at,
                        status="transient_infrastructure" if transient else "permanent_infrastructure",
                        transient=transient,
                        error=error,
                    )
                )
                if not transient or attempt >= self.max_retries:
                    return self._failure(
                        request,
                        error,
                        transport_retries=attempt,
                        route_verification_attempts=route_verification_attempts,
                        provider_error_attempts=provider_error_attempts,
                        attempt_history=attempt_history,
                        retryable_infrastructure=transient,
                        retry_exhausted=transient and attempt >= self.max_retries,
                        failure_classification=(
                            "transient_transport_exhausted" if transient else "permanent_transport_error"
                        ),
                    )
                last_error = exc
                attempt_history[-1]["backoff_seconds"] = self._backoff(attempt)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempt_history.append(
                    self._attempt_evidence(
                        attempt=attempt,
                        started_at=started_at,
                        status="permanent_harness_error",
                        transient=False,
                        error=error,
                    )
                )
                return self._failure(
                    request,
                    error,
                    transport_retries=attempt,
                    route_verification_attempts=route_verification_attempts,
                    provider_error_attempts=provider_error_attempts,
                    attempt_history=attempt_history,
                    retryable_infrastructure=False,
                    retry_exhausted=False,
                    failure_classification="permanent_harness_error",
                )
        return self._failure(
            request,
            f"exhausted {self.max_retries} retries: {last_error}",
            transport_retries=self.max_retries,
            route_verification_attempts=route_verification_attempts,
            provider_error_attempts=provider_error_attempts,
            attempt_history=attempt_history,
            retryable_infrastructure=True,
            retry_exhausted=True,
            failure_classification="transient_infrastructure_exhausted",
        )

    def _failure(
        self,
        request: ProviderRequest,
        error: str,
        *,
        transport_retries: int,
        route_verification_error: str | None = None,
        route_verification_attempts: Sequence[Mapping[str, Any]] = (),
        provider_error_code: int | None = None,
        provider_error_message: str | None = None,
        provider_error_attempts: Sequence[Mapping[str, Any]] = (),
        raw_response: Mapping[str, Any] | None = None,
        attempt_history: Sequence[Mapping[str, Any]] = (),
        retryable_infrastructure: bool = False,
        retry_exhausted: bool = False,
        failure_classification: str = "infrastructure_error",
    ) -> ProviderResponse:
        metadata: dict[str, Any] = {
            "provider": self.name,
            "infrastructure_error": True,
            "retryable_infrastructure": retryable_infrastructure,
            "retry_exhausted": retry_exhausted,
            "failure_classification": failure_classification,
            "infrastructure_attempts": len(attempt_history) or transport_retries + 1,
        }
        if transport_retries:
            metadata["transport_retries"] = transport_retries
        if self.endpoint_tag is not None:
            metadata["route"] = self._configured_route(route_verified=False, model_id=request.model)
        if route_verification_error is not None:
            metadata["route_verification_error"] = route_verification_error
        if route_verification_attempts:
            metadata["route_verification_attempts"] = [dict(item) for item in route_verification_attempts]
        if provider_error_message is not None:
            metadata["provider_error_code"] = provider_error_code
            metadata["provider_error_message"] = provider_error_message
        if provider_error_attempts:
            metadata["provider_error_attempts"] = [dict(item) for item in provider_error_attempts]
        if attempt_history:
            metadata["attempt_history"] = [dict(item) for item in attempt_history]
        if raw_response is not None and raw_response.get("id") is not None:
            metadata["generation_id"] = str(raw_response["id"])
        return ProviderResponse(
            code=None,
            model=request.model,
            metadata=metadata,
            raw_response=None if raw_response is None else dict(raw_response),
            error=error,
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

    def _backoff(self, attempt: int, exc: Exception | None = None) -> float:
        retry_after = self._parse_retry_after(exc) if isinstance(exc, urllib.error.HTTPError) else None
        if retry_after is not None:
            # Retry-After is authoritative and may exceed the configured cap.
            delay = retry_after
        else:
            delay = min(self.retry_base_delay * (2**attempt), self.retry_max_delay)
        jitter = random.uniform(0, delay * 0.25)
        actual = delay + jitter
        time.sleep(actual)
        return actual

    @staticmethod
    def _parse_retry_after(exc: urllib.error.HTTPError) -> float | None:
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header is None:
            return None
        try:
            value = float(header)
            return value if math.isfinite(value) and value >= 0 else None
        except ValueError:
            try:
                parsed = parsedate_to_datetime(header)
            except (TypeError, ValueError, OverflowError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())

    @classmethod
    def _is_transient_provider_error(cls, exc: OpenRouterAPIResponseError) -> bool:
        """Classify structured provider errors using the frozen retry taxonomy."""
        if exc.code in cls._RETRYABLE_CODES:
            return True
        if exc.code is not None:
            return False
        lowered = exc.message.casefold()
        return any(
            marker in lowered
            for marker in (
                "temporarily unavailable",
                "temporary provider unavailability",
                "temporarily rate-limited",
                "provider unavailable",
                "upstream unavailable",
                "provider overloaded",
            )
        )

    @staticmethod
    def _is_transient_transport_error(exc: Exception) -> bool:
        """Return whether a transport exception is safe to retry unchanged."""
        if isinstance(exc, urllib.error.URLError):
            reason = exc.reason
            if isinstance(reason, ssl.SSLCertVerificationError):
                return False
            if isinstance(reason, Exception):
                return OpenRouterProvider._is_transient_transport_error(reason)
            lowered = str(reason).casefold()
            return any(
                marker in lowered
                for marker in (
                    "connection reset",
                    "connection refused",
                    "network is unreachable",
                    "temporary",
                    "timed out",
                    "timeout",
                    "remote end closed",
                )
            )
        if isinstance(
            exc,
            (
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ),
        ):
            return True
        if isinstance(exc, OSError):
            return exc.errno in {
                errno.EAGAIN,
                errno.ECONNABORTED,
                errno.ECONNREFUSED,
                errno.ECONNRESET,
                errno.EHOSTUNREACH,
                errno.ENETDOWN,
                errno.ENETRESET,
                errno.ENETUNREACH,
                errno.EPIPE,
                errno.ETIMEDOUT,
                socket.EAI_AGAIN,
            }
        return False

    def _attempt_evidence(  # noqa: C901 - normalizes heterogeneous provider evidence
        self,
        *,
        attempt: int,
        started_at: str,
        status: str,
        transient: bool,
        error: str | None = None,
        http_status: int | None = None,
        raw: Mapping[str, Any] | None = None,
        response: ProviderResponse | None = None,
    ) -> dict[str, Any]:
        """Build one append-only-ledger-ready physical-attempt event."""
        evidence: dict[str, Any] = {
            "attempt_number": attempt + 1,
            "started_at_utc": started_at,
            "finished_at_utc": self._timestamp(),
            "status": status,
            "transient": transient,
        }
        if error is not None:
            evidence["error"] = error
        if http_status is not None:
            evidence["http_status"] = http_status
        generation_id = None if raw is None else raw.get("id")
        if generation_id is not None:
            evidence["generation_id"] = str(generation_id)
        if response is not None:
            if response.usage is not None:
                evidence["usage"] = response.usage.to_dict()
            finish_reason = response.metadata.get("finish_reason")
            if finish_reason is not None:
                evidence["finish_reason"] = str(finish_reason)
            route = response.metadata.get("route")
            if isinstance(route, Mapping):
                evidence["route_verified"] = route.get("route_verified") is True
            if response.error is not None and error is None:
                evidence["error"] = response.error
        elif isinstance(raw, Mapping) and isinstance(raw.get("usage"), Mapping):
            evidence["usage"] = self._usage(raw["usage"]).to_dict()
        return evidence

    def _finalize_attempts(
        self,
        response: ProviderResponse,
        attempt_history: Sequence[Mapping[str, Any]],
        *,
        retryable: bool,
        exhausted: bool,
        classification: str,
    ) -> ProviderResponse:
        """Attach the complete physical-attempt history to a provider response."""
        metadata = dict(response.metadata)
        metadata.update(
            {
                "attempt_history": [dict(item) for item in attempt_history],
                "infrastructure_attempts": len(attempt_history),
                "retryable_infrastructure": retryable,
                "retry_exhausted": exhausted,
                "failure_classification": classification,
            }
        )
        return replace(response, metadata=metadata)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _payload(self, request: ProviderRequest) -> dict[str, Any]:
        messages = (
            [{"role": message.role, "content": message.content} for message in request.messages]
            if request.messages
            else [{"role": "user", "content": request.prompt}]
        )
        # Harness-authored turns must never carry contracts, targets, or other
        # oracles. Assistant turns echo prior candidate code and are exempt.
        assert_provider_messages_exclude_oracle(messages)
        reasoning: dict[str, Any] = {"exclude": True}
        if self.reasoning_effort is not None:
            reasoning["effort"] = self.reasoning_effort
        elif self.reasoning_enabled is not None:
            reasoning["enabled"] = self.reasoning_enabled
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            # Excluding reasoning payloads keeps raw responses smaller and lets
            # code extraction focus on final answer text.
            "reasoning": reasoning,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.endpoint_tag is not None:
            assert self.output_token_parameter is not None
            assert self.max_output_tokens is not None
            payload["provider"] = {
                "only": [self.endpoint_tag],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            payload[self.output_token_parameter] = self.max_output_tokens
        return payload

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Metadata": "enabled",
                # OpenRouter intentionally strips router metadata from response
                # cache replays. Production route proof therefore requires a
                # fresh router pass even when a preset enables response caching.
                "X-OpenRouter-Cache": "false",
            },
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _response_from_raw(  # noqa: C901 - fail-closed provider response validation
        self,
        raw: dict[str, Any],
        request: ProviderRequest,
        *,
        transport_retries: int = 0,
        route_verification_attempts: Sequence[Mapping[str, Any]] = (),
        provider_error_attempts: Sequence[Mapping[str, Any]] = (),
    ) -> ProviderResponse:
        provider_error = self._provider_error(raw)
        if provider_error is not None:
            provider_error_code, provider_error_message = provider_error
            raise OpenRouterAPIResponseError(code=provider_error_code, message=provider_error_message)
        text = ""
        finish_reason: str | None = None
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            raw_finish_reason = choices[0].get("finish_reason")
            if raw_finish_reason is not None:
                finish_reason = str(raw_finish_reason)
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                text = message["content"]
        code = extract_code_from_text(text, request.entry_point)
        metadata: dict[str, Any] = {"provider": self.name}
        if transport_retries:
            metadata["transport_retries"] = transport_retries
        if self.reasoning_effort is not None:
            metadata["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_enabled is not None:
            metadata["reasoning_enabled"] = self.reasoning_enabled
        if route_verification_attempts:
            metadata["route_verification_attempts"] = [dict(item) for item in route_verification_attempts]
        if provider_error_attempts:
            metadata["provider_error_attempts"] = [dict(item) for item in provider_error_attempts]
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        generation_id = raw.get("id")
        if generation_id is not None:
            metadata["generation_id"] = str(generation_id)
        if self.endpoint_tag is not None:
            metadata["route"] = self._verified_route(raw, request)
        usage = self._usage(raw.get("usage") or {})
        if self.endpoint_tag is not None and finish_reason == "length":
            completion_tokens = usage.completion_tokens
            if self.max_output_tokens is None:
                raise AssertionError("pinned requests must configure max_output_tokens")
            if completion_tokens is None or completion_tokens < self.max_output_tokens:
                reported = "missing" if completion_tokens is None else str(completion_tokens)
                error = (
                    "endpoint output-capacity contract failed: finish_reason=length with "
                    f"{reported} reported completion tokens, below requested ceiling {self.max_output_tokens}"
                )
                metadata["infrastructure_error"] = True
                metadata["endpoint_capacity_error"] = error
                return ProviderResponse(
                    code=None,
                    model=str(raw.get("model") or request.model or ""),
                    metadata=metadata,
                    usage=usage,
                    raw_response=raw,
                    error=error,
                )
        if self.endpoint_tag is not None and usage.cost_usd is None and self._is_explicit_policy_refusal(raw):
            refusal = self._policy_refusal_text(raw)
            metadata.update(
                {
                    "failure_classification": "provider_policy_refusal",
                    "campaign_resolution": {
                        "schema_version": "qceval.policy_refusal_resolution.v1",
                        "disposition": "candidate_less_provider_failure",
                        "reason": "provider_policy_refusal",
                        "provider_reported_usage_present": False,
                        "accounting_source": "zero_normalization_for_unbilled_policy_refusal",
                        "route_evidence_source": "raw_response.openrouter_metadata",
                        "refusal_evidence_source": "raw_response.choices[0].message.refusal",
                    },
                }
            )
            return ProviderResponse(
                code=None,
                model=str(raw.get("model") or request.model or ""),
                metadata=metadata,
                usage=TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    reasoning_tokens=0,
                    cached_tokens=0,
                    cost_usd=0.0,
                ),
                raw_response=raw,
                error=refusal,
            )
        if self.endpoint_tag is not None and usage.cost_usd is None:
            raise OpenRouterRouteVerificationError("response omitted provider-reported cost")
        return ProviderResponse(
            code=code,
            model=str(raw.get("model") or request.model or ""),
            metadata=metadata,
            usage=usage,
            raw_response=raw,
        )

    @staticmethod
    def _policy_refusal_text(raw: Mapping[str, Any]) -> str | None:
        """Return an explicit policy-refusal message from a completion response."""
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            return None
        choice = choices[0]
        if choice.get("finish_reason") != "content_filter":
            return None
        message = choice.get("message")
        if not isinstance(message, Mapping):
            return None
        refusal = message.get("refusal")
        return refusal.strip() if isinstance(refusal, str) and refusal.strip() else None

    @classmethod
    def _is_explicit_policy_refusal(cls, raw: Mapping[str, Any]) -> bool:
        """Return whether the provider supplied direct policy-refusal evidence."""
        return cls._policy_refusal_text(raw) is not None

    def _route_verification_evidence(
        self,
        raw: Mapping[str, Any] | None,
        error: OpenRouterRouteVerificationError,
        *,
        attempt: int,
    ) -> dict[str, Any]:
        """Keep compact provider evidence for a rejected, parseable response."""
        evidence: dict[str, Any] = {
            "attempt_number": attempt + 1,
            "error": str(error),
        }
        if raw is None:
            return evidence
        if raw.get("id") is not None:
            evidence["generation_id"] = str(raw["id"])
        if raw.get("model") is not None:
            evidence["model"] = str(raw["model"])
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            finish_reason = choices[0].get("finish_reason")
            if finish_reason is not None:
                evidence["finish_reason"] = str(finish_reason)
        usage = raw.get("usage")
        if isinstance(usage, dict):
            evidence["usage"] = self._usage(usage).to_dict()
        router_metadata = raw.get("openrouter_metadata")
        evidence["openrouter_metadata_present"] = isinstance(router_metadata, Mapping)
        if isinstance(router_metadata, Mapping):
            evidence["openrouter_metadata"] = dict(router_metadata)
        return evidence

    @staticmethod
    def _provider_error(raw: Mapping[str, Any]) -> tuple[int | None, str] | None:
        """Parse a structured OpenRouter error envelope returned with HTTP 200."""
        error = raw.get("error")
        if not isinstance(error, Mapping):
            choices = raw.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                # OpenRouter may surface an upstream provider failure inside
                # the first choice while still returning HTTP 200. Treat that
                # envelope exactly like a top-level structured error so 5xx
                # failures use the bounded transient retry path instead of
                # being misclassified as missing-cost provenance failures.
                error = choices[0].get("error")
        if not isinstance(error, Mapping):
            return None
        raw_code = error.get("code")
        code = raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool) else None
        raw_message = error.get("message")
        message = str(raw_message) if raw_message is not None else "provider returned an unspecified error"
        return code, message

    @staticmethod
    def _provider_error_evidence(
        raw: Mapping[str, Any] | None,
        error: OpenRouterAPIResponseError,
        *,
        attempt: int,
    ) -> dict[str, Any]:
        """Keep compact evidence for a structured provider error response."""
        evidence: dict[str, Any] = {
            "attempt_number": attempt + 1,
            "code": error.code,
            "message": error.message,
        }
        if raw is None:
            return evidence
        if raw.get("id") is not None:
            evidence["generation_id"] = str(raw["id"])
        if raw.get("model") is not None:
            evidence["model"] = str(raw["model"])
        evidence["openrouter_metadata_present"] = isinstance(raw.get("openrouter_metadata"), Mapping)
        return evidence

    def _configured_route(self, *, route_verified: bool, model_id: str | None) -> dict[str, Any]:
        """Return immutable request-side route provenance."""
        route = {
            "configuration_id": self.configuration_id,
            "endpoint_tag": self.endpoint_tag,
            "max_output_tokens": self.max_output_tokens,
            "output_limit_source": self.output_limit_source,
            "endpoint_cap_status": self.endpoint_cap_status,
            "output_token_parameter": self.output_token_parameter,
            "route_revision": self.route_revision,
            "temperature": self.temperature,
            "allow_fallbacks": False,
            "require_parameters": True,
            "response_cache_disabled": True,
            "route_verified": route_verified,
        }
        if self.configuration_id is not None:
            reasoning_setting = self.reasoning_effort
            if reasoning_setting is None and self.reasoning_enabled is True:
                reasoning_setting = "enabled"
            if reasoning_setting is None or not model_id:
                raise OpenRouterRouteVerificationError(
                    "pinned campaign configuration omitted model or reasoning effort"
                )
            if self.configuration_id != expected_configuration_id(model_id, reasoning_setting):
                raise OpenRouterRouteVerificationError("configuration_id does not match the requested model and effort")
            route["configuration_identity_sha256"] = configuration_identity_sha256(
                model_id=model_id,
                effort=reasoning_setting,
                endpoint_tag=str(self.endpoint_tag),
                configured_output_tokens=int(self.max_output_tokens or 0),
                output_token_parameter=str(self.output_token_parameter),
                temperature_behavior="explicit_zero" if self.temperature is not None else "not_exposed",
                route_revision=str(self.route_revision),
            )
        return route

    def _verified_route(  # noqa: C901 - fail-closed route provenance validation
        self, raw: Mapping[str, Any], request: ProviderRequest
    ) -> dict[str, Any]:
        router = raw.get("openrouter_metadata")
        if not isinstance(router, Mapping):
            raise OpenRouterRouteVerificationError("response omitted openrouter_metadata")
        if router.get("requested") != request.model:
            raise OpenRouterRouteVerificationError(
                f"requested model {router.get('requested')!r} does not match {request.model!r}"
            )
        if router.get("attempt") != 1:
            raise OpenRouterRouteVerificationError(f"router attempt must be 1, got {router.get('attempt')!r}")
        if router.get("strategy") != "direct":
            raise OpenRouterRouteVerificationError(f"router strategy must be 'direct', got {router.get('strategy')!r}")

        endpoints = router.get("endpoints")
        if not isinstance(endpoints, Mapping):
            raise OpenRouterRouteVerificationError("router endpoint metadata is malformed")
        catalog_total = endpoints.get("total")
        if isinstance(catalog_total, bool) or not isinstance(catalog_total, int) or catalog_total < 1:
            raise OpenRouterRouteVerificationError(f"router endpoint catalog total is invalid: {catalog_total!r}")
        available = endpoints.get("available")
        if not isinstance(available, list) or len(available) != 1 or not isinstance(available[0], Mapping):
            raise OpenRouterRouteVerificationError("router did not report exactly one available endpoint")
        selected = available[0]
        if selected.get("selected") is not True:
            raise OpenRouterRouteVerificationError("the singular endpoint was not marked selected")

        attempts = router.get("attempts")
        if attempts is not None:
            if not isinstance(attempts, list) or len(attempts) != 1 or not isinstance(attempts[0], Mapping):
                raise OpenRouterRouteVerificationError("router reported a fallback attempt chain")
            if attempts[0].get("status") != 200:
                raise OpenRouterRouteVerificationError(
                    f"the singular router attempt did not return 200: {attempts[0].get('status')!r}"
                )

        pipeline = router.get("pipeline") or []
        if not isinstance(pipeline, list):
            raise OpenRouterRouteVerificationError("router pipeline metadata is malformed")
        if any(isinstance(stage, Mapping) and stage.get("type") == "context_compression" for stage in pipeline):
            raise OpenRouterRouteVerificationError("router applied context compression")

        route = self._configured_route(route_verified=True, model_id=request.model)
        route.update(
            {
                "selected_provider": selected.get("provider"),
                "selected_model": selected.get("model"),
                "router_attempt": router.get("attempt"),
                "router_strategy": router.get("strategy"),
                "router_endpoint_catalog_total": catalog_total,
                "router_available_endpoint_count": len(available),
                "router_metadata": dict(router),
            }
        )
        return route

    @staticmethod
    def _usage(raw_usage: dict[str, Any]) -> TokenUsage:
        completion_details = raw_usage.get("completion_tokens_details") or {}
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        return TokenUsage(
            prompt_tokens=raw_usage.get("prompt_tokens"),
            completion_tokens=raw_usage.get("completion_tokens"),
            total_tokens=raw_usage.get("total_tokens"),
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            cached_tokens=prompt_details.get("cached_tokens"),
            cost_usd=OpenRouterProvider._cost_usd(raw_usage.get("cost")),
        )

    @staticmethod
    def _cost_usd(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            cost = float(value)
        except (TypeError, ValueError):
            return None
        return cost if math.isfinite(cost) and cost >= 0.0 else None

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> str:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return f"openrouter HTTP {exc.code}: {body}"
