from __future__ import annotations

import http.client
import io
import json
import time
import urllib.error
from email.message import Message

import pytest

from qceval.models import ProviderRequest
from qceval.providers.openrouter import OpenRouterProvider


def test_openrouter_retries_429() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key", max_retries=2, retry_base_delay=0.01)
    calls = []

    def post_json(payload):
        calls.append(1)
        if len(calls) < 2:
            raise _make_http_error(429, b"rate limited")
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert len(calls) == 2


@pytest.mark.parametrize("status_code", range(520, 528))
def test_openrouter_does_not_retry_unfrozen_cloudflare_statuses(status_code: int) -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key", max_retries=2, retry_base_delay=0.01)
    calls = []

    def post_json(payload):
        calls.append(1)
        if len(calls) < 2:
            raise _make_http_error(status_code, b"upstream origin failure")
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert len(calls) == 1
    assert response.metadata["retryable_infrastructure"] is False
    assert response.metadata["failure_classification"] == "permanent_http_error"


def test_openrouter_does_not_retry_400() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key", max_retries=3, retry_base_delay=0.01)
    calls = []

    def post_json(payload):
        calls.append(1)
        raise _make_http_error(400, b"bad request")

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert len(calls) == 1


def test_openrouter_respects_retry_after() -> None:
    # Arrange
    headers = Message()
    headers["Retry-After"] = "0.05"
    provider = OpenRouterProvider(api_key="key", max_retries=1, retry_base_delay=0.01)
    timestamps = []

    def post_json(payload):
        timestamps.append(time.monotonic())
        if len(timestamps) < 2:
            raise urllib.error.HTTPError("url", 429, "msg", headers, io.BytesIO(b"rate limited"))
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert timestamps[1] - timestamps[0] >= 0.05


def test_openrouter_retries_three_times_with_exponential_backoff(monkeypatch) -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key", max_retries=3, retry_base_delay=1.0)
    calls = []
    delays = []

    def post_json(payload):
        calls.append(payload)
        raise urllib.error.URLError("temporary outage")

    provider._post_json = post_json  # type: ignore[method-assign]
    monkeypatch.setattr("qceval.providers.openrouter.random.uniform", lambda _low, _high: 0.0)
    monkeypatch.setattr("qceval.providers.openrouter.time.sleep", delays.append)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert len(calls) == 4
    assert delays == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_openrouter_frozen_transient_http_taxonomy(status_code: int, monkeypatch) -> None:
    provider = OpenRouterProvider(api_key="key", max_retries=5, retry_base_delay=1.0)
    calls = []
    delays = []
    provider._post_json = lambda payload: (_ for _ in ()).throw(  # type: ignore[method-assign]
        _make_http_error(status_code, b"temporary")
    )
    monkeypatch.setattr("qceval.providers.openrouter.random.uniform", lambda _low, _high: 0.0)
    monkeypatch.setattr("qceval.providers.openrouter.time.sleep", delays.append)
    original = provider._post_json

    def counted(payload):
        calls.append(payload)
        return original(payload)

    provider._post_json = counted  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    response = provider.generate(request)

    assert len(calls) == 6
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert response.metadata["retryable_infrastructure"] is True
    assert response.metadata["retry_exhausted"] is True
    assert response.metadata["infrastructure_attempts"] == 6
    assert len(response.metadata["attempt_history"]) == 6


def test_openrouter_retry_after_overrides_lower_delay_cap(monkeypatch) -> None:
    headers = Message()
    headers["Retry-After"] = "120"
    error = urllib.error.HTTPError("url", 429, "msg", headers, io.BytesIO(b"rate limited"))
    provider = OpenRouterProvider(api_key="key", max_retries=1, retry_base_delay=1, retry_max_delay=60)
    delays = []
    monkeypatch.setattr("qceval.providers.openrouter.random.uniform", lambda _low, _high: 0.0)
    monkeypatch.setattr("qceval.providers.openrouter.time.sleep", delays.append)

    actual = provider._backoff(0, error)

    assert actual == 120.0
    assert delays == [120.0]


def _make_decode_error() -> json.JSONDecodeError:
    try:
        json.loads('{"truncated": ')
    except json.JSONDecodeError as exc:
        return exc
    raise AssertionError("expected JSONDecodeError")


@pytest.mark.parametrize(
    "transient_error",
    [
        http.client.IncompleteRead(b"partial body"),
        _make_decode_error(),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
    ids=["incomplete_read", "json_decode", "unicode_decode"],
)
def test_openrouter_retries_truncated_or_invalid_response_bodies(transient_error: Exception) -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key", max_retries=2, retry_base_delay=0.01)
    calls = []

    def post_json(payload):
        calls.append(1)
        if len(calls) < 2:
            raise transient_error
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert len(calls) == 2
    assert response.metadata["transport_retries"] == 1


def test_openrouter_exhausted_truncated_responses_fail_with_retry_metadata(monkeypatch) -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key", max_retries=2, retry_base_delay=0.01)
    calls = []

    def post_json(payload):
        calls.append(1)
        raise http.client.IncompleteRead(b"partial body")

    provider._post_json = post_json  # type: ignore[method-assign]
    monkeypatch.setattr("qceval.providers.openrouter.time.sleep", lambda _delay: None)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert len(calls) == 3
    assert "IncompleteRead" in (response.error or "")
    assert response.metadata["transport_retries"] == 2


def test_openrouter_does_not_retry_unexpected_errors() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key", max_retries=3, retry_base_delay=0.01)
    calls = []

    def post_json(payload):
        calls.append(1)
        raise RuntimeError("programming error")

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert len(calls) == 1
    assert "RuntimeError" in (response.error or "")


def test_openrouter_does_not_retry_permanent_os_error() -> None:
    provider = OpenRouterProvider(api_key="key", max_retries=5, retry_base_delay=0)
    calls = []

    def post_json(payload):
        calls.append(payload)
        raise FileNotFoundError("permanent local configuration error")

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    response = provider.generate(request)

    assert len(calls) == 1
    assert response.metadata["retryable_infrastructure"] is False
    assert response.metadata["retry_exhausted"] is False


def test_openrouter_does_not_retry_permanent_url_configuration_error() -> None:
    provider = OpenRouterProvider(api_key="key", max_retries=5, retry_base_delay=0)
    calls = []

    def post_json(payload):
        calls.append(payload)
        raise urllib.error.URLError("unknown url type")

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    response = provider.generate(request)

    assert len(calls) == 1
    assert response.metadata["retryable_infrastructure"] is False


def test_openrouter_successful_first_attempt_has_no_retry_metadata() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key", max_retries=2, retry_base_delay=0.01)

    def post_json(payload):
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert "transport_retries" not in response.metadata


def test_openrouter_formats_http_error() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key")
    error = urllib.error.HTTPError("url", 429, "Too Many", {}, io.BytesIO(b"rate limited"))

    # Act
    message = provider._http_error(error)

    # Assert
    assert message == "openrouter HTTP 429: rate limited"


def _make_http_error(code: int, body: bytes = b"error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "msg", {}, io.BytesIO(body))
