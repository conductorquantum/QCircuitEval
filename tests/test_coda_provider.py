"""Tests for Coda provider request and response mapping."""

from __future__ import annotations

import io
import urllib.error
from typing import Any

from qceval.models import ProviderMessage, ProviderRequest
from qceval.providers.coda.provider import CodaProvider, _CodaHttpResponse


def test_coda_requires_api_key() -> None:
    provider = CodaProvider(api_key=None)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="coda/build")
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "coda api key is not configured"


def test_coda_rejects_invalid_constructor_args() -> None:
    errors: list[str] = []
    for kwargs in (
        {"api_key": "key", "max_retries": -1},
        {"api_key": "key", "retry_base_delay": -0.1},
        {"api_key": "key", "retry_max_delay": 0.0},
        {"api_key": "key", "mode": "chat"},
    ):
        try:
            CodaProvider(**kwargs)
        except ValueError as exc:
            errors.append(str(exc))
    assert errors == [
        "max_retries must be >= 0",
        "retry_base_delay must be >= 0",
        "retry_max_delay must be > 0",
        "coda mode must be build or learn",
    ]


def test_coda_rejects_system_role() -> None:
    provider = CodaProvider(api_key="key")
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="ignored",
        entry_point="answer",
        model="coda/build",
        messages=(ProviderMessage(role="system", content="nope"),),
    )
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "coda only supports user and assistant messages; got system"


def test_coda_one_shot_payload_uses_direct_agents_request() -> None:
    provider = CodaProvider(api_key="key", mode="learn", fast=True)
    captured: dict[str, Any] = {}

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        captured["payload"] = payload
        return _response(['data: {"type": "token", "content": "def answer():\\n    return 1"}\n'])

    provider._post = post  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="make code", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is True
    assert captured["payload"] == {
        "messages": [{"role": "user", "content": "make code"}],
        "mode": "learn",
        "fast": True,
    }


def test_coda_feedback_payload_uses_message_history_and_drops_empty_assistant() -> None:
    provider = CodaProvider(api_key="key")
    captured: dict[str, Any] = {}
    messages = (
        ProviderMessage(role="user", content="initial"),
        ProviderMessage(role="assistant", content="bad code"),
        ProviderMessage(role="user", content="feedback"),
        ProviderMessage(role="assistant", content=""),
    )

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        captured["payload"] = payload
        return _response(['data: {"type": "token", "content": "def answer():\\n    return 1"}\n'])

    provider._post = post  # type: ignore[method-assign]
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="ignored",
        entry_point="answer",
        messages=messages,
    )
    provider.generate(request)
    assert captured["payload"]["messages"] == [
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "bad code"},
        {"role": "user", "content": "feedback"},
    ]


def test_coda_response_maps_code_metadata_headers_and_raw_response() -> None:
    provider = CodaProvider(api_key="key", prefer_structured_response=True)
    provider._post = lambda payload: _response(  # type: ignore[method-assign]
        [
            'data: {"type": "token", "content": "def answer():\\n    return 1"}\n',
            'data: {"type": "structured_response", "data": {"qiskit": "def answer():\\n    return 2"}}\n',
            'data: {"type": "completed"}\n',
        ],
        headers={"x-run-id": "run-1", "x-thread-id": "thread-1"},
    )
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="coda/build")
    response = provider.generate(request)
    assert response.ok is True
    assert response.code == "def answer():\n    return 2"
    assert response.metadata["provider"] == "coda"
    assert response.metadata["extracted_source"] == "structured_response"
    assert response.metadata["run_id"] == "run-1"
    assert response.metadata["thread_id"] == "thread-1"
    assert response.raw_response["headers"] == {"run_id": "run-1", "thread_id": "thread-1"}


def test_coda_terminal_error_returns_provider_error() -> None:
    provider = CodaProvider(api_key="key")
    provider._post = lambda payload: _response(['data: {"type": "error", "content": "boom"}\n'])  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "coda event error: boom"


def test_coda_cancelled_returns_provider_error() -> None:
    provider = CodaProvider(api_key="key")
    provider._post = lambda payload: _response(['data: {"type": "cancelled"}\n'])  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "coda event cancelled"


def test_coda_no_content_returns_provider_error() -> None:
    provider = CodaProvider(api_key="key")
    provider._post = lambda payload: _response(['data: {"type": "heartbeat", "content": "ignored"}\n'])  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "coda returned no extractable code or assistant text"


def test_coda_http_error_surfaces_status_and_body() -> None:
    provider = CodaProvider(api_key="key")
    provider._post = lambda payload: (_ for _ in ()).throw(_make_http_error(422, b"bad body"))  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "coda HTTP 422: bad body"


def test_coda_reports_unexpected_generate_exception() -> None:
    provider = CodaProvider(api_key="key")
    provider._post = lambda payload: (_ for _ in ()).throw(ValueError("bad"))  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "ValueError: bad"


def test_coda_generate_many_preserves_order() -> None:
    provider = CodaProvider(api_key="key")

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        name = payload["messages"][0]["content"]
        return _response([f'data: {{"type": "token", "content": "def {name}():\\n    return 1"}}\n'])

    provider._post = post  # type: ignore[method-assign]
    requests = [
        ProviderRequest(task_id="01", framework="qiskit", prompt="first", entry_point="first"),
        ProviderRequest(task_id="02", framework="qiskit", prompt="second", entry_point="second"),
    ]
    responses = provider.generate_many(requests)
    assert [response.code for response in responses] == [
        "def first():\n    return 1",
        "def second():\n    return 1",
    ]


def _response(lines: list[str], headers: dict[str, str] | None = None) -> _CodaHttpResponse:
    return _CodaHttpResponse(lines=tuple(lines), headers=headers or {})


def _make_http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "msg", {}, io.BytesIO(body))
