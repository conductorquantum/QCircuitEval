"""Tests for Coda provider retry behavior."""

from __future__ import annotations

import io
import time
import urllib.error
from email.message import Message
from typing import Any

from qceval.models import ProviderRequest
from qceval.providers.coda.provider import CodaProvider, _CodaHttpResponse


def test_coda_retries_408() -> None:
    provider = CodaProvider(api_key="key", max_retries=2, retry_base_delay=0.01)
    calls: list[int] = []

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        calls.append(1)
        if len(calls) < 2:
            raise _make_http_error(408, b"timeout")
        return _response(['data: {"type": "token", "content": "def answer():\\n    return 1"}\n'])

    provider._post = post  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is True
    assert len(calls) == 2


def test_coda_retries_429() -> None:
    provider = CodaProvider(api_key="key", max_retries=2, retry_base_delay=0.01)
    calls: list[int] = []

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        calls.append(1)
        if len(calls) < 2:
            raise _make_http_error(429, b"rate limited")
        return _response(['data: {"type": "token", "content": "def answer():\\n    return 1"}\n'])

    provider._post = post  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is True
    assert len(calls) == 2


def test_coda_retries_5xx() -> None:
    provider = CodaProvider(api_key="key", max_retries=2, retry_base_delay=0.01)
    calls: list[int] = []

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        calls.append(1)
        if len(calls) < 2:
            raise _make_http_error(503, b"unavailable")
        return _response(['data: {"type": "token", "content": "def answer():\\n    return 1"}\n'])

    provider._post = post  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is True
    assert len(calls) == 2


def test_coda_does_not_retry_422() -> None:
    provider = CodaProvider(api_key="key", max_retries=3, retry_base_delay=0.01)
    calls: list[int] = []

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        calls.append(1)
        raise _make_http_error(422, b"validation")

    provider._post = post  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "coda HTTP 422: validation"
    assert len(calls) == 1


def test_coda_retries_transport_errors() -> None:
    provider = CodaProvider(api_key="key", max_retries=1, retry_base_delay=0.01)
    calls: list[int] = []

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        calls.append(1)
        if len(calls) < 2:
            raise urllib.error.URLError("temporary")
        return _response(['data: {"type": "token", "content": "def answer():\\n    return 1"}\n'])

    provider._post = post  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is True
    assert len(calls) == 2


def test_coda_respects_retry_after() -> None:
    headers = Message()
    headers["Retry-After"] = "0.05"
    provider = CodaProvider(api_key="key", max_retries=1, retry_base_delay=0.01)
    timestamps: list[float] = []

    def post(payload: dict[str, Any]) -> _CodaHttpResponse:
        timestamps.append(time.monotonic())
        if len(timestamps) < 2:
            raise urllib.error.HTTPError("url", 429, "msg", headers, io.BytesIO(b"rate limited"))
        return _response(['data: {"type": "token", "content": "def answer():\\n    return 1"}\n'])

    provider._post = post  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is True
    assert timestamps[1] - timestamps[0] >= 0.05


def test_coda_ignores_invalid_retry_after() -> None:
    headers = Message()
    headers["Retry-After"] = "later"
    error = urllib.error.HTTPError("url", 429, "msg", headers, io.BytesIO(b"rate limited"))
    retry_after = CodaProvider._parse_retry_after(error)
    assert retry_after is None


def test_coda_reports_final_transport_error() -> None:
    provider = CodaProvider(api_key="key", max_retries=0)
    provider._post = lambda payload: (_ for _ in ()).throw(urllib.error.URLError("down"))  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")
    response = provider.generate(request)
    assert response.ok is False
    assert response.error == "URLError: <urlopen error down>"


def _response(lines: list[str]) -> _CodaHttpResponse:
    return _CodaHttpResponse(lines=tuple(lines), headers={})


def _make_http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "msg", {}, io.BytesIO(body))
