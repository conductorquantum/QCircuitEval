from __future__ import annotations

import json
from pathlib import Path

from qceval.core.cache import ResponseCache
from qceval.models import ProviderMessage, ProviderRequest, ProviderResponse, TokenUsage


def test_response_cache_round_trips_provider_response(tmp_path: Path) -> None:
    # Arrange
    cache = ResponseCache(tmp_path)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer", model="m")
    key = cache.key_for(request, provider="openrouter", settings={"temperature": 0.0})
    response = ProviderResponse(
        code="def answer():\n    return 1\n",
        model="m",
        metadata={"provider": "openrouter"},
        usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3, cost_usd=0.0042),
        raw_response={"ignored": True},
    )

    # Act
    cache.put(key, response)
    cached = cache.get(key)

    # Assert
    assert cached is not None
    assert cached.code == response.code
    assert cached.usage is not None
    assert cached.usage.total_tokens == 3
    assert cached.usage.cost_usd == 0.0042
    assert cached.raw_response is None


def test_response_cache_key_changes_with_generation_settings(tmp_path: Path) -> None:
    # Arrange
    cache = ResponseCache(tmp_path)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer", model="m")

    # Act
    cold = cache.key_for(request, provider="openrouter", settings={"temperature": 0.0})
    warm = cache.key_for(request, provider="openrouter", settings={"temperature": 0.2})

    # Assert
    assert cold != warm


def test_response_cache_key_changes_with_route_revision_and_endpoint(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer", model="m")

    first = cache.key_for(
        request,
        provider="openrouter",
        settings={"openrouter_endpoint_tag": "author/a", "openrouter_route_revision": "route-a"},
    )
    second = cache.key_for(
        request,
        provider="openrouter",
        settings={"openrouter_endpoint_tag": "author/b", "openrouter_route_revision": "route-b"},
    )

    assert first != second


def test_response_cache_key_isolated_by_effort_configuration(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer", model="m")

    low = cache.key_for(
        request,
        provider="openrouter",
        settings={"configuration_id": "m__effort-low", "reasoning_effort": "low"},
    )
    high = cache.key_for(
        request,
        provider="openrouter",
        settings={"configuration_id": "m__effort-high", "reasoning_effort": "high"},
    )

    assert low != high
    assert low.filename() != high.filename()


def test_cache_key_changes_with_sample_index(tmp_path: Path) -> None:
    # Arrange
    cache = ResponseCache(tmp_path)
    first = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="prompt",
        entry_point="answer",
        model="m",
        sample_index=0,
        attempt_index=0,
    )
    second = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="prompt",
        entry_point="answer",
        model="m",
        sample_index=1,
        attempt_index=0,
    )

    # Act
    first_key = cache.key_for(first, provider="openrouter", settings={"temperature": 0.0})
    second_key = cache.key_for(second, provider="openrouter", settings={"temperature": 0.0})

    # Assert
    assert first_key != second_key
    assert first_key.filename() != second_key.filename()


def test_cache_key_includes_messages_hash(tmp_path: Path) -> None:
    # Arrange
    cache = ResponseCache(tmp_path)
    first = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="prompt",
        entry_point="answer",
        model="m",
        sample_index=0,
        attempt_index=1,
        messages=(ProviderMessage(role="user", content="fix syntax"),),
    )
    second = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="prompt",
        entry_point="answer",
        model="m",
        sample_index=0,
        attempt_index=1,
        messages=(ProviderMessage(role="user", content="fix runtime"),),
    )

    # Act
    first_key = cache.key_for(first, provider="openrouter", settings={"temperature": 0.0})
    second_key = cache.key_for(second, provider="openrouter", settings={"temperature": 0.0})

    # Assert
    assert first_key != second_key
    assert first_key.filename() != second_key.filename()


def test_response_cache_does_not_persist_api_keys(tmp_path: Path) -> None:
    # Arrange
    cache = ResponseCache(tmp_path)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer", model="m")
    key = cache.key_for(
        request,
        provider="openrouter",
        settings={"openrouter_api_key": "sk-or-secret", "temperature": 0.0},
    )

    # Act
    cache.put(key, ProviderResponse(code="code", model="m"))
    payload = json.dumps(json.loads(next((tmp_path / "responses").glob("*.json")).read_text(encoding="utf-8")))

    # Assert
    assert "sk-or-secret" not in payload


def test_response_cache_skips_provider_errors(tmp_path: Path) -> None:
    # Arrange
    cache = ResponseCache(tmp_path)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer", model="m")
    key = cache.key_for(request, provider="openrouter", settings={"openrouter_api_key": "bad-key"})

    # Act
    cache.put(key, ProviderResponse(code=None, model="m", error="401 unauthorized"))

    # Assert
    assert cache.get(key) is None
    assert list((tmp_path / "responses").glob("*.json")) == []


def test_response_cache_returns_none_for_corrupt_entries(tmp_path: Path) -> None:
    # Arrange
    cache = ResponseCache(tmp_path)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer", model="m")
    key = cache.key_for(request, provider="provider/with/slash", settings={"temperature": 0.0})
    path = cache.responses_dir / key.filename()
    path.write_text("{bad json", encoding="utf-8")

    # Act
    cached = cache.get(key)

    # Assert
    assert cached is None
    assert "/" not in key.filename()


def test_cache_filename_sanitizes_task_ids(tmp_path: Path) -> None:
    # Arrange
    cache = ResponseCache(tmp_path)
    request = ProviderRequest(
        task_id="../bad/task",
        framework="qiskit",
        prompt="prompt",
        entry_point="answer",
        model="m",
    )

    # Act
    key = cache.key_for(request, provider="provider", settings={})

    # Assert
    assert "/" not in key.filename()
    assert ".." not in key.filename()
