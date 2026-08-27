from __future__ import annotations

import pytest

from qceval.models import ProviderMessage, ProviderRequest
from qceval.providers.openrouter import OpenRouterProvider


def _router_metadata(
    model: str = "m",
    *,
    provider: str = "Author",
    pipeline: list[dict] | None = None,
    catalog_total: int = 1,
) -> dict:
    return {
        "requested": model,
        "strategy": "direct",
        "attempt": 1,
        "endpoints": {
            "total": catalog_total,
            "available": [{"provider": provider, "model": model, "selected": True}],
        },
        "attempts": [{"provider": provider, "model": model, "status": 200}],
        "pipeline": [] if pipeline is None else pipeline,
    }


def test_openrouter_requires_api_key() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key=None)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert response.error == "openrouter api key is not configured"


def test_openrouter_extracts_code_from_response() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key")
    provider._post_json = lambda payload: {  # type: ignore[method-assign]
        "model": payload["model"],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost": 0.00125},
        "choices": [{"message": {"content": "```python\ndef answer():\n    return 1\n```"}}],
    }
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert response.code == "def answer():\n    return 1"
    assert response.usage is not None
    assert response.usage.total_tokens == 5
    assert response.usage.cost_usd == 0.00125
    assert response.metadata["provider"] == "openrouter"
    assert response.metadata["infrastructure_attempts"] == 1
    assert response.metadata["attempt_history"][0]["status"] == "accepted_model_outcome"


@pytest.mark.parametrize("cost", [True, -1, "not-a-number", float("inf")])
def test_openrouter_ignores_invalid_reported_cost(cost: object) -> None:
    assert OpenRouterProvider._cost_usd(cost) is None


@pytest.mark.parametrize("content", [" \n\t", None])
def test_openrouter_rejects_empty_or_malformed_response_content(content: str | None) -> None:
    provider = OpenRouterProvider(api_key="key")
    provider._post_json = lambda payload: {  # type: ignore[method-assign]
        "model": payload["model"],
        "choices": [{"message": {"content": content}}],
    }
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    response = provider.generate(request)

    assert response.code == ""
    assert response.ok is False


def test_openrouter_generate_many_preserves_order() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key")

    def post_json(payload):
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": f"def {payload['messages'][0]['content']}():\n    return 1\n"}}],
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    requests = [
        ProviderRequest(task_id="01", framework="qiskit", prompt="first", entry_point="first", model="m"),
        ProviderRequest(task_id="02", framework="qiskit", prompt="second", entry_point="second", model="m"),
    ]

    # Act
    responses = provider.generate_many(requests)

    # Assert
    assert [response.code for response in responses] == [
        "def first():\n    return 1",
        "def second():\n    return 1",
    ]


def test_openrouter_one_shot_payload_uses_single_user_message() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key")
    captured = {}

    def post_json(payload):
        captured["payload"] = payload
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="make code", entry_point="answer", model="m")

    # Act
    provider.generate(request)

    # Assert
    assert captured["payload"]["messages"] == [{"role": "user", "content": "make code"}]
    assert captured["payload"]["reasoning"] == {"exclude": True}


def test_openrouter_payload_includes_configured_reasoning_effort() -> None:
    provider = OpenRouterProvider(api_key="key", reasoning_effort="xhigh")
    captured = {}

    def post_json(payload):
        captured["payload"] = payload
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="make code", entry_point="answer", model="m")

    provider.generate(request)

    assert captured["payload"]["reasoning"] == {"exclude": True, "effort": "xhigh"}


def test_openrouter_response_records_reasoning_effort() -> None:
    provider = OpenRouterProvider(api_key="key", reasoning_effort="high")
    provider._post_json = lambda payload: {  # type: ignore[method-assign]
        "model": payload["model"],
        "choices": [{"message": {"content": "def answer():\n    return 1\n"}}],
    }
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    response = provider.generate(request)

    assert response.metadata["provider"] == "openrouter"
    assert response.metadata["reasoning_effort"] == "high"


def test_openrouter_payload_enables_fixed_reasoning_model() -> None:
    provider = OpenRouterProvider(api_key="key", reasoning_enabled=True)
    captured = {}

    def post_json(payload):
        captured["payload"] = payload
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    response = provider.generate(request)

    assert captured["payload"]["reasoning"] == {"exclude": True, "enabled": True}
    assert response.metadata["provider"] == "openrouter"
    assert response.metadata["reasoning_enabled"] is True


def test_openrouter_repair_payload_uses_message_history() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key")
    captured = {}

    def post_json(payload):
        captured["payload"] = payload
        return {"model": payload["model"], "choices": [{"message": {"content": "def answer():\n    return 1\n"}}]}

    provider._post_json = post_json  # type: ignore[method-assign]
    messages = (
        ProviderMessage(role="user", content="initial"),
        ProviderMessage(role="assistant", content="bad code"),
        ProviderMessage(role="user", content="feedback"),
    )
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="ignored",
        entry_point="answer",
        model="m",
        messages=messages,
    )

    # Act
    provider.generate(request)

    # Assert
    assert captured["payload"]["messages"] == [
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "bad code"},
        {"role": "user", "content": "feedback"},
    ]


def test_openrouter_generate_many_preserves_order_with_messages() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key")

    def post_json(payload):
        name = payload["messages"][-1]["content"]
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": f"def {name}():\n    return 1\n"}}],
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    requests = [
        ProviderRequest(
            task_id="01",
            framework="qiskit",
            prompt="unused",
            entry_point="first",
            model="m",
            messages=(ProviderMessage(role="user", content="first"),),
        ),
        ProviderRequest(
            task_id="02",
            framework="qiskit",
            prompt="unused",
            entry_point="second",
            model="m",
            messages=(ProviderMessage(role="user", content="second"),),
        ),
    ]

    # Act
    responses = provider.generate_many(requests)

    # Assert
    assert [response.code for response in responses] == [
        "def first():\n    return 1",
        "def second():\n    return 1",
    ]


def test_openrouter_requires_model() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key")
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert response.error == "openrouter model is not configured"


def test_openrouter_allows_benign_target_dict_in_assistant_turn() -> None:
    """Regression (H9): echoed candidate code in assistant turns is exempt
    from the oracle substring deny-list."""
    provider = OpenRouterProvider(api_key="key")
    provider._post_json = lambda payload: {  # type: ignore[method-assign]
        "model": payload["model"],
        "choices": [{"message": {"content": "```python\ndef answer():\n    return 1\n```"}}],
    }
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        model="m",
        messages=(
            ProviderMessage(role="user", content="p"),
            ProviderMessage(role="assistant", content='spec = {"target": 2, "control": 0}\n'),
            ProviderMessage(role="user", content="Previous code ran but did not satisfy the task checks."),
        ),
    )

    response = provider.generate(request)

    assert response.ok is True


def test_openrouter_oracle_leak_in_user_turn_is_typed_harness_error() -> None:
    """Regression (H9): a deny-list hit in harness-authored text fails only the
    request with a typed harness error instead of raising."""
    provider = OpenRouterProvider(api_key="key")
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        model="m",
        messages=(ProviderMessage(role="user", content='fix the "target": mismatch'),),
    )

    response = provider.generate(request)

    assert response.ok is False
    assert response.metadata == {"harness_error": "oracle_isolation"}
    assert response.error is not None and response.error.startswith("harness safety violation")


def test_openrouter_reports_unexpected_generate_exception() -> None:
    # Arrange
    provider = OpenRouterProvider(api_key="key")
    provider._post_json = lambda payload: (_ for _ in ()).throw(ValueError("bad"))  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer", model="m")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert response.error == "ValueError: bad"


def test_openrouter_posts_json(monkeypatch) -> None:
    # Arrange
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"ok": true}'

    captured = {}

    def urlopen(request, timeout):
        captured["headers"] = dict(request.header_items())
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    provider = OpenRouterProvider(api_key="key", timeout=1)

    # Act
    payload = provider._post_json({"model": "m"})

    # Assert
    assert payload == {"ok": True}
    assert captured["headers"]["X-openrouter-metadata"] == "enabled"
    assert captured["headers"]["X-openrouter-cache"] == "false"


def test_openrouter_pinned_payload_is_singular_full_cap_and_omits_temperature() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=128_000,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_completion_tokens",
        route_revision="route-01",
        temperature=None,
        reasoning_effort="max",
        configuration_id="openai-gpt-5-6-sol__effort-max",
    )
    captured = {}

    def post_json(payload):
        captured["payload"] = payload
        return {
            "id": "gen-1",
            "model": payload["model"],
            "openrouter_metadata": _router_metadata(model=payload["model"], catalog_total=18),
            "choices": [{"finish_reason": "stop", "message": {"content": "def answer():\n    return 1\n"}}],
            "usage": {"cost": 0.01},
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        model="openai/gpt-5.6-sol",
    )

    response = provider.generate(request)

    assert captured["payload"]["provider"] == {
        "only": ["author/region"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert captured["payload"]["max_completion_tokens"] == 128_000
    assert "max_tokens" not in captured["payload"]
    assert "temperature" not in captured["payload"]
    assert response.metadata["generation_id"] == "gen-1"
    assert response.metadata["route"]["route_revision"] == "route-01"
    assert response.metadata["route"]["configuration_id"] == "openai-gpt-5-6-sol__effort-max"
    assert len(response.metadata["route"]["configuration_identity_sha256"]) == 64
    assert response.metadata["route"]["output_limit_source"] == "author_native"
    assert response.metadata["route"]["endpoint_cap_status"] == "catalog_numeric"
    assert response.metadata["route"]["route_verified"] is True
    assert response.metadata["route"]["response_cache_disabled"] is True
    assert response.metadata["route"]["selected_provider"] == "Author"
    assert response.metadata["route"]["router_endpoint_catalog_total"] == 18
    assert response.metadata["route"]["router_available_endpoint_count"] == 1


def test_openrouter_pinned_fixed_reasoning_configuration_uses_enabled_identity() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=128_000,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        temperature=0.0,
        reasoning_enabled=True,
        configuration_id="google-gemma-4-31b-it__effort-enabled",
    )
    provider._post_json = lambda payload: {  # type: ignore[method-assign]
        "id": "gen-enabled",
        "model": payload["model"],
        "openrouter_metadata": _router_metadata(model=payload["model"], catalog_total=1),
        "choices": [{"finish_reason": "stop", "message": {"content": "def answer():\n    return 1\n"}}],
        "usage": {"cost": 0.01},
    }
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        model="google/gemma-4-31b-it",
    )

    response = provider.generate(request)

    assert response.ok is True
    assert response.metadata["route"]["configuration_id"] == "google-gemma-4-31b-it__effort-enabled"
    assert len(response.metadata["route"]["configuration_identity_sha256"]) == 64


def test_openrouter_rejects_multiple_post_filter_available_endpoints() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=0,
    )
    metadata = _router_metadata(catalog_total=18)
    metadata["endpoints"]["available"].append({"provider": "Unexpected", "model": "m", "selected": False})
    provider._post_json = lambda payload: {  # type: ignore[method-assign]
        "id": "gen-rejected",
        "model": payload["model"],
        "openrouter_metadata": metadata,
        "choices": [{"message": {"content": "def answer():\n    return 1\n"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9, "cost": 0.02},
    }
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert response.ok is False
    assert response.metadata["infrastructure_error"] is True
    assert "exactly one available endpoint" in (response.error or "")
    assert response.metadata["generation_id"] == "gen-rejected"
    assert response.metadata["route_verification_attempts"] == [
        {
            "attempt_number": 1,
            "error": "router did not report exactly one available endpoint",
            "generation_id": "gen-rejected",
            "model": "m",
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9,
                "reasoning_tokens": None,
                "cached_tokens": None,
                "cost_usd": 0.02,
            },
            "openrouter_metadata_present": True,
            "openrouter_metadata": metadata,
        }
    ]
    assert response.raw_response is not None and response.raw_response["id"] == "gen-rejected"


def test_openrouter_does_not_retry_rejected_route_provenance() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=1,
        retry_base_delay=0,
    )
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        raw = {
            "id": f"gen-{calls}",
            "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": "def answer():\n    return 1\n"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost": 0.01},
        }
        if calls == 2:
            raw["openrouter_metadata"] = _router_metadata()
        return raw

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert response.ok is False
    assert calls == 1
    assert response.metadata["generation_id"] == "gen-1"
    assert response.metadata["retryable_infrastructure"] is False
    assert response.metadata["failure_classification"] == "route_provenance_failure"
    assert response.metadata["route_verification_attempts"] == [
        {
            "attempt_number": 1,
            "error": "response omitted openrouter_metadata",
            "generation_id": "gen-1",
            "model": "m",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
                "reasoning_tokens": None,
                "cached_tokens": None,
                "cost_usd": 0.01,
            },
            "openrouter_metadata_present": False,
        }
    ]


def test_openrouter_retries_structured_504_response_before_success() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=1,
        retry_base_delay=0,
    )
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"error": {"code": 504, "message": "Provider timed out after 300390ms"}}
        return {
            "id": "gen-success",
            "model": payload["model"],
            "openrouter_metadata": _router_metadata(),
            "choices": [{"finish_reason": "stop", "message": {"content": "def answer():\n    return 1\n"}}],
            "usage": {"cost": 0.02},
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert response.ok is True
    assert response.metadata["transport_retries"] == 1
    assert response.metadata["route"]["route_verified"] is True
    assert response.metadata.get("route_verification_attempts") is None
    assert response.metadata["provider_error_attempts"] == [
        {
            "attempt_number": 1,
            "code": 504,
            "message": "Provider timed out after 300390ms",
            "openrouter_metadata_present": False,
        }
    ]


def test_openrouter_retries_choice_level_structured_502_response_before_success() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=1,
        retry_base_delay=0,
    )
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "id": "gen-failed",
                "model": payload["model"],
                "openrouter_metadata": _router_metadata(),
                "choices": [
                    {
                        "error": {
                            "code": 502,
                            "message": "Network connection lost.",
                            "metadata": {"error_type": "provider_unavailable"},
                        },
                        "finish_reason": "error",
                        "message": {"content": None},
                    }
                ],
            }
        return {
            "id": "gen-success",
            "model": payload["model"],
            "openrouter_metadata": _router_metadata(),
            "choices": [{"finish_reason": "stop", "message": {"content": "def answer():\n    return 1\n"}}],
            "usage": {"cost": 0.02},
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert calls == 2
    assert response.ok is True
    assert response.metadata["transport_retries"] == 1
    assert response.metadata["route"]["route_verified"] is True
    assert response.metadata.get("route_verification_attempts") is None
    assert response.metadata["provider_error_attempts"] == [
        {
            "attempt_number": 1,
            "code": 502,
            "message": "Network connection lost.",
            "generation_id": "gen-failed",
            "model": "m",
            "openrouter_metadata_present": True,
        }
    ]


def test_openrouter_reports_exhausted_structured_504_response_as_infrastructure_error() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=1,
        retry_base_delay=0,
    )
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        return {"error": {"code": 504, "message": "Provider timed out after 300390ms"}}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert calls == 2
    assert response.ok is False
    assert response.error == "OpenRouter API error 504: Provider timed out after 300390ms"
    assert response.metadata["infrastructure_error"] is True
    assert response.metadata["transport_retries"] == 1
    assert response.metadata["provider_error_code"] == 504
    assert response.metadata["provider_error_message"] == "Provider timed out after 300390ms"
    assert response.metadata["provider_error_attempts"] == [
        {
            "attempt_number": 1,
            "code": 504,
            "message": "Provider timed out after 300390ms",
            "openrouter_metadata_present": False,
        },
        {
            "attempt_number": 2,
            "code": 504,
            "message": "Provider timed out after 300390ms",
            "openrouter_metadata_present": False,
        },
    ]
    assert response.metadata.get("route_verification_error") is None
    assert response.metadata["route"]["route_verified"] is False
    assert response.raw_response == {"error": {"code": 504, "message": "Provider timed out after 300390ms"}}


def test_openrouter_does_not_retry_nontransient_structured_error_response() -> None:
    provider = OpenRouterProvider(api_key="key", max_retries=3, retry_base_delay=0)
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        return {"error": {"code": 400, "message": "Bad request"}}

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert calls == 1
    assert response.metadata["provider_error_code"] == 400
    assert response.metadata.get("transport_retries") is None


def test_openrouter_preserves_mixed_provider_and_route_failure_evidence() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=1,
        retry_base_delay=0,
    )
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"error": {"code": 504, "message": "Provider timed out"}}
        return {
            "id": "gen-unverified",
            "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": "def answer():\n    return 1\n"}}],
            "usage": {"cost": 0.02},
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert response.metadata["provider_error_attempts"][0]["code"] == 504
    assert response.metadata["route_verification_attempts"][0]["generation_id"] == "gen-unverified"
    assert response.metadata["route_verification_error"] == "response omitted openrouter_metadata"
    assert response.raw_response is not None and response.raw_response["id"] == "gen-unverified"


def test_openrouter_rejects_unexpected_route_and_context_compression() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=0,
    )
    metadata = _router_metadata(pipeline=[{"type": "context_compression", "name": "context-compression"}])
    provider._post_json = lambda payload: {  # type: ignore[method-assign]
        "model": payload["model"],
        "openrouter_metadata": metadata,
        "choices": [{"message": {"content": "def answer():\n    return 1\n"}}],
    }
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert response.ok is False
    assert response.metadata["infrastructure_error"] is True
    assert response.metadata["route"]["route_verified"] is False
    assert "context compression" in (response.error or "")


def test_openrouter_finish_reason_length_is_an_accepted_model_outcome_without_retry() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="benchmark_floor",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=3,
    )
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        return {
            "model": payload["model"],
            "openrouter_metadata": _router_metadata(),
            "choices": [{"finish_reason": "length", "message": {"content": ""}}],
            "usage": {"completion_tokens": 10, "cost": 0.01},
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert calls == 1
    assert response.metadata["finish_reason"] == "length"
    assert response.metadata["route"]["route_verified"] is True
    assert response.metadata.get("infrastructure_error") is None


def test_openrouter_does_not_retry_pinned_response_without_reported_cost() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="author/region",
        max_output_tokens=10,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=1,
        retry_base_delay=0,
    )
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        usage = {} if calls == 1 else {"cost": 0.02}
        return {
            "model": payload["model"],
            "openrouter_metadata": _router_metadata(),
            "choices": [{"finish_reason": "error", "message": {"content": ""}}],
            "usage": usage,
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="p", entry_point="answer", model="m")

    response = provider.generate(request)

    assert calls == 1
    assert response.metadata["retryable_infrastructure"] is False
    assert response.metadata["failure_classification"] == "route_provenance_failure"
    assert response.usage is None


def test_openrouter_accepts_verified_unbilled_policy_refusal_as_model_outcome() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="anthropic",
        max_output_tokens=128_000,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=3,
        retry_base_delay=0,
    )
    calls = 0
    refusal = "This request triggered restrictions under the provider's usage policy."

    def post_json(payload):
        nonlocal calls
        calls += 1
        router_metadata = _router_metadata(
            "anthropic/claude-5-fable-20260609",
            provider="Anthropic",
            catalog_total=6,
        )
        router_metadata["requested"] = payload["model"]
        return {
            "id": "gen-refusal",
            "model": payload["model"],
            "openrouter_metadata": router_metadata,
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "message": {"content": None, "refusal": refusal},
                    "native_finish_reason": "refusal",
                }
            ],
            "usage": None,
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(
        task_id="02",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        model="anthropic/claude-fable-5",
    )

    response = provider.generate(request)

    assert calls == 1
    assert response.ok is False
    assert response.error == refusal
    assert response.metadata["route"]["route_verified"] is True
    assert response.metadata["failure_classification"] == "provider_policy_refusal"
    assert response.metadata["campaign_resolution"] == {
        "schema_version": "qceval.policy_refusal_resolution.v1",
        "disposition": "candidate_less_provider_failure",
        "reason": "provider_policy_refusal",
        "provider_reported_usage_present": False,
        "accounting_source": "zero_normalization_for_unbilled_policy_refusal",
        "route_evidence_source": "raw_response.openrouter_metadata",
        "refusal_evidence_source": "raw_response.choices[0].message.refusal",
    }
    assert response.metadata["attempt_history"][0]["status"] == "accepted_model_outcome"
    assert response.usage is not None
    assert response.usage.to_dict() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 0.0,
    }


def test_openrouter_detects_silent_lower_endpoint_cap_without_retry() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="xai",
        max_output_tokens=128_000,
        output_limit_source="benchmark_floor",
        endpoint_cap_status="undisclosed_first_party_exception",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=3,
    )
    calls = 0

    def post_json(payload):
        nonlocal calls
        calls += 1
        return {
            "id": "gen-clamped",
            "model": payload["model"],
            "openrouter_metadata": _router_metadata("x-ai/grok-4.6", provider="xAI"),
            "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
            "usage": {"completion_tokens": 16_384, "cost": 1.0},
        }

    provider._post_json = post_json  # type: ignore[method-assign]
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        model="x-ai/grok-4.6",
    )

    response = provider.generate(request)

    assert calls == 1
    assert response.ok is False
    assert response.metadata["infrastructure_error"] is True
    assert "below requested ceiling 128000" in str(response.metadata["endpoint_capacity_error"])
    assert response.metadata["generation_id"] == "gen-clamped"
    assert response.usage is not None and response.usage.cost_usd == 1.0


def test_openrouter_length_without_usage_is_an_infrastructure_error() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="xai",
        max_output_tokens=128_000,
        output_limit_source="benchmark_floor",
        endpoint_cap_status="undisclosed_first_party_exception",
        output_token_parameter="max_tokens",
        route_revision="route-01",
        max_retries=0,
    )
    provider._post_json = lambda payload: {  # type: ignore[method-assign]
        "model": payload["model"],
        "openrouter_metadata": _router_metadata("x-ai/grok-4.6", provider="xAI"),
        "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
    }
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        model="x-ai/grok-4.6",
    )

    response = provider.generate(request)

    assert response.ok is False
    assert response.metadata["infrastructure_error"] is True
    assert "missing reported completion tokens" in str(response.metadata["endpoint_capacity_error"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"endpoint_tag": "only-one"}, "must be configured together"),
        (
            {
                "endpoint_tag": "tag",
                "max_output_tokens": 1,
                "output_limit_source": "author_native",
                "endpoint_cap_status": "catalog_numeric",
                "output_token_parameter": "bad",
                "route_revision": "route",
            },
            "unsupported output token parameter",
        ),
        (
            {
                "endpoint_tag": "tag",
                "max_output_tokens": 1,
                "output_limit_source": "benchmark_floor",
                "endpoint_cap_status": "invented",
                "output_token_parameter": "max_tokens",
                "route_revision": "route",
            },
            "unsupported endpoint cap status",
        ),
    ],
)
def test_openrouter_rejects_invalid_pin_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenRouterProvider(api_key="key", **kwargs)


def test_openrouter_rejects_undisclosed_cap_exception_for_a_different_model() -> None:
    provider = OpenRouterProvider(
        api_key="key",
        endpoint_tag="xai",
        max_output_tokens=128000,
        output_limit_source="benchmark_floor",
        endpoint_cap_status="undisclosed_first_party_exception",
        output_token_parameter="max_tokens",
        route_revision="route-01",
    )
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        model="other/model",
    )

    response = provider.generate(request)

    assert response.ok is False
    assert response.metadata["infrastructure_error"] is True
    assert "restricted to x-ai/grok-4.6" in (response.error or "")
