from __future__ import annotations

import pytest

from qceval.production.endpoints import ModelCapability, assess_endpoint, select_endpoint


def _capability(**overrides) -> ModelCapability:
    values = {
        "model_id": "author/model",
        "reasoning_setting": "high",
        "configured_output_tokens": 100,
        "output_limit_source": "author_native",
        "author_native_max_output_tokens": 100,
        "native_context_tokens": 1000,
        "evidence_url": "https://author.example/model",
        "evidence_retrieved_at_utc": "2026-08-09T00:00:00Z",
        "author_endpoint_tags": ("author",),
    }
    values.update(overrides)
    return ModelCapability(**values)


def _endpoint(tag: str = "author", **overrides) -> dict:
    values = {
        "tag": tag,
        "provider_name": "Author",
        "model_id": "author/model",
        "name": "Author | author/model-20260810",
        "max_completion_tokens": 100,
        "context_length": 1000,
        "max_prompt_tokens": 900,
        "uptime_last_1d": 99.0,
        "supported_parameters": ["reasoning", "temperature", "max_completion_tokens", "max_tokens"],
        "quantization": "fp16",
        "pricing": {"completion": "0.00002", "prompt": "0.00001"},
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("limit", [None, "not-numeric", 99])
def test_full_cap_gate_rejects_missing_nonnumeric_or_lower_endpoint_limit(limit: object) -> None:
    assessment = assess_endpoint(_capability(), _endpoint(max_completion_tokens=limit), largest_prompt_tokens=100)

    assert assessment.accepted is False
    assert any("max_completion_tokens" in reason for reason in assessment.reasons)


@pytest.mark.parametrize("uptime", [None, "not-numeric", 94.999])
def test_uptime_gate_rejects_missing_nonnumeric_or_sub_95_percent(uptime: object) -> None:
    assessment = assess_endpoint(_capability(), _endpoint(uptime_last_1d=uptime), largest_prompt_tokens=100)

    assert assessment.accepted is False
    assert any("uptime_last_1d" in reason for reason in assessment.reasons)


def test_endpoint_must_fit_prompt_plus_full_output_and_expose_reasoning_and_ceiling() -> None:
    endpoint = _endpoint(context_length=199, supported_parameters=["temperature"])

    assessment = assess_endpoint(_capability(), endpoint, largest_prompt_tokens=100)

    assert assessment.accepted is False
    assert any("context_length" in reason for reason in assessment.reasons)
    assert any("reasoning control" in reason for reason in assessment.reasons)
    assert any("max_tokens" in reason for reason in assessment.reasons)


def test_selection_prefers_quality_then_precision_then_uptime_cost_and_tag() -> None:
    candidates = [
        _endpoint("community", quantization="fp32", pricing={"completion": "0", "prompt": "0"}),
        _endpoint("author", quantization="fp8", pricing={"completion": "0", "prompt": "0"}),
        _endpoint("authorized-b", quantization="fp16", pricing={"completion": "0.00003", "prompt": "0.00001"}),
        _endpoint("authorized-a", quantization="fp16", pricing={"completion": "0.00002", "prompt": "0.00001"}),
    ]
    capability = _capability(author_endpoint_tags=(), authorized_endpoint_tags=("authorized-a", "authorized-b"))

    selected = select_endpoint(capability, candidates, largest_prompt_tokens=100)

    assert selected["endpoint_tag"] == "authorized-a"
    assert selected["endpoint_served_model_id"] == "author/model-20260810"
    assert selected["output_token_parameter"] == "max_completion_tokens"
    assert selected["configured_output_tokens"] == 100
    assert selected["output_limit_source"] == "author_native"
    assert selected["endpoint_effective_output_capacity_tokens"] == 100
    assert selected["endpoint_cap_status"] == "catalog_numeric"
    assert selected["temperature"] == 0.0
    assert selected["route_revision"].startswith("route-")


def test_selection_prefers_uptime_before_price_after_quality_and_precision() -> None:
    candidates = [
        _endpoint("stable", uptime_last_1d=99.9, pricing={"completion": "0.00003", "prompt": "0.00001"}),
        _endpoint("cheap", uptime_last_1d=99.0, pricing={"completion": "0.00001", "prompt": "0.00001"}),
    ]

    selected = select_endpoint(_capability(author_endpoint_tags=()), candidates, largest_prompt_tokens=100)

    assert selected["endpoint_tag"] == "stable"


def test_model_context_must_fit_prompt_plus_configured_budget() -> None:
    assessment = assess_endpoint(
        _capability(native_context_tokens=199),
        _endpoint(context_length=1000),
        largest_prompt_tokens=100,
    )

    assert assessment.accepted is False
    assert any("author-documented context" in reason for reason in assessment.reasons)


def test_temperature_is_omitted_when_endpoint_does_not_expose_it() -> None:
    endpoint = _endpoint(supported_parameters=["reasoning", "max_tokens"])

    selected = select_endpoint(_capability(), [endpoint], largest_prompt_tokens=100)

    assert selected["output_token_parameter"] == "max_tokens"
    assert selected["temperature"] is None
    assert selected["temperature_behavior"] == "not_exposed"


def test_capability_registry_requires_author_evidence_and_output_limit_policy() -> None:
    with pytest.raises(ValueError, match="author_evidence is required"):
        ModelCapability.from_mapping({"model_id": "author/model", "reasoning_setting": "high"})
    with pytest.raises(ValueError, match="output_limit is required"):
        ModelCapability.from_mapping(
            {
                "model_id": "author/model",
                "reasoning_setting": "high",
                "author_evidence": {
                    "native_max_output_tokens": 100,
                    "url": "https://author.example/model",
                    "retrieved_at_utc": "2026-08-09T00:00:00Z",
                },
            }
        )


def test_author_native_limit_must_be_documented_and_match_configured_budget() -> None:
    base = {
        "model_id": "author/model",
        "reasoning_setting": "high",
        "output_limit": {"source": "author_native", "configured_output_tokens": 100},
        "author_evidence": {
            "native_max_output_tokens": None,
            "url": "https://author.example/model",
            "retrieved_at_utc": "2026-08-09T00:00:00Z",
        },
    }
    with pytest.raises(ValueError, match="require author evidence"):
        ModelCapability.from_mapping(base)
    base["author_evidence"]["native_max_output_tokens"] = 99
    with pytest.raises(ValueError, match="must equal the author-native limit"):
        ModelCapability.from_mapping(base)


def _glm_registry_entry() -> dict:
    return {
        "model_id": "z-ai/glm-5.2",
        "reasoning_efforts": ["max"],
        "output_limit": {"source": "author_native", "configured_output_tokens": 131072},
        "reasoning_parameter_names": ["reasoning"],
        "author_endpoint_tags": ["z-ai"],
        "authorized_provider_names": ["Z.AI"],
        "author_evidence": {
            "url": "https://docs.z.ai/guides/overview/concept-param",
            "retrieved_at_utc": "2026-08-10T21:39:39Z",
            "native_context_tokens": 1000000,
            "native_max_output_tokens": 131072,
            "default_max_output_tokens": 65536,
            "output_token_parameter": "max_tokens",
        },
    }


def test_glm_5_2_registry_uses_precise_native_output_limit() -> None:
    capability = ModelCapability.from_mapping(_glm_registry_entry())

    assert capability.configured_output_tokens == 131072
    assert capability.author_native_max_output_tokens == 131072
    assert capability.reasoning_setting == "max"
    assert capability.evidence_url == "https://docs.z.ai/guides/overview/concept-param"

    below_cap = assess_endpoint(
        capability,
        _endpoint(
            "z-ai",
            max_completion_tokens=131071,
            context_length=1_000_000,
            latency_last_30m=1.0,
            throughput_last_30m=1.0,
        ),
        largest_prompt_tokens=100,
    )
    full_cap = assess_endpoint(
        capability,
        _endpoint(
            "z-ai",
            max_completion_tokens=131072,
            context_length=1_000_000,
            latency_last_30m=1.0,
            throughput_last_30m=1.0,
        ),
        largest_prompt_tokens=100,
    )

    assert below_cap.accepted is False
    assert any("below configured output budget 131072" in reason for reason in below_cap.reasons)
    assert full_cap.accepted is True
    wrong_parameter = assess_endpoint(
        capability,
        _endpoint(
            "z-ai",
            max_completion_tokens=131072,
            context_length=1_000_000,
            latency_last_30m=1.0,
            throughput_last_30m=1.0,
            supported_parameters=["reasoning", "temperature", "max_completion_tokens"],
        ),
        largest_prompt_tokens=100,
    )
    assert wrong_parameter.accepted is False
    assert "GLM requires the exact max_tokens parameter" in wrong_parameter.reasons


def test_grok_4_6_registry_uses_xhigh_and_the_first_party_cap_exception() -> None:
    capability = ModelCapability.from_mapping(_grok_exception_payload())

    assert capability.reasoning_setting == "xhigh"
    assert capability.native_context_tokens == 500000
    assert capability.author_native_max_output_tokens is None
    assert capability.endpoint_cap_policy == "undisclosed_first_party_exception"
    assert capability.undisclosed_cap_allowed_endpoint_tags == ("xai",)
    assert capability.evidence_url == "https://docs.x.ai/developers/models/grok-4.6"


def test_benchmark_floor_accepts_undocumented_native_limit() -> None:
    capability = ModelCapability.from_mapping(
        {
            "model_id": "open/model",
            "reasoning_setting": "enabled",
            "output_limit": {"source": "benchmark_floor", "configured_output_tokens": 128000},
            "author_evidence": {
                "native_max_output_tokens": None,
                "native_context_tokens": 256000,
                "url": "https://author.example/model",
                "retrieved_at_utc": "2026-08-09T00:00:00Z",
            },
        }
    )

    assert capability.configured_output_tokens == 128000
    assert capability.output_limit_source == "benchmark_floor"
    assert capability.author_native_max_output_tokens is None


def test_benchmark_floor_rejects_a_different_configured_budget() -> None:
    payload = {
        "model_id": "open/model",
        "reasoning_setting": "enabled",
        "output_limit": {"source": "benchmark_floor", "configured_output_tokens": 64000},
        "author_evidence": {
            "url": "https://author.example/model",
            "retrieved_at_utc": "2026-08-09T00:00:00Z",
            "native_context_tokens": 256000,
            "native_max_output_tokens": None,
        },
    }

    with pytest.raises(ValueError, match="benchmark_floor must be exactly 128000 tokens"):
        ModelCapability.from_mapping(payload)


def _grok_exception_payload() -> dict:
    return {
        "model_id": "x-ai/grok-4.6",
        "reasoning_setting": "xhigh",
        "output_limit": {"source": "benchmark_floor", "configured_output_tokens": 128000},
        "reasoning_parameter_names": ["reasoning"],
        "author_endpoint_tags": ["xai"],
        "endpoint_completion_limit": {
            "policy": "undisclosed_first_party_exception",
            "allowed_endpoint_tags": ["xai"],
            "rationale": "First-party metadata is undisclosed; runtime lower-cap detection remains mandatory.",
            "approved_at_utc": "2026-08-10T00:00:00Z",
        },
        "author_evidence": {
            "url": "https://docs.x.ai/developers/models/grok-4.6",
            "retrieved_at_utc": "2026-08-09T00:00:00Z",
            "native_context_tokens": 500000,
            "native_max_output_tokens": None,
        },
    }


def test_grok_exception_accepts_only_the_exact_first_party_endpoint() -> None:
    capability = ModelCapability.from_mapping(_grok_exception_payload())
    selected = select_endpoint(
        capability,
        [
            _endpoint("xai/priority", max_completion_tokens=None, context_length=500000),
            _endpoint("xai", max_completion_tokens=None, context_length=500000),
        ],
        largest_prompt_tokens=100,
    )

    assert selected["endpoint_tag"] == "xai"
    assert selected["endpoint_max_completion_tokens"] is None
    assert selected["endpoint_effective_output_capacity_tokens"] is None
    assert selected["endpoint_cap_status"] == "undisclosed_first_party_exception"
    assert selected["endpoint_cap_exception"]["allowed_endpoint_tags"] == ["xai"]
    assert selected["route_revision"].startswith("route-")


def test_undisclosed_cap_exception_is_rejected_for_other_models_and_tags() -> None:
    other = _grok_exception_payload()
    other["model_id"] = "open/model"
    with pytest.raises(ValueError, match="restricted to x-ai/grok-4.6"):
        ModelCapability.from_mapping(other)

    wrong_tag = _grok_exception_payload()
    wrong_tag["endpoint_completion_limit"]["allowed_endpoint_tags"] = ["xai/priority"]
    with pytest.raises(ValueError, match="only the exact xai endpoint tag"):
        ModelCapability.from_mapping(wrong_tag)


def test_grok_exception_does_not_allow_third_party_or_low_numeric_caps() -> None:
    capability = ModelCapability.from_mapping(_grok_exception_payload())

    third_party = assess_endpoint(
        capability,
        _endpoint("community", max_completion_tokens=None, context_length=500000),
        largest_prompt_tokens=100,
    )
    low_first_party = assess_endpoint(
        capability,
        _endpoint("xai", max_completion_tokens=32768, context_length=500000),
        largest_prompt_tokens=100,
    )

    assert third_party.accepted is False
    assert any("max_completion_tokens" in reason for reason in third_party.reasons)
    assert low_first_party.accepted is False
    assert any("below configured output budget" in reason for reason in low_first_party.reasons)
