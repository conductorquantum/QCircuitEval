"""Minimum-output OpenRouter endpoint qualification and deterministic selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OUTPUT_PARAMETERS = ("max_completion_tokens", "max_tokens")
OUTPUT_LIMIT_SOURCES = frozenset({"author_native", "benchmark_floor"})
ENDPOINT_CAP_STATUSES = frozenset({"catalog_numeric", "undisclosed_first_party_exception"})
BENCHMARK_OUTPUT_FLOOR_TOKENS = 128_000
MINIMUM_UPTIME_LAST_1D = 95.0
_UNDISCLOSED_CAP_EXCEPTION_MODEL = "x-ai/grok-4.6"
_UNDISCLOSED_CAP_EXCEPTION_TAGS = frozenset({"xai"})
_PRECISION_RANK = {
    "fp32": 100,
    "bf16": 90,
    "fp16": 90,
    "fp8": 80,
    "fp6": 70,
    "fp4": 50,
    "mxfp4": 50,
    "nvfp4": 50,
    "int8": 60,
    "int4": 50,
    "unknown": 0,
}


@dataclass(frozen=True)
class ModelCapability:
    """Model output-budget and production reasoning policy."""

    model_id: str
    reasoning_setting: str
    configured_output_tokens: int
    output_limit_source: str
    evidence_url: str
    evidence_retrieved_at_utc: str
    native_context_tokens: int | None = None
    author_native_max_output_tokens: int | None = None
    reasoning_parameter_names: tuple[str, ...] = ("reasoning",)
    author_endpoint_tags: tuple[str, ...] = ()
    authorized_endpoint_tags: tuple[str, ...] = ()
    authorized_provider_names: tuple[str, ...] = ()
    endpoint_cap_policy: str = "catalog_numeric"
    undisclosed_cap_allowed_endpoint_tags: tuple[str, ...] = ()
    endpoint_cap_exception_rationale: str | None = None
    endpoint_cap_exception_approved_at_utc: str | None = None
    reasoning_efforts: tuple[str, ...] = ()
    required_quantizations: tuple[str, ...] = ()

    @classmethod
    def from_mapping(  # noqa: C901 - strict capability-registry validation
        cls, payload: Mapping[str, Any]
    ) -> ModelCapability:
        """Parse one strict capability-registry entry.

        Args:
            payload: Curated registry entry with author evidence.

        Returns:
            Validated model capability.
        """
        evidence = payload.get("author_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{payload.get('model_id')}: author_evidence is required")
        model_id = _required_text(payload, "model_id")
        output_limit = payload.get("output_limit")
        if not isinstance(output_limit, Mapping):
            raise ValueError(f"{model_id}: output_limit is required")
        output_limit_source = _required_text(output_limit, "source")
        if output_limit_source not in OUTPUT_LIMIT_SOURCES:
            raise ValueError(
                f"{model_id}: output limit source must be one of {', '.join(sorted(OUTPUT_LIMIT_SOURCES))}"
            )
        configured_output = _positive_int(output_limit.get("configured_output_tokens"), "configured_output_tokens")
        native_output_raw = evidence.get("native_max_output_tokens")
        native_output = (
            None if native_output_raw is None else _positive_int(native_output_raw, "native_max_output_tokens")
        )
        if output_limit_source == "author_native":
            if native_output is None:
                raise ValueError(f"{model_id}: author_native output limits require author evidence")
            if configured_output != native_output:
                raise ValueError(f"{model_id}: configured output tokens must equal the author-native limit")
        elif native_output is not None:
            raise ValueError(
                f"{model_id}: benchmark_floor is only valid when the author-native output limit is undocumented"
            )
        elif configured_output != BENCHMARK_OUTPUT_FLOOR_TOKENS:
            raise ValueError(f"{model_id}: benchmark_floor must be exactly {BENCHMARK_OUTPUT_FLOOR_TOKENS} tokens")
        context = evidence.get("native_context_tokens")
        native_context = None if context is None else _positive_int(context, "native_context_tokens")
        endpoint_cap_policy, exception_tags, exception_rationale, exception_approved_at = (
            _endpoint_completion_limit_policy(payload, model_id=model_id, output_limit_source=output_limit_source)
        )
        author_endpoint_tags = _text_tuple(payload.get("author_endpoint_tags") or [])
        if endpoint_cap_policy == "undisclosed_first_party_exception" and not set(exception_tags) <= set(
            author_endpoint_tags
        ):
            raise ValueError(f"{model_id}: undisclosed-cap exception tags must be author endpoint tags")
        raw_efforts = payload.get("reasoning_efforts")
        efforts: tuple[str, ...]
        if raw_efforts is None:
            efforts = (_required_text(payload, "reasoning_setting"),)
        else:
            efforts = _text_tuple(raw_efforts)
            if "reasoning_setting" in payload:
                raise ValueError(f"{model_id}: use reasoning_efforts or reasoning_setting, not both")
        if not efforts or len(set(efforts)) != len(efforts):
            raise ValueError(f"{model_id}: reasoning_efforts must be nonempty and unique")
        required_quantizations = tuple(
            value.lower() for value in _text_tuple(payload.get("required_quantizations") or [])
        )
        if len(set(required_quantizations)) != len(required_quantizations):
            raise ValueError(f"{model_id}: required_quantizations must be unique")
        return cls(
            model_id=model_id,
            # Retain the singular field for historical callers. New campaigns
            # expand the complete ordered effort tuple below.
            reasoning_setting=efforts[-1],
            configured_output_tokens=configured_output,
            output_limit_source=output_limit_source,
            evidence_url=_required_text(evidence, "url"),
            evidence_retrieved_at_utc=_required_text(evidence, "retrieved_at_utc"),
            native_context_tokens=native_context,
            author_native_max_output_tokens=native_output,
            reasoning_parameter_names=_text_tuple(payload.get("reasoning_parameter_names") or ["reasoning"]),
            author_endpoint_tags=author_endpoint_tags,
            authorized_endpoint_tags=_text_tuple(payload.get("authorized_endpoint_tags") or []),
            authorized_provider_names=_text_tuple(payload.get("authorized_provider_names") or []),
            endpoint_cap_policy=endpoint_cap_policy,
            undisclosed_cap_allowed_endpoint_tags=exception_tags,
            endpoint_cap_exception_rationale=exception_rationale,
            endpoint_cap_exception_approved_at_utc=exception_approved_at,
            reasoning_efforts=efforts,
            required_quantizations=required_quantizations,
        )


@dataclass(frozen=True)
class EndpointAssessment:
    """Qualification decision for one catalog endpoint."""

    accepted: bool
    reasons: tuple[str, ...]
    endpoint_tag: str | None
    output_token_parameter: str | None
    temperature_exposed: bool
    quality_rank: int
    precision_rank: int
    completion_price: float
    prompt_price: float
    uptime_last_1d: float
    effective_output_capacity_tokens: float | None
    endpoint_cap_status: str
    latency_last_30m: float
    throughput_last_30m: float


def select_endpoint(
    capability: ModelCapability,
    endpoints: Sequence[Mapping[str, Any]],
    *,
    largest_prompt_tokens: int,
) -> dict[str, Any]:
    """Select the unique best qualifying endpoint under the frozen policy.

    Args:
        capability: Author-evidenced model limits and reasoning policy.
        endpoints: Complete endpoint catalog entries for the model.
        largest_prompt_tokens: Conservative upper bound for the largest prompt.

    Returns:
        Frozen selected-route record and rejected-endpoint diagnostics.
    """
    if largest_prompt_tokens < 1:
        raise ValueError("largest_prompt_tokens must be positive")
    assessments = [
        (endpoint, assess_endpoint(capability, endpoint, largest_prompt_tokens=largest_prompt_tokens))
        for endpoint in endpoints
    ]
    accepted = [(endpoint, assessment) for endpoint, assessment in assessments if assessment.accepted]
    if not accepted:
        reasons = [
            {"endpoint_tag": assessment.endpoint_tag, "reasons": list(assessment.reasons)}
            for _, assessment in assessments
        ]
        raise ValueError(f"{capability.model_id}: no qualifying endpoint: {json.dumps(reasons, sort_keys=True)}")
    endpoint, assessment = min(accepted, key=lambda item: _selection_key(capability, item[1]))
    selected = {
        "model_id": capability.model_id,
        "reasoning_setting": capability.reasoning_setting,
        "reasoning_efforts": list(capability.reasoning_efforts or (capability.reasoning_setting,)),
        "configured_output_tokens": capability.configured_output_tokens,
        "output_limit_source": capability.output_limit_source,
        "author_native_max_output_tokens": capability.author_native_max_output_tokens,
        "native_context_tokens": capability.native_context_tokens,
        "author_evidence_url": capability.evidence_url,
        "author_evidence_retrieved_at_utc": capability.evidence_retrieved_at_utc,
        "largest_benchmark_prompt_tokens": largest_prompt_tokens,
        "endpoint_tag": assessment.endpoint_tag,
        "provider": endpoint.get("provider_name"),
        "endpoint_model_id": endpoint.get("model_id"),
        "endpoint_served_model_id": _served_model_id(endpoint),
        "endpoint_name": endpoint.get("name"),
        "endpoint_max_completion_tokens": endpoint.get("max_completion_tokens"),
        "endpoint_context_length": endpoint.get("context_length"),
        "endpoint_max_prompt_tokens": endpoint.get("max_prompt_tokens"),
        "endpoint_effective_output_capacity_tokens": assessment.effective_output_capacity_tokens,
        "endpoint_cap_status": assessment.endpoint_cap_status,
        "endpoint_cap_exception": (
            {
                "allowed_endpoint_tags": list(capability.undisclosed_cap_allowed_endpoint_tags),
                "rationale": capability.endpoint_cap_exception_rationale,
                "approved_at_utc": capability.endpoint_cap_exception_approved_at_utc,
            }
            if assessment.endpoint_cap_status == "undisclosed_first_party_exception"
            else None
        ),
        "output_token_parameter": assessment.output_token_parameter,
        "temperature": 0.0 if assessment.temperature_exposed else None,
        "temperature_behavior": "explicit_zero" if assessment.temperature_exposed else "not_exposed",
        "quantization": endpoint.get("quantization"),
        "required_quantizations": list(capability.required_quantizations),
        "pricing": endpoint.get("pricing"),
        "uptime_last_1d": endpoint.get("uptime_last_1d"),
        "supported_parameters": endpoint.get("supported_parameters"),
        "selection_rationale": {
            "quality_rank": assessment.quality_rank,
            "precision_rank": assessment.precision_rank,
            "completion_price": _finite_or_none(assessment.completion_price),
            "prompt_price": _finite_or_none(assessment.prompt_price),
            "uptime_last_1d": assessment.uptime_last_1d,
            "latency_last_30m": _finite_or_none(assessment.latency_last_30m),
            "throughput_last_30m": _finite_or_none(assessment.throughput_last_30m),
            "tie_break_endpoint_tag": assessment.endpoint_tag,
        },
    }
    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    selected["route_revision"] = "route-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    selected["rejected_endpoints"] = [
        {"endpoint_tag": item.endpoint_tag, "reasons": list(item.reasons)}
        for _, item in assessments
        if not item.accepted
    ]
    return selected


def assess_endpoint(  # noqa: C901 - fail-closed endpoint capability matrix
    capability: ModelCapability,
    endpoint: Mapping[str, Any],
    *,
    largest_prompt_tokens: int,
) -> EndpointAssessment:
    """Apply output-budget, context, uptime, reasoning, and parameter gates.

    Args:
        capability: Author-evidenced model limits and reasoning policy.
        endpoint: One raw OpenRouter endpoint catalog entry.
        largest_prompt_tokens: Conservative upper bound for the largest prompt.

    Returns:
        Qualification decision and deterministic ranking fields.
    """
    reasons: list[str] = []
    tag = _optional_text(endpoint.get("tag"))
    if tag is None:
        reasons.append("endpoint tag is missing")

    raw_completion_limit = endpoint.get("max_completion_tokens")
    completion_limit = _numeric(raw_completion_limit)
    endpoint_cap_status = "catalog_numeric"
    if completion_limit is None:
        if (
            raw_completion_limit is None
            and capability.model_id == _UNDISCLOSED_CAP_EXCEPTION_MODEL
            and capability.endpoint_cap_policy == "undisclosed_first_party_exception"
            and tag in set(capability.undisclosed_cap_allowed_endpoint_tags)
            and tag in _UNDISCLOSED_CAP_EXCEPTION_TAGS
        ):
            endpoint_cap_status = "undisclosed_first_party_exception"
        else:
            reasons.append("max_completion_tokens is missing or nonnumeric")
    elif completion_limit < capability.configured_output_tokens:
        reasons.append(
            f"max_completion_tokens {completion_limit:g} is below configured output budget "
            f"{capability.configured_output_tokens}"
        )

    required_context = largest_prompt_tokens + capability.configured_output_tokens
    if capability.native_context_tokens is not None and capability.native_context_tokens < required_context:
        reasons.append(f"author-documented context must be at least {required_context}")
    context = _numeric(endpoint.get("context_length"))
    if context is None or context < required_context:
        reasons.append(f"context_length must be at least {required_context}")
    max_prompt = _numeric(endpoint.get("max_prompt_tokens"))
    if max_prompt is not None and max_prompt < largest_prompt_tokens:
        reasons.append(f"max_prompt_tokens must be at least {largest_prompt_tokens}")

    uptime = _numeric(endpoint.get("uptime_last_1d"))
    if uptime is None or uptime < MINIMUM_UPTIME_LAST_1D:
        reasons.append(f"uptime_last_1d must be at least {MINIMUM_UPTIME_LAST_1D:g}")

    latency_last_30m = _metric_p50(endpoint.get("latency_last_30m"))
    throughput_last_30m = _metric_p50(endpoint.get("throughput_last_30m"))
    if capability.model_id == "z-ai/glm-5.2":
        if latency_last_30m is None or latency_last_30m < 0:
            reasons.append("GLM selection requires nonnegative latency_last_30m")
        if throughput_last_30m is None or throughput_last_30m < 0:
            reasons.append("GLM selection requires nonnegative throughput_last_30m")

    supported_raw = endpoint.get("supported_parameters")
    supported = {str(value) for value in supported_raw} if isinstance(supported_raw, list) else set()
    if not supported.intersection(capability.reasoning_parameter_names):
        reasons.append(
            "configured reasoning control is not exposed; expected one of "
            + ", ".join(capability.reasoning_parameter_names)
        )
    if capability.model_id == "z-ai/glm-5.2":
        output_parameter = "max_tokens" if "max_tokens" in supported else None
    else:
        output_parameter = next((name for name in OUTPUT_PARAMETERS if name in supported), None)
    if output_parameter is None:
        if capability.model_id == "z-ai/glm-5.2":
            reasons.append("GLM requires the exact max_tokens parameter")
        else:
            reasons.append("neither max_tokens nor max_completion_tokens is supported")

    quantization = str(endpoint.get("quantization") or "unknown").lower()
    if capability.required_quantizations and quantization not in set(capability.required_quantizations):
        reasons.append(
            f"quantization {quantization} is not allowed; expected one of "
            + ", ".join(capability.required_quantizations)
        )
    raw_pricing = endpoint.get("pricing")
    pricing: Mapping[str, Any] = raw_pricing if isinstance(raw_pricing, Mapping) else {}
    effective_capacity = (
        None
        if completion_limit is None or context is None
        else min(completion_limit, max(0.0, context - largest_prompt_tokens))
    )
    return EndpointAssessment(
        accepted=not reasons,
        reasons=tuple(reasons),
        endpoint_tag=tag,
        output_token_parameter=output_parameter,
        temperature_exposed="temperature" in supported,
        quality_rank=_quality_rank(capability, endpoint, tag),
        precision_rank=_PRECISION_RANK.get(quantization, 0),
        completion_price=_price(pricing.get("completion")),
        prompt_price=_price(pricing.get("prompt")),
        uptime_last_1d=-math.inf if uptime is None else uptime,
        effective_output_capacity_tokens=effective_capacity,
        endpoint_cap_status=endpoint_cap_status,
        latency_last_30m=math.inf if latency_last_30m is None else latency_last_30m,
        throughput_last_30m=-math.inf if throughput_last_30m is None else throughput_last_30m,
    )


def _selection_key(capability: ModelCapability, assessment: EndpointAssessment) -> tuple[Any, ...]:
    if capability.model_id == "z-ai/glm-5.2":
        return (
            assessment.latency_last_30m,
            -assessment.throughput_last_30m,
            -assessment.uptime_last_1d,
            assessment.completion_price,
            assessment.prompt_price,
            assessment.endpoint_tag or "",
        )
    return (
        assessment.quality_rank,
        -assessment.precision_rank,
        -assessment.uptime_last_1d,
        assessment.completion_price,
        assessment.prompt_price,
        assessment.endpoint_tag or "",
    )


def _quality_rank(capability: ModelCapability, endpoint: Mapping[str, Any], tag: str | None) -> int:
    if tag in capability.author_endpoint_tags:
        return 0
    if tag in capability.authorized_endpoint_tags:
        return 1
    if _optional_text(endpoint.get("provider_name")) in capability.authorized_provider_names:
        return 1
    return 2


def _endpoint_completion_limit_policy(
    payload: Mapping[str, Any], *, model_id: str, output_limit_source: str
) -> tuple[str, tuple[str, ...], str | None, str | None]:
    raw = payload.get("endpoint_completion_limit")
    if raw is None:
        return "catalog_numeric", (), None, None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{model_id}: endpoint_completion_limit must be an object")
    policy = _required_text(raw, "policy")
    if policy not in ENDPOINT_CAP_STATUSES:
        raise ValueError(f"{model_id}: unsupported endpoint completion-limit policy: {policy}")
    if policy == "catalog_numeric":
        extra = set(raw) - {"policy"}
        if extra:
            raise ValueError(f"{model_id}: catalog_numeric policy cannot define exception fields")
        return policy, (), None, None
    tags = _text_tuple(raw.get("allowed_endpoint_tags") or [])
    rationale = _required_text(raw, "rationale")
    approved_at = _required_text(raw, "approved_at_utc")
    if model_id != _UNDISCLOSED_CAP_EXCEPTION_MODEL:
        raise ValueError(f"{model_id}: undisclosed-cap exception is restricted to {_UNDISCLOSED_CAP_EXCEPTION_MODEL}")
    if output_limit_source != "benchmark_floor":
        raise ValueError(f"{model_id}: undisclosed-cap exception requires the benchmark output floor")
    if frozenset(tags) != _UNDISCLOSED_CAP_EXCEPTION_TAGS or len(tags) != len(_UNDISCLOSED_CAP_EXCEPTION_TAGS):
        raise ValueError(f"{model_id}: undisclosed-cap exception must allow only the exact xai endpoint tag")
    return policy, tags, rationale, approved_at


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    value = _optional_text(payload.get(name))
    if value is None:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _text_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(_optional_text(item) is None for item in value):
        raise ValueError("expected a list of nonempty strings")
    return tuple(str(item).strip() for item in value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if parsed < 1 or parsed != value:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_p50(value: Any) -> float | None:
    """Return a catalog metric's p50, retaining numeric fixture compatibility."""
    if isinstance(value, Mapping):
        return _numeric(value.get("p50"))
    return _numeric(value)


def _price(value: Any) -> float:
    parsed = _numeric(value)
    return math.inf if parsed is None or parsed < 0 else parsed


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _served_model_id(endpoint: Mapping[str, Any]) -> str | None:
    """Return the versioned model slug reported by router metadata."""
    name = _optional_text(endpoint.get("name"))
    if name is not None and " | " in name:
        served = _optional_text(name.rsplit(" | ", 1)[1])
        if served is not None:
            return served
    return _optional_text(endpoint.get("model_id"))
