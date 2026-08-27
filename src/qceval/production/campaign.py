"""Frozen identity rules for the selected production Pass@1 campaign."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

CAMPAIGN_NAME = os.environ.get("QCEVAL_PRODUCTION_CAMPAIGN", "max-reasoning-pass1")
BENCHMARK_CONTENT_COMMIT = "37bffc7ae6b98ecd2c78bdfba1d249c3c15ded70"

if CAMPAIGN_NAME == "max-reasoning-pass1":
    CAMPAIGN_SCHEMA_VERSION = "qceval.max_reasoning_pass1.v1"
    EFFORTS_BY_MODEL: dict[str, tuple[str, ...]] = {
        "openai/gpt-5.6-sol": ("max",),
        "x-ai/grok-4.6": ("xhigh",),
        "anthropic/claude-opus-5": ("max",),
        "anthropic/claude-fable-5": ("max",),
        "google/gemini-3.1-pro-preview": ("high",),
        "moonshotai/kimi-k3": ("max",),
        "z-ai/glm-5.2": ("max",),
        "google/gemma-4-31b-it": ("enabled",),
        "nvidia/nemotron-3-ultra-550b-a55b": ("high",),
    }
    OUTPUT_POLICY_BY_MODEL: dict[str, tuple[int, str, str]] = {
        "openai/gpt-5.6-sol": (128_000, "author_native", "catalog_numeric"),
        "x-ai/grok-4.6": (128_000, "benchmark_floor", "undisclosed_first_party_exception"),
        "anthropic/claude-opus-5": (128_000, "author_native", "catalog_numeric"),
        "anthropic/claude-fable-5": (128_000, "author_native", "catalog_numeric"),
        "google/gemini-3.1-pro-preview": (65_536, "author_native", "catalog_numeric"),
        "moonshotai/kimi-k3": (128_000, "benchmark_floor", "catalog_numeric"),
        "z-ai/glm-5.2": (131_072, "author_native", "catalog_numeric"),
        "google/gemma-4-31b-it": (128_000, "benchmark_floor", "catalog_numeric"),
        "nvidia/nemotron-3-ultra-550b-a55b": (128_000, "benchmark_floor", "catalog_numeric"),
    }
    REUSABLE_CONFIGURATION_IDS: tuple[str, ...] = ()
elif CAMPAIGN_NAME == "prompt-effort-sweep":
    CAMPAIGN_SCHEMA_VERSION = "qceval.prompt_effort_pass1.v1"
    EFFORTS_BY_MODEL = {
        "anthropic/claude-fable-5": ("low", "medium", "high", "xhigh", "max"),
        "anthropic/claude-opus-5": ("low", "medium", "high", "xhigh", "max"),
        "openai/gpt-5.6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
        "openai/gpt-5.6-terra": ("none", "low", "medium", "high", "xhigh", "max"),
        "openai/gpt-5.6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
    }
    OUTPUT_POLICY_BY_MODEL = dict.fromkeys(
        EFFORTS_BY_MODEL,
        (128_000, "author_native", "catalog_numeric"),
    )
    REUSABLE_CONFIGURATION_IDS = (
        "anthropic-claude-fable-5__effort-max",
        "anthropic-claude-opus-5__effort-max",
        "openai-gpt-5-6-sol__effort-low",
        "openai-gpt-5-6-sol__effort-max",
    )
else:
    raise RuntimeError(f"unsupported QCEVAL_PRODUCTION_CAMPAIGN: {CAMPAIGN_NAME}")

# Retained for the archived max-only importer. Current campaigns bind reused
# outputs by complete configuration ID instead.
HISTORICAL_MAX_IMPORT_MODELS: tuple[str, ...] = ()
BASE_MODEL_COUNT = len(EFFORTS_BY_MODEL)
CONFIGURATION_COUNT = sum(len(efforts) for efforts in EFFORTS_BY_MODEL.values())
FRAMEWORKS_PER_CONFIGURATION = 4
SHARD_COUNT = CONFIGURATION_COUNT * FRAMEWORKS_PER_CONFIGURATION
TASKS_PER_CONFIGURATION = 280
ASSIGNMENT_COUNT = CONFIGURATION_COUNT * TASKS_PER_CONFIGURATION
HISTORICAL_IMPORT_ASSIGNMENT_COUNT = len(REUSABLE_CONFIGURATION_IDS) * TASKS_PER_CONFIGURATION
FRESH_ASSIGNMENT_COUNT = ASSIGNMENT_COUNT - HISTORICAL_IMPORT_ASSIGNMENT_COUNT
REQUESTS_PER_ENDPOINT = 2
MAXIMUM_PROVIDER_REQUESTS = min(12, BASE_MODEL_COUNT * REQUESTS_PER_ENDPOINT)


def model_slug(model_id: str) -> str:
    """Return the canonical queue-safe model slug.

    Args:
        model_id: Provider-qualified model identifier.

    Returns:
        Queue-safe model slug.
    """
    return "".join(character if character.isalnum() else "-" for character in model_id).strip("-")


def configuration_id(model_id: str, effort: str) -> str:
    """Return the required human-readable configuration identifier.

    Args:
        model_id: Provider-qualified model identifier.
        effort: Frozen reasoning-effort level.

    Returns:
        Canonical configuration identifier.
    """
    return f"{model_slug(model_id)}__effort-{effort}"


def configuration_identity_payload(
    *,
    model_id: str,
    effort: str,
    endpoint_tag: str,
    configured_output_tokens: int,
    output_token_parameter: str,
    temperature_behavior: str,
    route_revision: str,
) -> dict[str, Any]:
    """Return every field that separates generated outputs and caches.

    Args:
        model_id: Provider-qualified model identifier.
        effort: Frozen reasoning-effort level.
        endpoint_tag: Exact selected endpoint tag.
        configured_output_tokens: Full requested output ceiling.
        output_token_parameter: Exact parameter spelling sent to the endpoint.
        temperature_behavior: Explicit-zero or omission behavior.
        route_revision: Immutable selected-route revision.

    Returns:
        JSON-compatible complete configuration identity.
    """
    return {
        "configuration_id": configuration_id(model_id, effort),
        "model_id": model_id,
        "reasoning_effort": effort,
        "endpoint_tag": endpoint_tag,
        "configured_output_tokens": configured_output_tokens,
        "output_token_parameter": output_token_parameter,
        "temperature_behavior": temperature_behavior,
        "route_revision": route_revision,
    }


def configuration_identity_sha256(**values: Any) -> str:
    """Hash the complete immutable configuration identity.

    Args:
        **values: Fields accepted by :func:`configuration_identity_payload`.

    Returns:
        Lowercase SHA-256 digest.
    """
    payload = configuration_identity_payload(**values)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_registry_efforts(models: Sequence[Mapping[str, Any]]) -> None:
    """Reject any roster or reasoning-setting drift from the frozen campaign.

    Args:
        models: Capability-registry model entries.
    """
    actual: dict[str, tuple[str, ...]] = {}
    for model in models:
        model_id = model.get("model_id")
        efforts = model.get("reasoning_efforts")
        if (
            not isinstance(model_id, str)
            or not isinstance(efforts, list)
            or not all(isinstance(value, str) for value in efforts)
        ):
            raise ValueError("every campaign model requires model_id and reasoning_efforts")
        actual[model_id] = tuple(efforts)
    if actual != EFFORTS_BY_MODEL:
        raise ValueError(f"capability registry does not match the frozen {CONFIGURATION_COUNT}-configuration roster")


def expand_configurations(routes: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Expand one frozen route per base model into configuration records.

    Args:
        routes: Selected endpoint route keyed by base model identifier.

    Returns:
        Configuration records keyed by canonical configuration identifier.
    """
    if set(routes) != set(EFFORTS_BY_MODEL):
        raise ValueError(f"endpoint map must contain exactly the {BASE_MODEL_COUNT} frozen base models")
    configurations: dict[str, dict[str, Any]] = {}
    for model_id, efforts in EFFORTS_BY_MODEL.items():
        route = routes[model_id]
        route_efforts = route.get("reasoning_efforts")
        if not isinstance(route_efforts, list) or tuple(route_efforts) != efforts:
            raise ValueError(f"{model_id}: selected route does not cover the frozen reasoning setting")
        for effort in efforts:
            config_id = configuration_id(model_id, effort)
            record = dict(route)
            record["configuration_id"] = config_id
            record["reasoning_setting"] = effort
            record["configuration_identity_sha256"] = configuration_identity_sha256(
                model_id=model_id,
                effort=effort,
                endpoint_tag=str(route["endpoint_tag"]),
                configured_output_tokens=int(route["configured_output_tokens"]),
                output_token_parameter=str(route["output_token_parameter"]),
                temperature_behavior=str(route["temperature_behavior"]),
                route_revision=str(route["route_revision"]),
            )
            configurations[config_id] = record
    if len(configurations) != CONFIGURATION_COUNT:
        raise ValueError(f"configuration expansion produced {len(configurations)}, expected {CONFIGURATION_COUNT}")
    return configurations


__all__ = [
    "ASSIGNMENT_COUNT",
    "BASE_MODEL_COUNT",
    "BENCHMARK_CONTENT_COMMIT",
    "CAMPAIGN_NAME",
    "CAMPAIGN_SCHEMA_VERSION",
    "CONFIGURATION_COUNT",
    "EFFORTS_BY_MODEL",
    "FRESH_ASSIGNMENT_COUNT",
    "HISTORICAL_IMPORT_ASSIGNMENT_COUNT",
    "HISTORICAL_MAX_IMPORT_MODELS",
    "MAXIMUM_PROVIDER_REQUESTS",
    "OUTPUT_POLICY_BY_MODEL",
    "REQUESTS_PER_ENDPOINT",
    "REUSABLE_CONFIGURATION_IDS",
    "SHARD_COUNT",
    "TASKS_PER_CONFIGURATION",
    "configuration_id",
    "configuration_identity_payload",
    "configuration_identity_sha256",
    "expand_configurations",
    "model_slug",
    "validate_registry_efforts",
]
