"""Registry for built-in provider implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qceval.providers.base import Provider
from qceval.providers.coda import CodaProvider
from qceval.providers.coda.provider import DEFAULT_CODA_AGENTS_URL, DEFAULT_CODA_TIMEOUT_SECONDS
from qceval.providers.openrouter import DEFAULT_CHAT_COMPLETIONS_URL, OpenRouterProvider
from qceval.providers.smoke import SmokeProvider

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_TEMPERATURE = 0.2


def provider_names() -> tuple[str, ...]:
    """Return names accepted by :func:`build_provider`."""
    return ("smoke", "openrouter", "coda")


def build_provider(name: str, *, model: str | None, config: Mapping[str, Any] | None = None) -> Provider:
    """Construct a built-in provider from CLI-style configuration.

    Args:
        name: Provider name returned by :func:`provider_names`.
        model: Model identifier passed to providers that need one.
        config: Provider-specific settings.  Supported keys include
            ``smoke_mode``, ``openrouter_api_key``, ``openrouter_base_url``,
            ``coda_api_key``, ``coda_agents_url``, ``coda_mode``,
            ``coda_fast``, ``coda_prefer_structured_response``, ``timeout``,
            ``temperature``, ``reasoning_effort``, ``reasoning_enabled``,
            ``openrouter_endpoint_tag``, ``openrouter_max_output_tokens``,
            ``openrouter_output_limit_source``,
            ``openrouter_endpoint_cap_status``,
            ``openrouter_output_token_parameter``, ``openrouter_route_revision``,
            ``configuration_id``,
            ``max_retries``, ``retry_base_delay``, and ``retry_max_delay``.

    Returns:
        Provider implementation matching ``name``.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    provider_config = dict(config or {})
    if name == "smoke":
        mode = str(_get_config(provider_config, "smoke_mode", "canonical"))
        return SmokeProvider(
            mode=mode,
            model=model or "smoke-canonical",
            reasoning_effort=_get_optional_str(provider_config, "reasoning_effort"),
            reasoning_enabled=_get_optional_bool(provider_config, "reasoning_enabled"),
            configuration_id=_get_optional_str(provider_config, "configuration_id"),
        )
    if name == "openrouter":
        endpoint_tag = _get_optional_str(provider_config, "openrouter_endpoint_tag")
        temperature = (
            provider_config.get("temperature")
            if endpoint_tag is not None
            else _get_config(provider_config, "temperature", DEFAULT_TEMPERATURE)
        )
        return OpenRouterProvider(
            api_key=_get_optional_str(provider_config, "openrouter_api_key"),
            base_url=str(_get_config(provider_config, "openrouter_base_url", DEFAULT_CHAT_COMPLETIONS_URL)),
            timeout=float(_get_config(provider_config, "timeout", DEFAULT_TIMEOUT_SECONDS)),
            temperature=None if temperature is None else float(temperature),
            reasoning_effort=_get_optional_str(provider_config, "reasoning_effort"),
            reasoning_enabled=_get_optional_bool(provider_config, "reasoning_enabled"),
            endpoint_tag=endpoint_tag,
            max_output_tokens=_get_optional_int(provider_config, "openrouter_max_output_tokens"),
            output_limit_source=_get_optional_str(provider_config, "openrouter_output_limit_source"),
            endpoint_cap_status=_get_optional_str(provider_config, "openrouter_endpoint_cap_status"),
            output_token_parameter=_get_optional_str(provider_config, "openrouter_output_token_parameter"),
            route_revision=_get_optional_str(provider_config, "openrouter_route_revision"),
            configuration_id=_get_optional_str(provider_config, "configuration_id"),
            max_retries=int(_get_config(provider_config, "max_retries", 3)),
            retry_base_delay=float(_get_config(provider_config, "retry_base_delay", 1.0)),
            retry_max_delay=float(_get_config(provider_config, "retry_max_delay", 60.0)),
        )
    if name == "coda":
        return CodaProvider(
            api_key=_get_optional_str(provider_config, "coda_api_key"),
            agents_url=str(_get_config(provider_config, "coda_agents_url", DEFAULT_CODA_AGENTS_URL)),
            timeout=float(_get_config(provider_config, "timeout", DEFAULT_CODA_TIMEOUT_SECONDS)),
            mode=str(_get_config(provider_config, "coda_mode", "build")),
            fast=bool(_get_config(provider_config, "coda_fast", False)),
            prefer_structured_response=bool(_get_config(provider_config, "coda_prefer_structured_response", False)),
            max_retries=int(_get_config(provider_config, "max_retries", 3)),
            retry_base_delay=float(_get_config(provider_config, "retry_base_delay", 1.0)),
            retry_max_delay=float(_get_config(provider_config, "retry_max_delay", 60.0)),
        )
    raise ValueError(f"unknown provider: {name}")


def _get_config(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


def _get_optional_str(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_optional_bool(config: Mapping[str, Any], key: str) -> bool | None:
    value = config.get(key)
    return None if value is None else bool(value)


def _get_optional_int(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    return None if value is None else int(value)
