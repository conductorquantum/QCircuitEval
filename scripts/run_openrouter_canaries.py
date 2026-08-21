#!/usr/bin/env python3
"""Run one output-qualified, exact-route sentinel canary per configuration."""

from __future__ import annotations

import argparse
import ast
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qceval.models import ProviderRequest
from qceval.production.campaign import CONFIGURATION_COUNT
from qceval.providers.openrouter import OpenRouterProvider


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=Path(".env"))
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    selection = _read_json(args.selection)
    configurations = selection.get("configurations")
    if (
        selection.get("campaign_eligible") is not True
        or not isinstance(configurations, Mapping)
        or len(configurations) != CONFIGURATION_COUNT
    ):
        parser.error(f"all {CONFIGURATION_COUNT} configurations must qualify before canaries")
    api_key = os.environ.get("OPENROUTER_API_KEY") or _dotenv_value(args.api_key_file, "OPENROUTER_API_KEY")
    if not api_key:
        parser.error("OPENROUTER_API_KEY is missing")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("", encoding="utf-8")
    failures = []
    for config_id in sorted(configurations):
        route = configurations[config_id]
        if not isinstance(route, Mapping):
            parser.error(f"{config_id}: selected route is malformed")
        model_id = str(route.get("model_id") or "")
        result = _run_canary(model_id, route, api_key=api_key, timeout=args.timeout)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if result["status"] != "passed":
            failures.append(config_id)
    summary = {
        "kind": "summary",
        "created_at_utc": _now(),
        "base_models": len({str(route.get("model_id")) for route in configurations.values()}),
        "configurations": len(configurations),
        "passed": len(configurations) - len(failures),
        "failed": failures,
        "benchmark_denominator_member": False,
    }
    with args.out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 2


def _run_canary(model_id: str, route: Mapping[str, Any], *, api_key: str, timeout: float) -> dict[str, Any]:
    # A long random hex string resembles a credential and can trigger provider
    # secret filters. Keep requests unique with a harmless six-digit integer.
    nonce = 100_000 + uuid.uuid4().int % 900_000
    provider = OpenRouterProvider(
        api_key=api_key,
        timeout=timeout,
        temperature=route.get("temperature"),
        reasoning_effort=None if route.get("reasoning_setting") == "enabled" else str(route["reasoning_setting"]),
        reasoning_enabled=True if route.get("reasoning_setting") == "enabled" else None,
        endpoint_tag=str(route["endpoint_tag"]),
        max_output_tokens=int(route["configured_output_tokens"]),
        output_limit_source=str(route["output_limit_source"]),
        endpoint_cap_status=str(route["endpoint_cap_status"]),
        output_token_parameter=str(route["output_token_parameter"]),
        route_revision=str(route["route_revision"]),
        configuration_id=str(route["configuration_id"]),
        max_retries=3,
    )
    request = ProviderRequest(
        task_id=f"canary-{nonce}",
        framework="qiskit",
        model=model_id,
        entry_point="sentinel",
        prompt=f"Reply with exactly this harmless token and nothing else: ROUTE_CANARY_{nonce}",
    )
    response = provider.generate(request)
    result = {
        "kind": "canary",
        "created_at_utc": _now(),
        "model_id": model_id,
        "configuration_id": route.get("configuration_id"),
        "endpoint_tag": route.get("endpoint_tag"),
        "route_revision": route.get("route_revision"),
        "configured_output_tokens": route.get("configured_output_tokens"),
        "output_limit_source": route.get("output_limit_source"),
        "endpoint_cap_status": route.get("endpoint_cap_status"),
        "author_native_max_output_tokens": route.get("author_native_max_output_tokens"),
        "output_token_parameter": route.get("output_token_parameter"),
        "generation_id": response.metadata.get("generation_id"),
        "usage": None if response.usage is None else response.usage.to_dict(),
        "provider_response_metadata": response.metadata,
        "error": response.error,
        "status": "failed",
    }
    response_route = response.metadata.get("route")
    if (
        response.error is not None
        or not isinstance(response_route, Mapping)
        or response_route.get("route_verified") is not True
    ):
        if result["error"] is None:
            result["error"] = "provider response did not contain a route-verified sentinel response"
        return result
    expected_provider = route.get("provider")
    expected_model = route.get("endpoint_served_model_id")
    if not isinstance(expected_model, str) or not expected_model:
        result["error"] = "selected route omitted endpoint_served_model_id"
        return result
    if response_route.get("selected_provider") != expected_provider:
        result["error"] = (
            f"selected provider {response_route.get('selected_provider')!r} does not match {expected_provider!r}"
        )
        return result
    if response_route.get("selected_model") != expected_model:
        result["error"] = f"selected model {response_route.get('selected_model')!r} does not match {expected_model!r}"
        return result
    if not _sentinel_matches(response.raw_response, nonce):
        result["error"] = "raw response did not contain the requested harmless sentinel token"
        return result
    result["status"] = "passed"
    return result


def _sentinel_matches(raw_response: Mapping[str, Any] | None, nonce: int) -> bool:
    if not isinstance(raw_response, Mapping):
        return False
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return False
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        return False
    return f"ROUTE_CANARY_{nonce}" in message["content"]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selection must contain a JSON object")
    return payload


def _dotenv_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if value[:1] in {'"', "'"}:
            parsed = ast.literal_eval(value)
            return str(parsed).strip() or None
        return value or None
    return None


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
