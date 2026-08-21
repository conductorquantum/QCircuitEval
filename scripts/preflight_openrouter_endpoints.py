#!/usr/bin/env python3
"""Archive endpoint catalogs and apply minimum-output production gates."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qceval.core.bench import SUPPORTED_FRAMEWORKS, Adaptor
from qceval.models import Framework, Suite
from qceval.production.campaign import (
    BASE_MODEL_COUNT,
    CONFIGURATION_COUNT,
    expand_configurations,
    validate_registry_efforts,
)
from qceval.production.endpoints import ModelCapability, select_endpoint

DEFAULT_REGISTRY = Path("production/models.full-cap.json")
SUITES: tuple[Suite, ...] = ("core", "qec")


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - fail-closed endpoint qualification
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=Path(".env"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--exclude-endpoint",
        action="append",
        default=[],
        metavar="MODEL_ID=ENDPOINT_TAG",
        help="Exclude an objectively failed frozen route during outage recovery; repeat as needed.",
    )
    args = parser.parse_args(argv)

    registry = _read_json(args.registry)
    models = registry.get("models")
    if not isinstance(models, list) or len(models) != BASE_MODEL_COUNT:
        parser.error(f"capability registry must contain exactly {BASE_MODEL_COUNT} base models")
    try:
        validate_registry_efforts(models)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        exclusions = _endpoint_exclusions(args.exclude_endpoint, models)
    except ValueError as exc:
        parser.error(str(exc))
    api_key = os.environ.get("OPENROUTER_API_KEY") or _dotenv_value(args.api_key_file, "OPENROUTER_API_KEY")
    if not api_key:
        parser.error("OPENROUTER_API_KEY is missing")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    captured_at = _now()
    prompt_bound = _largest_prompt_bound()
    catalog_models = []
    catalog_by_model: dict[str, dict[str, Any]] = {}
    for raw_model in models:
        if not isinstance(raw_model, dict) or not isinstance(raw_model.get("model_id"), str):
            parser.error("every capability entry must contain model_id")
        model_id = raw_model["model_id"]
        catalog = _fetch_catalog(model_id, api_key=api_key, timeout=args.timeout)
        catalog_models.append(catalog)
        catalog_by_model[model_id] = catalog

    raw_snapshot = {
        "schema_version": "1",
        "captured_at_utc": captured_at,
        "source": "https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints",
        "models": catalog_models,
    }
    raw_path = args.out_dir / "openrouter-endpoint-catalog.raw.json"
    _write_json(raw_path, raw_snapshot)

    selections: dict[str, Any] = {}
    failures: dict[str, list[str]] = {}
    for raw_model in models:
        model_id = str(raw_model["model_id"])
        model_failures: list[str] = []
        capability: ModelCapability | None = None
        try:
            capability = ModelCapability.from_mapping(raw_model)
        except ValueError as exc:
            model_failures.append(str(exc))
        catalog = catalog_by_model[model_id]
        if catalog.get("http_status") != 200:
            model_failures.append(f"endpoint catalog HTTP status {catalog.get('http_status')}")
        response = catalog.get("response")
        data = response.get("data") if isinstance(response, dict) else None
        endpoints = data.get("endpoints") if isinstance(data, dict) else None
        if not isinstance(endpoints, list):
            model_failures.append("endpoint catalog omitted data.endpoints")
        if capability is not None and isinstance(endpoints, list):
            try:
                eligible_endpoints = [
                    endpoint
                    for endpoint in endpoints
                    if not isinstance(endpoint, dict) or endpoint.get("tag") not in exclusions.get(model_id, set())
                ]
                selections[model_id] = select_endpoint(
                    capability,
                    eligible_endpoints,
                    largest_prompt_tokens=prompt_bound["utf8_byte_token_upper_bound"],
                )
            except ValueError as exc:
                model_failures.append(str(exc))
        if model_failures:
            failures[model_id] = model_failures

    selection_path = args.out_dir / "openrouter-endpoint-selection.json"
    configurations = expand_configurations(selections) if not failures and len(selections) == BASE_MODEL_COUNT else {}
    selection_payload = {
        "schema_version": "2",
        "selected_at_utc": _now(),
        "campaign_eligible": not failures
        and len(selections) == BASE_MODEL_COUNT
        and len(configurations) == CONFIGURATION_COUNT,
        "policy": {
            "minimum_uptime_last_1d_percent": 95,
            "output_limit_policy": "author_native_or_128000_benchmark_floor_with_frozen_grok_exception",
            "require_numeric_endpoint_completion_limit_by_default": True,
            "require_singular_endpoint": True,
            "one_endpoint_per_base_model": True,
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "largest_benchmark_prompt": prompt_bound,
        "route_recovery_exclusions": {model_id: sorted(tags) for model_id, tags in sorted(exclusions.items())},
        "models": selections,
        "configurations": configurations,
        "failures": failures,
    }
    _write_json(selection_path, selection_payload)

    hashes = {
        str(args.registry): _sha256(args.registry),
        raw_path.name: _sha256(raw_path),
        selection_path.name: _sha256(selection_path),
    }
    hashes_path = args.out_dir / "preflight-artifact-hashes.json"
    _write_json(
        hashes_path,
        {
            "schema_version": "1",
            "created_at_utc": _now(),
            "sha256": hashes,
        },
    )
    print(json.dumps({"campaign_eligible": selection_payload["campaign_eligible"], "failures": failures}, indent=2))
    return 0 if selection_payload["campaign_eligible"] else 2


def _endpoint_exclusions(raw_values: list[str], models: list[Any]) -> dict[str, set[str]]:
    known_models = {
        str(model["model_id"]) for model in models if isinstance(model, dict) and isinstance(model.get("model_id"), str)
    }
    exclusions: dict[str, set[str]] = {}
    for raw in raw_values:
        model_id, separator, endpoint_tag = raw.partition("=")
        if not separator or not model_id or not endpoint_tag:
            raise ValueError("--exclude-endpoint must use MODEL_ID=ENDPOINT_TAG")
        if model_id not in known_models:
            raise ValueError(f"--exclude-endpoint names unknown model {model_id!r}")
        exclusions.setdefault(model_id, set()).add(endpoint_tag)
    return exclusions


def _fetch_catalog(model_id: str, *, api_key: str, timeout: float) -> dict[str, Any]:
    author, separator, slug = model_id.partition("/")
    if not separator or not author or not slug:
        return {"model_id": model_id, "captured_at_utc": _now(), "http_status": None, "error": "invalid model id"}
    url = f"https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return {
                "model_id": model_id,
                "captured_at_utc": _now(),
                "url": url,
                "http_status": response.status,
                "response": json.loads(body),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"unparsed_body": body}
        return {
            "model_id": model_id,
            "captured_at_utc": _now(),
            "url": url,
            "http_status": exc.code,
            "response": parsed,
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "model_id": model_id,
            "captured_at_utc": _now(),
            "url": url,
            "http_status": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _largest_prompt_bound() -> dict[str, Any]:
    adapter = Adaptor()
    largest: tuple[int, Suite, Framework, str, str] | None = None
    for suite in SUITES:
        for framework in SUPPORTED_FRAMEWORKS:
            for task in adapter.load_tasks(framework, suite):
                size = len(task.prompt.encode("utf-8"))
                candidate = (size, suite, framework, task.task_id, hashlib.sha256(task.prompt.encode()).hexdigest())
                if largest is None or candidate > largest:
                    largest = candidate
    if largest is None:
        raise ValueError("benchmark contains no prompts")
    size, suite, framework, task_id, prompt_sha256 = largest
    return {
        "bound_method": "utf8_bytes_as_conservative_token_upper_bound",
        "utf8_byte_token_upper_bound": size,
        "suite": suite,
        "framework": framework,
        "task_id": task_id,
        "prompt_sha256": prompt_sha256,
    }


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    sys.exit(main())
