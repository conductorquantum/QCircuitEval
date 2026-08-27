#!/usr/bin/env python3
"""Generate the 36-shard schema-v2 Pass@1 queue from a complete endpoint map."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qceval.core.bench import SUPPORTED_FRAMEWORKS, Adaptor
from qceval.models import Framework, Suite
from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    CONFIGURATION_COUNT,
    SHARD_COUNT,
    configuration_id,
    model_slug,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows, assignments = generate_queue(_read_json(args.selection))
    except ValueError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    content = "".join("\t".join(row) + "\n" for row in rows)
    args.out.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "queue": str(args.out),
                "shards": len(rows),
                "assignments": assignments,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


def generate_queue(  # noqa: C901 - explicit queue protocol gates
    selection: Mapping[str, Any],
) -> tuple[list[tuple[str, ...]], int]:
    """Return exact endpoint-pinned rows and their logical assignment count."""
    if selection.get("campaign_eligible") is not True:
        raise ValueError("endpoint selection is campaign-blocked; refusing to create an official queue")
    configurations = selection.get("configurations")
    if not isinstance(configurations, Mapping) or len(configurations) != CONFIGURATION_COUNT:
        raise ValueError(f"endpoint selection must contain exactly {CONFIGURATION_COUNT} qualified configurations")
    adapter = Adaptor()
    suites: tuple[Suite, ...] = ("core", "qec")
    tasks_per_framework: dict[Framework, int] = {}
    for framework in SUPPORTED_FRAMEWORKS:
        count = 0
        for suite in suites:
            count += len(adapter.load_tasks(framework, suite))
        tasks_per_framework[framework] = count
    if set(tasks_per_framework.values()) != {70}:
        raise ValueError(f"each framework must contain exactly 70 tasks, got {tasks_per_framework}")

    rows: list[tuple[str, ...]] = []
    for config_id in sorted(configurations):
        route = configurations[config_id]
        if not isinstance(route, Mapping):
            raise ValueError(f"{config_id}: configuration route must be an object")
        model_id = str(route.get("model_id") or "")
        fields = _route_fields(model_id, config_id, route)
        for framework in SUPPORTED_FRAMEWORKS:
            job_id = f"{config_id}__pass1__{framework}"
            rows.append(
                (
                    job_id,
                    model_id,
                    fields["reasoning"],
                    "pass1",
                    framework,
                    "all",
                    "0",
                    fields["endpoint_tag"],
                    fields["max_output_tokens"],
                    fields["output_limit_source"],
                    fields["endpoint_cap_status"],
                    fields["output_token_parameter"],
                    fields["route_revision"],
                    fields["temperature_behavior"],
                    str(tasks_per_framework[framework]),
                    config_id,
                )
            )
    assignments = sum(int(row[14]) for row in rows)
    if len(rows) != SHARD_COUNT or assignments != ASSIGNMENT_COUNT:
        raise ValueError(
            f"queue expansion must be {SHARD_COUNT} shards and {ASSIGNMENT_COUNT} assignments, "
            f"got {len(rows)} and {assignments}"
        )
    return rows, assignments


def _route_fields(model_id: str, config_id: str, route: Mapping[str, Any]) -> dict[str, str]:
    required = {
        "reasoning": route.get("reasoning_setting"),
        "endpoint_tag": route.get("endpoint_tag"),
        "max_output_tokens": route.get("configured_output_tokens"),
        "output_limit_source": route.get("output_limit_source"),
        "endpoint_cap_status": route.get("endpoint_cap_status"),
        "output_token_parameter": route.get("output_token_parameter"),
        "route_revision": route.get("route_revision"),
        "temperature_behavior": route.get("temperature_behavior"),
    }
    if any(value is None or str(value).strip() == "" for value in required.values()):
        raise ValueError(f"{model_id}: selected route is incomplete")
    expected_configuration_id = configuration_id(model_id, str(required["reasoning"]))
    if config_id != expected_configuration_id or route.get("configuration_id") != config_id:
        raise ValueError(f"{model_id}: malformed configuration_id {config_id!r}")
    if required["output_token_parameter"] not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError(f"{model_id}: invalid output-token parameter")
    if required["output_limit_source"] not in {"author_native", "benchmark_floor"}:
        raise ValueError(f"{model_id}: invalid output-limit source")
    if required["endpoint_cap_status"] not in {"catalog_numeric", "undisclosed_first_party_exception"}:
        raise ValueError(f"{model_id}: invalid endpoint-cap status")
    if required["endpoint_cap_status"] != "catalog_numeric" and not (
        model_id == "x-ai/grok-4.6"
        and required["endpoint_cap_status"] == "undisclosed_first_party_exception"
        and required["endpoint_tag"] == "xai"
        and required["max_output_tokens"] == 128000
        and required["output_limit_source"] == "benchmark_floor"
    ):
        raise ValueError(f"{model_id}: endpoint cap status is outside the frozen campaign policy")
    if model_id == "z-ai/glm-5.2" and (
        required["reasoning"] != "max"
        or required["max_output_tokens"] != 131072
        or required["output_limit_source"] != "author_native"
        or required["endpoint_cap_status"] != "catalog_numeric"
        or required["output_token_parameter"] != "max_tokens"
    ):
        raise ValueError(
            "z-ai/glm-5.2 requires max reasoning, author-native max_tokens=131072, and a numeric endpoint cap"
        )
    if required["temperature_behavior"] not in {"explicit_zero", "not_exposed"}:
        raise ValueError(f"{model_id}: invalid temperature behavior")
    return {key: str(value) for key, value in required.items()}


def _slug(value: str) -> str:
    return model_slug(value)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selection must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
