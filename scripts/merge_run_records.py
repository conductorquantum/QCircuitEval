#!/usr/bin/env python3
"""Merge framework-sharded QCircuitEval JSONL outputs into one valid run."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qceval.core.io import write_output
from qceval.core.runner.records import _record_from_dict
from qceval.models import RunConfig
from qceval.reports import summarize

_FRAMEWORK_ORDER = {"qiskit": 0, "cirq": 1, "pennylane": 2, "cudaq": 3}
_SUITE_ORDER = {"core": 0, "qec": 1}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    merge_run_records(args.inputs, args.out)
    return 0


def merge_run_records(inputs: Sequence[Path], output: Path) -> None:
    """Merge compatible, complete JSONL shards and recompute their summary."""
    result_payloads: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for path in inputs:
        results, summary = _read_complete_shard(path)
        summaries.append(summary)
        for payload in results:
            key = _record_key(payload)
            previous = result_payloads.get(key)
            if previous is not None and previous != payload:
                raise ValueError(f"conflicting duplicate result {key} in {path}")
            result_payloads[key] = payload
    if not summaries:
        raise ValueError("at least one input shard is required")
    _validate_compatible(summaries)
    first = summaries[0]
    ordered_payloads = sorted(result_payloads.values(), key=_record_sort_key)
    records = [_record_from_dict(payload) for payload in ordered_payloads]
    config = _run_config(first, ordered_payloads)
    merged_summary = summarize(records, run_config=config)
    _record_route_protocol(merged_summary, ordered_payloads)
    payload = {
        "schema_version": first["schema_version"],
        "provider": first["provider"],
        "model": first.get("model"),
        "configuration_id": _configuration_id(ordered_payloads),
        "qceval": first["qceval"],
        "suites": first.get("suites", ["core"]),
        "run_id": str(uuid.uuid4()),
        "results": [record.to_dict() for record in records],
        "summary": merged_summary,
    }
    write_output(output, payload, "jsonl")


def _read_complete_shard(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON in {path}:{line_number}: {exc}") from exc
            kind = payload.pop("kind", None)
            if kind == "result":
                results.append(payload)
            elif kind == "summary":
                if summary is not None:
                    raise ValueError(f"multiple summary lines in {path}")
                summary = payload
            else:
                raise ValueError(f"unknown record kind in {path}:{line_number}: {kind!r}")
    if summary is None:
        raise ValueError(f"incomplete shard has no summary: {path}")
    return results, summary


def _validate_compatible(summaries: Sequence[dict[str, Any]]) -> None:
    reference = _compatibility_signature(summaries[0])
    for index, summary in enumerate(summaries[1:], start=2):
        if _compatibility_signature(summary) != reference:
            raise ValueError(f"input shard {index} has incompatible run metadata or protocol")


def _compatibility_signature(summary: dict[str, Any]) -> str:
    protocol = _route_agnostic_protocol(summary.get("summary", {}).get("run_protocol"))
    value = {
        "schema_version": summary.get("schema_version"),
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "qceval": summary.get("qceval"),
        "suites": summary.get("suites", ["core"]),
        "run_protocol": protocol,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _route_agnostic_protocol(protocol: Any) -> Any:
    """Ignore permitted endpoint-segment fields while checking merge semantics."""
    if not isinstance(protocol, dict):
        return protocol
    normalized = json.loads(json.dumps(protocol))
    generation = normalized.get("generation_parameters")
    if isinstance(generation, dict):
        for name in (
            "temperature",
            "endpoint_tag",
            "endpoint_cap_status",
            "output_token_parameter",
            "route_revision",
        ):
            generation.pop(name, None)
    return normalized


def _record_route_protocol(summary: dict[str, Any], results: Sequence[dict[str, Any]]) -> None:
    routes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        metadata = (result.get("provider_response") or {}).get("metadata") or {}
        route = metadata.get("route")
        if not isinstance(route, dict) or not route.get("endpoint_tag"):
            continue
        canonical = json.dumps(route, sort_keys=True, separators=(",", ":"))
        if canonical not in seen:
            seen.add(canonical)
            routes.append(route)
    if not routes:
        return
    generation = summary["run_protocol"]["generation_parameters"]
    for name in (
        "temperature",
        "endpoint_tag",
        "max_output_tokens",
        "output_limit_source",
        "endpoint_cap_status",
        "output_token_parameter",
        "route_revision",
        "configuration_id",
    ):
        values = sorted({json.dumps(route.get(name), sort_keys=True) for route in routes})
        generation[name] = {
            "value": json.loads(values[0]) if len(values) == 1 else [json.loads(value) for value in values],
            "source": "per_record_route_provenance",
        }


def _run_config(summary: dict[str, Any], results: Sequence[dict[str, Any]]) -> RunConfig:
    protocol = summary["summary"]["run_protocol"]
    generation = protocol.get("generation_parameters") or {}
    provider_config: dict[str, Any] = {}
    for name in ("temperature", "reasoning_effort", "reasoning_enabled", "configuration_id"):
        setting = generation.get(name) or {}
        if setting.get("source") == "explicit":
            provider_config[name] = setting.get("value")
    frameworks = tuple(
        sorted({str(payload["framework"]) for payload in results}, key=lambda item: _FRAMEWORK_ORDER.get(item, 99))
    )
    source_hint = summary.get("qceval", {}).get("source_hint")
    return RunConfig(
        provider=str(summary["provider"]),
        frameworks=frameworks,  # type: ignore[arg-type]
        source_hint=None if source_hint is None else Path(str(source_hint)),
        model=None if summary.get("model") is None else str(summary.get("model")),
        provider_config=provider_config,
        suites=tuple(summary.get("suites", ["core"])),
        samples_per_task=int(protocol["samples_per_task"]),
        pass_k=int(protocol["pass_k"]),
        max_attempts=int(protocol["max_attempts"]),
        feedback_max_chars=int(protocol.get("feedback_max_chars") or 2000),
    )


def _record_key(payload: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(payload.get("suite", "core")),
        str(payload["framework"]),
        str(payload["task_id"]),
        int(payload.get("sample_index", 0)),
        int(payload.get("attempt_index", 0)),
    )


def _configuration_id(results: Sequence[dict[str, Any]]) -> str | None:
    values = set()
    for result in results:
        metadata = (result.get("provider_response") or {}).get("metadata") or {}
        route = metadata.get("route") or {}
        value = route.get("configuration_id") if isinstance(route, dict) else None
        if value is not None:
            values.add(str(value))
    if len(values) > 1:
        raise ValueError("merged records contain multiple configuration identities")
    return next(iter(values), None)


def _record_sort_key(payload: dict[str, Any]) -> tuple[int, int, tuple[int, str], int, int]:
    suite, framework, task_id, sample_index, attempt_index = _record_key(payload)
    task_sort = (int(task_id), task_id) if task_id.isdigit() else (10**9, task_id)
    return (
        _SUITE_ORDER.get(suite, 99),
        _FRAMEWORK_ORDER.get(framework, 99),
        task_sort,
        sample_index,
        attempt_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
