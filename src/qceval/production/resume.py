"""Accepted-output preservation for endpoint route revisions."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

LogicalKey = tuple[str, str, str, int, int]


def accepted_records(paths: Sequence[Path], *, strict_provenance: bool = False) -> dict[LogicalKey, dict[str, Any]]:
    """Read accepted model outcomes and reject duplicate logical keys.

    Infrastructure failures are intentionally returned to the pending queue.
    Successful candidates and legitimate candidate-less model outcomes are
    accepted and therefore never regenerated after a route change.

    Args:
        paths: Route-segment JSONL artifacts to inspect.

    Returns:
        Accepted result payloads keyed by route-independent request identity.
    """
    accepted: dict[LogicalKey, dict[str, Any]] = {}
    for record in _result_records(paths):
        if record.get("status") == "infrastructure_error":
            continue
        provenance_error = _provenance_error(record)
        if provenance_error is not None:
            if strict_provenance:
                raise ValueError(provenance_error)
            continue
        key = logical_key(record)
        if key in accepted:
            raise ValueError(f"duplicate accepted logical key: {key}")
        accepted[key] = record
    return accepted


def _provenance_error(record: Mapping[str, Any]) -> str | None:
    response = record.get("provider_response")
    if not isinstance(response, Mapping):
        return "accepted record lacks provider response provenance"
    metadata = response.get("metadata")
    route = metadata.get("route") if isinstance(metadata, Mapping) else None
    if not isinstance(route, Mapping) or route.get("route_verified") is not True:
        return "accepted record lacks verified route provenance"
    usage = response.get("usage")
    cost = usage.get("cost_usd") if isinstance(usage, Mapping) else None
    if isinstance(cost, bool) or not isinstance(cost, int | float) or not math.isfinite(cost) or cost < 0:
        return "accepted record lacks provider-reported cost"
    return None


def pending_keys(assignments: Iterable[LogicalKey], paths: Sequence[Path]) -> list[LogicalKey]:
    """Return only never-started or infrastructure-failed assignments.

    Args:
        assignments: Complete logical assignment roster.
        paths: Existing route-segment JSONL artifacts.

    Returns:
        Ordered logical keys that still require a model outcome.
    """
    expected = list(assignments)
    if len(set(expected)) != len(expected):
        raise ValueError("assignment list contains duplicate logical keys")
    completed = accepted_records(paths)
    foreign = sorted(set(completed) - set(expected))
    if foreign:
        raise ValueError(f"accepted artifacts contain foreign logical keys: {foreign[:3]}")
    return [key for key in expected if key not in completed]


def logical_key(record: Mapping[str, Any]) -> LogicalKey:
    """Return the route-independent identity of one benchmark request.

    Args:
        record: Serialized benchmark result.

    Returns:
        Suite, framework, task, sample, and attempt identity tuple.
    """
    return (
        str(record.get("suite", "core")),
        str(record["framework"]),
        str(record["task_id"]),
        int(record.get("sample_index", 0)),
        int(record.get("attempt_index", 0)),
    )


def _result_records(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{number}: record must be an object")
            if payload.get("kind") == "result":
                yield payload
