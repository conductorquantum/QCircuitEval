from __future__ import annotations

import json
from pathlib import Path

import pytest

from qceval.production.resume import accepted_records, pending_keys


def _record(task_id: str, status: str, route: str) -> dict:
    return {
        "kind": "result",
        "suite": "core",
        "framework": "qiskit",
        "task_id": task_id,
        "sample_index": 0,
        "attempt_index": 0,
        "status": status,
        "provider_response": {
            "metadata": {"route": {"route_revision": route, "route_verified": True}},
            "usage": {"cost_usd": 0.0},
        },
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_outage_resume_retains_accepted_outputs_and_returns_only_failures_to_pending(tmp_path: Path) -> None:
    first_route = tmp_path / "route-a.jsonl"
    _write(
        first_route,
        [
            _record("01", "generated", "route-a"),
            _record("02", "provider_failed", "route-a"),
            _record("03", "infrastructure_error", "route-a"),
        ],
    )
    assignments = [("core", "qiskit", task_id, 0, 0) for task_id in ("01", "02", "03", "04")]

    accepted = accepted_records([first_route])
    pending = pending_keys(assignments, [first_route])

    assert set(accepted) == set(assignments[:2])
    assert pending == assignments[2:]


def test_resume_rejects_duplicate_accepted_logical_key_across_route_segments(tmp_path: Path) -> None:
    first = tmp_path / "route-a.jsonl"
    second = tmp_path / "route-b.jsonl"
    _write(first, [_record("01", "generated", "route-a")])
    _write(second, [_record("01", "generated", "route-b")])

    with pytest.raises(ValueError, match="duplicate accepted logical key"):
        accepted_records([first, second])


def test_resume_returns_costless_or_unverified_results_to_pending(tmp_path: Path) -> None:
    segment = tmp_path / "route-a.jsonl"
    costless = _record("01", "provider_failed", "route-a")
    costless["provider_response"]["usage"] = {"cost_usd": None}
    unverified = _record("02", "generated", "route-a")
    unverified["provider_response"]["metadata"]["route"]["route_verified"] = False
    _write(segment, [costless, unverified])
    assignments = [("core", "qiskit", task_id, 0, 0) for task_id in ("01", "02")]

    assert accepted_records([segment]) == {}
    assert pending_keys(assignments, [segment]) == assignments
