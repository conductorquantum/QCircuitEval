from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qceval.production.deferred import DeferredInfrastructureStore


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 11, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _key(task_id: str) -> tuple[str, str, str, int, int]:
    return ("core", "qiskit", task_id, 0, 0)


def _history(message: str = "temporary outage") -> list[dict]:
    return [
        {
            "attempt_number": number,
            "status": "transient_infrastructure",
            "transient": True,
            "error": message,
        }
        for number in range(1, 7)
    ]


def _accepted(task_id: str, generation_id: str = "gen-ok") -> dict:
    return {
        "kind": "result",
        "suite": "core",
        "framework": "qiskit",
        "task_id": task_id,
        "sample_index": 0,
        "attempt_index": 0,
        "status": "generated",
        "provider_response": {
            "metadata": {
                "generation_id": generation_id,
                "route": {"route_verified": True},
            },
            "usage": {"cost_usd": 0.25},
        },
    }


def _defer(store: DeferredInfrastructureStore, tmp_path: Path, task_id: str) -> bool:
    return store.defer_exhausted(
        model_id="author/model",
        job_id="author-model__qiskit",
        endpoint_tag="author/fp8",
        route_revision="route-frozen",
        key=_key(task_id),
        error_history=_history(),
        attempt_count=6,
        segment=tmp_path / f"{task_id}.jsonl",
    )


def test_deferred_state_persists_and_recovers_after_restart(tmp_path: Path) -> None:
    clock = _Clock()
    store = DeferredInfrastructureStore(tmp_path, clock=clock)

    opened = _defer(store, tmp_path, "01")
    restarted = DeferredInfrastructureStore(tmp_path, clock=clock)

    assert opened is False
    assert restarted.deferred_keys(model_id="author/model", job_id="author-model__qiskit") == {_key("01")}
    state = json.loads((tmp_path / "deferred-infrastructure-state.json").read_text(encoding="utf-8"))
    entry = next(iter(state["requests"].values()))
    assert entry["status"] == "deferred_infrastructure"
    assert entry["attempt_count"] == 6
    assert len(entry["error_history"]) == 6
    assert entry["next_eligible_retry_at_utc"] == "2026-08-11T00:30:00Z"


def test_deferred_request_identity_isolated_by_effort_configuration() -> None:
    key = _key("01")

    low = DeferredInfrastructureStore.request_id("author/model", key, configuration_id="author-model__effort-low")
    high = DeferredInfrastructureStore.request_id("author/model", key, configuration_id="author-model__effort-high")

    assert low != high


def test_two_distinct_consecutive_exhaustions_open_circuit_for_at_least_30_minutes(tmp_path: Path) -> None:
    clock = _Clock()
    store = DeferredInfrastructureStore(tmp_path, clock=clock)

    assert _defer(store, tmp_path, "01") is False
    assert _defer(store, tmp_path, "02") is True

    assert store.circuit_is_open(model_id="author/model", endpoint_tag="author/fp8", route_revision="route-frozen")
    assert (
        store.eligible_deferred_keys(
            model_id="author/model",
            job_id="author-model__qiskit",
            endpoint_tag="author/fp8",
            route_revision="route-frozen",
        )
        == set()
    )
    clock.advance(seconds=1799)
    assert (
        store.eligible_deferred_keys(
            model_id="author/model",
            job_id="author-model__qiskit",
            endpoint_tag="author/fp8",
            route_revision="route-frozen",
        )
        == set()
    )
    clock.advance(seconds=1)
    assert store.eligible_deferred_keys(
        model_id="author/model",
        job_id="author-model__qiskit",
        endpoint_tag="author/fp8",
        route_revision="route-frozen",
    ) == {_key("01"), _key("02")}


def test_successful_deferred_sweep_closes_circuit_and_deduplicates_accepted_key(tmp_path: Path) -> None:
    clock = _Clock()
    store = DeferredInfrastructureStore(tmp_path, clock=clock)
    _defer(store, tmp_path, "01")
    _defer(store, tmp_path, "02")
    clock.advance(seconds=1800)

    store.record_accepted(
        model_id="author/model",
        job_id="author-model__qiskit",
        endpoint_tag="author/fp8",
        route_revision="route-frozen",
        key=_key("01"),
        record=_accepted("01"),
        from_deferred_sweep=True,
    )
    resolved_again = store.reconcile_accepted(
        model_id="author/model",
        job_id="author-model__qiskit",
        endpoint_tag="author/fp8",
        route_revision="route-frozen",
        accepted={_key("01"): _accepted("01")},
    )

    assert resolved_again == 0
    assert not store.circuit_is_open(model_id="author/model", endpoint_tag="author/fp8", route_revision="route-frozen")
    assert store.deferred_keys(model_id="author/model", job_id="author-model__qiskit") == {_key("02")}


def test_same_logical_request_cannot_open_two_request_circuit_by_itself(tmp_path: Path) -> None:
    store = DeferredInfrastructureStore(tmp_path, clock=_Clock())

    assert _defer(store, tmp_path, "01") is False
    assert _defer(store, tmp_path, "01") is False

    assert not store.circuit_is_open(model_id="author/model", endpoint_tag="author/fp8", route_revision="route-frozen")
