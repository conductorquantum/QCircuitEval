"""Durable deferred-infrastructure recovery for production generation lanes."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from qceval.production.resume import LogicalKey

DEFERRED_STATE_SCHEMA_VERSION = "qceval.deferred_infrastructure.v1"
DEFERRED_LEDGER_SCHEMA_VERSION = "qceval.deferred_infrastructure_ledger.v1"


@dataclass(frozen=True)
class InfrastructureRetryPolicy:
    """Frozen operational retry and endpoint-circuit policy."""

    total_attempts: int = 6
    circuit_exhaustion_threshold: int = 2
    circuit_cooldown_seconds: int = 1800

    def __post_init__(self) -> None:
        if self.total_attempts != 6:
            raise ValueError("production infrastructure policy requires exactly six total attempts")
        if self.circuit_exhaustion_threshold != 2:
            raise ValueError("production circuit requires two consecutive exhausted logical requests")
        if self.circuit_cooldown_seconds < 1800:
            raise ValueError("production circuit cooldown must be at least 1800 seconds")


class DeferredInfrastructureStore:
    """Atomic recovery state plus append-only deferred transition evidence."""

    def __init__(
        self,
        out_dir: Path,
        *,
        policy: InfrastructureRetryPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.path = out_dir / "deferred-infrastructure-state.json"
        self.ledger_path = out_dir / "deferred-infrastructure-ledger.jsonl"
        self.policy = policy or InfrastructureRetryPolicy()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def reconcile_accepted(
        self,
        *,
        model_id: str,
        job_id: str,
        endpoint_tag: str,
        route_revision: str,
        accepted: Mapping[LogicalKey, Mapping[str, Any]],
        configuration_id: str | None = None,
    ) -> int:
        """Resolve deferred entries that gained an accepted result before restart.

        Args:
            model_id: Frozen provider model identifier.
            job_id: Frozen framework-shard identifier.
            endpoint_tag: Exact pinned endpoint tag.
            route_revision: Frozen route revision.
            accepted: Strictly validated accepted records by logical key.

        Returns:
            Number of deferred entries resolved by accepted artifacts.
        """
        resolved = 0
        with self._lock:
            for key, record in accepted.items():
                request_id = self.request_id(model_id, key, configuration_id=configuration_id)
                entry = self._state["requests"].get(request_id)
                if not isinstance(entry, dict) or entry.get("status") != "deferred_infrastructure":
                    continue
                self._resolve_entry(entry, record=record)
                resolved += 1
                self._close_circuit(model_id, endpoint_tag, route_revision)
            if resolved:
                self._commit(
                    {
                        "kind": "accepted_reconciliation",
                        "model_id": model_id,
                        "job_id": job_id,
                        "configuration_id": configuration_id,
                        "resolved_requests": resolved,
                    }
                )
        return resolved

    def record_accepted(
        self,
        *,
        model_id: str,
        job_id: str,
        endpoint_tag: str,
        route_revision: str,
        key: LogicalKey,
        record: Mapping[str, Any],
        from_deferred_sweep: bool,
        configuration_id: str | None = None,
    ) -> None:
        """Record an accepted logical outcome and close/reset its endpoint circuit.

        Args:
            model_id: Frozen provider model identifier.
            job_id: Frozen framework-shard identifier.
            endpoint_tag: Exact pinned endpoint tag.
            route_revision: Frozen route revision.
            key: Route-independent logical request identity.
            record: Strictly validated accepted result record.
            from_deferred_sweep: Whether the result came from a recovery sweep.
        """
        with self._lock:
            request_id = self.request_id(model_id, key, configuration_id=configuration_id)
            entry = self._state["requests"].get(request_id)
            if isinstance(entry, dict) and entry.get("status") == "deferred_infrastructure":
                self._resolve_entry(entry, record=record)
            circuit = self._circuit(model_id, endpoint_tag, route_revision)
            circuit["consecutive_exhausted_logical_requests"] = 0
            circuit["last_exhausted_request_id"] = None
            if circuit.get("status") == "open" and from_deferred_sweep:
                self._close_circuit(model_id, endpoint_tag, route_revision)
            self._commit(
                {
                    "kind": "accepted_result",
                    "request_id": request_id,
                    "logical_key": list(key),
                    "model_id": model_id,
                    "job_id": job_id,
                    "configuration_id": configuration_id,
                    "endpoint_tag": endpoint_tag,
                    "route_revision": route_revision,
                    "from_deferred_sweep": from_deferred_sweep,
                    "generation_id": _generation_id(record),
                }
            )

    def defer_exhausted(
        self,
        *,
        model_id: str,
        job_id: str,
        endpoint_tag: str,
        route_revision: str,
        key: LogicalKey,
        error_history: Sequence[Mapping[str, Any]],
        attempt_count: int,
        segment: Path,
        configuration_id: str | None = None,
    ) -> bool:
        """Persist one exhausted transient request and update its endpoint circuit.

        Args:
            model_id: Frozen provider model identifier.
            job_id: Frozen framework-shard identifier.
            endpoint_tag: Exact pinned endpoint tag.
            route_revision: Frozen route revision.
            key: Route-independent logical request identity.
            error_history: Complete six-attempt infrastructure history.
            attempt_count: Total physical attempts made for the logical request.
            segment: Durable JSONL segment containing the exhausted record.

        Returns:
            Whether the endpoint circuit is open after recording the exhaustion.
        """
        if attempt_count != self.policy.total_attempts:
            raise ValueError(f"deferred request must have {self.policy.total_attempts} attempts, found {attempt_count}")
        now = self._now()
        with self._lock:
            request_id = self.request_id(model_id, key, configuration_id=configuration_id)
            existing = self._state["requests"].get(request_id)
            first_deferred = (
                existing.get("first_deferred_at_utc") if isinstance(existing, Mapping) else self._format(now)
            )
            defer_count = int(existing.get("defer_count", 0)) + 1 if isinstance(existing, Mapping) else 1
            next_eligible = now + timedelta(seconds=self.policy.circuit_cooldown_seconds)
            self._state["requests"][request_id] = {
                "status": "deferred_infrastructure",
                "request_id": request_id,
                "logical_key": list(key),
                "model_id": model_id,
                "job_id": job_id,
                "configuration_id": configuration_id,
                "endpoint_tag": endpoint_tag,
                "route_revision": route_revision,
                "error_history": [dict(item) for item in error_history],
                "attempt_count": attempt_count,
                "first_deferred_at_utc": first_deferred,
                "last_deferred_at_utc": self._format(now),
                "next_eligible_retry_at_utc": self._format(next_eligible),
                "defer_count": defer_count,
                "segment": str(segment),
            }

            circuit = self._circuit(model_id, endpoint_tag, route_revision)
            already_open = circuit.get("status") == "open"
            if already_open:
                open_circuit = True
            elif circuit.get("last_exhausted_request_id") != request_id:
                circuit["consecutive_exhausted_logical_requests"] = (
                    int(circuit.get("consecutive_exhausted_logical_requests", 0)) + 1
                )
                circuit["last_exhausted_request_id"] = request_id
                open_circuit = (
                    circuit["consecutive_exhausted_logical_requests"] >= self.policy.circuit_exhaustion_threshold
                )
            else:
                open_circuit = False

            if open_circuit:
                cooldown_until = now + timedelta(seconds=self.policy.circuit_cooldown_seconds)
                circuit.update(
                    {
                        "status": "open",
                        "opened_at_utc": circuit.get("opened_at_utc") or self._format(now),
                        "last_opened_at_utc": self._format(now),
                        "cooldown_until_utc": self._format(cooldown_until),
                    }
                )
                for entry in self._active_entries(model_id, endpoint_tag, route_revision):
                    entry["next_eligible_retry_at_utc"] = self._format(cooldown_until)

            self._commit(
                {
                    "kind": "request_deferred",
                    "request_id": request_id,
                    "logical_key": list(key),
                    "model_id": model_id,
                    "job_id": job_id,
                    "configuration_id": configuration_id,
                    "endpoint_tag": endpoint_tag,
                    "route_revision": route_revision,
                    "attempt_count": attempt_count,
                    "error_history": [dict(item) for item in error_history],
                    "next_eligible_retry_at_utc": self._state["requests"][request_id]["next_eligible_retry_at_utc"],
                    "circuit_open": open_circuit,
                    "segment": str(segment),
                }
            )
            return open_circuit

    def deferred_keys(self, *, model_id: str, job_id: str) -> set[LogicalKey]:
        """Return active deferred logical keys for one frozen shard.

        Args:
            model_id: Frozen provider model identifier.
            job_id: Frozen framework-shard identifier.

        Returns:
            Active deferred logical keys for the shard.
        """
        with self._lock:
            return {
                _logical_key(entry["logical_key"])
                for entry in self._state["requests"].values()
                if entry.get("status") == "deferred_infrastructure"
                and entry.get("model_id") == model_id
                and entry.get("job_id") == job_id
            }

    def eligible_deferred_keys(
        self,
        *,
        model_id: str,
        job_id: str,
        endpoint_tag: str,
        route_revision: str,
    ) -> set[LogicalKey]:
        """Return unaccepted deferred keys eligible for one recovery sweep.

        Args:
            model_id: Frozen provider model identifier.
            job_id: Frozen framework-shard identifier.
            endpoint_tag: Exact pinned endpoint tag.
            route_revision: Frozen route revision.

        Returns:
            Deferred logical keys whose cooldown has elapsed.
        """
        now = self._now()
        with self._lock:
            circuit = self._circuit(model_id, endpoint_tag, route_revision)
            cooldown = _parse_timestamp(circuit.get("cooldown_until_utc"))
            if circuit.get("status") == "open" and cooldown is not None and now < cooldown:
                return set()
            eligible: set[LogicalKey] = set()
            for entry in self._active_entries(model_id, endpoint_tag, route_revision):
                if entry.get("job_id") != job_id:
                    continue
                next_eligible = _parse_timestamp(entry.get("next_eligible_retry_at_utc"))
                if next_eligible is None or next_eligible <= now:
                    eligible.add(_logical_key(entry["logical_key"]))
            return eligible

    def circuit_is_open(self, *, model_id: str, endpoint_tag: str, route_revision: str) -> bool:
        """Return whether the frozen endpoint circuit is currently open.

        Args:
            model_id: Frozen provider model identifier.
            endpoint_tag: Exact pinned endpoint tag.
            route_revision: Frozen route revision.

        Returns:
            Whether the route-scoped endpoint circuit is open.
        """
        with self._lock:
            return self._circuit(model_id, endpoint_tag, route_revision).get("status") == "open"

    def cooldown_until(
        self,
        *,
        model_id: str,
        endpoint_tag: str,
        route_revision: str,
    ) -> str | None:
        """Return the endpoint's durable cooldown deadline, if any.

        Args:
            model_id: Frozen provider model identifier.
            endpoint_tag: Exact pinned endpoint tag.
            route_revision: Frozen route revision.

        Returns:
            UTC cooldown deadline or ``None`` when no cooldown is active.
        """
        with self._lock:
            value = self._circuit(model_id, endpoint_tag, route_revision).get("cooldown_until_utc")
            return value if isinstance(value, str) else None

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible recovery-state snapshot."""
        with self._lock:
            return json.loads(json.dumps(self._state))

    @staticmethod
    def request_id(model_id: str, key: LogicalKey, *, configuration_id: str | None = None) -> str:
        """Return a model-scoped stable identifier for one logical assignment.

        Args:
            model_id: Frozen provider model identifier.
            key: Route-independent logical request identity.

        Returns:
            SHA-256 request identifier.
        """
        canonical = json.dumps([configuration_id or model_id, *key], separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": DEFERRED_STATE_SCHEMA_VERSION,
                "updated_at_utc": self._format(self._now()),
                "policy": asdict(self.policy),
                "requests": {},
                "circuits": {},
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != DEFERRED_STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported deferred state schema in {self.path}")
        if payload.get("policy") != asdict(self.policy):
            raise ValueError(f"deferred state policy mismatch in {self.path}")
        if not isinstance(payload.get("requests"), dict) or not isinstance(payload.get("circuits"), dict):
            raise ValueError(f"malformed deferred state in {self.path}")
        return payload

    def _circuit(self, model_id: str, endpoint_tag: str, route_revision: str) -> dict[str, Any]:
        identity = self._circuit_id(model_id, endpoint_tag, route_revision)
        return self._state["circuits"].setdefault(
            identity,
            {
                "status": "closed",
                "model_id": model_id,
                "endpoint_tag": endpoint_tag,
                "route_revision": route_revision,
                "consecutive_exhausted_logical_requests": 0,
                "last_exhausted_request_id": None,
                "opened_at_utc": None,
                "last_opened_at_utc": None,
                "cooldown_until_utc": None,
                "closed_at_utc": None,
            },
        )

    def _close_circuit(self, model_id: str, endpoint_tag: str, route_revision: str) -> None:
        circuit = self._circuit(model_id, endpoint_tag, route_revision)
        circuit.update(
            {
                "status": "closed",
                "consecutive_exhausted_logical_requests": 0,
                "last_exhausted_request_id": None,
                "cooldown_until_utc": None,
                "closed_at_utc": self._format(self._now()),
            }
        )

    def _active_entries(self, model_id: str, endpoint_tag: str, route_revision: str) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self._state["requests"].values()
            if entry.get("status") == "deferred_infrastructure"
            and entry.get("model_id") == model_id
            and entry.get("endpoint_tag") == endpoint_tag
            and entry.get("route_revision") == route_revision
        ]

    def _resolve_entry(self, entry: dict[str, Any], *, record: Mapping[str, Any]) -> None:
        entry.update(
            {
                "status": "accepted",
                "accepted_at_utc": self._format(self._now()),
                "generation_id": _generation_id(record),
                "next_eligible_retry_at_utc": None,
            }
        )

    def _commit(self, event: Mapping[str, Any]) -> None:
        now = self._format(self._now())
        self._state["updated_at_utc"] = now
        _atomic_write_json(self.path, self._state)
        payload = {
            "schema_version": DEFERRED_LEDGER_SCHEMA_VERSION,
            "created_at_utc": now,
            **event,
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("deferred recovery clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _format(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _circuit_id(model_id: str, endpoint_tag: str, route_revision: str) -> str:
        return json.dumps([model_id, endpoint_tag, route_revision], separators=(",", ":"))


def _logical_key(value: Sequence[Any]) -> LogicalKey:
    if len(value) != 5:
        raise ValueError("deferred logical key must contain five fields")
    return (str(value[0]), str(value[1]), str(value[2]), int(value[3]), int(value[4]))


def _generation_id(record: Mapping[str, Any]) -> str | None:
    response = record.get("provider_response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    value = metadata.get("generation_id") if isinstance(metadata, Mapping) else None
    return None if value is None else str(value)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
