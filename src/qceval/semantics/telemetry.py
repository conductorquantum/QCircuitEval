"""Bounded source-free semantic verification telemetry."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Protocol

MAX_EVENT_TEXT = 200


@dataclass(frozen=True)
class SemanticEvent:
    """One local route/cache/phase/status event without candidate source."""

    event: str
    correlation_id: str
    contract_hash: str
    target_hash: str
    input_hash: str
    route: str | None = None
    phase: str | None = None
    status: str | None = None
    reason: str | None = None
    elapsed_seconds: float | None = None
    monotonic_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("event", "correlation_id", "contract_hash", "target_hash", "input_hash"):
            value = getattr(self, name)
            if not value or len(value) > MAX_EVENT_TEXT:
                raise ValueError(f"semantic event {name} is empty or too long")
        for name in ("route", "phase", "status", "reason"):
            value = getattr(self, name)
            if value is not None and len(value) > MAX_EVENT_TEXT:
                raise ValueError(f"semantic event {name} is too long")
        if self.elapsed_seconds is not None and self.elapsed_seconds < 0:
            raise ValueError("semantic event elapsed time must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return bounded JSON-compatible event fields.

        Returns:
            Event dictionary with no source or exception payload.
        """
        return asdict(self)


class EventSink(Protocol):
    """Consume bounded local semantic events."""

    def emit(self, event: SemanticEvent) -> None:
        """Record an event.

        Args:
            event: Bounded source-free event.
        """
        ...


class InMemoryEventSink:
    """Deterministic test/debug event collector."""

    def __init__(self) -> None:
        self.events: list[SemanticEvent] = []

    def emit(self, event: SemanticEvent) -> None:
        """Append one event.

        Args:
            event: Bounded source-free event.
        """
        self.events.append(event)


def event_now(
    event: str,
    correlation_id: str,
    contract_hash: str,
    target_hash: str,
    input_hash: str,
    **fields: str | float | None,
) -> SemanticEvent:
    """Build an event stamped with a monotonic local time.

    Args:
        event: Stable event name.
        correlation_id: Run-local correlation identifier.
        contract_hash: Contract content hash.
        target_hash: Target content hash.
        input_hash: Candidate semantic hash.
        **fields: Optional route, phase, status, reason, and elapsed time.

    Returns:
        Bounded event.
    """
    raw_elapsed = fields.get("elapsed_seconds")
    elapsed = None if raw_elapsed is None else float(raw_elapsed)
    return SemanticEvent(
        event,
        correlation_id,
        contract_hash,
        target_hash,
        input_hash,
        route=None if fields.get("route") is None else str(fields["route"]),
        phase=None if fields.get("phase") is None else str(fields["phase"]),
        status=None if fields.get("status") is None else str(fields["status"]),
        reason=None if fields.get("reason") is None else str(fields["reason"]),
        elapsed_seconds=elapsed,
        monotonic_seconds=time.monotonic(),
    )
