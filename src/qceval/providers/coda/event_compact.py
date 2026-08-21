"""Bounded JSON-compatible Coda event diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qceval.providers.coda.event_types import (
    MAX_COMPACT_EVENTS,
    MAX_COMPACT_STRING_CHARS,
)


def compact_event(
    event: Mapping[str, Any],
    *,
    max_string_chars: int = MAX_COMPACT_STRING_CHARS,
) -> dict[str, Any]:
    """Build a bounded JSON-compatible event payload.

    Args:
        event: Normalized event mapping to compact.
        max_string_chars: Maximum retained length of each string value.

    Returns:
        A compact event containing supported diagnostic fields.
    """
    compact: dict[str, Any] = {}
    for key in ("type", "name", "content", "output", "data"):
        if key in event:
            compact[key] = _compact_json(
                event[key],
                max_string_chars=max_string_chars,
            )
    return compact


def _compact_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Compact and truncate events to fit within ``MAX_COMPACT_EVENTS``."""
    compact = [compact_event(event) for event in events]
    if len(compact) <= MAX_COMPACT_EVENTS:
        return tuple(compact)
    head_count = MAX_COMPACT_EVENTS // 2
    tail_count = MAX_COMPACT_EVENTS - head_count - 1
    omitted = len(compact) - head_count - tail_count
    marker = {"type": "truncated", "content": f"{omitted} events omitted"}
    return (*compact[:head_count], marker, *compact[-tail_count:])


def _compact_json(value: Any, *, max_string_chars: int) -> Any:
    """Recursively truncate strings and coerce non-JSON types to ``repr``."""
    if isinstance(value, str):
        return _truncate(value, max_string_chars)
    if isinstance(value, Mapping):
        return {str(key): _compact_json(item, max_string_chars=max_string_chars) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_compact_json(item, max_string_chars=max_string_chars) for item in value]
    if isinstance(value, int | float | bool) or value is None:
        return value
    return _truncate(repr(value), max_string_chars)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...[truncated {len(value) - max_chars} chars]"
