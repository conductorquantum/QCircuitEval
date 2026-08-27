"""SSE and JSON parsing for Coda agent events."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from qceval.providers.coda.event_compact import _compact_events
from qceval.providers.coda.event_fields import (
    _event_type,
    _message_text,
    _structured_data,
    _terminal_error,
    _token_text,
)
from qceval.providers.coda.event_types import CodaEventStream


def parse_coda_events(lines: Iterable[str | bytes]) -> CodaEventStream:
    """Parse Coda server-sent events from response lines.

    Args:
        lines: Text or UTF-8 byte lines from a Coda response.

    Returns:
        Normalized aggregate event stream, using plain JSON as a fallback.
    """
    decoded = [_decode_line(line) for line in lines]
    events = _parse_sse_lines(decoded)
    if events is None:
        events = _parse_json_fallback(decoded)
    return _event_stream(tuple(events))


def _decode_line(line: str | bytes) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace").rstrip("\r\n")
    return line.rstrip("\r\n")


def _parse_sse_lines(
    lines: Sequence[str],
) -> list[dict[str, Any]] | None:
    """Try to parse lines as an SSE stream."""
    events: list[dict[str, Any]] = []
    saw_sse = False
    event_name: str | None = None
    data_parts: list[str] = []
    for line in lines:
        if line.startswith(":"):
            saw_sse = True
            continue
        if not line:
            saw_sse = (
                _flush_sse_event(
                    events,
                    event_name,
                    data_parts,
                )
                or saw_sse
            )
            event_name = None
            data_parts = []
            continue
        if line.startswith("event:"):
            saw_sse = True
            event_name = line.removeprefix("event:").strip()
            continue
        if line.startswith("data:"):
            saw_sse = True
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                events.append({"type": "done", "data": data})
            elif event_name is None:
                events.append(_event_from_payload(data, event_name))
            else:
                data_parts.append(data)
    _flush_sse_event(events, event_name, data_parts)
    return events if saw_sse else None


def _flush_sse_event(
    events: list[dict[str, Any]],
    event_name: str | None,
    data_parts: list[str],
) -> bool:
    """Emit a buffered SSE event when data or a name was accumulated."""
    if not event_name and not data_parts:
        return False
    payload = "\n".join(data_parts).strip()
    events.append(_event_from_payload(payload, event_name))
    return True


def _event_from_payload(
    payload: str,
    event_name: str | None,
) -> dict[str, Any]:
    """Build a normalized event dictionary from an SSE payload."""
    decoded = _maybe_json(payload)
    if isinstance(decoded, Mapping):
        event = dict(decoded)
        if event_name:
            event.setdefault("name", event_name)
            event.setdefault("type", event_name)
        return _normalize_event(event)
    event_type = event_name or "message"
    content = payload if decoded is None else decoded
    return _normalize_event(
        {
            "type": event_type,
            "name": event_name or event_type,
            "content": str(content),
        }
    )


def _parse_json_fallback(
    lines: Sequence[str],
) -> list[dict[str, Any]]:
    """Parse a response body as plain JSON when SSE detection fails."""
    payload = "\n".join(lines).strip()
    decoded = _maybe_json(payload)
    if isinstance(decoded, list):
        return [_normalize_event(item) for item in decoded if isinstance(item, Mapping)]
    if isinstance(decoded, Mapping):
        events = decoded.get("events")
        if isinstance(events, list):
            return [_normalize_event(item) for item in events if isinstance(item, Mapping)]
        return [_normalize_event(decoded)]
    if payload:
        return [_normalize_event({"type": "message", "content": payload})]
    return []


def _maybe_json(value: str) -> Any | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure each event has a ``type`` and decode nested JSON data."""
    normalized = dict(event)
    if "event" in normalized and "type" not in normalized:
        normalized["type"] = normalized["event"]
    normalized.setdefault("type", "message")
    if isinstance(normalized.get("data"), str):
        nested = _maybe_json(str(normalized["data"]))
        if nested is not None:
            normalized["data"] = nested
    return normalized


def _event_stream(
    events: tuple[dict[str, Any], ...],
) -> CodaEventStream:
    """Aggregate normalized events into one ``CodaEventStream``."""
    from qceval.providers.coda.event_extraction import (
        _final_generated_code_text,
    )

    event_types = Counter(_event_type(event) for event in events)
    return CodaEventStream(
        events=events,
        compact_events=_compact_events(events),
        token_text="".join(_token_text(event) for event in events),
        message_texts=tuple(text for event in events if (text := _message_text(event))),
        final_code_texts=tuple(text for event in events if (text := _final_generated_code_text(event))),
        structured_data=tuple(data for event in events if (data := _structured_data(event)) is not None),
        event_types=dict(event_types),
        completed=any(_event_type(event) in {"completed", "done"} for event in events),
        terminal_error=_terminal_error(events),
        cancelled=any(_event_type(event) == "cancelled" for event in events),
    )
