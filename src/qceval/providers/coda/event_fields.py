"""Normalized Coda event field accessors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from qceval.providers.coda.event_types import PLUMBING_EVENT_TYPES


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("type") or "message")


def _token_text(event: Mapping[str, Any]) -> str:
    if _event_type(event) != "token":
        return ""
    text = _text_from_event(event)
    return "" if text == "<DONE>" else text.replace("<DONE>", "")


def _message_text(event: Mapping[str, Any]) -> str:
    event_type = _event_type(event)
    role = event.get("role")
    if event_type not in {"message", "assistant_message", "final_message"} and role != "assistant":
        return ""
    if event_type in PLUMBING_EVENT_TYPES:
        return ""
    return _text_from_event(event)


def _structured_data(
    event: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if _event_type(event) != "structured_response":
        return None
    data = event.get("data")
    if isinstance(data, Mapping):
        return data
    return {"code": data} if isinstance(data, str) else None


def _text_from_event(event: Mapping[str, Any]) -> str:
    for value in _text_values(event):
        if isinstance(value, str):
            return value
    return ""


def _text_values(event: Mapping[str, Any]) -> Iterable[Any]:
    for key in ("content", "text", "output"):
        yield event.get(key)
    data = event.get("data")
    if isinstance(data, Mapping):
        for key in ("content", "text", "output", "message"):
            yield data.get(key)


def _terminal_error(
    events: Sequence[Mapping[str, Any]],
) -> str | None:
    from qceval.providers.coda.event_compact import _compact_json

    for event in events:
        if _event_type(event) == "error":
            return _text_from_event(event) or _compact_json(
                event,
                max_string_chars=500,
            )
    return None
