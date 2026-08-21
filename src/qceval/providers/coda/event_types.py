"""Data models and limits for Coda event processing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_COMPACT_EVENTS = 100
MAX_COMPACT_STRING_CHARS = 4000

PLUMBING_EVENT_TYPES = frozenset(
    {
        "run_received",
        "node_start",
        "node_end",
        "tool_call",
        "tool_result",
        "thinking_token",
        "heartbeat",
        "simulator_result",
        "results_comparison",
        "code_quality_result",
        "resource_estimation_result",
        "test_execution_result",
        "compilation_result",
    }
)


@dataclass(frozen=True)
class CodaEventStream:
    """Parsed Coda event stream."""

    events: tuple[dict[str, Any], ...]
    compact_events: tuple[dict[str, Any], ...]
    token_text: str
    message_texts: tuple[str, ...]
    final_code_texts: tuple[str, ...]
    structured_data: tuple[Mapping[str, Any], ...]
    event_types: Mapping[str, int]
    completed: bool
    terminal_error: str | None
    cancelled: bool


@dataclass(frozen=True)
class CodaCodeExtraction:
    """Code extracted from a Coda event stream."""

    code: str | None
    text: str | None
    source: str | None


@dataclass(frozen=True)
class _GeneratedCodeText:
    """Code-like text emitted by one source in the event stream."""

    text: str
    source: str
    structured: bool
    index: int
