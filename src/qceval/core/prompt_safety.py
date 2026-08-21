"""Guards that keep grading oracles out of provider-facing text."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


class OracleLeakError(ValueError):
    """Harness-constructed provider text contained grading-oracle content.

    Raised only for text the harness authors itself (prompts, system text, and
    repair feedback). Model-authored candidate code is never scanned with the
    substring deny-list; oracle isolation for echoed candidate turns is
    enforced structurally by controlling which grader fields can enter a
    provider request at all.
    """


# Fragments that must never appear in provider prompts or repair feedback.
# Matching is case-insensitive against the full message text and against JSON keys.
ORACLE_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"semantic_verification", re.IGNORECASE),
    re.compile(r"behavior_verdict", re.IGNORECASE),
    re.compile(r"score_authority", re.IGNORECASE),
    re.compile(r"contract_hash", re.IGNORECASE),
    re.compile(r"target_hash", re.IGNORECASE),
    re.compile(r"contract_version", re.IGNORECASE),
    re.compile(r"canonical_solution", re.IGNORECASE),
    re.compile(r"canonical_probabilities", re.IGNORECASE),
    re.compile(r"expected_distribution", re.IGNORECASE),
    re.compile(r"case_results", re.IGNORECASE),
    re.compile(r"independent_derivations", re.IGNORECASE),
    re.compile(r"\"contract\"\s*:", re.IGNORECASE),
    re.compile(r"\"target\"\s*:", re.IGNORECASE),
    re.compile(r"\"evidence\"\s*:", re.IGNORECASE),
)

ORACLE_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {
        "canonical",
        "expected",
        "probabilities",
        "case_results",
        "accepted_peak",
        "contract",
        "target",
        "semantic",
        "verification",
        "behavior_verdict",
        "score_authority",
        "oracle",
        "hash",
        "evidence",
        "derivation",
        "manifest",
    }
)


def oracle_key_is_blocked(key: str) -> bool:
    """Return whether a mapping key must be excluded from provider feedback.

    Args:
        key: Candidate JSON or metadata key.

    Returns:
        ``True`` when the key name looks like grading-oracle content.
    """
    lowered = key.lower()
    return any(fragment in lowered for fragment in ORACLE_KEY_FRAGMENTS)


def provider_text_leaks_oracle(text: str) -> str | None:
    """Return the first matching oracle pattern in provider-facing text.

    Args:
        text: Prompt or feedback message that would be sent to a provider.

    Returns:
        Matching pattern text, or ``None`` when the message is safe.
    """
    for pattern in ORACLE_TEXT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return match.group(0)
    return None


def assert_provider_text_excludes_oracle(text: str) -> None:
    """Raise when provider-facing text contains grading-oracle content.

    Args:
        text: Prompt or feedback message.

    Raises:
        OracleLeakError: If oracle content is present.
    """
    leaked = provider_text_leaks_oracle(text)
    if leaked is not None:
        raise OracleLeakError(f"provider text must not include grading oracle content ({leaked!r})")


def assert_provider_messages_exclude_oracle(messages: Sequence[Mapping[str, Any]] | Sequence[Any]) -> None:
    """Raise when a harness-authored provider chat message leaks oracle content.

    Only harness-constructed turns (prompt, system, and feedback text) are
    scanned. Assistant turns echo the model's own previous candidate code, so
    benign source such as ``{"target": 2}`` must not trip the substring
    deny-list; oracle isolation for those turns is enforced structurally
    because the harness only ever places prior candidate code in them.

    Args:
        messages: Chat messages as mappings with ``role``/``content`` fields,
            or objects with ``role``/``content`` attributes.

    Raises:
        OracleLeakError: If oracle content is present in a harness-authored
            message.
    """
    for message in messages:
        if isinstance(message, Mapping):
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
        if str(role) == "assistant":
            continue
        assert_provider_text_excludes_oracle(str(content or ""))
