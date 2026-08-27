"""Feedback message construction helpers."""

from __future__ import annotations

from collections.abc import Sequence

from qceval.core.prompt_safety import assert_provider_text_excludes_oracle
from qceval.models import BenchmarkRecord, ProviderMessage


def _feedback_messages(
    prompt: str,
    previous_records: Sequence[BenchmarkRecord],
    feedback_message: str,
) -> tuple[ProviderMessage, ...]:
    """Assemble multi-turn repair chat without grading-oracle content.

    Args:
        prompt: Original task prompt.
        previous_records: Prior attempts in order.
        feedback_message: Latest redacted feedback user turn.

    Returns:
        Provider chat messages for the next repair attempt.
    """
    assert_provider_text_excludes_oracle(feedback_message)
    messages = [ProviderMessage(role="user", content=prompt)]
    for index, record in enumerate(previous_records):
        messages.append(ProviderMessage(role="assistant", content=record.provider_response.code or ""))
        next_feedback = feedback_message
        if index < len(previous_records) - 1:
            next_feedback = str(previous_records[index + 1].feedback.get("message_to_model") or "")
        if next_feedback:
            assert_provider_text_excludes_oracle(next_feedback)
            messages.append(ProviderMessage(role="user", content=next_feedback))
    return tuple(messages)
