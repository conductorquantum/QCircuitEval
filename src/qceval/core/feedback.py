"""Build bounded repair feedback from failed benchmark records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qceval.core.prompt_safety import assert_provider_text_excludes_oracle, oracle_key_is_blocked
from qceval.models import BenchmarkRecord, FeedbackPolicy
from qceval.serialization import to_jsonable


@dataclass(frozen=True)
class FeedbackResult:
    """Diagnostic feedback sent to a model for one repair attempt.

    Attributes:
        reason: Stable failure reason.
        message_to_model: Bounded user message for the next provider request.
        truncated: Whether any diagnostic text was truncated.
        metadata: JSON-compatible feedback metadata for output records.
    """

    reason: str
    message_to_model: str
    truncated: bool
    metadata: Mapping[str, Any]


def build_feedback(record: BenchmarkRecord, *, max_chars: int, policy: FeedbackPolicy | None = None) -> FeedbackResult:
    """Build fair repair feedback from one failed record.

    Feedback never includes behavior contracts, semantic targets, hashes,
    grader metrics, thresholds, taxonomy labels, or verifier reason codes. Only
    bounded candidate-observable execution information is shared with the model.

    Args:
        record: Failed benchmark record to diagnose.
        max_chars: Maximum characters for diagnostic text blocks.
        policy: Versioned feedback policy. Defaults to the current policy.

    Returns:
        Feedback result containing model-facing message and record metadata.

    Raises:
        ValueError: If ``max_chars`` is not positive.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    selected_policy = FeedbackPolicy() if policy is None else policy
    reason = _reason(record)
    message, truncated = _message(record, reason=reason, max_chars=max_chars)
    assert_provider_text_excludes_oracle(message)
    metadata = {
        "reason": reason,
        "policy_version": selected_policy.version,
        "source_attempt_index": record.attempt_index,
        "previous_status": record.status,
        "previous_error_type": None if record.evaluation is None else record.evaluation.error_type,
        "message_to_model": message,
        "truncated": truncated,
    }
    return FeedbackResult(reason=reason, message_to_model=message, truncated=truncated, metadata=metadata)


def _reason(record: BenchmarkRecord) -> str:
    if record.status == "provider_failed":
        return "provider_failed"
    if record.evaluation is None:
        return "provider_failed"
    if not record.evaluation.compiled:
        return "compile_failed"
    if not record.evaluation.ran:
        return "run_failed"
    return "failed"


def _message(record: BenchmarkRecord, *, reason: str, max_chars: int) -> tuple[str, bool]:
    if reason == "provider_failed":
        detail, truncated = _truncate(record.provider_response.error or "No provider error reported.", max_chars)
        return (
            "Previous generation failed before code was available.\n\n"
            f"Provider error:\n{detail}\n\n"
            "Return full corrected code for the required entry point.",
            truncated,
        )
    if reason == "compile_failed":
        detail, truncated = _truncate(_evaluation_error(record), max_chars)
        return (
            "Previous code did not compile.\n\n"
            f"Compile error:\n{detail}\n\n"
            f"Return full corrected code defining `{record.entry_point}`.",
            truncated,
        )
    if reason == "run_failed":
        detail, truncated = _truncate(_evaluation_error(record), max_chars)
        return (
            "Previous code compiled but failed at runtime.\n\n"
            f"Runtime error:\n{detail}\n\n"
            "Return full corrected code for the required entry point.",
            truncated,
        )
    detail, truncated = _truncate(_semantic_summary(record), max_chars)
    return (
        "Previous code ran but did not satisfy the task checks.\n\n"
        f"Candidate-observable output:\n{detail}\n\n"
        "Return full corrected code for the required entry point.",
        truncated,
    )


def _evaluation_error(record: BenchmarkRecord) -> str:
    if record.evaluation is None:
        return "No evaluation error reported."
    return record.evaluation.error or record.evaluation.error_type or "No evaluation error reported."


def _semantic_summary(record: BenchmarkRecord) -> str:
    evaluation = record.evaluation
    if evaluation is None:
        return "{}"
    payload: dict[str, Any] = {
        "output_vector": evaluation.probabilities,
        "execution_metadata": _safe_execution_metadata(evaluation.execution_metadata),
    }
    return json.dumps(to_jsonable(payload), sort_keys=True)


def _safe_execution_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "num_qubits",
        "measurement_count",
        "return_measurement_count",
        "non_measurement_operation_count",
        "entangling_gate_count",
        "operation_counts",
        "gate_family_counts",
        "probability_method",
        "kernel_argument_count",
        "measurement_qubits",
        "interaction_pairs",
        "returned_counts",
        "returned_probabilities",
    }
    return {key: metadata[key] for key in allowed if key in metadata and not oracle_key_is_blocked(key)}


def feedback_stop_reason(record: BenchmarkRecord) -> str | None:
    """Return a terminal reason when a feedback chain must not continue.

    Provider failures and verifier-side non-decisions are censored terminal
    outcomes. Candidate compilation, runtime, and semantic failures remain
    repairable. The attempt budget is handled separately by the runner.

    Args:
        record: Completed attempt record.

    Returns:
        Stable terminal reason, or ``None`` when repair may continue.
    """
    if record.status == "infrastructure_error":
        return "grader_nondecision"
    semantic_status = _semantic_status(record)
    evaluation = record.evaluation
    if semantic_status is not None:
        return _semantic_stop_reason(record, semantic_status)
    if evaluation is not None and evaluation.verified_status == "verified_pass":
        return "verified_pass"
    if record.status == "passed":
        if evaluation is not None and evaluation.verified_status is None:
            return "verified_pass"
        return "grader_nondecision"
    if record.status == "provider_failed" or record.evaluation is None:
        return "provider_failure"
    return None


def _semantic_stop_reason(record: BenchmarkRecord, semantic_status: str) -> str | None:
    if semantic_status == "verified_pass":
        return "verified_pass"
    if semantic_status == "resource_limit":
        return "resource_limit"
    if semantic_status == "execution_error" and _failure_origin(record) == "grader_verification":
        return "grader_nondecision"
    if semantic_status not in {"semantic_fail", "execution_error"}:
        return "grader_nondecision"
    return None


def terminal_reason(record: BenchmarkRecord, *, max_attempts: int) -> str | None:
    """Return the chain terminal reason after applying the attempt budget.

    Args:
        record: Completed attempt record.
        max_attempts: Configured attempt cap including the initial generation.

    Returns:
        Stable terminal reason, or ``None`` when repair may continue.
    """
    stop_reason = feedback_stop_reason(record)
    if stop_reason is not None:
        return stop_reason
    if record.attempt_index >= max_attempts - 1:
        return "max_attempts_exhausted"
    return None


def _semantic_status(record: BenchmarkRecord) -> str | None:
    evaluation = record.evaluation
    if evaluation is None or not isinstance(evaluation.semantic_result, Mapping):
        return None
    status = evaluation.semantic_result.get("status")
    return status if isinstance(status, str) else None


def _failure_origin(record: BenchmarkRecord) -> str | None:
    evaluation = record.evaluation
    if evaluation is None or not isinstance(evaluation.semantic_result, Mapping):
        return None
    diagnostics = evaluation.semantic_result.get("diagnostics")
    if not isinstance(diagnostics, list):
        return None
    for item in diagnostics:
        if isinstance(item, Mapping) and item.get("name") == "failure_origin":
            value = item.get("value")
            return value if isinstance(value, str) else None
    return None


def _truncate(value: str, max_chars: int) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    if max_chars <= 3:
        return value[:max_chars], True
    return value[: max_chars - 3] + "...", True
