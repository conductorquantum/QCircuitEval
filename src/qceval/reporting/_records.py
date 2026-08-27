"""Shared benchmark-record helpers for report generation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from qceval.models import BenchmarkRecord


def record_verified_status(record: BenchmarkRecord) -> str:
    """Resolve the normalized verified status for one record.

    Args:
        record: Benchmark record to inspect.

    Returns:
        Explicit semantic status when available, otherwise a pass/fail
        fallback.
    """
    evaluation = record.evaluation
    if evaluation is None:
        if record.status == "generated":
            return "ungraded"
        if record.status == "provider_failed":
            return "provider_error"
        return "verified_fail"
    if record.status == "infrastructure_error":
        return "infrastructure_error"
    if evaluation.verified_status:
        return str(evaluation.verified_status)
    details = evaluation.grader_details or {}
    if details.get("verified_status"):
        return str(details["verified_status"])
    return "verified_pass" if evaluation.passed else "verified_fail"


def verified_counts(records: list[BenchmarkRecord]) -> Counter[str]:
    """Count normalized verified statuses.

    Args:
        records: Benchmark records to aggregate.

    Returns:
        Frequency of each normalized verified status.
    """
    return Counter(record_verified_status(record) for record in records)


def failed_count(total: int, passed: int, counts: Counter[Any]) -> int:
    """Count verification failures excluding infrastructure failures.

    Args:
        total: Total result count.
        passed: Verified pass count.
        counts: Status counts including provider, compile, and run failures.

    Returns:
        Nonnegative count of semantic verification failures.
    """
    return max(
        0,
        total
        - passed
        - counts["generated"]
        - counts["provider_failed"]
        - counts["compile_failed"]
        - counts["run_failed"]
        - counts["infrastructure_error"],
    )
