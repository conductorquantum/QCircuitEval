"""Semantic-verification report aggregation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from qceval.models import BenchmarkRecord
from qceval.reporting._records import record_verified_status
from qceval.semantics.result_record import read_result_record

SEMANTIC_STATUSES = (
    "verified_pass",
    "semantic_fail",
    "execution_error",
    "resource_limit",
)


def semantic_summary(records: list[BenchmarkRecord]) -> dict[str, Any] | None:
    """Summarize decisive behavior and operational failures.

    Args:
        records: Completed benchmark records.

    Returns:
        Semantic rates, transitions, version groups, and resources, or ``None``
        when records have no semantic metadata.
    """
    if not any(_has_semantic_metadata(record) for record in records):
        return None
    statuses = Counter(_record_semantic_status(record) for record in records)
    total = len(records)
    passed = statuses["verified_pass"]
    decisive = passed + statuses["semantic_fail"]
    transitions = Counter(f"{record_verified_status(record)}->{_record_semantic_status(record)}" for record in records)
    version_groups, warnings = _semantic_version_groups(records)
    elapsed = sorted(_semantic_elapsed(record) for record in records if _semantic_record(record) is not None)
    by_framework: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_framework[record.framework][_record_semantic_status(record)] += 1
    nonsemantic = {status: statuses[status] for status in (*SEMANTIC_STATUSES[2:], "ungraded")}
    return {
        "assigned": total,
        "status_counts": dict(sorted(statuses.items())),
        "strict_pass_rate": passed / total if total else 0.0,
        "coverage": decisive / total if total else 0.0,
        "adjudicated_pass_rate": passed / decisive if decisive else 0.0,
        "nonsemantic_denominator": total - decisive,
        "nonsemantic_counts": nonsemantic,
        "status_transition_matrix": dict(sorted(transitions.items())),
        "version_groups": version_groups,
        "compatibility_warnings": warnings,
        "performance_seconds": {
            "p50": _percentile(elapsed, 0.50),
            "p95": _percentile(elapsed, 0.95),
        },
        "cache_state_counts": {"unknown": len(elapsed)},
        "by_framework": {name: dict(sorted(counts.items())) for name, counts in sorted(by_framework.items())},
    }


def _has_semantic_metadata(record: BenchmarkRecord) -> bool:
    evaluation = record.evaluation
    if evaluation is None:
        return False
    return evaluation.semantic_result is not None


def _semantic_record(record: BenchmarkRecord) -> dict[str, Any] | None:
    evaluation = record.evaluation
    if evaluation is None or not isinstance(evaluation.semantic_result, Mapping):
        return None
    try:
        return read_result_record(evaluation.semantic_result)
    except ValueError:
        return None


def _record_semantic_status(record: BenchmarkRecord) -> str:
    semantic = _semantic_record(record)
    if semantic is not None:
        return str(semantic["status"])
    if record.status in {"provider_failed", "compile_failed", "run_failed"}:
        return "execution_error"
    return "ungraded"


def _semantic_version_groups(records: list[BenchmarkRecord]) -> tuple[list[dict[str, Any]], list[str]]:
    groups: Counter[tuple[str, ...]] = Counter()
    identities_by_task: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for record in records:
        semantic = _semantic_record(record)
        if semantic is None:
            continue
        contract = semantic["contract"]
        target = semantic["target"]
        verifier = semantic["verifier"]
        identity = (
            str(contract["contract_version"]),
            str(contract["hash"]),
            str(target["version"]),
            str(target["hash"]),
            str(verifier["release_version"]),
        )
        task = (str(contract["suite"]), str(contract["task_id"]))
        groups[(task[0], task[1], *identity)] += 1
        identities_by_task[task].add(identity)
    warnings = [
        f"incompatible semantic versions for {suite}/{task_id}"
        for (suite, task_id), identities in sorted(identities_by_task.items())
        if len(identities) > 1
    ]
    rows = [
        {
            "suite": key[0],
            "task_id": key[1],
            "contract_version": key[2],
            "contract_hash": key[3],
            "target_version": key[4],
            "target_hash": key[5],
            "verifier_version": key[6],
            "records": count,
        }
        for key, count in sorted(groups.items())
    ]
    return rows, warnings


def _semantic_elapsed(record: BenchmarkRecord) -> float:
    semantic = _semantic_record(record)
    if semantic is None:
        return 0.0
    resources = semantic.get("resources") or {}
    try:
        return max(0.0, float(resources.get("wall_seconds", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    index = math.ceil((len(values) - 1) * quantile)
    return values[index]
