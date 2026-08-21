"""Aggregate benchmark records into machine-readable summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from qceval.models import BenchmarkRecord, RunConfig
from qceval.reporting._records import (
    failed_count,
    record_verified_status,
    verified_counts,
)
from qceval.reporting.costs import cost_summary
from qceval.reporting.error_taxonomy import error_taxonomy_summary
from qceval.reporting.feedback_lineage import feedback_lineage_summary
from qceval.reporting.protocol import feedback_summary, pass_at_k_summary, run_protocol, task_totals
from qceval.reporting.semantic import _record_semantic_status, semantic_summary


def summarize(records: list[BenchmarkRecord], run_config: RunConfig | None = None) -> dict[str, Any]:
    """Build an aggregate summary for benchmark records.

    Args:
        records: Completed records in output order.
        run_config: Optional protocol configuration for repeated sampling and
            feedback.

    Returns:
        JSON-compatible aggregate and per-task report data.
    """
    status_counts = Counter(record.status for record in records)
    total = len(records)
    counts = verified_counts(records)
    passed = counts["verified_pass"]
    assigned = _assigned_total(status_counts, total)
    scoreable = _scoreable_total(status_counts, total)
    protocol = run_protocol(run_config)
    summary = {
        "total_tasks": total,
        "assigned_tasks": assigned,
        "passed": passed,
        "failed": failed_count(total, passed, status_counts),
        "generated": status_counts["generated"],
        "provider_failures": status_counts["provider_failed"],
        "compile_failures": status_counts["compile_failed"],
        "run_failures": status_counts["run_failed"],
        "infrastructure_failures": status_counts["infrastructure_error"],
        "rerun_required_tasks": status_counts["infrastructure_error"],
        "scoreable_tasks": scoreable,
        "pass_rate": passed / assigned if assigned else 0.0,
        "pass_rate_denominator": "assigned_tasks",
        "verified_status_counts": dict(counts),
        "by_framework": _by_framework(records),
        "by_suite": _by_suite(records),
        "by_suite_framework": _by_suite_framework(records),
        "tasks": [_task_summary(record) for record in records],
        "run_protocol": protocol,
        "task_totals": task_totals(records),
        "cost": cost_summary(records),
        "routing": _routing_summary(records),
        "error_taxonomy": error_taxonomy_summary(records),
    }
    if protocol["samples_per_task"] > 1:
        summary["pass_at_k"] = pass_at_k_summary(records, protocol["pass_k"])
    if protocol["max_attempts"] > 1:
        summary["feedback"] = feedback_summary(records, protocol["max_attempts"])
        summary["feedback_lineage"] = feedback_lineage_summary(records, protocol["max_attempts"])
    semantic = semantic_summary(records)
    if semantic is not None:
        summary["semantic"] = semantic
    return summary


def _by_framework(records: list[BenchmarkRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped[record.framework].append(record)
    return {framework: _framework_summary(items) for framework, items in sorted(grouped.items())}


def _by_suite(records: list[BenchmarkRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped[record.suite].append(record)
    return {suite: _framework_summary(items) for suite, items in sorted(grouped.items())}


def _by_suite_framework(records: list[BenchmarkRecord]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[BenchmarkRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.suite][record.framework].append(record)
    return {
        suite: {framework: _framework_summary(items) for framework, items in sorted(frameworks.items())}
        for suite, frameworks in sorted(grouped.items())
    }


def _framework_summary(records: list[BenchmarkRecord]) -> dict[str, Any]:
    status_counts = Counter(record.status for record in records)
    counts = verified_counts(records)
    total = len(records)
    passed = counts["verified_pass"]
    assigned = _assigned_total(status_counts, total)
    scoreable = _scoreable_total(status_counts, total)
    return {
        "total_tasks": total,
        "assigned_tasks": assigned,
        "passed": passed,
        "failed": failed_count(total, passed, status_counts),
        "generated": status_counts["generated"],
        "provider_failures": status_counts["provider_failed"],
        "compile_failures": status_counts["compile_failed"],
        "run_failures": status_counts["run_failed"],
        "infrastructure_failures": status_counts["infrastructure_error"],
        "rerun_required_tasks": status_counts["infrastructure_error"],
        "scoreable_tasks": scoreable,
        "pass_rate": passed / assigned if assigned else 0.0,
        "pass_rate_denominator": "assigned_tasks",
        "verified_status_counts": dict(counts),
        "cost": cost_summary(records),
    }


def _assigned_total(status_counts: Counter[Any], total: int) -> int:
    """Return assigned graded records, excluding generation-only output."""
    return max(0, total - status_counts["generated"])


def _routing_summary(records: list[BenchmarkRecord]) -> dict[str, Any] | None:
    """Summarize endpoint-pinned provenance by frozen route revision."""
    segments: dict[tuple[str, str], dict[str, Any]] = {}
    pinned_records = 0
    verified_records = 0
    for record in records:
        metadata = record.provider_response.metadata
        route = metadata.get("route") if isinstance(metadata, Mapping) else None
        if not isinstance(route, Mapping) or not route.get("endpoint_tag"):
            continue
        pinned_records += 1
        route_revision = str(route.get("route_revision") or "unknown")
        endpoint_tag = str(route["endpoint_tag"])
        key = (route_revision, endpoint_tag)
        segment = segments.setdefault(
            key,
            {
                "route_revision": route_revision,
                "endpoint_tag": endpoint_tag,
                "selected_provider": route.get("selected_provider"),
                "selected_model": route.get("selected_model"),
                "max_output_tokens": route.get("max_output_tokens"),
                "output_limit_source": route.get("output_limit_source"),
                "endpoint_cap_status": route.get("endpoint_cap_status"),
                "output_token_parameter": route.get("output_token_parameter"),
                "records": 0,
                "route_verified_records": 0,
                "records_with_reported_cost": 0,
                "reported_cost_usd": 0.0,
                "finish_reason_length_records": 0,
            },
        )
        segment["records"] += 1
        if route.get("route_verified") is True:
            segment["route_verified_records"] += 1
            verified_records += 1
        usage = record.provider_response.usage
        if usage is not None and usage.cost_usd is not None:
            segment["records_with_reported_cost"] += 1
            segment["reported_cost_usd"] += usage.cost_usd
        if metadata.get("finish_reason") == "length":
            segment["finish_reason_length_records"] += 1
    if not segments:
        return None
    return {
        "pinned_records": pinned_records,
        "route_verified_records": verified_records,
        "all_pinned_records_route_verified": verified_records == pinned_records,
        "segments": [segments[key] for key in sorted(segments)],
    }


def _scoreable_total(status_counts: Counter[Any], total: int) -> int:
    """Return assigned records that do not require an infrastructure rerun."""
    return max(
        0,
        _assigned_total(status_counts, total) - status_counts["infrastructure_error"],
    )


def _task_summary(record: BenchmarkRecord) -> dict[str, Any]:
    evaluation = record.evaluation
    return {
        "framework": record.framework,
        "suite": record.suite,
        "task_id": record.task_id,
        "sample_index": record.sample_index,
        "attempt_index": record.attempt_index,
        "entry_point": record.entry_point,
        "category": record.category,
        "status": record.status,
        "passed": None if evaluation is None else evaluation.passed,
        "verified_status": record_verified_status(record),
        "metric": None if evaluation is None else evaluation.metric,
        "metric_name": None if evaluation is None else evaluation.metric_name,
        "error_type": None if evaluation is None else evaluation.error_type,
        "semantic_status": _record_semantic_status(record),
        "error_taxonomy": record.to_dict()["error_taxonomy"],
    }
