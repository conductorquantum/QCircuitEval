"""Provider-reported benchmark cost aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from qceval.models import BenchmarkRecord


def cost_summary(records: list[BenchmarkRecord]) -> dict[str, Any]:
    """Summarize reported request cost without treating missing values as zero.

    Args:
        records: Provider attempt records to aggregate by logical task.

    Returns:
        Cost totals, coverage counts, and coverage-gated per-task averages.
    """
    grouped: dict[tuple[str, str, str], list[BenchmarkRecord]] = defaultdict(list)
    reported_costs: list[float] = []
    for record in records:
        grouped[(record.suite, record.framework, record.task_id)].append(record)
        cost = _reported_cost(record)
        if cost is not None:
            reported_costs.append(cost)

    complete_task_costs: list[float] = []
    for task_records in grouped.values():
        costs = [_reported_cost(record) for record in task_records]
        if costs and all(cost is not None for cost in costs):
            complete_task_costs.append(sum(cost for cost in costs if cost is not None))

    record_count = len(records)
    task_count = len(grouped)
    reported_record_count = len(reported_costs)
    complete_task_count = len(complete_task_costs)
    complete_coverage = record_count == reported_record_count and task_count == complete_task_count
    return {
        "currency": "USD",
        "source": "provider_reported",
        "provider_records": record_count,
        "records_with_reported_cost": reported_record_count,
        "record_cost_coverage": reported_record_count / record_count if record_count else 0.0,
        "logical_tasks": task_count,
        "logical_tasks_with_complete_reported_cost": complete_task_count,
        "task_cost_coverage": complete_task_count / task_count if task_count else 0.0,
        "reported_total_cost_usd": sum(reported_costs),
        "mean_reported_cost_per_complete_task_usd": (
            sum(complete_task_costs) / complete_task_count if complete_task_count else None
        ),
        "mean_reported_cost_per_task_usd": (
            sum(complete_task_costs) / task_count if complete_coverage and task_count else None
        ),
    }


def _reported_cost(record: BenchmarkRecord) -> float | None:
    usage = record.provider_response.usage
    value = None if usage is None else usage.cost_usd
    if value is None:
        raw_response = record.provider_response.raw_response
        raw_usage = raw_response.get("usage") if isinstance(raw_response, Mapping) else None
        value = raw_usage.get("cost") if isinstance(raw_usage, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    cost = float(value)
    return cost if math.isfinite(cost) and cost >= 0.0 else None
