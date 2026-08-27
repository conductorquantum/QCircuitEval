"""Human-readable benchmark report formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tabulate import tabulate


def format_run_summary(summary: Mapping[str, Any]) -> str:
    """Return a concise human-readable run summary table.

    Args:
        summary: ``payload["summary"]`` returned by
            :class:`qceval.core.runner.BenchmarkRunner`.

    Returns:
        Plain-text table suitable for CLI output.
    """
    table = tabulate(
        _summary_rows(summary),
        headers=[
            "scope",
            "pass rate",
            "passed",
            "failed",
            "generated",
            "provider",
            "compile",
            "runtime",
            "infra",
            "total",
        ],
        tablefmt="github",
        disable_numparse=True,
    )
    lineage = summary.get("feedback_lineage")
    heading = "Benchmark attempt-record summary" if isinstance(lineage, Mapping) else "Benchmark summary"
    lines = [heading, table]
    pass_k = summary.get("pass_at_k")
    if isinstance(pass_k, Mapping):
        lines.append(
            "Pass@{k}: {rate} ({expected:.2f} expected passed / {tasks} tasks)".format(
                k=pass_k.get("k", "?"),
                rate=_format_rate(pass_k.get("pass_at_k", 0.0)),
                expected=float(pass_k.get("expected_passed", 0.0)),
                tasks=pass_k.get("tasks_evaluated", 0),
            )
        )
    feedback = summary.get("feedback")
    if isinstance(feedback, Mapping) and not isinstance(lineage, Mapping):
        lines.append(
            "Feedback: final_pass_rate={rate} max_attempts={attempts}".format(
                rate=_format_rate(feedback.get("final_pass_rate", 0.0)),
                attempts=feedback.get("max_attempts", "?"),
            )
        )
    if isinstance(lineage, Mapping):
        lines.append(
            "Feedback lineage: terminal_pass_rate={rate} passed={passed}/{assigned} "
            "complete={complete} invalid={invalid} provenance={provenance}".format(
                rate=_format_rate(lineage.get("terminal_pass_rate", 0.0)),
                passed=lineage.get("terminal_verified_passes", 0),
                assigned=lineage.get("assigned_chains", 0),
                complete=lineage.get("complete_chains", 0),
                invalid=lineage.get("invalid_chains", 0),
                provenance=_format_rate(lineage.get("provenance_coverage", 0.0)),
            )
        )
    semantic = summary.get("semantic")
    if isinstance(semantic, Mapping):
        lines.append(
            "Behavior grading: strict={strict} coverage={coverage} adjudicated={adjudicated} "
            "nonsemantic={nonsemantic}".format(
                strict=_format_rate(semantic.get("strict_pass_rate", 0.0)),
                coverage=_format_rate(semantic.get("coverage", 0.0)),
                adjudicated=_format_rate(semantic.get("adjudicated_pass_rate", 0.0)),
                nonsemantic=semantic.get("nonsemantic_denominator", 0),
            )
        )
    cost = summary.get("cost")
    if isinstance(cost, Mapping) and _int_stat(cost, "records_with_reported_cost"):
        average = cost.get("mean_reported_cost_per_task_usd")
        average_text = "unavailable" if average is None else _format_usd(average)
        lines.append(
            "Provider-reported cost: total={total} average/logical-task={average} "
            "record coverage={reported}/{records}".format(
                total=_format_usd(cost.get("reported_total_cost_usd")),
                average=average_text,
                reported=_int_stat(cost, "records_with_reported_cost"),
                records=_int_stat(cost, "provider_records"),
            )
        )
    return "\n".join(lines)


def _summary_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    rows = [_summary_row("overall", summary)]
    detail_rows = _suite_framework_rows(summary)
    if len(detail_rows) > 1:
        rows.extend(detail_rows)
    elif not detail_rows:
        framework_rows = _framework_rows(summary)
        if len(framework_rows) > 1:
            rows.extend(framework_rows)
    return rows


def _suite_framework_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    by_suite_framework = summary.get("by_suite_framework")
    if not isinstance(by_suite_framework, Mapping):
        return []
    rows: list[list[str]] = []
    for suite, frameworks in sorted(by_suite_framework.items()):
        if not isinstance(frameworks, Mapping):
            continue
        for framework, stats in sorted(frameworks.items()):
            if isinstance(stats, Mapping):
                rows.append(_summary_row(f"{suite}/{framework}", stats))
    return rows


def _framework_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    by_framework = summary.get("by_framework")
    if not isinstance(by_framework, Mapping):
        return []
    return [
        _summary_row(str(framework), stats)
        for framework, stats in sorted(by_framework.items())
        if isinstance(stats, Mapping)
    ]


def _summary_row(scope: str, stats: Mapping[str, Any]) -> list[str]:
    total = _int_stat(stats, "total_tasks")
    passed = _int_stat(stats, "passed")
    return [
        scope,
        _format_rate(stats.get("pass_rate", 0.0)),
        f"{passed}/{total}",
        str(_int_stat(stats, "failed")),
        str(_int_stat(stats, "generated")),
        str(_int_stat(stats, "provider_failures")),
        str(_int_stat(stats, "compile_failures")),
        str(_int_stat(stats, "run_failures")),
        str(_int_stat(stats, "infrastructure_failures")),
        str(total),
    ]


def _int_stat(stats: Mapping[str, Any], key: str) -> int:
    value = stats.get(key, 0)
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_rate(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "0.0%"


def _format_usd(value: Any) -> str:
    try:
        return f"${float(value):.6f}"
    except (TypeError, ValueError):
        return "unavailable"
