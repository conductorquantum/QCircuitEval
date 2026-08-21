"""Chain-level reporting for the versioned Feedback@N protocol."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qceval.core.feedback import terminal_reason
from qceval.core.lineage import (
    FEEDBACK_LINEAGE_SCHEMA_VERSION,
    REQUEST_TRACE_SCHEMA_VERSION,
    sha256_text,
)
from qceval.error_taxonomy import (
    AXIS_GROUP_LABELS,
    AXIS_GROUPS,
    ERROR_AXES,
    ERROR_TAXONOMY_VERSION,
)
from qceval.models import BenchmarkRecord
from qceval.reporting._records import record_verified_status

DIVERGING_PLOT_SCHEMA_VERSION = "qceval.feedback_diverging_plot.v1"
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class _ChainAudit:
    """Validated observations for one assigned feedback chain."""

    records: tuple[BenchmarkRecord, ...]
    valid: bool
    complete: bool
    provenance_complete: bool
    stop_reason: str
    issues: tuple[str, ...]


_TaxonomyState = tuple[str, set[str]] | None
_ChainTaxonomyState = tuple[_ChainAudit, _TaxonomyState, _TaxonomyState]


def feedback_lineage_summary(records: list[BenchmarkRecord], max_attempts: int) -> dict[str, Any]:
    """Aggregate chain outcomes, costs, and taxonomy transitions.

    All unconditional rates use the number of assigned chains. Taxonomy
    clearance and persistence are reported only when both the initial and
    terminal records are classifiable; unclassified or censored records remain
    explicit unknowns rather than being treated as axis absences.

    Args:
        records: Initial and repair-attempt records.
        max_attempts: Configured attempt cap, including the initial generation.

    Returns:
        JSON-compatible lineage report with framework strata.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    summary = _lineage_summary(records, max_attempts=max_attempts)
    by_framework: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        by_framework[record.framework].append(record)
    summary["by_framework"] = {
        framework: _lineage_summary(items, max_attempts=max_attempts)
        for framework, items in sorted(by_framework.items())
    }
    return summary


def _lineage_summary(records: list[BenchmarkRecord], *, max_attempts: int) -> dict[str, Any]:
    chains = [_audit_chain(items, max_attempts=max_attempts) for items in _group_chains(records).values()]
    assigned = len(chains)
    infrastructure_chains = [
        chain for chain in chains if any(record.status == "infrastructure_error" for record in chain.records)
    ]
    scoreable = assigned - len(infrastructure_chains)
    scoreable_terminal_passes = sum(
        chain.stop_reason == "verified_pass" and chain not in infrastructure_chains for chain in chains
    )
    valid = sum(chain.valid for chain in chains)
    complete = sum(chain.valid and chain.complete for chain in chains)
    provenance_complete = sum(chain.valid and chain.provenance_complete for chain in chains)
    issues = Counter(issue for chain in chains for issue in chain.issues)
    terminal_reasons = Counter(chain.stop_reason for chain in chains)
    generations = sum(len(chain.records) for chain in chains)
    return {
        "schema_version": FEEDBACK_LINEAGE_SCHEMA_VERSION,
        "unit": "feedback_chain",
        "denominator": "all_assigned_feedback_chains",
        "max_attempts": max_attempts,
        "max_repairs": max_attempts - 1,
        "assigned_chains": assigned,
        "scoreable_chains": scoreable,
        "infrastructure_chains": len(infrastructure_chains),
        "rerun_required": len(infrastructure_chains),
        "valid_chains": valid,
        "complete_chains": complete,
        "invalid_chains": assigned - valid,
        "incomplete_chains": sum(chain.valid and not chain.complete for chain in chains),
        "provenance_complete_chains": provenance_complete,
        "provenance_coverage": provenance_complete / assigned if assigned else 0.0,
        "terminal_verified_passes": terminal_reasons["verified_pass"],
        "terminal_pass_rate": scoreable_terminal_passes / scoreable if scoreable else 0.0,
        "terminal_pass_rate_denominator": "scoreable_feedback_chains",
        "mean_generations_per_chain": generations / assigned if assigned else 0.0,
        "levels": _attempt_levels(chains, max_attempts=max_attempts),
        "terminal_stop_reason_counts": dict(sorted(terminal_reasons.items())),
        "validation_issue_counts": dict(sorted(issues.items())),
        "token_usage": _token_usage(chains),
        "taxonomy_transitions": _taxonomy_transitions(chains),
    }


def _group_chains(records: list[BenchmarkRecord]) -> dict[tuple[str, ...], list[BenchmarkRecord]]:
    grouped: dict[tuple[str, ...], list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        chain_value = record.lineage.get("chain_id") if isinstance(record.lineage, Mapping) else None
        key: tuple[str, ...]
        if isinstance(chain_value, str) and chain_value:
            key = ("lineage", chain_value)
        else:
            key = (
                "legacy",
                record.provider,
                str(record.model),
                record.suite,
                record.framework,
                record.task_id,
                str(record.sample_index),
            )
        grouped[key].append(record)
    return grouped


def _audit_chain(records: list[BenchmarkRecord], *, max_attempts: int) -> _ChainAudit:
    ordered = tuple(sorted(records, key=lambda record: record.attempt_index))
    issues = _basic_chain_issues(ordered, max_attempts=max_attempts)
    issues.extend(_terminal_lineage_issues(ordered, max_attempts=max_attempts))
    provenance_complete = _provenance_complete(ordered, issues)
    valid = not issues
    final_reason = terminal_reason(ordered[-1], max_attempts=max_attempts) if ordered else None
    complete = bool(valid and final_reason is not None)
    if not valid:
        stop_reason = "invalid_chain"
    elif final_reason is None:
        stop_reason = "incomplete_chain"
    else:
        stop_reason = final_reason
    return _ChainAudit(
        records=ordered,
        valid=valid,
        complete=complete,
        provenance_complete=provenance_complete,
        stop_reason=stop_reason,
        issues=tuple(sorted(set(issues))),
    )


def _basic_chain_issues(records: tuple[BenchmarkRecord, ...], *, max_attempts: int) -> list[str]:
    issues: list[str] = []
    attempts = [record.attempt_index for record in records]
    if attempts != list(range(len(records))):
        issues.append("attempt_sequence_invalid")
    if attempts and attempts[-1] >= max_attempts:
        issues.append("attempt_budget_exceeded")
    identities = {
        (record.provider, record.model, record.suite, record.framework, record.task_id, record.sample_index)
        for record in records
    }
    if len(identities) > 1:
        issues.append("chain_identity_mismatch")
    return issues


def _terminal_lineage_issues(records: tuple[BenchmarkRecord, ...], *, max_attempts: int) -> list[str]:
    issues: list[str] = []
    for index, record in enumerate(records):
        expected_reason = terminal_reason(record, max_attempts=max_attempts)
        if index < len(records) - 1 and expected_reason is not None:
            issues.append("attempt_after_terminal_outcome")
            break
        if isinstance(record.lineage, Mapping) and record.lineage:
            if record.lineage.get("terminal") is not (expected_reason is not None):
                issues.append("lineage_terminal_mismatch")
            if record.lineage.get("stop_reason") != expected_reason:
                issues.append("lineage_stop_reason_mismatch")
    return issues


def _provenance_complete(records: tuple[BenchmarkRecord, ...], issues: list[str]) -> bool:
    complete = bool(records)
    chain_ids: set[str] = set()
    run_ids: set[str] = set()
    policy_versions: set[str] = set()
    for index, record in enumerate(records):
        record_complete, chain_value, run_value, policy_value = _record_provenance(record)
        complete = complete and record_complete
        chain_ids.update(_present(chain_value))
        run_ids.update(_present(run_value))
        policy_versions.update(_present(policy_value))
        issues.extend(_provenance_record_issues(record))
        issues.extend(_lineage_relationship_issues(records, index))
    if len(chain_ids) > 1:
        issues.append("lineage_chain_id_mismatch")
    if len(run_ids) > 1:
        issues.append("lineage_run_id_mismatch")
    if len(policy_versions) > 1:
        issues.append("lineage_policy_mismatch")
    return complete and not any(issue.startswith("lineage_") for issue in issues)


def _record_provenance(record: BenchmarkRecord) -> tuple[bool, str | None, str | None, str | None]:
    lineage = record.lineage
    if not isinstance(lineage, Mapping) or lineage.get("schema_version") != FEEDBACK_LINEAGE_SCHEMA_VERSION:
        return False, None, None, None
    chain_value = _nonempty_text(lineage.get("chain_id"))
    run_value = _nonempty_text(lineage.get("run_id"))
    policy_value = _nonempty_text(lineage.get("feedback_policy_version"))
    complete = _valid_request_trace(record.request_trace) and all(
        value is not None for value in (chain_value, run_value, policy_value)
    )
    return complete, chain_value, run_value, policy_value


def _provenance_record_issues(record: BenchmarkRecord) -> list[str]:
    issues: list[str] = []
    if record.request_trace and not _valid_request_trace(record.request_trace):
        issues.append("request_trace_invalid")
    lineage = record.lineage
    if lineage and (
        not isinstance(lineage, Mapping) or lineage.get("schema_version") != FEEDBACK_LINEAGE_SCHEMA_VERSION
    ):
        issues.append("lineage_schema_invalid")
    return issues


def _lineage_relationship_issues(records: tuple[BenchmarkRecord, ...], index: int) -> list[str]:
    record = records[index]
    lineage = record.lineage
    if not isinstance(lineage, Mapping) or not lineage:
        return []
    issues: list[str] = []
    previous = None if index == 0 else records[index - 1]
    expected_parent = None if previous is None else previous.attempt_index
    expected_parent_hash = None if previous is None else sha256_text(previous.provider_response.code)
    if lineage.get("attempt_index") != record.attempt_index:
        issues.append("lineage_attempt_mismatch")
    if lineage.get("parent_attempt_index") != expected_parent:
        issues.append("lineage_parent_mismatch")
    if lineage.get("parent_code_sha256") != expected_parent_hash:
        issues.append("lineage_parent_code_mismatch")
    if lineage.get("code_sha256") != sha256_text(record.provider_response.code):
        issues.append("lineage_code_mismatch")
    if lineage.get("feedback_source_attempt_index") != expected_parent:
        issues.append("lineage_feedback_source_mismatch")
    return issues


def _nonempty_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _present(value: str | None) -> set[str]:
    return set() if value is None else {value}


def _valid_request_trace(trace: Mapping[str, Any]) -> bool:
    if not isinstance(trace, Mapping) or trace.get("schema_version") != REQUEST_TRACE_SCHEMA_VERSION:
        return False
    messages = trace.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
        return False
    if not messages or not all(_valid_trace_message(message) for message in messages):
        return False
    prompt = trace.get("prompt")
    if not isinstance(prompt, str) or trace.get("prompt_sha256") != sha256_text(prompt):
        return False
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return trace.get("messages_sha256") == sha256_text(canonical)


def _valid_trace_message(message: Any) -> bool:
    return (
        isinstance(message, Mapping)
        and message.get("role") in {"user", "assistant", "system"}
        and isinstance(message.get("content"), str)
    )


def _attempt_levels(chains: list[_ChainAudit], *, max_attempts: int) -> list[dict[str, Any]]:
    assigned = len(chains)
    levels: list[dict[str, Any]] = []
    first_passes = [_first_verified_pass(chain) for chain in chains]
    for attempt_index in range(max_attempts):
        attempted = sum(chain.valid and len(chain.records) > attempt_index for chain in chains)
        first_passed = sum(value == attempt_index for value in first_passes)
        cumulative_passed = sum(value is not None and value <= attempt_index for value in first_passes)
        levels.append(
            {
                "attempt_index": attempt_index,
                "label": "initial" if attempt_index == 0 else f"repair_{attempt_index}",
                "attempted": attempted,
                "at_risk": attempted,
                "first_passed": first_passed,
                "first_pass_hazard": first_passed / attempted if attempted else 0.0,
                "cumulative_passed": cumulative_passed,
                "cumulative_pass_rate": cumulative_passed / assigned if assigned else 0.0,
            }
        )
    return levels


def _first_verified_pass(chain: _ChainAudit) -> int | None:
    if not chain.valid:
        return None
    for record in chain.records:
        if record_verified_status(record) == "verified_pass":
            return record.attempt_index
    return None


def _token_usage(chains: list[_ChainAudit]) -> dict[str, Any]:
    records = [record for chain in chains for record in chain.records]
    reported_records = [record for record in records if _reported_total(record) is not None]
    complete_chains = [
        chain
        for chain in chains
        if chain.valid and chain.records and all(_reported_total(record) is not None for record in chain.records)
    ]
    chain_totals = [sum(_reported_total(record) or 0 for record in chain.records) for chain in complete_chains]
    return {
        "records": len(records),
        "records_with_reported_total": len(reported_records),
        "chains_with_complete_reported_total": len(complete_chains),
        "reported_prompt_tokens": _sum_usage(records, "prompt_tokens"),
        "reported_completion_tokens": _sum_usage(records, "completion_tokens"),
        "reported_reasoning_tokens": _sum_usage(records, "reasoning_tokens"),
        "reported_cached_tokens": _sum_usage(records, "cached_tokens"),
        "records_with_reported_reasoning_tokens": _usage_coverage(records, "reasoning_tokens"),
        "records_with_reported_cached_tokens": _usage_coverage(records, "cached_tokens"),
        "reported_total_tokens": sum(_reported_total(record) or 0 for record in reported_records),
        "mean_reported_total_tokens_per_complete_chain": (
            sum(chain_totals) / len(chain_totals) if chain_totals else None
        ),
    }


def _reported_total(record: BenchmarkRecord) -> int | None:
    usage = record.provider_response.usage
    return None if usage is None else usage.total_tokens


def _sum_usage(records: list[BenchmarkRecord], field: str) -> int:
    values = []
    for record in records:
        usage = record.provider_response.usage
        value = None if usage is None else getattr(usage, field)
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
    return sum(values)


def _usage_coverage(records: list[BenchmarkRecord], field: str) -> int:
    return sum(
        record.provider_response.usage is not None
        and isinstance(getattr(record.provider_response.usage, field), int)
        and not isinstance(getattr(record.provider_response.usage, field), bool)
        for record in records
    )


def _taxonomy_transitions(chains: list[_ChainAudit]) -> dict[str, Any]:
    assigned = len(chains)
    versions: Counter[str] = Counter()
    outcome_transitions: Counter[str] = Counter()
    axis_counts: dict[str, Counter[str]] = {axis: Counter() for axis in ERROR_AXES}
    group_counts: dict[str, Counter[str]] = {group: Counter() for group in AXIS_GROUPS}
    states: list[_ChainTaxonomyState] = []
    initial_classifiable = 0
    terminal_classifiable = 0
    paired_classifiable = 0
    for chain in chains:
        initial, terminal = _chain_taxonomy_states(chain, versions)
        states.append((chain, initial, terminal))
        initial_classifiable += initial is not None
        terminal_classifiable += terminal is not None
        paired_classifiable += initial is not None and terminal is not None
        _accumulate_taxonomy_transition(initial, terminal, outcome_transitions, axis_counts)
        _accumulate_group_transitions(initial, terminal, group_counts)
    group_summaries = {
        group: {
            "label": AXIS_GROUP_LABELS[group],
            "axes": list(axes),
            **_axis_transition_summary(group_counts[group], assigned=assigned),
        }
        for group, axes in AXIS_GROUPS.items()
    }
    return {
        "taxonomy_version": ERROR_TAXONOMY_VERSION,
        "denominator": "all_assigned_feedback_chains",
        "assigned_chains": assigned,
        "initial_classifiable_chains": initial_classifiable,
        "terminal_classifiable_chains": terminal_classifiable,
        "paired_classifiable_chains": paired_classifiable,
        "initial_classification_coverage": initial_classifiable / assigned if assigned else 0.0,
        "terminal_classification_coverage": terminal_classifiable / assigned if assigned else 0.0,
        "paired_classification_coverage": paired_classifiable / assigned if assigned else 0.0,
        "taxonomy_version_counts": dict(sorted(versions.items())),
        "outcome_transition_counts": dict(sorted(outcome_transitions.items())),
        "axes": {axis: _axis_transition_summary(counts, assigned=assigned) for axis, counts in axis_counts.items()},
        "axis_groups": {group: list(axes) for group, axes in AXIS_GROUPS.items()},
        "group_labels": dict(AXIS_GROUP_LABELS),
        "groups": group_summaries,
        "diverging_plot": _diverging_plot_summary(
            states,
            group_summaries,
            assigned=assigned,
            paired_classification_coverage=paired_classifiable / assigned if assigned else 0.0,
        ),
    }


def _chain_taxonomy_states(
    chain: _ChainAudit,
    versions: Counter[str],
) -> tuple[_TaxonomyState, _TaxonomyState]:
    if not chain.valid or not chain.complete or not chain.records:
        return None, None
    return _taxonomy_state(chain.records[0], versions), _taxonomy_state(chain.records[-1], versions)


def _accumulate_taxonomy_transition(
    initial: _TaxonomyState,
    terminal: _TaxonomyState,
    outcome_transitions: Counter[str],
    axis_counts: dict[str, Counter[str]],
) -> None:
    initial_outcome = "unknown" if initial is None else initial[0]
    terminal_outcome = "unknown" if terminal is None else terminal[0]
    outcome_transitions[f"{initial_outcome}->{terminal_outcome}"] += 1
    if initial is None or terminal is None:
        for counts in axis_counts.values():
            counts["unknown"] += 1
        return
    initial_axes = initial[1]
    terminal_axes = terminal[1]
    for axis, counts in axis_counts.items():
        _accumulate_axis_transition(axis in initial_axes, axis in terminal_axes, counts)


def _accumulate_group_transitions(
    initial: _TaxonomyState,
    terminal: _TaxonomyState,
    group_counts: dict[str, Counter[str]],
) -> None:
    """Count each error family at most once per chain endpoint."""
    if initial is None or terminal is None:
        for counts in group_counts.values():
            counts["unknown"] += 1
        return
    initial_axes = initial[1]
    terminal_axes = terminal[1]
    for group, axes in AXIS_GROUPS.items():
        _accumulate_axis_transition(
            bool(initial_axes.intersection(axes)),
            bool(terminal_axes.intersection(axes)),
            group_counts[group],
        )


def _accumulate_axis_transition(initial_has: bool, terminal_has: bool, counts: Counter[str]) -> None:
    counts["initial"] += initial_has
    counts["terminal"] += terminal_has
    if initial_has and terminal_has:
        counts["persistent"] += 1
    elif initial_has:
        counts["cleared"] += 1
    elif terminal_has:
        counts["surfaced"] += 1
    else:
        counts["absent"] += 1


def _taxonomy_state(record: BenchmarkRecord, versions: Counter[str]) -> tuple[str, set[str]] | None:
    taxonomy = record.to_dict().get("error_taxonomy")
    if not isinstance(taxonomy, Mapping):
        return None
    version = str(taxonomy.get("taxonomy_version", "missing"))
    versions[version] += 1
    if version != ERROR_TAXONOMY_VERSION:
        return None
    outcome = taxonomy.get("outcome")
    axes_value = taxonomy.get("axes")
    unclassified = taxonomy.get("unclassified_reason_codes")
    axes: set[str] = set()
    if isinstance(axes_value, Sequence) and not isinstance(axes_value, str | bytes):
        axes = {axis for axis in axes_value if isinstance(axis, str) and axis in ERROR_AXES}
    if outcome == "verified_pass":
        return "verified_pass", set()
    if outcome == "observed_error" and axes and not _nonempty_strings(unclassified):
        return "observed_error", axes
    return None


def _nonempty_strings(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and any(isinstance(item, str) and item for item in value)
    )


def _axis_transition_summary(counts: Counter[str], *, assigned: int) -> dict[str, Any]:
    initial_errors = counts["cleared"] + counts["persistent"]
    initial_absences = counts["surfaced"] + counts["absent"]
    return {
        "initial_count": counts["initial"],
        "initial_rate": counts["initial"] / assigned if assigned else 0.0,
        "terminal_count": counts["terminal"],
        "terminal_rate": counts["terminal"] / assigned if assigned else 0.0,
        "cleared_count": counts["cleared"],
        "cleared_rate": counts["cleared"] / assigned if assigned else 0.0,
        "persistent_count": counts["persistent"],
        "persistent_rate": counts["persistent"] / assigned if assigned else 0.0,
        "surfaced_count": counts["surfaced"],
        "surfaced_rate": counts["surfaced"] / assigned if assigned else 0.0,
        "absent_count": counts["absent"],
        "absent_rate": counts["absent"] / assigned if assigned else 0.0,
        "unknown_count": counts["unknown"],
        "unknown_rate": counts["unknown"] / assigned if assigned else 0.0,
        "conditional_clearance_rate": counts["cleared"] / initial_errors if initial_errors else None,
        "conditional_persistence_rate": counts["persistent"] / initial_errors if initial_errors else None,
        "conditional_surface_rate": counts["surfaced"] / initial_absences if initial_absences else None,
        "net_reduction_count": counts["cleared"] - counts["surfaced"],
        "net_reduction_rate": (counts["cleared"] - counts["surfaced"]) / assigned if assigned else 0.0,
    }


def _diverging_plot_summary(
    states: list[_ChainTaxonomyState],
    groups: dict[str, dict[str, Any]],
    *,
    assigned: int,
    paired_classification_coverage: float,
) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    net_delta_count = 0
    for group, summary in groups.items():
        cleared_count = int(summary["cleared_count"])
        surfaced_count = int(summary["surfaced_count"])
        net_count = cleared_count - surfaced_count
        net_delta_count += net_count
        categories[group] = {
            "label": summary["label"],
            "axes": summary["axes"],
            "cleared_count": cleared_count,
            "cleared_rate": summary["cleared_rate"],
            "cleared_percentage_points": 100.0 * summary["cleared_rate"],
            "surfaced_count": surfaced_count,
            "surfaced_rate": summary["surfaced_rate"],
            "surfaced_percentage_points": -100.0 * summary["surfaced_rate"],
            "persistent_count": summary["persistent_count"],
            "persistent_rate": summary["persistent_rate"],
            "persistent_percentage_points": 100.0 * summary["persistent_rate"],
            "unknown_count": summary["unknown_count"],
            "unknown_rate": summary["unknown_rate"],
            "unknown_percentage_points": 100.0 * summary["unknown_rate"],
            "net_reduction_count": net_count,
            "net_reduction_rate": net_count / assigned if assigned else 0.0,
            "net_reduction_percentage_points": 100.0 * net_count / assigned if assigned else 0.0,
        }
    net_rate = net_delta_count / assigned if assigned else 0.0
    return {
        "schema_version": DIVERGING_PLOT_SCHEMA_VERSION,
        "unit": "error_family_transitions_per_assigned_feedback_chain",
        "denominator": "all_assigned_feedback_chains",
        "category_order": list(AXIS_GROUPS),
        "positive_direction": "cleared_after_feedback",
        "negative_direction": "surfaced_after_feedback",
        "categories": categories,
        "net_delta": {
            "definition": "sum_family_cleared_rate_minus_sum_family_surfaced_rate",
            "count": net_delta_count,
            "rate": net_rate,
            "percentage_points": 100.0 * net_rate,
            "confidence_interval_95": _task_cluster_bootstrap_interval(states),
        },
        "paired_classification_coverage": paired_classification_coverage,
        "interpretation_notes": [
            "Each family is a union of its taxonomy axes within a chain endpoint.",
            "A chain may contribute to more than one family, so stacked incidence can exceed 100%.",
            "Persistent and unknown transitions are reported but excluded from the signed net delta.",
        ],
    }


def _task_cluster_bootstrap_interval(states: list[_ChainTaxonomyState]) -> dict[str, Any]:
    clusters: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for chain, initial, terminal in states:
        if not chain.records:
            continue
        key = (chain.records[0].suite, chain.records[0].task_id)
        clusters[key][0] += 1
        clusters[key][1] += _family_delta(initial, terminal)
    common: dict[str, Any] = {
        "method": "task_cluster_percentile_bootstrap",
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "cluster_unit": "suite_task_id",
        "clusters": len(clusters),
        "seed": BOOTSTRAP_SEED,
    }
    if len(clusters) < 2:
        return {
            **common,
            "available": False,
            "resamples": 0,
            "low_rate": None,
            "high_rate": None,
            "low_percentage_points": None,
            "high_percentage_points": None,
            "reason": "insufficient_task_clusters",
        }
    cluster_values = list(clusters.values())
    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [cluster_values[rng.randrange(len(cluster_values))] for _ in cluster_values]
        sampled_assigned = sum(value[0] for value in sampled)
        sampled_delta = sum(value[1] for value in sampled)
        estimates.append(sampled_delta / sampled_assigned)
    estimates.sort()
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    low = _quantile(estimates, alpha)
    high = _quantile(estimates, 1.0 - alpha)
    return {
        **common,
        "available": True,
        "resamples": BOOTSTRAP_RESAMPLES,
        "low_rate": low,
        "high_rate": high,
        "low_percentage_points": 100.0 * low,
        "high_percentage_points": 100.0 * high,
        "reason": None,
    }


def _family_delta(initial: _TaxonomyState, terminal: _TaxonomyState) -> int:
    if initial is None or terminal is None:
        return 0
    initial_axes = initial[1]
    terminal_axes = terminal[1]
    return sum(
        int(bool(initial_axes.intersection(axes))) - int(bool(terminal_axes.intersection(axes)))
        for axes in AXIS_GROUPS.values()
    )


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


__all__ = ["feedback_lineage_summary"]
