"""Aggregate the versioned multi-label benchmark error taxonomy."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from qceval.error_taxonomy import (
    AXIS_GROUPS,
    AXIS_LABELS,
    ERROR_AXES,
    ERROR_TAXONOMY_VERSION,
    TAXONOMY_OUTCOMES,
    classify_error_taxonomy,
)
from qceval.models import BenchmarkRecord


def error_taxonomy_summary(records: list[BenchmarkRecord]) -> dict[str, Any]:
    """Build radar-ready error rates with one common assigned denominator.

    Args:
        records: Completed benchmark records in the plotted stratum.

    Returns:
        Counts, rates, coverage, and framework strata for taxonomy version 1.
    """
    taxonomies = [_record_taxonomy(record) for record in records]
    for taxonomy in taxonomies:
        _validate_taxonomy(taxonomy)
    total = len(records)
    summary = _taxonomy_counts(taxonomies, total)
    by_framework: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_suite: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_suite_framework: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record, taxonomy in zip(records, taxonomies, strict=True):
        by_framework[record.framework].append(taxonomy)
        by_suite[record.suite].append(taxonomy)
        by_suite_framework[record.suite][record.framework].append(taxonomy)
    summary.update(
        {
            "taxonomy_version": ERROR_TAXONOMY_VERSION,
            "multi_label": True,
            "denominator": "all_assigned_records_in_stratum",
            "axis_groups": {name: list(axes) for name, axes in AXIS_GROUPS.items()},
            "axis_labels": dict(AXIS_LABELS),
            "by_framework": {
                framework: _taxonomy_counts(items, len(items)) for framework, items in sorted(by_framework.items())
            },
            "by_suite": {suite: _taxonomy_counts(items, len(items)) for suite, items in sorted(by_suite.items())},
            "by_suite_framework": {
                suite: {
                    framework: _taxonomy_counts(items, len(items)) for framework, items in sorted(frameworks.items())
                }
                for suite, frameworks in sorted(by_suite_framework.items())
            },
        }
    )
    return summary


def _record_taxonomy(record: BenchmarkRecord) -> Mapping[str, Any]:
    if isinstance(record.error_taxonomy, Mapping):
        return record.error_taxonomy
    evaluation = None if record.evaluation is None else record.evaluation.to_dict()
    return classify_error_taxonomy(record.status, evaluation)


def _validate_taxonomy(taxonomy: Mapping[str, Any]) -> None:
    version = taxonomy.get("taxonomy_version")
    if version != ERROR_TAXONOMY_VERSION:
        raise ValueError(
            f"cannot aggregate error taxonomy version {version!r}; stratify records and use a compatible reader"
        )
    if taxonomy.get("multi_label") is not True:
        raise ValueError("error taxonomy must declare multi_label=true")
    if taxonomy.get("outcome") not in TAXONOMY_OUTCOMES:
        raise ValueError("error taxonomy outcome is invalid")
    axes = taxonomy.get("axes")
    if not isinstance(axes, Sequence) or isinstance(axes, str | bytes):
        raise ValueError("error taxonomy axes must be an array")
    if not all(isinstance(axis, str) and axis in ERROR_AXES for axis in axes):
        raise ValueError("error taxonomy contains an unknown axis")
    if len(set(axes)) != len(axes):
        raise ValueError("error taxonomy axes must not contain duplicates")


def _taxonomy_counts(taxonomies: list[Mapping[str, Any]], total: int) -> dict[str, Any]:
    outcomes = Counter(str(item.get("outcome", "ungraded")) for item in taxonomies)
    axis_counts: Counter[str] = Counter()
    unclassified_reasons: Counter[str] = Counter()
    versions = Counter(str(item.get("taxonomy_version", "missing")) for item in taxonomies)
    classified_errors = 0
    for taxonomy in taxonomies:
        axes = _known_axes(taxonomy.get("axes"))
        axis_counts.update(axes)
        unclassified_reasons.update(_strings(taxonomy.get("unclassified_reason_codes")))
        if taxonomy.get("outcome") == "observed_error" and axes:
            classified_errors += 1
    observed_errors = outcomes["observed_error"]
    return {
        "assigned": total,
        "outcome_counts": dict(sorted(outcomes.items())),
        "axis_counts": {axis: axis_counts[axis] for axis in ERROR_AXES},
        "axis_rates": {axis: axis_counts[axis] / total if total else 0.0 for axis in ERROR_AXES},
        "observed_error_records": observed_errors,
        "classified_errors": classified_errors,
        "unclassified_errors": observed_errors - classified_errors,
        "classification_coverage": classified_errors / observed_errors if observed_errors else 0.0,
        "unclassified_reason_counts": dict(sorted(unclassified_reasons.items())),
        "version_counts": dict(sorted(versions.items())),
    }


def _known_axes(value: Any) -> set[str]:
    return set(_strings(value)).intersection(ERROR_AXES)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, str))


__all__ = ["error_taxonomy_summary"]
