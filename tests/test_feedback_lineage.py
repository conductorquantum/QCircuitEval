from __future__ import annotations

from qceval.error_taxonomy import ERROR_AXES
from qceval.models import BenchmarkRecord, ProviderResponse, QCEvalEvaluation, TokenUsage
from qceval.reports import feedback_lineage_summary


def _taxonomy(outcome: str, *axes: str) -> dict[str, object]:
    return {
        "taxonomy_version": "1",
        "multi_label": True,
        "outcome": outcome,
        "axes": list(axes),
        "reason_codes": [],
        "grader_reason_codes": [],
        "unclassified_reason_codes": [],
        "parameter_case_status_counts": {},
    }


def _record(
    task_id: str,
    attempt: int,
    *,
    passed: bool,
    taxonomy: dict[str, object],
    tokens: int,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        framework="qiskit",
        task_id=task_id,
        attempt_index=attempt,
        entry_point="answer",
        category=None,
        provider="stub",
        model="m",
        status="passed" if passed else "failed",
        provider_response=ProviderResponse(
            code=f"code-{task_id}-{attempt}",
            model="m",
            usage=TokenUsage(prompt_tokens=tokens - 1, completion_tokens=1, total_tokens=tokens),
        ),
        evaluation=QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=passed,
            verified_status="verified_pass" if passed else "verified_fail",
        ),
        error_taxonomy=taxonomy,
    )


def test_feedback_lineage_reports_hazards_costs_and_axis_transitions() -> None:
    generation = "generation_execution_reliability"
    behavior = "behavioral_target_mismatch"
    records = [
        _record("01", 0, passed=False, taxonomy=_taxonomy("observed_error", generation), tokens=10),
        _record("01", 1, passed=True, taxonomy=_taxonomy("verified_pass"), tokens=20),
        _record("02", 0, passed=False, taxonomy=_taxonomy("observed_error", behavior), tokens=30),
        _record("02", 1, passed=False, taxonomy=_taxonomy("observed_error", behavior), tokens=40),
        _record("03", 0, passed=True, taxonomy=_taxonomy("verified_pass"), tokens=50),
        _record("04", 0, passed=False, taxonomy=_taxonomy("observed_error", generation), tokens=60),
        _record("04", 1, passed=False, taxonomy=_taxonomy("observed_error", behavior), tokens=70),
    ]

    summary = feedback_lineage_summary(records, max_attempts=2)

    assert summary["assigned_chains"] == 4
    assert summary["complete_chains"] == 4
    assert summary["terminal_pass_rate"] == 0.5
    assert summary["levels"][0]["first_pass_hazard"] == 0.25
    assert summary["levels"][1]["first_pass_hazard"] == 1 / 3
    assert summary["token_usage"]["reported_total_tokens"] == 280
    assert summary["token_usage"]["reported_reasoning_tokens"] == 0
    assert summary["token_usage"]["records_with_reported_reasoning_tokens"] == 0
    assert summary["token_usage"]["mean_reported_total_tokens_per_complete_chain"] == 70
    transitions = summary["taxonomy_transitions"]
    assert transitions["paired_classification_coverage"] == 1.0
    assert transitions["axes"][generation]["cleared_count"] == 2
    assert transitions["axes"][behavior]["persistent_count"] == 1
    assert transitions["axes"][behavior]["surfaced_count"] == 1
    assert set(transitions["axes"]) == set(ERROR_AXES)
    assert transitions["groups"]["execution"]["cleared_count"] == 2
    assert transitions["groups"]["semantic"]["persistent_count"] == 1
    assert transitions["groups"]["semantic"]["surfaced_count"] == 1
    plot = transitions["diverging_plot"]
    assert plot["category_order"] == ["execution", "algorithmic", "semantic"]
    assert plot["categories"]["execution"]["cleared_percentage_points"] == 50.0
    assert plot["categories"]["semantic"]["surfaced_percentage_points"] == -25.0
    assert plot["net_delta"]["percentage_points"] == 25.0
    interval = plot["net_delta"]["confidence_interval_95"]
    assert interval["available"] is True
    assert interval["clusters"] == 4
    assert interval["resamples"] == 10_000
    assert interval["low_percentage_points"] <= 25.0 <= interval["high_percentage_points"]


def test_feedback_lineage_keeps_unclassified_transitions_unknown() -> None:
    unclassified = _taxonomy("observed_error")
    unclassified["unclassified_reason_codes"] = ["new_reason"]
    records = [
        _record("01", 0, passed=False, taxonomy=unclassified, tokens=10),
        _record("01", 1, passed=False, taxonomy=_taxonomy("observed_error", "behavioral_target_mismatch"), tokens=10),
    ]

    summary = feedback_lineage_summary(records, max_attempts=2)

    transitions = summary["taxonomy_transitions"]
    assert transitions["paired_classifiable_chains"] == 0
    assert all(axis["unknown_count"] == 1 for axis in transitions["axes"].values())
    assert all(group["unknown_count"] == 1 for group in transitions["groups"].values())
    interval = transitions["diverging_plot"]["net_delta"]["confidence_interval_95"]
    assert interval == {
        "method": "task_cluster_percentile_bootstrap",
        "confidence_level": 0.95,
        "cluster_unit": "suite_task_id",
        "clusters": 1,
        "seed": 0,
        "available": False,
        "resamples": 0,
        "low_rate": None,
        "high_rate": None,
        "low_percentage_points": None,
        "high_percentage_points": None,
        "reason": "insufficient_task_clusters",
    }


def test_feedback_lineage_error_families_are_unions_of_axes() -> None:
    construction = "construction_resource_fidelity"
    shortcut = "shortcut_provenance_violation"
    records = [
        _record(
            "01",
            0,
            passed=False,
            taxonomy=_taxonomy("observed_error", construction, shortcut),
            tokens=10,
        ),
        _record("01", 1, passed=True, taxonomy=_taxonomy("verified_pass"), tokens=10),
    ]

    summary = feedback_lineage_summary(records, max_attempts=2)

    transitions = summary["taxonomy_transitions"]
    assert transitions["axes"][construction]["cleared_count"] == 1
    assert transitions["axes"][shortcut]["cleared_count"] == 1
    assert transitions["groups"]["algorithmic"]["cleared_count"] == 1
    plot = transitions["diverging_plot"]
    assert plot["categories"]["algorithmic"]["cleared_count"] == 1
    assert plot["net_delta"]["count"] == 1
