"""Synthetic validation of the scientific reporting rules."""

from __future__ import annotations

from qceval.models import (
    BenchmarkRecord,
    Framework,
    OutcomeStatus,
    ProviderResponse,
    QCEvalEvaluation,
    RunConfig,
    Suite,
)
from qceval.reports import feedback_lineage_summary, summarize


def _one_shot_record(
    status: OutcomeStatus,
    *,
    suite: Suite,
    framework: Framework,
    task_id: str,
    incomplete: bool = False,
) -> BenchmarkRecord:
    evaluation: QCEvalEvaluation | None
    if incomplete or status == "provider_failed":
        evaluation = None
    elif status == "compile_failed":
        evaluation = QCEvalEvaluation(compiled=False, ran=False, passed=False)
    elif status == "run_failed":
        evaluation = QCEvalEvaluation(compiled=True, ran=False, passed=False)
    elif status == "infrastructure_error":
        evaluation = QCEvalEvaluation(
            compiled=False,
            ran=False,
            passed=False,
            error_type="InfrastructureError",
        )
    else:
        evaluation = QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=status == "passed",
        )
    return BenchmarkRecord(
        framework=framework,
        suite=suite,
        task_id=task_id,
        entry_point="answer",
        category=None,
        provider="synthetic",
        model="m",
        status=status,
        provider_response=ProviderResponse(code="" if status == "provider_failed" else "code", model="m"),
        evaluation=evaluation,
    )


def test_synthetic_pass_at_1_uses_strict_denominators_and_separate_strata() -> None:
    records = [
        _one_shot_record("passed", suite="core", framework="qiskit", task_id="01"),
        _one_shot_record("provider_failed", suite="core", framework="cirq", task_id="01"),
        _one_shot_record("compile_failed", suite="core", framework="pennylane", task_id="01"),
        _one_shot_record("run_failed", suite="core", framework="cudaq", task_id="01"),
        _one_shot_record("failed", suite="qec", framework="qiskit", task_id="01"),
        _one_shot_record("infrastructure_error", suite="qec", framework="cirq", task_id="01"),
        _one_shot_record("passed", suite="qec", framework="pennylane", task_id="01"),
        _one_shot_record("failed", suite="qec", framework="cudaq", task_id="01", incomplete=True),
        _one_shot_record("passed", suite="qec", framework="qiskit", task_id="02"),
    ]
    config = RunConfig(
        provider="synthetic",
        frameworks=("qiskit", "cirq", "pennylane", "cudaq"),
        source_hint=None,
        model="m",
        suites=("core", "qec"),
    )

    summary = summarize(records, run_config=config)

    assert summary["run_protocol"]["samples_per_task"] == 1
    assert summary["run_protocol"]["pass_k"] == 1
    assert summary["pass_rate_denominator"] == "assigned_tasks"
    assert summary["assigned_tasks"] == len(records)
    assert summary["passed"] == 3
    assert summary["pass_rate"] == 3 / 9
    assert summary["scoreable_tasks"] == 8
    assert summary["rerun_required_tasks"] == 1
    assert summary["provider_failures"] == 1
    assert summary["compile_failures"] == 1
    assert summary["run_failures"] == 1
    assert summary["infrastructure_failures"] == 1
    assert summary["failed"] == 2

    assert set(summary["by_suite"]) == {"core", "qec"}
    assert set(summary["by_framework"]) == {"qiskit", "cirq", "pennylane", "cudaq"}
    assert summary["by_suite"]["core"]["pass_rate"] == 1 / 4
    assert summary["by_suite"]["qec"]["pass_rate"] == 2 / 5
    assert summary["by_suite_framework"]["qec"]["qiskit"]["pass_rate"] == 1 / 2
    assert summary["by_framework"]["qiskit"]["pass_rate"] == 2 / 3

    incomplete = next(task for task in summary["tasks"] if task["suite"] == "qec" and task["framework"] == "cudaq")
    assert incomplete["passed"] is None
    assert incomplete["verified_status"] == "verified_fail"


def test_synthetic_missing_output_and_incomplete_evaluation_cannot_improve_rate() -> None:
    records = [
        _one_shot_record("passed", suite="core", framework="qiskit", task_id="01"),
        _one_shot_record("provider_failed", suite="core", framework="qiskit", task_id="02"),
        _one_shot_record("failed", suite="core", framework="qiskit", task_id="03", incomplete=True),
    ]

    summary = summarize(records)

    assert summary["assigned_tasks"] == 3
    assert summary["passed"] == 1
    assert summary["provider_failures"] == 1
    assert summary["failed"] == 1
    assert summary["pass_rate"] == 1 / 3


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


def _feedback_record(
    *,
    suite: Suite,
    framework: Framework,
    task_id: str,
    attempt_index: int,
    passed: bool,
    axes: tuple[str, ...] = (),
) -> BenchmarkRecord:
    return BenchmarkRecord(
        framework=framework,
        suite=suite,
        task_id=task_id,
        attempt_index=attempt_index,
        entry_point="answer",
        category=None,
        provider="synthetic",
        model="m",
        status="passed" if passed else "failed",
        provider_response=ProviderResponse(code=f"{suite}-{framework}-{task_id}-{attempt_index}", model="m"),
        evaluation=QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=passed,
            verified_status="verified_pass" if passed else "verified_fail",
        ),
        error_taxonomy=_taxonomy("verified_pass" if passed else "observed_error", *axes),
    )


def test_synthetic_bootstrap_is_deterministic_and_clusters_suite_tasks() -> None:
    records = [
        _feedback_record(
            suite="core",
            framework="qiskit",
            task_id="01",
            attempt_index=0,
            passed=False,
            axes=("generation_execution_reliability",),
        ),
        _feedback_record(
            suite="core",
            framework="qiskit",
            task_id="01",
            attempt_index=1,
            passed=True,
        ),
        _feedback_record(
            suite="core",
            framework="cirq",
            task_id="01",
            attempt_index=0,
            passed=False,
            axes=("construction_resource_fidelity",),
        ),
        _feedback_record(
            suite="core",
            framework="cirq",
            task_id="01",
            attempt_index=1,
            passed=True,
        ),
        _feedback_record(
            suite="qec",
            framework="qiskit",
            task_id="01",
            attempt_index=0,
            passed=False,
            axes=("behavioral_target_mismatch",),
        ),
        _feedback_record(
            suite="qec",
            framework="qiskit",
            task_id="01",
            attempt_index=1,
            passed=True,
        ),
        _feedback_record(
            suite="qec",
            framework="cirq",
            task_id="01",
            attempt_index=0,
            passed=True,
        ),
    ]

    first = feedback_lineage_summary(records, max_attempts=2)
    second = feedback_lineage_summary(records, max_attempts=2)
    first_interval = first["taxonomy_transitions"]["diverging_plot"]["net_delta"]["confidence_interval_95"]
    second_interval = second["taxonomy_transitions"]["diverging_plot"]["net_delta"]["confidence_interval_95"]

    assert first["assigned_chains"] == 4
    assert first_interval == second_interval
    assert first_interval["method"] == "task_cluster_percentile_bootstrap"
    assert first_interval["cluster_unit"] == "suite_task_id"
    # The two frameworks for each task stay together, while the same task ID
    # in Core and QEC remains two distinct task clusters.
    assert first_interval["clusters"] == 2
    assert first_interval["seed"] == 0
    assert first_interval["resamples"] == 10_000
    assert first_interval["available"] is True
