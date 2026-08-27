from __future__ import annotations

import math
from dataclasses import replace

import pytest

from qceval.core.runner.records import _status, _timeout_evaluation
from qceval.models import BenchmarkRecord, ProviderResponse, QCEvalEvaluation, RunConfig, TokenUsage
from qceval.reports import format_run_summary, pass_at_k, summarize


def _record(
    status: str,
    passed: bool = False,
    *,
    task_id: str = "01",
    cost_usd: float | None = None,
) -> BenchmarkRecord:
    evaluation = None
    if status not in {"generated", "provider_failed"}:
        evaluation = QCEvalEvaluation(compiled=True, ran=True, passed=passed)
    return BenchmarkRecord(
        framework="qiskit",
        task_id=task_id,
        entry_point="answer",
        category=None,
        provider="smoke",
        model="m",
        status=status,  # type: ignore[arg-type]
        provider_response=ProviderResponse(
            code="code",
            model="m",
            usage=None if cost_usd is None else TokenUsage(cost_usd=cost_usd),
        ),
        evaluation=evaluation,
    )


def _originless_execution_error_record(
    *,
    task_id: str = "01",
    sample_index: int = 0,
    attempt_index: int = 0,
) -> BenchmarkRecord:
    evaluation = QCEvalEvaluation(
        compiled=True,
        ran=True,
        passed=False,
        semantic_result={"status": "execution_error"},
    )
    return BenchmarkRecord(
        framework="qiskit",
        task_id=task_id,
        sample_index=sample_index,
        attempt_index=attempt_index,
        entry_point="answer",
        category=None,
        provider="smoke",
        model="m",
        status=_status(evaluation),
        provider_response=ProviderResponse(code="unsupported", model="m"),
        evaluation=evaluation,
    )


def test_originless_resource_limit_is_candidate_failure() -> None:
    evaluation = QCEvalEvaluation(
        compiled=True,
        ran=True,
        passed=False,
        semantic_result={"status": "resource_limit"},
    )

    assert _status(evaluation) == "failed"


def test_evaluation_timeout_is_a_candidate_resource_limit() -> None:
    evaluation = _timeout_evaluation(180)
    record = BenchmarkRecord(
        framework="cudaq",
        task_id="04",
        entry_point="qaoa_maxcut_ansatz",
        category=None,
        provider="openrouter",
        model="m",
        status=_status(evaluation),
        provider_response=ProviderResponse(code="candidate", model="m"),
        evaluation=evaluation,
    )

    assert record.status == "failed"
    assert evaluation.compiled is True
    assert evaluation.ran is False
    assert evaluation.verified_status == "resource_limit"
    assert evaluation.error_type == "EvaluationTimeout"
    assert record.to_dict()["error_taxonomy"]["outcome"] == "resource_limit"


@pytest.mark.parametrize(
    "evaluation",
    [
        QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=False,
            semantic_result={
                "status": "execution_error",
                "diagnostics": [{"name": "failure_origin", "value": "grader_verification"}],
            },
        ),
        QCEvalEvaluation(
            compiled=False,
            ran=False,
            passed=False,
            error_type="InfrastructureError",
        ),
    ],
)
def test_explicit_grader_faults_remain_infrastructure(evaluation: QCEvalEvaluation) -> None:
    assert _status(evaluation) == "infrastructure_error"


def test_summary_counts_statuses() -> None:
    # Arrange
    records = [_record("passed", True), _record("failed"), _record("provider_failed")]

    # Act
    summary = summarize(records)

    # Assert
    assert summary["total_tasks"] == 3
    assert summary["passed"] == 1
    assert summary["provider_failures"] == 1
    assert summary["pass_rate"] == 1 / 3
    assert summary["by_suite"]["core"]["total_tasks"] == 3


def test_summary_reports_average_provider_cost_per_logical_task() -> None:
    records = [
        _record("passed", True, task_id="01", cost_usd=0.1),
        _record("failed", task_id="02", cost_usd=0.3),
    ]

    summary = summarize(records)

    assert summary["cost"]["reported_total_cost_usd"] == pytest.approx(0.4)
    assert summary["cost"]["record_cost_coverage"] == 1.0
    assert summary["cost"]["task_cost_coverage"] == 1.0
    assert summary["cost"]["mean_reported_cost_per_task_usd"] == pytest.approx(0.2)
    assert summary["by_framework"]["qiskit"]["cost"]["mean_reported_cost_per_task_usd"] == pytest.approx(0.2)
    assert "average/logical-task=$0.200000" in format_run_summary(summary)


def test_summary_withholds_average_cost_when_any_record_is_missing_cost() -> None:
    summary = summarize(
        [
            _record("passed", True, task_id="01", cost_usd=0.1),
            _record("failed", task_id="02"),
        ]
    )

    assert summary["cost"]["reported_total_cost_usd"] == pytest.approx(0.1)
    assert summary["cost"]["records_with_reported_cost"] == 1
    assert summary["cost"]["mean_reported_cost_per_task_usd"] is None


def test_summary_reads_legacy_openrouter_cost_from_raw_response() -> None:
    record = _record("passed", True)
    record = replace(
        record,
        provider_response=ProviderResponse(
            code="code",
            model="m",
            raw_response={"usage": {"cost": 0.125}},
        ),
    )

    summary = summarize([record])

    assert summary["cost"]["mean_reported_cost_per_task_usd"] == pytest.approx(0.125)


def test_summary_separates_generated_records_from_failures() -> None:
    summary = summarize([_record("generated")])

    assert summary["generated"] == 1
    assert summary["assigned_tasks"] == 0
    assert summary["failed"] == 0
    assert summary["verified_status_counts"] == {"ungraded": 1}


def test_summary_headline_rate_includes_assigned_infrastructure_failure() -> None:
    infrastructure = BenchmarkRecord(
        framework="qiskit",
        task_id="02",
        entry_point="answer",
        category=None,
        provider="smoke",
        model="m",
        status="infrastructure_error",
        provider_response=ProviderResponse(code="", model="m"),
        evaluation=QCEvalEvaluation(
            compiled=False,
            ran=False,
            passed=False,
            error_type="InfrastructureError",
        ),
    )

    summary = summarize([_record("passed", True), infrastructure])

    assert summary["assigned_tasks"] == 2
    assert summary["scoreable_tasks"] == 1
    assert summary["rerun_required_tasks"] == 1
    assert summary["pass_rate_denominator"] == "assigned_tasks"
    assert summary["pass_rate"] == 0.5
    assert summary["by_framework"]["qiskit"]["pass_rate"] == 0.5


def test_task_summary_includes_metric_name() -> None:
    # Arrange
    record = BenchmarkRecord(
        framework="qiskit",
        task_id="01",
        entry_point="answer",
        category=None,
        provider="smoke",
        model=None,
        status="passed",
        provider_response=ProviderResponse(code="code"),
        evaluation=QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=True,
            metric=0.0,
            metric_name="hellinger_infidelity",
        ),
    )

    # Act
    summary = summarize([record])

    # Assert
    assert summary["tasks"][0]["metric_name"] == "hellinger_infidelity"
    assert summary["tasks"][0]["suite"] == "core"


def test_pass_at_k_estimator_edges() -> None:
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 8, 5) == 1.0
    assert pass_at_k(10, 3, 5) == pytest.approx(1 - math.comb(7, 5) / math.comb(10, 5))


def test_summary_counts_verified_pass_only() -> None:
    records = [
        _record("passed", True),
        BenchmarkRecord(
            framework="qiskit",
            task_id="02",
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="passed",
            provider_response=ProviderResponse(code="code", model="m"),
            evaluation=QCEvalEvaluation(
                compiled=True,
                ran=True,
                passed=True,
                verified_status="unverified_pass",
            ),
        ),
    ]

    summary = summarize(records)

    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["verified_status_counts"] == {"verified_pass": 1, "unverified_pass": 1}


def test_summary_pass_at_k_groups_samples() -> None:
    # Arrange
    records = [
        _record("passed", True),
        BenchmarkRecord(
            framework="qiskit",
            task_id="01",
            sample_index=1,
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="failed",
            provider_response=ProviderResponse(code="code", model="m"),
            evaluation=QCEvalEvaluation(compiled=True, ran=True, passed=False),
        ),
    ]
    config = RunConfig(
        provider="smoke",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        samples_per_task=2,
        pass_k=2,
    )

    # Act
    summary = summarize(records, run_config=config)

    # Assert
    assert summary["pass_at_k"]["tasks"][0]["n"] == 2
    assert summary["pass_at_k"]["tasks"][0]["c"] == 1
    assert summary["pass_at_k"]["pass_at_k"] == 1.0


def test_summary_excludes_infrastructure_from_pass_at_k_denominator() -> None:
    records = [
        _record("passed", True),
        BenchmarkRecord(
            framework="qiskit",
            task_id="01",
            sample_index=1,
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="failed",
            provider_response=ProviderResponse(code="wrong", model="m"),
            evaluation=QCEvalEvaluation(compiled=True, ran=True, passed=False),
        ),
        BenchmarkRecord(
            framework="qiskit",
            task_id="01",
            sample_index=2,
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="infrastructure_error",
            provider_response=ProviderResponse(code="", model="m"),
            evaluation=QCEvalEvaluation(
                compiled=False,
                ran=False,
                passed=False,
                error_type="InfrastructureError",
            ),
        ),
    ]
    config = RunConfig(
        provider="smoke",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        samples_per_task=3,
        pass_k=2,
    )

    summary = summarize(records, run_config=config)["pass_at_k"]

    assert summary["tasks_evaluated"] == 0
    assert summary["tasks_requiring_rerun"] == 1
    assert summary["tasks"][0]["n"] == 2
    assert summary["tasks"][0]["infrastructure_samples"] == 1
    assert summary["tasks"][0]["estimate"] is None


def test_pass_at_k_counts_originless_candidate_execution_error_as_failure() -> None:
    records = [
        _record("passed", True),
        _originless_execution_error_record(sample_index=1),
    ]
    config = RunConfig(
        provider="smoke",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        samples_per_task=2,
        pass_k=1,
    )

    summary = summarize(records, run_config=config)["pass_at_k"]

    assert records[1].status == "failed"
    assert summary["tasks_requiring_rerun"] == 0
    assert summary["tasks"][0]["n"] == 2
    assert summary["tasks"][0]["c"] == 1
    assert summary["pass_at_k"] == 0.5


def test_format_run_summary_includes_overall_and_scope_rows() -> None:
    # Arrange
    records = [
        _record("passed", True),
        BenchmarkRecord(
            framework="cirq",
            task_id="01",
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="compile_failed",
            provider_response=ProviderResponse(code="code", model="m"),
            evaluation=QCEvalEvaluation(compiled=False, ran=False, passed=False),
        ),
    ]
    summary = summarize(records)

    # Act
    text = format_run_summary(summary)

    # Assert
    assert "Benchmark summary" in text
    assert "| scope" in text
    assert "overall" in text
    assert "core/qiskit" in text
    assert "core/cirq" in text
    assert "50.0%" in text


def test_feedback_cumulative_summary() -> None:
    # Arrange
    records = [
        BenchmarkRecord(
            framework="qiskit",
            task_id="01",
            attempt_index=0,
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="failed",
            provider_response=ProviderResponse(code="bad", model="m"),
            evaluation=QCEvalEvaluation(compiled=True, ran=True, passed=False),
        ),
        BenchmarkRecord(
            framework="qiskit",
            task_id="01",
            attempt_index=1,
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="passed",
            provider_response=ProviderResponse(code="good", model="m"),
            evaluation=QCEvalEvaluation(compiled=True, ran=True, passed=True),
        ),
        BenchmarkRecord(
            framework="qiskit",
            task_id="02",
            attempt_index=0,
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="compile_failed",
            provider_response=ProviderResponse(code="bad", model="m"),
            evaluation=QCEvalEvaluation(compiled=False, ran=False, passed=False),
        ),
    ]
    config = RunConfig(provider="smoke", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=2)

    # Act
    summary = summarize(records, run_config=config)

    # Assert
    assert summary["feedback"]["levels"][0]["attempted"] == 2
    assert summary["feedback"]["levels"][0]["first_compiled"] == 1
    assert summary["feedback"]["levels"][0]["first_passed"] == 0
    assert summary["feedback"]["levels"][1]["attempted"] == 1
    assert summary["feedback"]["levels"][1]["first_passed"] == 1
    assert summary["feedback"]["final_pass_rate"] == 0.5
    protocol = summary["run_protocol"]
    assert protocol["max_repairs"] == 1
    assert protocol["feedback_policy"]["version"] == "feedback.execution_trace.v1"
    assert protocol["generation_parameters"]["seed"]["source"] == "not_supported"
    text = format_run_summary(summary)
    assert "Benchmark attempt-record summary" in text
    assert "Feedback lineage: terminal_pass_rate=50.0%" in text


def test_feedback_summary_excludes_infrastructure_chains() -> None:
    records = [
        _record("passed", True),
        BenchmarkRecord(
            framework="qiskit",
            task_id="02",
            entry_point="answer",
            category=None,
            provider="smoke",
            model="m",
            status="infrastructure_error",
            provider_response=ProviderResponse(code="", model="m"),
            evaluation=QCEvalEvaluation(
                compiled=False,
                ran=False,
                passed=False,
                error_type="InfrastructureError",
            ),
        ),
    ]
    config = RunConfig(
        provider="smoke",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        max_attempts=2,
    )

    summary = summarize(records, run_config=config)

    assert summary["feedback"]["final_pass_rate"] == 1.0
    assert summary["feedback"]["infrastructure_chains"] == 1
    assert summary["feedback_lineage"]["terminal_pass_rate"] == 1.0
    assert summary["feedback_lineage"]["rerun_required"] == 1


def test_feedback_counts_originless_candidate_execution_error_as_failure() -> None:
    records = [
        _record("passed", True),
        _originless_execution_error_record(task_id="02", attempt_index=0),
        _originless_execution_error_record(task_id="02", attempt_index=1),
    ]
    config = RunConfig(
        provider="smoke",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        max_attempts=2,
    )

    summary = summarize(records, run_config=config)

    assert summary["feedback"]["infrastructure_chains"] == 0
    assert summary["feedback"]["rerun_required"] == 0
    assert summary["feedback"]["final_pass_rate"] == 0.5
    assert summary["feedback_lineage"]["terminal_pass_rate"] == 0.5


def test_openrouter_protocol_records_reasoning_effort() -> None:
    config = RunConfig(
        provider="openrouter",
        frameworks=("qiskit",),
        source_hint=None,
        model="z-ai/glm-5.2",
        provider_config={"temperature": 0.2, "reasoning_effort": "xhigh"},
    )

    protocol = summarize([], run_config=config)["run_protocol"]

    assert protocol["generation_parameters"]["reasoning_effort"] == {"value": "xhigh", "source": "explicit"}
    assert protocol["generation_parameters"]["reasoning_enabled"] == {
        "value": None,
        "source": "not_applicable",
    }


def test_openrouter_protocol_records_output_limit_source() -> None:
    config = RunConfig(
        provider="openrouter",
        frameworks=("qiskit",),
        source_hint=None,
        model="google/gemma-4-31b-it",
        provider_config={
            "openrouter_endpoint_tag": "provider/fp16",
            "openrouter_max_output_tokens": 128000,
            "openrouter_output_limit_source": "benchmark_floor",
            "openrouter_endpoint_cap_status": "undisclosed_first_party_exception",
            "openrouter_output_token_parameter": "max_tokens",
            "openrouter_route_revision": "route-01",
        },
    )

    parameters = summarize([], run_config=config)["run_protocol"]["generation_parameters"]

    assert parameters["max_output_tokens"] == {"value": 128000, "source": "explicit"}
    assert parameters["output_limit_source"] == {"value": "benchmark_floor", "source": "explicit"}
    assert parameters["endpoint_cap_status"] == {
        "value": "undisclosed_first_party_exception",
        "source": "explicit",
    }


def test_openrouter_protocol_records_reasoning_enabled() -> None:
    config = RunConfig(
        provider="openrouter",
        frameworks=("qiskit",),
        source_hint=None,
        model="poolside/laguna-xs-2.1",
        provider_config={"reasoning_enabled": True},
    )

    protocol = summarize([], run_config=config)["run_protocol"]

    assert protocol["generation_parameters"]["reasoning_effort"] == {
        "value": None,
        "source": "not_applicable",
    }
    assert protocol["generation_parameters"]["reasoning_enabled"] == {"value": True, "source": "explicit"}


def test_openrouter_protocol_records_model_default_reasoning_when_unspecified() -> None:
    config = RunConfig(
        provider="openrouter",
        frameworks=("qiskit",),
        source_hint=None,
        model="model-with-provider-defaults",
    )

    parameters = summarize([], run_config=config)["run_protocol"]["generation_parameters"]

    assert parameters["reasoning_effort"] == {"value": None, "source": "model_default"}
    assert parameters["reasoning_enabled"] == {"value": None, "source": "model_default"}
