from __future__ import annotations

from qceval.core.feedback import build_feedback, feedback_stop_reason
from qceval.models import BenchmarkRecord, ProviderResponse, QCEvalEvaluation


def _record(status: str, evaluation: QCEvalEvaluation | None) -> BenchmarkRecord:
    return BenchmarkRecord(
        framework="qiskit",
        task_id="01",
        entry_point="answer",
        category=None,
        provider="stub",
        model="m",
        status=status,  # type: ignore[arg-type]
        provider_response=ProviderResponse(code="def answer():\n    pass\n", model="m", error="provider boom"),
        evaluation=evaluation,
    )


def test_build_feedback_compile_failure() -> None:
    # Arrange
    record = _record(
        "compile_failed",
        QCEvalEvaluation(compiled=False, ran=False, passed=False, error="SyntaxError: bad", error_type="SyntaxError"),
    )

    # Act
    feedback = build_feedback(record, max_chars=2000)

    # Assert
    assert feedback.reason == "compile_failed"
    assert "Previous code did not compile." in feedback.message_to_model
    assert "SyntaxError: bad" in feedback.message_to_model
    assert "canonical" not in feedback.message_to_model.lower()


def test_build_feedback_runtime_failure() -> None:
    # Arrange
    record = _record(
        "run_failed",
        QCEvalEvaluation(compiled=True, ran=False, passed=False, error="RuntimeError: boom", error_type="RuntimeError"),
    )

    # Act
    feedback = build_feedback(record, max_chars=2000)

    # Assert
    assert feedback.reason == "run_failed"
    assert "Previous code compiled but failed at runtime." in feedback.message_to_model
    assert "RuntimeError: boom" in feedback.message_to_model


def test_build_feedback_semantic_failure() -> None:
    # Arrange
    record = _record(
        "failed",
        QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=False,
            metric=0.5,
            metric_name="hellinger_infidelity",
            probabilities=[0.2, 0.8],
            execution_metadata={"num_qubits": 1, "measurement_count": 1, "operation_counts": {"h": 1}},
            grader_details={
                "passed": False,
                "metric": 0.5,
                "metric_name": "hellinger_infidelity",
                "threshold": 0.01,
                "canonical_solution": "secret",
                "canonical_probabilities": [1.0, 0.0],
                "expected_distribution": [1.0, 0.0],
                "case_results": [{"expected": "secret"}],
            },
        ),
    )

    # Act
    feedback = build_feedback(record, max_chars=2000)

    # Assert
    assert feedback.reason == "failed"
    assert "output_vector" in feedback.message_to_model
    assert "hellinger_infidelity" not in feedback.message_to_model
    assert "0.01" not in feedback.message_to_model
    assert "num_qubits" in feedback.message_to_model
    assert "canonical_solution" not in feedback.message_to_model
    assert "canonical_probabilities" not in feedback.message_to_model
    assert "expected_distribution" not in feedback.message_to_model
    assert "case_results" not in feedback.message_to_model
    assert "taxonomy" not in feedback.message_to_model


def test_build_feedback_truncation() -> None:
    # Arrange
    record = _record(
        "compile_failed",
        QCEvalEvaluation(compiled=False, ran=False, passed=False, error="x" * 100, error_type="SyntaxError"),
    )

    # Act
    feedback = build_feedback(record, max_chars=10)

    # Assert
    assert feedback.truncated is True
    assert "xxxxxxx..." in feedback.message_to_model


def test_feedback_stop_reason_separates_candidate_and_grader_execution_errors() -> None:
    candidate = _record(
        "failed",
        QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=False,
            semantic_result={
                "status": "execution_error",
                "diagnostics": [{"name": "failure_origin", "value": "candidate_execution"}],
            },
        ),
    )
    grader = _record(
        "failed",
        QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=False,
            semantic_result={
                "status": "execution_error",
                "diagnostics": [{"name": "failure_origin", "value": "grader_verification"}],
            },
        ),
    )

    assert feedback_stop_reason(candidate) is None
    assert feedback_stop_reason(grader) == "grader_nondecision"


def test_feedback_treats_originless_execution_error_as_candidate_failure() -> None:
    record = _record(
        "failed",
        QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=False,
            semantic_result={"status": "execution_error"},
        ),
    )

    assert feedback_stop_reason(record) is None


def test_feedback_stops_on_resource_limit() -> None:
    record = _record(
        "failed",
        QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=False,
            semantic_result={"status": "resource_limit"},
        ),
    )

    assert feedback_stop_reason(record) == "resource_limit"


def test_feedback_does_not_treat_explicit_unverified_pass_as_verified() -> None:
    record = _record(
        "passed",
        QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=True,
            verified_status="unverified_pass",
        ),
    )

    assert feedback_stop_reason(record) == "grader_nondecision"
