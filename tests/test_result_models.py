from __future__ import annotations

import pytest

from qceval.models import BenchmarkRecord, ProviderResponse, QCEvalEvaluation, QCEvalTask


@pytest.mark.parametrize("code", [None, "", " \n\t"])
def test_provider_response_requires_nonempty_code(code: str | None) -> None:
    response = ProviderResponse(code=code)

    assert response.ok is False


def test_provider_metadata_includes_canonical_solution() -> None:
    # Arrange
    task = QCEvalTask(
        task_id="01",
        framework="qiskit",
        prompt="p",
        entry_point="answer",
        category="cat",
        canonical_class={"type": "exact_distribution"},
        raw={"canonical_solution": "code"},
    )

    # Act
    metadata = task.provider_metadata()

    # Assert
    assert metadata["canonical_solution"] == "code"


def test_benchmark_record_serializes_evaluation() -> None:
    # Arrange
    evaluation = QCEvalEvaluation(compiled=True, ran=True, passed=True, metric=0.0, probabilities=[1.0])
    record = BenchmarkRecord(
        framework="qiskit",
        task_id="01",
        entry_point="answer",
        category="cat",
        provider="smoke",
        model="m",
        status="passed",
        provider_response=ProviderResponse(code="code", model="m"),
        evaluation=evaluation,
    )

    # Act
    payload = record.to_dict()

    # Assert
    assert payload["status"] == "passed"
    assert payload["suite"] == "core"
    assert payload["evaluation"]["probabilities"] == [1.0]


def test_evaluation_serializes_metric_name() -> None:
    # Arrange
    evaluation = QCEvalEvaluation(
        compiled=True,
        ran=True,
        passed=True,
        metric=0.001,
        metric_name="hellinger_infidelity",
    )

    # Act
    payload = evaluation.to_dict()

    # Assert
    assert payload["metric"] == 0.001
    assert payload["metric_name"] == "hellinger_infidelity"
