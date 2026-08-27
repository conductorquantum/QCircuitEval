from __future__ import annotations

import pytest

from qceval.core.bench import Adaptor
from qceval.evals.evaluator import build_evaluator, load_tasks


def test_load_tasks_loads_core_by_default() -> None:
    # Arrange / Act
    tasks = load_tasks("qiskit")

    # Assert
    assert "01" in tasks
    assert all(not task_id.startswith("qec") for task_id in tasks)


def test_load_tasks_loads_qec_suite() -> None:
    # Arrange / Act
    tasks = load_tasks("qiskit", suite="qec")

    # Assert
    assert next(iter(tasks)) == "qec01"
    assert all(task["category"] == "QEC" for task in tasks.values())
    assert all(task["canonical_class"]["type"] == "case_table" for task in tasks.values())


def test_adapter_loads_qec_tasks_with_suite_metadata() -> None:
    # Arrange
    adapter = Adaptor()

    # Act
    task = adapter.load_tasks("qiskit", suite="qec")[0]

    # Assert
    assert task.suite == "qec"


def test_semantic_contract_grades_qiskit_bit_flip_correct() -> None:
    # Arrange
    evaluator = build_evaluator("qiskit", suite="qec")
    task = load_tasks("qiskit", suite="qec")["qec03"]

    # Act
    _, details = evaluator.grade_code(task_id="qec03", code=task["canonical_solution"], entry_point=task["entry_point"])

    # Assert
    assert details["passed"] is True, (
        details["reason"],
        [
            (item["reason_code"], item["value"], item["preconditions"])
            for item in details["semantic_verification"]["evidence"]
        ],
    )
    assert details["grader_type"] == "semantic_contract"
    assert details["num_cases"] == 8


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane", "cudaq"])
@pytest.mark.parametrize("task_id", [f"qec{index:02d}" for index in range(1, 13)])
def test_qec_canonical_passes_shared_semantic_contract(framework: str, task_id: str) -> None:
    # Arrange
    evaluator = build_evaluator(framework, suite="qec")
    tasks = load_tasks(framework, suite="qec")
    task = tasks[task_id]

    # Act
    _, details = evaluator.grade_code(
        task_id=task_id,
        code=task["canonical_solution"],
        entry_point=task["entry_point"],
    )

    # Assert
    assert details["semantic_status"] == "verified_pass", (task_id, framework, details["semantic_status"])
    assert details["passed"] is True, (
        details["reason"],
        [
            (item["reason_code"], item["value"], item["preconditions"])
            for item in details["semantic_verification"]["evidence"]
        ],
    )


@pytest.mark.parametrize("framework", ["cirq", "pennylane", "cudaq"])
def test_qec_framework_matches_qiskit_cases_and_metadata(framework: str) -> None:
    # Arrange
    qiskit_tasks = load_tasks("qiskit", suite="qec")
    framework_tasks = load_tasks(framework, suite="qec")

    # Assert
    assert set(framework_tasks) == set(qiskit_tasks)
    for task_id, qiskit_task in qiskit_tasks.items():
        framework_task = framework_tasks[task_id]
        metadata_checks = dict(framework_task["canonical_class"]["metadata_checks"])
        if framework == "cudaq":
            assert metadata_checks.pop("forbid_returned_unitary") is True
        assert metadata_checks == qiskit_task["canonical_class"]["metadata_checks"]
        assert framework_task["canonical_class"]["cases"] == qiskit_task["canonical_class"]["cases"]
