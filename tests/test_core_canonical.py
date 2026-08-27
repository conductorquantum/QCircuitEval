"""Canonical Core suite sweep across all frameworks (58 x 4 = 232)."""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")
OPERATIONAL_FAILURE_STATUSES = frozenset({"execution_error", "resource_limit"})


def _core_task_ids() -> list[str]:
    return sorted(load_tasks("qiskit", suite="core"))


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("task_id", _core_task_ids())
def test_core_canonical_passes_shared_semantic_contract(framework: str, task_id: str) -> None:
    evaluator = build_evaluator(framework, suite="core")
    task = load_tasks(framework, suite="core")[task_id]

    _, details = evaluator.grade_code(
        task_id=task_id,
        code=task["canonical_solution"],
        entry_point=task["entry_point"],
    )

    semantic = details.get("semantic_verification") or {}
    status = str(details.get("semantic_status") or semantic.get("status") or "")
    assert status not in OPERATIONAL_FAILURE_STATUSES, (task_id, framework, status, details.get("reason"))
    assert details["passed"] is True, (
        details.get("reason"),
        status,
        [
            (item.get("reason_code") or item.get("reason"), item.get("value"), item.get("preconditions"))
            for item in semantic.get("evidence") or ()
        ],
    )


def test_core_and_qec_canonical_suite_sizes() -> None:
    """Document the 280-instance full-suite identity (58+12) x 4."""
    assert len(_core_task_ids()) == 58
    assert len(load_tasks("qiskit", suite="qec")) == 12
    assert (58 + 12) * len(FRAMEWORKS) == 280
