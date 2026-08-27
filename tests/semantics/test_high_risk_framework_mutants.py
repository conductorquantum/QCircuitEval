"""Task-specific regressions for Phase 2.5 framework-boundary risks."""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator
from qceval.evals.tasks import load_tasks
from qceval.models import Framework

FRAMEWORKS: tuple[Framework, ...] = ("qiskit", "cirq", "pennylane", "cudaq")

_WIRE_ORDER_MUTATIONS = {
    "qiskit": ("    qc.cx(2, 1)\n", "    qc.cx(0, 1)\n"),
    "cirq": (
        "    circuit.append(cirq.CNOT(q[2], q[1]))\n",
        "    circuit.append(cirq.CNOT(q[0], q[1]))\n",
    ),
    "pennylane": (
        "        qml.CNOT(wires=[2, 1])\n",
        "        qml.CNOT(wires=[0, 1])\n",
    ),
    "cudaq": ("        x.ctrl(q[2], q[1])\n", "        x.ctrl(q[0], q[1])\n"),
}

_TERMINAL_OBSERVATION_ALTERNATES = {
    "qiskit": (
        "    qc.measure([0, 1, 2], [0, 1, 2])",
        "    qc.measure([2, 1, 0], [2, 1, 0])",
    ),
    "cirq": (
        '    circuit.append(cirq.measure(q[2], q[1], q[0], key="result"))',
        '    circuit.append(cirq.measure(q[0], key="result0"))\n'
        '    circuit.append(cirq.measure(q[1], key="result1"))\n'
        '    circuit.append(cirq.measure(q[2], key="result2"))',
    ),
    "pennylane": (
        "        return qml.probs(wires=[2, 1, 0])",
        "        return qml.probs(wires=[0, 1, 2])",
    ),
    "cudaq": (
        "        mz(q[0])\n        mz(q[1])\n        mz(q[2])",
        "        mz(q[2])\n        mz(q[1])\n        mz(q[0])",
    ),
}

_TERMINAL_OBSERVATION_MUTATIONS = {
    "qiskit": (
        "    qc.measure([0, 1, 2], [0, 1, 2])",
        "    qc.measure([2, 1, 0], [0, 1, 2])",
    ),
    "cirq": (
        '    circuit.append(cirq.measure(q[2], q[1], q[0], key="result"))',
        '    circuit.append(cirq.measure(q[0], q[1], q[2], key="result"))',
    ),
    "pennylane": (
        "        return qml.probs(wires=[2, 1, 0])",
        "        return qml.probs(wires=[2, 1])",
    ),
    "cudaq": (
        "        mz(q[0])\n        mz(q[1])\n        mz(q[2])",
        "        mz(q[0])\n        mz(q[1])\n        mz(q[1])",
    ),
}


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, f"mutation anchor changed: {old!r}"
    return source.replace(old, new, 1)


def _grade(framework: Framework, task_id: str, code: str) -> dict[str, object]:
    task = load_tasks(framework, "core")[task_id]
    _, details = build_evaluator(framework, "core").grade_code(
        task_id=task_id,
        code=code,
        entry_point=task["entry_point"],
    )
    return details


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_task21_asymmetric_wire_order_mutant_is_rejected(framework: Framework) -> None:
    """The complete MAJ truth table distinguishes every declared input wire."""
    task = load_tasks(framework, "core")["21"]
    candidate = _replace_once(task["canonical_solution"], *_WIRE_ORDER_MUTATIONS[framework])

    details = _grade(framework, "21", candidate)

    assert details["passed"] is False, (framework, details.get("reason"))
    assert details["semantic_status"] == "semantic_fail", (framework, details.get("reason"))


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_task21_equivalent_terminal_observation_order_is_accepted(framework: Framework) -> None:
    """Reordering observation syntax while preserving the register must pass."""
    task = load_tasks(framework, "core")["21"]
    candidate = _replace_once(task["canonical_solution"], *_TERMINAL_OBSERVATION_ALTERNATES[framework])

    details = _grade(framework, "21", candidate)

    assert details["passed"] is True, (framework, details.get("reason"))
    assert details["semantic_status"] == "verified_pass"


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_task21_wrong_terminal_register_interpretation_is_rejected(framework: Framework) -> None:
    """Wrong qubit-to-output binding must not survive framework normalization."""
    task = load_tasks(framework, "core")["21"]
    candidate = _replace_once(task["canonical_solution"], *_TERMINAL_OBSERVATION_MUTATIONS[framework])

    details = _grade(framework, "21", candidate)

    assert details["passed"] is False, (framework, details.get("reason"))
    assert details["semantic_status"] == "semantic_fail", (framework, details.get("reason"))


def test_qiskit_task33_wrong_dynamic_feedback_condition_is_rejected() -> None:
    """The IPE instrument checks the classically controlled correction branch."""
    task = load_tasks("qiskit", "core")["33"]
    candidate = _replace_once(
        task["canonical_solution"],
        "    with qc.if_test((cr[0], True)):\n",
        "    with qc.if_test((cr[0], False)):\n",
    )

    details = _grade("qiskit", "33", candidate)

    assert details["passed"] is False, details.get("reason")
    assert details["semantic_status"] == "semantic_fail", details.get("reason")


def test_cudaq_task22_full_relation_replay_rejects_a_wrong_carry_wire() -> None:
    """CUDA-Q lowering must verify the adder relation beyond its baked example."""
    task = load_tasks("cudaq", "core")["22"]
    accepted = _grade("cudaq", "22", task["canonical_solution"])
    candidate = _replace_once(
        task["canonical_solution"],
        "        x.ctrl(q[2], q[1])\n",
        "        x.ctrl(q[0], q[1])\n",
    )
    rejected = _grade("cudaq", "22", candidate)

    assert accepted["passed"] is True, accepted.get("reason")
    assert accepted["semantic_status"] == "verified_pass"
    assert rejected["passed"] is False, rejected.get("reason")
    assert rejected["semantic_status"] == "semantic_fail", rejected.get("reason")


def test_pennylane_task01_rejects_a_correct_raw_probability_shortcut() -> None:
    """A numerically correct array is not evidence of the required QNode program."""
    task = load_tasks("pennylane", "core")["01"]
    candidate = f"""\
import numpy as np

def {task["entry_point"]}():
    return np.array([1.0, 0.0, 0.0, 0.0])
"""

    details = _grade("pennylane", "01", candidate)

    assert details["passed"] is False, details.get("reason")
    assert details["semantic_status"] == "execution_error"
    assert details["reason"] == "captured_tape_required"
