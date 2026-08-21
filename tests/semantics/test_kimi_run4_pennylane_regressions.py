"""Regressions for valid PennyLane responses rejected during the kimi run4 audit."""

from __future__ import annotations

from qceval.evals.evaluator import build_evaluator

_SWAPTEST_NESTED_QNODE = """\
import pennylane as qml

def swaptest_zaxis(unknown_state):
    dev = qml.device("default.qubit", wires=3, shots=None)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.StatePrep(unknown_state(), wires=1)
        qml.CSWAP(wires=[0, 1, 2])
        qml.Hadamard(wires=0)
        return qml.probs(wires=[0])

    return circuit()
"""

_W_STATE_NEGATIVE_CONTROLS = """\
import pennylane as qml
import numpy as np

def W_State_4():
    dev = qml.device("default.qubit", wires=4, shots=None)

    @qml.qnode(dev)
    def circuit():
        qml.RY(np.pi / 3, wires=0)
        qml.ctrl(qml.RY, control=0, control_values=[0])(
            2 * np.arcsin(1 / np.sqrt(3)), wires=1
        )
        qml.ctrl(qml.PauliX, control=[0, 1], control_values=[0, 0])(wires=3)
        qml.ctrl(qml.Hadamard, control=[0, 1], control_values=[0, 0])(wires=2)
        qml.CNOT(wires=[2, 3])
        return qml.probs(wires=[0, 1, 2, 3])

    return circuit()
"""

_CTQW_ISING_TROTTER = """\
import pennylane as qml

def ctqw_path4_spatial_search():
    n = 2
    gamma = 0.3
    t = 1.5
    steps = 4
    dt = t / steps

    dev = qml.device("default.qubit", wires=n, shots=None)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        for _ in range(steps):
            qml.RX(-gamma * dt, wires=0)

            qml.CNOT(wires=[0, 1])
            qml.RX(-gamma * dt / 2, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.IsingYY(-gamma * dt / 2, wires=[0, 1])

            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.ControlledPhaseShift(dt, wires=[0, 1])
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)

            qml.IsingYY(-gamma * dt / 2, wires=[0, 1])

            qml.CNOT(wires=[0, 1])
            qml.RX(-gamma * dt / 2, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.RX(-gamma * dt, wires=0)

        return qml.probs(wires=[1, 0])

    return circuit()
"""

_CTQW_OPAQUE_FULL_REGISTER_UNITARY = """\
import numpy as np
import pennylane as qml
from scipy.linalg import expm

def ctqw_path4_spatial_search():
    gamma = 0.3
    t = 1.5
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    oracle = np.diag([0.0, 0.0, 0.0, 1.0])
    hamiltonian = -gamma * adjacency - oracle
    unitary = expm(-1j * hamiltonian * t)

    dev = qml.device("default.qubit", wires=2, shots=None)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.QubitUnitary(unitary, wires=[0, 1])
        return qml.probs(wires=[0, 1])

    return circuit()
"""


def _grade(task_id: str, entry_point: str, code: str) -> dict:
    _, details = build_evaluator("pennylane", suite="core").grade_code(
        task_id=task_id,
        code=code,
        entry_point=entry_point,
    )
    return details


def test_pennylane_nested_qnode_call_grades_top_level_tape() -> None:
    details = _grade("06", "swaptest_zaxis", _SWAPTEST_NESTED_QNODE)
    assert details["passed"] is True, details.get("reason")


def test_pennylane_negative_control_values_lower_correctly() -> None:
    details = _grade("24", "W_State_4", _W_STATE_NEGATIVE_CONTROLS)
    assert details["passed"] is True, details.get("reason")


def test_pennylane_native_ising_gates_are_not_dense_unitaries() -> None:
    details = _grade("57", "ctqw_path4_spatial_search", _CTQW_ISING_TROTTER)
    assert details["passed"] is True, details.get("reason")


def test_pennylane_opaque_full_register_unitary_remains_forbidden() -> None:
    details = _grade("57", "ctqw_path4_spatial_search", _CTQW_OPAQUE_FULL_REGISTER_UNITARY)
    assert details["passed"] is False
    assert "forbid_full_register_dense_unitary" in str(details.get("reason"))
