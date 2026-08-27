from __future__ import annotations

import pytest

from qceval.frameworks.cirq import execute_cirq_task
from qceval.frameworks.pennylane import execute_pennylane_task
from qceval.frameworks.qiskit import execute_qiskit_task


def test_framework_executors_accept_call_args() -> None:
    # Arrange
    qiskit_code = """
from qiskit import QuantumCircuit

def answer(x):
    qc = QuantumCircuit(1, 1)
    if x:
        qc.x(0)
    qc.measure(0, 0)
    return qc
"""
    cirq_code = """
import cirq

def answer(x):
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    if x:
        circuit.append(cirq.X(q))
    circuit.append(cirq.measure(q, key="result"))
    return circuit
"""
    pennylane_code = """
import pennylane as qml

def answer(x):
    dev = qml.device("default.qubit", wires=1, shots=None)
    @qml.qnode(dev)
    def circuit():
        if x:
            qml.PauliX(wires=0)
        return qml.probs(wires=[0])
    return circuit()
"""

    # Act
    qiskit = execute_qiskit_task(task_id="01", code=qiskit_code, entry_point="answer", inputs={}, call_args=(1,))
    cirq = execute_cirq_task(task_id="01", code=cirq_code, entry_point="answer", inputs={}, call_args=(1,))
    pennylane = execute_pennylane_task(
        task_id="01", code=pennylane_code, entry_point="answer", inputs={}, call_args=(1,)
    )

    # Assert
    assert qiskit.probabilities == [0.0, 1.0]
    assert cirq.probabilities == [0.0, 1.0]
    assert pennylane.probabilities == [0.0, 1.0]


def test_cudaq_executor_accepts_call_args() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    from qceval.frameworks.cudaq import execute_cudaq_task

    code = (
        "import numpy as np\n\n"
        "def answer(bit: int):\n"
        "    return np.array([0.0, 1.0]) if bit else np.array([1.0, 0.0])\n"
    )

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={}, call_args=(1,))

    # Assert
    assert result.probabilities == [0.0, 1.0]
