"""Deterministic task inputs used by framework executors."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from qceval.models import Framework

# First declared witness state for the core task 06 SWAP test. Additional
# witness states come from the contract's diagnostic points and are bound by
# the evaluator through :func:`task6_witness_state`.
TASK6_DEFAULT_WITNESS = (2.0943951023931953, 0.6283185307179586)


def global_inputs(framework: Framework) -> dict[str, Any]:
    """Return deterministic runtime inputs for task entry points.

    Args:
        framework: Framework whose task functions will receive the inputs.

    Returns:
        Mapping from task id to arguments consumed by special-case tasks.
    """
    return {
        "04": [_task4_graph(), [_angle() for _ in range(5)], [_angle() for _ in range(5)]],
        "06": task6_witness_state(framework, *TASK6_DEFAULT_WITNESS),
        "29": [1, 0],
        "39": [_angle(), _angle()],
        "40": [_angle() for _ in range(8)],
        "41": [_angle() for _ in range(6)],
        "42": [_angle(), _angle(), _angle()],
    }


def _angle() -> float:
    return float((25 * np.pi) / 54)


def _task4_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]])
    return graph


def task6_witness_state(framework: Framework, theta: float, phi: float) -> Any:
    """Build one framework-native unknown input state for the SWAP test.

    The witness is ``rz(phi) @ ry(theta) |0>``, whose overlap with ``|0>`` is
    ``cos(theta / 2) ** 2``. Qiskit and Cirq receive a one-qubit preparation
    circuit, PennyLane receives a callable returning the state vector, and
    CUDA-Q receives the explicit state vector because kernels cannot accept a
    foreign circuit object.

    Args:
        framework: Framework whose task function will receive the state.
        theta: Polar angle of the witness state in radians.
        phi: Relative phase of the witness state in radians.

    Returns:
        Framework-native representation of the witness state.
    """
    if framework == "qiskit":
        return _task6_qiskit(theta, phi)
    if framework == "cirq":
        return _task6_cirq(theta, phi)
    if framework == "pennylane":
        return _task6_pennylane(theta, phi)
    return task6_statevector(theta, phi)


def task6_statevector(theta: float, phi: float) -> Any:
    """Return the explicit witness state vector ``rz(phi) ry(theta) |0>``.

    Args:
        theta: Polar angle of the witness state in radians.
        phi: Relative phase of the witness state in radians.

    Returns:
        Length-2 complex state vector.
    """
    return np.array(
        [
            np.cos(theta / 2.0) * np.exp(-0.5j * phi),
            np.sin(theta / 2.0) * np.exp(0.5j * phi),
        ]
    )


def _task6_qiskit(theta: float, phi: float) -> Any:
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return qc


def _task6_cirq(theta: float, phi: float) -> Any:
    import cirq

    qubit = cirq.LineQubit(0)
    return cirq.Circuit(cirq.ry(theta)(qubit), cirq.rz(phi)(qubit))


def _task6_pennylane(theta: float, phi: float) -> Any:
    import pennylane as qml

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit() -> Any:
        qml.RY(theta, wires=0)
        qml.RZ(phi, wires=0)
        return qml.state()

    return circuit
