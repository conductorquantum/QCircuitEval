"""Unitary fallback source generation for smoke provider."""

from __future__ import annotations

from qceval.providers.smoke.utils import _target_qubits


def _unitary_code(entry_point: str, framework: str, target: str) -> str | None:
    if framework == "cirq":
        return _cirq_unitary_code(entry_point, target)
    if framework == "pennylane":
        return _pennylane_unitary_code(entry_point, target)
    if framework == "cudaq":
        return _cudaq_unitary_code(entry_point, target)
    return None


def _cirq_unitary_code(entry_point: str, target: str) -> str:
    n_qubits = _target_qubits(target)
    op = _cirq_unitary_op(target)
    return (
        "import cirq\n"
        "import numpy as np\n\n"
        f"def {entry_point}(*args, **kwargs):\n"
        f"    q = cirq.LineQubit.range({n_qubits})\n"
        "    circuit = cirq.Circuit()\n"
        f"{op}\n"
        "    return circuit\n"
    )


def _cirq_unitary_op(target: str) -> str:
    if target == "cx":
        return "    circuit.append(cirq.CNOT(q[0], q[1]))"
    if target == "ccx":
        return "    circuit.append(cirq.CCX(q[0], q[1], q[2]))"
    if target == "controlled_h":
        return "    circuit.append(cirq.H(q[1]).controlled_by(q[0]))"
    return (
        "    theta, phi, lam = args[:3]\n"
        "    matrix = np.array([\n"
        "        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],\n"
        "        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],\n"
        "    ], dtype=complex)\n"
        "    circuit.append(cirq.MatrixGate(matrix).on(q[0]))"
    )


def _pennylane_unitary_code(entry_point: str, target: str) -> str:
    n_qubits = _target_qubits(target)
    op = _pennylane_unitary_op(target)
    wires = list(range(n_qubits))
    return (
        "import numpy as np\n"
        "import pennylane as qml\n\n"
        f"def {entry_point}(*args, **kwargs):\n"
        f'    dev = qml.device("default.qubit", wires={n_qubits}, shots=None)\n\n'
        "    @qml.qnode(dev)\n"
        "    def circuit():\n"
        f"{op}\n"
        f"        return qml.probs(wires={wires!r})\n\n"
        "    return circuit()\n"
    )


def _pennylane_unitary_op(target: str) -> str:
    if target == "cx":
        return "        qml.CNOT(wires=[0, 1])"
    if target == "ccx":
        return "        qml.Toffoli(wires=[0, 1, 2])"
    if target == "controlled_h":
        return (
            "        h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)\n"
            "        qml.ControlledQubitUnitary(h, wires=[0, 1])"
        )
    return (
        "        theta, phi, lam = args[:3]\n"
        "        matrix = np.array([\n"
        "            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],\n"
        "            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],\n"
        "        ], dtype=complex)\n"
        "        qml.QubitUnitary(matrix, wires=0)"
    )


def _cudaq_unitary_code(entry_point: str, target: str) -> str:
    matrix = _unitary_matrix_expr(target)
    return f"import numpy as np\n\ndef {entry_point}(*args, **kwargs):\n{matrix}\n    return matrix\n"


def _unitary_matrix_expr(target: str) -> str:
    if target == "cx":
        return "    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)"
    if target == "ccx":
        return "    matrix = np.eye(8, dtype=complex)\n    matrix[[6, 7], :] = matrix[[7, 6], :]"
    if target == "controlled_h":
        return (
            "    matrix = np.eye(4, dtype=complex)\n"
            "    matrix[2:, 2:] = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)"
        )
    return (
        "    theta, phi, lam = args[:3]\n"
        "    matrix = np.array([\n"
        "        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],\n"
        "        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],\n"
        "    ], dtype=complex)"
    )
