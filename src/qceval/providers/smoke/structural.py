"""Structural fallback source generation for smoke provider."""

from __future__ import annotations


def _structural_code(entry_point: str, framework: str) -> str | None:
    if framework == "cirq":
        return (
            "import cirq\n\n"
            f"def {entry_point}(param):\n"
            "    q = cirq.LineQubit.range(2)\n"
            "    return cirq.Circuit(cirq.rx(param[0]).on(q[0]), cirq.rx(param[1]).on(q[1]))\n"
        )
    if framework == "pennylane":
        return (
            "import pennylane as qml\n\n"
            f"def {entry_point}(param):\n"
            '    dev = qml.device("default.qubit", wires=2, shots=None)\n\n'
            "    @qml.qnode(dev)\n"
            "    def circuit():\n"
            "        qml.RX(param[0], wires=0)\n"
            "        qml.RX(param[1], wires=1)\n"
            "        return qml.probs(wires=[0, 1])\n\n"
            "    return circuit()\n"
        )
    if framework == "cudaq":
        return f"import numpy as np\n\ndef {entry_point}(param):\n    return np.array([1.0, 0.0, 0.0, 0.0])\n"
    return None
