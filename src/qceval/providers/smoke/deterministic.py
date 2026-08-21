"""Deterministic-state fallback source generation for smoke provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _deterministic_code(entry_point: str, framework: str, spec: Mapping[str, Any]) -> str | None:
    bitstring = str(spec["expected_dominants"][0])
    min_ops = int(spec.get("min_non_measure_ops", 0) or 0)
    if framework == "cirq":
        return _cirq_deterministic_code(entry_point, bitstring, min_ops)
    if framework == "pennylane":
        return _pennylane_deterministic_code(entry_point, bitstring, min_ops)
    if framework == "cudaq":
        return _cudaq_deterministic_code(entry_point, bitstring, min_ops)
    return None


def _cirq_deterministic_code(entry_point: str, bitstring: str, min_ops: int) -> str:
    x_ops = [f"    circuit.append(cirq.X(q[{index}]))" for index, bit in enumerate(bitstring) if bit == "1"]
    filler = _cirq_identity_ops(bitstring, min_ops, len(x_ops))
    body = "\n".join([*filler, *x_ops, '    circuit.append(cirq.measure(*q, key="result"))'])
    return (
        "import cirq\n\n"
        f"def {entry_point}(*args, **kwargs):\n"
        f"    q = cirq.LineQubit.range({len(bitstring)})\n"
        "    circuit = cirq.Circuit()\n"
        f"{body}\n"
        "    return circuit\n"
    )


def _cirq_identity_ops(bitstring: str, min_ops: int, existing_ops: int) -> list[str]:
    if existing_ops >= min_ops:
        return []
    return [f"    circuit.append(cirq.H(q[0]))\n    circuit.append(cirq.H(q[0]))  # keeps |{bitstring}> unchanged"]


def _pennylane_deterministic_code(entry_point: str, bitstring: str, min_ops: int) -> str:
    x_ops = [f"        qml.PauliX(wires={index})" for index, bit in enumerate(bitstring) if bit == "1"]
    filler = _pennylane_identity_ops(bitstring, min_ops, len(x_ops))
    ops = "\n".join([*filler, *x_ops]) or "        pass"
    wires = list(range(len(bitstring)))
    return (
        "import pennylane as qml\n\n"
        f"def {entry_point}(*args, **kwargs):\n"
        f'    dev = qml.device("default.qubit", wires={len(bitstring)}, shots=None)\n\n'
        "    @qml.qnode(dev)\n"
        "    def circuit():\n"
        f"{ops}\n"
        f"        return qml.probs(wires={wires!r})\n\n"
        "    return circuit()\n"
    )


def _pennylane_identity_ops(bitstring: str, min_ops: int, existing_ops: int) -> list[str]:
    if existing_ops >= min_ops:
        return []
    return [f"        qml.Hadamard(wires=0)\n        qml.Hadamard(wires=0)  # keeps |{bitstring}> unchanged"]


def _cudaq_deterministic_code(entry_point: str, bitstring: str, min_ops: int) -> str:
    x_ops = [f"        x(q[{index}])" for index, bit in enumerate(bitstring) if bit == "1"]
    filler = _cudaq_identity_ops(bitstring, min_ops, len(x_ops))
    ops = "\n".join([*filler, *x_ops]) or "        pass"
    return (
        "import cudaq\n\n"
        f"def {entry_point}(*args, **kwargs):\n"
        "    @cudaq.kernel\n"
        "    def kernel():\n"
        f"        q = cudaq.qvector({len(bitstring)})\n"
        f"{ops}\n\n"
        "    return kernel\n"
    )


def _cudaq_identity_ops(bitstring: str, min_ops: int, existing_ops: int) -> list[str]:
    if existing_ops >= min_ops:
        return []
    return [f"        h(q[0])\n        h(q[0])  # keeps |{bitstring}> unchanged"]
