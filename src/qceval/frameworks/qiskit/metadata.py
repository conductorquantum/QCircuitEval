"""Qiskit circuit metadata helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qceval.evals.structure import OperationSignature, detect_repeated_blocks

IGNORED_GATE_FAMILIES = {"measure", "reset", "barrier"}
MAX_EXACT_UNITARY_QUBITS = 8


def circuit_without_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return copy of circuit with measurement operations removed.

    Args:
        circuit: Source Qiskit circuit.

    Returns:
        New circuit containing all non-measurement operations.
    """
    stripped = QuantumCircuit(*circuit.qregs)
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            stripped.append(instruction.operation, instruction.qubits, [])
    return stripped


def measurement_pairs(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    """Return ``(qubit_index, clbit_index)`` pairs for measurements.

    Args:
        circuit: Qiskit circuit to inspect.

    Returns:
        Measurement pairs in circuit operation order.
    """
    pairs: list[tuple[int, int]] = []
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            qubit_index = circuit.find_bit(instruction.qubits[0]).index
            clbit_index = circuit.find_bit(instruction.clbits[0]).index
            pairs.append((qubit_index, clbit_index))
    return pairs


def circuit_unitary(circuit: QuantumCircuit) -> np.ndarray | None:
    """Return circuit unitary, or ``None`` when not available.

    Args:
        circuit: Qiskit circuit to convert.

    Returns:
        Complex unitary matrix with measurements removed, or ``None`` if Qiskit
        cannot construct an operator.
    """
    if circuit.num_qubits > MAX_EXACT_UNITARY_QUBITS:
        return None
    try:
        return np.asarray(Operator(circuit_without_measurements(circuit)).data, dtype=complex)
    except Exception:
        return None


def circuit_metadata(circuit: QuantumCircuit) -> dict[str, Any]:
    """Return structural metadata for a Qiskit circuit.

    Args:
        circuit: Qiskit circuit to inspect.

    Returns:
        JSON-compatible metadata with qubit counts, measurement layout,
        operation counts, and entangling-gate count.
    """
    pairs = measurement_pairs(circuit)
    op_counts: dict[str, int] = {}
    gate_family_counts: dict[str, int] = {}
    interaction_pairs: list[list[int]] = []
    entangling = 0
    block_ops: list[OperationSignature] = []
    for instruction in circuit.data:
        name = instruction.operation.name
        op_counts[name] = op_counts.get(name, 0) + 1
        entangling += int(name != "measure" and len(instruction.qubits) >= 2)
        if name in IGNORED_GATE_FAMILIES:
            continue
        gate_family_counts[name] = gate_family_counts.get(name, 0) + 1
        ordered_qubit_indices = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        block_ops.append((name, ordered_qubit_indices))
        qubit_indices = sorted(ordered_qubit_indices)
        interaction_pairs.extend([[a, b] for a, b in combinations(qubit_indices, 2)])
    return {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "measurement_count": len(pairs),
        "non_measurement_operation_count": sum(count for name, count in op_counts.items() if name != "measure"),
        "circuit_depth": circuit.depth(),
        "repeated_block_count": detect_repeated_blocks(block_ops),
        "measurement_pairs": [[q, c] for q, c in pairs],
        "operation_counts": op_counts,
        "gate_family_counts": gate_family_counts,
        "interaction_pairs": interaction_pairs,
        "entangling_gate_count": entangling,
        "has_measurements": bool(pairs),
    }
