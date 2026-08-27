"""PennyLane tape metadata helpers."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pennylane as qml

from qceval.evals.structure import OperationSignature, detect_repeated_blocks

PENNYLANE_NORMALIZE = {
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "S": "s",
    "T": "t",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PhaseShift": "p",
    "CNOT": "cx",
    "CZ": "cz",
    "CY": "cy",
    "CH": "ch",
    "SWAP": "swap",
    "Toffoli": "ccx",
    "CCZ": "ccz",
    "CSWAP": "cswap",
    "IsingXX": "rxx",
    "IsingYY": "ryy",
    "IsingZZ": "rzz",
}

# Unitary matrices above this size are not needed for probability-only
# graders and make large QEC tasks spend minutes building dense matrices.
MAX_EXACT_UNITARY_WIRES = 6


def _metadata_from_tape(tape: Any | None, *, output_wires: Sequence[int] | None = None) -> dict[str, Any]:
    if tape is None:
        return {}
    op_counts: dict[str, int] = {}
    gate_family_counts: dict[str, int] = {}
    interaction_pairs: list[list[int]] = []
    block_ops: list[OperationSignature] = []
    for op in tape.operations:
        name = getattr(op, "name", type(op).__name__)
        op_counts[name] = op_counts.get(name, 0) + 1
        if isinstance(op, qml.measurements.MeasurementProcess):
            continue
        family = PENNYLANE_NORMALIZE.get(name, name.lower())
        gate_family_counts[family] = gate_family_counts.get(family, 0) + 1
        wires = list(op.wires)
        if all(isinstance(wire, int) for wire in wires):
            block_ops.append((family, tuple(int(wire) for wire in wires)))
        if all(isinstance(wire, int) for wire in wires):
            wire_indices = sorted(wires)
            interaction_pairs.extend([[a, b] for a, b in combinations(wire_indices, 2)])
    measured_wires = _measurement_wires(tape, output_wires=output_wires)
    measurement_pairs = [[int(wire), index] for index, wire in enumerate(measured_wires) if isinstance(wire, int)]
    return {
        "probability_method": "statevector",
        "num_qubits": len(tape.wires),
        "measurement_count": len(measured_wires),
        "return_measurement_count": len(tape.measurements),
        "non_measurement_operation_count": len(tape.operations),
        "circuit_depth": _estimate_tape_depth(tape.operations),
        "repeated_block_count": detect_repeated_blocks(block_ops),
        "measurement_pairs": measurement_pairs,
        "measurement_wires": [str(wire) for wire in measured_wires],
        "operation_counts": op_counts,
        "gate_family_counts": gate_family_counts,
        "interaction_pairs": interaction_pairs,
        "entangling_gate_count": sum(1 for op in tape.operations if len(op.wires) >= 2),
    }


def _estimate_tape_depth(operations: Sequence[Any]) -> int:
    wire_depths: dict[Any, int] = {}
    max_depth = 0
    for op in operations:
        wires = list(op.wires)
        if not wires:
            continue
        depth = max((wire_depths.get(wire, 0) for wire in wires), default=0) + 1
        for wire in wires:
            wire_depths[wire] = depth
        max_depth = max(max_depth, depth)
    return max_depth


def _unitary_from_tape(tape: Any) -> np.ndarray | None:
    if len(tape.wires) > MAX_EXACT_UNITARY_WIRES:
        return None
    try:
        # Sort wires so wire 0 is the most-significant qubit regardless of gate
        # insertion order, matching the big-endian target-unitary convention.
        return np.asarray(qml.matrix(tape.operations, wire_order=sorted(tape.wires)), dtype=complex)
    except Exception:
        return None


def _measurement_wires(tape: Any, *, output_wires: Sequence[int] | None = None) -> list[Any]:
    # qml.state() returns the full statevector; it is a state read-out, not a
    # projective/classical measurement, so it contributes no measured wires.
    if tape.measurements and all(
        isinstance(measurement, qml.measurements.StateMP) for measurement in tape.measurements
    ):
        return []
    for measurement in tape.measurements:
        wires = list(measurement.wires)
        if wires:
            return wires
    if output_wires is not None:
        return [int(wire) for wire in output_wires]
    return sorted(tape.wires, reverse=True)
