"""Convert PennyLane tapes into the dense framework-neutral circuit IR."""

from __future__ import annotations

from typing import Any

import numpy as np

from qceval.evals.ir.core import (
    Circuit,
    Gate,
    _to_little_endian_matrix,
)


def from_pennylane(tape: Any) -> Circuit:
    """Convert a PennyLane tape to the neutral IR.

    Args:
        tape: PennyLane quantum tape to lower.

    Returns:
        The lowered framework-neutral circuit.

    Raises:
        NotImplementedError: If the tape contains a conditional operation or
            uses a wire after it has been measured.
    """
    import pennylane as qml

    labels = [int(wire) for operation in tape.operations for wire in operation.wires]
    labels += [int(wire) for measurement in tape.measurements for wire in measurement.wires]
    num_qubits = max(labels) + 1 if labels else 0
    gates: list[Gate] = []
    measured_wires: set[int] = set()
    for operation in tape.operations:
        operation_name = getattr(
            operation,
            "name",
            type(operation).__name__,
        )
        type_name = type(operation).__name__
        wires = tuple(int(wire) for wire in operation.wires)
        if "MidMeasure" in type_name or "MidMeasure" in str(operation_name):
            measured_wires.update(wires)
            continue
        if "Conditional" in type_name:
            raise NotImplementedError(
                f"pennylane operation {operation_name!r} is non-unitary and cannot be equivalence-checked"
            )
        if measured_wires.intersection(wires):
            raise NotImplementedError(
                "pennylane mid-circuit measurement is non-unitary and cannot be equivalence-checked"
            )
        matrix = np.asarray(qml.matrix(operation), dtype=complex)
        gates.append(
            Gate.full(
                _to_little_endian_matrix(matrix, len(wires)),
                wires,
                name=str(operation_name),
            )
        )
    return Circuit(num_qubits=num_qubits, gates=tuple(gates))
