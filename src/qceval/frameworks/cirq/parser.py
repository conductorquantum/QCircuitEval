"""Convert Cirq circuits into the dense framework-neutral circuit IR."""

from __future__ import annotations

from typing import Any

import numpy as np

from qceval.evals.ir.core import (
    Circuit,
    Gate,
    _to_little_endian_matrix,
)


def from_cirq(circuit: Any) -> Circuit:
    """Convert a Cirq circuit to the neutral IR.

    Args:
        circuit: Cirq circuit to lower.

    Returns:
        The lowered framework-neutral circuit.

    Raises:
        NotImplementedError: If the circuit contains a parameterized,
            non-unitary, or mid-circuit-measurement operation.
    """
    import cirq

    order = sorted(circuit.all_qubits())
    qubit_index = {qubit: index for index, qubit in enumerate(order)}
    gates: list[Gate] = []
    measured_qubits: set[Any] = set()
    for operation in circuit.all_operations():
        if cirq.is_measurement(operation):
            measured_qubits.update(operation.qubits)
            continue
        if measured_qubits.intersection(operation.qubits):
            raise NotImplementedError("cirq mid-circuit measurement is non-unitary and cannot be equivalence-checked")
        if cirq.is_parameterized(operation) or not cirq.has_unitary(operation):
            raise NotImplementedError(f"cirq operation {operation!r} is non-unitary and cannot be equivalence-checked")
        matrix = np.asarray(cirq.unitary(operation), dtype=complex)
        qargs = tuple(qubit_index[qubit] for qubit in operation.qubits)
        name = str(operation.gate) if operation.gate is not None else type(operation).__name__
        gates.append(
            Gate.full(
                _to_little_endian_matrix(matrix, len(qargs)),
                qargs,
                name=name,
            )
        )
    return Circuit(num_qubits=len(order), gates=tuple(gates))
