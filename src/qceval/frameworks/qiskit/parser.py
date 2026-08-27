"""Convert Qiskit circuits into the dense framework-neutral circuit IR."""

from __future__ import annotations

from typing import Any

import numpy as np

from qceval.evals.ir.core import (
    Circuit,
    Gate,
)

_QISKIT_SKIP = frozenset({"barrier"})
_QISKIT_MEASUREMENT = frozenset({"measure", "measure_all", "measure_active"})
_QISKIT_NON_UNITARY = frozenset(
    {
        "reset",
        "initialize",
        "delay",
        "snapshot",
        "save_state",
        "save_statevector",
        "save_density_matrix",
        "save_probabilities",
        "save_amplitudes",
        "set_statevector",
        "set_density_matrix",
    }
)
_QISKIT_CONTROL_FLOW = frozenset(
    {
        "if_else",
        "while_loop",
        "for_loop",
        "switch_case",
        "break_loop",
        "continue_loop",
        "box",
    }
)


def from_qiskit(circuit: Any) -> Circuit:
    """Convert a Qiskit ``QuantumCircuit`` to the neutral IR.

    Args:
        circuit: Qiskit circuit to lower.

    Returns:
        The lowered framework-neutral circuit.

    Raises:
        NotImplementedError: If the circuit contains unsupported non-unitary,
            classically controlled, or control-flow operations.
    """
    gates: list[Gate] = []
    measured_qubits: set[int] = set()
    for instruction in circuit.data:
        operation = instruction.operation
        name = str(operation.name)
        qargs = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if name in _QISKIT_CONTROL_FLOW or getattr(operation, "condition", None) is not None:
            raise NotImplementedError(
                f"qiskit operation {name!r} is classically controlled or control-flow and cannot be equivalence-checked"
            )
        if name in _QISKIT_MEASUREMENT:
            measured_qubits.update(qargs)
            continue
        if name in _QISKIT_SKIP:
            continue
        if name in _QISKIT_NON_UNITARY:
            raise NotImplementedError(f"qiskit operation {name!r} is non-unitary and cannot be equivalence-checked")
        if measured_qubits.intersection(qargs):
            raise NotImplementedError("qiskit mid-circuit measurement is non-unitary and cannot be equivalence-checked")
        gates.extend(_qiskit_operation_gates(operation, qargs))
    return Circuit(
        num_qubits=int(circuit.num_qubits),
        gates=tuple(gates),
    )


def _qiskit_operation_gates(
    operation: Any,
    qargs: tuple[int, ...],
) -> tuple[Gate, ...]:
    """Convert an operation, recursively decomposing custom gates."""
    name = str(operation.name)
    if name in _QISKIT_NON_UNITARY or name in _QISKIT_MEASUREMENT or name in _QISKIT_CONTROL_FLOW:
        raise NotImplementedError(f"qiskit operation {name!r} is non-unitary and cannot be equivalence-checked")
    if getattr(operation, "condition", None) is not None:
        raise NotImplementedError(
            f"qiskit operation {name!r} is classically controlled and cannot be equivalence-checked"
        )
    try:
        matrix = np.asarray(operation.to_matrix(), dtype=complex)
    except Exception:  # noqa: BLE001 - fall through to decomposition.
        definition = getattr(operation, "definition", None)
        if definition is None:
            raise NotImplementedError(f"qiskit operation {name!r} has no unitary matrix") from None
        gates: list[Gate] = []
        for nested in definition.data:
            nested_operation = nested.operation
            nested_name = str(nested_operation.name)
            if nested_name in _QISKIT_SKIP:
                continue
            if nested_name in _QISKIT_MEASUREMENT or nested_name in _QISKIT_NON_UNITARY:
                raise NotImplementedError(
                    f"qiskit gate definition contains non-unitary operation {nested_name!r}"
                ) from None
            nested_qargs = tuple(qargs[definition.find_bit(qubit).index] for qubit in nested.qubits)
            gates.extend(
                _qiskit_operation_gates(
                    nested_operation,
                    nested_qargs,
                )
            )
        return tuple(gates)
    return (Gate.full(matrix, qargs, name=name),)
