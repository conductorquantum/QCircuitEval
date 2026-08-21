"""Cirq circuit metadata and exact simulation helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import cirq
import numpy as np

from qceval.evals.structure import OperationSignature, detect_repeated_blocks

CIRQ_NORMALIZE = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "T": "t",
    "Rx": "rx",
    "Ry": "ry",
    "Rz": "rz",
    "CNOT": "cx",
    "CX": "cx",
    "CZ": "cz",
    "CY": "cy",
    "CH": "ch",
    "SWAP": "swap",
    "ISWAP": "iswap",
    "TOFFOLI": "ccx",
    "CCX": "ccx",
    "CCZ": "ccz",
    "FREDKIN": "cswap",
    "CSWAP": "cswap",
}
MAX_EXACT_UNITARY_QUBITS = 8


def _deferred_probabilities(circuit: cirq.Circuit) -> np.ndarray:
    """Exact measured distribution of a dynamic circuit via deferral.

    ``cirq.defer_measurements`` rewrites mid-circuit measurements and
    classical controls into coherent form on ancilla qubits, which is an
    exact transformation. Deferred measurement operations move to the end of
    the circuit, so the projection follows the *original* measurement-key
    order to keep the grading bit order stable.

    Bit order matches the Program IR path (``_classical_render_order``):
    sequential single-qubit keys render with the *first* measured key as the
    least-significant bit, matching Qiskit's classical register packing for
    iterative protocols; a single joint key keeps its listed qubit order.
    """
    ordered_keys = _ordered_measurement_keys(circuit)
    result_keys = [key for key in ordered_keys if key == "result"]
    selected_keys = result_keys or ordered_keys
    deferred = cirq.defer_measurements(circuit)
    qubit_order = sorted(deferred.all_qubits())
    measured = _deferred_measurement_qubits(deferred, selected_keys)
    if not measured:
        measured = list(qubit_order)
    positions = [qubit_order.index(qubit) for qubit in measured]
    basis_probs = _exact_basis_probabilities(deferred, qubit_order)
    out = np.zeros(2 ** len(measured), dtype=float)
    for basis_index, probability in enumerate(basis_probs):
        out[_measured_index(basis_index, positions, len(qubit_order))] += float(probability)
    total = float(out.sum())
    return out / total if total > 0 else out


def _ordered_measurement_keys(circuit: cirq.Circuit) -> list[str]:
    ordered: list[str] = []
    for operation in circuit.all_operations():
        if cirq.is_measurement(operation):
            key = cirq.measurement_key_name(operation)
            if key not in ordered:
                ordered.append(key)
    return ordered


def _deferred_measurement_qubits(circuit: cirq.Circuit, selected_keys: list[str]) -> list[cirq.Qid]:
    key_qubits: dict[str, tuple[cirq.Qid, ...]] = {}
    for operation in circuit.all_operations():
        if cirq.is_measurement(operation):
            key_qubits.setdefault(cirq.measurement_key_name(operation), operation.qubits)
    measured = [qubit for key in selected_keys for qubit in key_qubits.get(key, ())]
    if len(selected_keys) > 1 and all(len(key_qubits.get(key, ())) == 1 for key in selected_keys):
        # Dynamic-circuit style: mirror the Program IR classical render order
        # so both grading paths agree (first measured key least significant).
        measured.reverse()
    return measured


def _exact_basis_probabilities(circuit: cirq.Circuit, qubit_order: list[cirq.Qid]) -> np.ndarray:
    """Exact computational-basis probabilities, tolerating channels like reset."""
    try:
        state = np.asarray(
            cirq.final_state_vector(
                circuit,
                qubit_order=qubit_order,
                ignore_terminal_measurements=True,
                dtype=np.complex128,
            ),
            dtype=complex,
        )
        return np.abs(state) ** 2
    except ValueError:
        density = cirq.final_density_matrix(
            circuit,
            qubit_order=qubit_order,
            ignore_measurement_results=True,
            dtype=np.complex128,
        )
        return np.real(np.diag(np.asarray(density, dtype=complex)))


def exact_probabilities(circuit: cirq.Circuit) -> np.ndarray:
    """Compute exact probabilities for measured qubits.

    If the circuit has measurement operations, result-key measurements are
    preferred and all measured qubits are projected in circuit qubit order.  If
    no measurements exist, all qubits are treated as measured.

    Args:
        circuit: Cirq circuit to simulate.

    Returns:
        Normalized probability vector.
    """
    qubit_order = sorted(circuit.all_qubits())
    measured = _measurement_qubits(circuit) or qubit_order
    measured_positions = [qubit_order.index(q) for q in measured]
    state = exact_statevector(circuit)
    basis_probs = np.abs(np.asarray(state, dtype=complex)) ** 2
    out = np.zeros(2 ** len(measured), dtype=float)
    for basis_index, probability in enumerate(basis_probs):
        # Cirq basis indexes follow sorted qubit order; grader output follows
        # the selected measurement qubits as a compact result register.
        out[_measured_index(basis_index, measured_positions, len(qubit_order))] += float(probability)
    total = float(out.sum())
    return out / total if total > 0 else out


def exact_statevector(circuit: cirq.Circuit) -> np.ndarray:
    """Compute exact final statevector with terminal measurements ignored.

    Args:
        circuit: Cirq circuit to simulate.

    Returns:
        Complex statevector amplitudes in sorted-qubit order.
    """
    return np.asarray(
        cirq.final_state_vector(
            circuit,
            qubit_order=sorted(circuit.all_qubits()),
            ignore_terminal_measurements=True,
            dtype=np.complex128,
        ),
        dtype=complex,
    )


def circuit_unitary(circuit: cirq.Circuit) -> np.ndarray | None:
    """Return circuit unitary, or ``None`` when not available.

    Args:
        circuit: Cirq circuit to convert.

    Returns:
        Complex unitary matrix with terminal measurements dropped, or ``None``
        if Cirq cannot construct a unitary.
    """
    if len(circuit.all_qubits()) > MAX_EXACT_UNITARY_QUBITS:
        return None
    try:
        return np.asarray(cirq.unitary(cirq.drop_terminal_measurements(circuit)), dtype=complex)
    except Exception:
        return None


def circuit_metadata(circuit: cirq.Circuit) -> dict[str, Any]:
    """Return structural metadata for a Cirq circuit.

    Args:
        circuit: Cirq circuit to inspect.

    Returns:
        JSON-compatible metadata with qubit count, measurements, operation
        counts, and entangling-gate count.
    """
    ops = list(circuit.all_operations())
    qubit_order = sorted(circuit.all_qubits())
    op_counts: dict[str, int] = {}
    gate_family_counts: dict[str, int] = {}
    interaction_pairs: list[list[int]] = []
    non_measurement_ops = 0
    block_ops: list[OperationSignature] = []
    for op in ops:
        name = str(op.gate).split("(")[0] if op.gate is not None else type(op).__name__
        op_counts[name] = op_counts.get(name, 0) + 1
        if cirq.is_measurement(op):
            continue
        non_measurement_ops += 1
        family = CIRQ_NORMALIZE.get(name, name.lower())
        gate_family_counts[family] = gate_family_counts.get(family, 0) + 1
        ordered_qubit_indices = tuple(qubit_order.index(qubit) for qubit in op.qubits)
        block_ops.append((family, ordered_qubit_indices))
        qubit_indices = sorted(ordered_qubit_indices)
        interaction_pairs.extend([[a, b] for a, b in combinations(qubit_indices, 2)])
    measurement_pairs = [[qubit_order.index(qubit), index] for index, qubit in enumerate(_measurement_qubits(circuit))]
    return {
        "num_qubits": len(circuit.all_qubits()),
        "measurement_count": sum(len(op.qubits) for op in ops if cirq.is_measurement(op)),
        "non_measurement_operation_count": non_measurement_ops,
        "circuit_depth": len(circuit),
        "repeated_block_count": detect_repeated_blocks(block_ops),
        "measurement_pairs": measurement_pairs,
        "operation_counts": op_counts,
        "gate_family_counts": gate_family_counts,
        "interaction_pairs": interaction_pairs,
        "entangling_gate_count": sum(1 for op in ops if not cirq.is_measurement(op) and len(op.qubits) >= 2),
        "measurement_qubits": [str(q) for q in _measurement_qubits(circuit)],
    }


def _measurement_qubits(circuit: cirq.Circuit) -> list[cirq.Qid]:
    measurement_ops = [op for op in circuit.all_operations() if cirq.is_measurement(op)]
    result_ops = [op for op in measurement_ops if "result" in cirq.measurement_key_names(op)]
    selected = result_ops or measurement_ops
    qubits: list[cirq.Qid] = []
    for op in selected:
        qubits.extend(op.qubits)
    return qubits


def _measured_index(basis_index: int, positions: list[int], n_qubits: int) -> int:
    measured_index = 0
    for position in positions:
        bit = (basis_index >> (n_qubits - position - 1)) & 1
        measured_index = (measured_index << 1) | bit
    return measured_index
