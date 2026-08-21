from __future__ import annotations

import numpy as np
import pytest

from qceval.evals.ir import Circuit, Gate, from_framework, full_unitary

X = np.asarray([[0, 1], [1, 0]], dtype=complex)
H = np.asarray([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SWAP = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def _zero_state(num_qubits: int) -> np.ndarray:
    state = np.zeros(1 << num_qubits, dtype=complex)
    state[0] = 1.0
    return state


def test_batched_dense_construction_golden_single_and_two_qubit_gates() -> None:
    assert np.allclose(full_unitary(Circuit(2, (Gate.full(X, (0,), name="x"),))), np.kron(np.eye(2), X))
    assert np.allclose(full_unitary(Circuit(2, (Gate.full(H, (1,), name="h"),))), np.kron(H, np.eye(2)))
    assert np.allclose(full_unitary(Circuit(2, (Gate.full(SWAP, (0, 1), name="swap"),))), SWAP)


def test_qiskit_converter_matches_operator_exactly() -> None:
    # Qiskit is natively little-endian (qubit 0 = least-significant bit), the
    # same convention as the IR, so the converted unitary must equal
    # Operator(qc).data with NO permutation.
    qiskit = pytest.importorskip("qiskit")
    qi = pytest.importorskip("qiskit.quantum_info")

    for control, target in [(0, 1), (1, 0)]:
        qc = qiskit.QuantumCircuit(2)
        qc.cx(control, target)
        circuit = from_framework(qc, framework="qiskit")
        expected = np.asarray(qi.Operator(qc).data, dtype=complex)
        assert np.allclose(full_unitary(circuit), expected)


def test_qiskit_statevector_convention_pinned() -> None:
    qiskit = pytest.importorskip("qiskit")
    qi = pytest.importorskip("qiskit.quantum_info")

    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    qc.x(2)
    qc.cx(1, 2)
    circuit = from_framework(qc, framework="qiskit")

    ours = full_unitary(circuit) @ _zero_state(3)
    theirs = np.asarray(qi.Statevector(qc).data, dtype=complex)
    assert np.allclose(ours, theirs)


def test_cirq_statevector_convention_pinned() -> None:
    cirq = pytest.importorskip("cirq")

    q = cirq.LineQubit.range(3)
    native = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.T(q[1]), cirq.X(q[2]), cirq.CNOT(q[1], q[2]))
    circuit = from_framework(native, framework="cirq")

    ours = full_unitary(circuit) @ _zero_state(3)
    theirs = np.asarray(cirq.final_state_vector(native), dtype=complex)
    # Cirq indexes the first (sorted) qubit as the most-significant bit; the IR
    # is little-endian, so compare through the bit-reversal permutation.
    perm = [int(format(i, "03b")[::-1], 2) for i in range(8)]
    assert np.allclose(ours, theirs[perm])


def test_pennylane_statevector_convention_pinned() -> None:
    qml = pytest.importorskip("pennylane")

    def build() -> None:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.T(wires=1)
        qml.PauliX(wires=2)
        qml.CNOT(wires=[1, 2])

    with qml.tape.QuantumTape() as tape:
        build()
        qml.probs(wires=[0, 1, 2])

    circuit = from_framework(tape, framework="pennylane")
    ours = full_unitary(circuit) @ _zero_state(3)

    device = qml.device("default.qubit", wires=3)

    @qml.qnode(device)
    def run() -> object:
        build()
        return qml.state()

    theirs = np.asarray(run(), dtype=complex)
    # PennyLane also treats wire 0 as the most-significant bit.
    perm = [int(format(i, "03b")[::-1], 2) for i in range(8)]
    assert np.allclose(ours, theirs[perm])


def test_qiskit_gate_argument_order_regression() -> None:
    # CX(0,1)·X0 == X1·X0·CX(0,1) natively; a converter that swaps gate
    # argument order (control/target) breaks this identity.  Regression for
    # the endianness bug where Qiskit local matrices were bit-reversed.
    qiskit = pytest.importorskip("qiskit")

    qc1 = qiskit.QuantumCircuit(2)
    qc1.x(0)
    qc1.cx(0, 1)
    qc2 = qiskit.QuantumCircuit(2)
    qc2.cx(0, 1)
    qc2.x(0)
    qc2.x(1)

    left = from_framework(qc1, framework="qiskit")
    right = from_framework(qc2, framework="qiskit")
    assert np.allclose(full_unitary(left), full_unitary(right))


def test_qiskit_custom_gate_definition_is_decomposed() -> None:
    qiskit = pytest.importorskip("qiskit")

    sub = qiskit.QuantumCircuit(2, name="sub")
    sub.h(0)
    sub.cx(0, 1)
    qc = qiskit.QuantumCircuit(2)
    qc.append(sub.to_gate(), [0, 1])

    circuit = from_framework(qc, framework="qiskit")

    assert [gate.name for gate in circuit.gates] == ["h", "cx"]


def test_gate_rejects_ambiguous_full_matrix_with_controls() -> None:
    with pytest.raises(ValueError, match="full-local"):
        Gate(SWAP, wires=(0, 1), controls=(0,), name="bad")
