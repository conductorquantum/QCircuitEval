"""Regressions for valid responses rejected during the full prompt audit."""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks


def _assert_passes(framework: str, task_id: str, entry_point: str, code: str, *, suite: str = "core") -> None:
    _, details = build_evaluator(framework, suite=suite).grade_code(
        task_id=task_id,
        code=code,
        entry_point=entry_point,
    )
    assert details["passed"] is True, details.get("reason")


def test_qiskit_unused_classical_bits_do_not_widen_observation() -> None:
    _assert_passes(
        "qiskit",
        "07",
        "swaptest_individual",
        """\
from qiskit import QuantumCircuit

def swaptest_individual():
    qc = QuantumCircuit(9, 9)
    qc.x(1); qc.x(2)
    for i in range(3):
        anc = i + 6
        qc.h(anc); qc.cswap(anc, i, i + 3); qc.h(anc)
        qc.measure(anc, anc)
    return qc
""",
    )


@pytest.mark.parametrize(
    ("task_id", "entry_point", "body"),
    (
        (
            "30",
            "or_circuit",
            "qc.cx(0, 2); qc.cx(1, 2); qc.ccx(0, 1, 2); qc.measure(2, 2)",
        ),
        (
            "35",
            "parity_check_3bit",
            "qc.cx(0, 3); qc.cx(1, 3); qc.cx(2, 3); qc.measure(3, 3)",
        ),
    ),
)
def test_qiskit_same_index_sparse_measurement_is_observed(
    task_id: str,
    entry_point: str,
    body: str,
) -> None:
    width = 3 if task_id == "30" else 4
    _assert_passes(
        "qiskit",
        task_id,
        entry_point,
        f"""\
from qiskit import QuantumCircuit

def {entry_point}():
    qc = QuantumCircuit({width}, {width})
    {body}
    return qc
""",
    )


def test_three_qubit_bernstein_vazirani_phase_oracle_passes() -> None:
    _assert_passes(
        "qiskit",
        "20",
        "Bernstein_Vazirani_011",
        """\
from qiskit import QuantumCircuit

def Bernstein_Vazirani_011():
    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    qc.z(0); qc.z(1)
    qc.h(range(3))
    qc.measure(range(3), range(3))
    return qc
""",
    )


def test_compact_simon_oracle_passes() -> None:
    _assert_passes(
        "pennylane",
        "13",
        "Simon_11",
        """\
import pennylane as qml

def Simon_11():
    dev = qml.device("default.qubit", wires=3, shots=None)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(0); qml.Hadamard(1)
        qml.CNOT([0, 2]); qml.CNOT([1, 2])
        qml.Hadamard(0); qml.Hadamard(1)
        return qml.probs(wires=[0, 1])
    return circuit()
""",
    )


def test_hadamard_first_bell_pair_teleportation_passes() -> None:
    """The teleportation recipe accepts either Bell-pair Hadamard placement."""
    _assert_passes(
        "cirq",
        "18",
        "Quantum_Teleportation",
        """\
import cirq
from numpy import pi

def Quantum_Teleportation():
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(
        cirq.rx(pi / 2)(q[0]),
        cirq.H(q[2]),
        cirq.CNOT(q[2], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[0]),
        cirq.CNOT(q[1], q[2]),
        cirq.CZ(q[0], q[2]),
        cirq.measure(q[2], key="m"),
    )
""",
    )


def test_cirq_controlled_composite_hadamard_test_passes() -> None:
    _assert_passes(
        "cirq",
        "58",
        "hadamard_test_matrix_element",
        """\
import cirq

def hadamard_test_matrix_element():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1))
    for op in (cirq.CZ(q1, q2), cirq.H(q1), cirq.T(q1), cirq.CX(q1, q2)):
        circuit.append(op.controlled_by(q0))
    circuit.append(cirq.H(q0))
    circuit.append(cirq.measure(q0, key="result"))
    return circuit
""",
    )


@pytest.mark.parametrize(
    ("framework", "code"),
    (
        (
            "qiskit",
            """\
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

def qpe_x_gate():
    qc = QuantumCircuit(4, 3)
    qc.h(range(3))
    qc.cx(0, 3)
    qc.compose(QFT(3, inverse=True), range(3), inplace=True)
    qc.measure(range(3), range(3))
    return qc
""",
        ),
        (
            "cirq",
            """\
import cirq

def qpe_x_gate():
    q0, q1, q2, target = cirq.LineQubit.range(4)
    return cirq.Circuit(
        cirq.H.on_each(q0, q1, q2),
        cirq.CNOT(q0, target),
        cirq.SWAP(q0, q2), cirq.H(q0),
        (cirq.CZ ** -0.5)(q1, q0), cirq.H(q1),
        (cirq.CZ ** -0.25)(q2, q0),
        (cirq.CZ ** -0.5)(q2, q1), cirq.H(q2),
        cirq.measure(q2, q1, q0, key="result"),
    )
""",
        ),
    ),
)
def test_order_two_qpe_needs_only_one_nonidentity_controlled_power(
    framework: str,
    code: str,
) -> None:
    _assert_passes(framework, "09", "qpe_x_gate", code)


def test_qiskit_gate_built_controlled_hadamard_test_passes() -> None:
    _assert_passes(
        "qiskit",
        "58",
        "hadamard_test_matrix_element",
        """\
from qiskit import QuantumCircuit

def hadamard_test_matrix_element():
    qc = QuantumCircuit(3, 1)
    qc.h(0); qc.h(1)
    unitary = QuantumCircuit(2)
    unitary.cz(0, 1); unitary.h(0); unitary.t(0); unitary.cx(0, 1)
    qc.append(unitary.to_gate().control(), [0, 1, 2])
    qc.h(0); qc.measure(0, 0)
    return qc
""",
    )


def test_qiskit_large_inverse_qft_composite_passes_shor_response() -> None:
    _assert_passes(
        "qiskit",
        "32",
        "shor_7mod15",
        """\
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator
import numpy as np

def shor_7mod15():
    qc = QuantumCircuit(12, 8)
    phase = list(range(8)); target = list(range(8, 12))
    qc.h(phase); qc.x(target[0])
    for k in range(8):
        a = pow(7, 2 ** k, 15)
        if a != 1:
            matrix = np.zeros((16, 16), dtype=complex)
            for x in range(16):
                matrix[15 if x == 15 else (a * x) % 15, x] = 1.0
            qc.append(Operator(matrix).to_instruction().control(), [phase[k]] + target)
    qc.append(QFT(8, inverse=True), phase)
    qc.measure(phase, phase)
    return qc
""",
    )


def test_pennylane_native_pauli_rot_trotterization_passes() -> None:
    _assert_passes(
        "pennylane",
        "54",
        "heisenberg_trotter_second_order",
        """\
import pennylane as qml

def heisenberg_trotter_second_order():
    dev = qml.device("default.qubit", wires=3, shots=None)
    @qml.qnode(dev)
    def circuit():
        qml.PauliX(2)
        for _ in range(2):
            for wires in ([0, 1], [1, 2]):
                qml.PauliRot(0.5, "XX", wires=wires)
                qml.PauliRot(0.4, "YY", wires=wires)
                qml.PauliRot(0.25, "ZZ", wires=wires)
            for wire in range(3):
                qml.PauliRot(0.3, "Z", wires=wire)
            for wires in ([1, 2], [0, 1]):
                qml.PauliRot(0.5, "XX", wires=wires)
                qml.PauliRot(0.4, "YY", wires=wires)
                qml.PauliRot(0.25, "ZZ", wires=wires)
        return qml.probs(wires=[2, 1, 0])
    return circuit()
""",
    )


def test_commuting_qec_encode_decode_order_passes() -> None:
    _assert_passes(
        "qiskit",
        "qec01",
        "bit_flip_encode_decode",
        """\
from qiskit import QuantumCircuit

def bit_flip_encode_decode(logical_bit: int):
    qc = QuantumCircuit(3, 1)
    if logical_bit:
        qc.x(0)
    qc.cx(0, 1); qc.cx(0, 2)
    qc.cx(0, 1); qc.cx(0, 2)
    qc.measure(0, 0)
    return qc
""",
        suite="qec",
    )


def test_u_decomposition_proof_completes_for_equivalent_pi_literal() -> None:
    _assert_passes(
        "qiskit",
        "42",
        "U_gate_decompose",
        """\
from qiskit import QuantumCircuit
import numpy as np

def U_gate_decompose(theta, phi, lam):
    qc = QuantumCircuit(1)
    qc.rz(lam, 0); qc.sx(0); qc.rz(theta + np.pi, 0)
    qc.sx(0); qc.rz(phi + np.pi, 0)
    return qc
""",
    )


def test_constant_zero_deutsch_jozsa_accepts_a_four_qubit_phase_oracle_form() -> None:
    _assert_passes(
        "qiskit",
        "12",
        "Deutsch_Jozsa_Constant_4",
        """\
from qiskit import QuantumCircuit

def Deutsch_Jozsa_Constant_4():
    qc = QuantumCircuit(4, 4)
    qc.h(range(4))
    qc.h(range(4))
    qc.measure(range(4), range(4))
    return qc
""",
    )


def test_qsub_canonical_is_input_independent_without_witness_stripping() -> None:
    task = load_tasks("qiskit", suite="core")["23"]
    source = task["canonical_solution"]
    assert "cir.x(2*i+2)" not in source
    _assert_passes("qiskit", "23", "QSub", source)


def test_period_finding_accepts_inverse_qft_readout() -> None:
    _assert_passes(
        "qiskit",
        "47",
        "period_finding_mod4_phase_kickback",
        """\
from math import pi
from qiskit import QuantumCircuit

def period_finding_mod4_phase_kickback():
    qc = QuantumCircuit(3, 2)
    qc.h(0); qc.h(1)
    qc.x(2); qc.h(2)
    qc.cx(0, 2)
    qc.swap(0, 1)
    qc.h(0)
    qc.cp(-pi / 2, 0, 1)
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc
""",
    )


def test_first_order_trotter_accepts_reverse_factor_order() -> None:
    _assert_passes(
        "qiskit",
        "49",
        "ising_trotter_evolution",
        """\
from qiskit import QuantumCircuit

def ising_trotter_evolution():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    for _ in range(2):
        qc.rx(0.24, 1)
        qc.rx(0.32, 0)
        qc.rzz(0.56, 0, 1)
    qc.measure([0, 1], [0, 1])
    return qc
""",
    )
