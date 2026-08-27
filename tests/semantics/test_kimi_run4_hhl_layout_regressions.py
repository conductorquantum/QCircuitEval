"""Regressions for kimi-run4 core task 10 (HHL_4x4) register-layout grading.

The task 10 prompts pin register roles (three clock qubits, two solution
qubits, one success ancilla) and the measurement/render conventions, but not
physical qubit placement. Grading must therefore be layout-independent: a
correct solution using a non-canonical placement passes, while behaviorally
wrong candidates keep failing without the misleading layout-presuming
``terminal_observation_mismatch`` diagnostic.
"""

from __future__ import annotations

import json

import pytest

from qceval.assets._resources import task_resource
from qceval.evals.evaluator import build_evaluator

AUDITED_QISKIT = """
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import QFT
from qiskit.circuit.library import RYGate
import math


def _rzz(qc, theta, q0, q1):
    qc.cx(q1, q0)
    qc.rz(theta, q0)
    qc.cx(q1, q0)


def _crzz(qc, theta, c, q0, q1):
    qc.cx(q1, q0)
    qc.rz(theta / 2, q0)
    qc.cx(c, q0)
    qc.rz(-theta / 2, q0)
    qc.cx(c, q0)
    qc.cx(q1, q0)


def _crxx(qc, theta, c, q0, q1):
    qc.h(q0)
    qc.h(q1)
    _crzz(qc, theta, c, q0, q1)
    qc.h(q0)
    qc.h(q1)


def _cu_A(qc, tau, c, q0, q1):
    qc.p(2.5 * tau, c)
    _crzz(qc, 2 * tau, c, q0, q1)
    _crxx(qc, tau, c, q0, q1)


def _cu_A_inv(qc, tau, c, q0, q1):
    qc.p(-2.5 * tau, c)
    _crzz(qc, -2 * tau, c, q0, q1)
    _crxx(qc, -tau, c, q0, q1)


def _qft_inv(qc, q):
    qc.h(q[2])
    qc.cp(-np.pi / 2, q[2], q[1])
    qc.h(q[1])
    qc.cp(-np.pi / 4, q[2], q[0])
    qc.cp(-np.pi / 2, q[1], q[0])
    qc.h(q[0])


def _qft(qc, q):
    qc.h(q[0])
    qc.cp(np.pi / 2, q[1], q[0])
    qc.cp(np.pi / 4, q[2], q[0])
    qc.h(q[1])
    qc.cp(np.pi / 2, q[2], q[1])
    qc.h(q[2])


def _ucry(qc, angles, controls, target):
    n = len(controls)
    if n == 0:
        qc.ry(angles[0], target)
        return
    if n == 1:
        a = (angles[0] + angles[1]) / 2.0
        b = (angles[0] - angles[1]) / 2.0
        qc.ry(a, target)
        qc.cx(controls[0], target)
        qc.ry(b, target)
        qc.cx(controls[0], target)
        return
    half = 1 << (n - 1)
    avg = [(angles[i] + angles[i + half]) / 2.0 for i in range(half)]
    diff = [(angles[i] - angles[i + half]) / 2.0 for i in range(half)]
    _ucry(qc, avg, controls[:-1], target)
    qc.cx(controls[-1], target)
    _ucry(qc, diff, controls[:-1], target)
    qc.cx(controls[-1], target)


def HHL_4x4() -> QuantumCircuit:
    sol = QuantumRegister(2, 'sol')
    clock = QuantumRegister(3, 'clock')
    anc = QuantumRegister(1, 'anc')
    c = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(sol, clock, anc, c)

    # Prepare |b> = (|00> + |01>)/sqrt(2)
    qc.h(sol[0])

    # Phase estimation
    for k in range(3):
        qc.h(clock[k])

    t = np.pi / 4.0
    for k in range(3):
        tk = t * (2 ** k)
        _cu_A(qc, tk, clock[k], sol[0], sol[1])

    _qft_inv(qc, [clock[0], clock[1], clock[2]])

    # Controlled rotation by 1/lambda
    C = 1.0
    angles = [0.0] * 8
    for lam in (1, 2, 3, 4):
        angles[lam] = 2.0 * math.asin(C / lam)
    _ucry(qc, angles, [clock[0], clock[1], clock[2]], anc[0])

    # Uncompute phase estimation
    _qft(qc, [clock[0], clock[1], clock[2]])
    for k in range(3):
        tk = t * (2 ** k)
        _cu_A_inv(qc, tk, clock[k], sol[0], sol[1])
    for k in range(3):
        qc.h(clock[k])

    # Measure solution qubits and HHL success ancilla
    qc.measure([sol[0], sol[1], anc[0]], [c[0], c[1], c[2]])
    return qc
"""

AUDITED_CIRQ = """
import cirq
import numpy as np
from scipy.linalg import expm
import math

def HHL_4x4():
    q0, q1 = cirq.LineQubit.range(2)
    c0, c1, c2 = cirq.LineQubit.range(2, 5)
    anc = cirq.LineQubit(5)
    t = math.pi / 4.0

    def controlled_U(tau, ctrl):
        ops = []
        ops.append(cirq.Rz(rads=-2.5 * tau)(ctrl))

        ops.append(cirq.CNOT(q0, q1))
        ops.append(cirq.Rz(rads=-tau)(q0))
        ops.append(cirq.CNOT(ctrl, q0))
        ops.append(cirq.Rz(rads=tau)(q0))
        ops.append(cirq.CNOT(ctrl, q0))
        ops.append(cirq.CNOT(q0, q1))

        ops.append(cirq.H(q0))
        ops.append(cirq.H(q1))
        ops.append(cirq.CNOT(q0, q1))
        ops.append(cirq.Rz(rads=-tau / 2)(q0))
        ops.append(cirq.CNOT(ctrl, q0))
        ops.append(cirq.Rz(rads=tau / 2)(q0))
        ops.append(cirq.CNOT(ctrl, q0))
        ops.append(cirq.CNOT(q0, q1))
        ops.append(cirq.H(q0))
        ops.append(cirq.H(q1))
        return ops

    def qpe():
        ops = []
        ops.append([cirq.H(c) for c in (c0, c1, c2)])
        for k, c in enumerate((c0, c1, c2)):
            ops.extend(controlled_U(-t * (2 ** k), c))
        ops.append(cirq.H(c0))
        ops.append(cirq.CZPowGate(exponent=-0.5)(c0, c1))
        ops.append(cirq.H(c1))
        ops.append(cirq.CZPowGate(exponent=-0.25)(c0, c2))
        ops.append(cirq.CZPowGate(exponent=-0.5)(c1, c2))
        ops.append(cirq.H(c2))
        return cirq.Circuit(ops)

    def rotation_block():
        n = 3
        size = 1 << n
        thetas = [0.0] * size
        thetas[1] = math.pi
        thetas[2] = 2.0 * math.asin(1.0 / 2.0)
        thetas[3] = 2.0 * math.asin(1.0 / 3.0)
        thetas[4] = 2.0 * math.asin(1.0 / 4.0)

        gammas = [0.0] * size
        for mask in range(size):
            s = 0.0
            for state in range(size):
                parity = bin(state & mask).count('1')
                sign = -1 if (parity & 1) else 1
                s += thetas[state] * sign
            gammas[mask] = s / size

        gray = [i ^ (i >> 1) for i in range(size)]
        alphas = [gammas[g] for g in gray]
        clock = (c0, c1, c2)

        ops = []
        for j in range(size - 1):
            if abs(alphas[j]) > 1e-12:
                ops.append(cirq.Ry(rads=alphas[j])(anc))
            diff = gray[j + 1] ^ gray[j]
            bit = diff.bit_length() - 1
            ops.append(cirq.CNOT(clock[bit], anc))
        if abs(alphas[-1]) > 1e-12:
            ops.append(cirq.Ry(rads=alphas[-1])(anc))
        return cirq.Circuit(ops)

    circuit = cirq.Circuit()
    circuit.append(cirq.H(q0))

    qpe_circ = qpe()
    circuit += qpe_circ
    circuit += rotation_block()
    circuit += cirq.inverse(qpe_circ)

    circuit.append(cirq.measure(q1, q0, anc, key='result'))
    return circuit
"""

PERMUTED_QISKIT = """
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np
from scipy.linalg import expm
from qiskit.circuit.library import UnitaryGate, QFT, RYGate
import math


def HHL_4x4() -> QuantumCircuit:
    b = QuantumRegister(2, name="b")
    clock = QuantumRegister(3, name="clock")
    ancilla = QuantumRegister(1, name="ancilla")
    classical_ancilla = ClassicalRegister(1, name="c_anc")
    classical_results = ClassicalRegister(2, name="c_res")
    circuit = QuantumCircuit(b, clock, ancilla, classical_ancilla, classical_results)
    circuit.h(b[0])
    circuit.h(clock)
    A = (1 / 2) * np.array([[3, 0, 0, -1], [0, 7, -1, 0], [0, -1, 7, 0], [-1, 0, 0, 3]])
    t = np.pi / 4
    for idx, c in enumerate(clock):
        circuit.append(UnitaryGate(expm(1j * A * t * (2**idx))).control(num_ctrl_qubits=1), [c, b[0], b[1]])
    circuit.append(QFT(3, inverse=True), clock)
    for value, scale in (("001", 1.0), ("010", 2.0), ("011", 3.0), ("100", 4.0)):
        theta = 2 * math.asin(1 / scale)
        circuit.append(RYGate(theta).control(3, ctrl_state=value), [clock[0], clock[1], clock[2], ancilla[0]])
    circuit.measure(ancilla, classical_ancilla)
    circuit.append(QFT(3, inverse=False), clock)
    for idx in range(3):
        Uin = expm(-1j * t * A * 2 ** (2 - idx))
        circuit.append(UnitaryGate(Uin).control(num_ctrl_qubits=1), [clock[2 - idx], b[0], b[1]])
    circuit.h(clock)
    circuit.measure(b, classical_results)
    return circuit
"""

PERMUTED_CIRQ = """
import cirq
import numpy as np
from scipy.linalg import expm


def HHL_4x4():
    b = [cirq.LineQubit(i) for i in range(2)]
    clock = [cirq.LineQubit(2 + i) for i in range(3)]
    anc = cirq.LineQubit(5)
    circuit = cirq.Circuit()
    A = 0.5 * np.array([[3, 0, 0, -1], [0, 7, -1, 0], [0, -1, 7, 0], [-1, 0, 0, 3]], dtype=complex)
    t = np.pi / 4
    circuit.append(cirq.H(b[0]))
    circuit.append([cirq.H(c) for c in clock])
    for idx in range(3):
        gate = cirq.MatrixGate(expm(1j * A * t * (2 ** idx)), name="U")
        circuit.append(gate.on(b[0], b[1]).controlled_by(clock[idx]))
    circuit.append(cirq.qft(clock[2], clock[1], clock[0], inverse=True, without_reverse=False))
    for k, cs in [(1.0, (1, 0, 0)), (2.0, (0, 1, 0)), (3.0, (1, 1, 0)), (4.0, (0, 0, 1))]:
        circuit.append(
            cirq.ry(2 * np.arcsin(1 / k)).on(anc).controlled_by(clock[0], clock[1], clock[2], control_values=cs)
        )
    circuit.append(cirq.qft(clock[2], clock[1], clock[0], inverse=False, without_reverse=False))
    for idx in range(3):
        gate = cirq.MatrixGate(expm(-1j * A * t * (2 ** (2 - idx))), name="Ui")
        circuit.append(gate.on(b[0], b[1]).controlled_by(clock[2 - idx]))
    circuit.append([cirq.H(c) for c in clock])
    circuit.append(cirq.measure(b[1], b[0], anc, key="result"))
    return circuit
"""

AUDITED = {"qiskit": AUDITED_QISKIT, "cirq": AUDITED_CIRQ}
PERMUTED = {"qiskit": PERMUTED_QISKIT, "cirq": PERMUTED_CIRQ}


def _grade(framework: str, code: str) -> dict:
    _, details = build_evaluator(framework, suite="core").grade_code(task_id="10", code=code, entry_point="HHL_4x4")
    return details


def _canonical_solution(framework: str) -> str:
    for line in task_resource("core", framework).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["task_id"] == "10":
            return record["canonical_solution"]
    raise AssertionError(f"core task 10 is missing from the {framework} assets")


@pytest.mark.parametrize("framework", ["qiskit", "cirq"])
def test_audited_kimi_run4_candidates_still_fail_without_layout_diagnostic(framework: str) -> None:
    details = _grade(framework, AUDITED[framework])
    assert details["passed"] is False
    assert details["reason"] != "terminal_observation_mismatch"


@pytest.mark.parametrize("framework", ["qiskit", "cirq"])
def test_correct_solution_with_non_canonical_layout_passes(framework: str) -> None:
    details = _grade(framework, PERMUTED[framework])
    assert details["passed"] is True, details["reason"]


@pytest.mark.parametrize("framework", ["qiskit", "cirq"])
def test_canonical_solution_still_passes(framework: str) -> None:
    details = _grade(framework, _canonical_solution(framework))
    assert details["passed"] is True, details["reason"]
