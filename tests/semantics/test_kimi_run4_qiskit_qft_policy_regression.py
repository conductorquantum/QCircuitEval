"""Regression: library-shortcut policy must not blanket-block low-level QFT usage (kimi run 4, qiskit core/51)."""

from __future__ import annotations

from qceval.evals.evaluator import build_evaluator

_LIBRARY_QFT_CANDIDATE = """\
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

def quantum_counting_marked_11():
    qc = QuantumCircuit(5, 3)
    qc.h([0, 1, 2])
    qc.h([3, 4])
    grover = QuantumCircuit(2, name="G")
    grover.cz(0, 1)
    grover.h([0, 1])
    grover.x([0, 1])
    grover.cz(0, 1)
    grover.x([0, 1])
    grover.h([0, 1])
    ggate = grover.to_gate()
    for j, ctrl in enumerate([0, 1, 2]):
        for _ in range(2 ** j):
            qc.append(ggate.control(1), [ctrl, 3, 4])
            qc.z(ctrl)
    qc.append(QFT(num_qubits=3, inverse=True), [0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc
"""

_HELPER_NAMED_QFT_CANDIDATE = """\
from math import pi
from qiskit import QuantumCircuit

def quantum_counting_marked_11():
    qc = QuantumCircuit(5, 3)

    def controlled_h(control, target):
        qc.s(target)
        qc.h(target)
        qc.t(target)
        qc.cx(control, target)
        for _ in range(7):
            qc.t(target)
        qc.h(target)
        qc.sdg(target)

    def controlled_cz(control, a, b):
        qc.h(b)
        qc.ccx(control, a, b)
        qc.h(b)

    def controlled_grover(control):
        controlled_cz(control, 3, 4)
        for target in (3, 4):
            controlled_h(control, target)
        qc.cx(control, 3)
        qc.cx(control, 4)
        controlled_cz(control, 3, 4)
        qc.cx(control, 3)
        qc.cx(control, 4)
        for target in (3, 4):
            controlled_h(control, target)

    def qft(qubits):
        n = len(qubits)
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
        for i in range(n):
            for j in range(i):
                qc.cp(-pi / (2 ** (i - j)), qubits[j], qubits[i])
            qc.h(qubits[i])

    for qubit in range(5):
        qc.h(qubit)
    for control in range(3):
        for _ in range(2 ** control):
            controlled_grover(control)
            qc.z(control)
    qft([0, 1, 2])
    for qubit in range(3):
        qc.measure(qubit, qubit)
    return qc
"""

_PHASE_ESTIMATION_IMPORT_CANDIDATE = """\
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseEstimation

def quantum_counting_marked_11():
    qc = QuantumCircuit(5, 3)
    grover = QuantumCircuit(2, name="G")
    grover.cz(0, 1)
    grover.h([0, 1])
    grover.x([0, 1])
    grover.cz(0, 1)
    grover.x([0, 1])
    grover.h([0, 1])
    qc.h([3, 4])
    qc.append(PhaseEstimation(3, grover), [0, 1, 2, 3, 4])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc
"""

_PHASE_ESTIMATION_ATTRIBUTE_CANDIDATE = """\
import qiskit.circuit.library as library
from qiskit import QuantumCircuit

def quantum_counting_marked_11():
    qc = QuantumCircuit(5, 3)
    grover = QuantumCircuit(2, name="G")
    grover.cz(0, 1)
    grover.h([0, 1])
    grover.x([0, 1])
    grover.cz(0, 1)
    grover.x([0, 1])
    grover.h([0, 1])
    qc.h([3, 4])
    qc.append(library.PhaseEstimation(3, grover), [0, 1, 2, 3, 4])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc
"""


def _grade(code: str) -> dict:
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id="51",
        code=code,
        entry_point="quantum_counting_marked_11",
    )
    return details


def test_low_level_library_qft_composition_passes() -> None:
    details = _grade(_LIBRARY_QFT_CANDIDATE)
    assert details["passed"] is True, details.get("reason")


def test_candidate_defined_helper_named_qft_passes() -> None:
    details = _grade(_HELPER_NAMED_QFT_CANDIDATE)
    assert details["passed"] is True, details.get("reason")


def test_phase_estimation_library_import_still_fails() -> None:
    details = _grade(_PHASE_ESTIMATION_IMPORT_CANDIDATE)
    assert details["passed"] is False
    assert details.get("reason") == "requirement_failed:forbidden_imports"


def test_phase_estimation_attribute_shortcut_still_fails() -> None:
    details = _grade(_PHASE_ESTIMATION_ATTRIBUTE_CANDIDATE)
    assert details["passed"] is False
    # The structural forbidden-import check now resolves module aliases, so it
    # catches ``library.PhaseEstimation`` before the semantic library-shortcut
    # rule is reached.
    assert details.get("reason") == "requirement_failed:forbidden_imports"
