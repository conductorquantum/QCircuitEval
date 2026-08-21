"""Regression tests for demonstrated parser and shortcut-gaming attacks."""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks

_CUSTOM_GATE_SPOOFS = {
    "qiskit": """\
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

def Bell_State():
    fake = Gate("cx", 2, [])
    definition = QuantumCircuit(2)
    definition.cz(0, 1)
    fake.definition = definition
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.append(fake, [0, 1])
    circuit.measure([0, 1], [0, 1])
    return circuit
""",
    "cirq": """\
import cirq
import numpy as np

class FakeCX(cirq.Gate):
    def _num_qubits_(self):
        return 2
    def _unitary_(self):
        return np.diag([1, 1, 1, -1])
    def __str__(self):
        return "cx"

def Bell_State():
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q[0]), FakeCX().on(*q), cirq.measure(q[1], q[0], key="result"))
""",
    "pennylane": """\
import pennylane as qml
import numpy as np

class CNOT(qml.operation.Operation):
    num_wires = 2
    num_params = 0
    @staticmethod
    def compute_matrix():
        return np.diag([1, 1, 1, -1])

def Bell_State():
    dev = qml.device("default.qubit", wires=2, shots=None)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        CNOT(wires=[0, 1])
        return qml.probs(wires=[1, 0])
    return circuit()
""",
}

_DENSE_UNITARY_LAUNDERING = {
    "qiskit": """\
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

def Bell_State():
    recipe = QuantumCircuit(2)
    recipe.h(0)
    recipe.cx(0, 1)
    circuit = QuantumCircuit(2, 2)
    circuit.append(UnitaryGate(Operator(recipe).data), [0, 1])
    circuit.measure([0, 1], [0, 1])
    return circuit
""",
    "cirq": """\
import cirq

def Bell_State():
    q = cirq.LineQubit.range(2)
    recipe = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))
    matrix = cirq.unitary(recipe)
    return cirq.Circuit(
        cirq.MatrixGate(matrix).on(*q),
        cirq.measure(q[1], q[0], key="result"),
    )
""",
    "pennylane": """\
import pennylane as qml
import numpy as np

def Bell_State():
    dev = qml.device("default.qubit", wires=2, shots=None)
    matrix = qml.matrix(
        qml.prod(qml.CNOT(wires=[0, 1]), qml.Hadamard(wires=0))
    )
    @qml.qnode(dev)
    def circuit():
        qml.QubitUnitary(np.asarray(matrix), wires=[0, 1])
        return qml.probs(wires=[1, 0])
    return circuit()
""",
}


@pytest.mark.parametrize("framework", tuple(_CUSTOM_GATE_SPOOFS))
def test_custom_gate_names_cannot_spoof_builtin_semantics(framework: str) -> None:
    _, details = build_evaluator(framework, suite="core").grade_code(
        task_id="16",
        code=_CUSTOM_GATE_SPOOFS[framework],
        entry_point="Bell_State",
    )

    assert details["passed"] is False


@pytest.mark.parametrize("framework", tuple(_DENSE_UNITARY_LAUNDERING))
def test_full_register_dense_unitary_cannot_launder_complete_answer(framework: str) -> None:
    _, details = build_evaluator(framework, suite="core").grade_code(
        task_id="16",
        code=_DENSE_UNITARY_LAUNDERING[framework],
        entry_point="Bell_State",
    )

    assert details["passed"] is False
    assert details["reason"] == "requirement_failed:forbid_full_register_dense_unitary"


def test_qec_controlled_x_name_spoof_is_rejected() -> None:
    code = """\
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

def bit_flip_encode_decode(logical_bit: int):
    fake = Gate("cx", 2, [])
    definition = QuantumCircuit(2)
    definition.cz(0, 1)
    fake.definition = definition
    circuit = QuantumCircuit(3, 1)
    if logical_bit:
        circuit.x(0)
    for target in (1, 2, 2, 1):
        circuit.append(fake, [0, target])
    circuit.measure(0, 0)
    return circuit
"""
    _, details = build_evaluator("qiskit", suite="qec").grade_code(
        task_id="qec01",
        code=code,
        entry_point="bit_flip_encode_decode",
    )

    assert details["passed"] is False


def test_qec_canceling_error_padding_cannot_fake_argument_dependency() -> None:
    task = load_tasks("qiskit", suite="qec")["qec03"]
    code = task["canonical_solution"].replace(
        "        qc.x(error_qubit)\n",
        "        qc.x(error_qubit)\n        qc.x(error_qubit)\n",
    )
    _, details = build_evaluator("qiskit", suite="qec").grade_code(
        task_id="qec03",
        code=code,
        entry_point="bit_flip_correct",
    )

    assert details["passed"] is False
    assert details["reason"] == "requirement_failed:case_program_invariance"


def test_qec12_direct_codeword_initialization_violates_fixed_encoding_convention() -> None:
    code = """\
from qiskit import QuantumCircuit

def rep5_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(9, 1)
    if logical_bit:
        for wire in range(5):
            qc.x(wire)
    if error_qubit is not None:
        qc.x(error_qubit)
    for left, right, ancilla in ((0, 1, 5), (1, 2, 6), (2, 3, 7), (3, 4, 8)):
        qc.cx(left, ancilla); qc.cx(right, ancilla)
    controls = [5, 6, 7, 8]
    for bits, target in (((1, 0, 0, 0), 0), ((1, 1, 0, 0), 1),
                         ((0, 1, 1, 0), 2), ((0, 0, 1, 1), 3),
                         ((0, 0, 0, 1), 4)):
        for index, bit in enumerate(bits):
            if bit == 0:
                qc.x(controls[index])
        qc.mcx(controls, target)
        for index, bit in enumerate(bits):
            if bit == 0:
                qc.x(controls[index])
    qc.measure(0, 0)
    return qc
"""
    _, details = build_evaluator("qiskit", suite="qec").grade_code(
        task_id="qec12",
        code=code,
        entry_point="rep5_correct",
    )

    assert details["passed"] is False
    assert details["reason"] == "requirement_failed:case_program_invariance"


def test_decomposed_state_preparation_cannot_launder_provenance() -> None:
    code = """\
from math import sqrt
from qiskit import QuantumCircuit

def shor_encode_zero():
    amplitudes = [0j] * 512
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                index = (7 * a) | (56 * b) | (448 * c)
                amplitudes[index] = 1 / sqrt(8)
    circuit = QuantumCircuit(9, 9)
    circuit.prepare_state(amplitudes, range(9))
    circuit = circuit.decompose(reps=20)
    circuit.measure(range(9), range(9))
    return circuit
"""
    _, details = build_evaluator("qiskit", suite="qec").grade_code(
        task_id="qec06",
        code=code,
        entry_point="shor_encode_zero",
    )

    assert details["passed"] is False
    assert details["reason"] == "requirement_failed:forbid_state_preparation_source"


def test_obfuscated_decomposed_state_preparation_cannot_launder_provenance() -> None:
    code = """\
from math import sqrt
from qiskit import QuantumCircuit

def Bell_State():
    circuit = QuantumCircuit(2, 2)
    amplitudes = [1 / sqrt(2), 0, 0, 1 / sqrt(2)]
    builder = getattr(circuit, "".join(("prepare", "_state")))
    builder(amplitudes, [0, 1])
    circuit = circuit.decompose(reps=20)
    circuit.measure([0, 1], [0, 1])
    return circuit
"""
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id="16",
        code=code,
        entry_point="Bell_State",
    )

    assert details["passed"] is False
    assert details["reason"] == "requirement_failed:forbid_state_preparation_source"


@pytest.mark.parametrize(
    ("task_id", "entry_point", "body"),
    (
        (
            "11",
            "Deutsch_Jozsa_Balance_4",
            "circuit.x(2); circuit.x(3); circuit.cx(0, 4)",
        ),
        (
            "12",
            "Deutsch_Jozsa_Constant_4",
            "circuit.h(4); circuit.h(4)",
        ),
        (
            "20",
            "Bernstein_Vazirani_011",
            "circuit.x(0); circuit.x(1); circuit.cx(2, 3)",
        ),
    ),
)
def test_direct_output_synthesis_cannot_impersonate_query_algorithm(
    task_id: str,
    entry_point: str,
    body: str,
) -> None:
    output_count = 4 if task_id in {"11", "12"} else 3
    qubit_count = output_count + 1
    code = f"""\
from qiskit import QuantumCircuit

def {entry_point}():
    circuit = QuantumCircuit({qubit_count}, {output_count})
    {body}
    circuit.measure(range({output_count}), range({output_count}))
    return circuit
"""
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id=task_id,
        code=code,
        entry_point=entry_point,
    )

    assert details["passed"] is False
    assert str(details["reason"]).startswith("requirement_failed:")


def test_hardened_query_algorithm_canonicals_still_pass() -> None:
    for framework in ("qiskit", "cirq", "pennylane", "cudaq"):
        tasks = load_tasks(framework, suite="core")
        for task_id in ("11", "12", "20"):
            task = tasks[task_id]
            _, details = build_evaluator(framework, suite="core").grade_code(
                task_id=task_id,
                code=task["canonical_solution"],
                entry_point=task["entry_point"],
            )
            assert details["passed"] is True, (framework, task_id, details.get("reason"))


_ORDINARY_GATE_ALGORITHM_SHORTCUTS = {
    "13": (
        "Simon_11",
        """\
from qiskit import QuantumCircuit

def Simon_11():
    circuit = QuantumCircuit(4, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit
""",
        "requirement_failed:query_hadamard_layers",
    ),
    "14": (
        "Simon_110",
        """\
from qiskit import QuantumCircuit

def Simon_110():
    circuit = QuantumCircuit(6, 3)
    circuit.h(0)
    circuit.h(1)
    circuit.cx(1, 2)
    circuit.measure(range(3), range(3))
    return circuit
""",
        "requirement_failed:query_hadamard_layers",
    ),
    "32": (
        "shor_7mod15",
        """\
from qiskit import QuantumCircuit

def shor_7mod15():
    circuit = QuantumCircuit(12, 8)
    circuit.h(6)
    circuit.h(7)
    circuit.cx(0, 8)
    circuit.cx(0, 8)
    circuit.measure(range(8), range(8))
    return circuit
""",
        "requirement_failed:controlled_evolution_register",
    ),
    "53": (
        "hhl_2x2_diagonal_system",
        """\
from math import atan
from qiskit import QuantumCircuit

def hhl_2x2_diagonal_system():
    circuit = QuantumCircuit(4, 2)
    circuit.ry(2 * atan(1 / 2), 2)
    circuit.x(3)
    circuit.cx(0, 2)
    circuit.cx(1, 2)
    circuit.measure(2, 0)
    circuit.measure(3, 1)
    return circuit
""",
        "requirement_failed:hhl_clock_transforms",
    ),
}


@pytest.mark.parametrize("task_id", tuple(_ORDINARY_GATE_ALGORITHM_SHORTCUTS))
def test_ordinary_gate_output_synthesis_cannot_impersonate_algorithm(task_id: str) -> None:
    entry_point, code, reason = _ORDINARY_GATE_ALGORITHM_SHORTCUTS[task_id]
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id=task_id,
        code=code,
        entry_point=entry_point,
    )

    assert details["passed"] is False
    assert details["reason"] == reason


_DIRECT_PREPARATION_ALGORITHM_SHORTCUTS = {
    "01": (
        "grover_search_oracle_00",
        """\
from qiskit import QuantumCircuit

def grover_search_oracle_00():
    circuit = QuantumCircuit(2, 2)
    circuit.measure([0, 1], [0, 1])
    return circuit
""",
        "requirement_failed:min_non_measurement_operation_count",
    ),
    "03": (
        "grover_3SAT",
        """\
from qiskit import QuantumCircuit

def grover_3SAT():
    circuit = QuantumCircuit(3, 3)
    circuit.x(1)
    circuit.h(2)
    circuit.measure([0, 1, 2], [0, 1, 2])
    return circuit
""",
        "requirement_failed:min_entangling_gate_count",
    ),
    "05": (
        "grover_knapsack",
        """\
from qiskit import QuantumCircuit

def grover_knapsack():
    circuit = QuantumCircuit(5, 5)
    circuit.x(0)
    circuit.x(4)
    circuit.measure(range(5), range(5))
    return circuit
""",
        "requirement_failed:min_entangling_gate_count",
    ),
    "18": (
        "Quantum_Teleportation",
        """\
from numpy import pi
from qiskit import QuantumCircuit

def Quantum_Teleportation():
    circuit = QuantumCircuit(3, 1)
    circuit.rx(pi / 2, 0)
    circuit.swap(0, 2)
    circuit.measure(2, 0)
    return circuit
""",
        "requirement_failed:teleportation_bell_pair",
    ),
    "45": (
        "amplitude_amplification_two_marked",
        """\
from qiskit import QuantumCircuit

def amplitude_amplification_two_marked():
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.x(1)
    circuit.cx(0, 1)
    circuit.measure(range(3), range(3))
    return circuit
""",
        "requirement_failed:grover_hadamard_layers",
    ),
}


@pytest.mark.parametrize("task_id", tuple(_DIRECT_PREPARATION_ALGORITHM_SHORTCUTS))
def test_direct_preparation_cannot_impersonate_named_algorithm(task_id: str) -> None:
    """The trivial output-synthesis solutions from the assets review must fail."""
    entry_point, code, reason = _DIRECT_PREPARATION_ALGORITHM_SHORTCUTS[task_id]
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id=task_id,
        code=code,
        entry_point=entry_point,
    )

    assert details["passed"] is False
    assert details["reason"] == reason


def test_grover_identity_padding_cannot_fake_hadamard_layers() -> None:
    """Adjacent self-canceling Hadamard padding does not satisfy the recipe."""
    code = """\
from qiskit import QuantumCircuit

def grover_search_oracle_00():
    circuit = QuantumCircuit(2, 2)
    for _ in range(3):
        circuit.h(0); circuit.h(0)
        circuit.h(1); circuit.h(1)
    circuit.cz(0, 1); circuit.cz(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit
"""
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id="01",
        code=code,
        entry_point="grover_search_oracle_00",
    )

    assert details["passed"] is False
    # The structural net-unitary check rejects the fully canceling circuit
    # first; the Grover recipe check is the backstop when structure passes.
    assert details["reason"] in {
        "requirement_failed:net_unitary_nonlocal",
        "requirement_failed:grover_hadamard_layers",
    }


_STATE_TRIVIAL_GROVER_PADDING = {
    "01": (
        "grover_search_oracle_00",
        """\
from numpy import pi
from qiskit import QuantumCircuit

def grover_search_oracle_00():
    circuit = QuantumCircuit(2, 2)
    for _ in range(4):
        circuit.h(0); circuit.h(1)
    circuit.cz(0, 1)
    circuit.cp(pi / 2, 0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit
""",
    ),
    "03": (
        "grover_3SAT",
        """\
from qiskit import QuantumCircuit

def grover_3SAT():
    circuit = QuantumCircuit(3, 3)
    for _ in range(4):
        circuit.h(0); circuit.h(1); circuit.h(2)
    circuit.x(1); circuit.h(2)
    circuit.cz(0, 1)
    circuit.cp(1.0, 0, 2)
    circuit.measure(range(3), range(3))
    return circuit
""",
    ),
    "05": (
        "grover_knapsack",
        """\
from qiskit import QuantumCircuit

def grover_knapsack():
    circuit = QuantumCircuit(5, 5)
    for _ in range(4):
        for wire in range(5):
            circuit.h(wire)
    circuit.x(0); circuit.x(4)
    for _ in range(5):
        circuit.cp(0.5, 1, 2)
        circuit.cp(0.5, 2, 3)
    circuit.measure(range(5), range(5))
    return circuit
""",
    ),
    "45": (
        "amplitude_amplification_two_marked",
        """\
from qiskit import QuantumCircuit

def amplitude_amplification_two_marked():
    circuit = QuantumCircuit(3, 3)
    for _ in range(4):
        for wire in range(3):
            circuit.h(wire)
    circuit.h(0); circuit.cx(0, 2)
    circuit.x(1); circuit.cx(0, 1)
    circuit.cz(0, 1); circuit.z(0); circuit.z(1)
    circuit.cz(1, 2); circuit.z(1); circuit.z(2)
    circuit.measure(range(3), range(3))
    return circuit
""",
    ),
}


@pytest.mark.parametrize("task_id", tuple(_STATE_TRIVIAL_GROVER_PADDING))
def test_state_trivial_padding_cannot_fake_grover_reflections(task_id: str) -> None:
    entry_point, code = _STATE_TRIVIAL_GROVER_PADDING[task_id]
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id=task_id,
        code=code,
        entry_point=entry_point,
    )

    assert details["passed"] is False
    assert str(details["reason"]).startswith("requirement_failed:")


def test_grover_oracle_ancillas_must_be_uncomputed() -> None:
    code = """\
from qiskit import QuantumCircuit

def grover_3SAT():
    circuit = QuantumCircuit(4, 3)
    circuit.h([0, 1, 2])
    circuit.x(0)
    circuit.cz(0, 1)
    circuit.x(0)
    circuit.h([0, 1, 2])
    circuit.x([0, 1, 2])
    circuit.h(2)
    circuit.ccx(0, 1, 2)
    circuit.h(2)
    circuit.x([0, 1, 2])
    circuit.h([0, 1, 2])
    circuit.x(3)
    circuit.measure([0, 1, 2], [0, 1, 2])
    return circuit
"""
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id="03",
        code=code,
        entry_point="grover_3SAT",
    )

    assert details["passed"] is False
    assert details["reason"] == "requirement_failed:grover_ancillas_not_restored"


def test_identity_gadgets_cannot_hide_direct_teleportation_swap() -> None:
    code = """\
from numpy import pi
from qiskit import QuantumCircuit

def Quantum_Teleportation():
    circuit = QuantumCircuit(3, 3)
    circuit.rx(pi / 2, 0)
    circuit.h(1)
    circuit.cx(1, 2); circuit.x(2); circuit.cx(1, 2); circuit.x(2)
    circuit.h(1)
    circuit.cz(0, 1); circuit.x(0); circuit.cz(0, 1); circuit.x(0)
    circuit.z(1); circuit.z(1)
    circuit.cx(1, 2); circuit.z(1); circuit.cx(1, 2); circuit.z(1)
    circuit.cz(0, 2); circuit.z(0); circuit.cz(0, 2); circuit.z(0)
    circuit.swap(0, 2)
    circuit.measure(2, 2)
    return circuit
"""
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id="18",
        code=code,
        entry_point="Quantum_Teleportation",
    )

    assert details["passed"] is False
    assert str(details["reason"]).startswith("requirement_failed:teleportation_")


def test_nested_unitary_gate_cannot_launder_qft_matrix() -> None:
    code = """\
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Operator

class Laundered(Gate):
    def __init__(self, matrix):
        super().__init__("laundered", 6, [])
        self._matrix = matrix

    def _define(self):
        circuit = QuantumCircuit(6)
        circuit.append(UnitaryGate(self._matrix), range(6))
        self.definition = circuit

def qft_6():
    circuit = QuantumCircuit(6)
    circuit.append(Laundered(Operator(QFT(6)).data), range(6))
    return circuit
"""
    _, details = build_evaluator("qiskit", suite="core").grade_code(
        task_id="08",
        code=code,
        entry_point="qft_6",
    )

    assert details["passed"] is False
    assert details["reason"] == "requirement_failed:forbid_full_register_dense_unitary"


_QEC_CASE_TABLE_SHORTCUTS = {
    "qec02": (
        "bit_flip_syndrome",
        """\
from qiskit import QuantumCircuit

def bit_flip_syndrome(error_qubit):
    circuit = QuantumCircuit(5, 2)
    if error_qubit is not None:
        circuit.x(error_qubit)
    pairs = [(0, 3), (1, 3), (1, 4), (2, 4)]
    for control, target in pairs + pairs:
        circuit.cx(control, target)
    if error_qubit == 0:
        circuit.x(3)
    elif error_qubit == 1:
        circuit.x(3); circuit.x(4)
    elif error_qubit == 2:
        circuit.x(4)
    circuit.measure(3, 0)
    circuit.measure(4, 1)
    return circuit
""",
    ),
    "qec03": (
        "bit_flip_correct",
        """\
from qiskit import QuantumCircuit

def bit_flip_correct(logical_bit, error_qubit):
    circuit = QuantumCircuit(3, 1)
    if logical_bit:
        circuit.x(0)
    if error_qubit is not None:
        circuit.x(error_qubit)
    pairs = [(0, 1), (0, 2)]
    for control, target in pairs + pairs:
        circuit.cx(control, target)
    circuit.ccx(1, 2, 0)
    if error_qubit is not None:
        circuit.x(error_qubit)
    circuit.measure(0, 0)
    return circuit
""",
    ),
    "qec05": (
        "repetition_logical_cnot",
        """\
from qiskit import QuantumCircuit

def repetition_logical_cnot(control_bit, target_bit):
    circuit = QuantumCircuit(6, 2)
    if control_bit:
        circuit.x(0)
    if control_bit ^ target_bit:
        circuit.x(3)
    pairs = [(0, 3), (1, 4), (2, 5)]
    for control, target in pairs + pairs:
        circuit.cx(control, target)
    circuit.measure(0, 0)
    circuit.measure(3, 1)
    return circuit
""",
    ),
}


@pytest.mark.parametrize("task_id", tuple(_QEC_CASE_TABLE_SHORTCUTS))
def test_qec_case_table_answer_synthesis_is_rejected(task_id: str) -> None:
    entry_point, code = _QEC_CASE_TABLE_SHORTCUTS[task_id]
    _, details = build_evaluator("qiskit", suite="qec").grade_code(
        task_id=task_id,
        code=code,
        entry_point=entry_point,
    )

    assert details["passed"] is False
    assert details["reason"] == "requirement_failed:case_program_invariance"
