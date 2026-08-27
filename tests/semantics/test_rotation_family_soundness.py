"""Regressions for H2: finite-point fallback must not decide quantifier="all".

Core tasks 39-41 promise verification over all real parameters via the
structured rotation-family source-identity proof. A candidate whose
perturbation vanishes at every finite diagnostic point interpolates the
graded executions while being wrong elsewhere, so a decisive AST refutation
(or an unsupported source spelling) must never be upgraded to a pass by
finite-point sampling.
"""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks

# Each perturbation is a polynomial in the first family parameter whose roots
# are exactly the first components of the contract's diagnostic points, so an
# extra ry(g) is the identity at every graded point yet detunes the state for
# all other real parameter values.
_INTERPOLATION_ATTACKS = {
    "39": (
        "quantum_state_preparation",
        """\
from qiskit import QuantumRegister, QuantumCircuit

def quantum_state_preparation(parameters):
    g = (
        parameters[0]
        * (parameters[0] - 3.141592653589793)
        * (parameters[0] - 0.37)
        * (parameters[0] + 1.1)
        * (parameters[0] - 1.5707963267948966)
    )
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    qc.rx(parameters[0], qr[0])
    qc.ry(parameters[1], qr[0])
    qc.ry(g, qr[0])
    return qc
""",
    ),
    "40": (
        "VQE_2",
        """\
from qiskit import QuantumRegister, QuantumCircuit

def VQE_2(parameters):
    g = (
        parameters[0]
        * (parameters[0] - 3.141592653589793)
        * (parameters[0] - 0.37)
        * (parameters[0] - 1.454441043328608)
        * (parameters[0] + 2.0)
    )
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.x(qr[0])
    qc.rz(parameters[0], qr[0])
    qc.rz(parameters[1], qr[0])
    qc.ry(parameters[2], qr[0])
    qc.rz(parameters[3], qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(parameters[4], qr[0])
    qc.rz(parameters[5], qr[1])
    qc.ry(parameters[6], qr[0])
    qc.rz(parameters[7], qr[1])
    qc.ry(g, qr[0])
    return qc
""",
    ),
    "41": (
        "VQE_Z2",
        """\
from qiskit import QuantumRegister, QuantumCircuit

def VQE_Z2(param):
    g = (
        param[0]
        * (param[0] - 0.37)
        * (param[0] - 1.454441043328608)
        * (param[0] + 2.0)
    )
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.rz(param[0], qr[0])
    qc.ry(param[1], qr[0])
    qc.rz(param[2], qr[0])
    qc.rz(param[3], qr[1])
    qc.ry(param[4], qr[1])
    qc.rz(param[5], qr[1])
    qc.ry(g, qr[0])
    return qc
""",
    ),
}


@pytest.mark.parametrize("task_id", sorted(_INTERPOLATION_ATTACKS))
def test_diagnostic_point_interpolation_does_not_pass(task_id: str) -> None:
    entry_point, code = _INTERPOLATION_ATTACKS[task_id]
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id=task_id, code=code, entry_point=entry_point)
    assert details["passed"] is False, (task_id, details["reason"])
    assert details["semantic_status"] == "semantic_fail", (task_id, details["semantic_status"])


@pytest.mark.parametrize("framework", ("qiskit", "cirq", "pennylane", "cudaq"))
@pytest.mark.parametrize("task_id", ("39", "40", "41"))
def test_rotation_family_canonical_solutions_pass(framework: str, task_id: str) -> None:
    evaluator = build_evaluator(framework, suite="core")
    task = load_tasks(framework, suite="core")[task_id]
    _, details = evaluator.grade_code(
        task_id=task_id,
        code=task["canonical_solution"],
        entry_point=task["entry_point"],
    )
    assert details["passed"] is True, (task_id, framework, details["semantic_status"], details["reason"])


def test_task41_interleaved_disjoint_wire_order_passes() -> None:
    """Per-qubit gate order is preserved; strands on disjoint wires commute."""
    code = """\
from qiskit import QuantumRegister, QuantumCircuit

def VQE_Z2(param):
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.rz(param[3], qr[1])
    qc.rz(param[0], qr[0])
    qc.ry(param[4], qr[1])
    qc.ry(param[1], qr[0])
    qc.rz(param[5], qr[1])
    qc.rz(param[2], qr[0])
    return qc
"""
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id="41", code=code, entry_point="VQE_Z2")
    assert details["passed"] is True, (details["semantic_status"], details["reason"])


def test_ast_rejected_wrong_gate_order_stays_failed() -> None:
    """A decisive structured-family refutation is terminal, never resampled."""
    code = """\
from qiskit import QuantumRegister, QuantumCircuit

def quantum_state_preparation(parameters):
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    qc.ry(parameters[1], qr[0])
    qc.rx(parameters[0], qr[0])
    return qc
"""
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id="39", code=code, entry_point="quantum_state_preparation")
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


# Each candidate matches the accepted gate structure exactly but rebinds the
# parameter vector name first, so the gate angles are constants rather than
# the contract's universally quantified parameters. Proving the family from
# the rebound name would be unsound; the prover must fail closed instead.
_REBOUND_VECTOR_ATTACKS = {
    "39": (
        "quantum_state_preparation",
        """\
from qiskit import QuantumRegister, QuantumCircuit

def quantum_state_preparation(parameters):
    parameters = [1.0, 2.0]
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    qc.rx(parameters[0], qr[0])
    qc.ry(parameters[1], qr[0])
    return qc
""",
    ),
    "40": (
        "VQE_2",
        """\
from qiskit import QuantumRegister, QuantumCircuit

def VQE_2(parameters):
    parameters = [0.1] * 8
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.x(qr[0])
    qc.rz(parameters[0], qr[0])
    qc.rz(parameters[1], qr[0])
    qc.ry(parameters[2], qr[0])
    qc.rz(parameters[3], qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(parameters[4], qr[0])
    qc.rz(parameters[5], qr[1])
    qc.ry(parameters[6], qr[0])
    qc.rz(parameters[7], qr[1])
    return qc
""",
    ),
    "41": (
        "VQE_Z2",
        """\
from qiskit import QuantumRegister, QuantumCircuit

def VQE_Z2(param):
    param = [1.0] * 6
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.rz(param[0], qr[0])
    qc.ry(param[1], qr[0])
    qc.rz(param[2], qr[0])
    qc.rz(param[3], qr[1])
    qc.ry(param[4], qr[1])
    qc.rz(param[5], qr[1])
    return qc
""",
    ),
}


@pytest.mark.parametrize("task_id", sorted(_REBOUND_VECTOR_ATTACKS))
def test_rebound_parameter_vector_does_not_prove_family(task_id: str) -> None:
    entry_point, code = _REBOUND_VECTOR_ATTACKS[task_id]
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id=task_id, code=code, entry_point=entry_point)
    assert details["passed"] is False, (task_id, details["reason"])
    assert details["semantic_status"] == "execution_error", (task_id, details["semantic_status"])


def test_element_assignment_to_parameter_vector_fails_closed() -> None:
    code = """\
from qiskit import QuantumRegister, QuantumCircuit

def VQE_Z2(param):
    param[3] = 0.0
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.rz(param[0], qr[0])
    qc.ry(param[1], qr[0])
    qc.rz(param[2], qr[0])
    qc.rz(param[3], qr[1])
    qc.ry(param[4], qr[1])
    qc.rz(param[5], qr[1])
    return qc
"""
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id="41", code=code, entry_point="VQE_Z2")
    assert details["passed"] is False
    assert details["semantic_status"] == "execution_error"


def test_element_assignment_to_parameter_alias_fails_closed() -> None:
    code = """\
from qiskit import QuantumCircuit

def quantum_state_preparation(parameters):
    angles = [float(value) for value in parameters]
    angles[0] = 0.0
    qc = QuantumCircuit(1)
    qc.rx(angles[0], 0)
    qc.ry(angles[1], 0)
    return qc
"""
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id="39", code=code, entry_point="quantum_state_preparation")
    assert details["passed"] is False
    assert details["semantic_status"] == "execution_error"


def test_element_assignment_to_cudaq_kernel_vector_fails_closed() -> None:
    pytest.importorskip("cudaq")
    code = """\
import cudaq

def quantum_state_preparation(parameters):
    @cudaq.kernel
    def kernel(vals: list[float]):
        vals[0] = 0.0
        q = cudaq.qvector(1)
        rx(vals[0], q[0])
        ry(vals[1], q[0])
    return kernel
"""
    evaluator = build_evaluator("cudaq", suite="core")
    _, details = evaluator.grade_code(task_id="39", code=code, entry_point="quantum_state_preparation")
    assert details["passed"] is False
    assert details["semantic_status"] == "execution_error"


def test_flag_guarded_rebinding_is_not_a_none_default() -> None:
    """Only ``if <vector> is None: <vector> = ...`` is a dead-branch default."""
    code = """\
from qiskit import QuantumRegister, QuantumCircuit

def quantum_state_preparation(parameters):
    flag = None
    if flag is None:
        parameters = [1.0, 2.0]
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    qc.rx(parameters[0], qr[0])
    qc.ry(parameters[1], qr[0])
    return qc
"""
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id="39", code=code, entry_point="quantum_state_preparation")
    assert details["passed"] is False
    assert details["semantic_status"] == "execution_error"


def test_none_default_rebinding_of_vector_argument_still_passes() -> None:
    """The grader always binds a real vector, so the default branch is dead."""
    code = """\
from qiskit import QuantumRegister, QuantumCircuit

def quantum_state_preparation(parameters):
    if parameters is None:
        parameters = [0.0, 0.0]
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    qc.rx(parameters[0], qr[0])
    qc.ry(parameters[1], qr[0])
    return qc
"""
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id="39", code=code, entry_point="quantum_state_preparation")
    assert details["passed"] is True, (details["semantic_status"], details["reason"])


def test_unsupported_source_spelling_fails_closed_for_universal_claim() -> None:
    """Control flow the AST prover cannot decide must not fall back to points."""
    code = """\
from qiskit import QuantumRegister, QuantumCircuit

def VQE_Z2(param):
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    for qubit in range(2):
        qc.rz(param[3 * qubit], qr[qubit])
        qc.ry(param[3 * qubit + 1], qr[qubit])
        qc.rz(param[3 * qubit + 2], qr[qubit])
    return qc
"""
    evaluator = build_evaluator("qiskit", suite="core")
    _, details = evaluator.grade_code(task_id="41", code=code, entry_point="VQE_Z2")
    assert details["passed"] is False
    assert details["semantic_status"] == "execution_error"
