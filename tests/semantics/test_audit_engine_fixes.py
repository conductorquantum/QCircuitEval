"""End-to-end regressions for the grader-audit engine fixes.

Each test grades real candidate source through :func:`build_evaluator`, so it
exercises execution, lowering, requirements, and the semantic portfolio
exactly as production grading does.
"""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks


@pytest.fixture(scope="module")
def qiskit_core():
    return build_evaluator("qiskit", suite="core")


@pytest.fixture(scope="module")
def cirq_core():
    return build_evaluator("cirq", suite="core")


@pytest.fixture(scope="module")
def pennylane_core():
    return build_evaluator("pennylane", suite="core")


@pytest.fixture(scope="module")
def cudaq_core():
    return build_evaluator("cudaq", suite="core")


@pytest.fixture(scope="module")
def qiskit_qec():
    return build_evaluator("qiskit", suite="qec")


def _grade(evaluator, task_id: str, code: str, entry_point: str):
    _, details = evaluator.grade_code(task_id=task_id, code=code, entry_point=entry_point)
    return details


def _evidence_reasons(details) -> str:
    record = details.get("semantic_verification") or {}
    return " ".join(str(item.get("reason_code") or item.get("reason") or "") for item in record.get("evidence") or ())


# ---------------------------------------------------------------------------
# Fix 1: structured-rotation spellings get exact source proofs; bounded
# symbolic families retain their analytic fallback.
# ---------------------------------------------------------------------------


def test_task39_float_cast_comprehension_passes(qiskit_core) -> None:
    code = """
from qiskit import QuantumCircuit
def quantum_state_preparation(parameters):
    angles = [float(x) for x in parameters]
    qc = QuantumCircuit(1)
    qc.rx(angles[0], 0)
    qc.ry(angles[1], 0)
    return qc
"""
    details = _grade(qiskit_core, "39", code, "quantum_state_preparation")
    assert details["passed"] is True, details["reason"]


def test_task39_wrong_gate_order_fails_decisively(qiskit_core) -> None:
    code = """
from qiskit import QuantumCircuit
def quantum_state_preparation(parameters):
    angles = [float(x) for x in parameters]
    qc = QuantumCircuit(1)
    qc.ry(angles[1], 0)
    qc.rx(angles[0], 0)
    return qc
"""
    details = _grade(qiskit_core, "39", code, "quantum_state_preparation")
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"
    assert "structured_rotation_family_mismatch" in _evidence_reasons(details)


def test_task40_input_guard_and_merged_rotations_pass(qiskit_core) -> None:
    code = """
from qiskit import QuantumCircuit
def VQE_2(parameters):
    if len(parameters) != 8:
        raise ValueError("need 8")
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.rz(float(parameters[0]) + float(parameters[1]), 0)
    qc.ry(parameters[2], 0)
    qc.rz(parameters[3], 1)
    qc.cx(0, 1)
    qc.rz(parameters[4], 0)
    qc.rz(parameters[5], 1)
    qc.ry(parameters[6], 0)
    qc.rz(parameters[7], 1)
    return qc
"""
    details = _grade(qiskit_core, "40", code, "VQE_2")
    assert details["passed"] is True, details["reason"]


def test_task42_loop_spelling_passes_and_wrong_family_fails(qiskit_core) -> None:
    correct = """
import numpy as np
from qiskit import QuantumCircuit
def U_gate_decompose(theta, phi, lam):
    qc = QuantumCircuit(1)
    angles = [lam, theta + np.pi, phi + np.pi]
    for i, a in enumerate(angles):
        qc.rz(a, 0)
        if i < 2:
            qc.sx(0)
    return qc
"""
    details = _grade(qiskit_core, "42", correct, "U_gate_decompose")
    assert details["passed"] is True, details["reason"]

    wrong = """
import numpy as np
from qiskit import QuantumCircuit
def U_gate_decompose(theta, phi, lam):
    qc = QuantumCircuit(1)
    for angle in [lam, theta + np.pi, phi]:
        qc.rz(angle, 0)
        qc.sx(0)
    return qc
"""
    details = _grade(qiskit_core, "42", wrong, "U_gate_decompose")
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_task42_forbidden_gate_shortcut_still_fails(qiskit_core) -> None:
    code = """
from qiskit import QuantumCircuit
def U_gate_decompose(theta, phi, lam):
    qc = QuantumCircuit(1)
    qc.u(theta, phi, lam, 0)
    return qc
"""
    details = _grade(qiskit_core, "42", code, "U_gate_decompose")
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_task42_cirq_gate_stored_in_variable_passes(cirq_core) -> None:
    """The stored-gate assignment must not be double-counted as an application."""
    code = """
import cirq
def U_gate_decompose(theta, phi, lam):
    q = cirq.LineQubit(0)
    sx = cirq.XPowGate(exponent=0.5)
    c = cirq.Circuit()
    c.append(cirq.rz(lam).on(q))
    c.append(sx.on(q))
    c.append(cirq.rz(theta + 3.141592653589793).on(q))
    c.append(sx.on(q))
    c.append(cirq.rz(phi + 3.141592653589793).on(q))
    return c
"""
    details = _grade(cirq_core, "42", code, "U_gate_decompose")
    assert details["passed"] is True, details["reason"]


def test_task41_cirq_tuple_unpack_and_on_spelling_passes(cirq_core) -> None:
    code = """
import cirq
def VQE_Z2(param):
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit()
    c.append(cirq.rz(float(param[0])).on(q0))
    c.append(cirq.ry(param[1]).on(q0))
    c.append(cirq.rz(param[2]).on(q0))
    c.append(cirq.rz(param[3]).on(q1))
    c.append(cirq.ry(param[4]).on(q1))
    c.append(cirq.rz(param[5]).on(q1))
    return c
"""
    details = _grade(cirq_core, "41", code, "VQE_Z2")
    assert details["passed"] is True, details["reason"]


def test_task42_pennylane_nested_qnode_any_name_passes(pennylane_core) -> None:
    code = """
import pennylane as qml
from math import pi
def U_gate_decompose(theta, phi, lam):
    theta = 0.0 if theta is None else theta
    phi = 0.0 if phi is None else phi
    lam = 0.0 if lam is None else lam
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def my_qnode():
        qml.RZ(lam, wires=0)
        qml.SX(wires=0)
        qml.RZ(theta + pi, wires=0)
        qml.SX(wires=0)
        qml.RZ(phi + pi, wires=0)
        return qml.probs(wires=[0])
    return my_qnode()
"""
    details = _grade(pennylane_core, "42", code, "U_gate_decompose")
    assert details["passed"] is True, details["reason"]


def test_task42_cudaq_builder_api_passes(cudaq_core) -> None:
    code = """
import cudaq
from math import pi
def U_gate_decompose(theta, phi, lam):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(1)
    kernel.rz(lam, q[0])
    kernel.rx(pi / 2, q[0])
    kernel.rz(theta + pi, q[0])
    kernel.rx(pi / 2, q[0])
    kernel.rz(phi + pi, q[0])
    return kernel
"""
    details = _grade(cudaq_core, "42", code, "U_gate_decompose")
    assert details["passed"] is True, details["reason"]


def test_task39_cudaq_nested_kernel_closure_passes(cudaq_core) -> None:
    code = """
import cudaq
def quantum_state_preparation(parameters):
    a0 = float(parameters[0])
    a1 = float(parameters[1])
    @cudaq.kernel
    def kern():
        q = cudaq.qvector(1)
        rx(a0, q[0])
        ry(a1, q[0])
    return kern
"""
    details = _grade(cudaq_core, "39", code, "quantum_state_preparation")
    assert details["passed"] is True, details["reason"]


def test_task40_cudaq_list_signature_kernel_passes(cudaq_core) -> None:
    code = """
import cudaq
def VQE_2(parameters):
    p = [float(v) for v in parameters]
    @cudaq.kernel
    def kern(vals: list[float]):
        q = cudaq.qvector(2)
        x(q[0])
        rz(vals[0], q[0])
        rz(vals[1], q[0])
        ry(vals[2], q[0])
        rz(vals[3], q[1])
        cx(q[0], q[1])
        rz(vals[4], q[0])
        rz(vals[5], q[1])
        ry(vals[6], q[0])
        rz(vals[7], q[1])
    return kern
"""
    details = _grade(cudaq_core, "40", code, "VQE_2")
    assert details["passed"] is True, details["reason"]


# ---------------------------------------------------------------------------
# Fix 2: grover_reflections accepts prompt-mandated elementary decompositions.
# ---------------------------------------------------------------------------

_DECOMPOSED_GROVER_05 = """
import math
from qiskit import QuantumCircuit

def mcp_vchain(qc, ang, controls, anc):
    qc.ccx(controls[0], controls[1], anc[0])
    for i, c in enumerate(controls[2:]):
        qc.ccx(anc[i], c, anc[i + 1])
    qc.p(ang, anc[len(controls) - 2])
    for i in reversed(range(len(controls) - 2)):
        qc.ccx(anc[i], controls[i + 2], anc[i + 1])
    qc.ccx(controls[0], controls[1], anc[0])

def {entry_point}():
    n = 5
    theta = math.asin(1 / math.sqrt(2 ** n))
    phi = 2 * math.asin(math.sin(math.pi / 26) / math.sin(theta))
    marked = "10001"
    qc = QuantumCircuit(n + 4, n)
    search = list(range(n))
    anc = list(range(n, n + 4))
    zeros = [i for i in range(n) if marked[n - 1 - i] == "0"]
    qc.h(search)
    for _ in range(6):
        for q in zeros:
            qc.x(q)
        mcp_vchain(qc, phi, search, anc)
        for q in zeros:
            qc.x(q)
        qc.h(search)
        for q in search:
            qc.x(q)
        mcp_vchain(qc, phi, search, anc)
        for q in search:
            qc.x(q)
        qc.h(search)
    qc.measure(search, range(n))
    return qc
"""


def test_task05_elementary_decomposed_grover_passes(qiskit_core) -> None:
    """Phase-matched exact search with every multi-controlled phase decomposed
    into Toffoli V-chains plus single-qubit P gates prepares P(10001) = 1 and
    must grade as a pass."""
    task = load_tasks("qiskit", suite="core")["05"]
    code = _DECOMPOSED_GROVER_05.format(entry_point=task["entry_point"])
    details = _grade(qiskit_core, "05", code, task["entry_point"])
    assert details["passed"] is True, (details["reason"], _evidence_reasons(details))


@pytest.mark.parametrize("task_id", ["01", "05", "45"])
def test_grover_canonicals_still_pass(qiskit_core, task_id: str) -> None:
    task = load_tasks("qiskit", suite="core")[task_id]
    details = _grade(qiskit_core, task_id, task["canonical_solution"], task["entry_point"])
    assert details["passed"] is True, details["reason"]


def test_task45_direct_state_preparation_still_fails(qiskit_core) -> None:
    task = load_tasks("qiskit", suite="core")["45"]
    code = f"""
from qiskit import QuantumCircuit
def {task["entry_point"]}():
    qc = QuantumCircuit(4, 4)
    qc.x(0)
    qc.x(1)
    qc.measure(range(4), range(4))
    return qc
"""
    details = _grade(qiskit_core, "45", code, task["entry_point"])
    assert details["passed"] is False


def test_task05_product_phase_fake_grover_fails(qiskit_core) -> None:
    """Per-wire phase segments are product patterns and must not count as
    reflections (nor pass any other requirement)."""
    task = load_tasks("qiskit", suite="core")["05"]
    code = f"""
from qiskit import QuantumCircuit
def {task["entry_point"]}():
    qc = QuantumCircuit(5, 5)
    search = list(range(5))
    qc.h(search)
    for _ in range(12):
        qc.z(0)
        qc.z(4)
        qc.h(search)
    qc.measure(search, range(5))
    return qc
"""
    details = _grade(qiskit_core, "05", code, task["entry_point"])
    assert details["passed"] is False


# ---------------------------------------------------------------------------
# Fix 3: task 47 phase-kickback recipe accepts exact preparation variants.
# ---------------------------------------------------------------------------


def test_task47_h_then_z_minus_preparation_passes(qiskit_core) -> None:
    code = """
from qiskit import QuantumCircuit
def period_finding_mod4_phase_kickback():
    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.z(2)
    qc.cx(0, 2)
    qc.h(0)
    qc.h(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc
"""
    details = _grade(qiskit_core, "47", code, "period_finding_mod4_phase_kickback")
    assert details["passed"] is True, details["reason"]


def test_task47_x_prep_with_cz_oracle_passes(qiskit_core) -> None:
    code = """
from qiskit import QuantumCircuit
def period_finding_mod4_phase_kickback():
    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.h(1)
    qc.x(2)
    qc.cz(0, 2)
    qc.h(0)
    qc.h(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc
"""
    details = _grade(qiskit_core, "47", code, "period_finding_mod4_phase_kickback")
    assert details["passed"] is True, details["reason"]


def test_task47_direct_output_synthesis_still_fails(qiskit_core) -> None:
    code = """
from qiskit import QuantumCircuit
def period_finding_mod4_phase_kickback():
    qc = QuantumCircuit(3, 2)
    qc.x(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc
"""
    details = _grade(qiskit_core, "47", code, "period_finding_mod4_phase_kickback")
    assert details["passed"] is False


@pytest.mark.parametrize("task_id", ["11", "20", "47"])
def test_kickback_guarded_canonicals_still_pass(qiskit_core, task_id: str) -> None:
    task = load_tasks("qiskit", suite="core")[task_id]
    details = _grade(qiskit_core, task_id, task["canonical_solution"], task["entry_point"])
    assert details["passed"] is True, details["reason"]


# ---------------------------------------------------------------------------
# Fix 4: the Clifford gate-class check is angle-aware.
# ---------------------------------------------------------------------------


def test_task44_quarter_turn_rz_decomposition_passes(qiskit_core) -> None:
    code = """
import numpy as np
from qiskit import QuantumCircuit
def CX_gate_decompose_Clifford():
    qc = QuantumCircuit(2)
    qc.rz(np.pi / 2, 1); qc.sx(1); qc.rz(np.pi / 2, 1)
    qc.cz(0, 1)
    qc.rz(np.pi / 2, 1); qc.sx(1); qc.rz(np.pi / 2, 1)
    return qc
"""
    details = _grade(qiskit_core, "44", code, "CX_gate_decompose_Clifford")
    assert details["passed"] is True, details["reason"]


def test_task44_non_clifford_angle_still_fails(qiskit_core) -> None:
    code = """
import numpy as np
from qiskit import QuantumCircuit
def CX_gate_decompose_Clifford():
    qc = QuantumCircuit(2)
    qc.rz(0.3, 1); qc.sx(1); qc.rz(np.pi / 2, 1)
    qc.cz(0, 1)
    qc.rz(np.pi / 2, 1); qc.sx(1); qc.rz(np.pi / 2, 1)
    return qc
"""
    details = _grade(qiskit_core, "44", code, "CX_gate_decompose_Clifford")
    assert details["passed"] is False
    assert "forbidden_gate_family:rz" in details["reason"]


def test_task44_canonical_still_passes(qiskit_core) -> None:
    task = load_tasks("qiskit", suite="core")["44"]
    details = _grade(qiskit_core, "44", task["canonical_solution"], task["entry_point"])
    assert details["passed"] is True, details["reason"]


# ---------------------------------------------------------------------------
# Fix 5: required_encoder_state_before_ancilla_use is enforced.
# ---------------------------------------------------------------------------


def test_qec08_encoderless_bypass_now_fails(qiskit_qec) -> None:
    code = """
from qiskit import QuantumCircuit

def steane_z_syndrome(error_qubit: int | None):
    qc = QuantumCircuit(10, 3)
    if error_qubit is not None:
        qc.x(error_qubit)
    for d in [0, 2, 4, 6]:
        qc.cx(d, 7)
    for d in [1, 2, 5, 6]:
        qc.cx(d, 8)
    for d in [3, 4, 5, 6]:
        qc.cx(d, 9)
    qc.measure(7, 0); qc.measure(8, 1); qc.measure(9, 2)
    return qc
"""
    details = _grade(qiskit_qec, "qec08", code, "steane_z_syndrome")
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"
    assert "required_encoder_state_before_ancilla_use" in _evidence_reasons(details)


def test_qec11_hadamard_bypass_now_fails(qiskit_qec) -> None:
    code = """
from qiskit import QuantumCircuit

def shor_x_syndrome(error_qubit: int | None):
    qc = QuantumCircuit(11, 2)
    for q in range(9):
        qc.h(q)
    if error_qubit is not None:
        qc.z(error_qubit)
    qc.h(9)
    for d in range(6):
        qc.cx(9, d)
    qc.h(9)
    qc.h(10)
    for d in range(3, 9):
        qc.cx(10, d)
    qc.h(10)
    qc.measure(9, 0); qc.measure(10, 1)
    return qc
"""
    details = _grade(qiskit_qec, "qec11", code, "shor_x_syndrome")
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"
    assert "required_encoder_state_before_ancilla_use" in _evidence_reasons(details)


def test_qec08_reordered_steane_encoder_still_passes(qiskit_qec) -> None:
    code = """
from qiskit import QuantumCircuit

def steane_z_syndrome(error_qubit: int | None):
    qc = QuantumCircuit(10, 3)
    qc.h(3); qc.h(1); qc.h(0)
    qc.cx(1, 6); qc.cx(3, 5); qc.cx(0, 2)
    qc.cx(1, 5); qc.cx(3, 6); qc.cx(0, 4)
    qc.cx(3, 4); qc.cx(1, 2); qc.cx(0, 6)
    if error_qubit is not None:
        qc.x(error_qubit)
    for d in [0, 2, 4, 6]:
        qc.cx(d, 7)
    for d in [1, 2, 5, 6]:
        qc.cx(d, 8)
    for d in [3, 4, 5, 6]:
        qc.cx(d, 9)
    qc.measure(7, 0); qc.measure(8, 1); qc.measure(9, 2)
    return qc
"""
    details = _grade(qiskit_qec, "qec08", code, "steane_z_syndrome")
    assert details["passed"] is True, (details["reason"], _evidence_reasons(details))


# ---------------------------------------------------------------------------
# Fix 6: robustness of point binding and CUDA-Q kernel launches.
# ---------------------------------------------------------------------------


def test_point_arguments_preserve_float_type() -> None:
    from qceval.evals.evaluator import _point_arguments

    values = _point_arguments((0.0, 1.0, 2, None))
    assert values == (0.0, 1.0, 2, None)
    assert isinstance(values[0], float)
    assert isinstance(values[2], int)


def test_task35_parameterized_cudaq_kernel_fails_typed_without_crash(cudaq_core) -> None:
    """A kernel taking runtime arguments used to SIGSEGV the grader process."""
    code = """
import cudaq

def parity_check_3bit():
    @cudaq.kernel
    def kernel(bits: list[int]):
        q = cudaq.qvector(4)
        for i in range(3):
            if bits[i] == 1:
                x(q[i])
        x.ctrl(q[0], q[3])
        x.ctrl(q[1], q[3])
        x.ctrl(q[2], q[3])
        mz(q)
    return kernel
"""
    details = _grade(cudaq_core, "35", code, "parity_check_3bit")
    assert details["passed"] is False
    assert details["semantic_status"] in {"semantic_fail", "execution_error"}
