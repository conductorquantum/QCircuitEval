"""Regressions for CUDA-Q grader defects found in the kimi run4 audit."""

from __future__ import annotations

from qceval.evals.evaluator import build_evaluator

_GROVER_CLEAN_ANCILLA = """\
import cudaq

@cudaq.kernel
def grover_3SAT():
    q = cudaq.qvector(3)
    anc = cudaq.qubit()

    h(q[0])
    h(q[1])
    h(q[2])

    x(q[0])
    x.ctrl([q[0], q[1]], anc)
    z(anc)
    x.ctrl([q[0], q[1]], anc)
    x(q[0])

    h(q[0])
    h(q[1])
    h(q[2])

    x(q[0])
    x(q[1])
    x(q[2])

    x.ctrl([q[0], q[1]], anc)
    z.ctrl(anc, q[2])
    x.ctrl([q[0], q[1]], anc)

    x(q[0])
    x(q[1])
    x(q[2])

    h(q[0])
    h(q[1])
    h(q[2])

    mz(q)
"""

_U_DECOMPOSE_WRONG_MIDDLE = """\
import cudaq

@cudaq.kernel
def U_gate_decompose(theta: float, phi: float, lam: float):
    q = cudaq.qubit()
    pi = 3.141592653589793
    rz(lam + pi, q)
    rx(pi / 2.0, q)
    rz(theta + pi, q)
    rx(pi / 2.0, q)
    rz(phi, q)
"""

_U_DECOMPOSE_CORRECT = """\
import cudaq

@cudaq.kernel
def U_gate_decompose(theta: float, phi: float, lam: float):
    q = cudaq.qubit()
    pi = 3.141592653589793
    rz(lam, q)
    rx(pi / 2.0, q)
    rz(theta + pi, q)
    rx(pi / 2.0, q)
    rz(phi + pi, q)
"""


def test_grover_clean_ancilla_phase_oracle_passes() -> None:
    _, details = build_evaluator("cudaq", suite="core").grade_code(
        task_id="03",
        code=_GROVER_CLEAN_ANCILLA,
        entry_point="grover_3SAT",
    )
    assert details["passed"] is True, details.get("reason")


def test_u_gate_qubit_allocation_wrong_rotation_fails_semantically() -> None:
    _, details = build_evaluator("cudaq", suite="core").grade_code(
        task_id="42",
        code=_U_DECOMPOSE_WRONG_MIDDLE,
        entry_point="U_gate_decompose",
    )
    reason = str(details.get("reason", ""))
    assert details["passed"] is False
    assert "symbolic_call_unsupported" not in reason, reason
    assert any(token in reason for token in ("counterexample", "mismatch", "semantic")), reason


def test_u_gate_qubit_allocation_correct_decomposition_passes() -> None:
    _, details = build_evaluator("cudaq", suite="core").grade_code(
        task_id="42",
        code=_U_DECOMPOSE_CORRECT,
        entry_point="U_gate_decompose",
    )
    assert details["passed"] is True, details.get("reason")
