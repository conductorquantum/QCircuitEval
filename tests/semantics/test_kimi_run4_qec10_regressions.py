"""Adversarial regressions for qec10 outer phase-flip encoder enforcement.

The kimi-run4 production audit found two incorrect qec10 candidates passing:
a Cirq submission whose cross-block gates run AFTER the inner bit-flip
encoders (state-trivial on |0_L> but not the demanded outer encoder), and a
CUDA-Q submission that omits the outer phase-flip encoder entirely. Both in
fact prepare exactly |0_L> = GHZ x GHZ x GHZ before any ancilla use (GHZ
blocks are invariant under X^3, so trailing cross-block CNOTs act trivially),
and the 2026-07 prompt/contract audit removed the unprompted
construction-order requirement: the prompt defines the codeword state, not a
gate ordering. These submissions are therefore accepted, while the newly
enforced required_encoder_state_before_ancilla_use requirement rejects
encoder-less submissions whose data register is not in the declared codeword
state (see tests/semantics/test_audit_engine_fixes.py).
"""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks

CIRQ_SHORTCUT = """import cirq

def shor_z_syndrome(error_qubit: int | None) -> cirq.Circuit:
    q = cirq.LineQubit.range(15)
    circuit = cirq.Circuit()

    # Inner bit-flip encoding for each block of three qubits
    for block in (0, 3, 6):
        circuit.append(cirq.H(q[block]))
        circuit.append(cirq.CNOT(q[block], q[block + 1]))
        circuit.append(cirq.CNOT(q[block], q[block + 2]))

    # Outer phase-flip encoding between blocks
    for i in range(3):
        circuit.append(cirq.CNOT(q[i], q[i + 3]))
        circuit.append(cirq.CNOT(q[i], q[i + 6]))

    # Optional single X error on a data qubit
    if error_qubit is not None:
        circuit.append(cirq.X(q[error_qubit]))

    # Extract Z-stabilizer syndromes using ancillas q[9]..q[14]
    stabilizers = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
    for (a, b), anc in zip(stabilizers, range(9, 15)):
        circuit.append(cirq.CNOT(q[a], q[anc]))
        circuit.append(cirq.CNOT(q[b], q[anc]))

    # Measure only the ancilla qubits
    circuit.append(cirq.measure(q[14], q[13], q[12], q[11], q[10], q[9], key="result"))
    return circuit
"""

CUDAQ_SHORTCUT = """import cudaq

def shor_z_syndrome(error_qubit: int | None):
    if error_qubit is None:
        error_qubit = -1

    @cudaq.kernel
    def kernel(error_qubit: int):
        data = cudaq.qvector(9)
        anc = cudaq.qvector(6)

        # Encode |0_L>: outer phase-flip on block leaders, then inner bit-flip
        h(data[0])
        h(data[3])
        h(data[6])

        x.ctrl(data[0], data[1])
        x.ctrl(data[0], data[2])
        x.ctrl(data[3], data[4])
        x.ctrl(data[3], data[5])
        x.ctrl(data[6], data[7])
        x.ctrl(data[6], data[8])

        # Inject optional X error on a data qubit
        for i in range(9):
            if i == error_qubit:
                x(data[i])

        # Z-stabilizer syndrome extraction into ancilla qubits 9-14
        x.ctrl(data[0], anc[0])
        x.ctrl(data[1], anc[0])

        x.ctrl(data[1], anc[1])
        x.ctrl(data[2], anc[1])

        x.ctrl(data[3], anc[2])
        x.ctrl(data[4], anc[2])

        x.ctrl(data[4], anc[3])
        x.ctrl(data[5], anc[3])

        x.ctrl(data[6], anc[4])
        x.ctrl(data[7], anc[4])

        x.ctrl(data[7], anc[5])
        x.ctrl(data[8], anc[5])

        mz(anc)

    return kernel
"""

CIRQ_CORRECT = """import cirq

def shor_z_syndrome(error_qubit: int | None) -> cirq.Circuit:
    q = cirq.LineQubit.range(15)
    circuit = cirq.Circuit()
    circuit.append(cirq.CNOT(q[0], q[3]))
    circuit.append(cirq.CNOT(q[0], q[6]))
    for block in (0, 3, 6):
        circuit.append(cirq.H(q[block]))
        circuit.append(cirq.CNOT(q[block], q[block + 1]))
        circuit.append(cirq.CNOT(q[block], q[block + 2]))
    if error_qubit is not None:
        circuit.append(cirq.X(q[error_qubit]))
    stabilizers = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
    for (a, b), anc in zip(stabilizers, range(9, 15)):
        circuit.append(cirq.CNOT(q[a], q[anc]))
        circuit.append(cirq.CNOT(q[b], q[anc]))
    circuit.append(cirq.measure(q[14], q[13], q[12], q[11], q[10], q[9], key="result"))
    return circuit
"""

CUDAQ_CORRECT = """import cudaq

def shor_z_syndrome(error_qubit: int | None):
    if error_qubit is None:
        error_qubit = -1

    @cudaq.kernel
    def kernel(error_qubit: int):
        data = cudaq.qvector(9)
        anc = cudaq.qvector(6)

        x.ctrl(data[0], data[3])
        x.ctrl(data[0], data[6])
        h(data[0])
        h(data[3])
        h(data[6])
        x.ctrl(data[0], data[1])
        x.ctrl(data[0], data[2])
        x.ctrl(data[3], data[4])
        x.ctrl(data[3], data[5])
        x.ctrl(data[6], data[7])
        x.ctrl(data[6], data[8])

        for i in range(9):
            if i == error_qubit:
                x(data[i])

        x.ctrl(data[0], anc[0])
        x.ctrl(data[1], anc[0])
        x.ctrl(data[1], anc[1])
        x.ctrl(data[2], anc[1])
        x.ctrl(data[3], anc[2])
        x.ctrl(data[4], anc[2])
        x.ctrl(data[4], anc[3])
        x.ctrl(data[5], anc[3])
        x.ctrl(data[6], anc[4])
        x.ctrl(data[7], anc[4])
        x.ctrl(data[7], anc[5])
        x.ctrl(data[8], anc[5])

        mz(anc)

    return kernel
"""


def _grade(framework: str, code: str) -> dict:
    evaluator = build_evaluator(framework, suite="qec")
    _, details = evaluator.grade_code(task_id="qec10", code=code, entry_point="shor_z_syndrome")
    return details


def test_qec10_cirq_accepts_post_inner_cross_block_gates() -> None:
    """Trailing cross-block CNOTs leave |0_L> intact; the prompt pins no order."""
    details = _grade("cirq", CIRQ_SHORTCUT)
    assert details["passed"] is True, details.get("reason")
    assert details["semantic_status"] == "verified_pass"


def test_qec10_cudaq_accepts_direct_ghz_block_encoder() -> None:
    """Direct GHZ-per-block preparation is exactly the prompted |0_L> state."""
    details = _grade("cudaq", CUDAQ_SHORTCUT)
    assert details["passed"] is True, details.get("reason")
    assert details["semantic_status"] == "verified_pass"


def test_qec10_cirq_full_encoder_still_passes() -> None:
    details = _grade("cirq", CIRQ_CORRECT)
    assert details["passed"] is True, details.get("reason")
    assert details["semantic_status"] == "verified_pass"


def test_qec10_cudaq_full_encoder_still_passes() -> None:
    details = _grade("cudaq", CUDAQ_CORRECT)
    assert details["passed"] is True, details.get("reason")
    assert details["semantic_status"] == "verified_pass"


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane", "cudaq"])
def test_qec10_canonical_solutions_still_pass(framework: str) -> None:
    task = load_tasks(framework, suite="qec")["qec10"]
    details = _grade(framework, task["canonical_solution"])
    assert details["passed"] is True, details.get("reason")
    assert details["semantic_status"] == "verified_pass"
