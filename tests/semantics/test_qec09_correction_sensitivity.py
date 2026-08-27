"""Regression tests for the qec09 logical-one convention and correction sensitivity.

Finding H7: qec09 previously accepted (and required) applying X to all seven
data qubits BEFORE the Steane encoder, a state outside the codespace. The task
now uses the qec07 convention: logical |1_L> is transversal X on all seven
data qubits AFTER encoding |0_L>.

Finding M1: qec09 previously observed only decoded qubit 0, so deleting the
correction stage or mistargeting every syndrome still reproduced all sixteen
expected outputs. The contract now observes all seven decoded data qubits, so
any missing or mistargeted correction leaves a visible residual.
"""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks

HEADER = """from qiskit import QuantumCircuit


def steane_x_correct(logical_bit: int, error_qubit: int | None):
    qc = QuantumCircuit(10, 7)
"""

ENCODER = """    qc.h(0); qc.h(1); qc.h(3)
    qc.cx(0, 2); qc.cx(3, 5); qc.cx(1, 6)
    qc.cx(0, 4); qc.cx(3, 6); qc.cx(1, 5)
    qc.cx(0, 6); qc.cx(1, 2); qc.cx(3, 4)
"""

LOGICAL_X_ALL_SEVEN = """    if logical_bit == 1:
        for i in range(7):
            qc.x(i)
"""

ERROR_AND_SYNDROME = """    if error_qubit is not None:
        qc.x(error_qubit)
    for d in [0, 2, 4, 6]:
        qc.cx(d, 7)
    for d in [1, 2, 5, 6]:
        qc.cx(d, 8)
    for d in [3, 4, 5, 6]:
        qc.cx(d, 9)
"""

CORRECTION = """    qc.x(8); qc.x(9); qc.mcx([7, 8, 9], 0); qc.x(8); qc.x(9)
    qc.x(7); qc.x(9); qc.mcx([7, 8, 9], 1); qc.x(7); qc.x(9)
    qc.x(9); qc.mcx([7, 8, 9], 2); qc.x(9)
    qc.x(7); qc.x(8); qc.mcx([7, 8, 9], 3); qc.x(7); qc.x(8)
    qc.x(8); qc.mcx([7, 8, 9], 4); qc.x(8)
    qc.x(7); qc.mcx([7, 8, 9], 5); qc.x(7)
    qc.mcx([7, 8, 9], 6)
"""

#: Every syndrome fires a correction, but each one targets the wrong qubit.
WRONG_TARGET_CORRECTION = """    qc.x(8); qc.x(9); qc.mcx([7, 8, 9], 1); qc.x(8); qc.x(9)
    qc.x(7); qc.x(9); qc.mcx([7, 8, 9], 2); qc.x(7); qc.x(9)
    qc.x(9); qc.mcx([7, 8, 9], 3); qc.x(9)
    qc.x(7); qc.x(8); qc.mcx([7, 8, 9], 4); qc.x(7); qc.x(8)
    qc.x(8); qc.mcx([7, 8, 9], 5); qc.x(8)
    qc.x(7); qc.mcx([7, 8, 9], 6); qc.x(7)
    qc.mcx([7, 8, 9], 0)
"""

DECODE_AND_MEASURE = """    qc.cx(3, 4); qc.cx(1, 2); qc.cx(0, 6)
    qc.cx(1, 5); qc.cx(3, 6); qc.cx(0, 4)
    qc.cx(1, 6); qc.cx(3, 5); qc.cx(0, 2)
    qc.h(3); qc.h(1); qc.h(0)
    qc.cx(2, 0); qc.cx(0, 2); qc.cx(0, 4); qc.cx(0, 5)
    for i in range(7):
        qc.measure(i, i)
    return qc
"""


def _grade(code: str) -> dict:
    evaluator = build_evaluator("qiskit", suite="qec")
    _, details = evaluator.grade_code(task_id="qec09", code=code, entry_point="steane_x_correct")
    return details


def test_variant_blocks_reassemble_the_packaged_canonical_solution() -> None:
    # Guards the block strings below against drifting from the shipped asset.
    task = load_tasks("qiskit", suite="qec")["qec09"]
    assembled = HEADER + ENCODER + LOGICAL_X_ALL_SEVEN + ERROR_AND_SYNDROME + CORRECTION + DECODE_AND_MEASURE
    assert assembled == task["canonical_solution"]


def test_qec09_true_codeword_convention_passes() -> None:
    task = load_tasks("qiskit", suite="qec")["qec09"]
    details = _grade(task["canonical_solution"])
    assert details["passed"] is True, details.get("reason")
    assert details["semantic_status"] == "verified_pass"


def test_qec09_rejects_pre_encoding_transversal_x() -> None:
    # H7: X on all seven qubits BEFORE the encoder is not the Steane logical
    # one; the old canonical construction must now fail.
    code = HEADER + LOGICAL_X_ALL_SEVEN + ENCODER + ERROR_AND_SYNDROME + CORRECTION + DECODE_AND_MEASURE
    details = _grade(code)
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_qec09_rejects_single_wire_logical_input() -> None:
    # The prompt states the transversal-X-after-encoding convention, so an
    # encoder that feeds the logical bit into one input wire is off-contract.
    code = (
        HEADER
        + "    if logical_bit == 1:\n        qc.x(0)\n"
        + ENCODER
        + ERROR_AND_SYNDROME
        + CORRECTION
        + DECODE_AND_MEASURE
    )
    details = _grade(code)
    assert details["passed"] is False


def test_qec09_rejects_deleted_correction_stage() -> None:
    # M1: without the correction stage every injected error leaves a visible
    # residual on the observed decoded data qubits.
    code = HEADER + ENCODER + LOGICAL_X_ALL_SEVEN + ERROR_AND_SYNDROME + DECODE_AND_MEASURE
    details = _grade(code)
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_qec09_rejects_wrong_syndrome_to_target_mapping() -> None:
    # M1: corrections that fire on every syndrome but flip the wrong data
    # qubit satisfy the structural rule yet must fail the graded cases.
    code = HEADER + ENCODER + LOGICAL_X_ALL_SEVEN + ERROR_AND_SYNDROME + WRONG_TARGET_CORRECTION + DECODE_AND_MEASURE
    details = _grade(code)
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane", "cudaq"])
def test_qec09_prompt_states_logical_one_convention(framework: str) -> None:
    prompt = load_tasks(framework, suite="qec")["qec09"]["prompt"]
    assert "Realize logical one by applying X to every data qubit 0 through 6 after encoding |0_L>" in prompt
