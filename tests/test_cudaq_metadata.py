from __future__ import annotations

import ast

import pytest

from qceval.frameworks.cudaq import execute_cudaq_task
from qceval.frameworks.cudaq import metadata as cudaq_metadata
from qceval.frameworks.cudaq.metadata_source import _eval_int


def test_cudaq_kernel_metadata_counts_operations() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq

def answer():
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        h(q[0])
        h(q[0])

    return kernel
"""

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.metadata["non_measurement_operation_count"] == 2
    assert result.metadata["operation_counts"]["h"] == 2


def test_cudaq_source_metadata_recognizes_x_ctrl_pair_as_cx() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq


def answer():
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        x.ctrl(q[0], q[1])

    return kernel
"""

    # Act
    result = execute_cudaq_task(task_id="16", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.metadata["gate_family_counts"]["cx"] >= 1
    assert (0, 1) in {tuple(pair) for pair in result.metadata["interaction_pairs"]}


def test_cudaq_source_metadata_recognizes_x_ctrl_list_as_ccx() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq


def answer():
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x.ctrl([q[0], q[1]], q[2])

    return kernel
"""

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.metadata["gate_family_counts"]["ccx"] >= 1
    interactions = {tuple(pair) for pair in result.metadata["interaction_pairs"]}
    assert {(0, 1), (0, 2), (1, 2)} <= interactions


def test_cudaq_source_metadata_simple_singles_no_interactions() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq


def answer():
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        h(q[0])
        x(q[0])

    return kernel
"""

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.metadata["interaction_pairs"] == []
    assert result.metadata["gate_family_counts"]["h"] == 1
    assert result.metadata["gate_family_counts"]["x"] == 1


def test_cudaq_accepts_direct_kernel_with_call_args() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq

@cudaq.kernel
def answer(bit: int):
    data = cudaq.qvector(1)
    if bit == 1:
        x(data[0])
    mz(data[0])
"""

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={}, call_args=(1,))

    # Assert
    assert result.probabilities[1] == pytest.approx(1.0, abs=0.05)
    assert result.metadata["probability_method"] in {"sample_fallback", "statevector_replay"}
    assert result.metadata["measurement_qubits"] == [0]


def test_cudaq_retries_returned_zero_arg_kernel_after_call_args_consumed() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq

def answer(bit: int):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(1)
    if bit == 1:
        kernel.x(q[0])
    kernel.mz(q[0])
    return kernel
"""

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={}, call_args=(1,))

    # Assert
    assert result.probabilities[1] == pytest.approx(1.0, abs=0.05)
    assert result.metadata["kernel_argument_count"] == 1


def test_cudaq_metadata_parses_noncanonical_measurement_and_alloc_names() -> None:
    # Arrange
    code = """
import cudaq

@cudaq.kernel
def answer(left: int, right: int):
    ctrl_q = cudaq.qvector(3)
    tgt_q = cudaq.qvector(3)
    mz(tgt_q[0])
    mz(ctrl_q[0])
"""

    # Assert
    assert cudaq_metadata._measurement_indices_from_code(code) == [3, 0]
    assert cudaq_metadata._allocated_qubits_from_code(code) == 6


def test_cudaq_metadata_parses_vector_measurement_and_dynamic_alloc_size() -> None:
    code = """
import cudaq

@cudaq.kernel
def answer():
    width = 2
    q = cudaq.qvector(width)
    mz(q)
"""

    metadata = cudaq_metadata._operation_metadata_from_code(code)

    assert cudaq_metadata._allocated_qubits_from_code(code) == 2
    assert metadata["measurement_qubits"] == [0, 1]
    assert metadata["measurement_count"] == 2


def test_cudaq_metadata_resolves_alias_interaction_pairs() -> None:
    code = """
import cudaq

@cudaq.kernel
def answer():
    number_of_qubits = 2
    qubits = cudaq.qvector(number_of_qubits)
    aux = qubits[0]
    target = qubits[number_of_qubits - 1]
    x.ctrl(aux, target)
"""

    metadata = cudaq_metadata._operation_metadata_from_code(code)

    assert [0, 1] in metadata["interaction_pairs"]
    assert metadata["gate_family_counts"]["cx"] >= 1


def test_cudaq_metadata_keeps_multiple_qvector_offsets() -> None:
    code = """
import cudaq

@cudaq.kernel
def answer():
    left = cudaq.qvector(2)
    right = cudaq.qvector(2)
    target = right[1]
    mz(right)
    x.ctrl(left[1], target)
"""

    metadata = cudaq_metadata._operation_metadata_from_code(code)

    assert cudaq_metadata._allocated_qubits_from_code(code) == 4
    assert metadata["measurement_qubits"] == [2, 3]
    assert [1, 3] in metadata["interaction_pairs"]


def test_cudaq_metadata_resolves_qubit_allocations_and_multicontrol_aliases() -> None:
    code = """
import cudaq

@cudaq.kernel
def answer():
    control_a = cudaq.qubit()
    control_b = cudaq.qubit()
    target = cudaq.qubit()
    x.ctrl([control_a, control_b], target)
    mz(target)
"""

    metadata = cudaq_metadata._operation_metadata_from_code(code)

    assert cudaq_metadata._allocated_qubits_from_code(code) == 3
    assert metadata["measurement_qubits"] == [2]
    assert metadata["gate_family_counts"]["ccx"] >= 1
    assert {tuple(pair) for pair in metadata["interaction_pairs"]} >= {(0, 1), (0, 2), (1, 2)}


def test_cudaq_metadata_unknown_dynamic_allocation_is_not_guessed() -> None:
    code = """
import cudaq

@cudaq.kernel
def answer(width: int):
    q = cudaq.qvector(width)
    aux = cudaq.qubit()
    mz(q[0])
"""

    assert cudaq_metadata._allocated_qubits_from_code(code) is None
    assert cudaq_metadata._measurement_indices_from_code(code) == [0]


def test_cudaq_metadata_falls_back_to_regex_when_source_is_invalid() -> None:
    code = "def answer(:\n    mz(q[2])\n"

    assert cudaq_metadata._measurement_indices_from_code(code) == [2]
    assert cudaq_metadata._allocated_qubits_from_code(code) is None


def test_cudaq_metadata_integer_expression_helper_handles_known_forms() -> None:
    def expression(source: str) -> ast.AST:
        return ast.parse(source).body[0].value  # type: ignore[union-attr]

    assert _eval_int(expression("-2"), {}) == -2
    assert _eval_int(expression("4 // 2"), {}) == 2
    assert _eval_int(expression("4 // 0"), {}) is None
    assert _eval_int(expression("2 / 1"), {}) is None


def test_cudaq_base_metadata_tolerates_missing_target() -> None:
    assert cudaq_metadata._base_metadata(object())["cudaq_target"] is None


def test_cudaq_metadata_recognizes_controlled_phase_r1_ctrl() -> None:
    code = """
import cudaq
from math import pi

@cudaq.kernel
def answer():
    q = cudaq.qvector(2)
    h(q[0])
    h(q[1])
    r1.ctrl(pi / 2, q[0], q[1])
"""

    metadata = cudaq_metadata._operation_metadata_from_code(code)

    assert metadata["gate_family_counts"]["cr1"] >= 1
    assert metadata["entangling_gate_count"] >= 1
    assert [0, 1] in metadata["interaction_pairs"]


def test_cudaq_metadata_recognizes_controlled_rotation_rz_ctrl() -> None:
    code = """
import cudaq

@cudaq.kernel
def answer():
    q = cudaq.qvector(2)
    rz.ctrl(0.5, q[0], q[1])
"""

    metadata = cudaq_metadata._operation_metadata_from_code(code)

    assert metadata["gate_family_counts"]["crz"] >= 1
    assert metadata["entangling_gate_count"] >= 1
    assert [0, 1] in metadata["interaction_pairs"]


def test_cudaq_metadata_recognizes_direct_controlled_phase_helper() -> None:
    code = """
import cudaq
from math import pi

@cudaq.kernel
def answer():
    q = cudaq.qvector(2)
    cr1(pi / 4, q[0], q[1])
"""

    metadata = cudaq_metadata._operation_metadata_from_code(code)

    assert metadata["gate_family_counts"]["cr1"] >= 1
    assert metadata["entangling_gate_count"] >= 1
    assert [0, 1] in metadata["interaction_pairs"]


def test_cudaq_metadata_controlled_rotation_qft_has_entangling_evidence() -> None:
    code = """
import cudaq
from math import pi

@cudaq.kernel
def answer():
    q = cudaq.qvector(3)
    for i in range(3):
        h(q[i])
    r1.ctrl(pi / 2, q[0], q[1])
    r1.ctrl(pi / 4, q[0], q[2])
    r1.ctrl(pi / 2, q[1], q[2])
"""

    metadata = cudaq_metadata._operation_metadata_from_code(code)

    # A native controlled-phase QFT must register as entangling so structural
    # checks accept it instead of collapsing to a Hadamard-only stub.
    assert metadata["entangling_gate_count"] >= 3
    interactions = {tuple(pair) for pair in metadata["interaction_pairs"]}
    assert {(0, 1), (0, 2), (1, 2)} <= interactions
