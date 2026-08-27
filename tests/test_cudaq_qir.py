"""Adaptive-QIR lowering and AST-parity tests."""

# ruff: noqa: F821

from __future__ import annotations

from dataclasses import replace

import cudaq
import numpy as np
import pytest

from qceval.frameworks.cudaq.lowering import CudaqLoweringAdapter
from qceval.frameworks.cudaq.parser import from_cudaq
from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.frameworks.cudaq.qir import lower_cudaq_qir
from qceval.frameworks.cudaq.qir_parser import QirParseError, QirParseLimits, parse_adaptive_qir
from qceval.semantics.ir import ClassicalCondition, OperationKind, Provenance
from qceval.semantics.lowering.base import LoweringStatus, SourceMetadata
from qceval.semantics.verifiers.dynamic import ExactBranchSimulator


@cudaq.kernel
def _bell_kernel():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


@cudaq.kernel
def _multi_control_kernel():
    q = cudaq.qvector(5)
    x.ctrl([q[0], q[1]], q[2])
    swap.ctrl(q[0], q[3], q[4])
    r1.ctrl(0.5, q[0], q[1], q[2])
    mz(q[2])
    mz(q[4])


@cudaq.kernel
def _argument_kernel(bits: list[int]):
    q = cudaq.qvector(3)
    ancilla = cudaq.qubit()
    for index in range(3):
        if bits[index] == 1:
            x(q[index])
    x.ctrl(q[0], ancilla)
    x.ctrl(q[1], ancilla)
    x.ctrl(q[2], ancilla)
    mz(ancilla)


@cudaq.kernel
def _feedback_kernel():
    q = cudaq.qvector(2)
    h(q[0])
    measured = mz(q[0])
    if measured:
        x(q[1])
    mz(q[1])


@cudaq.kernel
def _tuple_data_kernel():
    q = cudaq.qvector(3)
    angles = [0.6, 0.3]
    for first, second in [(0, 1), (1, 2)]:
        x.ctrl(q[first], q[second])
        rz(2.0 * angles[0], q[second])


@cudaq.kernel
def _short_circuit_kernel(values: list[int]):
    q = cudaq.qvector(1)
    if values[0] != 0 and values[1] != 0:
        x(q[0])


def _bound_kernel_wrapper():
    theta = 0.25
    count = 2

    @cudaq.kernel
    def kernel(value: float, repetitions: int):
        q = cudaq.qvector(1)
        for _ in range(repetitions):
            ry(value, q[0])

    return lambda: kernel(theta, count)


@cudaq.kernel
def _adjoint_kernel():
    q = cudaq.qvector(1)
    h(q[0])
    s.adj(q[0])
    t.adj(q[0])


_MULTI_KERNEL_SOURCE = """\
import cudaq


def solver():
    @cudaq.kernel
    def rejected_attempt():
        q = cudaq.qvector(7)
        h(q[0])
        mz(q)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        h(q[0])
        mz(q[0])
        mz(q[1])
        mz(q[2])

    return kernel
"""


@cudaq.kernel
def _second_of_two_kernels():
    q = cudaq.qvector(3)
    h(q[0])
    mz(q[0])
    mz(q[1])
    mz(q[2])


_BELL_SOURCE = """\
import cudaq


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)
"""


def _metadata() -> SourceMetadata:
    return SourceMetadata("cudaq", "a" * 64, "qpp-cpu")


def test_qir_static_bell_matches_ast_replay_state() -> None:
    carrier = CudaqProgram(_BELL_SOURCE, "bell", kernel=_bell_kernel)

    qir_program = lower_cudaq_qir(carrier, _metadata())
    ast_circuit = from_cudaq(CudaqProgram(_BELL_SOURCE, "bell"))
    ast_state = np.zeros(4, dtype=np.complex128)
    ast_state[0] = 1 / np.sqrt(2)
    ast_state[3] = 1 / np.sqrt(2)
    static_program = replace(
        qir_program,
        num_clbits=0,
        operations=tuple(
            operation for operation in qir_program.operations if operation.kind is not OperationKind.MEASUREMENT
        ),
        classical_render_order=(),
    )
    qir_state = ExactBranchSimulator().run(static_program, max_branches=1)[0].statevector

    assert ast_circuit.num_qubits == qir_program.num_qubits == 2
    assert np.allclose(qir_state, ast_state)
    assert qir_program.diagnostics == ("cudaq_qir_adaptive",)


def test_qir_generalized_calls_preserve_controls_targets_and_measurements() -> None:
    program = lower_cudaq_qir(CudaqProgram("", "_multi_control_kernel", kernel=_multi_control_kernel), _metadata())

    gates = [operation for operation in program.operations if operation.kind is OperationKind.GATE]
    assert [(operation.name, operation.quantum_wires) for operation in gates] == [
        ("x", (2,)),
        ("swap", (3, 4)),
        ("r1", (2,)),
    ]
    assert [tuple(control.wire for control in operation.controls) for operation in gates] == [
        (0, 1),
        (0,),
        (0, 1),
    ]
    measurements = [operation for operation in program.operations if operation.kind is OperationKind.MEASUREMENT]
    assert [(item.quantum_wires, item.classical_bits) for item in measurements] == [
        ((2,), (0,)),
        ((4,), (1,)),
    ]
    assert program.classical_render_order == (1, 0)


def test_qir_translation_specializes_concrete_classical_arguments() -> None:
    program = lower_cudaq_qir(
        CudaqProgram("", "_argument_kernel", ([1, 0, 1],), _argument_kernel),
        _metadata(),
    )

    gates = [operation for operation in program.operations if operation.kind is OperationKind.GATE]
    unconditional_x = [
        operation.quantum_wires for operation in gates if operation.name == "x" and not operation.controls
    ]
    assert unconditional_x == [(0,), (2,)]
    assert all(operation.condition is None for operation in gates)


def test_qir_feedback_branch_becomes_classical_condition() -> None:
    program = lower_cudaq_qir(CudaqProgram("", "_feedback_kernel", kernel=_feedback_kernel), _metadata())

    measurements = [operation for operation in program.operations if operation.kind is OperationKind.MEASUREMENT]
    conditioned = [
        operation
        for operation in program.operations
        if operation.kind is OperationKind.GATE and operation.condition is not None
    ]

    assert [(item.quantum_wires, item.classical_bits) for item in measurements] == [
        ((0,), (0,)),
        ((1,), (1,)),
    ]
    assert len(conditioned) == 1
    assert conditioned[0].name == "x"
    assert conditioned[0].quantum_wires == (1,)
    assert conditioned[0].condition == ClassicalCondition((0,), 1)


def test_qir_unwraps_a_bound_zero_argument_kernel_closure() -> None:
    program = lower_cudaq_qir(
        CudaqProgram("", "_bound_kernel_wrapper", kernel=_bound_kernel_wrapper()),
        _metadata(),
    )

    assert [(operation.name, operation.parameters[0].value) for operation in program.operations] == [
        ("ry", "0.25"),
        ("ry", "0.25"),
    ]


def test_qir_folds_compiler_classical_arrays_tuples_and_float_memory() -> None:
    program = lower_cudaq_qir(CudaqProgram("", "_tuple_data_kernel", kernel=_tuple_data_kernel), _metadata())

    assert [(operation.name, operation.quantum_wires) for operation in program.operations] == [
        ("x", (1,)),
        ("rz", (1,)),
        ("x", (2,)),
        ("rz", (2,)),
    ]
    assert [operation.parameters[0].value for operation in program.operations if operation.name == "rz"] == [
        "1.2",
        "1.2",
    ]


def test_qir_resolves_short_circuit_phi_nodes_from_concrete_arguments() -> None:
    false_program = lower_cudaq_qir(
        CudaqProgram("", "_short_circuit_kernel", ([1, 0],), _short_circuit_kernel),
        _metadata(),
    )
    true_program = lower_cudaq_qir(
        CudaqProgram("", "_short_circuit_kernel", ([1, 1],), _short_circuit_kernel),
        _metadata(),
    )

    assert false_program.operations == ()
    assert [(operation.name, operation.quantum_wires) for operation in true_program.operations] == [("x", (0,))]


def test_qir_lowers_only_the_executed_kernel_from_multi_kernel_source() -> None:
    # Source-text scanning miscounted registers and measurements when the
    # candidate defined several kernels; compiler IR sees only the executed one.
    program = lower_cudaq_qir(
        CudaqProgram(_MULTI_KERNEL_SOURCE, "solver", kernel=_second_of_two_kernels),
        _metadata(),
    )

    assert program.num_qubits == 3
    assert program.num_clbits == 3
    measurements = [operation for operation in program.operations if operation.kind is OperationKind.MEASUREMENT]
    assert [(item.quantum_wires, item.classical_bits) for item in measurements] == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
    ]


def test_qir_normalizes_adjoint_phase_gates() -> None:
    program = lower_cudaq_qir(CudaqProgram("", "_adjoint_kernel", kernel=_adjoint_kernel), _metadata())

    assert [operation.name for operation in program.operations] == ["h", "sdg", "tdg"]


def test_qir_parser_rejects_unknown_instructions_and_unspecialized_loops() -> None:
    unknown = _minimal_qir("  call void @unknown_runtime()")
    loop = """\
%Qubit = type opaque
%Result = type opaque
define void @__nvqpp__mlirgen__fixture() #0 {
  br label %loop
loop:
  br label %loop
}
attributes #0 = { "entry_point" "requiredQubits"="1" "requiredResults"="0" }
"""

    with pytest.raises(QirParseError, match="unsupported adaptive-QIR instruction"):
        parse_adaptive_qir(unknown, provenance=Provenance("cudaq", "test"))
    with pytest.raises(QirParseError, match="CFG traversal exceeds"):
        parse_adaptive_qir(
            loop,
            provenance=Provenance("cudaq", "test"),
            limits=QirParseLimits(max_instructions=10),
        )


def test_cudaq_adapter_requires_the_executed_native_kernel() -> None:
    result = CudaqLoweringAdapter().lower(CudaqProgram(_BELL_SOURCE, "bell"), _metadata(), None)

    assert result.status is LoweringStatus.UNSUPPORTED
    assert result.error is not None
    assert result.error.reason == "unsupported_cudaq_qir"
    assert "requires the executed native kernel" in str(result.error.detail)


def _minimal_qir(instruction: str) -> str:
    return f"""\
%Qubit = type opaque
%Result = type opaque
define void @__nvqpp__mlirgen__fixture() #0 {{
{instruction}
  ret void
}}
attributes #0 = {{ "entry_point" "requiredQubits"="1" "requiredResults"="0" }}
"""


def test_qir_signed_division_truncates_toward_zero_like_llvm() -> None:
    from qceval.frameworks.cudaq.qir.models import _State
    from qceval.frameworks.cudaq.qir.ssa import _evaluate_arithmetic

    cases = {
        ("sdiv", -7, 2): -3,  # Python -7 // 2 == -4; LLVM truncates to -3
        ("sdiv", 7, -2): -3,
        ("sdiv", -7, -2): 3,
        ("sdiv", 7, 2): 3,
        ("srem", -7, 2): -1,  # Python -7 % 2 == 1; LLVM keeps the dividend sign
        ("srem", 7, -2): 1,
        ("srem", -7, -2): -1,
        ("srem", 7, 2): 1,
        ("udiv", 7, 2): 3,
        ("urem", 7, 2): 1,
    }
    for (opcode, left, right), expected in cases.items():
        state = _State()
        _evaluate_arithmetic("%out", f"{opcode} i64 {left}, {right}", state)
        assert state.values["%out"] == expected, (opcode, left, right)
        # LLVM identity: left == sdiv * right + srem must always hold.
    for left, right in [(-7, 2), (7, -2), (-9, 4), (9, -4)]:
        div_state, rem_state = _State(), _State()
        _evaluate_arithmetic("%q", f"sdiv i64 {left}, {right}", div_state)
        _evaluate_arithmetic("%r", f"srem i64 {left}, {right}", rem_state)
        assert div_state.values["%q"] * right + rem_state.values["%r"] == left
