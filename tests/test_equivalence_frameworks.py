from __future__ import annotations

import numpy as np
import pytest

from qceval.evals.ir import Circuit, Gate, from_framework, full_unitary
from qceval.frameworks.cudaq.program import CudaqProgram

X = np.asarray([[0, 1], [1, 0]], dtype=complex)


def _assert_self_and_mutation(circuit: Circuit) -> None:
    assert _equivalent(circuit, circuit)
    mutated = Circuit(circuit.num_qubits, circuit.gates + (Gate.full(X, (0,), name="mutation"),))
    assert not _equivalent(circuit, mutated)


def _equivalent(left: Circuit, right: Circuit) -> bool:
    left_unitary = full_unitary(left)
    right_unitary = full_unitary(right)
    overlap = np.vdot(right_unitary.ravel(), left_unitary.ravel())
    phase = 1.0 + 0.0j if abs(overlap) < 1e-15 else overlap / abs(overlap)
    return bool(np.allclose(left_unitary, phase * right_unitary, atol=1e-9))


def _relabel(circuit: Circuit, mapping: dict[int, int]) -> Circuit:
    return Circuit(
        circuit.num_qubits,
        tuple(
            Gate(
                gate.matrix,
                wires=tuple(mapping.get(wire, wire) for wire in gate.wires),
                controls=tuple(
                    type(control)(mapping.get(control.wire, control.wire), control.value) for control in gate.controls
                ),
                targets=tuple(mapping.get(wire, wire) for wire in gate.targets),
                representation=gate.representation,
                name=gate.name,
            )
            for gate in circuit.gates
        ),
    )


def test_qiskit_subset_bell_and_relabel() -> None:
    qiskit = pytest.importorskip("qiskit")
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    circuit = from_framework(qc, framework="qiskit")

    _assert_self_and_mutation(circuit)

    x0 = Circuit(2, (Gate.full(X, (0,), name="x0"),))
    x1 = _relabel(x0, {0: 1, 1: 0})
    assert not _equivalent(x0, x1)
    assert _equivalent(_relabel(x0, {0: 1, 1: 0}), x1)


def test_qiskit_reset_is_rejected_as_non_unitary() -> None:
    qiskit = pytest.importorskip("qiskit")
    circuit = qiskit.QuantumCircuit(1)
    circuit.reset(0)

    with pytest.raises(NotImplementedError, match="reset"):
        from_framework(circuit, framework="qiskit")


def test_cirq_subset_ghz() -> None:
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(3)
    native = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2]), cirq.measure(*q))
    circuit = from_framework(native, framework="cirq")

    _assert_self_and_mutation(circuit)


def test_pennylane_subset_qft_fragment() -> None:
    qml = pytest.importorskip("pennylane")
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.ControlledPhaseShift(np.pi / 2, wires=[1, 0])
        qml.Hadamard(wires=1)
        qml.probs(wires=[0, 1])

    circuit = from_framework(tape, framework="pennylane")

    _assert_self_and_mutation(circuit)


def test_cudaq_source_subset_bell() -> None:
    code = """
import cudaq


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="bell"), framework="cudaq")

    _assert_self_and_mutation(circuit)


def test_cudaq_static_range_loop_is_unrolled() -> None:
    code = """
import cudaq


@cudaq.kernel
def looped():
    q = cudaq.qvector(2)
    for i in range(2):
        h(q[0])
    mz(q)
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="looped"), framework="cudaq")

    assert len(circuit.gates) == 2


def test_cudaq_whole_register_gate_is_broadcast() -> None:
    code = """
import cudaq


@cudaq.kernel
def broadcast():
    q = cudaq.qvector(3)
    h(q)
    mz(q)
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="broadcast"), framework="cudaq")

    assert len(circuit.gates) == 3
    assert [gate.wires for gate in circuit.gates] == [(0,), (1,), (2,)]


def test_cudaq_controlled_swap_uses_two_targets() -> None:
    code = """
import cudaq


@cudaq.kernel
def cswap_kernel():
    q = cudaq.qvector(3)
    swap.ctrl(q[0], q[1], q[2])
    mz(q)
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="cswap_kernel"), framework="cudaq")

    assert len(circuit.gates) == 1
    assert circuit.gates[0].targets == (1, 2)
    assert tuple(control.wire for control in circuit.gates[0].controls) == (0,)


def test_cudaq_adjoint_gate_is_converted() -> None:
    code = """
import cudaq


@cudaq.kernel
def adjoint_kernel():
    q = cudaq.qvector(1)
    t.adj(q[0])
    mz(q)
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="adjoint_kernel"), framework="cudaq")

    assert len(circuit.gates) == 1
    assert np.allclose(circuit.gates[0].matrix, np.asarray([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex))


def test_cudaq_constant_rotation_binding_is_converted() -> None:
    code = """
import cudaq
from math import pi


def parameterized():
    theta = pi / 2

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        ry(theta, q[0])
        mz(q)

    return kernel
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="parameterized"), framework="cudaq")

    assert len(circuit.gates) == 1
    assert circuit.gates[0].name == "ry"


def test_cudaq_legacy_parser_uses_statement_order_for_rebound_constants() -> None:
    code = """
import cudaq
from math import pi


@cudaq.kernel
def rebound():
    q = cudaq.qvector(1)
    theta = pi / 2
    rx(theta, q[0])
    theta = pi
    ry(theta, q[0])
    mz(q)
"""

    circuit = from_framework(CudaqProgram(code=code, entry_point="rebound"), framework="cudaq")

    assert [gate.name for gate in circuit.gates] == ["rx", "ry"]
    assert np.allclose(
        circuit.gates[0].matrix,
        np.asarray(
            [[np.cos(np.pi / 4), -1j * np.sin(np.pi / 4)], [-1j * np.sin(np.pi / 4), np.cos(np.pi / 4)]],
            dtype=complex,
        ),
    )
    assert np.allclose(circuit.gates[1].matrix, np.asarray([[0, -1], [1, 0]], dtype=complex))


def test_cudaq_nonconstant_rotation_is_rejected() -> None:
    code = """
import cudaq


def dynamic(theta):
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        ry(theta, q[0])
        mz(q)

    return kernel
"""
    with pytest.raises(NotImplementedError, match="non-constant angle"):
        from_framework(CudaqProgram(code=code, entry_point="dynamic"), framework="cudaq")


def test_cudaq_unsupported_gate_call_is_rejected() -> None:
    code = """
import cudaq


@cudaq.kernel
def unsupported():
    q = cudaq.qvector(1)
    u3(0.1, 0.2, 0.3, q[0])
    mz(q)
"""
    with pytest.raises(NotImplementedError, match="unsupported CUDA-Q gate"):
        from_framework(CudaqProgram(code=code, entry_point="unsupported"), framework="cudaq")


def test_cudaq_non_static_control_flow_is_rejected() -> None:
    code = """
import cudaq


@cudaq.kernel
def branchy():
    q = cudaq.qvector(1)
    if True:
        h(q[0])
    mz(q)
"""
    with pytest.raises(NotImplementedError, match="control flow"):
        from_framework(CudaqProgram(code=code, entry_point="branchy"), framework="cudaq")


def test_qiskit_mid_circuit_measurement_is_rejected() -> None:
    qiskit = pytest.importorskip("qiskit")
    circuit = qiskit.QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.x(0)

    with pytest.raises(NotImplementedError, match="mid-circuit measurement"):
        from_framework(circuit, framework="qiskit")


def test_qiskit_gates_on_unmeasured_qubits_after_measurement_are_allowed() -> None:
    qiskit = pytest.importorskip("qiskit")
    circuit = qiskit.QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.x(1)

    converted = from_framework(circuit, framework="qiskit")

    assert [gate.name for gate in converted.gates] == ["h", "x"]
    assert converted.gates[0].wires == (0,)
    assert converted.gates[1].wires == (1,)


def test_qiskit_control_flow_is_rejected() -> None:
    qiskit = pytest.importorskip("qiskit")
    circuit = qiskit.QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(0)

    with pytest.raises(NotImplementedError, match="control-flow|classically controlled"):
        from_framework(circuit, framework="qiskit")


def test_cirq_mid_circuit_measurement_is_rejected() -> None:
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit(0)
    native = cirq.Circuit(cirq.H(q), cirq.measure(q, key="m"), cirq.X(q))

    with pytest.raises(NotImplementedError, match="mid-circuit measurement"):
        from_framework(native, framework="cirq")


def test_cirq_gates_on_unmeasured_qubits_after_measurement_are_allowed() -> None:
    """Cirq may schedule measurements earliest; disjoint later gates remain unitary."""

    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(2)
    native = cirq.Circuit(cirq.H(q[0]), cirq.measure(q[0], key="m"), cirq.X(q[1]))
    circuit = from_framework(native, framework="cirq")

    assert [gate.name for gate in circuit.gates] == ["H", "X"]
    assert circuit.gates[0].wires == (0,)
    assert circuit.gates[1].wires == (1,)


def test_cirq_unitary_operation_without_gate_is_converted() -> None:
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(2)
    operation = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1])))
    native = cirq.Circuit(operation)

    circuit = from_framework(native, framework="cirq")
    expected_big_endian = native.unitary(qubit_order=q)
    permutation = [0, 2, 1, 3]
    expected = expected_big_endian[np.ix_(permutation, permutation)]

    assert np.allclose(full_unitary(circuit), expected)


def test_cirq_reset_is_rejected_as_non_unitary() -> None:
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit(0)
    native = cirq.Circuit(cirq.reset(q))

    with pytest.raises(NotImplementedError, match="non-unitary"):
        from_framework(native, framework="cirq")


def test_pennylane_mid_circuit_measurement_is_rejected() -> None:
    qml = pytest.importorskip("pennylane")
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.measure(0)
        qml.PauliX(wires=0)
        qml.probs(wires=[0])

    with pytest.raises(NotImplementedError, match="mid-circuit measurement"):
        from_framework(tape, framework="pennylane")


def test_pennylane_gates_on_unmeasured_wires_after_measurement_are_allowed() -> None:
    qml = pytest.importorskip("pennylane")
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.measure(0)
        qml.PauliX(wires=1)
        qml.probs(wires=[0, 1])

    circuit = from_framework(tape, framework="pennylane")

    assert [gate.name for gate in circuit.gates] == ["Hadamard", "PauliX"]
    assert circuit.gates[0].wires == (0,)
    assert circuit.gates[1].wires == (1,)


def test_cudaq_mid_circuit_measurement_is_rejected() -> None:
    code = """
import cudaq


@cudaq.kernel
def mid():
    q = cudaq.qvector(1)
    h(q[0])
    mz(q[0])
    x(q[0])
"""
    with pytest.raises(NotImplementedError, match="mid-circuit measurement"):
        from_framework(CudaqProgram(code=code, entry_point="mid"), framework="cudaq")


def test_cudaq_whole_register_mid_circuit_measurement_is_rejected() -> None:
    code = """
import cudaq


@cudaq.kernel
def mid():
    q = cudaq.qvector(2)
    h(q[0])
    mz(q)
    x(q[1])
"""
    with pytest.raises(NotImplementedError, match="mid-circuit measurement"):
        from_framework(CudaqProgram(code=code, entry_point="mid"), framework="cudaq")


def test_cudaq_gates_on_unmeasured_qubits_after_measurement_are_allowed() -> None:
    code = """
import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    h(q[0])
    mz(q[0])
    x(q[1])
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="kernel"), framework="cudaq")

    assert [gate.name for gate in circuit.gates] == ["h", "x"]
    assert circuit.gates[0].wires == (0,)
    assert circuit.gates[1].wires == (1,)


def test_cudaq_resolves_the_kernel_returned_by_the_requested_entry_point() -> None:
    code = """
import cudaq


@cudaq.kernel
def decoy():
    q = cudaq.qvector(1)
    h(q[0])


@cudaq.kernel
def actual():
    q = cudaq.qvector(2)
    x(q[1])


def requested():
    return actual
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="requested"), framework="cudaq")

    assert circuit.num_qubits == 2
    assert [(gate.name, gate.wires) for gate in circuit.gates] == [("x", (1,))]


def test_cudaq_multiple_registers_are_rejected_instead_of_aliasing_wires() -> None:
    code = """
import cudaq


@cudaq.kernel
def kernel():
    left = cudaq.qvector(1)
    right = cudaq.qvector(1)
    h(left[0])
    x(right[0])
"""
    with pytest.raises(NotImplementedError, match="exactly one"):
        from_framework(CudaqProgram(code=code, entry_point="kernel"), framework="cudaq")


def test_cudaq_rz_matches_documented_symmetric_z_rotation() -> None:
    from qceval.frameworks.cudaq.values import _rotation_matrix

    theta = 0.7
    expected_rz = np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
    expected_r1 = np.diag([1.0, np.exp(1j * theta)])

    assert np.allclose(_rotation_matrix("rz", theta), expected_rz)
    assert np.allclose(_rotation_matrix("r1", theta), expected_r1)
    assert not np.allclose(_rotation_matrix("rz", theta), expected_r1)


def test_cudaq_rz_and_r1_are_not_conflated_when_controlled() -> None:
    rz_code = """
import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    rz.ctrl(0.7, q[0], q[1])
    mz(q)
"""
    r1_code = """
import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    r1.ctrl(0.7, q[0], q[1])
    mz(q)
"""
    rz_circuit = from_framework(CudaqProgram(code=rz_code, entry_point="kernel"), framework="cudaq")
    r1_circuit = from_framework(CudaqProgram(code=r1_code, entry_point="kernel"), framework="cudaq")

    assert not _equivalent(rz_circuit, r1_circuit)


def test_cudaq_uncontrolled_rz_matches_r1_up_to_global_phase() -> None:
    rz_code = """
import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(1)
    rz(0.7, q[0])
    mz(q)
"""
    r1_code = """
import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(1)
    r1(0.7, q[0])
    mz(q)
"""
    rz_circuit = from_framework(CudaqProgram(code=rz_code, entry_point="kernel"), framework="cudaq")
    r1_circuit = from_framework(CudaqProgram(code=r1_code, entry_point="kernel"), framework="cudaq")

    assert _equivalent(rz_circuit, r1_circuit)


def test_cudaq_controlled_rotation_and_adjoint_are_converted() -> None:
    code = """
import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    rz.ctrl(0.5, q[0], q[1])
    ry.adj(0.25, q[1])
    mz(q)
"""
    circuit = from_framework(CudaqProgram(code=code, entry_point="kernel"), framework="cudaq")

    assert [gate.name for gate in circuit.gates] == ["rz", "ry"]
    assert tuple(control.wire for control in circuit.gates[0].controls) == (0,)
    assert circuit.gates[0].targets == (1,)
    assert np.allclose(
        circuit.gates[1].matrix,
        np.asarray([[np.cos(0.125), np.sin(0.125)], [-np.sin(0.125), np.cos(0.125)]], dtype=complex),
    )
