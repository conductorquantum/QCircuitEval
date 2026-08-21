"""Static cross-framework Program IR and legacy-bridge conformance."""

# ruff: noqa: F821

from __future__ import annotations

import cudaq
import numpy as np
import pytest
from qiskit import QuantumCircuit

from qceval.evals.ir import full_unitary
from qceval.frameworks.cirq.lowering import CirqLoweringAdapter
from qceval.frameworks.cudaq.lowering import CudaqLoweringAdapter
from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.frameworks.pennylane.lowering import PennyLaneLoweringAdapter
from qceval.semantics.ir import OperationKind
from qceval.semantics.lowering.base import LoweringStatus, SourceMetadata
from qceval.semantics.lowering.bridge import (
    lower_cirq_with_legacy_bridge,
    lower_cudaq_with_legacy_bridge,
    lower_pennylane_with_legacy_bridge,
    lower_qiskit_with_legacy_bridge,
)
from qceval.semantics.lowering.registry import default_lowering_registry
from qceval.semantics.verifiers.dynamic import ExactBranchSimulator


def _metadata(framework: str) -> SourceMetadata:
    return SourceMetadata(framework, source_hash=framework[0] * 64, backend="cpu")


def _cudaq_source(*, measured: bool = False) -> str:
    measurement = "    mz(q[0])\n    mz(q[1])\n" if measured else ""
    return (
        "import cudaq\n\n"
        "@cudaq.kernel\n"
        "def bell():\n"
        "    q = cudaq.qvector(2)\n"
        "    h(q[0])\n"
        "    x.ctrl(q[0], q[1])\n"
        f"{measurement}"
    )


@cudaq.kernel
def _cudaq_bell_kernel():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@cudaq.kernel
def _cudaq_measured_bell_kernel():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q[0])
    mz(q[1])


def test_bell_materialization_agrees_across_all_four_legacy_bridges() -> None:
    import cirq
    import pennylane as qml

    qiskit = QuantumCircuit(2)
    qiskit.h(0)
    qiskit.cx(0, 1)
    q0, q1 = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    tape = qml.tape.QuantumScript([qml.Hadamard(0), qml.CNOT([0, 1])], [qml.state()])
    cudaq_program = CudaqProgram(_cudaq_source(), "bell", kernel=_cudaq_bell_kernel)

    bridges = [
        lower_qiskit_with_legacy_bridge(qiskit, _metadata("qiskit")),
        lower_cirq_with_legacy_bridge(cirq_circuit, _metadata("cirq")),
        lower_pennylane_with_legacy_bridge(tape, _metadata("pennylane")),
        lower_cudaq_with_legacy_bridge(cudaq_program, _metadata("cudaq")),
    ]

    assert all(item.lowering.status is LoweringStatus.SUCCESS for item in bridges)
    unitaries = [full_unitary(item.legacy_circuit) for item in bridges if item.legacy_circuit is not None]
    assert len(unitaries) == 4
    assert all(np.allclose(unitaries[0], unitary, atol=1e-12) for unitary in unitaries[1:])


def test_cirq_terminal_measurement_preserves_key_order_and_invert_mask() -> None:
    import cirq

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q1, q0, key="result", invert_mask=(True, False)))

    result = CirqLoweringAdapter().lower(circuit, _metadata("cirq"), None)

    assert result.program is not None
    measurement = result.program.operations[-1]
    assert measurement.kind is OperationKind.MEASUREMENT
    assert measurement.quantum_wires == (1, 0)
    assert measurement.classical_bits == (0, 1)
    assert dict(measurement.semantic_data) == {"invert_mask": "10", "key": "result"}


def test_pennylane_terminal_probability_process_is_explicit() -> None:
    import pennylane as qml

    tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.probs(wires=[1, 0])])

    result = PennyLaneLoweringAdapter().lower(tape, _metadata("pennylane"), None)

    assert result.program is not None
    measurement = result.program.operations[-1]
    assert measurement.kind is OperationKind.MEASUREMENT
    assert measurement.name == "probabilities"
    assert measurement.quantum_wires == (1, 0)
    assert measurement.classical_bits == (0, 1)


def test_cudaq_source_replay_preserves_terminal_measurement_order() -> None:
    program = CudaqProgram(_cudaq_source(measured=True), "bell", kernel=_cudaq_measured_bell_kernel)

    result = CudaqLoweringAdapter().lower(program, _metadata("cudaq"), None)

    assert result.program is not None
    measurements = [item for item in result.program.operations if item.kind is OperationKind.MEASUREMENT]
    assert [item.quantum_wires for item in measurements] == [(0,), (1,)]
    assert [item.classical_bits for item in measurements] == [(0,), (1,)]
    assert result.program.classical_render_order == (1, 0)


def test_nonunitary_or_dynamic_features_fail_with_typed_capabilities() -> None:
    import cirq
    import pennylane as qml

    qubit = cirq.LineQubit(0)
    cirq_result = CirqLoweringAdapter().lower(
        cirq.Circuit(cirq.depolarize(0.1)(qubit)),
        _metadata("cirq"),
        None,
    )
    tape = qml.tape.QuantumScript([qml.DepolarizingChannel(0.1, wires=0)], [qml.state()])
    pennylane_result = PennyLaneLoweringAdapter().lower(tape, _metadata("pennylane"), None)
    dynamic_source = (
        "import cudaq\n\n@cudaq.kernel\ndef dynamic():\n    q = cudaq.qvector(1)\n    if True:\n        x(q[0])\n"
    )
    cudaq_result = CudaqLoweringAdapter().lower(
        CudaqProgram(dynamic_source, "dynamic"),
        _metadata("cudaq"),
        None,
    )

    assert cirq_result.status is LoweringStatus.UNSUPPORTED
    assert cirq_result.error is not None and cirq_result.error.reason == "unsupported_channel_or_mixture"
    assert pennylane_result.status is LoweringStatus.UNSUPPORTED
    assert pennylane_result.error is not None
    assert pennylane_result.error.reason == "unsupported_nonunitary_operation"
    assert cudaq_result.status is LoweringStatus.UNSUPPORTED
    assert cudaq_result.error is not None and cudaq_result.error.reason == "unsupported_cudaq_qir"


def test_default_registry_exposes_four_typed_adapters_and_fingerprints() -> None:
    registry = default_lowering_registry()
    fingerprints = {
        name: registry.get(name).framework_fingerprint().framework for name in ("qiskit", "cirq", "pennylane", "cudaq")
    }

    assert fingerprints == {name: name for name in fingerprints}
    assert all(registry.get(name).capabilities().features for name in fingerprints)


def test_cirq_half_turn_exponents_match_radian_rotations_across_frameworks() -> None:
    import cirq
    import pennylane as qml

    qiskit = QuantumCircuit(2)
    qiskit.h(0)
    qiskit.rx(0.37, 0)
    qiskit.rz(-0.2, 1)
    qiskit.cp(0.41, 0, 1)
    q0, q1 = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.rx(0.37)(q0),
        cirq.rz(-0.2)(q1),
        (cirq.CZ ** (0.41 / np.pi))(q0, q1),
    )
    tape = qml.tape.QuantumScript(
        [
            qml.Hadamard(0),
            qml.RX(0.37, 0),
            qml.RZ(-0.2, 1),
            qml.ControlledPhaseShift(0.41, [0, 1]),
        ],
        [qml.state()],
    )
    programs = [
        lower_qiskit_with_legacy_bridge(qiskit, _metadata("qiskit")).lowering.program,
        CirqLoweringAdapter().lower(cirq_circuit, _metadata("cirq"), None).program,
        PennyLaneLoweringAdapter().lower(tape, _metadata("pennylane"), None).program,
    ]
    assert all(program is not None for program in programs)
    states = [ExactBranchSimulator().run(program, max_branches=1)[0].statevector for program in programs if program]

    assert all(abs(np.vdot(states[0], state)) == pytest.approx(1.0) for state in states[1:])


def _little_endian(state: object, num_qubits: int) -> np.ndarray:
    """Reorder a big-endian framework statevector onto IR little-endian wires."""
    amplitudes = np.asarray(state, dtype=complex).reshape(-1)
    reordered = np.zeros_like(amplitudes)
    for index, amplitude in enumerate(amplitudes):
        reordered[int(format(index, f"0{num_qubits}b")[::-1], 2)] = amplitude
    return reordered


@pytest.mark.parametrize("gate_name", ["cswap", "iswap"])
def test_cirq_cswap_and_iswap_simulate_exactly(gate_name: str) -> None:
    import cirq

    qubits = cirq.LineQubit.range(3)
    if gate_name == "cswap":
        operations = [cirq.X(qubits[0]), cirq.H(qubits[1]), cirq.CSWAP(qubits[0], qubits[1], qubits[2])]
    else:
        operations = [cirq.X(qubits[0]), cirq.H(qubits[1]), cirq.ISWAP(qubits[0], qubits[1])]
    circuit = cirq.Circuit(operations)

    result = CirqLoweringAdapter().lower(circuit, _metadata("cirq"), None)

    assert result.status is LoweringStatus.SUCCESS and result.program is not None
    simulated = ExactBranchSimulator().run(result.program, max_branches=1)[0].statevector
    native = cirq.final_state_vector(circuit, qubit_order=sorted(circuit.all_qubits()))
    expected = _little_endian(native, len(circuit.all_qubits()))
    assert abs(np.vdot(expected, simulated)) == pytest.approx(1.0)


@pytest.mark.parametrize("gate_name", ["iswap", "adjoint_s", "adjoint_t"])
def test_pennylane_iswap_and_adjoint_phase_gates_simulate_exactly(gate_name: str) -> None:
    import pennylane as qml

    operations = {
        "iswap": [qml.PauliX(0), qml.Hadamard(1), qml.ISWAP([0, 1])],
        "adjoint_s": [qml.Hadamard(0), qml.Hadamard(1), qml.adjoint(qml.S(0))],
        "adjoint_t": [qml.Hadamard(0), qml.Hadamard(1), qml.adjoint(qml.T(0))],
    }[gate_name]
    tape = qml.tape.QuantumScript(operations, [qml.state()])

    result = PennyLaneLoweringAdapter().lower(tape, _metadata("pennylane"), None)

    assert result.status is LoweringStatus.SUCCESS and result.program is not None
    simulated = ExactBranchSimulator().run(result.program, max_branches=1)[0].statevector
    native = qml.execute([tape], qml.device("default.qubit", wires=2))[0]
    expected = _little_endian(native, 2)
    assert abs(np.vdot(expected, simulated)) == pytest.approx(1.0)


def test_cirq_invert_mask_measurement_records_inverted_bit_in_branch_simulation() -> None:
    import cirq

    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(qubit, key="result", invert_mask=(True,)))

    result = CirqLoweringAdapter().lower(circuit, _metadata("cirq"), None)

    assert result.status is LoweringStatus.SUCCESS and result.program is not None
    branches = ExactBranchSimulator().run(result.program, max_branches=4)
    assert len(branches) == 1
    # Measuring |0> under invert_mask=(True,) records 1 without exciting the qubit.
    assert branches[0].classical_bits == (1,)
    assert branches[0].statevector[0] == pytest.approx(1.0)
