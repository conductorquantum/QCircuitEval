"""Program IR invariants and Qiskit bridge parity tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qceval.evals.ir import full_unitary
from qceval.frameworks.qiskit.lowering import QiskitLoweringAdapter
from qceval.semantics.ir import (
    IR_VERSION,
    ClassicalCondition,
    IRValidationError,
    IRValidationLimits,
    Operation,
    OperationKind,
    Parameter,
    ParameterKind,
    Program,
    Provenance,
    program_hash,
    validate_program,
)
from qceval.semantics.lowering.base import LoweringStatus, SourceMetadata
from qceval.semantics.lowering.bridge import lower_qiskit_with_legacy_bridge


def _metadata(source_hash: str = "a" * 64) -> SourceMetadata:
    return SourceMetadata("qiskit", source_hash=source_hash, backend="statevector")


def test_qiskit_lowering_preserves_wires_measurements_and_phase() -> None:
    circuit = QuantumCircuit(3, 2, global_phase=0.25)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.measure(2, 0)
    circuit.measure(0, 1)

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)

    assert result.status is LoweringStatus.SUCCESS
    assert result.program is not None
    assert result.program.num_qubits == 3
    assert result.program.num_clbits == 2
    assert result.program.classical_render_order == (1, 0)
    assert result.program.global_phase == Parameter(ParameterKind.NUMBER, "0.25")
    assert [(item.kind, item.quantum_wires, item.classical_bits) for item in result.program.operations] == [
        (OperationKind.GATE, (0,), ()),
        (OperationKind.GATE, (2,), ()),
        (OperationKind.MEASUREMENT, (2,), (0,)),
        (OperationKind.MEASUREMENT, (0,), (1,)),
    ]
    assert result.program.operations[1].controls[0].wire == 0


def test_qiskit_custom_gate_behavior_is_preserved_as_dense_payload() -> None:
    definition = QuantumCircuit(1, name="custom_h")
    definition.h(0)
    circuit = QuantumCircuit(1)
    circuit.append(definition.to_gate(), [0])

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)

    assert result.program is not None
    operation = result.program.operations[0]
    assert operation.name == "dense_unitary"
    assert "matrix_complex128_hex" in dict(operation.semantic_data)


def test_qiskit_large_untrusted_gate_fails_closed() -> None:
    definition = QuantumCircuit(7, name="large_custom")
    circuit = QuantumCircuit(7)
    circuit.append(definition.to_gate(), range(7))

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)

    assert result.status is LoweringStatus.UNSUPPORTED
    assert result.error is not None
    assert result.error.reason == "custom_unitary_exceeds_dense_limit"


def test_qiskit_large_composite_gate_uses_its_inspectable_definition() -> None:
    from qiskit.circuit.library import QFT

    circuit = QuantumCircuit(8)
    with pytest.warns(DeprecationWarning):
        circuit.append(QFT(8, inverse=True), range(8))

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)

    assert result.status is LoweringStatus.SUCCESS
    assert result.program is not None
    assert result.program.operations[0].definition


def test_qiskit_gate_built_custom_matrix_records_verified_origin() -> None:
    definition = QuantumCircuit(2)
    definition.h(0)
    definition.cz(0, 1)
    circuit = QuantumCircuit(2)
    circuit.append(definition.to_gate(), [0, 1])

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)

    assert result.program is not None
    assert dict(result.program.operations[0].semantic_data)["matrix_origin"] == "qiskit_gate_definition"


def test_qiskit_builtin_multi_controlled_x_keeps_structural_controls() -> None:
    circuit = QuantumCircuit(5)
    circuit.mcx([0, 1, 2, 3], 4)

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)

    assert result.status is LoweringStatus.SUCCESS
    assert result.program is not None
    operation = result.program.operations[0]
    assert operation.name == "mcx"
    assert operation.quantum_wires == (4,)
    assert tuple(control.wire for control in operation.controls) == (0, 1, 2, 3)


def test_qiskit_single_branch_if_is_lowered_to_a_classical_condition() -> None:
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)

    assert result.status is LoweringStatus.SUCCESS
    assert result.program is not None
    assert result.program.operations[0].condition is not None
    assert result.program.operations[0].condition.bits == (0,)
    assert result.program.operations[0].condition.value == 1


def test_qiskit_if_else_remains_typed_unsupported() -> None:
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test((circuit.clbits[0], True)) as else_:
        circuit.x(0)
    with else_:
        circuit.z(0)

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)

    assert result.status is LoweringStatus.UNSUPPORTED
    assert result.error is not None
    assert result.error.reason == "unsupported_if_else_shape"


def test_invalid_return_type_is_execution_error() -> None:
    result = QiskitLoweringAdapter().lower({"not": "a circuit"}, _metadata(), None)

    assert result.status is LoweringStatus.EXECUTION_ERROR
    assert result.error is not None
    assert result.error.reason == "invalid_return_type"


def test_program_hash_excludes_provenance_diagnostics_and_barriers() -> None:
    gate = Operation(OperationKind.GATE, "cnot", quantum_wires=(1,), controls=())
    barrier = Operation(OperationKind.BARRIER, "barrier", quantum_wires=(0, 1))
    first = Program(
        IR_VERSION,
        2,
        0,
        (barrier, gate),
        None,
        (),
        Provenance("qiskit", "1", source_hash="a" * 64),
        diagnostics=("first",),
    )
    second = replace(
        first,
        operations=(replace(gate, name="cx"),),
        provenance=Provenance("qiskit", "2", source_hash="b" * 64),
        diagnostics=("second",),
    )

    assert program_hash(first) == program_hash(second)


def test_ir_validation_rejects_bad_measurement_and_preflights_limits() -> None:
    invalid = Program(
        IR_VERSION,
        1,
        1,
        (Operation(OperationKind.MEASUREMENT, "measure", quantum_wires=(0,), classical_bits=()),),
        None,
        (0,),
        Provenance("fixture", "1"),
    )
    with pytest.raises(IRValidationError, match="equal nonempty"):
        validate_program(invalid)

    empty = replace(invalid, operations=())
    with pytest.raises(IRValidationError, match="outside configured limit"):
        validate_program(empty, IRValidationLimits(max_qubits=0))


def test_qiskit_legacy_bridge_preserves_dense_unitary_behavior() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.h(0)

    bridged = lower_qiskit_with_legacy_bridge(circuit, _metadata())

    assert bridged.lowering.status is LoweringStatus.SUCCESS
    assert bridged.lowering.program is not None
    assert bridged.legacy_circuit is not None
    expected = np.asarray(Operator(circuit).data, dtype=complex)
    assert np.allclose(full_unitary(bridged.legacy_circuit), expected, atol=1e-12)


def _run_lowered(circuit: QuantumCircuit, max_branches: int = 8):
    from qceval.semantics.verifiers.dynamic import ExactBranchSimulator

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)
    assert result.status is LoweringStatus.SUCCESS
    assert result.program is not None
    validate_program(result.program)
    return ExactBranchSimulator().run(result.program, max_branches=max_branches)


def test_reset_inside_false_if_test_branch_is_skipped() -> None:
    circuit = QuantumCircuit(2, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.x(1)
    with circuit.if_test((circuit.clbits[0], 0)):
        circuit.reset(1)

    branches = _run_lowered(circuit)

    assert len(branches) == 1
    branch = branches[0]
    assert branch.classical_bits == (1,)
    # Reset must NOT be applied: q1 stays |1>, so basis index 0b11 == 3.
    expected = np.zeros(4, dtype=complex)
    expected[3] = 1.0
    assert np.allclose(branch.statevector, expected, atol=1e-12)


def test_reset_inside_true_if_test_branch_is_applied() -> None:
    circuit = QuantumCircuit(2, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.x(1)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.reset(1)

    branches = _run_lowered(circuit)

    assert len(branches) == 1
    branch = branches[0]
    assert branch.classical_bits == (1,)
    # Reset applied: q1 back to |0>, q0 stays |1>, basis index 0b01 == 1.
    expected = np.zeros(4, dtype=complex)
    expected[1] = 1.0
    assert np.allclose(branch.statevector, expected, atol=1e-12)


def test_measurement_inside_false_if_test_branch_retains_prior_clbit() -> None:
    # Matches Qiskit AerSimulator semantics: a measure inside an untaken
    # if_test body never executes, so the target clbit keeps its prior value.
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.x(1)
    with circuit.if_test((circuit.clbits[0], 0)):
        circuit.measure(1, 1)

    branches = _run_lowered(circuit)

    assert len(branches) == 1
    branch = branches[0]
    assert branch.classical_bits == (1, 0)
    expected = np.zeros(4, dtype=complex)
    expected[3] = 1.0
    assert np.allclose(branch.statevector, expected, atol=1e-12)


def test_measurement_inside_true_if_test_branch_writes_clbit() -> None:
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.x(1)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.measure(1, 1)

    branches = _run_lowered(circuit)

    assert len(branches) == 1
    branch = branches[0]
    assert branch.classical_bits == (1, 1)
    expected = np.zeros(4, dtype=complex)
    expected[3] = 1.0
    assert np.allclose(branch.statevector, expected, atol=1e-12)


def test_conditional_measurement_on_superposed_control_splits_correctly() -> None:
    # h(0); measure(0,0); then conditionally measure q1 (in |1>) into c1 only
    # when c0 == 1. The c0 == 0 branch must leave c1 at its prior value 0.
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.x(1)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.measure(1, 1)

    branches = _run_lowered(circuit)

    outcomes = {branch.classical_bits: branch.probability for branch in branches}
    assert set(outcomes) == {(0, 0), (1, 1)}
    assert outcomes[(0, 0)] == pytest.approx(0.5, abs=1e-12)
    assert outcomes[(1, 1)] == pytest.approx(0.5, abs=1e-12)


def test_conditional_state_preparation_respects_condition() -> None:
    from qceval.semantics.lowering.utils import bounded_statevector_semantic_data
    from qceval.semantics.verifiers.dynamic import ExactBranchSimulator

    def _program(condition_value: int) -> Program:
        preparation = Operation(
            OperationKind.STATE_PREPARATION,
            "statevector",
            quantum_wires=(0,),
            semantic_data=bounded_statevector_semantic_data(np.asarray([0.0, 1.0])),
            condition=ClassicalCondition(bits=(0,), value=condition_value),
        )
        return Program(
            IR_VERSION,
            1,
            1,
            (preparation,),
            None,
            (0,),
            Provenance("qiskit", "1", source_hash="a" * 64),
        )

    skipped = ExactBranchSimulator().run(_program(1), max_branches=4)
    assert len(skipped) == 1
    assert np.allclose(skipped[0].statevector, [1.0, 0.0], atol=1e-12)

    applied = ExactBranchSimulator().run(_program(0), max_branches=4)
    assert len(applied) == 1
    assert np.allclose(applied[0].statevector, [0.0, 1.0], atol=1e-12)


def _phase_aligned_error(actual: np.ndarray, expected: np.ndarray) -> float:
    anchor = int(np.argmax(np.abs(expected)))
    return float(np.max(np.abs(actual * (expected[anchor] / actual[anchor]) - expected)))


def _qft_probe_circuit(num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.rx(0.3 * (qubit + 1), qubit)
    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


@pytest.mark.parametrize("num_qubits", [7, 8])
def test_qiskit_large_named_qft_matches_native_operator(num_qubits: int) -> None:
    from qiskit.circuit.library import QFTGate
    from qiskit.quantum_info import Statevector

    circuit = _qft_probe_circuit(num_qubits)
    circuit.append(QFTGate(num_qubits), range(num_qubits))

    branches = _run_lowered(circuit, max_branches=1)
    expected = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)

    assert len(branches) == 1
    assert _phase_aligned_error(branches[0].statevector, expected) < 1e-9


def test_qiskit_six_qubit_qft_still_exact() -> None:
    from qiskit.circuit.library import QFTGate
    from qiskit.quantum_info import Statevector

    circuit = _qft_probe_circuit(6)
    circuit.append(QFTGate(6), range(6))

    branches = _run_lowered(circuit, max_branches=1)
    expected = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)

    assert _phase_aligned_error(branches[0].statevector, expected) < 1e-9


@pytest.mark.parametrize("num_qubits", [7, 8])
def test_qiskit_large_inverse_named_qft_matches_native_operator(num_qubits: int) -> None:
    from qiskit.circuit.library import QFTGate
    from qiskit.quantum_info import Statevector

    circuit = _qft_probe_circuit(num_qubits)
    circuit.append(QFTGate(num_qubits).inverse(), range(num_qubits))

    branches = _run_lowered(circuit, max_branches=1)
    expected = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)

    assert _phase_aligned_error(branches[0].statevector, expected) < 1e-9


@pytest.mark.parametrize(
    "options",
    [{}, {"do_swaps": False}, {"inverse": True}, {"do_swaps": False, "inverse": True}],
)
def test_qiskit_deprecated_qft_variants_lower_through_definition(options: dict) -> None:
    # The deprecated QFT circuit class shares the "qft" instruction name but
    # its variants (do_swaps=False, inverse) do not implement the full DFT the
    # shared named-QFT primitive assumes; they must never reach it by name.
    from qiskit.quantum_info import Statevector

    circuit = _qft_probe_circuit(7)
    with pytest.warns(DeprecationWarning):
        from qiskit.circuit.library import QFT

        circuit.append(QFT(7, **options), range(7))

    result = QiskitLoweringAdapter().lower(circuit, _metadata(), None)
    assert result.status is LoweringStatus.SUCCESS
    assert result.program is not None
    assert result.program.operations[-1].name != "qft"

    from qceval.semantics.verifiers.dynamic import ExactBranchSimulator

    branches = ExactBranchSimulator().run(result.program, max_branches=1)
    expected = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)
    assert _phase_aligned_error(branches[0].statevector, expected) < 1e-9
