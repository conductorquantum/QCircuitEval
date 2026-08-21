from __future__ import annotations

import pytest

from qceval.frameworks.qiskit import counts_to_array, execute_qiskit_task
from qceval.frameworks.qiskit import executor as qiskit_eval


def test_qiskit_executes_circuit_and_counts_paths() -> None:
    # Arrange
    circuit_code = (
        "from qiskit import QuantumCircuit\n"
        "def answer():\n"
        "    qc=QuantumCircuit(1,1); qc.x(0); qc.measure(0,0); return qc\n"
    )
    counts_code = "def answer():\n    return {'0': 1, '1': 3}\n"

    # Act
    circuit = execute_qiskit_task(task_id="01", code=circuit_code, entry_point="answer", inputs={})
    counts = execute_qiskit_task(task_id="01", code=counts_code, entry_point="answer", inputs={})

    # Assert
    assert circuit.probabilities == [0.0, 1.0]
    assert circuit.metadata["measurement_count"] == 1
    assert counts.probabilities == [0.25, 0.75]
    assert counts.metadata["returned_counts"] is True


def test_qiskit_rejects_bad_return_and_bad_counts() -> None:
    # Arrange
    bad_code = "def answer():\n    return 1\n"

    # Act
    with pytest.raises(TypeError):
        execute_qiskit_task(task_id="01", code=bad_code, entry_point="answer", inputs={})
    with pytest.raises(TypeError):
        counts_to_array(1)  # type: ignore[arg-type]

    # Assert
    assert counts_to_array([{"0": 1, "1": 1}]).tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        counts_to_array({})


def test_qiskit_fallback_and_unitary_error_paths(monkeypatch) -> None:
    # Arrange
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    monkeypatch.setattr(qiskit_eval, "exact_probabilities", lambda circuit: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.setattr(qiskit_eval, "circuit_unitary", lambda circuit: None)

    # Act
    result = qiskit_eval._execute_circuit(circuit)

    # Assert
    assert result.metadata["probability_method"] == "qasm_fallback"
    assert result.unitary is None


def test_qiskit_qasm_probabilities_direct() -> None:
    # Arrange
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    # Act
    probabilities = qiskit_eval._qasm_probabilities(circuit)

    # Assert
    assert probabilities.tolist() == [1.0, 0.0]


def test_qiskit_metadata_emits_gate_families_and_interactions() -> None:
    # Arrange
    code = """
from qiskit import QuantumCircuit


def answer():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    return qc
"""

    # Act
    result = execute_qiskit_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    metadata = result.metadata
    assert metadata["gate_family_counts"]["h"] == 1
    assert metadata["gate_family_counts"]["cx"] == 1
    assert metadata["gate_family_counts"]["ccx"] == 1
    interactions = {tuple(pair) for pair in metadata["interaction_pairs"]}
    assert {(0, 1), (0, 2), (1, 2)} <= interactions


def test_qiskit_mid_circuit_measurement_yields_exact_collapsed_distribution() -> None:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.h(0)

    result = qiskit_eval._execute_circuit(circuit)

    assert result.probabilities == pytest.approx([0.5, 0.5], abs=1e-12)
    assert result.metadata["probability_method"] == "deferred_statevector"
    assert result.statevector is None


def test_qiskit_terminal_measurements_keep_statevector_method() -> None:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    result = qiskit_eval._execute_circuit(circuit)

    assert result.probabilities == pytest.approx([0.5, 0.0, 0.0, 0.5], abs=1e-12)
    assert result.metadata["probability_method"] == "statevector"
    assert qiskit_eval.requires_measurement_deferral(circuit) is False


def test_qiskit_measure_reset_reuse_is_exact() -> None:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.reset(0)
    circuit.h(0)
    circuit.measure(0, 1)

    result = qiskit_eval._execute_circuit(circuit)

    assert result.probabilities == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=1e-12)
    assert result.metadata["probability_method"] == "deferred_statevector"


def test_qiskit_entangled_mid_measurement_keeps_correlations() -> None:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.h(0)
    circuit.measure(1, 1)

    result = qiskit_eval._execute_circuit(circuit)

    assert result.probabilities == pytest.approx([0.5, 0.0, 0.0, 0.5], abs=1e-12)
    assert result.metadata["probability_method"] == "deferred_statevector"


def test_qiskit_classical_feedback_is_marked_sampled_not_exact() -> None:
    from qiskit import QuantumCircuit

    from qceval.semantics.verifiers.distribution_materializers import AdaptiveDistributionMaterializer

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(1)
    circuit.measure(1, 1)

    result = qiskit_eval._execute_circuit(circuit)

    assert qiskit_eval.requires_measurement_deferral(circuit) is True
    assert result.metadata["probability_method"] in AdaptiveDistributionMaterializer._SAMPLED_METHODS


def test_qiskit_overwritten_classical_bit_uses_last_write() -> None:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.measure(1, 0)

    result = qiskit_eval._execute_circuit(circuit)

    assert result.probabilities == pytest.approx([1.0, 0.0], abs=1e-12)
    assert result.metadata["probability_method"] == "deferred_statevector"
