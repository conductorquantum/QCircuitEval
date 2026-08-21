from __future__ import annotations

import pytest

from qceval.frameworks.cirq import execute_cirq_task
from qceval.frameworks.cirq import metadata as cirq_eval


def test_cirq_executes_circuit_and_counts_paths() -> None:
    # Arrange
    pytest.importorskip("cirq")
    circuit_code = (
        "import cirq\n"
        "def answer():\n"
        "    q=cirq.LineQubit(0); return cirq.Circuit(cirq.X(q), cirq.measure(q, key='result'))\n"
    )
    counts_code = "def answer():\n    return {'0': 1, '1': 3}\n"

    # Act
    circuit = execute_cirq_task(task_id="01", code=circuit_code, entry_point="answer", inputs={})
    counts = execute_cirq_task(task_id="01", code=counts_code, entry_point="answer", inputs={})

    # Assert
    assert circuit.probabilities == [0.0, 1.0]
    assert circuit.metadata["measurement_count"] == 1
    assert counts.probabilities == [0.25, 0.75]


def test_cirq_rejects_bad_return() -> None:
    # Arrange
    pytest.importorskip("cirq")
    code = "def answer():\n    return 1\n"

    # Act
    with pytest.raises(TypeError) as exc:
        execute_cirq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert "Expected Cirq Circuit" in str(exc.value)


def test_cirq_unitary_error_path(monkeypatch) -> None:
    # Arrange
    cirq = pytest.importorskip("cirq")
    circuit = cirq.Circuit()
    monkeypatch.setattr(cirq_eval.cirq, "unitary", lambda circuit: (_ for _ in ()).throw(ValueError("boom")))

    # Act
    unitary = cirq_eval.circuit_unitary(circuit)

    # Assert
    assert unitary is None


def test_cirq_metadata_emits_gate_families_and_interactions() -> None:
    # Arrange
    pytest.importorskip("cirq")
    code = """
import cirq


def answer():
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CCX(q[0], q[1], q[2]))
"""

    # Act
    result = execute_cirq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    metadata = result.metadata
    assert metadata["gate_family_counts"]["h"] == 1
    assert metadata["gate_family_counts"]["cx"] == 1
    assert metadata["gate_family_counts"]["ccx"] == 1
    interactions = {tuple(pair) for pair in metadata["interaction_pairs"]}
    assert {(0, 1), (0, 2), (1, 2)} <= interactions


def test_cirq_deferred_multi_key_bit_order_matches_program_ir_path() -> None:
    # Arrange: asymmetric feed-forward outcome a=1, b=0 distinguishes the two
    # historical bit conventions ("01" versus "10").
    cirq = pytest.importorskip("cirq")
    import numpy as np

    from qceval.frameworks.cirq.adapter import CirqLoweringAdapter
    from qceval.semantics.lowering.base import SourceMetadata
    from qceval.semantics.verifiers.dynamic import ExactBranchSimulator

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key="a"),
        cirq.X(q1).with_classical_controls("a"),
        cirq.X(q1),
        cirq.measure(q1, key="b"),
    )

    # Act
    deferred = cirq_eval._deferred_probabilities(circuit)
    lowering = CirqLoweringAdapter().lower(circuit, SourceMetadata("cirq", source_hash="c" * 64, backend="cpu"), None)

    # Assert: the Program IR path renders a=1, b=0 as "01" (first measured key
    # least significant, matching Qiskit classical register packing) and the
    # deferred executor path must agree.
    assert lowering.program is not None
    branches = ExactBranchSimulator().run(lowering.program, max_branches=8)
    assert len(branches) == 1
    rendered = "".join(str(branches[0].classical_bits[index]) for index in lowering.program.classical_render_order)
    assert rendered == "01"
    expected = np.zeros(4)
    expected[int(rendered, 2)] = 1.0
    assert np.allclose(deferred, expected)


def test_cirq_invert_mask_measurement_grades_as_recording_one() -> None:
    # Arrange: measuring |0> with invert_mask=(True,) must record 1.
    pytest.importorskip("cirq")
    code = (
        "import cirq\n"
        "def answer():\n"
        "    q=cirq.LineQubit(0)\n"
        "    return cirq.Circuit(cirq.measure(q, key='result', invert_mask=(True,)))\n"
    )

    # Act
    result = execute_cirq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.probabilities == [0.0, 1.0]
