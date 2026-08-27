from __future__ import annotations

import pytest

from qceval.frameworks.pennylane import execute_pennylane_task


def test_pennylane_executes_qnode_and_array_paths() -> None:
    # Arrange
    pytest.importorskip("pennylane")
    qnode_code = (
        "import pennylane as qml\n"
        "def answer():\n"
        "    dev=qml.device('default.qubit', wires=1, shots=None)\n"
        "    @qml.qnode(dev)\n"
        "    def circuit():\n"
        "        qml.PauliX(wires=0); return qml.probs(wires=[0])\n"
        "    return circuit()\n"
    )
    array_code = "import numpy as np\ndef answer():\n    return np.array([0.25, 0.75])\n"

    # Act
    qnode = execute_pennylane_task(task_id="01", code=qnode_code, entry_point="answer", inputs={})
    array = execute_pennylane_task(task_id="01", code=array_code, entry_point="answer", inputs={})

    # Assert
    assert qnode.probabilities == [0.0, 1.0]
    assert qnode.metadata["measurement_count"] == 1
    assert qnode.metadata["return_measurement_count"] == 1
    assert array.probabilities == [0.25, 0.75]


def test_pennylane_metadata_emits_gate_families_and_interactions() -> None:
    # Arrange
    pytest.importorskip("pennylane")
    code = """
import pennylane as qml


def answer():
    dev = qml.device("default.qubit", wires=3, shots=None)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Toffoli(wires=[0, 1, 2])
        return qml.probs(wires=[2, 1, 0])

    return circuit()
"""

    # Act
    result = execute_pennylane_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    metadata = result.metadata
    assert metadata["gate_family_counts"]["h"] == 1
    assert metadata["gate_family_counts"]["cx"] == 1
    assert metadata["gate_family_counts"]["ccx"] == 1
    interactions = {tuple(pair) for pair in metadata["interaction_pairs"]}
    assert {(0, 1), (0, 2), (1, 2)} <= interactions


def test_pennylane_preserves_explicit_terminal_wire_order() -> None:
    """Explicit qml.probs order is retained instead of silently normalized."""
    pytest.importorskip("pennylane")
    template = """
import pennylane as qml


def answer():
    dev = qml.device("default.qubit", wires=2, shots=None)

    {shots_decorator}
    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        qml.Identity(wires=1)
        return {measurement}

    return circuit()
"""
    ascending = execute_pennylane_task(
        task_id="16",
        code=template.format(measurement="qml.probs(wires=[0, 1])", shots_decorator=""),
        entry_point="answer",
        inputs={},
    )
    implicit = execute_pennylane_task(
        task_id="16",
        code=template.format(measurement="qml.sample()", shots_decorator="@qml.set_shots(shots=10)"),
        entry_point="answer",
        inputs={},
    )

    assert ascending.probabilities == [0.0, 0.0, 1.0, 0.0]
    assert implicit.probabilities == [0.0, 1.0, 0.0, 0.0]
    assert ascending.metadata["measurement_wires"] == ["0", "1"]


def test_pennylane_no_tape_path_returns_empty_structural_fields() -> None:
    # Arrange
    pytest.importorskip("pennylane")
    code = "import numpy as np\ndef answer():\n    return np.array([0.25, 0.75])\n"

    # Act
    result = execute_pennylane_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.metadata["gate_family_counts"] == {}
    assert result.metadata["interaction_pairs"] == []


def test_pennylane_sample_and_error_paths() -> None:
    # Arrange
    sample_1d = "import numpy as np\ndef answer():\n    return np.array([-1, 1, 1])\n"
    sample_2d = "import numpy as np\ndef answer():\n    return np.array([[0, 1], [1, 0]])\n"
    bad = "import numpy as np\ndef answer():\n    return np.zeros((1, 1, 1))\n"

    # Act
    one = execute_pennylane_task(task_id="01", code=sample_1d, entry_point="answer", inputs={})
    two = execute_pennylane_task(task_id="01", code=sample_2d, entry_point="answer", inputs={})
    with pytest.raises(TypeError) as exc:
        execute_pennylane_task(task_id="01", code=bad, entry_point="answer", inputs={})

    # Assert
    assert one.probabilities == [1 / 3, 2 / 3]
    assert two.probabilities == [0.0, 0.5, 0.5, 0.0]
    assert "Expected 1-D or 2-D PennyLane array" in str(exc.value)
