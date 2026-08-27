from __future__ import annotations

import pytest

from qceval.models import ProviderRequest
from qceval.providers.smoke import SmokeProvider


@pytest.mark.parametrize(
    ("framework", "task_id", "entry_point", "prompt", "spec", "expected"),
    [
        (
            "cirq",
            "01",
            "answer",
            "",
            {
                "type": "deterministic_dominant",
                "expected_dominants": ["1"],
                "min_dominant_probability": 0.95,
                "min_non_measure_ops": 1,
            },
            "cirq.X",
        ),
        (
            "pennylane",
            "01",
            "answer",
            "",
            {
                "type": "deterministic_dominant",
                "expected_dominants": ["1"],
                "min_dominant_probability": 0.95,
                "min_non_measure_ops": 1,
            },
            "qml.PauliX",
        ),
        (
            "cirq",
            "13",
            "answer",
            "",
            {"type": "support_uniformity", "support": ["00", "11"], "threshold": 0.02},
            "'00': 0.5",
        ),
        (
            "cirq",
            "08",
            "qft_2",
            "",
            {"type": "support_uniformity", "support": "all", "threshold": 0.02},
            "'11': 0.25",
        ),
        (
            "pennylane",
            "08",
            "answer",
            "prepare 3 qubits",
            {"type": "support_uniformity", "support": "all", "threshold": 0.02},
            "0.125",
        ),
        (
            "cirq",
            "25",
            "answer",
            "",
            {"type": "peak_match", "accepted_peak_sets": [["001", "111"]], "top_k": 2},
            "'111': 0.5",
        ),
        (
            "pennylane",
            "02",
            "answer",
            "",
            {"type": "peak_match", "expected_peaks": ["011", "100"], "top_k": 2},
            "0.5",
        ),
        (
            "pennylane",
            "06",
            "answer",
            "",
            {"type": "exact_distribution", "expected_distribution": {"0": 1, "1": 0}},
            "1.0",
        ),
        (
            "cirq",
            "06",
            "answer",
            "",
            {"type": "exact_distribution", "threshold": 0.001},
            "return {",
        ),
        (
            "cirq",
            "04",
            "answer",
            "",
            {"type": "peak_match", "top_k": 8, "threshold": 0.08},
            "return {",
        ),
        (
            "cirq",
            "41",
            "answer",
            "",
            {"type": "structural", "structural_name": "vqe_z2_ansatz"},
            "cirq.rx",
        ),
        (
            "pennylane",
            "41",
            "answer",
            "",
            {"type": "structural", "structural_name": "vqe_z2_ansatz"},
            "qml.RX",
        ),
    ],
)
def test_smoke_provider_generates_spec_variants(framework, task_id, entry_point, prompt, spec, expected) -> None:
    # Arrange
    provider = SmokeProvider()
    request = ProviderRequest(
        task_id=task_id,
        framework=framework,
        prompt=prompt,
        entry_point=entry_point,
        metadata={"canonical_class": spec},
    )

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert expected in str(response.code)
    assert "RuntimeError" not in str(response.code)


@pytest.mark.parametrize(
    ("framework", "target", "expected"),
    [
        ("cirq", "cx", "cirq.CNOT"),
        ("cirq", "ccx", "cirq.CCX"),
        ("cirq", "controlled_h", "controlled_by"),
        ("cirq", "u_gate", "cirq.MatrixGate"),
        ("pennylane", "cx", "qml.CNOT"),
        ("pennylane", "ccx", "qml.Toffoli"),
        ("pennylane", "controlled_h", "qml.ControlledQubitUnitary"),
        ("pennylane", "u_gate", "qml.QubitUnitary"),
    ],
)
def test_smoke_provider_generates_unitary_variants(framework, target, expected) -> None:
    # Arrange
    provider = SmokeProvider()
    request = ProviderRequest(
        task_id="42",
        framework=framework,
        prompt="",
        entry_point="answer",
        metadata={"canonical_class": {"type": "exact_distribution", "comparison": "unitary", "target_unitary": target}},
    )

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert expected in str(response.code)


@pytest.mark.parametrize(
    ("framework", "spec"),
    [
        ("qiskit", {"type": "deterministic_dominant", "expected_dominants": ["0"]}),
        ("qiskit", {"type": "support_uniformity", "support": ["0", "1"]}),
        ("qiskit", {"type": "exact_distribution", "comparison": "unitary", "target_unitary": "cx"}),
        ("qiskit", {"type": "structural", "structural_name": "vqe_z2_ansatz"}),
        ("cirq", {"type": "unknown"}),
    ],
)
def test_smoke_provider_falls_back_when_spec_generation_is_unsupported(framework, spec) -> None:
    # Arrange
    provider = SmokeProvider()
    request = ProviderRequest(
        task_id="01",
        framework=framework,
        prompt="",
        entry_point="answer",
        metadata={"canonical_class": spec},
    )

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert "no canonical solution" in str(response.code)
