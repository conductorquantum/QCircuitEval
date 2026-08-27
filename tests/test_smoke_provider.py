from __future__ import annotations

import pytest

from qceval.models import ProviderRequest
from qceval.providers.base import fan_out_generate
from qceval.providers.smoke import SmokeProvider


def test_smoke_provider_returns_canonical_code() -> None:
    # Arrange
    provider = SmokeProvider()
    request = ProviderRequest(
        task_id="01",
        framework="qiskit",
        prompt="prompt text",
        entry_point="answer",
        metadata={"canonical_solution": "def answer():\n    return 1\n"},
    )

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert response.code == "def answer():\n    return 1\n"
    assert response.usage is not None
    assert response.usage.total_tokens == 6


def test_smoke_provider_records_matrix_metadata() -> None:
    provider = SmokeProvider(
        reasoning_effort="max",
        configuration_id="smoke-canonical__effort-max",
    )
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer")

    response = provider.generate(request)

    assert response.metadata["reasoning_effort"] == "max"
    assert response.metadata["configuration_id"] == "smoke-canonical__effort-max"


def test_smoke_provider_records_unnamed_reasoning() -> None:
    provider = SmokeProvider(reasoning_enabled=True)
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="prompt", entry_point="answer")

    response = provider.generate(request)

    assert response.metadata["reasoning_enabled"] is True


def test_smoke_provider_generate_many_preserves_order() -> None:
    # Arrange
    provider = SmokeProvider()
    requests = [
        ProviderRequest(task_id="02", framework="qiskit", prompt="two", entry_point="answer"),
        ProviderRequest(
            task_id="01",
            framework="qiskit",
            prompt="one",
            entry_point="answer",
            metadata={"canonical_solution": "def answer():\n    return 1\n"},
        ),
    ]

    # Act
    responses = provider.generate_many(requests)

    # Assert
    assert responses[0].raw_response == {"mode": "canonical", "task_id": "02"}
    assert responses[1].raw_response == {"mode": "canonical", "task_id": "01"}


def test_smoke_provider_rejects_invalid_support_spec() -> None:
    # Arrange
    provider = SmokeProvider()
    request = ProviderRequest(
        task_id="01",
        framework="cirq",
        prompt="",
        entry_point="answer",
        metadata={"canonical_class": {"type": "support_uniformity", "support": None}},
    )

    # Act / Assert
    with pytest.raises(ValueError, match="support_uniformity requires support bitstrings"):
        provider.generate(request)


def test_smoke_provider_reports_error_mode() -> None:
    # Arrange
    provider = SmokeProvider(mode="error")
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert response.error == "smoke provider configured to fail"


def test_smoke_provider_empty_mode_returns_failed_response() -> None:
    # Arrange
    provider = SmokeProvider(mode="empty")
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is False
    assert response.code == ""


def test_fan_out_generate_uses_serial_path_for_one_worker() -> None:
    # Arrange
    provider = SmokeProvider()
    request = ProviderRequest(task_id="01", framework="qiskit", prompt="", entry_point="answer")

    # Act
    responses = fan_out_generate(provider, [request], workers=1)

    # Assert
    assert len(responses) == 1
    assert responses[0].ok is True


def test_smoke_provider_uses_fallback_code_without_canonical() -> None:
    # Arrange
    provider = SmokeProvider()
    request = ProviderRequest(task_id="02", framework="cirq", prompt="", entry_point="answer")

    # Act
    response = provider.generate(request)

    # Assert
    assert response.ok is True
    assert "def answer" in str(response.code)
    assert "no canonical solution" in str(response.code)


def test_smoke_provider_rejects_unknown_mode() -> None:
    # Arrange
    mode = "weird"

    # Act
    try:
        SmokeProvider(mode=mode)
    except ValueError as exc:
        error = str(exc)
    else:
        error = ""

    # Assert
    assert error == "smoke mode must be canonical, empty, or error"
