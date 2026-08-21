from __future__ import annotations

from pathlib import Path

from qceval.providers.coda import CodaProvider
from qceval.providers.openrouter import OpenRouterProvider
from qceval.providers.registry import build_provider, provider_names
from qceval.providers.smoke import SmokeProvider
from qceval.serialization import to_jsonable


class ListLike:
    def tolist(self) -> list[int]:
        return [1, 2]


class ReprOnly:
    def __repr__(self) -> str:
        return "repr-only"


def test_registry_builds_smoke_provider() -> None:
    # Arrange
    config = {"smoke_mode": "empty"}

    # Act
    provider = build_provider("smoke", model="m", config=config)

    # Assert
    assert isinstance(provider, SmokeProvider)
    assert provider.mode == "empty"


def test_registry_builds_openrouter_provider() -> None:
    # Arrange
    config = {
        "openrouter_api_key": "key",
        "timeout": 3,
        "temperature": 0.1,
        "reasoning_enabled": True,
    }

    # Act
    provider = build_provider("openrouter", model="m", config=config)

    # Assert
    assert isinstance(provider, OpenRouterProvider)
    assert provider.timeout == 3
    assert provider.reasoning_enabled is True


def test_registry_builds_coda_provider() -> None:
    # Arrange
    config = {
        "coda_api_key": "key",
        "coda_mode": "learn",
        "coda_fast": True,
        "coda_prefer_structured_response": True,
        "timeout": 7,
        "retry_max_delay": 11,
    }

    # Act
    provider = build_provider("coda", model="coda/learn-fast", config=config)

    # Assert
    assert isinstance(provider, CodaProvider)
    assert provider.mode == "learn"
    assert provider.fast is True
    assert provider.prefer_structured_response is True
    assert provider.timeout == 7
    assert provider.retry_max_delay == 11


def test_registry_rejects_unknown_provider() -> None:
    # Arrange
    name = "missing"

    # Act
    try:
        build_provider(name, model=None)
    except ValueError as exc:
        error = str(exc)
    else:
        error = ""

    # Assert
    assert error == "unknown provider: missing"


def test_provider_names_are_stable() -> None:
    # Arrange
    expected = ("smoke", "openrouter", "coda")

    # Act
    names = provider_names()

    # Assert
    assert names == expected


def test_to_jsonable_handles_common_non_json_types() -> None:
    # Arrange
    value = {"path": Path("relative/a"), "complex": 1 + 2j, "array": ListLike(), "object": ReprOnly()}

    # Act
    payload = to_jsonable(value)

    # Assert
    assert payload == {
        "path": "relative/a",
        "complex": {"real": 1.0, "imag": 2.0},
        "array": [1, 2],
        "object": "repr-only",
    }
