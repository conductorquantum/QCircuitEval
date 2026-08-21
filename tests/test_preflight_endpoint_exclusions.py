from __future__ import annotations

import pytest
from scripts.preflight_openrouter_endpoints import _endpoint_exclusions


def test_endpoint_exclusions_group_repeated_objective_failures() -> None:
    models = [{"model_id": "author/model"}, {"model_id": "author/other"}]

    exclusions = _endpoint_exclusions(
        ["author/model=provider/fp8", "author/model=provider/bf16"],
        models,
    )

    assert exclusions == {"author/model": {"provider/fp8", "provider/bf16"}}


@pytest.mark.parametrize("value", ["missing-separator", "author/model=", "=provider"])
def test_endpoint_exclusions_reject_malformed_values(value: str) -> None:
    with pytest.raises(ValueError, match="MODEL_ID=ENDPOINT_TAG"):
        _endpoint_exclusions([value], [{"model_id": "author/model"}])


def test_endpoint_exclusions_reject_unknown_models() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        _endpoint_exclusions(["unknown/model=provider"], [{"model_id": "author/model"}])
