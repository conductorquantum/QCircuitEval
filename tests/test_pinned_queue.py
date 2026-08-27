from __future__ import annotations

import pytest
from scripts.generate_pinned_queue import generate_queue

from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    OUTPUT_POLICY_BY_MODEL,
    SHARD_COUNT,
    expand_configurations,
)


def _selection(*, eligible: bool = True) -> dict:
    models = {}
    for index, (model_id, efforts) in enumerate(EFFORTS_BY_MODEL.items()):
        output_tokens, output_source, cap_status = OUTPUT_POLICY_BY_MODEL[model_id]
        models[model_id] = {
            "model_id": model_id,
            "reasoning_setting": efforts[-1],
            "reasoning_efforts": list(efforts),
            "endpoint_tag": "xai" if model_id == "x-ai/grok-4.6" else f"endpoint-{index}",
            "configured_output_tokens": output_tokens,
            "output_limit_source": output_source,
            "endpoint_cap_status": cap_status,
            "output_token_parameter": "max_tokens" if model_id == "z-ai/glm-5.2" else "max_completion_tokens",
            "route_revision": f"route-{index}",
            "temperature_behavior": "explicit_zero" if index % 2 else "not_exposed",
        }
    return {
        "campaign_eligible": eligible,
        "models": models,
        "configurations": expand_configurations(models),
    }


def test_pinned_queue_expands_to_exact_campaign_cardinality() -> None:
    rows, assignments = generate_queue(_selection())

    assert len(rows) == SHARD_COUNT
    assert assignments == ASSIGNMENT_COUNT
    assert {row[4] for row in rows} == {"qiskit", "cirq", "pennylane", "cudaq"}
    assert {row[5] for row in rows} == {"all"}
    assert {row[9] for row in rows} == {"author_native", "benchmark_floor"}
    assert {row[10] for row in rows} == {"catalog_numeric", "undisclosed_first_party_exception"}
    assert {row[14] for row in rows} == {"70"}
    assert len({row[0] for row in rows}) == SHARD_COUNT
    assert len({row[15] for row in rows}) == CONFIGURATION_COUNT


def test_pinned_queue_refuses_campaign_blocked_selection() -> None:
    with pytest.raises(ValueError, match="campaign-blocked"):
        generate_queue(_selection(eligible=False))


def test_pinned_queue_rejects_malformed_configuration_identity() -> None:
    selection = _selection()
    config_id, route = next(iter(selection["configurations"].items()))
    route["configuration_id"] = "wrong__effort-max"

    with pytest.raises(ValueError, match="malformed configuration_id"):
        generate_queue(selection)
