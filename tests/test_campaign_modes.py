"""Campaign-mode regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qceval.production.campaign import (
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    OUTPUT_POLICY_BY_MODEL,
    configuration_id,
    expand_configurations,
    validate_registry_efforts,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_effort_sweep_has_exact_requested_roster() -> None:
    script = """
import json
from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    BASE_MODEL_COUNT,
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    FRESH_ASSIGNMENT_COUNT,
    REUSABLE_CONFIGURATION_IDS,
)
print(json.dumps({
    'models': BASE_MODEL_COUNT,
    'configurations': CONFIGURATION_COUNT,
    'assignments': ASSIGNMENT_COUNT,
    'fresh_assignments': FRESH_ASSIGNMENT_COUNT,
    'efforts': EFFORTS_BY_MODEL,
    'reusable': REUSABLE_CONFIGURATION_IDS,
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["QCEVAL_PRODUCTION_CAMPAIGN"] = "prompt-effort-sweep"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload == {
        "assignments": 7840,
        "configurations": 28,
        "efforts": {
            "anthropic/claude-fable-5": ["low", "medium", "high", "xhigh", "max"],
            "anthropic/claude-opus-5": ["low", "medium", "high", "xhigh", "max"],
            "openai/gpt-5.6-luna": ["none", "low", "medium", "high", "xhigh", "max"],
            "openai/gpt-5.6-sol": ["none", "low", "medium", "high", "xhigh", "max"],
            "openai/gpt-5.6-terra": ["none", "low", "medium", "high", "xhigh", "max"],
        },
        "fresh_assignments": 6720,
        "models": 5,
        "reusable": [
            "anthropic-claude-fable-5__effort-max",
            "anthropic-claude-opus-5__effort-max",
            "openai-gpt-5-6-sol__effort-low",
            "openai-gpt-5-6-sol__effort-max",
        ],
    }


def _route_map() -> dict[str, dict[str, object]]:
    routes: dict[str, dict[str, object]] = {}
    for index, (model_id, efforts) in enumerate(EFFORTS_BY_MODEL.items()):
        output_tokens, output_source, cap_status = OUTPUT_POLICY_BY_MODEL[model_id]
        routes[model_id] = {
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
    return routes


def test_default_campaign_expands_to_frozen_max_reasoning_cardinality() -> None:
    routes = _route_map()
    validate_registry_efforts(
        [{"model_id": model_id, "reasoning_efforts": list(efforts)} for model_id, efforts in EFFORTS_BY_MODEL.items()]
    )
    configurations = expand_configurations(routes)

    assert len(configurations) == CONFIGURATION_COUNT
    assert set(configurations) == {
        configuration_id(model_id, effort) for model_id, efforts in EFFORTS_BY_MODEL.items() for effort in efforts
    }


def test_registry_effort_drift_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match the frozen"):
        validate_registry_efforts([{"model_id": "openai/gpt-5.6-sol", "reasoning_efforts": ["max"]}])
