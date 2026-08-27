from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.create_effort_sweep_manifest import _validate_historical_imports, validate_run_manifest
from scripts.generate_pinned_queue import generate_queue

from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    BASE_MODEL_COUNT,
    BENCHMARK_CONTENT_COMMIT,
    CAMPAIGN_SCHEMA_VERSION,
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    FRESH_ASSIGNMENT_COUNT,
    HISTORICAL_IMPORT_ASSIGNMENT_COUNT,
    OUTPUT_POLICY_BY_MODEL,
    SHARD_COUNT,
    expand_configurations,
)


def _queue(path: Path) -> None:
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
            "temperature_behavior": "not_exposed",
        }
    selection = {
        "campaign_eligible": True,
        "models": models,
        "configurations": expand_configurations(models),
    }
    rows, assignments = generate_queue(selection)
    assert assignments == ASSIGNMENT_COUNT
    path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def _manifest(queue: Path) -> dict[str, object]:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "harness_commit": "harness",
        "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
        "base_models": BASE_MODEL_COUNT,
        "configurations": CONFIGURATION_COUNT,
        "shards": SHARD_COUNT,
        "logical_requests": ASSIGNMENT_COUNT,
        "fresh_logical_requests": FRESH_ASSIGNMENT_COUNT,
        "historical_imported_requests": HISTORICAL_IMPORT_ASSIGNMENT_COUNT,
        "historical_imports": {
            "configuration_ids": [],
            "configurations": 0,
            "records": 0,
        },
        "artifacts": {"queue": {"sha256": hashlib.sha256(queue.read_bytes()).hexdigest()}},
    }


def test_current_campaign_manifest_requires_every_assignment_to_be_fresh(tmp_path: Path) -> None:
    queue = tmp_path / "pass1.tsv"
    _queue(queue)
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(json.dumps(_manifest(queue)), encoding="utf-8")

    payload = validate_run_manifest(manifest, queue=queue, harness_commit="harness")

    assert payload["fresh_logical_requests"] == ASSIGNMENT_COUNT
    assert payload["historical_imported_requests"] == 0


def test_current_campaign_manifest_rejects_historical_candidate_reuse(tmp_path: Path) -> None:
    queue = tmp_path / "pass1.tsv"
    _queue(queue)
    payload = _manifest(queue)
    payload["historical_imports"] = {
        "configuration_ids": ["openai-gpt-5-6-sol__effort-max"],
        "configurations": 1,
        "records": 280,
    }
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fresh generation"):
        validate_run_manifest(manifest, queue=queue, harness_commit="harness")


def test_current_campaign_rejects_even_a_supplied_empty_import_manifest(tmp_path: Path) -> None:
    import_manifest = tmp_path / "imports.json"
    import_manifest.write_text("{}", encoding="utf-8")

    assert _validate_historical_imports(None)["records"] == 0
    with pytest.raises(ValueError, match="requires fresh generation"):
        _validate_historical_imports(import_manifest)


def test_registry_and_campaign_pin_the_same_content_commit() -> None:
    registry = json.loads(Path("production/models.full-cap.json").read_text(encoding="utf-8"))

    assert registry["policy"]["benchmark_content_commit"] == BENCHMARK_CONTENT_COMMIT
