from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from scripts.filter_pass1_queue import filter_queue
from scripts.generate_pinned_queue import generate_queue
from scripts.run_pass1_generation import read_queue

from qceval.production.campaign import EFFORTS_BY_MODEL, OUTPUT_POLICY_BY_MODEL, expand_configurations


def _selection() -> dict:
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
            "output_token_parameter": "max_tokens",
            "route_revision": f"route-{index}",
            "temperature_behavior": "explicit_zero",
        }
    return {"campaign_eligible": True, "models": models, "configurations": expand_configurations(models)}


def test_filter_queue_preserves_parent_provenance_and_removes_one_model(tmp_path: Path) -> None:
    rows, _assignments = generate_queue(_selection())
    parent = tmp_path / "pass1.tsv"
    parent.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    scoped = tmp_path / "pass1-no-glm.tsv"
    scope_manifest = tmp_path / "scope.json"

    manifest = filter_queue(
        parent,
        scoped,
        scope_manifest,
        excluded_models=["z-ai/glm-5.2"],
    )

    jobs = read_queue(scoped, validate_campaign=False)
    assert len(jobs) == 32
    assert sum(job.assigned_tasks for job in jobs) == 2240
    assert {job.model_id for job in jobs} == set(EFFORTS_BY_MODEL) - {"z-ai/glm-5.2"}
    assert manifest["parent_queue_sha256"] == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert manifest["queue_sha256"] == hashlib.sha256(scoped.read_bytes()).hexdigest()
    assert manifest["excluded_models"] == ["z-ai/glm-5.2"]
    assert scope_manifest.is_file()


def test_filter_queue_rejects_unknown_exclusion(tmp_path: Path) -> None:
    rows, _assignments = generate_queue(_selection())
    parent = tmp_path / "pass1.tsv"
    parent.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="not present"):
        filter_queue(
            parent,
            tmp_path / "scoped.tsv",
            tmp_path / "scope.json",
            excluded_models=["missing/model"],
        )
