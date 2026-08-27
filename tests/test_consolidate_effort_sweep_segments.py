from __future__ import annotations

from pathlib import Path

import pytest
from scripts.consolidate_effort_sweep_segments import _fresh_jobs, _validate_source_plan
from scripts.generate_pinned_queue import generate_queue
from scripts.run_pass1_generation import read_queue

from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    OUTPUT_POLICY_BY_MODEL,
    SHARD_COUNT,
    expand_configurations,
)


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
            "output_token_parameter": "max_tokens" if model_id == "z-ai/glm-5.2" else "max_completion_tokens",
            "route_revision": f"route-{index}",
            "temperature_behavior": "not_exposed",
        }
    return {"campaign_eligible": True, "models": models, "configurations": expand_configurations(models)}


def test_fresh_jobs_include_every_configuration_for_new_content(tmp_path: Path) -> None:
    queue = tmp_path / "queue.tsv"
    rows, _assignments = generate_queue(_selection())
    queue.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    fresh = _fresh_jobs(read_queue(queue), frozenset())

    assert len(fresh) == SHARD_COUNT
    assert len({job.configuration_id for job in fresh}) == CONFIGURATION_COUNT
    assert sum(job.assigned_tasks for job in fresh) == ASSIGNMENT_COUNT


def test_source_plan_must_select_requested_model(tmp_path: Path) -> None:
    segments = tmp_path / "generation" / "segments"
    segments.mkdir(parents=True)
    (segments.parent / "generation-plan.json").write_text(
        '{"selected_model_lanes":["openai/gpt-5.6-terra"]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not select"):
        _validate_source_plan("openai/gpt-5.6-luna", segments)
