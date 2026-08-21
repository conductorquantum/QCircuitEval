from __future__ import annotations

from pathlib import Path

from scripts.generate_pinned_queue import generate_queue
from scripts.run_configuration_smokes import diagnostic_jobs, offline_regrade_command
from scripts.run_pass1_generation import read_queue

from qceval.production.campaign import (
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    OUTPUT_POLICY_BY_MODEL,
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
            "temperature_behavior": "explicit_zero",
        }
    rows, _ = generate_queue(
        {"campaign_eligible": True, "models": models, "configurations": expand_configurations(models)}
    )
    path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def test_diagnostics_choose_one_task_per_configuration_and_regrade_offline(tmp_path: Path) -> None:
    queue = tmp_path / "queue.tsv"
    _queue(queue)

    jobs = diagnostic_jobs(read_queue(queue))
    command = offline_regrade_command(jobs[0], tmp_path / "input.jsonl", tmp_path / "out.jsonl")

    assert len(jobs) == CONFIGURATION_COUNT
    assert {job.framework for job in jobs} == {"qiskit"}
    assert "--regrade" in command
    assert "--rerun" not in command
    assert "--openrouter-api-key" not in command
    assert "--configuration-id" not in command
