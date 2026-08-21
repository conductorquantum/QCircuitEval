from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.filter_pass1_queue import filter_queue
from scripts.generate_pinned_queue import generate_queue

from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    BASE_MODEL_COUNT,
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    OUTPUT_POLICY_BY_MODEL,
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
            "temperature_behavior": "explicit_zero",
        }
    return {"campaign_eligible": True, "models": models, "configurations": expand_configurations(models)}


def test_offline_runner_plan_validates_frozen_queue_and_candidates(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    selection = _selection()
    rows, assignments = generate_queue(selection)
    assert assignments == ASSIGNMENT_COUNT
    artifacts = []
    for config_id, route in selection["configurations"].items():
        candidate = candidates / f"{config_id}.jsonl"
        candidate.write_text(f"candidate-{config_id}\n", encoding="utf-8")
        artifacts.append(
            {
                "model_id": route["model_id"],
                "configuration_id": config_id,
                "path": str(candidate),
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "records": 280,
            }
        )
    queue = tmp_path / "pass1.tsv"
    queue.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    (candidates / "manifest.json").write_text(
        json.dumps(
            {
                "base_models": BASE_MODEL_COUNT,
                "configurations": CONFIGURATION_COUNT,
                "records": ASSIGNMENT_COUNT,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "grading"

    completed = subprocess.run(
        [
            "production/run_offline_aws.sh",
            "--queue",
            str(queue),
            "--candidates-dir",
            str(candidates),
            "--out-dir",
            str(output),
            "--plan-only",
        ],
        check=False,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "validated 36 offline shards and 9 candidate artifacts" in completed.stdout
    assert (output / "offline-queue.tsv").read_bytes() == queue.read_bytes()


def test_offline_runner_plan_accepts_parent_linked_eight_model_scope(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    selection = _selection()
    rows, assignments = generate_queue(selection)
    assert assignments == ASSIGNMENT_COUNT
    parent_queue = tmp_path / "pass1.tsv"
    parent_queue.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    queue = tmp_path / "pass1-no-glm.tsv"
    filter_queue(
        parent_queue,
        queue,
        tmp_path / "scope.json",
        excluded_models=["z-ai/glm-5.2"],
    )
    artifacts = []
    for config_id, route in selection["configurations"].items():
        if route["model_id"] == "z-ai/glm-5.2":
            continue
        candidate = candidates / f"{config_id}.jsonl"
        candidate.write_text(f"candidate-{config_id}\n", encoding="utf-8")
        artifacts.append(
            {
                "model_id": route["model_id"],
                "configuration_id": config_id,
                "path": str(candidate),
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "records": 280,
            }
        )
    (candidates / "manifest.json").write_text(
        json.dumps(
            {
                "base_models": 8,
                "configurations": 8,
                "records": 2240,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "grading"

    completed = subprocess.run(
        [
            "production/run_offline_aws.sh",
            "--queue",
            str(queue),
            "--candidates-dir",
            str(candidates),
            "--out-dir",
            str(output),
            "--scope-manifest",
            str(tmp_path / "scope.json"),
            "--plan-only",
        ],
        check=False,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "validated 32 offline shards and 8 candidate artifacts" in completed.stdout
    assert (output / "offline-queue.tsv").read_bytes() == queue.read_bytes()


def test_offline_runner_plan_accepts_manifest_bound_multi_configuration_campaign(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    selection = _selection()
    rows, assignments = generate_queue(selection)
    assert assignments == ASSIGNMENT_COUNT

    source_config = rows[0][15]
    duplicate_config = f"{source_config}-duplicate"
    duplicate_rows = []
    for row in rows:
        if row[15] != source_config:
            continue
        duplicate = list(row)
        duplicate[0] = duplicate[0].replace(source_config, duplicate_config)
        duplicate[15] = duplicate_config
        duplicate_rows.append(tuple(duplicate))
    rows.extend(duplicate_rows)

    queue = tmp_path / "pass1.tsv"
    queue.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    artifacts = []
    routes = dict(selection["configurations"])
    routes[duplicate_config] = routes[source_config]
    for config_id, route in routes.items():
        candidate = candidates / f"{config_id}.jsonl"
        candidate.write_text(f"candidate-{config_id}\n", encoding="utf-8")
        artifacts.append(
            {
                "model_id": route["model_id"],
                "configuration_id": config_id,
                "path": str(candidate),
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "records": 280,
            }
        )
    (candidates / "manifest.json").write_text(
        json.dumps(
            {
                "base_models": BASE_MODEL_COUNT,
                "configurations": CONFIGURATION_COUNT + 1,
                "records": ASSIGNMENT_COUNT + 280,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    campaign_manifest = tmp_path / "campaign.json"
    campaign_manifest.write_text(
        json.dumps(
            {
                "base_models": BASE_MODEL_COUNT,
                "configurations": CONFIGURATION_COUNT + 1,
                "shards": len(rows),
                "logical_requests": ASSIGNMENT_COUNT + 280,
                "artifacts": {"queue": {"sha256": hashlib.sha256(queue.read_bytes()).hexdigest()}},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "grading"

    completed = subprocess.run(
        [
            "production/run_offline_aws.sh",
            "--queue",
            str(queue),
            "--candidates-dir",
            str(candidates),
            "--out-dir",
            str(output),
            "--campaign-manifest",
            str(campaign_manifest),
            "--plan-only",
        ],
        check=False,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "validated 40 offline shards and 10 candidate artifacts" in completed.stdout
    assert (output / "offline-queue.tsv").read_bytes() == queue.read_bytes()


def test_offline_runner_rejects_a_staged_shard_when_validation_fails() -> None:
    script = (Path(__file__).parents[1] / "production/run_offline_aws.sh").read_text(encoding="utf-8")

    assert (
        """if ! validate_shard "$staged" "$model" "$configuration" "$framework" 70; then
    rm -f "$staged"
    return 1
  fi"""
        in script
    )


def test_offline_runner_uses_isolated_cudaq_capacity_and_rejects_grader_timeouts() -> None:
    script = (Path(__file__).parents[1] / "production/run_offline_aws.sh").read_text(encoding="utf-8")

    assert "CUDAQ_EVAL_TIMEOUT=900" in script
    assert "CUDAQ_EVALUATION_WORKERS=1" in script
    assert 'if [[ "$framework" == "cudaq" ]]; then' in script
    assert 'workers="$CUDAQ_EVALUATION_WORKERS"' in script
    assert 'timeout="$CUDAQ_EVAL_TIMEOUT"' in script
    assert '"offline shard contains a grader evaluation timeout"' in script


def test_offline_runner_records_each_calibration_failure_before_selecting_workers() -> None:
    script = (Path(__file__).parents[1] / "production/run_offline_aws.sh").read_text(encoding="utf-8")

    assert "calibration_status=failed" in script
    assert "calibration_status=passed" in script
    assert (
        'printf \'%s\\t%s\\t%s\\n\' "$workers" "$elapsed_ms" "$calibration_status" '
        '>>"$CALIBRATION_DIR/attempts.tsv"' in script
    )


def test_offline_runner_records_and_validates_aws_launch_provenance() -> None:
    script = (Path(__file__).parents[1] / "production/run_offline_aws.sh").read_text(encoding="utf-8")

    assert '"$OUT_DIR/aws-launch-readback.json"' in script
    assert ".ImageId == $image" in script
    assert ".InstanceType == $instance_type" in script
    assert ".SubnetId == $subnet" in script
    assert ".SecurityGroupIds == [$provisioning_group]" in script
    assert '"$OUT_DIR/evaluation-attachment-readback.json"' in script
    assert ".SecurityGroupIds == [$evaluation_group]" in script
