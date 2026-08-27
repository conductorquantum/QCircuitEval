from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from scripts.audit_max_reasoning_campaign import audit_campaign
from scripts.generate_pinned_queue import generate_queue
from scripts.run_pass1_generation import _assignments, read_queue

from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    BASE_MODEL_COUNT,
    BENCHMARK_CONTENT_COMMIT,
    CAMPAIGN_SCHEMA_VERSION,
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    OUTPUT_POLICY_BY_MODEL,
    SHARD_COUNT,
    expand_configurations,
)


def test_final_audit_accepts_one_complete_fresh_campaign(tmp_path: Path) -> None:
    paths = _complete_campaign(tmp_path)

    report = audit_campaign(**paths)

    assert report["publication_ready"] is True
    assert report["concerns"] == []
    assert report["coverage"]["fresh_generation_records"] == ASSIGNMENT_COUNT
    assert report["coverage"]["offline_regraded_records"] == ASSIGNMENT_COUNT
    assert report["coverage"]["provider_cost_covered_records"] == ASSIGNMENT_COUNT
    assert report["checks"]["offline_grading"]["all_instances_terminated"] is True


def test_final_audit_fails_closed_when_termination_readback_is_incomplete(tmp_path: Path) -> None:
    paths = _complete_campaign(tmp_path)
    termination = paths["offline_dir"] / "termination-readback.json"
    rows = json.loads(termination.read_text(encoding="utf-8"))
    termination.write_text(json.dumps(rows[:-1]), encoding="utf-8")

    report = audit_campaign(**paths)

    assert report["publication_ready"] is False
    assert any("AWS launch and termination readbacks" in concern for concern in report["concerns"])


def test_final_audit_accepts_parent_linked_model_subset(tmp_path: Path) -> None:
    paths = _complete_campaign(tmp_path)
    paths = _exclude_model(paths, "z-ai/glm-5.2")

    report = audit_campaign(**paths)

    assert report["publication_ready"] is True
    assert report["concerns"] == []
    assert report["scope"] == {
        "base_models": 8,
        "configurations": 8,
        "framework_shards": 32,
        "logical_records": 2240,
        "fresh_records": 2240,
        "historical_records": 0,
    }
    assert report["coverage"]["fresh_generation_records"] == 2240
    assert report["coverage"]["offline_regraded_records"] == 2240


def test_final_audit_rejects_scope_manifest_count_mismatch(tmp_path: Path) -> None:
    paths = _exclude_model(_complete_campaign(tmp_path), "z-ai/glm-5.2")
    scope_manifest = paths["scope_manifest"]
    payload = json.loads(scope_manifest.read_text(encoding="utf-8"))
    payload["logical_requests"] += 1
    scope_manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_campaign(**paths)

    assert report["publication_ready"] is False
    assert any("scope manifest counts" in concern for concern in report["concerns"])


def test_final_audit_accepts_exact_task_scoped_route_recovery(tmp_path: Path) -> None:
    paths = _complete_campaign(tmp_path)
    job = next(
        job
        for job in read_queue(paths["queue"])
        if job.model_id == "google/gemma-4-31b-it" and job.framework == "cudaq"
    )
    key = next(iter(_assignments(job)))
    paths = _apply_route_recovery(paths, job, [key])

    report = audit_campaign(**paths)

    assert report["publication_ready"] is True
    assert report["checks"]["route_recovery_manifest"]["recovered_assignments"] == 1


def test_final_audit_rejects_replacement_route_on_undeclared_task(tmp_path: Path) -> None:
    paths = _complete_campaign(tmp_path)
    job = next(
        job
        for job in read_queue(paths["queue"])
        if job.model_id == "google/gemma-4-31b-it" and job.framework == "cudaq"
    )
    first, second = list(_assignments(job))[:2]
    paths = _apply_route_recovery(paths, job, [first])
    selection = json.loads((tmp_path / "route-selection.json").read_text(encoding="utf-8"))
    selected = selection["configurations"][job.configuration_id]
    replacement_route = _replacement_route(selected)
    segment = next((paths["generation_dir"] / "segments" / job.job_id).glob("*.jsonl"))
    _replace_record_route(segment, second, replacement_route)

    report = audit_campaign(**paths)

    assert report["publication_ready"] is False
    assert any("record route differs from the frozen queue" in concern for concern in report["concerns"])


def _complete_campaign(tmp_path: Path) -> dict[str, Path]:  # noqa: C901 - explicit end-to-end audit fixture
    queue = tmp_path / "pass1.tsv"
    selection = _selection()
    rows, assignments = generate_queue(selection)
    assert assignments == ASSIGNMENT_COUNT
    queue.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    jobs = read_queue(queue)
    generation_dir = tmp_path / "generation"
    candidates_dir = tmp_path / "candidates"
    offline_dir = tmp_path / "offline"
    for directory in (generation_dir, candidates_dir, offline_dir):
        directory.mkdir()

    generation_by_configuration: dict[str, list[dict]] = defaultdict(list)
    all_attempts: list[dict] = []
    for job in jobs:
        assert job.configuration_id is not None
        records = [_record(job, key, status="generated") for key in _assignments(job)]
        generation_by_configuration[job.configuration_id].extend(records)
        segment = generation_dir / "segments" / job.job_id / "route-s001-all.jsonl"
        segment.parent.mkdir(parents=True)
        _write_run(segment, records, _summary(job.configuration_id, records))
        all_attempts.extend(
            {
                "schema_version": "qceval.provider_attempt_ledger.v1",
                "event_id": hashlib.sha256(f"{job.job_id}:{index}".encode()).hexdigest(),
                "configuration_id": job.configuration_id,
                "provider_cost_usd": 1.0,
            }
            for index, _record_payload in enumerate(records)
        )
    (generation_dir / "provider-attempt-ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_attempts),
        encoding="utf-8",
    )
    (generation_dir / "controller-summary.json").write_text(
        json.dumps(
            {
                "status": "generation_complete",
                "expected_logical_requests": ASSIGNMENT_COUNT,
                "accepted_logical_requests": ASSIGNMENT_COUNT,
                "deferred_infrastructure_requests": 0,
                "open_endpoint_circuits": 0,
                "accepted_scan_errors": {},
                "model_lane_status": dict.fromkeys(EFFORTS_BY_MODEL, "complete"),
                "provider_cost_usd": float(ASSIGNMENT_COUNT),
                "physical_provider_attempts": ASSIGNMENT_COUNT,
                "infrastructure_records": 0,
            }
        ),
        encoding="utf-8",
    )

    candidate_artifacts = []
    model_by_configuration = {job.configuration_id: job.model_id for job in jobs}
    for config_id, records in generation_by_configuration.items():
        candidate = candidates_dir / f"{config_id}__pass1.generated.jsonl"
        _write_run(candidate, records, _summary(config_id, records))
        candidate_artifacts.append(
            {
                "model_id": model_by_configuration[config_id],
                "configuration_id": config_id,
                "path": str(candidate),
                "sha256": _sha256(candidate),
                "records": 280,
                "route_verified_records": 280,
                "cost_covered_records": 280,
            }
        )
    candidate_manifest = {
        "base_models": BASE_MODEL_COUNT,
        "configurations": CONFIGURATION_COUNT,
        "records": ASSIGNMENT_COUNT,
        "imported_configurations": 0,
        "imported_records": 0,
        "imports_manifest": None,
        "queue_sha256": _sha256(queue),
        "artifacts": candidate_artifacts,
    }
    (candidates_dir / "manifest.json").write_text(
        json.dumps(candidate_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    shutil.copyfile(queue, offline_dir / "offline-queue.tsv")
    shutil.copyfile(candidates_dir / "manifest.json", offline_dir / "candidate-manifest.json")
    offline_by_configuration: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        assert job.configuration_id is not None
        records = [_record(job, key, status="passed") for key in _assignments(job)]
        offline_by_configuration[job.configuration_id].extend(records)
        _write_run(offline_dir / "shards" / f"{job.job_id}.jsonl", records, _summary(job.configuration_id, records))
        done = offline_dir / "state" / "done" / f"{job.job_id}.job"
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("done\n", encoding="utf-8")
    for state in ("pending", "running", "failed"):
        (offline_dir / "state" / state).mkdir(parents=True)

    score_rows = []
    for config_id, records in offline_by_configuration.items():
        merged = offline_dir / "merged" / f"{config_id}__pass1.regraded.jsonl"
        summary = _summary(config_id, records)
        _write_run(merged, records, summary)
        model_id = model_by_configuration[config_id]
        score_rows.append(
            {
                "model": model_id,
                "protocol": "pass1",
                "score": 1.0,
                "logical_tasks": 280,
                "provider_records": 280,
                "records_with_reported_cost": 280,
                "records_missing_reported_cost": 0,
                "reported_cost_coverage": 1.0,
                "observed_cost_usd": 280.0,
            }
        )
    (offline_dir / "score-cost.json").write_text(json.dumps(score_rows), encoding="utf-8")
    calibration = offline_dir / "calibration"
    calibration.mkdir()
    (calibration / "attempts.tsv").write_text("2\t30\tpassed\n4\t20\tpassed\n8\t25\tpassed\n", encoding="utf-8")

    instance_ids = [f"i-{index:017d}" for index in range(6)]
    (offline_dir / "instance-ids.txt").write_text("\n".join(instance_ids) + "\n", encoding="utf-8")
    launch = [
        {
            "InstanceId": instance_id,
            "ImageId": "ami-test",
            "InstanceType": "c7i.2xlarge",
            "SubnetId": "subnet-test",
            "SecurityGroupIds": ["sg-provisioning"],
            "State": "running",
        }
        for instance_id in instance_ids
    ]
    (offline_dir / "aws-launch-readback.json").write_text(json.dumps(launch), encoding="utf-8")
    evaluation_attachment = [
        {"InstanceId": instance_id, "SecurityGroupIds": ["sg-evaluation"], "State": "running"}
        for instance_id in instance_ids
    ]
    (offline_dir / "evaluation-attachment-readback.json").write_text(
        json.dumps(evaluation_attachment), encoding="utf-8"
    )
    termination = [{"InstanceId": instance_id, "State": "terminated"} for instance_id in instance_ids]
    (offline_dir / "termination-readback.json").write_text(json.dumps(termination), encoding="utf-8")
    (offline_dir / "evaluation-security-group.json").write_text(
        json.dumps({"SecurityGroups": [{"GroupId": "sg-evaluation", "IpPermissionsEgress": []}]}),
        encoding="utf-8",
    )

    artifact_paths = {name: tmp_path / f"{name}.json" for name in _manifest_artifact_names() - {"queue"}}
    for name, path in artifact_paths.items():
        payload = {"name": name}
        if name == "benchmark_content_manifest":
            payload.update(
                {
                    "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
                    "byte_identical": True,
                }
            )
        path.write_text(json.dumps(payload), encoding="utf-8")
    artifact_paths["queue"] = queue
    harness_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    run_manifest = tmp_path / "run-manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "harness_commit": harness_commit,
                "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
                "base_models": BASE_MODEL_COUNT,
                "configurations": CONFIGURATION_COUNT,
                "shards": SHARD_COUNT,
                "logical_requests": ASSIGNMENT_COUNT,
                "fresh_logical_requests": ASSIGNMENT_COUNT,
                "historical_imported_requests": 0,
                "historical_imports": {"configuration_ids": [], "configurations": 0, "records": 0},
                "artifacts": {
                    name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
                    for name, path in artifact_paths.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "run_manifest": run_manifest,
        "queue": queue,
        "generation_dir": generation_dir,
        "candidates_dir": candidates_dir,
        "offline_dir": offline_dir,
    }


def _exclude_model(paths: dict[str, Path], model_id: str) -> dict[str, Path]:
    parent_queue = paths["queue"]
    parent_jobs = read_queue(parent_queue)
    excluded_configurations = {job.configuration_id for job in parent_jobs if job.model_id == model_id}
    assert None not in excluded_configurations
    scoped_queue = parent_queue.with_name("pass1-scoped.tsv")
    parent_lines = parent_queue.read_text(encoding="utf-8").splitlines()
    scoped_lines = [line for line, job in zip(parent_lines, parent_jobs, strict=True) if job.model_id != model_id]
    scoped_queue.write_text("".join(f"{line}\n" for line in scoped_lines), encoding="utf-8")
    scoped_jobs = read_queue(scoped_queue, validate_campaign=False)
    scoped_models = sorted({job.model_id for job in scoped_jobs})
    scope_manifest = parent_queue.with_name("scope.json")
    scope_manifest.write_text(
        json.dumps(
            {
                "schema_version": "qceval.pass1_scope.v1",
                "parent_queue": str(parent_queue.resolve()),
                "parent_queue_sha256": _sha256(parent_queue),
                "parent_shards": len(parent_jobs),
                "parent_logical_requests": sum(job.assigned_tasks for job in parent_jobs),
                "excluded_models": [model_id],
                "queue": str(scoped_queue.resolve()),
                "queue_sha256": _sha256(scoped_queue),
                "base_models": len(scoped_models),
                "models": scoped_models,
                "configurations": len({job.configuration_id for job in scoped_jobs}),
                "shards": len(scoped_jobs),
                "logical_requests": sum(job.assigned_tasks for job in scoped_jobs),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    candidates_dir = paths["candidates_dir"]
    candidate_manifest_path = candidates_dir / "manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_manifest["artifacts"] = [
        row for row in candidate_manifest["artifacts"] if row["configuration_id"] not in excluded_configurations
    ]
    candidate_manifest.update(
        {
            "base_models": len(scoped_models),
            "configurations": len(scoped_models),
            "records": len(scoped_models) * 280,
            "queue_sha256": _sha256(scoped_queue),
        }
    )
    candidate_manifest_path.write_text(json.dumps(candidate_manifest, sort_keys=True) + "\n", encoding="utf-8")

    offline_dir = paths["offline_dir"]
    shutil.copyfile(scoped_queue, offline_dir / "offline-queue.tsv")
    shutil.copyfile(candidate_manifest_path, offline_dir / "candidate-manifest.json")
    excluded_jobs = [job for job in parent_jobs if job.model_id == model_id]
    for job in excluded_jobs:
        (offline_dir / "shards" / f"{job.job_id}.jsonl").unlink()
        (offline_dir / "state" / "done" / f"{job.job_id}.job").unlink()
    for config_id in excluded_configurations:
        assert config_id is not None
        (offline_dir / "merged" / f"{config_id}__pass1.regraded.jsonl").unlink()
    score_cost_path = offline_dir / "score-cost.json"
    score_cost = json.loads(score_cost_path.read_text(encoding="utf-8"))
    score_cost_path.write_text(json.dumps([row for row in score_cost if row["model"] != model_id]), encoding="utf-8")
    return {
        **paths,
        "queue": scoped_queue,
        "scope_manifest": scope_manifest,
    }


def _apply_route_recovery(paths: dict[str, Path], job, keys: list[tuple]) -> dict[str, Path]:
    assert job.configuration_id is not None
    config_identity = hashlib.sha256(f"{job.configuration_id}:replacement".encode()).hexdigest()
    selected = {
        "configuration_id": job.configuration_id,
        "configuration_identity_sha256": config_identity,
        "model_id": job.model_id,
        "endpoint_tag": "replacement/fp8",
        "configured_output_tokens": job.configured_output_tokens,
        "output_limit_source": job.output_limit_source,
        "endpoint_cap_status": job.endpoint_cap_status,
        "output_token_parameter": job.output_token_parameter,
        "route_revision": "route-replacement",
        "temperature": 0.0,
        "temperature_behavior": "explicit_zero",
        "provider": "Replacement",
        "endpoint_served_model_id": f"{job.model_id}-served",
    }
    selection = paths["queue"].parent / "route-selection.json"
    selection.write_text(
        json.dumps({"configurations": {job.configuration_id: selected}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replacement_route = _replacement_route(selected)
    canary = paths["queue"].parent / "route-canary.jsonl"
    canary.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                {
                    "kind": "canary",
                    "status": "passed",
                    "configuration_id": job.configuration_id,
                    "model_id": job.model_id,
                    "provider_response_metadata": {"route": replacement_route},
                },
                {
                    "kind": "summary",
                    "benchmark_denominator_member": False,
                    "models": 1,
                    "passed": 1,
                    "failed": [],
                },
            )
        ),
        encoding="utf-8",
    )

    segment = next((paths["generation_dir"] / "segments" / job.job_id).glob("*.jsonl"))
    candidate_manifest_path = paths["candidates_dir"] / "manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_row = next(
        row for row in candidate_manifest["artifacts"] if row["configuration_id"] == job.configuration_id
    )
    candidate = Path(candidate_row["path"])
    offline_shard = paths["offline_dir"] / "shards" / f"{job.job_id}.jsonl"
    offline_merged = paths["offline_dir"] / "merged" / f"{job.configuration_id}__pass1.regraded.jsonl"
    for artifact in (segment, candidate, offline_shard, offline_merged):
        for key in keys:
            _replace_record_route(artifact, key, replacement_route)
    candidate_row["sha256"] = _sha256(candidate)
    candidate_manifest_path.write_text(json.dumps(candidate_manifest, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(candidate_manifest_path, paths["offline_dir"] / "candidate-manifest.json")

    route_recovery_manifest = paths["queue"].parent / "route-recovery-manifest.json"
    route_recovery_manifest.write_text(
        json.dumps(
            {
                "schema_version": "qceval.route_recovery.v1",
                "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
                "queue": paths["queue"].name,
                "queue_sha256": _sha256(paths["queue"]),
                "revisions": [
                    {
                        "configuration_id": job.configuration_id,
                        "model_id": job.model_id,
                        "selection": {"path": selection.name, "sha256": _sha256(selection)},
                        "canary": {"path": canary.name, "sha256": _sha256(canary)},
                        "assignments": [
                            {
                                "suite": key[0],
                                "framework": key[1],
                                "task_id": key[2],
                                "sample_index": key[3],
                                "attempt_index": key[4],
                            }
                            for key in keys
                        ],
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**paths, "route_recovery_manifest": route_recovery_manifest}


def _replacement_route(selection: dict) -> dict:
    return {
        "configuration_id": selection["configuration_id"],
        "configuration_identity_sha256": selection["configuration_identity_sha256"],
        "endpoint_tag": selection["endpoint_tag"],
        "max_output_tokens": selection["configured_output_tokens"],
        "output_limit_source": selection["output_limit_source"],
        "endpoint_cap_status": selection["endpoint_cap_status"],
        "output_token_parameter": selection["output_token_parameter"],
        "route_revision": selection["route_revision"],
        "temperature": selection["temperature"],
        "route_verified": True,
        "allow_fallbacks": False,
        "require_parameters": True,
        "selected_provider": selection["provider"],
        "selected_model": selection["endpoint_served_model_id"],
    }


def _replace_record_route(path: Path, key: tuple, replacement_route: dict) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    matches = 0
    for row in rows:
        if row.get("kind") != "result":
            continue
        row_key = (
            row.get("suite"),
            row.get("framework"),
            row.get("task_id"),
            row.get("sample_index"),
            row.get("attempt_index"),
        )
        if row_key == key:
            row["provider_response"]["metadata"]["route"] = replacement_route
            row["provider_response"]["metadata"]["attempt_history"] = [
                {
                    "status": "accepted_model_outcome",
                    "generation_id": "gen-test-replacement",
                    "route_verified": True,
                    "usage": row["provider_response"]["usage"],
                }
            ]
            matches += 1
    assert matches == 1
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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


def _record(job, key, *, status: str) -> dict:
    suite, framework, task_id, sample_index, attempt_index = key
    temperature = 0.0 if job.temperature_behavior == "explicit_zero" else None
    return {
        "kind": "result",
        "model": job.model_id,
        "suite": suite,
        "framework": framework,
        "task_id": task_id,
        "sample_index": sample_index,
        "attempt_index": attempt_index,
        "status": status,
        "provider_response": {
            "usage": {"cost_usd": 1.0, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "metadata": {
                "route": {
                    "configuration_id": job.configuration_id,
                    "endpoint_tag": job.endpoint_tag,
                    "max_output_tokens": job.configured_output_tokens,
                    "output_limit_source": job.output_limit_source,
                    "endpoint_cap_status": job.endpoint_cap_status,
                    "output_token_parameter": job.output_token_parameter,
                    "route_revision": job.route_revision,
                    "temperature": temperature,
                    "route_verified": True,
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            },
        },
    }


def _summary(config_id: str, records: list[dict]) -> dict:
    frameworks = {framework: {"assigned_tasks": 70, "infrastructure_failures": 0} for framework in _frameworks()}
    return {
        "kind": "summary",
        "configuration_id": config_id,
        "summary": {
            "total_tasks": len(records),
            "pass_rate": 1.0,
            "by_suite": {
                "core": {"assigned_tasks": 232, "infrastructure_failures": 0},
                "qec": {"assigned_tasks": 48, "infrastructure_failures": 0},
            },
            "by_framework": frameworks,
        },
    }


def _write_run(path: Path, records: list[dict], summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in [*records, summary]),
        encoding="utf-8",
    )


def _manifest_artifact_names() -> set[str]:
    return {
        "capability_registry",
        "raw_openrouter_catalog",
        "endpoint_selection",
        "preflight_hashes",
        "queue",
        "benchmark_content_manifest",
        "canaries",
        "diagnostics_manifest",
    }


def _frameworks() -> tuple[str, ...]:
    return "qiskit", "cirq", "pennylane", "cudaq"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
