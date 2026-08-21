from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.generate_pinned_queue import generate_queue
from scripts.run_pass1_generation import (
    CampaignLedger,
    QueueJob,
    SegmentScope,
    _interleaved_model_jobs,
    _run_job,
    _run_model_lane,
    _segment_scopes,
    build_command,
    campaign_summary,
    main,
    read_queue,
)

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
from qceval.production.deferred import DeferredInfrastructureStore


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
            "temperature_behavior": "not_exposed" if index == 0 else "explicit_zero",
        }
    return {"campaign_eligible": True, "models": models, "configurations": expand_configurations(models)}


def _write_queue(path: Path) -> None:
    rows, assignments = generate_queue(_selection())
    assert assignments == ASSIGNMENT_COUNT
    path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def test_controller_queue_requires_exact_official_expansion(tmp_path: Path) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)

    jobs = read_queue(queue)

    assert len(jobs) == SHARD_COUNT
    assert len({job.model_id for job in jobs}) == BASE_MODEL_COUNT
    assert len({job.configuration_id for job in jobs}) == CONFIGURATION_COUNT
    assert all(job.assigned_tasks == 70 for job in jobs)
    assert {job.endpoint_cap_status for job in jobs} == {
        "catalog_numeric",
        "undisclosed_first_party_exception",
    }


def test_controller_rejects_old_fourteen_column_queue(tmp_path: Path) -> None:
    queue = tmp_path / "old.tsv"
    queue.write_text("\t".join(["value"] * 14) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly fifteen or sixteen columns"):
        read_queue(queue)


def test_controller_keeps_complete_legacy_queue_readable(tmp_path: Path) -> None:
    frameworks = ("qiskit", "cirq", "pennylane", "cudaq")
    lines = []
    for model_index in range(9):
        for framework in frameworks:
            lines.append(
                "\t".join(
                    (
                        f"legacy-{model_index}-{framework}",
                        f"legacy/model-{model_index}",
                        "max",
                        "pass1",
                        framework,
                        "all",
                        "0",
                        f"endpoint-{model_index}",
                        "128000",
                        "author_native",
                        "catalog_numeric",
                        "max_tokens",
                        f"route-{model_index}",
                        "explicit_zero",
                        "70",
                    )
                )
            )
    queue = tmp_path / "legacy.tsv"
    queue.write_text("\n".join(lines) + "\n", encoding="utf-8")

    jobs = read_queue(queue)

    assert len(jobs) == 36
    assert {job.queue_schema_version for job in jobs} == {1}
    assert {job.configuration_id for job in jobs} == {None}


def test_model_lane_interleaves_efforts_deterministically(tmp_path: Path) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)
    jobs = [job for job in read_queue(queue) if job.model_id == "openai/gpt-5.6-sol"]

    ordered = _interleaved_model_jobs(jobs)

    efforts = EFFORTS_BY_MODEL["openai/gpt-5.6-sol"]
    assert [job.reasoning_setting for job in ordered[: len(efforts)]] == list(efforts)
    assert [job.reasoning_setting for job in ordered[len(efforts) : 2 * len(efforts)]] == [
        *efforts[1:],
        efforts[0],
    ]


def test_model_lane_contains_one_configuration_across_four_frameworks(tmp_path: Path) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)
    jobs = [job for job in read_queue(queue) if job.model_id == "openai/gpt-5.6-sol"]

    ordered = _interleaved_model_jobs(jobs)

    assert len(ordered) == 4
    assert {job.framework for job in ordered} == {"qiskit", "cirq", "pennylane", "cudaq"}
    assert {job.reasoning_setting for job in ordered} == {"max"}


def test_model_lane_continues_other_shards_after_a_single_deferred_request(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)
    jobs = [job for job in read_queue(queue) if job.model_id == "openai/gpt-5.6-sol"]
    called = []

    def run_job(job, **kwargs):
        called.append(job.job_id)
        return "deferred_infrastructure" if len(called) == 1 else "complete"

    monkeypatch.setattr("scripts.run_pass1_generation._run_job", run_job)
    generation = tmp_path / "generation"
    generation.mkdir()
    recovery = DeferredInfrastructureStore(generation)
    status = _run_model_lane(
        jobs,
        args=_runner_args(),
        segments_dir=generation / "segments",
        logs_dir=generation / "logs",
        cache_dir=generation / "cache",
        ledger=CampaignLedger(generation, expected_assignments=280),
        recovery=recovery,
    )

    assert status == "deferred_infrastructure"
    assert len(called) == 4


def test_controller_command_is_generation_only_endpoint_pinned_and_two_wide(tmp_path: Path) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)
    job = next(job for job in read_queue(queue) if job.model_id == "openai/gpt-5.6-sol")

    command = build_command(
        job,
        output=tmp_path / "segment.jsonl",
        api_key_file=tmp_path / ".env",
        source_hint=BENCHMARK_CONTENT_COMMIT,
        cache_dir=tmp_path / "cache",
        scope=SegmentScope(suite="core", task_numbers=(1, 3)),
    )

    assert command[command.index("--generation-concurrency") + 1] == "2"
    assert command[command.index("--max-retries") + 1] == "5"
    assert command[command.index("--rerun") + 1] == job.framework
    assert "--regrade" not in command
    assert command[command.index("--openrouter-endpoint-tag") + 1] == job.endpoint_tag
    assert command[command.index("--openrouter-endpoint-cap-status") + 1] == "catalog_numeric"
    assert command[command.index("--configuration-id") + 1] == job.configuration_id
    assert command[command.index("--tasks") + 1 :] == ["1", "3", "--reasoning-effort", job.reasoning_setting]
    assert "--temperature" not in command


def test_resume_scopes_contain_only_pending_suite_local_tasks() -> None:
    assignments = [
        ("core", "qiskit", "01", 0, 0),
        ("core", "qiskit", "02", 0, 0),
        ("qec", "qiskit", "qec01", 0, 0),
    ]
    pending = [assignments[1], assignments[2]]

    assert _segment_scopes(assignments, pending, existing=True) == [
        SegmentScope(suite="core", task_numbers=(2,)),
        SegmentScope(suite="qec", task_numbers=(1,)),
    ]


def test_resume_scopes_reject_malformed_suite_task_ids() -> None:
    assignment = ("qec", "qiskit", "qec-01", 0, 0)

    with pytest.raises(ValueError, match="invalid qec task ID"):
        _segment_scopes([assignment], [assignment], existing=True)


def test_campaign_summary_counts_same_task_key_separately_per_model_lane(tmp_path: Path) -> None:
    for job_id in ("model-a__qiskit", "model-b__qiskit"):
        directory = tmp_path / "segments" / job_id
        directory.mkdir(parents=True)
        (directory / "route-s001-all.jsonl").write_text(
            '{"kind":"result","suite":"core","framework":"qiskit","task_id":"01",'
            '"sample_index":0,"attempt_index":0,"status":"generated",'
            '"provider_response":{"metadata":{"route":{"route_verified":true}},'
            '"usage":{"cost_usd":0.5}}}\n',
            encoding="utf-8",
        )

    summary = campaign_summary(tmp_path, expected_assignments=2520)

    assert summary["accepted_logical_requests"] == 2
    assert summary["provider_cost_usd"] == 1.0
    assert summary["accepted_scan_errors"] == {}


def test_campaign_summary_tolerates_an_in_progress_jsonl_line(tmp_path: Path) -> None:
    directory = tmp_path / "segments" / "model-a__qiskit"
    directory.mkdir(parents=True)
    (directory / "route-s001-all.jsonl").write_text('{"kind":"result"', encoding="utf-8")

    summary = campaign_summary(tmp_path, expected_assignments=2520)

    assert summary["accepted_logical_requests"] == 0
    assert "model-a__qiskit" in summary["accepted_scan_errors"]


def test_controller_rejects_nonnumeric_endpoint_cap_status(tmp_path: Path) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)
    rows = queue.read_text(encoding="utf-8").splitlines()
    fields = rows[0].split("\t")
    fields[10] = "undisclosed_first_party_exception"
    rows[0] = "\t".join(fields)
    queue.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen model output policy"):
        read_queue(queue)


def _runner_job() -> QueueJob:
    return QueueJob(
        job_id="openai-gpt-5-6-sol__effort-max__pass1__qiskit",
        model_id="openai/gpt-5.6-sol",
        reasoning_setting="max",
        protocol="pass1",
        framework="qiskit",
        suite="all",
        max_tasks=0,
        endpoint_tag="openai",
        configured_output_tokens=128000,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-frozen",
        temperature_behavior="explicit_zero",
        assigned_tasks=70,
        configuration_id="openai-gpt-5-6-sol__effort-max",
    )


def _runner_args() -> argparse.Namespace:
    return argparse.Namespace(
        api_key_file=None,
        source_hint=BENCHMARK_CONTENT_COMMIT,
        provider_timeout=600.0,
        max_retries=5,
        retry_base_delay=1.0,
        retry_max_delay=60.0,
    )


def _accepted_record(task_id: str) -> dict:
    return {
        "kind": "result",
        "suite": "core",
        "framework": "qiskit",
        "task_id": task_id,
        "sample_index": 0,
        "attempt_index": 0,
        "status": "generated",
        "provider_response": {
            "metadata": {
                "generation_id": f"gen-{task_id}",
                "route": {
                    "route_verified": True,
                    "configuration_id": "openai-gpt-5-6-sol__effort-max",
                    "endpoint_tag": "openai",
                    "max_output_tokens": 128000,
                    "output_limit_source": "author_native",
                    "endpoint_cap_status": "catalog_numeric",
                    "output_token_parameter": "max_tokens",
                    "route_revision": "route-frozen",
                    "temperature": 0.0,
                },
                "attempt_history": [
                    {
                        "attempt_number": 1,
                        "status": "accepted_model_outcome",
                        "transient": False,
                        "generation_id": f"gen-{task_id}",
                        "usage": {"cost_usd": 0.25},
                    }
                ],
            },
            "usage": {"cost_usd": 0.25},
        },
    }


def _exhausted_record(task_id: str) -> dict:
    history = [
        {
            "attempt_number": number,
            "status": "transient_infrastructure",
            "transient": True,
            "http_status": 504,
            "error": "temporary provider timeout",
        }
        for number in range(1, 7)
    ]
    return {
        "kind": "result",
        "suite": "core",
        "framework": "qiskit",
        "task_id": task_id,
        "sample_index": 0,
        "attempt_index": 0,
        "status": "infrastructure_error",
        "provider_response": {
            "metadata": {
                "infrastructure_error": True,
                "retryable_infrastructure": True,
                "retry_exhausted": True,
                "infrastructure_attempts": 6,
                "failure_classification": "transient_http_exhausted",
                "attempt_history": history,
                "route": {"route_verified": False},
            },
            "usage": None,
        },
    }


def test_job_runs_deferred_sweep_and_never_regenerates_accepted_output(tmp_path: Path, monkeypatch) -> None:
    assignments = [("core", "qiskit", "01", 0, 0), ("core", "qiskit", "02", 0, 0)]
    monkeypatch.setattr("scripts.run_pass1_generation._assignments", lambda job: assignments)
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        output = Path(command[command.index("--out") + 1])
        records = [_exhausted_record("01"), _accepted_record("02")] if calls == 1 else [_accepted_record("01")]
        output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.run_pass1_generation.subprocess.run", run)
    generation = tmp_path / "generation"
    for path in (generation / "segments", generation / "logs", generation / "cache"):
        path.mkdir(parents=True)
    ledger = CampaignLedger(generation, expected_assignments=2)
    now = [datetime(2026, 8, 11, tzinfo=UTC)]
    recovery = DeferredInfrastructureStore(generation, clock=lambda: now[0])

    first_status = _run_job(
        _runner_job(),
        args=_runner_args(),
        segments_dir=generation / "segments",
        logs_dir=generation / "logs",
        cache_dir=generation / "cache",
        ledger=ledger,
        recovery=recovery,
    )
    now[0] += timedelta(minutes=30, seconds=1)
    status = _run_job(
        _runner_job(),
        args=_runner_args(),
        segments_dir=generation / "segments",
        logs_dir=generation / "logs",
        cache_dir=generation / "cache",
        ledger=ledger,
        recovery=recovery,
    )

    assert first_status == "deferred_infrastructure"
    assert status == "complete"
    assert calls == 2
    assert (
        recovery.deferred_keys(model_id="openai/gpt-5.6-sol", job_id="openai-gpt-5-6-sol__effort-max__pass1__qiskit")
        == set()
    )
    attempt_events = (generation / "provider-attempt-ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(attempt_events) == 8


def test_job_opens_circuit_after_two_exhausted_logical_requests_and_leaves_later_work_pending(
    tmp_path: Path, monkeypatch
) -> None:
    assignments = [
        ("core", "qiskit", "01", 0, 0),
        ("core", "qiskit", "02", 0, 0),
        ("core", "qiskit", "03", 0, 0),
    ]
    monkeypatch.setattr("scripts.run_pass1_generation._assignments", lambda job: assignments)
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps(_exhausted_record(f"{calls:02d}")) + "\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.run_pass1_generation.subprocess.run", run)
    generation = tmp_path / "generation"
    for path in (generation / "segments", generation / "logs", generation / "cache"):
        path.mkdir(parents=True)
    ledger = CampaignLedger(generation, expected_assignments=3)
    recovery = DeferredInfrastructureStore(generation)

    status = _run_job(
        _runner_job(),
        args=_runner_args(),
        segments_dir=generation / "segments",
        logs_dir=generation / "logs",
        cache_dir=generation / "cache",
        ledger=ledger,
        recovery=recovery,
    )

    assert status == "deferred_infrastructure"
    assert calls == 2
    assert recovery.circuit_is_open(model_id="openai/gpt-5.6-sol", endpoint_tag="openai", route_revision="route-frozen")
    assert ("core", "qiskit", "03", 0, 0) not in recovery.deferred_keys(
        model_id="openai/gpt-5.6-sol", job_id="openai-gpt-5-6-sol__effort-max__pass1__qiskit"
    )


def test_controller_continues_unrelated_model_lanes_when_one_lane_defers(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)
    called = []

    def lane(jobs, **kwargs):
        model_id = jobs[0].model_id
        called.append(model_id)
        return "deferred_infrastructure" if model_id == "anthropic/claude-fable-5" else "complete"

    monkeypatch.setattr("scripts.run_pass1_generation._run_model_lane", lane)
    monkeypatch.setattr("scripts.run_pass1_generation._validate_harness_commit", lambda expected: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    run_manifest = tmp_path / "run-manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "harness_commit": "frozen",
                "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
                "base_models": BASE_MODEL_COUNT,
                "configurations": CONFIGURATION_COUNT,
                "shards": SHARD_COUNT,
                "logical_requests": ASSIGNMENT_COUNT,
                "fresh_logical_requests": FRESH_ASSIGNMENT_COUNT,
                "historical_imported_requests": HISTORICAL_IMPORT_ASSIGNMENT_COUNT,
                "historical_imports": {"configuration_ids": [], "configurations": 0, "records": 0},
                "artifacts": {"queue": {"sha256": hashlib.sha256(queue.read_bytes()).hexdigest()}},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--queue",
            str(queue),
            "--out-dir",
            str(tmp_path / "generation"),
            "--source-hint",
            BENCHMARK_CONTENT_COMMIT,
            "--harness-commit",
            "frozen",
            "--run-manifest",
            str(run_manifest),
        ]
    )

    assert exit_code == 2
    assert len(called) == BASE_MODEL_COUNT
    assert len(set(called)) == BASE_MODEL_COUNT


def test_controller_can_resume_one_model_lane_from_the_full_frozen_campaign(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)
    called = []

    def lane(jobs, **kwargs):
        called.append(jobs[0].model_id)
        return "complete"

    monkeypatch.setattr("scripts.run_pass1_generation._run_model_lane", lane)
    monkeypatch.setattr("scripts.run_pass1_generation._validate_harness_commit", lambda expected: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    run_manifest = tmp_path / "run-manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "harness_commit": "frozen",
                "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
                "base_models": BASE_MODEL_COUNT,
                "configurations": CONFIGURATION_COUNT,
                "shards": SHARD_COUNT,
                "logical_requests": ASSIGNMENT_COUNT,
                "fresh_logical_requests": FRESH_ASSIGNMENT_COUNT,
                "historical_imported_requests": HISTORICAL_IMPORT_ASSIGNMENT_COUNT,
                "historical_imports": {"configuration_ids": [], "configurations": 0, "records": 0},
                "artifacts": {"queue": {"sha256": hashlib.sha256(queue.read_bytes()).hexdigest()}},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--queue",
            str(queue),
            "--out-dir",
            str(tmp_path / "generation"),
            "--source-hint",
            BENCHMARK_CONTENT_COMMIT,
            "--harness-commit",
            "frozen",
            "--run-manifest",
            str(run_manifest),
            "--only-model",
            "x-ai/grok-4.6",
        ]
    )

    assert exit_code == 0
    assert called == ["x-ai/grok-4.6"]
    summary = json.loads((tmp_path / "generation" / "controller-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "selected_model_lanes_complete"


def test_controller_rejects_historical_imports_for_new_content(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "pass1.tsv"
    _write_queue(queue)
    called = []

    def lane(jobs, **kwargs):
        called.extend(job.configuration_id for job in jobs)
        return "complete"

    monkeypatch.setattr("scripts.run_pass1_generation._run_model_lane", lane)
    monkeypatch.setattr("scripts.run_pass1_generation._validate_harness_commit", lambda expected: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    run_manifest = tmp_path / "run-manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "harness_commit": "frozen",
                "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
                "base_models": BASE_MODEL_COUNT,
                "configurations": CONFIGURATION_COUNT,
                "shards": SHARD_COUNT,
                "logical_requests": ASSIGNMENT_COUNT,
                "fresh_logical_requests": FRESH_ASSIGNMENT_COUNT,
                "historical_imported_requests": HISTORICAL_IMPORT_ASSIGNMENT_COUNT,
                "artifacts": {"queue": {"sha256": hashlib.sha256(queue.read_bytes()).hexdigest()}},
                "historical_imports": {
                    "configuration_ids": ["openai-gpt-5-6-sol__effort-max"],
                    "configurations": 1,
                    "records": 280,
                },
            }
        ),
        encoding="utf-8",
    )
    imported = "openai-gpt-5-6-sol__effort-max"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--queue",
                str(queue),
                "--out-dir",
                str(tmp_path / "generation"),
                "--source-hint",
                BENCHMARK_CONTENT_COMMIT,
                "--harness-commit",
                "frozen",
                "--run-manifest",
                str(run_manifest),
                "--only-model",
                "openai/gpt-5.6-sol",
                "--skip-configuration",
                imported,
            ]
        )

    assert exc_info.value.code == 2
    assert called == []
