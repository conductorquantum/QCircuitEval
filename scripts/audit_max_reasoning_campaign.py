#!/usr/bin/env python3
"""Fail closed unless the fresh maximum-reasoning campaign is publication-ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_effort_sweep_manifest import validate_run_manifest
from scripts.run_pass1_generation import QueueJob, _assignments, read_queue

from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    BASE_MODEL_COUNT,
    BENCHMARK_CONTENT_COMMIT,
)
from qceval.production.resume import accepted_records, logical_key

FRAMEWORKS = frozenset({"qiskit", "cirq", "pennylane", "cudaq"})
SUITE_COUNTS = {"core": 232, "qec": 48}
RUN_MANIFEST_ARTIFACTS = frozenset(
    {
        "capability_registry",
        "raw_openrouter_catalog",
        "endpoint_selection",
        "preflight_hashes",
        "queue",
        "benchmark_content_manifest",
        "canaries",
        "diagnostics_manifest",
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--scope-manifest", type=Path)
    parser.add_argument("--route-recovery-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_campaign(
        run_manifest=args.run_manifest,
        queue=args.queue,
        generation_dir=args.generation_dir,
        candidates_dir=args.candidates_dir,
        offline_dir=args.offline_dir,
        scope_manifest=args.scope_manifest,
        route_recovery_manifest=args.route_recovery_manifest,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "publication_ready": report["publication_ready"],
                "concerns": report["concerns"],
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["publication_ready"] else 2


def audit_campaign(  # noqa: C901 - coordinates independent fail-closed evidence checks
    *,
    run_manifest: Path,
    queue: Path,
    generation_dir: Path,
    candidates_dir: Path,
    offline_dir: Path,
    scope_manifest: Path | None = None,
    route_recovery_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate campaign provenance, generation, grading, and AWS cleanup evidence."""
    checks: dict[str, dict[str, Any]] = {}
    concerns: list[str] = []

    def check(name: str, callback: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
        try:
            evidence = callback()
        except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
            concern = f"{name}: {exc}"
            concerns.append(concern)
            checks[name] = {"passed": False, "error": str(exc)}
            return None
        checks[name] = {"passed": True, **evidence}
        return evidence

    queue_context = check("queue", lambda: _audit_queue(queue))
    scope_context = (
        check("scope_manifest", lambda: _audit_scope_manifest(scope_manifest, queue))
        if scope_manifest is not None
        else None
    )
    manifest_queue = Path(scope_context["parent_queue"]) if scope_context is not None else queue
    manifest_evidence = check("run_manifest", lambda: _audit_run_manifest(run_manifest, manifest_queue))
    route_recovery_context = (
        check("route_recovery_manifest", lambda: _audit_route_recovery_manifest(route_recovery_manifest, queue))
        if route_recovery_manifest is not None
        else None
    )

    generation_context: dict[str, Any] | None = None
    candidate_context: dict[str, Any] | None = None
    offline_context: dict[str, Any] | None = None
    scope_is_valid = scope_manifest is None or scope_context is not None
    route_recovery_is_valid = route_recovery_manifest is None or route_recovery_context is not None
    if queue_context is not None and scope_is_valid and route_recovery_is_valid:
        jobs = queue_context["jobs"]
        expected_by_configuration = queue_context["expected_by_configuration"]
        route_recoveries = route_recovery_context["route_recoveries"] if route_recovery_context is not None else {}
        generation_context = check(
            "generation",
            lambda: _audit_generation(
                generation_dir,
                jobs,
                expected_by_configuration,
                route_recoveries,
                scoped=scope_manifest is not None,
            ),
        )
        candidate_context = check(
            "candidates",
            lambda: _audit_candidates(
                candidates_dir,
                queue,
                jobs,
                expected_by_configuration,
                generation_context["records_by_configuration"] if generation_context is not None else None,
                route_recoveries,
            ),
        )
        if candidate_context is not None:
            offline_context = check(
                "offline_grading",
                lambda: _audit_offline(
                    offline_dir,
                    queue,
                    candidates_dir / "manifest.json",
                    jobs,
                    expected_by_configuration,
                    candidate_context["records_by_configuration"],
                    route_recoveries,
                ),
            )

    source: dict[str, Any] = {
        "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
        "run_manifest": str(run_manifest.resolve()),
        "run_manifest_sha256": _sha256_if_file(run_manifest),
        "queue": str(queue.resolve()),
        "queue_sha256": _sha256_if_file(queue),
    }
    if manifest_evidence is not None:
        source["generation_harness_commit"] = manifest_evidence["harness_commit"]
    if scope_manifest is not None:
        source["scope_manifest"] = str(scope_manifest.resolve())
        source["scope_manifest_sha256"] = _sha256_if_file(scope_manifest)
    if route_recovery_manifest is not None:
        source["route_recovery_manifest"] = str(route_recovery_manifest.resolve())
        source["route_recovery_manifest_sha256"] = _sha256_if_file(route_recovery_manifest)
    if candidate_context is not None:
        source["candidate_manifest_sha256"] = candidate_context["manifest_sha256"]
    if offline_context is not None:
        source["termination_readback_sha256"] = offline_context["termination_readback_sha256"]

    coverage: dict[str, Any] = {}
    if generation_context is not None:
        coverage.update(
            {
                "fresh_generation_records": generation_context["accepted_records"],
                "generation_cost_usd": generation_context["provider_cost_usd"],
                "accepted_generation_cost_usd": generation_context["accepted_provider_cost_usd"],
                "physical_provider_attempts": generation_context["physical_provider_attempts"],
            }
        )
    if offline_context is not None:
        coverage.update(
            {
                "offline_regraded_records": offline_context["records"],
                "unique_configuration_task_keys": offline_context["unique_configuration_task_keys"],
                "route_verified_records": offline_context["route_verified_records"],
                "provider_cost_covered_records": offline_context["cost_covered_records"],
                "token_usage_covered_records": offline_context["token_usage_covered_records"],
                "passed_records": offline_context["passed_records"],
                "provider_cost_usd": offline_context["provider_cost_usd"],
            }
        )

    expected_checks = {
        "queue",
        "run_manifest",
        "generation",
        "candidates",
        "offline_grading",
    }
    if scope_manifest is not None:
        expected_checks.add("scope_manifest")
    if route_recovery_manifest is not None:
        expected_checks.add("route_recovery_manifest")
    scope = _scope_from_queue(queue_context) if queue_context is not None else {}
    return {
        "schema_version": "qceval.max_reasoning_final_audit.v1",
        "completed_at_utc": _now(),
        "publication_ready": not concerns and set(checks) == expected_checks,
        "scope": scope,
        "source": source,
        "coverage": coverage,
        "checks": {name: _public_evidence(value) for name, value in checks.items()},
        "concerns": concerns,
    }


def _audit_queue(path: Path) -> dict[str, Any]:
    jobs = read_queue(path, validate_campaign=False)
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("queue is empty or contains duplicate job IDs")
    expected_by_configuration: dict[str, set[tuple[str, str, str, int, int]]] = defaultdict(set)
    jobs_by_configuration: dict[str, list[QueueJob]] = defaultdict(list)
    for job in jobs:
        if job.configuration_id is None:
            raise ValueError("queue contains a configuration without a schema-v2 identity")
        expected_by_configuration[job.configuration_id].update(_assignments(job))
        jobs_by_configuration[job.configuration_id].append(job)
    if any(len(keys) != 280 for keys in expected_by_configuration.values()):
        raise ValueError("queue does not expand to 280 unique tasks per configuration")
    for config_id, config_jobs in jobs_by_configuration.items():
        if (
            len(config_jobs) != 4
            or {job.framework for job in config_jobs} != FRAMEWORKS
            or {job.assigned_tasks for job in config_jobs} != {70}
            or len({job.model_id for job in config_jobs}) != 1
            or {job.protocol for job in config_jobs} != {"pass1"}
        ):
            raise ValueError(f"queue configuration is not one four-framework Pass@1 matrix: {config_id}")
    return {
        "jobs": jobs,
        "expected_by_configuration": dict(expected_by_configuration),
        "sha256": _sha256(path),
        "shards": len(jobs),
        "configurations": len(expected_by_configuration),
        "assignments": sum(job.assigned_tasks for job in jobs),
    }


def _audit_scope_manifest(path: Path, queue: Path) -> dict[str, Any]:
    payload = _json(path)
    if payload.get("schema_version") != "qceval.pass1_scope.v1":
        raise ValueError("scope manifest has an unsupported schema version")
    parent_queue = Path(str(payload.get("parent_queue")))
    scoped_queue = Path(str(payload.get("queue")))
    if scoped_queue.resolve() != queue.resolve() or payload.get("queue_sha256") != _sha256(queue):
        raise ValueError("scope manifest does not identify the audited queue")
    if not parent_queue.is_file() or payload.get("parent_queue_sha256") != _sha256(parent_queue):
        raise ValueError("scope manifest parent queue is missing or hash-invalid")

    parent_jobs = read_queue(parent_queue)
    scoped_jobs = read_queue(queue, validate_campaign=False)
    parent_models = {job.model_id for job in parent_jobs}
    scoped_models = {job.model_id for job in scoped_jobs}
    excluded_models = parent_models - scoped_models
    expected = {
        "parent_shards": len(parent_jobs),
        "parent_logical_requests": sum(job.assigned_tasks for job in parent_jobs),
        "excluded_models": sorted(excluded_models),
        "base_models": len(scoped_models),
        "models": sorted(scoped_models),
        "configurations": len({job.configuration_id for job in scoped_jobs}),
        "shards": len(scoped_jobs),
        "logical_requests": sum(job.assigned_tasks for job in scoped_jobs),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("scope manifest counts or model identities do not match its queues")
    parent_rows = set(parent_queue.read_text(encoding="utf-8").splitlines())
    scoped_rows = set(queue.read_text(encoding="utf-8").splitlines())
    if not scoped_rows or not scoped_rows < parent_rows:
        raise ValueError("scoped queue is not a strict row-preserving subset of its parent")
    return {**expected, "parent_queue": str(parent_queue.resolve()), "sha256": _sha256(path)}


def _audit_route_recovery_manifest(path: Path, queue: Path) -> dict[str, Any]:  # noqa: C901
    """Validate exact task-scoped route replacements and their live preflight evidence."""
    payload = _json(path)
    if payload.get("schema_version") != "qceval.route_recovery.v1":
        raise ValueError("route recovery manifest has an unsupported schema version")
    declared_queue = _resolve_manifest_path(path, payload.get("queue"))
    if (
        declared_queue.resolve() != queue.resolve()
        or payload.get("queue_sha256") != _sha256(queue)
        or payload.get("benchmark_content_commit") != BENCHMARK_CONTENT_COMMIT
    ):
        raise ValueError("route recovery manifest does not identify the audited benchmark queue")
    revisions = payload.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        raise ValueError("route recovery manifest has no revisions")

    jobs = read_queue(queue, validate_campaign=False)
    expected_assignments: dict[tuple[str, str, str, str, int, int], QueueJob] = {}
    for job in jobs:
        assert job.configuration_id is not None
        for key in _assignments(job):
            expected_assignments[(job.configuration_id, *key)] = job

    route_recoveries: dict[tuple[str, str, str, str, int, int], dict[str, Any]] = {}
    revision_evidence: list[dict[str, Any]] = []
    for revision in revisions:
        if not isinstance(revision, Mapping):
            raise ValueError("route recovery revision is not an object")
        config_id = revision.get("configuration_id")
        model_id = revision.get("model_id")
        if not isinstance(config_id, str) or not isinstance(model_id, str):
            raise ValueError("route recovery revision lacks configuration/model identity")
        selection_path, selection_sha = _verified_manifest_artifact(path, revision.get("selection"), "selection")
        canary_path, canary_sha = _verified_manifest_artifact(path, revision.get("canary"), "canary")
        selection = _json(selection_path)
        configurations = selection.get("configurations")
        selected = configurations.get(config_id) if isinstance(configurations, Mapping) else None
        if not isinstance(selected, Mapping) or selected.get("model_id") != model_id:
            raise ValueError("route recovery selection does not contain the declared configuration")
        route_expectation = _route_expectation_from_selection(selected)

        canary_rows = _jsonl_objects(canary_path)
        if len(canary_rows) != 2:
            raise ValueError("route recovery canary must contain one result and one summary")
        canary, summary = canary_rows
        canary_metadata = canary.get("provider_response_metadata")
        canary_route = canary_metadata.get("route") if isinstance(canary_metadata, Mapping) else None
        if (
            canary.get("kind") != "canary"
            or canary.get("status") != "passed"
            or canary.get("configuration_id") != config_id
            or canary.get("model_id") != model_id
            or not isinstance(canary_route, Mapping)
            or not _route_matches_expectation(canary_route, route_expectation)
            or summary.get("kind") != "summary"
            or summary.get("benchmark_denominator_member") is not False
            or summary.get("models") != 1
            or summary.get("passed") != 1
            or summary.get("failed") != []
        ):
            raise ValueError("route recovery canary does not prove the declared exact route")

        assignments = revision.get("assignments")
        if not isinstance(assignments, list) or not assignments:
            raise ValueError("route recovery revision has no exact task assignments")
        for assignment in assignments:
            key = _route_recovery_key(config_id, assignment)
            job = expected_assignments.get(key)
            if job is None or job.model_id != model_id:
                raise ValueError(f"route recovery assignment is outside the audited queue: {key}")
            if key in route_recoveries:
                raise ValueError(f"duplicate route recovery assignment: {key}")
            if (
                route_expectation["endpoint_tag"] == job.endpoint_tag
                and route_expectation["route_revision"] == job.route_revision
            ):
                raise ValueError("route recovery assignment does not change its frozen route")
            route_recoveries[key] = route_expectation
        revision_evidence.append(
            {
                "configuration_id": config_id,
                "model_id": model_id,
                "endpoint_tag": route_expectation["endpoint_tag"],
                "route_revision": route_expectation["route_revision"],
                "assignments": len(assignments),
                "selection_sha256": selection_sha,
                "canary_sha256": canary_sha,
            }
        )
    return {
        "sha256": _sha256(path),
        "recovered_assignments": len(route_recoveries),
        "revisions": revision_evidence,
        "route_recoveries": route_recoveries,
    }


def _scope_from_queue(queue_context: Mapping[str, Any]) -> dict[str, Any]:
    jobs = queue_context["jobs"]
    assignments = int(queue_context["assignments"])
    return {
        "base_models": len({job.model_id for job in jobs}),
        "configurations": int(queue_context["configurations"]),
        "framework_shards": int(queue_context["shards"]),
        "logical_records": assignments,
        "fresh_records": assignments,
        "historical_records": 0,
    }


def _audit_run_manifest(path: Path, queue: Path) -> dict[str, Any]:
    payload = _json(path)
    harness_commit = payload.get("harness_commit")
    if not isinstance(harness_commit, str) or len(harness_commit) != 40:
        raise ValueError("run manifest lacks a full generation harness commit")
    validate_run_manifest(path, queue=queue, harness_commit=harness_commit)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != RUN_MANIFEST_ARTIFACTS:
        raise ValueError("run manifest does not contain the exact immutable artifact inventory")
    verified = 0
    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"run manifest artifact {name!r} is not an object")
        artifact = Path(str(raw.get("path")))
        if not artifact.is_file():
            raise ValueError(f"run manifest artifact is missing: {artifact}")
        if raw.get("sha256") != _sha256(artifact) or raw.get("bytes") != artifact.stat().st_size:
            raise ValueError(f"run manifest artifact changed after freeze: {name}")
        verified += 1
    return {
        "harness_commit": harness_commit,
        "artifacts_verified": verified,
        "fresh_logical_requests": payload["fresh_logical_requests"],
    }


def _audit_generation(  # noqa: C901 - validates every generation provenance layer
    root: Path,
    jobs: Sequence[QueueJob],
    expected_by_configuration: Mapping[str, set[tuple[str, str, str, int, int]]],
    route_recoveries: Mapping[tuple[str, str, str, str, int, int], Mapping[str, Any]],
    *,
    scoped: bool,
) -> dict[str, Any]:
    summary = _json(root / "controller-summary.json")
    lane_status = summary.get("model_lane_status")
    if scoped:
        if summary.get("accepted_scan_errors") != {}:
            raise ValueError("generation controller reports accepted-record scan errors")
    elif (
        summary.get("status") != "generation_complete"
        or summary.get("expected_logical_requests") != ASSIGNMENT_COUNT
        or summary.get("accepted_logical_requests") != ASSIGNMENT_COUNT
        or summary.get("deferred_infrastructure_requests") != 0
        or summary.get("open_endpoint_circuits") != 0
        or summary.get("accepted_scan_errors") != {}
        or not isinstance(lane_status, Mapping)
        or len(lane_status) != BASE_MODEL_COUNT
        or set(lane_status.values()) != {"complete"}
    ):
        raise ValueError("generation controller did not finish every model lane without unresolved infrastructure")

    segment_root = root / "segments"
    actual_job_dirs = {path.name for path in segment_root.iterdir() if path.is_dir()}
    expected_job_dirs = {job.job_id for job in jobs}
    if (not scoped and actual_job_dirs != expected_job_dirs) or (scoped and not expected_job_dirs <= actual_job_dirs):
        raise ValueError("generation segment directories do not cover the audited queue")
    records_by_configuration: dict[str, dict[tuple[str, str, str, int, int], dict[str, Any]]] = defaultdict(dict)
    provider_cost = 0.0
    recovery_cost = 0.0
    recovery_attempts = 0
    recovery_transients = 0
    recovery_generation_ids: set[str] = set()
    for job in jobs:
        paths = sorted((segment_root / job.job_id).glob("*.jsonl"))
        records = accepted_records(paths, strict_provenance=True)
        if set(records) != set(_assignments(job)):
            raise ValueError(f"{job.job_id}: accepted generation keys do not match the frozen assignment set")
        for key, record in records.items():
            _validate_route(record, job, route_recoveries)
            assert job.configuration_id is not None
            if key in records_by_configuration[job.configuration_id]:
                raise ValueError(f"duplicate generation key in {job.configuration_id}: {key}")
            records_by_configuration[job.configuration_id][key] = record
            provider_cost += _cost(record)
            recovery_key = (job.configuration_id, *key)
            if recovery_key in route_recoveries:
                attempts, cost, transients, generation_id = _recovery_attempt_evidence(record)
                recovery_attempts += attempts
                recovery_cost += cost
                recovery_transients += transients
                if generation_id in recovery_generation_ids:
                    raise ValueError("recovered records contain duplicate provider generation IDs")
                recovery_generation_ids.add(generation_id)
    if {key: set(records) for key, records in records_by_configuration.items()} != dict(expected_by_configuration):
        raise ValueError("generation records do not cover the exact configuration assignment matrix")

    attempt_path = root / "provider-attempt-ledger.jsonl"
    all_attempts = _jsonl_objects(attempt_path)
    selected_configurations = set(expected_by_configuration)
    attempts = (
        [row for row in all_attempts if row.get("configuration_id") in selected_configurations]
        if scoped
        else all_attempts
    )
    expected_assignments = sum(len(keys) for keys in expected_by_configuration.values())
    event_ids = [row.get("event_id") for row in attempts]
    if len(event_ids) < expected_assignments or any(not isinstance(value, str) for value in event_ids):
        raise ValueError("provider-attempt ledger has fewer physical requests than the logical denominator")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("provider-attempt ledger contains duplicate event IDs")
    ledger_generation_ids = {
        generation_id for attempt in attempts if isinstance((generation_id := attempt.get("generation_id")), str)
    }
    if recovery_generation_ids & ledger_generation_ids:
        raise ValueError("recovered physical attempts are duplicated in the controller ledger")
    if scoped:
        controller_cost = recovery_cost
        for attempt in attempts:
            value = attempt.get("provider_cost_usd")
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
                raise ValueError("scoped provider-attempt ledger contains invalid cost evidence")
            controller_cost += float(value)
        if controller_cost + 1e-8 < provider_cost:
            raise ValueError("scoped attempt-ledger cost is smaller than accepted provider-record cost")
    else:
        controller_cost = summary.get("provider_cost_usd")
        if (
            isinstance(controller_cost, bool)
            or not isinstance(controller_cost, int | float)
            or not math.isfinite(controller_cost)
            or controller_cost < provider_cost
        ):
            raise ValueError("generation controller cost is smaller than accepted provider-record cost")
        if summary.get("physical_provider_attempts") != len(attempts):
            raise ValueError("generation controller and provider-attempt ledger disagree")
    return {
        "accepted_records": sum(len(records) for records in records_by_configuration.values()),
        "records_by_configuration": dict(records_by_configuration),
        "provider_cost_usd": float(controller_cost),
        "accepted_provider_cost_usd": provider_cost,
        "physical_provider_attempts": len(attempts) + recovery_attempts,
        "transient_infrastructure_records": sum(
            attempt.get("status") == "transient_infrastructure" for attempt in attempts
        )
        + recovery_transients
        if scoped
        else int(summary.get("infrastructure_records", 0)),
    }


def _audit_candidates(  # noqa: C901 - validates every candidate artifact and task key
    root: Path,
    queue: Path,
    jobs: Sequence[QueueJob],
    expected_by_configuration: Mapping[str, set[tuple[str, str, str, int, int]]],
    generation_records: Mapping[str, Mapping[tuple[str, str, str, int, int], dict[str, Any]]] | None,
    route_recoveries: Mapping[tuple[str, str, str, str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    artifacts = manifest.get("artifacts")
    expected_configurations = len(expected_by_configuration)
    expected_models = len({job.model_id for job in jobs})
    expected_records = sum(len(keys) for keys in expected_by_configuration.values())
    if (
        manifest.get("base_models") != expected_models
        or manifest.get("configurations") != expected_configurations
        or manifest.get("records") != expected_records
        or manifest.get("imported_configurations") != 0
        or manifest.get("imported_records") != 0
        or manifest.get("imports_manifest") is not None
        or manifest.get("queue_sha256") != _sha256(queue)
        or not isinstance(artifacts, list)
        or len(artifacts) != expected_configurations
    ):
        raise ValueError("candidate manifest does not prove the scoped fully fresh 280-record artifacts")
    job_by_scope = _job_by_scope(jobs)
    records_by_configuration: dict[str, dict[tuple[str, str, str, int, int], dict[str, Any]]] = {}
    seen: set[str] = set()
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise ValueError("candidate manifest artifact is not an object")
        config_id = raw.get("configuration_id")
        if not isinstance(config_id, str) or config_id in seen or config_id not in expected_by_configuration:
            raise ValueError(f"invalid candidate configuration identity: {config_id!r}")
        seen.add(config_id)
        path = Path(str(raw.get("path")))
        if not path.is_file() or raw.get("sha256") != _sha256(path):
            raise ValueError(f"candidate artifact is missing or hash-invalid: {config_id}")
        records, summary = _read_run(path)
        keyed = {logical_key(record): record for record in records}
        if len(keyed) != 280 or set(keyed) != expected_by_configuration[config_id]:
            raise ValueError(f"candidate artifact has incomplete or duplicate task keys: {config_id}")
        if (
            raw.get("records") != 280
            or raw.get("route_verified_records") != 280
            or raw.get("cost_covered_records") != 280
            or summary.get("configuration_id") != config_id
        ):
            raise ValueError(f"candidate manifest evidence is incomplete: {config_id}")
        for key, record in keyed.items():
            job = job_by_scope[(config_id, str(record.get("framework")))]
            _validate_route(record, job, route_recoveries)
            if generation_records is not None and record.get("provider_response") != generation_records[config_id][
                key
            ].get("provider_response"):
                raise ValueError(f"candidate materialization changed provider provenance: {config_id} {key}")
        records_by_configuration[config_id] = keyed
    if seen != set(expected_by_configuration):
        raise ValueError("candidate manifest omits one or more frozen configurations")
    return {
        "manifest_sha256": _sha256(manifest_path),
        "artifacts": len(artifacts),
        "records": expected_records,
        "records_by_configuration": records_by_configuration,
    }


def _audit_offline(  # noqa: C901 - final publication gate intentionally checks every evidence layer
    root: Path,
    queue: Path,
    candidate_manifest: Path,
    jobs: Sequence[QueueJob],
    expected_by_configuration: Mapping[str, set[tuple[str, str, str, int, int]]],
    candidate_records: Mapping[str, Mapping[tuple[str, str, str, int, int], dict[str, Any]]],
    route_recoveries: Mapping[tuple[str, str, str, str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    expected_configurations = len(expected_by_configuration)
    expected_assignments = sum(len(keys) for keys in expected_by_configuration.values())
    expected_shards = len(jobs)
    if _sha256(root / "offline-queue.tsv") != _sha256(queue):
        raise ValueError("offline queue copy differs from the frozen queue")
    if _sha256(root / "candidate-manifest.json") != _sha256(candidate_manifest):
        raise ValueError("offline candidate manifest copy differs from the materialized candidate manifest")

    expected_job_ids = {job.job_id for job in jobs}
    shard_paths = {path.stem: path for path in (root / "shards").glob("*.jsonl")}
    if set(shard_paths) != expected_job_ids:
        raise ValueError("offline shard files do not exactly match all 36 frozen queue jobs")
    for state_name, expected_count in (("pending", 0), ("running", 0), ("failed", 0), ("done", expected_shards)):
        state_dir = root / "state" / state_name
        count = sum(path.is_file() for path in state_dir.glob("*.job")) if state_dir.is_dir() else 0
        if count != expected_count:
            raise ValueError(f"offline state/{state_name} contains {count} jobs; expected {expected_count}")

    offline_by_configuration: dict[str, dict[tuple[str, str, str, int, int], dict[str, Any]]] = defaultdict(dict)
    job_by_id = {job.job_id: job for job in jobs}
    for job_id, path in shard_paths.items():
        job = job_by_id[job_id]
        records, _summary = _read_run(path)
        keyed = {logical_key(record): record for record in records}
        if len(records) != 70 or len(keyed) != 70 or set(keyed) != set(_assignments(job)):
            raise ValueError(f"offline shard is not the exact 70-task assignment: {job_id}")
        assert job.configuration_id is not None
        for key, record in keyed.items():
            _validate_offline_record(record, job, route_recoveries)
            if key in offline_by_configuration[job.configuration_id]:
                raise ValueError(f"duplicate offline task key in {job.configuration_id}: {key}")
            offline_by_configuration[job.configuration_id][key] = record

    merged_paths = {path.name: path for path in (root / "merged").glob("*.jsonl")}
    expected_names = {f"{config_id}__pass1.regraded.jsonl" for config_id in expected_by_configuration}
    if set(merged_paths) != expected_names:
        raise ValueError("merged offline artifacts do not exactly match the audited configurations")
    job_by_configuration = _job_by_configuration(jobs)
    job_by_scope = _job_by_scope(jobs)
    status_counts: Counter[str] = Counter()
    total_cost = 0.0
    token_covered = 0
    passed = 0
    merged_hashes: dict[str, str] = {}
    model_costs: dict[str, float] = {}
    model_scores: dict[str, float] = {}
    for config_id, expected_keys in expected_by_configuration.items():
        path = merged_paths[f"{config_id}__pass1.regraded.jsonl"]
        records, summary = _read_run(path)
        keyed = {logical_key(record): record for record in records}
        if len(records) != 280 or len(keyed) != 280 or set(keyed) != expected_keys:
            raise ValueError(f"merged artifact is not the exact 280-task configuration: {config_id}")
        if keyed != offline_by_configuration[config_id]:
            raise ValueError(f"merged artifact differs from its four validated shards: {config_id}")
        config_cost = 0.0
        for key, record in keyed.items():
            job = job_by_scope[(config_id, str(record.get("framework")))]
            _validate_offline_record(record, job, route_recoveries)
            candidate = candidate_records[config_id][key]
            if record.get("provider_response") != candidate.get("provider_response"):
                raise ValueError(f"offline grading changed provider provenance for {config_id} {key}")
            status = str(record.get("status"))
            status_counts[status] += 1
            passed += status == "passed"
            cost = _cost(record)
            config_cost += cost
            total_cost += cost
            token_covered += _token_usage_covered(record)
        _validate_merged_summary(summary, config_id)
        model_id = job_by_configuration[config_id].model_id
        model_costs[model_id] = config_cost
        model_scores[model_id] = _finite_rate((summary.get("summary") or {}).get("pass_rate"))
        merged_hashes[config_id] = _sha256(path)

    score_cost = json.loads((root / "score-cost.json").read_text(encoding="utf-8"))
    if not isinstance(score_cost, list) or len(score_cost) != expected_configurations:
        raise ValueError("score-cost.json must contain exactly one row per audited configuration")
    score_models: set[str] = set()
    for row in score_cost:
        if not isinstance(row, Mapping):
            raise ValueError("score-cost row is not an object")
        model = row.get("model")
        if not isinstance(model, str) or model in score_models or model not in model_costs:
            raise ValueError(f"score-cost contains an invalid model identity: {model!r}")
        score_models.add(model)
        if (
            row.get("protocol") != "pass1"
            or row.get("logical_tasks") != 280
            or row.get("provider_records") != 280
            or row.get("records_with_reported_cost") != 280
            or row.get("records_missing_reported_cost") != 0
            or row.get("reported_cost_coverage") != 1.0
            or not math.isclose(float(row.get("observed_cost_usd", -1)), model_costs[model], rel_tol=0, abs_tol=1e-8)
            or not math.isclose(float(row.get("score", -1)), model_scores[model], rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"score-cost row does not reproduce merged evidence: {model}")
    if score_models != set(model_costs):
        raise ValueError("score-cost omits one or more campaign models")
    if token_covered != expected_assignments:
        raise ValueError("offline records do not all retain prompt, completion, and total token usage")

    calibration_rows = [line.split("\t") for line in (root / "calibration" / "attempts.tsv").read_text().splitlines()]
    if (
        len(calibration_rows) != 3
        or {row[0] for row in calibration_rows if len(row) == 3} != {"2", "4", "8"}
        or not any(len(row) == 3 and row[2] == "passed" for row in calibration_rows)
    ):
        raise ValueError("offline worker calibration is incomplete")

    instance_ids = [line for line in (root / "instance-ids.txt").read_text().splitlines() if line]
    launch = json.loads((root / "aws-launch-readback.json").read_text(encoding="utf-8"))
    evaluation_attachment = json.loads((root / "evaluation-attachment-readback.json").read_text(encoding="utf-8"))
    terminated = json.loads((root / "termination-readback.json").read_text(encoding="utf-8"))
    if (
        len(instance_ids) != 6
        or len(set(instance_ids)) != 6
        or not isinstance(launch, list)
        or len(launch) != 6
        or {row.get("InstanceId") for row in launch if isinstance(row, Mapping)} != set(instance_ids)
        or len({row.get("ImageId") for row in launch if isinstance(row, Mapping)}) != 1
        or len({row.get("InstanceType") for row in launch if isinstance(row, Mapping)}) != 1
        or len({row.get("SubnetId") for row in launch if isinstance(row, Mapping)}) != 1
        or len({tuple(row.get("SecurityGroupIds") or []) for row in launch if isinstance(row, Mapping)}) != 1
        or any(len(row.get("SecurityGroupIds") or []) != 1 for row in launch if isinstance(row, Mapping))
        or any(row.get("State") not in {"pending", "running"} for row in launch if isinstance(row, Mapping))
        or not isinstance(evaluation_attachment, list)
        or len(evaluation_attachment) != 6
        or {row.get("InstanceId") for row in evaluation_attachment if isinstance(row, Mapping)} != set(instance_ids)
        or len({tuple(row.get("SecurityGroupIds") or []) for row in evaluation_attachment if isinstance(row, Mapping)})
        != 1
        or any(len(row.get("SecurityGroupIds") or []) != 1 for row in evaluation_attachment if isinstance(row, Mapping))
        or any(row.get("State") != "running" for row in evaluation_attachment if isinstance(row, Mapping))
        or not isinstance(terminated, list)
        or len(terminated) != 6
        or {row.get("InstanceId") for row in terminated if isinstance(row, Mapping)} != set(instance_ids)
        or {row.get("State") for row in terminated if isinstance(row, Mapping)} != {"terminated"}
    ):
        raise ValueError("AWS launch and termination readbacks do not prove one terminated six-worker pool")
    evaluation_groups = _json(root / "evaluation-security-group.json").get("SecurityGroups")
    if (
        not isinstance(evaluation_groups, list)
        or len(evaluation_groups) != 1
        or evaluation_groups[0].get("IpPermissionsEgress") != []
    ):
        raise ValueError("evaluation security-group snapshot does not prove zero egress")
    evaluation_group_id = evaluation_groups[0].get("GroupId")
    if {tuple(row.get("SecurityGroupIds") or []) for row in evaluation_attachment if isinstance(row, Mapping)} != {
        (evaluation_group_id,)
    }:
        raise ValueError("evaluation attachment readback does not use the snapshotted zero-egress group")

    return {
        "records": sum(len(records) for records in offline_by_configuration.values()),
        "unique_configuration_task_keys": sum(len(records) for records in offline_by_configuration.values()),
        "route_verified_records": expected_assignments,
        "cost_covered_records": expected_assignments,
        "token_usage_covered_records": token_covered,
        "passed_records": passed,
        "status_counts": dict(sorted(status_counts.items())),
        "provider_cost_usd": total_cost,
        "offline_shards": len(shard_paths),
        "merged_artifacts": len(merged_paths),
        "merged_sha256": merged_hashes,
        "instances": len(instance_ids),
        "all_instances_terminated": True,
        "evaluation_egress_rules": 0,
        "termination_readback_sha256": _sha256(root / "termination-readback.json"),
    }


def _validate_route(
    record: Mapping[str, Any],
    job: QueueJob,
    route_recoveries: Mapping[tuple[str, str, str, str, int, int], Mapping[str, Any]],
) -> None:
    response = record.get("provider_response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    route = metadata.get("route") if isinstance(metadata, Mapping) else None
    if not isinstance(route, Mapping):
        raise ValueError(f"{job.job_id}: record lacks route provenance")
    assert job.configuration_id is not None
    recovery_key = (job.configuration_id, *logical_key(record))
    recovery = route_recoveries.get(recovery_key)
    if recovery is not None:
        if not _route_matches_expectation(route, recovery):
            raise ValueError(f"{job.job_id}: recovered record route differs from its authorized replacement")
    else:
        expected_temperature: float | None = 0.0 if job.temperature_behavior == "explicit_zero" else None
        expected = {
            "configuration_id": job.configuration_id,
            "endpoint_tag": job.endpoint_tag,
            "max_output_tokens": job.configured_output_tokens,
            "output_limit_source": job.output_limit_source,
            "endpoint_cap_status": job.endpoint_cap_status,
            "output_token_parameter": job.output_token_parameter,
            "route_revision": job.route_revision,
            "temperature": expected_temperature,
            "route_verified": True,
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if not _route_matches_expectation(route, expected):
            raise ValueError(f"{job.job_id}: record route differs from the frozen queue")
    if record.get("model") != job.model_id or record.get("framework") != job.framework:
        raise ValueError(f"{job.job_id}: record model or framework identity is invalid")
    _cost(record)


def _validate_offline_record(
    record: Mapping[str, Any],
    job: QueueJob,
    route_recoveries: Mapping[tuple[str, str, str, str, int, int], Mapping[str, Any]],
) -> None:
    _validate_route(record, job, route_recoveries)
    if record.get("status") in {"generated", "infrastructure_error"}:
        raise ValueError(f"{job.job_id}: offline record is ungraded or infrastructure-failed")


def _validate_merged_summary(summary: Mapping[str, Any], config_id: str) -> None:
    if summary.get("configuration_id") != config_id:
        raise ValueError(f"merged summary has the wrong configuration identity: {config_id}")
    payload = summary.get("summary")
    if not isinstance(payload, Mapping) or payload.get("total_tasks") != 280:
        raise ValueError(f"merged summary does not report 280 tasks: {config_id}")
    by_suite = payload.get("by_suite")
    by_framework = payload.get("by_framework")
    if not isinstance(by_suite, Mapping) or not isinstance(by_framework, Mapping):
        raise ValueError(f"merged summary lacks suite/framework splits: {config_id}")
    for suite, expected in SUITE_COUNTS.items():
        row = by_suite.get(suite)
        if (
            not isinstance(row, Mapping)
            or row.get("assigned_tasks") != expected
            or row.get("infrastructure_failures") != 0
        ):
            raise ValueError(f"merged {suite} summary is incomplete: {config_id}")
    for framework in FRAMEWORKS:
        row = by_framework.get(framework)
        if not isinstance(row, Mapping) or row.get("assigned_tasks") != 70 or row.get("infrastructure_failures") != 0:
            raise ValueError(f"merged {framework} summary is incomplete: {config_id}")


def _job_by_configuration(jobs: Sequence[QueueJob]) -> dict[str, QueueJob]:
    result: dict[str, QueueJob] = {}
    for job in jobs:
        assert job.configuration_id is not None
        result.setdefault(job.configuration_id, job)
    return result


def _job_by_scope(jobs: Sequence[QueueJob]) -> dict[tuple[str, str], QueueJob]:
    result: dict[tuple[str, str], QueueJob] = {}
    for job in jobs:
        assert job.configuration_id is not None
        result[(job.configuration_id, job.framework)] = job
    return result


def _cost(record: Mapping[str, Any]) -> float:
    response = record.get("provider_response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    value = usage.get("cost_usd") if isinstance(usage, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise ValueError("record lacks finite non-negative provider-reported cost")
    return float(value)


def _finite_rate(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("summary lacks a finite numeric pass rate")
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("summary pass rate lies outside [0, 1]")
    return rate


def _token_usage_covered(record: Mapping[str, Any]) -> int:
    response = record.get("provider_response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping):
        return 0
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
    return 1


def _recovery_attempt_evidence(record: Mapping[str, Any]) -> tuple[int, float, int, str]:
    response = record.get("provider_response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    attempts = metadata.get("attempt_history") if isinstance(metadata, Mapping) else None
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("recovered record lacks physical-attempt evidence")
    accepted = [
        attempt
        for attempt in attempts
        if isinstance(attempt, Mapping) and attempt.get("status") == "accepted_model_outcome"
    ]
    if len(accepted) != 1 or attempts[-1] is not accepted[0]:
        raise ValueError("recovered record attempt history does not end in one accepted outcome")
    accepted_usage = accepted[0].get("usage")
    response_usage = response.get("usage") if isinstance(response, Mapping) else None
    generation_id = accepted[0].get("generation_id")
    if (
        accepted[0].get("route_verified") is not True
        or not isinstance(generation_id, str)
        or accepted_usage != response_usage
    ):
        raise ValueError("recovered record attempt history does not prove its accepted provider response")
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or attempt.get("status") not in {
            "accepted_model_outcome",
            "transient_infrastructure",
        }:
            raise ValueError("recovered record attempt history contains an unsupported outcome")
    transients = sum(attempt.get("status") == "transient_infrastructure" for attempt in attempts)
    return len(attempts), _cost(record), transients, generation_id


def _resolve_manifest_path(manifest: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest artifact path is missing")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else manifest.parent / candidate


def _verified_manifest_artifact(manifest: Path, raw: Any, label: str) -> tuple[Path, str]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"route recovery {label} artifact is not an object")
    path = _resolve_manifest_path(manifest, raw.get("path"))
    sha256 = raw.get("sha256")
    if not path.is_file() or not isinstance(sha256, str) or sha256 != _sha256(path):
        raise ValueError(f"route recovery {label} artifact is missing or hash-invalid")
    return path, sha256


def _route_expectation_from_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    temperature = selection.get("temperature")
    expected = {
        "configuration_id": selection.get("configuration_id"),
        "configuration_identity_sha256": selection.get("configuration_identity_sha256"),
        "endpoint_tag": selection.get("endpoint_tag"),
        "max_output_tokens": selection.get("configured_output_tokens"),
        "output_limit_source": selection.get("output_limit_source"),
        "endpoint_cap_status": selection.get("endpoint_cap_status"),
        "output_token_parameter": selection.get("output_token_parameter"),
        "route_revision": selection.get("route_revision"),
        "temperature": temperature,
        "route_verified": True,
        "allow_fallbacks": False,
        "require_parameters": True,
        "selected_provider": selection.get("provider"),
        "selected_model": selection.get("endpoint_served_model_id"),
    }
    if (
        selection.get("temperature_behavior") == "explicit_zero"
        and temperature != 0.0
        or any(expected[key] is None for key in expected if key != "temperature")
    ):
        raise ValueError("route recovery selection lacks required exact-route fields")
    return expected


def _route_matches_expectation(route: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(route.get(key) == value for key, value in expected.items())


def _route_recovery_key(config_id: str, raw: Any) -> tuple[str, str, str, str, int, int]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "suite",
        "framework",
        "task_id",
        "sample_index",
        "attempt_index",
    }:
        raise ValueError("route recovery assignment does not contain the exact logical-key fields")
    suite = raw.get("suite")
    framework = raw.get("framework")
    task_id = raw.get("task_id")
    sample_index = raw.get("sample_index")
    attempt_index = raw.get("attempt_index")
    if (
        not isinstance(suite, str)
        or not isinstance(framework, str)
        or not isinstance(task_id, str)
        or isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or isinstance(attempt_index, bool)
        or not isinstance(attempt_index, int)
    ):
        raise ValueError("route recovery assignment contains invalid logical-key values")
    return config_id, suite, framework, task_id, sample_index, attempt_index


def _read_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _jsonl_objects(path)
    results = [row for row in rows if row.get("kind") == "result"]
    summaries = [row for row in rows if row.get("kind") == "summary"]
    if len(summaries) != 1 or not rows or rows[-1] != summaries[0] or len(rows) != len(results) + 1:
        raise ValueError(f"run artifact must end in exactly one summary: {path}")
    return results, summaries[0]


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{number}: row is not an object")
        rows.append(payload)
    return rows


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _public_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"jobs", "expected_by_configuration", "records_by_configuration", "route_recoveries"}
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
