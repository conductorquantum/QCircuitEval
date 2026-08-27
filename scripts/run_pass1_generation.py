#!/usr/bin/env python3
"""Run the endpoint-pinned 2,520-request maximum-reasoning Pass@1 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from qceval.core.bench import SUPPORTED_FRAMEWORKS, Adaptor
from qceval.models import Framework, Suite
from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    BASE_MODEL_COUNT,
    BENCHMARK_CONTENT_COMMIT,
    CAMPAIGN_SCHEMA_VERSION,
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    FRESH_ASSIGNMENT_COUNT,
    HISTORICAL_IMPORT_ASSIGNMENT_COUNT,
    MAXIMUM_PROVIDER_REQUESTS,
    OUTPUT_POLICY_BY_MODEL,
    REQUESTS_PER_ENDPOINT,
    REUSABLE_CONFIGURATION_IDS,
    SHARD_COUNT,
)
from qceval.production.campaign import (
    configuration_id as expected_configuration_id,
)
from qceval.production.deferred import (
    DeferredInfrastructureStore,
    InfrastructureRetryPolicy,
)
from qceval.production.resume import LogicalKey, accepted_records, logical_key, pending_keys

_FRAMEWORK_ORDER: dict[str, int] = {framework: index for index, framework in enumerate(SUPPORTED_FRAMEWORKS)}
_OUTPUT_LIMIT_SOURCES = frozenset({"author_native", "benchmark_floor"})
_ENDPOINT_CAP_STATUSES = frozenset({"catalog_numeric", "undisclosed_first_party_exception"})
_OUTPUT_PARAMETERS = frozenset({"max_tokens", "max_completion_tokens"})
_TEMPERATURE_BEHAVIORS = frozenset({"explicit_zero", "not_exposed"})
_REASONING_SETTINGS = frozenset({"max", "xhigh", "high", "medium", "low", "minimal", "none", "enabled"})


@dataclass(frozen=True)
class QueueJob:
    """One fixed endpoint/framework generation shard."""

    job_id: str
    model_id: str
    reasoning_setting: str
    protocol: str
    framework: str
    suite: str
    max_tasks: int
    endpoint_tag: str
    configured_output_tokens: int
    output_limit_source: str
    endpoint_cap_status: str
    output_token_parameter: str
    route_revision: str
    temperature_behavior: str
    assigned_tasks: int
    configuration_id: str | None = None
    queue_schema_version: int = 2


@dataclass(frozen=True)
class SegmentScope:
    """Task subset assigned to one durable route segment."""

    suite: str
    task_numbers: tuple[int, ...] = ()


class CampaignLedger:
    """Thread-safe append-only campaign ledger and aggregate summary writer."""

    def __init__(self, out_dir: Path, *, expected_assignments: int) -> None:
        self.out_dir = out_dir
        self.expected_assignments = expected_assignments
        self.path = out_dir / "request-ledger.jsonl"
        self.attempt_path = out_dir / "provider-attempt-ledger.jsonl"
        self.summary_path = out_dir / "controller-summary.json"
        self._lock = threading.Lock()
        self._attempt_event_ids = self._load_attempt_event_ids()

    def append(self, event: Mapping[str, Any]) -> None:
        payload = {"created_at_utc": _now(), **event}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            summary = campaign_summary(self.out_dir, expected_assignments=self.expected_assignments)
            _atomic_write_json(self.summary_path, summary)
            print(json.dumps({"event": payload, "campaign": summary}, sort_keys=True), flush=True)

    def append_provider_attempts(self, *, job: QueueJob, segment: Path, record: Mapping[str, Any]) -> int:
        """Append each physical provider attempt once without changing the denominator."""
        key = logical_key(record)
        response = record.get("provider_response")
        metadata = response.get("metadata") if isinstance(response, Mapping) else None
        history = metadata.get("attempt_history") if isinstance(metadata, Mapping) else None
        if not isinstance(history, list):
            return 0
        appended = 0
        with self._lock, self.attempt_path.open("a", encoding="utf-8") as handle:
            for raw_attempt in history:
                if not isinstance(raw_attempt, Mapping):
                    raise ValueError("provider attempt history entries must be objects")
                attempt_number = raw_attempt.get("attempt_number")
                event_identity = {
                    "job_id": job.job_id,
                    "logical_key": list(key),
                    "segment": str(segment),
                    "attempt_number": attempt_number,
                }
                event_id = hashlib.sha256(
                    json.dumps(event_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if event_id in self._attempt_event_ids:
                    continue
                attempt = dict(raw_attempt)
                usage = attempt.get("usage")
                cost = usage.get("cost_usd") if isinstance(usage, Mapping) else None
                payload = {
                    "schema_version": "qceval.provider_attempt_ledger.v1",
                    "event_id": event_id,
                    "model_id": job.model_id,
                    "job_id": job.job_id,
                    "logical_key": list(key),
                    "endpoint_tag": job.endpoint_tag,
                    "route_revision": job.route_revision,
                    "configured_output_tokens": job.configured_output_tokens,
                    "output_token_parameter": job.output_token_parameter,
                    "reasoning_setting": job.reasoning_setting,
                    "configuration_id": job.configuration_id,
                    "temperature_behavior": job.temperature_behavior,
                    "segment": str(segment),
                    "provider_generation_id": attempt.get("generation_id"),
                    "provider_usage": usage,
                    "provider_cost_usd": cost,
                    **attempt,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                self._attempt_event_ids.add(event_id)
                appended += 1
            if appended:
                handle.flush()
                os.fsync(handle.fileno())
        return appended

    def _load_attempt_event_ids(self) -> set[str]:
        if not self.attempt_path.exists():
            return set()
        event_ids: set[str] = set()
        for number, line in enumerate(self.attempt_path.read_text(encoding="utf-8").splitlines(), start=1):
            payload = json.loads(line)
            event_id = payload.get("event_id") if isinstance(payload, Mapping) else None
            if not isinstance(event_id, str):
                raise ValueError(f"{self.attempt_path}:{number}: attempt event lacks event_id")
            if event_id in event_ids:
                raise ValueError(f"{self.attempt_path}:{number}: duplicate attempt event_id")
            event_ids.add(event_id)
        return event_ids


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - durable campaign orchestration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, help="Optional raw OpenRouter key file; otherwise use env/.env.")
    parser.add_argument("--source-hint", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--provider-timeout", type=float, default=600.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries after the initial request; production requires 5 (6 total attempts).",
    )
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--retry-max-delay", type=float, default=60.0)
    parser.add_argument(
        "--only-model",
        action="append",
        choices=tuple(EFFORTS_BY_MODEL),
        help="Run only the selected model lane while retaining the full frozen queue and manifest.",
    )
    parser.add_argument(
        "--skip-configuration",
        action="append",
        default=[],
        help="Skip a configuration supplied by an explicitly archived import while retaining the frozen queue.",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        jobs = read_queue(args.queue)
        if any(job.queue_schema_version != 2 for job in jobs):
            raise ValueError("the maximum-reasoning campaign refuses historical schema-v1 queue records")
        _validate_runtime_args(args)
        _validate_harness_commit(args.harness_commit)
        run_manifest = _validate_run_manifest(
            args.run_manifest,
            queue=args.queue,
            harness_commit=args.harness_commit,
        )
    except ValueError as exc:
        parser.error(str(exc))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_model_ids = frozenset(args.only_model or EFFORTS_BY_MODEL)
    known_configuration_ids = {job.configuration_id for job in jobs}
    skipped_configuration_ids = frozenset(args.skip_configuration)
    unknown_skips = skipped_configuration_ids - known_configuration_ids
    if unknown_skips:
        parser.error(f"--skip-configuration contains unknown IDs: {sorted(unknown_skips)}")
    historical_imports = run_manifest.get("historical_imports")
    declared_imports = (
        frozenset(historical_imports.get("configuration_ids") or ())
        if isinstance(historical_imports, Mapping)
        else frozenset()
    )
    undeclared_skips = skipped_configuration_ids - declared_imports
    if undeclared_skips:
        parser.error(
            "--skip-configuration requires matching historical import evidence in the run manifest: "
            f"{sorted(undeclared_skips)}"
        )
    plan = {
        "schema_version": "2",
        "created_at_utc": _now(),
        "queue": str(args.queue.resolve()),
        "queue_sha256": _sha256(args.queue),
        "harness_commit": args.harness_commit,
        "benchmark_content_commit": args.source_hint,
        "run_manifest": str(args.run_manifest.resolve()),
        "run_manifest_sha256": _sha256(args.run_manifest),
        "models": len({job.model_id for job in jobs}),
        "shards": len(jobs),
        "assignments": sum(job.assigned_tasks for job in jobs),
        "configurations": len({job.configuration_id for job in jobs}),
        "model_lanes": BASE_MODEL_COUNT,
        "selected_model_lanes": sorted(selected_model_ids),
        "skipped_imported_configurations": sorted(skipped_configuration_ids),
        "requests_per_endpoint": REQUESTS_PER_ENDPOINT,
        "maximum_simultaneous_provider_requests": MAXIMUM_PROVIDER_REQUESTS,
        "configuration_interleaving": "framework_rounds_with_rotated_configuration_order",
        "infrastructure_retry_policy": asdict(InfrastructureRetryPolicy()),
        "provider_max_retries_after_initial_attempt": args.max_retries,
        "generation_only": True,
        "jobs": [asdict(job) for job in jobs],
    }
    _atomic_write_json(args.out_dir / "generation-plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.api_key_file is not None and not args.api_key_file.is_file():
        parser.error("--api-key-file must point to a readable raw key file")
    if args.api_key_file is None and not _default_openrouter_credential_available():
        parser.error("OPENROUTER_API_KEY is missing from the environment and repository .env")
    segments = args.out_dir / "segments"
    logs = args.out_dir / "logs"
    cache = args.out_dir / "cache"
    for path in (segments, logs, cache):
        path.mkdir(parents=True, exist_ok=True)
    ledger = CampaignLedger(args.out_dir, expected_assignments=ASSIGNMENT_COUNT)
    recovery = DeferredInfrastructureStore(args.out_dir)
    by_model: dict[str, list[QueueJob]] = defaultdict(list)
    for job in jobs:
        if job.model_id in selected_model_ids and job.configuration_id not in skipped_configuration_ids:
            by_model[job.model_id].append(job)
    empty_selected_lanes = selected_model_ids - set(by_model)
    if empty_selected_lanes:
        parser.error(f"selected model lanes contain no executable jobs: {sorted(empty_selected_lanes)}")
    ledger.append(
        {
            "kind": "campaign_started",
            "models": BASE_MODEL_COUNT,
            "configurations": CONFIGURATION_COUNT,
            "shards": SHARD_COUNT,
            "assignments": ASSIGNMENT_COUNT,
            "selected_model_lanes": sorted(selected_model_ids),
            "skipped_imported_configurations": sorted(skipped_configuration_ids),
        }
    )
    recovery_deadline = datetime.now(UTC) + timedelta(hours=24)
    lane_results: dict[str, str] = {}
    recovery_round = 0
    while True:
        recovery_round += 1
        lane_results = _run_lane_round(
            by_model,
            args=args,
            segments_dir=segments,
            logs_dir=logs,
            cache_dir=cache,
            ledger=ledger,
            recovery=recovery,
        )
        if set(lane_results.values()) == {"complete"}:
            break
        if set(lane_results.values()) - {"complete", "deferred_infrastructure"}:
            break
        wait_seconds = _next_recovery_wait_seconds(recovery, deadline=recovery_deadline)
        if wait_seconds is None:
            break
        ledger.append(
            {
                "kind": "deferred_recovery_wait",
                "round": recovery_round,
                "wait_seconds": wait_seconds,
                "absolute_deadline_utc": recovery_deadline.isoformat().replace("+00:00", "Z"),
            }
        )
        time.sleep(wait_seconds)
    ledger.append({"kind": "campaign_finished", "model_lane_status": dict(sorted(lane_results.items()))})
    final = campaign_summary(args.out_dir, expected_assignments=ASSIGNMENT_COUNT)
    final["model_lane_status"] = dict(sorted(lane_results.items()))
    scoped_execution = selected_model_ids != frozenset(EFFORTS_BY_MODEL) or bool(skipped_configuration_ids)
    if scoped_execution and set(lane_results.values()) == {"complete"}:
        final["status"] = "selected_model_lanes_complete"
        exit_code = 0
    elif final["accepted_logical_requests"] == ASSIGNMENT_COUNT and set(lane_results.values()) == {"complete"}:
        final["status"] = "generation_complete"
        exit_code = 0
    elif {"deferred_infrastructure", "paused_infrastructure"} & set(lane_results.values()):
        if datetime.now(UTC) >= recovery_deadline:
            final["status"] = "model_blocked_after_24h"
            final["comparison_withheld"] = True
        else:
            final["status"] = "deferred_infrastructure"
        exit_code = 2
    else:
        final["status"] = "runner_error"
        exit_code = 1
    final["finished_at_utc"] = _now()
    _atomic_write_json(ledger.summary_path, final)
    return exit_code


def _run_lane_round(
    by_model: Mapping[str, list[QueueJob]],
    *,
    args: argparse.Namespace,
    segments_dir: Path,
    logs_dir: Path,
    cache_dir: Path,
    ledger: CampaignLedger,
    recovery: DeferredInfrastructureStore,
) -> dict[str, str]:
    results: dict[str, str] = {}
    lane_workers = max(1, MAXIMUM_PROVIDER_REQUESTS // REQUESTS_PER_ENDPOINT)
    with ThreadPoolExecutor(max_workers=lane_workers, thread_name_prefix="model-lane") as pool:
        futures = {
            pool.submit(
                _run_model_lane,
                model_jobs,
                args=args,
                segments_dir=segments_dir,
                logs_dir=logs_dir,
                cache_dir=cache_dir,
                ledger=ledger,
                recovery=recovery,
            ): model_id
            for model_id, model_jobs in sorted(by_model.items())
        }
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                results[model_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve other independent model lanes.
                results[model_id] = "runner_error"
                ledger.append(
                    {
                        "kind": "model_lane_failed",
                        "model_id": model_id,
                        "classification": "runner_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return results


def _next_recovery_wait_seconds(
    recovery: DeferredInfrastructureStore,
    *,
    deadline: datetime,
) -> float | None:
    now = datetime.now(UTC)
    if now >= deadline:
        return None
    snapshot = recovery.snapshot()
    eligible_times = []
    for entry in snapshot.get("requests", {}).values():
        if not isinstance(entry, Mapping) or entry.get("status") != "deferred_infrastructure":
            continue
        raw = entry.get("next_eligible_retry_at_utc")
        if isinstance(raw, str):
            eligible_times.append(datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC))
    if not eligible_times:
        return None
    next_retry = min(eligible_times)
    wake = min(next_retry, deadline)
    return max(0.0, (wake - now).total_seconds())


def read_queue(path: Path, *, validate_campaign: bool = True) -> list[QueueJob]:
    """Parse a queue, optionally enforcing the full frozen campaign cardinality."""
    jobs: list[QueueJob] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split("\t")
        if len(fields) not in {15, 16}:
            raise ValueError(f"queue line {line_number} must contain exactly fifteen or sixteen columns")
        schema_version = 2 if len(fields) == 16 else 1
        try:
            job = QueueJob(
                job_id=fields[0],
                model_id=fields[1],
                reasoning_setting=fields[2],
                protocol=fields[3],
                framework=fields[4],
                suite=fields[5],
                max_tasks=int(fields[6]),
                endpoint_tag=fields[7],
                configured_output_tokens=int(fields[8]),
                output_limit_source=fields[9],
                endpoint_cap_status=fields[10],
                output_token_parameter=fields[11],
                route_revision=fields[12],
                temperature_behavior=fields[13],
                assigned_tasks=int(fields[14]),
                configuration_id=fields[15] if schema_version == 2 else None,
                queue_schema_version=schema_version,
            )
        except ValueError as exc:
            raise ValueError(f"queue line {line_number} contains a nonnumeric count") from exc
        _validate_job(job, line_number=line_number)
        jobs.append(job)
    if validate_campaign:
        _validate_campaign(jobs)
    return jobs


def build_command(
    job: QueueJob,
    *,
    output: Path,
    api_key_file: Path | None,
    source_hint: str,
    cache_dir: Path,
    scope: SegmentScope,
    provider_timeout: float = 600.0,
    max_retries: int = 5,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 60.0,
) -> list[str]:
    """Build one exact-route generation-only shard command."""
    executable = Path(sys.executable).with_name("qceval")
    command = [
        str(executable),
        "run",
        "--provider",
        "openrouter",
        "--model",
        job.model_id,
        "--framework",
        job.framework,
        "--suite",
        scope.suite,
        "--source-hint",
        source_hint,
        "--out",
        str(output),
        "--output-format",
        "jsonl",
        "--timeout",
        str(provider_timeout),
        "--max-retries",
        str(max_retries),
        "--retry-base-delay",
        str(retry_base_delay),
        "--retry-max-delay",
        str(retry_max_delay),
        "--generation-concurrency",
        "2",
        "--evaluation-workers",
        "1",
        "--samples-per-task",
        "1",
        "--pass-k",
        "1",
        "--max-attempts",
        "1",
        "--cache-dir",
        str(cache_dir),
        "--rerun",
        job.framework,
        "--openrouter-endpoint-tag",
        job.endpoint_tag,
        "--openrouter-max-output-tokens",
        str(job.configured_output_tokens),
        "--openrouter-output-limit-source",
        job.output_limit_source,
        "--openrouter-endpoint-cap-status",
        job.endpoint_cap_status,
        "--openrouter-output-token-parameter",
        job.output_token_parameter,
        "--openrouter-route-revision",
        job.route_revision,
        "--configuration-id",
        str(job.configuration_id),
        "--stop-on-infrastructure-error",
        "--progress",
    ]
    if api_key_file is not None:
        command[4:4] = ["--openrouter-api-key-file", str(api_key_file)]
    if scope.task_numbers:
        command.extend(["--tasks", *(str(number) for number in scope.task_numbers)])
    if job.reasoning_setting == "enabled":
        command.append("--reasoning-enabled")
    else:
        command.extend(["--reasoning-effort", job.reasoning_setting])
    if job.temperature_behavior == "explicit_zero":
        command.extend(["--temperature", "0.0"])
    return command


def campaign_summary(  # noqa: C901 - explicit durable-ledger accounting
    out_dir: Path, *, expected_assignments: int
) -> dict[str, Any]:
    """Aggregate durable route segments without treating retries as assignments."""
    paths = sorted((out_dir / "segments").glob("*/*.jsonl")) if (out_dir / "segments").exists() else []
    physical_records = 0
    infrastructure_records = 0
    provider_cost = 0.0
    cost_records = 0
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}
    for payload in _result_payloads(paths):
        physical_records += 1
        if payload.get("status") == "infrastructure_error":
            infrastructure_records += 1
        usage = (payload.get("provider_response") or {}).get("usage") or {}
        cost = usage.get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            provider_cost += float(cost)
            cost_records += 1
        for source, target in (
            ("prompt_tokens", "prompt"),
            ("completion_tokens", "completion"),
            ("reasoning_tokens", "reasoning"),
            ("total_tokens", "total"),
        ):
            value = usage.get(source)
            if isinstance(value, int) and not isinstance(value, bool):
                tokens[target] += value
    accepted_count = 0
    accepted_scan_errors: dict[str, str] = {}
    job_dirs = sorted(path for path in (out_dir / "segments").glob("*") if path.is_dir())
    for job_dir in job_dirs:
        try:
            accepted_count += len(accepted_records(sorted(job_dir.glob("*.jsonl"))))
        except (OSError, ValueError) as exc:
            # This aggregate is refreshed while independent qceval processes
            # append JSONL. A summary read may briefly observe an incomplete
            # final line; final shard validation still uses pending_keys after
            # the writer exits and remains strict.
            accepted_scan_errors[job_dir.name] = str(exc)
    projected = None if accepted_count == 0 else provider_cost / accepted_count * expected_assignments
    deferred_requests = 0
    open_endpoint_circuits = 0
    recovery_path = out_dir / "deferred-infrastructure-state.json"
    if recovery_path.exists():
        try:
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            deferred_requests = sum(
                entry.get("status") == "deferred_infrastructure"
                for entry in recovery.get("requests", {}).values()
                if isinstance(entry, Mapping)
            )
            open_endpoint_circuits = sum(
                circuit.get("status") == "open"
                for circuit in recovery.get("circuits", {}).values()
                if isinstance(circuit, Mapping)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # The controller's durable store performs strict recovery reads;
            # this live aggregate stays available during an atomic replace.
            pass
    attempt_ledger_path = out_dir / "provider-attempt-ledger.jsonl"
    physical_provider_attempts = (
        len(attempt_ledger_path.read_text(encoding="utf-8").splitlines()) if attempt_ledger_path.exists() else 0
    )
    return {
        "schema_version": "1",
        "updated_at_utc": _now(),
        "status": "running",
        "expected_logical_requests": expected_assignments,
        "accepted_logical_requests": accepted_count,
        "physical_records": physical_records,
        "infrastructure_records": infrastructure_records,
        "deferred_infrastructure_requests": deferred_requests,
        "open_endpoint_circuits": open_endpoint_circuits,
        "physical_provider_attempts": physical_provider_attempts,
        "provider_cost_usd": provider_cost,
        "records_with_reported_cost": cost_records,
        "projected_provider_cost_usd": projected,
        "tokens": tokens,
        "accepted_scan_errors": accepted_scan_errors,
    }


def _run_model_lane(
    jobs: list[QueueJob],
    *,
    args: argparse.Namespace,
    segments_dir: Path,
    logs_dir: Path,
    cache_dir: Path,
    ledger: CampaignLedger,
    recovery: DeferredInfrastructureStore | None = None,
) -> str:
    if recovery is None:
        recovery = DeferredInfrastructureStore(ledger.out_dir)
    ordered = _interleaved_model_jobs(jobs)
    model_id = ordered[0].model_id
    ledger.append(
        {
            "kind": "model_lane_started",
            "model_id": model_id,
            "configuration_order": [job.configuration_id for job in ordered],
        }
    )
    has_deferred = False
    for job in ordered:
        status = _run_job(
            job,
            args=args,
            segments_dir=segments_dir,
            logs_dir=logs_dir,
            cache_dir=cache_dir,
            ledger=ledger,
            recovery=recovery,
        )
        if status == "deferred_infrastructure" and not recovery.circuit_is_open(
            model_id=job.model_id,
            endpoint_tag=job.endpoint_tag,
            route_revision=job.route_revision,
        ):
            has_deferred = True
            ledger.append(
                {
                    "kind": "shard_deferred_lane_continues",
                    "model_id": model_id,
                    "configuration_id": job.configuration_id,
                    "job_id": job.job_id,
                }
            )
            continue
        if status != "complete":
            ledger.append({"kind": "model_lane_paused", "model_id": model_id, "classification": status})
            return status
    if has_deferred:
        ledger.append({"kind": "model_lane_deferred", "model_id": model_id})
        return "deferred_infrastructure"
    ledger.append({"kind": "model_lane_completed", "model_id": model_id})
    return "complete"


def _interleaved_model_jobs(jobs: Sequence[QueueJob]) -> list[QueueJob]:
    """Return deterministic framework rounds with rotated configurations.

    Every configured setting appears once before the next framework begins.
    Rotating the starting setting prevents the same configuration from always
    occupying the earliest temporal position.
    """
    if not jobs:
        return []
    model_id = jobs[0].model_id
    official_efforts = EFFORTS_BY_MODEL.get(model_id)
    if official_efforts is None or any(job.model_id != model_id for job in jobs):
        raise ValueError("model lanes must contain one known base model")
    present_efforts = {job.reasoning_setting for job in jobs}
    efforts = tuple(effort for effort in official_efforts if effort in present_efforts)
    if not efforts:
        raise ValueError(f"{model_id}: model lane has no executable effort")
    by_pair = {(job.framework, job.reasoning_setting): job for job in jobs}
    ordered: list[QueueJob] = []
    for framework_index, framework in enumerate(SUPPORTED_FRAMEWORKS):
        rotation = framework_index % len(efforts)
        rotated = efforts[rotation:] + efforts[:rotation]
        ordered.extend(by_pair[(framework, effort)] for effort in rotated)
    if len(ordered) != len(jobs) or len(set(ordered)) != len(jobs):
        raise ValueError(f"{model_id}: model lane cannot be deterministically interleaved")
    return ordered


def _run_job(  # noqa: C901 - bounded retry and circuit-breaker state machine
    job: QueueJob,
    *,
    args: argparse.Namespace,
    segments_dir: Path,
    logs_dir: Path,
    cache_dir: Path,
    ledger: CampaignLedger,
    recovery: DeferredInfrastructureStore | None = None,
) -> str:
    if recovery is None:
        recovery = DeferredInfrastructureStore(ledger.out_dir)
    job_dir = segments_dir / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    assignments = _assignments(job)
    swept: set[LogicalKey] = set()
    started_incomplete = bool(pending_keys(assignments, sorted(job_dir.glob("*.jsonl"))))
    while True:
        existing = sorted(job_dir.glob("*.jsonl"))
        accepted = accepted_records(existing, strict_provenance=True)
        _validate_accepted_configuration(job, accepted.values())
        recovery.reconcile_accepted(
            model_id=job.model_id,
            job_id=job.job_id,
            endpoint_tag=job.endpoint_tag,
            route_revision=job.route_revision,
            accepted=accepted,
            configuration_id=job.configuration_id,
        )
        pending = pending_keys(assignments, existing)
        if not pending:
            kind = "shard_completed" if started_incomplete else "shard_already_complete"
            ledger.append(
                {
                    "kind": kind,
                    "job_id": job.job_id,
                    "configuration_id": job.configuration_id,
                    "records": len(assignments),
                }
            )
            return "complete"

        deferred = recovery.deferred_keys(model_id=job.model_id, job_id=job.job_id)
        normal_pending = [key for key in pending if key not in deferred]
        circuit_open = recovery.circuit_is_open(
            model_id=job.model_id,
            endpoint_tag=job.endpoint_tag,
            route_revision=job.route_revision,
        )
        eligible = recovery.eligible_deferred_keys(
            model_id=job.model_id,
            job_id=job.job_id,
            endpoint_tag=job.endpoint_tag,
            route_revision=job.route_revision,
        )
        eligible_pending = [key for key in assignments if key in pending and key in eligible and key not in swept]

        if circuit_open:
            if not eligible_pending:
                ledger.append(
                    {
                        "kind": "endpoint_circuit_waiting",
                        "job_id": job.job_id,
                        "model_id": job.model_id,
                        "configuration_id": job.configuration_id,
                        "endpoint_tag": job.endpoint_tag,
                        "route_revision": job.route_revision,
                        "cooldown_until_utc": recovery.cooldown_until(
                            model_id=job.model_id,
                            endpoint_tag=job.endpoint_tag,
                            route_revision=job.route_revision,
                        ),
                        "deferred_requests": len(deferred),
                    }
                )
                return "deferred_infrastructure"
            selected = eligible_pending[:1]
            from_deferred_sweep = True
        elif normal_pending:
            selected = normal_pending
            from_deferred_sweep = False
        elif eligible_pending:
            # A sweep is bounded to the endpoint lane width. A controller restart
            # can revisit still-deferred work without duplicating accepted keys.
            selected = eligible_pending[:2]
            from_deferred_sweep = True
        else:
            ledger.append(
                {
                    "kind": "deferred_sweep_pending",
                    "job_id": job.job_id,
                    "model_id": job.model_id,
                    "configuration_id": job.configuration_id,
                    "endpoint_tag": job.endpoint_tag,
                    "route_revision": job.route_revision,
                    "deferred_requests": len(deferred),
                    "cooldown_until_utc": recovery.cooldown_until(
                        model_id=job.model_id,
                        endpoint_tag=job.endpoint_tag,
                        route_revision=job.route_revision,
                    ),
                }
            )
            return "deferred_infrastructure"

        scopes = _segment_scopes(assignments, selected, existing=bool(existing))
        if not scopes:
            raise ValueError(f"{job.job_id}: selected pending work did not produce a segment scope")
        scope = scopes[0]
        selected_scope_keys = [key for key in selected if key[0] == scope.suite or scope.suite == "all"]
        sequence = len(list(job_dir.glob("*.jsonl"))) + 1
        segment = job_dir / f"{job.route_revision}-s{sequence:03d}-{scope.suite}.jsonl"
        log_path = logs_dir / f"{job.job_id}-s{sequence:03d}-{scope.suite}.log"
        command = build_command(
            job,
            output=segment,
            api_key_file=None if args.api_key_file is None else args.api_key_file.resolve(),
            source_hint=args.source_hint,
            cache_dir=cache_dir,
            scope=scope,
            provider_timeout=args.provider_timeout,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            retry_max_delay=args.retry_max_delay,
        )
        ledger.append(
            {
                "kind": "segment_started",
                "job_id": job.job_id,
                "model_id": job.model_id,
                "configuration_id": job.configuration_id,
                "framework": job.framework,
                "route_revision": job.route_revision,
                "scope": asdict(scope),
                "from_deferred_sweep": from_deferred_sweep,
                "segment": str(segment),
                "command": command,
            }
        )
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=Path.cwd(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
            log.flush()
            os.fsync(log.fileno())
        records = list(_result_payloads([segment])) if segment.exists() else []
        infrastructure = sum(record.get("status") == "infrastructure_error" for record in records)
        physical_attempts = sum(
            ledger.append_provider_attempts(job=job, segment=segment, record=record) for record in records
        )
        ledger.append(
            {
                "kind": "segment_finished",
                "job_id": job.job_id,
                "configuration_id": job.configuration_id,
                "segment": str(segment),
                "exit_code": completed.returncode,
                "records": len(records),
                "infrastructure_records": infrastructure,
                "physical_provider_attempts": physical_attempts,
                "from_deferred_sweep": from_deferred_sweep,
                "log": str(log_path),
            }
        )
        if completed.returncode != 0:
            return "runner_error"
        if not records:
            ledger.append(
                {
                    "kind": "segment_empty",
                    "job_id": job.job_id,
                    "configuration_id": job.configuration_id,
                    "segment": str(segment),
                }
            )
            return "runner_error"

        accepted_before = set(accepted)
        accepted_in_segment = accepted_records([segment], strict_provenance=True)
        _validate_accepted_configuration(job, accepted_in_segment.values())
        duplicates = sorted(accepted_before & set(accepted_in_segment))
        if duplicates:
            raise ValueError(f"{job.job_id}: segment regenerated accepted logical keys: {duplicates[:3]}")

        nonretryable_infrastructure = False
        circuit_opened = False
        for record in records:
            key = logical_key(record)
            if key not in assignments:
                raise ValueError(f"{job.job_id}: segment contains foreign logical key {key}")
            if record.get("status") != "infrastructure_error":
                recovery.record_accepted(
                    model_id=job.model_id,
                    job_id=job.job_id,
                    endpoint_tag=job.endpoint_tag,
                    route_revision=job.route_revision,
                    key=key,
                    record=record,
                    from_deferred_sweep=from_deferred_sweep,
                    configuration_id=job.configuration_id,
                )
                continue

            metadata = _provider_metadata(record)
            history = metadata.get("attempt_history")
            attempts = metadata.get("infrastructure_attempts")
            retryable = metadata.get("retryable_infrastructure") is True
            exhausted = metadata.get("retry_exhausted") is True
            if retryable and exhausted and attempts == 6 and isinstance(history, list) and len(history) == 6:
                circuit_opened = (
                    recovery.defer_exhausted(
                        model_id=job.model_id,
                        job_id=job.job_id,
                        endpoint_tag=job.endpoint_tag,
                        route_revision=job.route_revision,
                        key=key,
                        error_history=[item for item in history if isinstance(item, Mapping)],
                        attempt_count=attempts,
                        segment=segment,
                        configuration_id=job.configuration_id,
                    )
                    or circuit_opened
                )
            else:
                nonretryable_infrastructure = True
                ledger.append(
                    {
                        "kind": "nonretryable_infrastructure",
                        "job_id": job.job_id,
                        "model_id": job.model_id,
                        "configuration_id": job.configuration_id,
                        "logical_key": list(key),
                        "endpoint_tag": job.endpoint_tag,
                        "route_revision": job.route_revision,
                        "failure_classification": metadata.get("failure_classification"),
                        "retryable_infrastructure": retryable,
                        "retry_exhausted": exhausted,
                        "infrastructure_attempts": attempts,
                        "segment": str(segment),
                    }
                )

        if from_deferred_sweep:
            swept.update(selected_scope_keys)
        if nonretryable_infrastructure:
            return "paused_infrastructure"
        if circuit_opened:
            ledger.append(
                {
                    "kind": "endpoint_circuit_opened",
                    "job_id": job.job_id,
                    "model_id": job.model_id,
                    "configuration_id": job.configuration_id,
                    "endpoint_tag": job.endpoint_tag,
                    "route_revision": job.route_revision,
                    "cooldown_until_utc": recovery.cooldown_until(
                        model_id=job.model_id,
                        endpoint_tag=job.endpoint_tag,
                        route_revision=job.route_revision,
                    ),
                }
            )
            return "deferred_infrastructure"


def _assignments(job: QueueJob) -> list[LogicalKey]:
    adapter = Adaptor()
    assignments: list[LogicalKey] = []
    for suite in ("core", "qec"):
        for task in adapter.load_tasks(cast(Framework, job.framework), cast(Suite, suite)):
            assignments.append((suite, job.framework, task.task_id, 0, 0))
    if len(assignments) != job.assigned_tasks:
        raise ValueError(f"{job.job_id}: assignment expansion mismatch")
    return assignments


def _segment_scopes(assignments: list[LogicalKey], pending: list[LogicalKey], *, existing: bool) -> list[SegmentScope]:
    if not existing and pending == assignments:
        return [SegmentScope(suite="all")]
    scopes: list[SegmentScope] = []
    for suite in ("core", "qec"):
        numbers = tuple(_suite_local_task_number(suite, key[2]) for key in pending if key[0] == suite)
        if numbers:
            scopes.append(SegmentScope(suite=suite, task_numbers=numbers))
    return scopes


def _suite_local_task_number(suite: str, task_id: str) -> int:
    """Convert a durable task ID to the numeric selector accepted by the CLI."""
    numeric = task_id.removeprefix("qec") if suite == "qec" else task_id
    if not numeric.isdigit():
        raise ValueError(f"invalid {suite} task ID in resume state: {task_id!r}")
    return int(numeric)


def _result_payloads(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("kind") == "result":
                records.append(payload)
    return records


def _provider_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    response = record.get("provider_response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    return metadata if isinstance(metadata, Mapping) else {}


def _validate_accepted_configuration(job: QueueJob, records: Iterable[Mapping[str, Any]]) -> None:
    """Reject accepted records whose route belongs to another effort config."""
    expected_temperature = 0.0 if job.temperature_behavior == "explicit_zero" else None
    for record in records:
        metadata = _provider_metadata(record)
        route = metadata.get("route")
        identity = (
            (
                route.get("configuration_id"),
                route.get("endpoint_tag"),
                route.get("max_output_tokens"),
                route.get("output_limit_source"),
                route.get("endpoint_cap_status"),
                route.get("output_token_parameter"),
                route.get("route_revision"),
                route.get("temperature"),
            )
            if isinstance(route, Mapping)
            else None
        )
        expected = (
            job.configuration_id,
            job.endpoint_tag,
            job.configured_output_tokens,
            job.output_limit_source,
            job.endpoint_cap_status,
            job.output_token_parameter,
            job.route_revision,
            expected_temperature,
        )
        if identity != expected:
            raise ValueError(f"{job.job_id}: accepted record has incompatible configuration provenance")


def _validate_job(job: QueueJob, *, line_number: int) -> None:  # noqa: C901 - frozen queue contract
    if not all((job.job_id, job.model_id, job.endpoint_tag, job.route_revision)):
        raise ValueError(f"queue line {line_number} contains an empty identity field")
    if job.reasoning_setting not in _REASONING_SETTINGS:
        raise ValueError(f"queue line {line_number} has an invalid reasoning setting")
    if job.protocol != "pass1" or job.suite != "all" or job.max_tasks != 0:
        raise ValueError(f"queue line {line_number} is not a full Pass@1 shard")
    if job.framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"queue line {line_number} has an invalid framework")
    if job.configured_output_tokens < 1 or job.assigned_tasks != 70:
        raise ValueError(f"queue line {line_number} has an invalid assignment or output count")
    if job.output_limit_source not in _OUTPUT_LIMIT_SOURCES:
        raise ValueError(f"queue line {line_number} has an invalid output-limit source")
    if job.endpoint_cap_status not in _ENDPOINT_CAP_STATUSES:
        raise ValueError(f"queue line {line_number} has an invalid endpoint-cap status")
    if (
        job.queue_schema_version == 1
        and job.endpoint_cap_status == "undisclosed_first_party_exception"
        and (
            job.model_id != "x-ai/grok-4.6"
            or job.endpoint_tag != "xai"
            or job.output_limit_source != "benchmark_floor"
            or job.configured_output_tokens != 128000
        )
    ):
        raise ValueError(f"queue line {line_number} applies the Grok exception outside its frozen scope")
    if job.model_id == "z-ai/glm-5.2" and (
        job.reasoning_setting != "max"
        or job.configured_output_tokens != 131072
        or job.output_limit_source != "author_native"
        or job.endpoint_cap_status != "catalog_numeric"
        or job.output_token_parameter != "max_tokens"
    ):
        raise ValueError(f"queue line {line_number} violates the frozen GLM-5.2 max_tokens=131072 request contract")
    if job.output_token_parameter not in _OUTPUT_PARAMETERS:
        raise ValueError(f"queue line {line_number} has an invalid output parameter")
    if job.temperature_behavior not in _TEMPERATURE_BEHAVIORS:
        raise ValueError(f"queue line {line_number} has an invalid temperature behavior")
    if job.queue_schema_version == 1:
        if job.configuration_id is not None:
            raise ValueError(f"queue line {line_number} mixes schema-v1 and schema-v2 identity")
        return
    if job.configuration_id != expected_configuration_id(job.model_id, job.reasoning_setting):
        raise ValueError(f"queue line {line_number} has a malformed configuration_id")
    if job.model_id not in EFFORTS_BY_MODEL or job.reasoning_setting not in EFFORTS_BY_MODEL[job.model_id]:
        raise ValueError(f"queue line {line_number} is outside the frozen reasoning settings")
    expected_output_policy = OUTPUT_POLICY_BY_MODEL[job.model_id]
    actual_output_policy = (
        job.configured_output_tokens,
        job.output_limit_source,
        job.endpoint_cap_status,
    )
    if actual_output_policy != expected_output_policy:
        raise ValueError(f"queue line {line_number} violates the frozen model output policy")


def _validate_campaign(jobs: list[QueueJob]) -> None:  # noqa: C901 - frozen campaign cardinality contract
    schemas = {job.queue_schema_version for job in jobs}
    if len(schemas) != 1:
        raise ValueError("queue cannot mix schema-v1 and schema-v2 records")
    if schemas == {1}:
        _validate_legacy_campaign(jobs)
        return
    if len(jobs) != SHARD_COUNT or sum(job.assigned_tasks for job in jobs) != ASSIGNMENT_COUNT:
        raise ValueError(f"official queue must contain {SHARD_COUNT} shards and {ASSIGNMENT_COUNT} assignments")
    if len({job.job_id for job in jobs}) != SHARD_COUNT:
        raise ValueError("official queue contains duplicate job IDs")
    by_model: dict[str, list[QueueJob]] = defaultdict(list)
    by_configuration: dict[str, list[QueueJob]] = defaultdict(list)
    for job in jobs:
        by_model[job.model_id].append(job)
        assert job.configuration_id is not None
        by_configuration[job.configuration_id].append(job)
    if set(by_model) != set(EFFORTS_BY_MODEL):
        raise ValueError("official queue must contain exactly the nine frozen base models")
    if len(by_configuration) != CONFIGURATION_COUNT:
        raise ValueError(f"official queue must contain exactly {CONFIGURATION_COUNT} configurations")
    for config_id, config_jobs in by_configuration.items():
        if {job.framework for job in config_jobs} != set(SUPPORTED_FRAMEWORKS) or len(config_jobs) != 4:
            raise ValueError(f"{config_id}: configuration must contain one shard per framework")
    for model_id, model_jobs in by_model.items():
        expected_efforts = set(EFFORTS_BY_MODEL[model_id])
        if {job.reasoning_setting for job in model_jobs} != expected_efforts:
            raise ValueError(f"{model_id}: model lane does not contain its frozen reasoning setting")
        route_signatures = {
            (
                job.endpoint_tag,
                job.configured_output_tokens,
                job.output_limit_source,
                job.endpoint_cap_status,
                job.output_token_parameter,
                job.route_revision,
                job.temperature_behavior,
            )
            for job in model_jobs
        }
        if len(route_signatures) != 1:
            raise ValueError(f"{model_id}: all configurations must share one exact endpoint route")


def _validate_legacy_campaign(jobs: list[QueueJob]) -> None:
    """Keep the historical fifteen-column campaign readable, never runnable as v2."""
    if len(jobs) != 36 or sum(job.assigned_tasks for job in jobs) != 2520:
        raise ValueError("historical queue must contain 36 shards and 2520 assignments")
    if len({job.job_id for job in jobs}) != 36 or len({job.model_id for job in jobs}) != 9:
        raise ValueError("historical queue must contain 36 unique shards across nine models")
    for model_id in {job.model_id for job in jobs}:
        model_jobs = [job for job in jobs if job.model_id == model_id]
        if {job.framework for job in model_jobs} != set(SUPPORTED_FRAMEWORKS) or len(model_jobs) != 4:
            raise ValueError(f"{model_id}: historical model lane must contain one shard per framework")


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.provider_timeout <= 0:
        raise ValueError("--provider-timeout must be positive")
    if args.max_retries != 5:
        raise ValueError("production Pass@1 requires --max-retries 5 (six total infrastructure attempts)")
    if args.retry_base_delay < 0 or args.retry_max_delay <= 0:
        raise ValueError("invalid retry configuration")
    if args.source_hint != BENCHMARK_CONTENT_COMMIT:
        raise ValueError(f"production source hint must be exactly {BENCHMARK_CONTENT_COMMIT}")


def _default_openrouter_credential_available() -> bool:
    value = os.environ.get("OPENROUTER_API_KEY")
    if value and value.strip():
        return True
    path = Path.cwd() / ".env"
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == "OPENROUTER_API_KEY" and value.strip():
            return True
    return False


def _validate_harness_commit(expected: str) -> None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    actual = completed.stdout.strip()
    if actual != expected:
        raise ValueError(f"harness commit mismatch: expected {expected}, found {actual}")
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, check=False).returncode != 0:
            raise ValueError("tracked worktree must be clean before production generation")


def _validate_run_manifest(path: Path, *, queue: Path, harness_commit: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError("--run-manifest must point to the frozen campaign manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("run manifest has an unsupported schema")
    if payload.get("harness_commit") != harness_commit:
        raise ValueError("run manifest harness commit does not match --harness-commit")
    if payload.get("benchmark_content_commit") != BENCHMARK_CONTENT_COMMIT:
        raise ValueError("run manifest benchmark content commit is invalid")
    historical_imports = payload.get("historical_imports")
    if (
        payload.get("fresh_logical_requests") != FRESH_ASSIGNMENT_COUNT
        or payload.get("historical_imported_requests") != HISTORICAL_IMPORT_ASSIGNMENT_COUNT
        or not isinstance(historical_imports, Mapping)
        or historical_imports.get("configuration_ids") != sorted(REUSABLE_CONFIGURATION_IDS)
        or historical_imports.get("records") != HISTORICAL_IMPORT_ASSIGNMENT_COUNT
    ):
        raise ValueError(
            "run manifest does not bind the exact reusable configuration set required for fresh generation"
        )
    cardinality = (
        payload.get("base_models"),
        payload.get("configurations"),
        payload.get("shards"),
        payload.get("logical_requests"),
    )
    if cardinality != (BASE_MODEL_COUNT, CONFIGURATION_COUNT, SHARD_COUNT, ASSIGNMENT_COUNT):
        raise ValueError("run manifest campaign cardinality is invalid")
    artifacts = payload.get("artifacts")
    queue_artifact = artifacts.get("queue") if isinstance(artifacts, Mapping) else None
    if not isinstance(queue_artifact, Mapping) or queue_artifact.get("sha256") != _sha256(queue):
        raise ValueError("run manifest queue hash does not match the supplied queue")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
