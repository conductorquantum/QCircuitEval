#!/usr/bin/env python3
"""Generate and offline-regrade one diagnostic task per model configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pass1_generation import QueueJob, SegmentScope, build_command, read_queue

from qceval.production.campaign import BENCHMARK_CONTENT_COMMIT, CONFIGURATION_COUNT
from qceval.production.resume import accepted_records


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - fail-closed diagnostic orchestration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--source-hint", default=BENCHMARK_CONTENT_COMMIT)
    parser.add_argument("--provider-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    jobs = diagnostic_jobs(read_queue(args.queue))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    results: list[dict[str, Any]] = []
    for job in jobs:
        assert job.configuration_id is not None
        generation = args.out_dir / f"{job.configuration_id}.generated.jsonl"
        regraded = args.out_dir / f"{job.configuration_id}.regraded.jsonl"
        generation_command = build_command(
            job,
            output=generation,
            api_key_file=None if args.api_key_file is None else args.api_key_file.resolve(),
            source_hint=args.source_hint,
            cache_dir=cache_dir,
            scope=SegmentScope(suite="core", task_numbers=(1,)),
            provider_timeout=args.provider_timeout,
        )
        generation_run = subprocess.run(generation_command, check=False, text=True)
        if generation_run.returncode != 0:
            results.append(_result(job, generation, regraded, "generation_failed"))
            continue
        try:
            accepted = accepted_records([generation], strict_provenance=True)
            if len(accepted) != 1:
                raise ValueError(f"expected one accepted diagnostic generation, found {len(accepted)}")
            record = next(iter(accepted.values()))
            route = _route(record)
            if route.get("configuration_id") != job.configuration_id:
                raise ValueError("diagnostic generation has incompatible configuration provenance")
        except (OSError, ValueError) as exc:
            results.append(_result(job, generation, regraded, "generation_invalid", error=str(exc)))
            continue
        regrade_run = subprocess.run(offline_regrade_command(job, generation, regraded), check=False, text=True)
        if regrade_run.returncode != 0:
            results.append(_result(job, generation, regraded, "regrade_failed"))
            continue
        try:
            grade_records = _result_records(regraded)
            if len(grade_records) != 1 or grade_records[0].get("status") == "infrastructure_error":
                raise ValueError("offline diagnostic did not preserve exactly one model outcome")
            if _route(grade_records[0]).get("configuration_id") != job.configuration_id:
                raise ValueError("offline diagnostic lost configuration provenance")
        except (OSError, ValueError) as exc:
            results.append(_result(job, generation, regraded, "regrade_invalid", error=str(exc)))
            continue
        results.append(_result(job, generation, regraded, "passed"))

    passed = sum(result["status"] == "passed" for result in results)
    manifest = {
        "schema_version": "qceval.configuration_smoke.v1",
        "created_at_utc": _now(),
        "benchmark_denominator_member": False,
        "provider_requests": len(jobs),
        "offline_regrades": sum(Path(result["regraded_path"]).exists() for result in results),
        "configurations": len(results),
        "passed": passed,
        "failed": [result["configuration_id"] for result in results if result["status"] != "passed"],
        "results": results,
    }
    manifest_path = args.out_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if passed == CONFIGURATION_COUNT else 2


def diagnostic_jobs(jobs: Sequence[QueueJob]) -> list[QueueJob]:
    """Choose one deterministic Qiskit shard per schema-v2 configuration."""
    selected: dict[str, QueueJob] = {}
    for job in jobs:
        if job.queue_schema_version != 2 or job.configuration_id is None:
            raise ValueError("diagnostic smokes require a schema-v2 queue")
        if job.framework == "qiskit":
            selected[job.configuration_id] = job
    if len(selected) != CONFIGURATION_COUNT:
        raise ValueError(f"diagnostic smokes require exactly {CONFIGURATION_COUNT} configurations")
    return [selected[config_id] for config_id in sorted(selected)]


def offline_regrade_command(job: QueueJob, generation: Path, output: Path) -> list[str]:
    """Build a credential-free, provider-call-free single-task regrade command."""
    executable = Path(sys.executable).with_name("qceval")
    return [
        str(executable),
        "run",
        "--provider",
        "openrouter",
        "--model",
        job.model_id,
        "--framework",
        job.framework,
        "--suite",
        "core",
        "--source-hint",
        BENCHMARK_CONTENT_COMMIT,
        "--tasks",
        "1",
        "--regrade",
        job.framework,
        "--input",
        str(generation),
        "--out",
        str(output),
        "--output-format",
        "jsonl",
        "--evaluation-workers",
        "1",
        "--samples-per-task",
        "1",
        "--pass-k",
        "1",
        "--max-attempts",
        "1",
    ]


def _result(job: QueueJob, generation: Path, regraded: Path, status: str, *, error: str | None = None) -> dict:
    return {
        "configuration_id": job.configuration_id,
        "model_id": job.model_id,
        "reasoning_effort": job.reasoning_setting,
        "status": status,
        "error": error,
        "generated_path": str(generation),
        "generated_sha256": _sha256(generation) if generation.is_file() else None,
        "regraded_path": str(regraded),
        "regraded_sha256": _sha256(regraded) if regraded.is_file() else None,
    }


def _route(record: Mapping[str, Any]) -> Mapping[str, Any]:
    response = record.get("provider_response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    route = metadata.get("route") if isinstance(metadata, Mapping) else None
    return route if isinstance(route, Mapping) else {}


def _result_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if isinstance(payload, dict) and payload.get("kind") == "result":
            records.append(payload)
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
