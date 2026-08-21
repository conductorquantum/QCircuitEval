#!/usr/bin/env python3
"""Derive a smaller Pass@1 queue while preserving parent-queue provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pass1_generation import read_queue


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scope-manifest", required=True, type=Path)
    parser.add_argument("--exclude-model", action="append", default=[])
    args = parser.parse_args(argv)

    manifest = filter_queue(
        args.queue,
        args.out,
        args.scope_manifest,
        excluded_models=args.exclude_model,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def filter_queue(
    queue: Path,
    out: Path,
    scope_manifest: Path,
    *,
    excluded_models: Sequence[str],
) -> dict[str, object]:
    """Write a filtered schema-v2 queue and its parent-linked scope manifest."""
    jobs = read_queue(queue)
    exclusions = frozenset(excluded_models)
    if not exclusions:
        raise ValueError("at least one --exclude-model value is required")
    available_models = {job.model_id for job in jobs}
    unknown = exclusions - available_models
    if unknown:
        raise ValueError(f"excluded models are not present in the parent queue: {sorted(unknown)}")

    raw_lines = [line for line in queue.read_text(encoding="utf-8").splitlines() if line]
    if len(raw_lines) != len(jobs):
        raise ValueError("parent queue contains blank or unparsable rows")
    selected_lines = [line for line, job in zip(raw_lines, jobs, strict=True) if job.model_id not in exclusions]
    selected_jobs = [job for job in jobs if job.model_id not in exclusions]
    if not selected_jobs:
        raise ValueError("scope exclusion removed every queue job")

    frameworks_by_configuration: dict[str, set[str]] = {}
    assignments_by_configuration: Counter[str] = Counter()
    for job in selected_jobs:
        if job.queue_schema_version != 2 or job.configuration_id is None:
            raise ValueError("scoped queue requires schema-v2 configuration identities")
        frameworks_by_configuration.setdefault(job.configuration_id, set()).add(job.framework)
        assignments_by_configuration[job.configuration_id] += job.assigned_tasks
    expected_frameworks = {"qiskit", "cirq", "pennylane", "cudaq"}
    if any(frameworks != expected_frameworks for frameworks in frameworks_by_configuration.values()):
        raise ValueError("each scoped configuration must retain exactly four framework shards")
    if set(assignments_by_configuration.values()) != {280}:
        raise ValueError("each scoped configuration must retain exactly 280 assignments")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{line}\n" for line in selected_lines), encoding="utf-8")
    selected_models = sorted({job.model_id for job in selected_jobs})
    manifest: dict[str, object] = {
        "schema_version": "qceval.pass1_scope.v1",
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parent_queue": str(queue.resolve()),
        "parent_queue_sha256": _sha256(queue),
        "parent_shards": len(jobs),
        "parent_logical_requests": sum(job.assigned_tasks for job in jobs),
        "excluded_models": sorted(exclusions),
        "queue": str(out.resolve()),
        "queue_sha256": _sha256(out),
        "base_models": len(selected_models),
        "models": selected_models,
        "configurations": len(frameworks_by_configuration),
        "shards": len(selected_jobs),
        "logical_requests": sum(job.assigned_tasks for job in selected_jobs),
    }
    scope_manifest.parent.mkdir(parents=True, exist_ok=True)
    scope_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
