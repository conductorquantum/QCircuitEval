#!/usr/bin/env python3
"""Consolidate all fresh model lanes for the current benchmark content."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.materialize_pass1_candidates import _accepted_job_records, _load_import_manifest
from scripts.run_pass1_generation import QueueJob, read_queue

from qceval.production.campaign import (
    CONFIGURATION_COUNT,
    EFFORTS_BY_MODEL,
    FRESH_ASSIGNMENT_COUNT,
    SHARD_COUNT,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--imports-manifest",
        type=Path,
        help="Legacy-only import evidence; current content requires every configuration to be fresh.",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="MODEL=SEGMENTS_DIR",
        help="Fresh segment root for one model; repeat exactly once per base model.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sources: dict[str, Path] = {}
    for value in args.source:
        model_id, separator, raw_path = value.partition("=")
        if not separator or model_id in sources:
            parser.error(f"invalid or duplicate --source: {value}")
        sources[model_id] = Path(raw_path)
    try:
        manifest = consolidate_segments(
            args.queue,
            sources,
            args.imports_manifest,
            args.out_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def consolidate_segments(
    queue: Path,
    sources: Mapping[str, Path],
    imports_manifest: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    if set(sources) != set(EFFORTS_BY_MODEL):
        raise ValueError(f"segment sources must contain exactly {sorted(EFFORTS_BY_MODEL)}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"consolidated segment directory must be absent or empty: {out_dir}")
    jobs = read_queue(queue)
    imported = _load_import_manifest(imports_manifest)
    fresh_jobs = _fresh_jobs(jobs, frozenset(imported))
    source_evidence = {
        model_id: _validate_source_plan(model_id, source) for model_id, source in sorted(sources.items())
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    configurations: set[str] = set()
    for job in fresh_jobs:
        assert job.configuration_id is not None
        source_job_dir = sources[job.model_id] / job.job_id
        paths = sorted(source_job_dir.glob("*.jsonl"))
        _accepted_job_records(job, paths)
        destination_job_dir = out_dir / job.job_id
        destination_job_dir.mkdir()
        configurations.add(job.configuration_id)
        for source_path in paths:
            destination = destination_job_dir / source_path.name
            shutil.copy2(source_path, destination)
            files.append(
                {
                    "job_id": job.job_id,
                    "configuration_id": job.configuration_id,
                    "model_id": job.model_id,
                    "source_path": str(source_path.resolve()),
                    "path": str(destination.resolve()),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
    manifest = {
        "schema_version": "qceval.consolidated_effort_segments.v1",
        "queue": str(queue.resolve()),
        "queue_sha256": _sha256(queue),
        "imports_manifest": str(imports_manifest.resolve()) if imports_manifest is not None else None,
        "imports_manifest_sha256": _sha256(imports_manifest) if imports_manifest is not None else None,
        "source_evidence": source_evidence,
        "models": len(sources),
        "fresh_configurations": len(configurations),
        "fresh_shards": len(fresh_jobs),
        "fresh_assignments": sum(job.assigned_tasks for job in fresh_jobs),
        "imported_configurations": len(imported),
        "segment_files": len(files),
        "files": files,
    }
    expected_fresh_configurations = CONFIGURATION_COUNT - len(imported)
    expected_fresh_shards = SHARD_COUNT - 4 * len(imported)
    if (
        manifest["fresh_configurations"] != expected_fresh_configurations
        or manifest["fresh_shards"] != expected_fresh_shards
        or manifest["fresh_assignments"] != FRESH_ASSIGNMENT_COUNT
        or manifest["imported_configurations"] != len(imported)
    ):
        raise ValueError(
            f"fresh segment consolidation did not produce the expected {expected_fresh_configurations} configurations"
        )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _fresh_jobs(jobs: Sequence[QueueJob], imported: frozenset[str]) -> list[QueueJob]:
    fresh = [job for job in jobs if job.configuration_id not in imported]
    if any(job.queue_schema_version != 2 or job.configuration_id is None for job in fresh):
        raise ValueError("fresh segment consolidation requires schema-v2 configuration identities")
    if sum(job.assigned_tasks for job in fresh) != FRESH_ASSIGNMENT_COUNT:
        raise ValueError(
            f"imported configuration set does not leave exactly {FRESH_ASSIGNMENT_COUNT:,} fresh assignments"
        )
    return sorted(fresh, key=lambda job: (job.model_id, job.configuration_id or "", job.framework))


def _validate_source_plan(model_id: str, segments_dir: Path) -> dict[str, Any]:
    if not segments_dir.is_dir():
        raise ValueError(f"{model_id}: segment source is missing: {segments_dir}")
    plan_path = segments_dir.parent / "generation-plan.json"
    if not plan_path.is_file():
        raise ValueError(f"{model_id}: source generation plan is missing")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = plan.get("selected_model_lanes")
    if selected is not None and model_id not in selected:
        raise ValueError(f"{model_id}: source generation plan did not select this model")
    return {
        "segments_dir": str(segments_dir.resolve()),
        "generation_plan": str(plan_path.resolve()),
        "generation_plan_sha256": _sha256(plan_path),
        "harness_commit": plan.get("harness_commit"),
        "run_manifest_sha256": plan.get("run_manifest_sha256"),
        "queue_sha256": plan.get("queue_sha256"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
