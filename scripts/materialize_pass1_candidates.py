#!/usr/bin/env python3
"""Materialize authoritative generation-only Pass@1 candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_run_records import _record_route_protocol, _record_sort_key, _run_config
from scripts.run_pass1_generation import QueueJob, _assignments, read_queue

from qceval.core.io import write_output
from qceval.core.runner.records import _record_from_dict
from qceval.models import BenchmarkRecord
from qceval.production.resume import accepted_records
from qceval.reports import summarize


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--segments-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--imports-manifest",
        type=Path,
        help="Validated historical-candidate import manifest for configurations omitted from generation.",
    )
    args = parser.parse_args(argv)
    manifest = materialize_candidates(
        args.queue,
        args.segments_dir,
        args.out_dir,
        imports_manifest=args.imports_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def materialize_candidates(  # noqa: C901 - explicit candidate completeness gates
    queue: Path,
    segments_dir: Path,
    out_dir: Path,
    *,
    imports_manifest: Path | None = None,
) -> dict[str, Any]:
    """Write one complete candidate artifact per configuration from route segments."""
    jobs = read_queue(queue, validate_campaign=False)
    if not jobs or any(job.queue_schema_version != 2 or job.configuration_id is None for job in jobs):
        raise ValueError("maximum-reasoning candidate materialization refuses schema-v1 queue records")
    by_configuration: dict[str, list[QueueJob]] = defaultdict(list)
    for job in jobs:
        assert job.configuration_id is not None
        by_configuration[job.configuration_id].append(job)
    expected_frameworks = {"qiskit", "cirq", "pennylane", "cudaq"}
    if any(
        len(config_jobs) != 4
        or {job.framework for job in config_jobs} != expected_frameworks
        or {job.assigned_tasks for job in config_jobs} != {70}
        for config_jobs in by_configuration.values()
    ):
        raise ValueError("each candidate configuration requires four 70-task framework shards")
    imported_artifacts = _load_import_manifest(imports_manifest)
    unknown_imports = set(imported_artifacts) - set(by_configuration)
    if unknown_imports:
        raise ValueError(f"historical import contains unknown configurations: {sorted(unknown_imports)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for config_id, config_jobs in sorted(by_configuration.items()):
        model_id = config_jobs[0].model_id
        imported_artifact = imported_artifacts.get(config_id)
        if imported_artifact is not None:
            artifacts.append(
                _materialize_imported_candidate(
                    config_id,
                    config_jobs,
                    imported_artifact,
                    out_dir,
                )
            )
            continue
        payloads: list[dict[str, Any]] = []
        reference_summary: dict[str, Any] | None = None
        for job in sorted(config_jobs, key=lambda item: item.framework):
            job_dir = segments_dir / job.job_id
            paths = sorted(job_dir.glob("*.jsonl"))
            job_payloads = _accepted_job_records(job, paths)
            payloads.extend(job_payloads)
            if reference_summary is None:
                reference_summary = _first_summary(paths)
        if reference_summary is None:
            raise ValueError(f"{config_id}: no complete source summary is available")
        counts = Counter(str(payload["framework"]) for payload in payloads)
        if len(payloads) != 280 or set(counts.values()) != {70} or len(counts) != 4:
            raise ValueError(f"{config_id}: expected four 70-record framework shards, found {dict(counts)}")
        ordered = sorted(payloads, key=_record_sort_key)
        benchmark_records: list[BenchmarkRecord] = [_record_from_dict(payload) for payload in ordered]
        config = _run_config(reference_summary, ordered)
        merged_summary = summarize(benchmark_records, run_config=config)
        _record_route_protocol(merged_summary, ordered)
        output = out_dir / f"{config_id}__pass1.generated.jsonl"
        write_output(
            output,
            {
                "schema_version": reference_summary["schema_version"],
                "provider": reference_summary["provider"],
                "model": model_id,
                "configuration_id": config_id,
                "qceval": reference_summary["qceval"],
                "suites": ["core", "qec"],
                "run_id": str(uuid.uuid4()),
                "results": [record.to_dict() for record in benchmark_records],
                "summary": merged_summary,
            },
            "jsonl",
        )
        artifacts.append(
            {
                "model_id": model_id,
                "configuration_id": config_id,
                "reasoning_effort": config_jobs[0].reasoning_setting,
                "path": str(output),
                "sha256": _sha256(output),
                "records": len(benchmark_records),
                "framework_records": dict(sorted(counts.items())),
                "route_verified_records": sum(_route_verified(payload) for payload in ordered),
                "cost_covered_records": sum(_reported_cost(payload) is not None for payload in ordered),
            }
        )
    manifest = {
        "schema_version": "1",
        "queue": str(queue),
        "queue_sha256": _sha256(queue),
        "base_models": len({artifact["model_id"] for artifact in artifacts}),
        "configurations": len(artifacts),
        "records": sum(int(artifact["records"]) for artifact in artifacts),
        "imports_manifest": str(imports_manifest.resolve()) if imports_manifest is not None else None,
        "imports_manifest_sha256": _sha256(imports_manifest) if imports_manifest is not None else None,
        "imported_configurations": len(imported_artifacts),
        "imported_records": sum(
            int(artifact["records"]) for artifact in artifacts if artifact.get("historical_import") is True
        ),
        "artifacts": artifacts,
    }
    expected_models = len({job.model_id for job in jobs})
    expected_configurations = len(by_configuration)
    expected_records = sum(job.assigned_tasks for job in jobs)
    if (
        len(jobs) != expected_configurations * 4
        or expected_records != expected_configurations * 280
        or manifest["base_models"] != expected_models
        or manifest["configurations"] != expected_configurations
        or manifest["records"] != expected_records
    ):
        raise ValueError(
            "candidate materialization did not reproduce the queue's complete four-framework, 280-record configurations"
        )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _load_import_manifest(  # noqa: C901 - explicit import provenance gates
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"historical import manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {
        "qceval.historical_max_import.v1",
        "qceval.current_prompt_import.v1",
    }:
        raise ValueError("historical import manifest has an unsupported schema")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("historical import manifest artifacts must be a list")
    artifacts: dict[str, dict[str, Any]] = {}
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, Mapping):
            raise ValueError("historical import artifact must be an object")
        artifact = dict(raw_artifact)
        config_id = artifact.get("configuration_id")
        if not isinstance(config_id, str) or config_id in artifacts:
            raise ValueError("historical import configuration IDs must be unique strings")
        source = Path(str(artifact.get("path")))
        if not source.is_absolute():
            source = path.parent / source
        artifact["path"] = str(source)
        artifacts[config_id] = artifact
    expected_count = manifest.get("configurations")
    expected_records = manifest.get("records")
    if expected_count != len(artifacts) or expected_records != sum(
        int(artifact.get("records", -1)) for artifact in artifacts.values()
    ):
        raise ValueError("historical import manifest cardinality does not match its artifacts")
    return artifacts


def _materialize_imported_candidate(  # noqa: C901 - explicit import provenance gates
    config_id: str,
    config_jobs: Sequence[QueueJob],
    artifact: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    model_id = config_jobs[0].model_id
    if any(job.model_id != model_id or job.configuration_id != config_id for job in config_jobs):
        raise ValueError(f"{config_id}: queue shards disagree on configuration identity")
    if len(config_jobs) != 4 or {job.framework for job in config_jobs} != {
        "qiskit",
        "cirq",
        "pennylane",
        "cudaq",
    }:
        raise ValueError(f"{config_id}: historical import requires four queue framework shards")
    reasoning_effort = config_jobs[0].reasoning_setting
    if artifact.get("reasoning_effort") != reasoning_effort:
        raise ValueError(f"{config_id}: imported reasoning effort does not match the queue")
    if artifact.get("model_id") != model_id or artifact.get("records") != 280:
        raise ValueError(f"{config_id}: historical import identity or cardinality is invalid")
    source = Path(str(artifact.get("path")))
    if not source.is_file() or _sha256(source) != artifact.get("sha256"):
        raise ValueError(f"{config_id}: historical import artifact hash does not match")
    payloads = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    results = [payload for payload in payloads if payload.get("kind") == "result"]
    summaries = [payload for payload in payloads if payload.get("kind") == "summary"]
    counts = Counter(str(payload.get("framework")) for payload in results)
    logical_keys = {
        (
            payload.get("suite"),
            payload.get("framework"),
            payload.get("task_id"),
            payload.get("sample_index"),
            payload.get("attempt_index"),
        )
        for payload in results
    }
    if len(results) != 280 or len(logical_keys) != 280 or len(counts) != 4 or set(counts.values()) != {70}:
        raise ValueError(f"{config_id}: historical import does not contain four unique 70-record shards")
    if len(summaries) != 1 or summaries[0].get("configuration_id") != config_id:
        raise ValueError(f"{config_id}: historical import summary identity is invalid")
    for payload in results:
        response = payload.get("provider_response")
        metadata = response.get("metadata") if isinstance(response, Mapping) else None
        route = metadata.get("route") if isinstance(metadata, Mapping) else None
        evidence = payload.get("campaign_import")
        if payload.get("model") != model_id or not isinstance(route, Mapping):
            raise ValueError(f"{config_id}: historical record identity is invalid")
        if route.get("configuration_id") != config_id or not _route_verified(payload):
            raise ValueError(f"{config_id}: historical record lacks configuration-bound route provenance")
        if _reported_cost(payload) is None:
            raise ValueError(f"{config_id}: historical record lacks provider-reported cost")
        if not isinstance(evidence, Mapping) or evidence.get("configuration_id") != config_id:
            raise ValueError(f"{config_id}: historical record lacks import provenance")
        if evidence.get("configuration_identity_sha256") != route.get("configuration_identity_sha256"):
            raise ValueError(f"{config_id}: historical import configuration identity does not match its route")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{config_id}__pass1.generated.jsonl"
    if output.resolve() != source.resolve():
        shutil.copy2(source, output)
    return {
        "model_id": model_id,
        "configuration_id": config_id,
        "reasoning_effort": reasoning_effort,
        "path": str(output),
        "sha256": _sha256(output),
        "records": 280,
        "framework_records": dict(sorted(counts.items())),
        "route_verified_records": 280,
        "cost_covered_records": 280,
        "historical_import": True,
        "import_source_path": artifact.get("source_path"),
        "import_source_sha256": artifact.get("source_sha256"),
        "endpoint_tag": artifact.get("endpoint_tag"),
        "route_revision": artifact.get("route_revision"),
        "configuration_identity_sha256": artifact.get("configuration_identity_sha256"),
    }


def _accepted_job_records(  # noqa: C901 - explicit accepted-outcome gates
    job: QueueJob, paths: Sequence[Path]
) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError(f"{job.job_id}: no route segments are available")
    accepted = accepted_records(paths, strict_provenance=True)
    expected = set(_assignments(job))
    if set(accepted) != expected:
        missing = sorted(expected - set(accepted))
        foreign = sorted(set(accepted) - expected)
        raise ValueError(f"{job.job_id}: incomplete accepted set; missing={missing[:3]} foreign={foreign[:3]}")
    payloads = []
    for key in sorted(expected):
        payload = {name: value for name, value in accepted[key].items() if name != "kind"}
        if payload.get("model") != job.model_id or payload.get("framework") != job.framework:
            raise ValueError(f"{job.job_id}: accepted record identity does not match its queue shard")
        if payload.get("status") == "infrastructure_error":
            raise ValueError(f"{job.job_id}: infrastructure record entered the accepted set")
        if not _route_verified(payload):
            raise ValueError(f"{job.job_id}: accepted record lacks verified route provenance")
        if _reported_cost(payload) is None:
            raise ValueError(f"{job.job_id}: accepted record lacks provider-reported cost")
        route = ((payload.get("provider_response") or {}).get("metadata") or {}).get("route") or {}
        if route.get("configuration_id") != job.configuration_id:
            raise ValueError(f"{job.job_id}: accepted record belongs to another configuration")
        payloads.append(payload)
    if len(payloads) != 70:
        raise ValueError(f"{job.job_id}: expected 70 accepted records, found {len(payloads)}")
    return payloads


def _first_summary(paths: Sequence[Path]) -> dict[str, Any]:
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("kind") == "summary":
                return {name: value for name, value in payload.items() if name != "kind"}
    raise ValueError("route segments do not contain a completed source summary")


def _route_verified(payload: Mapping[str, Any]) -> bool:
    response = payload.get("provider_response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    route = metadata.get("route") if isinstance(metadata, Mapping) else None
    return isinstance(route, Mapping) and route.get("route_verified") is True


def _reported_cost(payload: Mapping[str, Any]) -> float | None:
    response = payload.get("provider_response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    value = usage.get("cost_usd") if isinstance(usage, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    cost = float(value)
    return cost if math.isfinite(cost) and cost >= 0.0 else None


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
