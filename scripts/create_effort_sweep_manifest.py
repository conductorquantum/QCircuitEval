#!/usr/bin/env python3
"""Freeze all gate evidence for the fresh maximum-reasoning Pass@1 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pass1_generation import read_queue

from qceval.production.campaign import (
    ASSIGNMENT_COUNT,
    BASE_MODEL_COUNT,
    BENCHMARK_CONTENT_COMMIT,
    CAMPAIGN_SCHEMA_VERSION,
    CONFIGURATION_COUNT,
    FRESH_ASSIGNMENT_COUNT,
    HISTORICAL_IMPORT_ASSIGNMENT_COUNT,
    HISTORICAL_MAX_IMPORT_MODELS,
    REUSABLE_CONFIGURATION_IDS,
    SHARD_COUNT,
    configuration_id,
)

PROHIBITED_SOURCE_FRAGMENTS = ("c6b11ef",)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--raw-catalog", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--preflight-hashes", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--benchmark-content-manifest", type=Path, required=True)
    parser.add_argument("--canaries", type=Path, required=True)
    parser.add_argument("--diagnostics-manifest", type=Path, required=True)
    parser.add_argument(
        "--historical-imports-manifest",
        type=Path,
        help="Legacy-only import evidence; the current campaign rejects historical candidates.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = create_manifest(
            registry=args.registry,
            raw_catalog=args.raw_catalog,
            selection=args.selection,
            preflight_hashes=args.preflight_hashes,
            queue=args.queue,
            benchmark_content_manifest=args.benchmark_content_manifest,
            canaries=args.canaries,
            diagnostics_manifest=args.diagnostics_manifest,
            historical_imports_manifest=args.historical_imports_manifest,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_manifest": str(args.out), "sha256": _sha256(args.out)}, sort_keys=True))
    return 0


def create_manifest(  # noqa: C901 - binds every fail-closed pre-generation artifact gate
    *,
    registry: Path,
    raw_catalog: Path,
    selection: Path,
    preflight_hashes: Path,
    queue: Path,
    benchmark_content_manifest: Path,
    canaries: Path,
    diagnostics_manifest: Path,
    historical_imports_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate every pre-generation gate and return an immutable run manifest."""
    paths = {
        "capability_registry": registry,
        "raw_openrouter_catalog": raw_catalog,
        "endpoint_selection": selection,
        "preflight_hashes": preflight_hashes,
        "queue": queue,
        "benchmark_content_manifest": benchmark_content_manifest,
        "canaries": canaries,
        "diagnostics_manifest": diagnostics_manifest,
    }
    if historical_imports_manifest is not None:
        paths["historical_imports_manifest"] = historical_imports_manifest
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} is missing: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(fragment in text or fragment in str(path) for fragment in PROHIBITED_SOURCE_FRAGMENTS):
            raise ValueError(f"{name} contains a prohibited historical artifact reference")

    jobs = read_queue(queue)
    if any(job.queue_schema_version != 2 for job in jobs):
        raise ValueError("run manifest refuses schema-v1 queue records")
    content = _json(benchmark_content_manifest)
    if content.get("benchmark_content_commit") != BENCHMARK_CONTENT_COMMIT or content.get("byte_identical") is not True:
        raise ValueError(f"benchmark-content manifest does not prove frozen content {BENCHMARK_CONTENT_COMMIT}")
    selected = _json(selection)
    if (
        selected.get("campaign_eligible") is not True
        or len(selected.get("models") or {}) != BASE_MODEL_COUNT
        or len(selected.get("configurations") or {}) != CONFIGURATION_COUNT
    ):
        raise ValueError(
            f"endpoint selection did not qualify all {BASE_MODEL_COUNT} models and {CONFIGURATION_COUNT} configurations"
        )
    _validate_canaries(canaries)
    diagnostics = _json(diagnostics_manifest)
    if (
        diagnostics.get("benchmark_denominator_member") is not False
        or diagnostics.get("configurations") != CONFIGURATION_COUNT
        or diagnostics.get("passed") != CONFIGURATION_COUNT
        or diagnostics.get("provider_requests") != CONFIGURATION_COUNT
    ):
        raise ValueError(
            f"diagnostic smoke manifest did not pass all {CONFIGURATION_COUNT} configurations outside the denominator"
        )
    historical_imports = _validate_historical_imports(historical_imports_manifest)

    harness_commit = _git("rev-parse", "HEAD")
    if (
        subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode != 0
    ):
        raise ValueError("tracked worktree must be clean before freezing the run manifest")
    artifacts = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "created_at_utc": _now(),
        "harness_commit": harness_commit,
        "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
        "queue_schema_version": 2,
        "base_models": BASE_MODEL_COUNT,
        "configurations": CONFIGURATION_COUNT,
        "shards": SHARD_COUNT,
        "logical_requests": ASSIGNMENT_COUNT,
        "fresh_logical_requests": FRESH_ASSIGNMENT_COUNT,
        "historical_imported_requests": HISTORICAL_IMPORT_ASSIGNMENT_COUNT,
        "historical_imports": historical_imports,
        "canary_requests": CONFIGURATION_COUNT,
        "diagnostic_requests": CONFIGURATION_COUNT,
        "artifacts": artifacts,
    }


def _validate_historical_imports(path: Path | None) -> dict[str, Any]:
    expected_ids = set(REUSABLE_CONFIGURATION_IDS) or {
        configuration_id(model_id, "max") for model_id in HISTORICAL_MAX_IMPORT_MODELS
    }
    if not expected_ids:
        if path is not None:
            raise ValueError("the current benchmark content requires fresh generation for every configuration")
        return {
            "schema_version": "none",
            "configuration_ids": [],
            "configurations": 0,
            "records": 0,
            "manifest_sha256": None,
        }
    if path is None:
        raise ValueError("historical import evidence is required for the frozen campaign")
    payload = _json(path)
    artifacts = payload.get("artifacts")
    if (
        payload.get("schema_version") not in {"qceval.historical_max_import.v1", "qceval.current_prompt_import.v1"}
        or payload.get("configurations") != len(expected_ids)
        or payload.get("records") != HISTORICAL_IMPORT_ASSIGNMENT_COUNT
        or not isinstance(artifacts, list)
    ):
        raise ValueError("candidate import manifest has invalid campaign cardinality")
    actual_ids: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("historical max import artifact must be an object")
        config_id = artifact.get("configuration_id")
        candidate_path = Path(str(artifact.get("path")))
        if (
            not isinstance(config_id, str)
            or config_id in actual_ids
            or artifact.get("records") != 280
            or artifact.get("route_verified_records") != 280
            or artifact.get("cost_covered_records") != 280
            or not candidate_path.is_file()
            or artifact.get("sha256") != _sha256(candidate_path)
        ):
            raise ValueError("historical max import artifact is incomplete or hash-invalid")
        actual_ids.add(config_id)
    if actual_ids != expected_ids:
        raise ValueError("historical max import manifest contains the wrong configurations")
    return {
        "schema_version": payload["schema_version"],
        "configuration_ids": sorted(actual_ids),
        "configurations": len(actual_ids),
        "records": payload["records"],
        "manifest_sha256": _sha256(path),
    }


def validate_run_manifest(path: Path, *, queue: Path, harness_commit: str) -> dict[str, Any]:
    """Validate the immutable evidence bundle immediately before generation."""
    payload = _json(path)
    if payload.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
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
    if cardinality != (
        BASE_MODEL_COUNT,
        CONFIGURATION_COUNT,
        SHARD_COUNT,
        ASSIGNMENT_COUNT,
    ):
        raise ValueError("run manifest campaign cardinality is invalid")
    artifacts = payload.get("artifacts")
    queue_artifact = artifacts.get("queue") if isinstance(artifacts, Mapping) else None
    if not isinstance(queue_artifact, Mapping) or queue_artifact.get("sha256") != _sha256(queue):
        raise ValueError("run manifest queue hash does not match the supplied queue")
    return payload


def _validate_canaries(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summaries = [row for row in rows if isinstance(row, Mapping) and row.get("kind") == "summary"]
    canary_rows = [row for row in rows if isinstance(row, Mapping) and row.get("kind") == "canary"]
    if len(summaries) != 1 or rows[-1] != summaries[0]:
        raise ValueError("canary artifact must end in exactly one summary")
    summary = summaries[0]
    if (
        len(canary_rows) != CONFIGURATION_COUNT
        or summary.get("configurations") != CONFIGURATION_COUNT
        or summary.get("passed") != CONFIGURATION_COUNT
        or summary.get("failed") != []
        or summary.get("benchmark_denominator_member") is not False
    ):
        raise ValueError(f"canary artifact did not pass all {CONFIGURATION_COUNT} configurations")
    if len({row.get("configuration_id") for row in canary_rows}) != CONFIGURATION_COUNT:
        raise ValueError("canary artifact contains duplicate configuration identities")


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
