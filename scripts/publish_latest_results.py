#!/usr/bin/env python3
"""Publish the latest complete Pass@1 matrix as canonical ``qceval run`` artifacts.

The published tree is byte-for-byte what ``qceval run`` writes for a
directory-shaped matrix destination: one ``<configuration_id>.json`` run
envelope per configuration plus a ``qceval.effort_sweep.v1`` manifest. Integrity
and campaign provenance are recorded separately in ``provenance.json`` so the
manifest keeps the exact shape the CLI emits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_run_records import _record_route_protocol, _run_config

from qceval.cli_plan import (
    ALL_REASONING_TOKEN,
    ReasoningJob,
    jobs_from_registry,
    load_registry_efforts,
    sweep_manifest_path,
    sweep_output_path,
    write_sweep_manifest,
)
from qceval.core.io import write_output
from qceval.core.runner.records import _record_from_dict
from qceval.reports import summarize

MERGED_SUFFIX = "__pass1.regraded.jsonl"
PROVENANCE_SCHEMA_VERSION = "qceval.published_results.v1"
PROVENANCE_FILENAME = "provenance.json"
RUN_SCHEMA_VERSION = "qceval.run.v2"
# Offline-regraded campaign artifacts carry every field of a fresh run envelope
# except ``run_identity``, which the merge utility does not reconstruct.
RUN_ENVELOPE_KEYS = frozenset(
    ("schema_version", "run_id", "provider", "model", "suites", "qceval", "results", "summary", "configuration_id")
)
DEFAULT_REGISTRIES = (
    Path("production/models.prompt-effort-sweep.json"),
    Path("production/models.max-reasoning.json"),
)
DEFAULT_OUT_DIR = Path("results/published")


@dataclass(frozen=True)
class Campaign:
    """A completed campaign that can supply published configurations."""

    name: str
    merged_dir: Path
    audit: Path
    ready_key: tuple[str, ...]
    configurations_key: tuple[str, ...]
    records_key: tuple[str, ...]
    commit_file: Path
    commit_key: tuple[str, ...]


PROMPT_EFFORT_CAMPAIGN = Campaign(
    name="prompt-effort-pass1-20260818T032316Z",
    merged_dir=Path("results/prompt-effort-pass1-20260818T032316Z/offline-grading/merged"),
    audit=Path("results/prompt-effort-pass1-20260818T032316Z/analysis/effort-sweep.json"),
    ready_key=("acceptance", "publication_ready"),
    configurations_key=("acceptance", "configurations"),
    records_key=("acceptance", "records"),
    commit_file=Path("results/prompt-effort-pass1-20260818T032316Z/run-manifest.json"),
    commit_key=("benchmark_content_commit",),
)
MAX_REASONING_CAMPAIGN = Campaign(
    name="max-reasoning-pass1-20260815T190022Z",
    merged_dir=Path("results/max-reasoning-pass1-20260815T190022Z/offline-grading-no-glm-retry4/merged"),
    audit=Path("results/max-reasoning-pass1-20260815T190022Z/final-audit-no-glm.json"),
    ready_key=("publication_ready",),
    configurations_key=("scope", "configurations"),
    records_key=("coverage", "offline_regraded_records"),
    commit_file=Path("results/max-reasoning-pass1-20260815T190022Z/final-audit-no-glm.json"),
    commit_key=("source", "benchmark_content_commit"),
)
# Earlier campaigns supply only the configurations the newest one does not cover.
DEFAULT_CAMPAIGNS = (PROMPT_EFFORT_CAMPAIGN, MAX_REASONING_CAMPAIGN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--registry", type=Path, action="append", dest="registries")
    args = parser.parse_args(argv)
    try:
        provenance = publish_latest_results(
            args.repo_root.resolve(),
            args.out_dir,
            registries=tuple(args.registries) if args.registries else DEFAULT_REGISTRIES,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(provenance["scope"], indent=2, sort_keys=True))
    return 0


def publish_latest_results(
    repo: Path,
    out_dir: Path,
    *,
    registries: Sequence[Path] = DEFAULT_REGISTRIES,
    campaigns: Sequence[Campaign] = DEFAULT_CAMPAIGNS,
) -> dict[str, Any]:
    """Materialize the registry-defined matrix and return its provenance record.

    Args:
        repo: Repository root that contains the registries and campaigns.
        out_dir: Publication directory, relative to ``repo`` unless absolute.
        registries: Capability registries defining the published matrix.
        campaigns: Candidate campaigns in descending precedence.

    Returns:
        The provenance record written alongside the published artifacts.
    """
    jobs = jobs_from_registry(
        load_registry_efforts([_require_file(repo / path) for path in registries]),
        requested_effort=ALL_REASONING_TOKEN,
        model_filter=None,
    )
    inventories = [(campaign, _campaign_inventory(repo, campaign)) for campaign in campaigns]
    sources: dict[str, tuple[Campaign, Path]] = {}
    for campaign, inventory in inventories:
        for configuration, path in inventory.items():
            sources.setdefault(configuration, (campaign, path))

    relative_out = out_dir if not out_dir.is_absolute() else out_dir.relative_to(repo)
    destination_dir = repo / relative_out
    _require_empty(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for job in jobs:
        configuration = job.configuration_id()
        if configuration not in sources:
            raise ValueError(f"no completed campaign artifact for configuration: {configuration}")
        campaign, source = sources[configuration]
        payload = _run_payload(source, job)
        out = sweep_output_path(relative_out, job, multi_model=True)
        destination = repo / out
        write_output(destination, payload)
        _verify_written(destination, payload)
        entries.append(
            {
                "model": job.model,
                "reasoning_effort": job.effort,
                "configuration_id": configuration,
                "out": str(out),
                "exit_code": 0,
            }
        )
        artifacts.append(
            {
                "configuration_id": configuration,
                "model": job.model,
                "reasoning_effort": job.effort,
                "out": str(out),
                "records": len(payload["results"]),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "campaign": campaign.name,
                "source": source.relative_to(repo).as_posix(),
                "source_sha256": _sha256(source),
            }
        )

    write_sweep_manifest(repo / sweep_manifest_path(relative_out), entries)
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "scope": {
            "models": len({job.model for job in jobs}),
            "configurations": len(jobs),
            "records": sum(int(artifact["records"]) for artifact in artifacts),
            "pass_k": 1,
            "attempts_per_task": 1,
        },
        "registries": [{"path": Path(path).as_posix(), "sha256": _sha256(repo / path)} for path in registries],
        "campaigns": [
            {
                "name": campaign.name,
                "audit": campaign.audit.as_posix(),
                "audit_sha256": _sha256(repo / campaign.audit),
                "benchmark_content_commit": _lookup(
                    _read_object(repo / campaign.commit_file), campaign.commit_key, campaign.commit_file
                ),
                "published_configurations": sorted(
                    artifact["configuration_id"] for artifact in artifacts if artifact["campaign"] == campaign.name
                ),
            }
            for campaign, _ in inventories
        ],
        "artifacts": artifacts,
    }
    provenance_path = destination_dir / PROVENANCE_FILENAME
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def _campaign_inventory(repo: Path, campaign: Campaign) -> dict[str, Path]:
    """Validate one campaign's acceptance evidence and return its artifacts."""
    audit = _read_object(_require_file(repo / campaign.audit))
    if _lookup(audit, campaign.ready_key, campaign.audit) is not True:
        raise ValueError(f"{campaign.name}: campaign is not publication-ready")
    merged_dir = repo / campaign.merged_dir
    if not merged_dir.is_dir():
        raise ValueError(f"{campaign.name}: merged result directory is missing: {campaign.merged_dir}")
    inventory = {path.name.removesuffix(MERGED_SUFFIX): path for path in sorted(merged_dir.glob(f"*{MERGED_SUFFIX}"))}
    if not inventory:
        raise ValueError(f"{campaign.name}: merged result directory has no artifacts")
    if _lookup(audit, campaign.configurations_key, campaign.audit) != len(inventory):
        raise ValueError(f"{campaign.name}: audit configuration count does not match its merged artifacts")
    records = sum(_record_count(path) for path in inventory.values())
    if _lookup(audit, campaign.records_key, campaign.audit) != records:
        raise ValueError(f"{campaign.name}: audit record count does not match its merged artifacts")
    return inventory


def _run_payload(path: Path, job: ReasoningJob) -> dict[str, Any]:
    """Rebuild the run envelope that produced one merged JSONL artifact."""
    configuration = job.configuration_id()
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        kind = row.pop("kind", None)
        if kind == "result":
            results.append(row)
        elif kind == "summary":
            summaries.append(row)
        else:
            raise ValueError(f"{path}:{line_number}: unsupported row kind: {kind!r}")
    if len(summaries) != 1 or not results:
        raise ValueError(f"{path}: expected result rows followed by exactly one summary row")
    results = [_align_result_status(row) for row in results]
    results = [_redact_provider_telemetry(row) for row in results]
    records = [_record_from_dict(row) for row in results]
    rebuilt_summary = summarize(records, run_config=_run_config(summaries[0], results))
    _record_route_protocol(rebuilt_summary, results)
    payload = summaries[0] | {"results": results, "summary": rebuilt_summary}
    _validate_payload(payload, path, job=job, configuration=configuration)
    return payload


def _align_result_status(row: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a claimed status that disagrees with its embedded evaluation."""
    payload = row.get("evaluation")
    if not isinstance(payload, Mapping):
        return row
    compiled = payload.get("compiled")
    ran = payload.get("ran")
    passed = payload.get("passed")
    if not all(isinstance(flag, bool) for flag in (compiled, ran, passed)):
        return row
    derived = "compile_failed" if not compiled else "run_failed" if not ran else ("passed" if passed else "failed")
    if row.get("status") == derived:
        return row
    aligned = dict(row)
    aligned["status"] = derived
    return aligned


def _redact_provider_telemetry(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop vendor payloads that public score artifacts do not need."""
    redacted = dict(row)
    response = redacted.get("provider_response")
    if isinstance(response, Mapping):
        cleaned = dict(response)
        cleaned["raw_response"] = None
        redacted["provider_response"] = cleaned
    _strip_generation_ids(redacted)
    return redacted


def _strip_generation_ids(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("generation_id", None)
        for nested in value.values():
            _strip_generation_ids(nested)
        return
    if isinstance(value, list):
        for item in value:
            _strip_generation_ids(item)


def _validate_payload(payload: Mapping[str, Any], path: Path, *, job: ReasoningJob, configuration: str) -> None:
    _validate_envelope(payload, path, job=job, configuration=configuration)
    _validate_results(payload["results"], path, job=job, configuration=configuration)
    _validate_pass1_summary(payload, path)


def _validate_envelope(payload: Mapping[str, Any], path: Path, *, job: ReasoningJob, configuration: str) -> None:
    if set(payload) != RUN_ENVELOPE_KEYS:
        raise ValueError(f"{path}: run envelope fields do not match a published Pass@1 run")
    if payload["schema_version"] != RUN_SCHEMA_VERSION or payload["configuration_id"] != configuration:
        raise ValueError(f"{path}: run envelope schema or configuration identity is invalid")
    if payload["model"] != job.model or payload["provider"] != "openrouter":
        raise ValueError(f"{path}: run envelope provider or model does not match the registry job")


def _validate_results(
    results: Sequence[Mapping[str, Any]], path: Path, *, job: ReasoningJob, configuration: str
) -> None:
    identities = {
        (row["suite"], row["framework"], row["task_id"], row["sample_index"], row["attempt_index"]) for row in results
    }
    if len(identities) != len(results):
        raise ValueError(f"{path}: run contains duplicate task identities")
    if any(row["status"] == "infrastructure_error" for row in results):
        raise ValueError(f"{path}: run contains unresolved infrastructure failures")
    if any(
        isinstance(evaluation := row.get("evaluation"), Mapping) and evaluation.get("error_type") == "EvaluationTimeout"
        for row in results
    ):
        raise ValueError(f"{path}: run contains grader evaluation timeouts")
    if {row["model"] for row in results} != {job.model} or {row["provider"] for row in results} != {"openrouter"}:
        raise ValueError(f"{path}: result provider or model is inconsistent with the run envelope")
    routes = {_route(row).get("configuration_id") for row in results}
    if routes != {configuration}:
        raise ValueError(f"{path}: result route provenance does not match {configuration}")
    efforts = {effort for row in results if (effort := _metadata(row).get("reasoning_effort")) is not None}
    if not efforts <= {job.reasoning_effort()}:
        raise ValueError(f"{path}: result reasoning effort does not match {configuration}")


def _validate_pass1_summary(payload: Mapping[str, Any], path: Path) -> None:
    protocol = payload["summary"].get("run_protocol") or {}
    if (protocol.get("samples_per_task"), protocol.get("pass_k"), protocol.get("max_attempts")) != (1, 1, 1):
        raise ValueError(f"{path}: run is not a one-sample, one-attempt Pass@1 artifact")
    if payload["summary"].get("assigned_tasks") != len(payload["results"]):
        raise ValueError(f"{path}: summary denominator does not match its result records")


def _verify_written(path: Path, payload: Mapping[str, Any]) -> None:
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"{path}: written artifact does not round-trip to its run payload")


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return (row.get("provider_response") or {}).get("metadata") or {}


def _route(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _metadata(row).get("route") or {}


def _record_count(path: Path) -> int:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(len(lines) - 1, 0)


def _lookup(payload: Mapping[str, Any], keys: Sequence[str], path: Path) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"{path}: missing required field: {'.'.join(keys)}")
        value = value[key]
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return path


def _require_empty(path: Path) -> None:
    if path.exists() and any(item for item in path.iterdir() if item.name != "README.md"):
        raise ValueError(f"publication directory already contains artifacts: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
