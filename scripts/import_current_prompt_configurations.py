#!/usr/bin/env python3
"""Validate and import completed configurations for the current prompt set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qceval.core.bench import SUPPORTED_FRAMEWORKS, Adaptor
from qceval.production.campaign import (
    BENCHMARK_CONTENT_COMMIT,
    EFFORTS_BY_MODEL,
    REUSABLE_CONFIGURATION_IDS,
    configuration_id,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="CONFIGURATION_ID=PATH",
        help="Completed 280-record current-prompt candidate; repeat once per reusable configuration.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sources: dict[str, Path] = {}
    for value in args.source:
        config_id, separator, raw_path = value.partition("=")
        if not separator or config_id in sources:
            parser.error(f"invalid or duplicate --source: {value}")
        sources[config_id] = Path(raw_path)
    try:
        manifest = import_configurations(sources, args.out_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def import_configurations(sources: Mapping[str, Path], out_dir: Path) -> dict[str, Any]:
    """Copy exact current-prompt candidates with explicit reuse provenance."""
    expected_ids = set(REUSABLE_CONFIGURATION_IDS)
    if set(sources) != expected_ids:
        raise ValueError(f"current-prompt import requires exactly {sorted(expected_ids)}")
    identities = {
        configuration_id(model_id, effort): (model_id, effort)
        for model_id, efforts in EFFORTS_BY_MODEL.items()
        for effort in efforts
    }
    prompt_hashes = _prompt_hashes()
    out_dir.mkdir(parents=True, exist_ok=True)
    imported_at = _now()
    artifacts = []
    for config_id, source in sorted(sources.items()):
        if not source.is_file():
            raise ValueError(f"{config_id}: source is missing: {source}")
        model_id, effort = identities[config_id]
        source_hash = _sha256(source)
        payloads = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        results = [payload for payload in payloads if payload.get("kind") == "result"]
        summaries = [payload for payload in payloads if payload.get("kind") == "summary"]
        route = _validate_results(config_id, model_id, effort, results, prompt_hashes)
        if len(summaries) != 1:
            raise ValueError(f"{config_id}: expected exactly one summary")
        summary = summaries[0]
        qceval = summary.get("qceval")
        if (
            summary.get("model") != model_id
            or summary.get("configuration_id") not in {None, config_id}
            or not isinstance(qceval, Mapping)
            or qceval.get("source_hint") != BENCHMARK_CONTENT_COMMIT
        ):
            raise ValueError(f"{config_id}: summary does not prove the current prompt set")
        evidence = {
            "mode": "current_prompt_configuration_reuse",
            "configuration_id": config_id,
            "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
            "source_path": str(source.resolve()),
            "source_sha256": source_hash,
            "imported_at_utc": imported_at,
            "configuration_identity_sha256": route["configuration_identity_sha256"],
        }
        copied = []
        for payload in results:
            record = json.loads(json.dumps(payload))
            record["campaign_import"] = evidence
            copied.append(json.dumps(record, sort_keys=True))
        copied_summary = json.loads(json.dumps(summary))
        copied_summary["configuration_id"] = config_id
        copied_summary["campaign_import"] = evidence
        copied.append(json.dumps(copied_summary, sort_keys=True))
        output = out_dir / f"{config_id}__pass1.generated.jsonl"
        output.write_text("\n".join(copied) + "\n", encoding="utf-8")
        artifacts.append(
            {
                "model_id": model_id,
                "configuration_id": config_id,
                "reasoning_effort": effort,
                "path": str(output.resolve()),
                "sha256": _sha256(output),
                "source_path": str(source.resolve()),
                "source_sha256": source_hash,
                "records": 280,
                "route_verified_records": 280,
                "cost_covered_records": 280,
                "endpoint_tag": route["endpoint_tag"],
                "route_revision": route["route_revision"],
                "configuration_identity_sha256": route["configuration_identity_sha256"],
            }
        )
    manifest = {
        "schema_version": "qceval.current_prompt_import.v1",
        "created_at_utc": imported_at,
        "benchmark_content_commit": BENCHMARK_CONTENT_COMMIT,
        "configurations": len(artifacts),
        "records": sum(int(artifact["records"]) for artifact in artifacts),
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _validate_results(
    config_id: str,
    model_id: str,
    effort: str,
    results: list[dict[str, Any]],
    prompt_hashes: Mapping[tuple[str, str, str], str],
) -> dict[str, Any]:
    _require_framework_shape(config_id, results)
    routes = [_validated_route(config_id, model_id, effort, payload, prompt_hashes) for payload in results]
    identity = {
        (
            route.get("endpoint_tag"),
            route.get("route_revision"),
            route.get("configuration_identity_sha256"),
        )
        for route in routes
    }
    if len(identity) != 1:
        raise ValueError(f"{config_id}: candidate spans multiple route identities")
    return routes[0]


def _require_framework_shape(config_id: str, results: list[dict[str, Any]]) -> None:
    if len(results) != 280:
        raise ValueError(f"{config_id}: expected 280 records, found {len(results)}")
    counts = Counter(str(payload.get("framework")) for payload in results)
    keys = {
        (
            str(payload.get("framework")),
            str(payload.get("suite")),
            str(payload.get("task_id")),
            payload.get("sample_index"),
            payload.get("attempt_index"),
        )
        for payload in results
    }
    if len(keys) != 280 or set(counts) != set(SUPPORTED_FRAMEWORKS) or set(counts.values()) != {70}:
        raise ValueError(f"{config_id}: candidate does not contain four unique 70-task framework sets")


def _validated_route(
    config_id: str,
    model_id: str,
    effort: str,
    payload: Mapping[str, Any],
    prompt_hashes: Mapping[tuple[str, str, str], str],
) -> dict[str, Any]:
    key = (str(payload.get("framework")), str(payload.get("suite")), str(payload.get("task_id")))
    trace = payload.get("request_trace")
    response = payload.get("provider_response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    route = metadata.get("route") if isinstance(metadata, Mapping) else None
    usage = response.get("usage") if isinstance(response, Mapping) else None
    cost = usage.get("cost_usd") if isinstance(usage, Mapping) else None
    if payload.get("model") != model_id or not isinstance(trace, Mapping):
        raise ValueError(f"{config_id}: record identity is invalid")
    if trace.get("prompt_sha256") != prompt_hashes.get(key):
        raise ValueError(f"{config_id}: record does not match the current prompt hash: {key}")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{config_id}: record lacks provider metadata")
    _require_effort(config_id, effort, payload, metadata)
    _require_route_provenance(config_id, route)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(float(cost)):
        raise ValueError(f"{config_id}: record lacks provider-reported cost")
    return dict(route)


def _require_effort(config_id: str, effort: str, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    reported_effort = metadata.get("reasoning_effort")
    refused = (
        payload.get("status") == "provider_failed"
        and metadata.get("failure_classification") == "provider_policy_refusal"
    )
    if reported_effort not in {None, effort} or (reported_effort is None and not refused):
        raise ValueError(f"{config_id}: record has the wrong reasoning effort")


def _require_route_provenance(config_id: str, route: Any) -> None:
    if (
        not isinstance(route, Mapping)
        or route.get("route_verified") is not True
        or route.get("configuration_id") != config_id
        or route.get("allow_fallbacks") is not False
        or route.get("require_parameters") is not True
        or not isinstance(route.get("configuration_identity_sha256"), str)
    ):
        raise ValueError(f"{config_id}: record lacks verified configuration-bound route provenance")


def _prompt_hashes() -> dict[tuple[str, str, str], str]:
    adapter = Adaptor()
    return {
        (framework, suite, task.task_id): hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
        for framework in SUPPORTED_FRAMEWORKS
        for suite in ("core", "qec")
        for task in adapter.load_tasks(framework, suite)
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
