#!/usr/bin/env python3
"""Validate and import archived max-effort candidates into the effort sweep."""

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

from qceval.production.campaign import configuration_id, configuration_identity_sha256

LEGACY_BENCHMARK_CONTENT_COMMIT = "02061df263c1204f61776cbdb8d7295f820f029c"
IMPORT_MODELS = frozenset(
    {
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5",
        "openai/gpt-5.6-sol",
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="MODEL=PATH",
        help="Archived 280-result max candidate; repeat once per imported model.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--source-canaries", type=Path, required=True)
    args = parser.parse_args(argv)
    sources: dict[str, Path] = {}
    for value in args.source:
        model_id, separator, raw_path = value.partition("=")
        if not separator or model_id in sources:
            parser.error(f"invalid or duplicate --source: {value}")
        sources[model_id] = Path(raw_path)
    try:
        manifest = import_historical_max_outputs(
            sources,
            args.out_dir,
            source_run_manifest=args.source_run_manifest,
            source_canaries=args.source_canaries,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def import_historical_max_outputs(
    sources: Mapping[str, Path],
    out_dir: Path,
    *,
    source_run_manifest: Path,
    source_canaries: Path,
) -> dict[str, Any]:
    """Validate archived candidates and attach explicit import provenance."""
    if set(sources) != IMPORT_MODELS:
        raise ValueError(f"historical max import requires exactly {sorted(IMPORT_MODELS)}")
    source_campaign = _validate_source_campaign(source_run_manifest, source_canaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    imported_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifacts = []
    for model_id, source in sorted(sources.items()):
        if not source.is_file():
            raise ValueError(f"historical source is missing: {source}")
        source_sha256 = _sha256(source)
        payloads = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        results = [payload for payload in payloads if payload.get("kind") == "result"]
        summaries = [payload for payload in payloads if payload.get("kind") == "summary"]
        route = _validate_results(model_id, results)
        if len(summaries) != 1:
            raise ValueError(f"{model_id}: expected exactly one source summary")
        _validate_summary(model_id, summaries[0])
        canary = source_campaign["canaries_by_model"][model_id]
        _validate_canary_route(model_id, canary, route)
        config_id = configuration_id(model_id, "max")
        temperature_behavior = "explicit_zero" if route.get("temperature") == 0.0 else "not_exposed"
        identity_hash = configuration_identity_sha256(
            model_id=model_id,
            effort="max",
            endpoint_tag=str(route["endpoint_tag"]),
            configured_output_tokens=int(route["max_output_tokens"]),
            output_token_parameter=str(route["output_token_parameter"]),
            temperature_behavior=temperature_behavior,
            route_revision=str(route["route_revision"]),
        )
        import_evidence = {
            "mode": "archived_max_output",
            "source_path": str(source.resolve()),
            "source_sha256": source_sha256,
            "imported_at_utc": imported_at,
            "configuration_id": config_id,
            "configuration_identity_sha256": identity_hash,
            "source_run_manifest_sha256": source_campaign["run_manifest_sha256"],
            "source_canaries_sha256": source_campaign["canaries_sha256"],
            "source_canary_generation_id": canary["generation_id"],
        }
        output = out_dir / f"{config_id}__pass1.generated.jsonl"
        lines = []
        for payload in results:
            copied = json.loads(json.dumps(payload))
            copied_route = copied["provider_response"]["metadata"]["route"]
            copied_route["configuration_id"] = config_id
            copied_route["configuration_identity_sha256"] = identity_hash
            copied["campaign_import"] = import_evidence
            lines.append(json.dumps(copied, sort_keys=True))
        summary = json.loads(json.dumps(summaries[0]))
        summary["configuration_id"] = config_id
        summary["campaign_import"] = import_evidence
        lines.append(json.dumps(summary, sort_keys=True))
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        artifacts.append(
            {
                "model_id": model_id,
                "configuration_id": config_id,
                "reasoning_effort": "max",
                "path": str(output.resolve()),
                "sha256": _sha256(output),
                "source_path": str(source.resolve()),
                "source_sha256": source_sha256,
                "records": 280,
                "route_verified_records": 280,
                "cost_covered_records": 280,
                "endpoint_tag": route["endpoint_tag"],
                "route_revision": route["route_revision"],
                "configuration_identity_sha256": identity_hash,
                "source_canary_generation_id": canary["generation_id"],
            }
        )
    manifest = {
        "schema_version": "qceval.historical_max_import.v1",
        "created_at_utc": imported_at,
        "configurations": len(artifacts),
        "records": sum(item["records"] for item in artifacts),
        "source_campaign": {key: value for key, value in source_campaign.items() if key != "canaries_by_model"},
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _validate_results(  # noqa: C901 - explicit historical-artifact gates
    model_id: str, results: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(results) != 280:
        raise ValueError(f"{model_id}: expected 280 archived max records, found {len(results)}")
    keys = {
        (
            payload.get("suite"),
            payload.get("framework"),
            payload.get("task_id"),
            payload.get("sample_index"),
            payload.get("attempt_index"),
        )
        for payload in results
    }
    if len(keys) != 280:
        raise ValueError(f"{model_id}: archived max records contain duplicate logical keys")
    counts = Counter(str(payload.get("framework")) for payload in results)
    if len(counts) != 4 or set(counts.values()) != {70}:
        raise ValueError(f"{model_id}: expected four 70-task frameworks, found {dict(counts)}")
    routes = []
    for payload in results:
        if payload.get("model") != model_id:
            raise ValueError(f"{model_id}: archived record has the wrong model")
        response = payload.get("provider_response")
        metadata = response.get("metadata") if isinstance(response, Mapping) else None
        route = metadata.get("route") if isinstance(metadata, Mapping) else None
        usage = response.get("usage") if isinstance(response, Mapping) else None
        cost = usage.get("cost_usd") if isinstance(usage, Mapping) else None
        if not isinstance(metadata, Mapping) or metadata.get("reasoning_effort") != "max":
            raise ValueError(f"{model_id}: archived record is not max effort")
        if not isinstance(route, Mapping) or not _valid_route(route):
            raise ValueError(f"{model_id}: archived record lacks exact verified route provenance")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(float(cost)):
            raise ValueError(f"{model_id}: archived record lacks provider-reported cost")
        routes.append(dict(route))
    identity_fields = {
        (
            route["endpoint_tag"],
            route["max_output_tokens"],
            route["output_token_parameter"],
            route["route_revision"],
            route.get("temperature"),
        )
        for route in routes
    }
    if len(identity_fields) != 1:
        raise ValueError(f"{model_id}: archived max records span multiple route identities")
    return routes[0]


def _validate_summary(model_id: str, summary: Mapping[str, Any]) -> None:
    qceval = summary.get("qceval")
    if (
        summary.get("model") != model_id
        or summary.get("provider") != "openrouter"
        or summary.get("schema_version") != "qceval.run.v2"
        or summary.get("suites") != ["core", "qec"]
        or not isinstance(qceval, Mapping)
        or qceval.get("source_hint") != LEGACY_BENCHMARK_CONTENT_COMMIT
    ):
        raise ValueError(f"{model_id}: archived summary does not prove the frozen benchmark source")


def _validate_source_campaign(run_manifest_path: Path, canaries_path: Path) -> dict[str, Any]:
    if not run_manifest_path.is_file() or not canaries_path.is_file():
        raise ValueError("historical source campaign manifest or canaries are missing")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    canaries_sha256 = _sha256(canaries_path)
    canary_evidence = run_manifest.get("canaries") if isinstance(run_manifest, Mapping) else None
    if (
        not isinstance(run_manifest, Mapping)
        or run_manifest.get("benchmark_content_commit") != LEGACY_BENCHMARK_CONTENT_COMMIT
        or run_manifest.get("benchmark_content_byte_identical") is not True
        or not isinstance(canary_evidence, Mapping)
        or canary_evidence.get("sha256") != canaries_sha256
        or canary_evidence.get("benchmark_denominator_member") is not False
    ):
        raise ValueError("historical source campaign does not prove its benchmark or canary evidence")
    rows = [json.loads(line) for line in canaries_path.read_text(encoding="utf-8").splitlines()]
    by_model = {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("kind") == "canary" and row.get("model_id") in IMPORT_MODELS
    }
    if set(by_model) != IMPORT_MODELS:
        raise ValueError("historical source canaries do not cover every imported model")
    return {
        "run_manifest_path": str(run_manifest_path.resolve()),
        "run_manifest_sha256": _sha256(run_manifest_path),
        "canaries_path": str(canaries_path.resolve()),
        "canaries_sha256": canaries_sha256,
        "benchmark_content_commit": LEGACY_BENCHMARK_CONTENT_COMMIT,
        "benchmark_content_byte_identical": True,
        "canaries_by_model": by_model,
    }


def _validate_canary_route(
    model_id: str,
    canary: Mapping[str, Any],
    candidate_route: Mapping[str, Any],
) -> None:
    metadata = canary.get("provider_response_metadata")
    canary_route = metadata.get("route") if isinstance(metadata, Mapping) else None
    fields = (
        "endpoint_tag",
        "max_output_tokens",
        "output_limit_source",
        "endpoint_cap_status",
        "output_token_parameter",
        "route_revision",
        "temperature",
    )
    if (
        canary.get("status") != "passed"
        or not isinstance(canary_route, Mapping)
        or canary_route.get("route_verified") is not True
        or any(canary_route.get(field) != candidate_route.get(field) for field in fields)
    ):
        raise ValueError(f"{model_id}: archived candidate route does not match its passed source canary")


def _valid_route(route: Mapping[str, Any]) -> bool:
    return (
        route.get("route_verified") is True
        and route.get("allow_fallbacks") is False
        and route.get("require_parameters") is True
        and route.get("max_output_tokens") == 128000
        and route.get("output_limit_source") == "author_native"
        and route.get("endpoint_cap_status") == "catalog_numeric"
        and isinstance(route.get("endpoint_tag"), str)
        and isinstance(route.get("output_token_parameter"), str)
        and isinstance(route.get("route_revision"), str)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
