from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.import_historical_max_outputs import IMPORT_MODELS, import_historical_max_outputs


def _candidate(path: Path, model_id: str, *, missing_cost: bool = False) -> None:
    records = []
    for framework in ("qiskit", "cirq", "pennylane", "cudaq"):
        for number in range(70):
            records.append(
                {
                    "kind": "result",
                    "model": model_id,
                    "suite": "core",
                    "framework": framework,
                    "task_id": f"{number + 1:02d}",
                    "sample_index": 0,
                    "attempt_index": 0,
                    "status": "generated",
                    "provider_response": {
                        "metadata": {
                            "reasoning_effort": "max",
                            "route": {
                                "route_verified": True,
                                "allow_fallbacks": False,
                                "require_parameters": True,
                                "endpoint_tag": "provider/region",
                                "max_output_tokens": 128000,
                                "output_limit_source": "author_native",
                                "endpoint_cap_status": "catalog_numeric",
                                "output_token_parameter": "max_tokens",
                                "route_revision": "route-old",
                                "temperature": None,
                            },
                        },
                        "usage": {} if missing_cost and number == 0 else {"cost_usd": 0.1},
                    },
                }
            )
    records.append(
        {
            "kind": "summary",
            "model": model_id,
            "provider": "openrouter",
            "schema_version": "qceval.run.v2",
            "suites": ["core", "qec"],
            "qceval": {"source_hint": "02061df263c1204f61776cbdb8d7295f820f029c"},
        }
    )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _source_campaign(tmp_path: Path) -> tuple[Path, Path]:
    canaries = tmp_path / "canaries.jsonl"
    rows = []
    for model_id in sorted(IMPORT_MODELS):
        rows.append(
            {
                "kind": "canary",
                "model_id": model_id,
                "status": "passed",
                "generation_id": f"canary-{model_id}",
                "provider_response_metadata": {
                    "route": {
                        "route_verified": True,
                        "endpoint_tag": "provider/region",
                        "max_output_tokens": 128000,
                        "output_limit_source": "author_native",
                        "endpoint_cap_status": "catalog_numeric",
                        "output_token_parameter": "max_tokens",
                        "route_revision": "route-old",
                        "temperature": None,
                    }
                },
            }
        )
    canaries.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    run_manifest = tmp_path / "run-manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "benchmark_content_commit": "02061df263c1204f61776cbdb8d7295f820f029c",
                "benchmark_content_byte_identical": True,
                "canaries": {
                    "sha256": hashlib.sha256(canaries.read_bytes()).hexdigest(),
                    "benchmark_denominator_member": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_manifest, canaries


def _import(sources: dict[str, Path], tmp_path: Path):
    run_manifest, canaries = _source_campaign(tmp_path)
    return import_historical_max_outputs(
        sources,
        tmp_path / "out",
        source_run_manifest=run_manifest,
        source_canaries=canaries,
    )


def test_import_historical_max_outputs_preserves_route_and_adds_configuration_identity(tmp_path: Path) -> None:
    sources = {}
    for index, model_id in enumerate(sorted(IMPORT_MODELS)):
        source = tmp_path / f"source-{index}.jsonl"
        _candidate(source, model_id)
        sources[model_id] = source

    manifest = _import(sources, tmp_path)

    assert manifest["configurations"] == 3
    assert manifest["records"] == 840
    artifact = manifest["artifacts"][0]
    payload = json.loads(Path(artifact["path"]).read_text(encoding="utf-8").splitlines()[0])
    route = payload["provider_response"]["metadata"]["route"]
    assert route["endpoint_tag"] == "provider/region"
    assert route["route_revision"] == "route-old"
    assert route["configuration_id"].endswith("__effort-max")
    assert route["configuration_identity_sha256"] == artifact["configuration_identity_sha256"]
    assert payload["campaign_import"]["source_sha256"] == artifact["source_sha256"]
    assert payload["campaign_import"]["source_canary_generation_id"].startswith("canary-")
    assert manifest["source_campaign"]["benchmark_content_byte_identical"] is True


def test_import_historical_max_outputs_rejects_missing_cost(tmp_path: Path) -> None:
    sources = {}
    for index, model_id in enumerate(sorted(IMPORT_MODELS)):
        source = tmp_path / f"source-{index}.jsonl"
        _candidate(source, model_id, missing_cost=index == 0)
        sources[model_id] = source

    with pytest.raises(ValueError, match="lacks provider-reported cost"):
        _import(sources, tmp_path)


def test_import_historical_max_outputs_rejects_missing_metadata(tmp_path: Path) -> None:
    sources = {}
    for index, model_id in enumerate(sorted(IMPORT_MODELS)):
        source = tmp_path / f"source-{index}.jsonl"
        _candidate(source, model_id)
        if index == 0:
            records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
            del records[0]["provider_response"]["metadata"]
            source.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
        sources[model_id] = source

    with pytest.raises(ValueError, match="is not max effort"):
        _import(sources, tmp_path)


def test_import_historical_max_outputs_rejects_wrong_benchmark_source(tmp_path: Path) -> None:
    sources = {}
    for index, model_id in enumerate(sorted(IMPORT_MODELS)):
        source = tmp_path / f"source-{index}.jsonl"
        _candidate(source, model_id)
        if index == 0:
            records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
            records[-1]["qceval"]["source_hint"] = "wrong"
            source.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
        sources[model_id] = source

    with pytest.raises(ValueError, match="frozen benchmark source"):
        _import(sources, tmp_path)
