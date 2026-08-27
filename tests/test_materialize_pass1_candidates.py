from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.materialize_pass1_candidates import (
    _accepted_job_records,
    _load_import_manifest,
    _materialize_imported_candidate,
)
from scripts.run_pass1_generation import QueueJob


def _job() -> QueueJob:
    return QueueJob(
        job_id="openai-gpt-5-6-sol__effort-max__pass1__qiskit",
        model_id="openai/gpt-5.6-sol",
        reasoning_setting="max",
        protocol="pass1",
        framework="qiskit",
        suite="all",
        max_tasks=0,
        endpoint_tag="author",
        configured_output_tokens=128000,
        output_limit_source="author_native",
        endpoint_cap_status="catalog_numeric",
        output_token_parameter="max_tokens",
        route_revision="route-a",
        temperature_behavior="explicit_zero",
        assigned_tasks=70,
        configuration_id="openai-gpt-5-6-sol__effort-max",
    )


def _jobs() -> list[QueueJob]:
    return [
        QueueJob(**{**_job().__dict__, "job_id": f"imported-{framework}", "framework": framework})
        for framework in ("qiskit", "cirq", "pennylane", "cudaq")
    ]


def _record(suite: str, task_id: str, *, status: str = "generated", verified: bool = True, cost=0.1) -> dict:
    return {
        "kind": "result",
        "suite": suite,
        "framework": "qiskit",
        "task_id": task_id,
        "sample_index": 0,
        "attempt_index": 0,
        "model": "openai/gpt-5.6-sol",
        "status": status,
        "provider_response": {
            "metadata": {
                "route": {
                    "route_verified": verified,
                    "configuration_id": "openai-gpt-5-6-sol__effort-max",
                }
            },
            "usage": {"cost_usd": cost},
        },
    }


def _complete_records() -> list[dict]:
    return [
        *(_record("core", f"{number:02d}") for number in range(1, 59)),
        *(_record("qec", f"qec{number:02d}") for number in range(1, 13)),
    ]


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _imported_candidate(path: Path) -> None:
    records = []
    config_id = "openai-gpt-5-6-sol__effort-max"
    for framework in ("qiskit", "cirq", "pennylane", "cudaq"):
        for number in range(70):
            records.append(
                {
                    "kind": "result",
                    "suite": "core",
                    "framework": framework,
                    "task_id": str(number),
                    "sample_index": 0,
                    "attempt_index": 0,
                    "model": "openai/gpt-5.6-sol",
                    "status": "generated",
                    "provider_response": {
                        "metadata": {
                            "route": {
                                "route_verified": True,
                                "configuration_id": config_id,
                                "configuration_identity_sha256": "identity",
                            }
                        },
                        "usage": {"cost_usd": 0.1},
                    },
                    "campaign_import": {
                        "configuration_id": config_id,
                        "configuration_identity_sha256": "identity",
                    },
                }
            )
    records.append({"kind": "summary", "configuration_id": config_id})
    _write(path, records)


def test_materializer_preserves_retry_success_and_drops_infrastructure_record(tmp_path: Path) -> None:
    first = tmp_path / "route-a.jsonl"
    second = tmp_path / "route-b.jsonl"
    records = _complete_records()
    failed = _record("core", "05", status="infrastructure_error", verified=False, cost=None)
    successful = next(record for record in records if record["suite"] == "core" and record["task_id"] == "05")
    _write(first, [failed, *(record for record in records if record is not successful)])
    _write(second, [successful])

    accepted = _accepted_job_records(_job(), [first, second])

    assert len(accepted) == 70
    assert sum(record["task_id"] == "05" for record in accepted) == 1
    assert all(record["status"] != "infrastructure_error" for record in accepted)


@pytest.mark.parametrize(
    ("verified", "cost", "message"),
    [(False, 0.1, "verified route"), (True, None, "provider-reported cost")],
)
def test_materializer_rejects_incomplete_provenance(
    tmp_path: Path, verified: bool, cost: float | None, message: str
) -> None:
    path = tmp_path / "route.jsonl"
    records = _complete_records()
    records[0] = _record("core", "01", verified=verified, cost=cost)
    _write(path, records)

    with pytest.raises(ValueError, match=message):
        _accepted_job_records(_job(), [path])


def test_materializer_copies_validated_historical_candidate(tmp_path: Path) -> None:
    source = tmp_path / "import.jsonl"
    _imported_candidate(source)
    artifact = {
        "configuration_id": "openai-gpt-5-6-sol__effort-max",
        "model_id": "openai/gpt-5.6-sol",
        "reasoning_effort": "max",
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_path": "/archive/sol.jsonl",
        "source_sha256": "source-hash",
        "records": 280,
        "endpoint_tag": "azure/eu",
        "route_revision": "historical-route",
        "configuration_identity_sha256": "identity",
    }

    materialized = _materialize_imported_candidate(
        "openai-gpt-5-6-sol__effort-max",
        _jobs(),
        artifact,
        tmp_path / "out",
    )

    assert materialized["historical_import"] is True
    assert materialized["endpoint_tag"] == "azure/eu"
    assert Path(materialized["path"]).read_bytes() == source.read_bytes()


def test_import_manifest_rejects_cardinality_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "qceval.historical_max_import.v1",
                "configurations": 1,
                "records": 279,
                "artifacts": [
                    {
                        "configuration_id": "openai-gpt-5-6-sol__effort-max",
                        "path": "candidate.jsonl",
                        "records": 280,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cardinality"):
        _load_import_manifest(manifest)
