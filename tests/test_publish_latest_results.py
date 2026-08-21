from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.publish_latest_results import (
    MERGED_SUFFIX,
    PROVENANCE_FILENAME,
    Campaign,
    publish_latest_results,
)

from qceval.cli import main as qceval_main

PRIMARY = Campaign(
    name="primary",
    merged_dir=Path("results/primary/merged"),
    audit=Path("results/primary/audit.json"),
    ready_key=("acceptance", "publication_ready"),
    configurations_key=("acceptance", "configurations"),
    records_key=("acceptance", "records"),
    commit_file=Path("results/primary/run-manifest.json"),
    commit_key=("benchmark_content_commit",),
)
SUPPLEMENTAL = Campaign(
    name="supplemental",
    merged_dir=Path("results/supplemental/merged"),
    audit=Path("results/supplemental/audit.json"),
    ready_key=("publication_ready",),
    configurations_key=("scope", "configurations"),
    records_key=("coverage", "offline_regraded_records"),
    commit_file=Path("results/supplemental/audit.json"),
    commit_key=("source", "benchmark_content_commit"),
)
CAMPAIGNS = (PRIMARY, SUPPLEMENTAL)
REGISTRIES = (Path("production/models.primary.json"), Path("production/models.max.json"))


def _result(
    configuration: str,
    model: str,
    effort: str,
    task_number: int,
    *,
    status: str = "passed",
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": "result",
        "suite": "core",
        "framework": "qiskit",
        "task_id": f"{task_number:03d}",
        "sample_index": 0,
        "attempt_index": 0,
        "status": status,
        "model": model,
        "provider": "openrouter",
        "entry_point": "build",
        "provider_response": {
            "metadata": {
                "reasoning_effort": None if effort == "enabled" else effort,
                "route": {"configuration_id": configuration},
            }
        },
    }
    if evaluation is not None:
        payload["evaluation"] = evaluation
    return payload


def _artifact(path: Path, *, model: str, effort: str, records: int = 2, **overrides: Any) -> None:
    configuration = f"{model}__effort-{effort}"
    rows = [_result(configuration, model, effort, number) for number in range(1, records + 1)]
    for key, value in overrides.items():
        rows[0][key] = value
    summary = {
        "kind": "summary",
        "schema_version": "qceval.run.v2",
        "run_id": f"run-{configuration}",
        "provider": "openrouter",
        "model": model,
        "suites": ["core"],
        "qceval": {"version": "test"},
        "configuration_id": configuration,
        "summary": {
            "assigned_tasks": records,
            "pass_rate": 1.0,
            "run_protocol": {"samples_per_task": 1, "pass_k": 1, "max_attempts": 1},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in (*rows, summary)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _registry(path: Path, models: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": [{"model_id": model, "reasoning_efforts": efforts} for model, efforts in models.items()]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _audit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo(tmp_path: Path, *, records: int = 2, **overrides: Any) -> Path:
    """Build a repository whose registries span two completed campaigns."""
    repo = tmp_path / "repo"
    _registry(repo / REGISTRIES[0], {"model-a": ["low", "high"]})
    _registry(repo / REGISTRIES[1], {"model-b": ["max"]})
    primary = repo / PRIMARY.merged_dir
    for effort in ("low", "high"):
        _artifact(primary / f"model-a__effort-{effort}{MERGED_SUFFIX}", model="model-a", effort=effort, records=records)
    supplemental = repo / SUPPLEMENTAL.merged_dir
    _artifact(supplemental / f"model-b__effort-max{MERGED_SUFFIX}", model="model-b", effort="max", records=records)
    # Also present in the newer campaign, and therefore not the published source.
    _artifact(
        supplemental / f"model-a__effort-high{MERGED_SUFFIX}",
        model="model-a",
        effort="high",
        records=records,
        **overrides,
    )
    _audit(
        repo / PRIMARY.audit,
        {"acceptance": {"publication_ready": True, "configurations": 2, "records": 2 * records}},
    )
    _audit(repo / PRIMARY.commit_file, {"benchmark_content_commit": "a" * 40})
    _audit(
        repo / SUPPLEMENTAL.audit,
        {
            "publication_ready": True,
            "scope": {"configurations": 2},
            "coverage": {"offline_regraded_records": 2 * records},
            "source": {"benchmark_content_commit": "b" * 40},
        },
    )
    return repo


def _publish(repo: Path, out_dir: Path = Path("results/published")) -> dict[str, Any]:
    return publish_latest_results(repo, out_dir, registries=REGISTRIES, campaigns=CAMPAIGNS)


def test_published_tree_is_a_directory_shaped_matrix_run(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    provenance = _publish(repo)

    published = repo / "results/published"
    assert sorted(path.name for path in published.iterdir()) == [
        "manifest.json",
        "model-a__effort-high.json",
        "model-a__effort-low.json",
        "model-b__effort-max.json",
        PROVENANCE_FILENAME,
    ]
    assert provenance["scope"] == {
        "models": 2,
        "configurations": 3,
        "records": 6,
        "pass_k": 1,
        "attempts_per_task": 1,
    }
    envelope = json.loads((published / "model-a__effort-low.json").read_text(encoding="utf-8"))
    assert envelope["configuration_id"] == "model-a__effort-low"
    assert len(envelope["results"]) == 2
    assert all("kind" not in row for row in envelope["results"])


def test_published_artifacts_match_a_real_matrix_run(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    reference = tmp_path / "reference"
    assert (
        qceval_main(
            [
                "run",
                "--provider",
                "smoke",
                "--reasoning-effort",
                "all",
                "--framework",
                "qiskit",
                "--tasks",
                "1",
                "--out",
                f"{reference}/",
            ]
        )
        == 0
    )

    _publish(repo)

    published = repo / "results/published"
    reference_manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    published_manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    assert published_manifest["schema_version"] == reference_manifest["schema_version"]
    assert {key for job in published_manifest["jobs"] for key in job} == {
        key for job in reference_manifest["jobs"] for key in job
    }
    assert published_manifest["jobs"] == [
        {
            "configuration_id": "model-a__effort-low",
            "exit_code": 0,
            "model": "model-a",
            "out": "results/published/model-a__effort-low.json",
            "reasoning_effort": "low",
        },
        {
            "configuration_id": "model-a__effort-high",
            "exit_code": 0,
            "model": "model-a",
            "out": "results/published/model-a__effort-high.json",
            "reasoning_effort": "high",
        },
        {
            "configuration_id": "model-b__effort-max",
            "exit_code": 0,
            "model": "model-b",
            "out": "results/published/model-b__effort-max.json",
            "reasoning_effort": "max",
        },
    ]
    reference_envelope = json.loads((reference / "smoke-canonical__effort-low.json").read_text(encoding="utf-8"))
    published_envelope = json.loads((published / "model-a__effort-low.json").read_text(encoding="utf-8"))
    # Offline-regraded artifacts carry every run field except the identity block,
    # which the merge utility does not reconstruct.
    assert set(reference_envelope) - set(published_envelope) == {"run_identity"}
    assert set(published_envelope) - set(reference_envelope) == set()
    for path in (published / "model-a__effort-low.json", published / "manifest.json"):
        text = path.read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


def test_newest_campaign_wins_for_overlapping_configurations(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    provenance = _publish(repo)

    sources = {artifact["configuration_id"]: artifact["campaign"] for artifact in provenance["artifacts"]}
    assert sources == {
        "model-a__effort-low": "primary",
        "model-a__effort-high": "primary",
        "model-b__effort-max": "supplemental",
    }
    campaigns = {campaign["name"]: campaign["published_configurations"] for campaign in provenance["campaigns"]}
    assert campaigns["supplemental"] == ["model-b__effort-max"]


def test_publish_omits_provider_raw_response_and_generation_ids(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    artifact = repo / PRIMARY.merged_dir / f"model-a__effort-low{MERGED_SUFFIX}"
    rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("kind") != "result":
            continue
        row["provider_response"] = {
            "code": "def build():\n    return None\n",
            "raw_response": {"id": "gen-secret", "choices": [{"native_finish_reason": "stop"}]},
            "metadata": {
                "generation_id": "gen-secret",
                "reasoning_effort": "low",
                "route": {"configuration_id": "model-a__effort-low"},
                "attempt_history": [{"attempt_number": 1, "generation_id": "gen-secret"}],
            },
        }
    artifact.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    _publish(repo)

    envelope = json.loads((repo / "results/published/model-a__effort-low.json").read_text(encoding="utf-8"))
    response = envelope["results"][0]["provider_response"]
    assert response["raw_response"] is None
    assert "generation_id" not in json.dumps(envelope["results"][0])


def test_publish_rejects_missing_configuration(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / SUPPLEMENTAL.merged_dir / f"model-b__effort-max{MERGED_SUFFIX}").unlink()
    _audit(
        repo / SUPPLEMENTAL.audit,
        {
            "publication_ready": True,
            "scope": {"configurations": 1},
            "coverage": {"offline_regraded_records": 2},
            "source": {"benchmark_content_commit": "b" * 40},
        },
    )

    with pytest.raises(ValueError, match="no completed campaign artifact"):
        _publish(repo)


def test_publish_rewrites_status_to_match_evaluation(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _artifact(
        repo / PRIMARY.merged_dir / f"model-a__effort-low{MERGED_SUFFIX}",
        model="model-a",
        effort="low",
        status="failed",
        evaluation={"compiled": False, "ran": False, "passed": False},
    )

    _publish(repo)

    envelope = json.loads((repo / "results/published/model-a__effort-low.json").read_text(encoding="utf-8"))
    assert envelope["results"][0]["status"] == "compile_failed"
    assert envelope["summary"]["compile_failures"] == 1


def test_publish_rejects_infrastructure_failures(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _artifact(
        repo / PRIMARY.merged_dir / f"model-a__effort-low{MERGED_SUFFIX}",
        model="model-a",
        effort="low",
        status="infrastructure_error",
    )

    with pytest.raises(ValueError, match="unresolved infrastructure failures"):
        _publish(repo)


def test_publish_rejects_grader_evaluation_timeouts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _artifact(
        repo / PRIMARY.merged_dir / f"model-a__effort-low{MERGED_SUFFIX}",
        model="model-a",
        effort="low",
        status="compile_failed",
        evaluation={
            "compiled": False,
            "ran": False,
            "passed": False,
            "error_type": "EvaluationTimeout",
        },
    )

    with pytest.raises(ValueError, match="grader evaluation timeouts"):
        _publish(repo)


def test_publish_rejects_audit_that_disagrees_with_its_artifacts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _audit(
        repo / PRIMARY.audit,
        {"acceptance": {"publication_ready": True, "configurations": 2, "records": 99}},
    )

    with pytest.raises(ValueError, match="audit record count"):
        _publish(repo)


def test_publish_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)

    with pytest.raises(ValueError, match="already contains artifacts"):
        _publish(repo)
