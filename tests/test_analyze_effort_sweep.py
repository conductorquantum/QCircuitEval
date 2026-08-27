from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_effort_sweep import analyze, write_outputs

from qceval.production.campaign import configuration_id


def _record(effort: str, framework: str, suite: str, *, passed: bool, finish_reason: str = "stop") -> dict:
    model = "openai/gpt-5.6-sol"
    config = configuration_id(model, effort)
    completion_tokens = 128000 if finish_reason == "length" else 10
    return {
        "kind": "result",
        "model": model,
        "suite": suite,
        "framework": framework,
        "task_id": "01" if suite == "core" else "qec01",
        "status": "passed" if passed else "failed",
        "provider_response": {
            "code": "def answer():\n    return 1",
            "metadata": {
                "reasoning_effort": effort,
                "finish_reason": finish_reason,
                "route": {
                    "configuration_id": config,
                    "endpoint_tag": "openai",
                    "max_output_tokens": 128000,
                    "output_limit_source": "author_native",
                    "endpoint_cap_status": "catalog_numeric",
                    "output_token_parameter": "max_completion_tokens",
                    "route_revision": "route-openai",
                    "temperature": 0.0,
                    "route_verified": True,
                },
                "attempt_history": [
                    {
                        "status": "accepted_model_outcome",
                        "started_at_utc": "2026-08-11T00:00:00Z",
                        "finished_at_utc": "2026-08-11T00:00:01Z",
                    }
                ],
            },
            "usage": {
                "cost_usd": 0.1,
                "prompt_tokens": 5,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": 3,
            },
            "raw_response": {"choices": [{"finish_reason": finish_reason, "message": {"content": "candidate"}}]},
        },
        "evaluation": {"verified_status": "verified_pass" if passed else "semantic_fail"},
    }


def _write(path: Path, effort: str, *, high: bool) -> None:
    records = []
    for framework in ("qiskit", "cirq", "pennylane", "cudaq"):
        records.append(_record(effort, framework, "core", passed=high))
        records.append(
            _record(
                effort,
                framework,
                "qec",
                passed=high,
                finish_reason="stop" if high or framework != "qiskit" else "length",
            )
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
        + json.dumps({"kind": "summary", "summary": {}})
        + "\n",
        encoding="utf-8",
    )


def test_analysis_computes_paired_cluster_delta_and_exclusive_failures(tmp_path: Path) -> None:
    low = tmp_path / "low.jsonl"
    high = tmp_path / "high.jsonl"
    _write(low, "low", high=False)
    _write(high, "high", high=True)

    report = analyze([low, high], bootstrap_samples=100, require_complete_campaign=False)
    delta = next(
        row for row in report["paired_effort_deltas"] if row["effort_a"] == "low" and row["effort_b"] == "high"
    )
    low_failures = next(
        row
        for row in report["failure_causes"]
        if row["configuration_id"] == configuration_id("openai/gpt-5.6-sol", "low")
    )

    assert delta["delta_b_minus_a"] == 1.0
    assert delta["clusters"] == 2
    assert sum(low_failures["counts"].values()) == 8
    assert low_failures["counts"]["full-ceiling length outcome"] == 1
    assert low_failures["counts"]["verifier rejection"] == 7

    output = tmp_path / "analysis"
    write_outputs(report, output)
    assert (output / "effort-sweep.json").is_file()
    assert (output / "effort-versus-pass1.png").stat().st_size > 0
    assert (output / "safeguards-and-refusals.png").stat().st_size > 0


def test_analysis_requires_explicit_audit_to_resolve_candidate_timeout(tmp_path: Path) -> None:
    path = tmp_path / "timeout.jsonl"
    record = _record("low", "cudaq", "core", passed=False)
    record["status"] = "infrastructure_error"
    record["evaluation"] = {
        "verified_status": "resource_limit",
        "error_type": "InfrastructureError",
        "error": "evaluation timed out after 600.000s",
    }
    path.write_text(
        json.dumps(record) + "\n" + json.dumps({"kind": "summary", "summary": {}}) + "\n",
        encoding="utf-8",
    )

    unresolved = analyze([path], bootstrap_samples=10, require_complete_campaign=False)
    assert unresolved["acceptance"]["unresolved_infrastructure_failures"] == 1

    audit = tmp_path / "resource-limit-audit.tsv"
    audit.write_text(
        "configuration_id\tsuite\tframework\ttask_id\taudit_decision\n"
        f"{configuration_id('openai/gpt-5.6-sol', 'low')}\tcore\tcudaq\t01\t"
        "confirmed_candidate_resource_limit\n",
        encoding="utf-8",
    )
    resolved = analyze(
        [path],
        resource_limit_audit=audit,
        bootstrap_samples=10,
        require_complete_campaign=False,
    )
    assert resolved["acceptance"]["unresolved_infrastructure_failures"] == 0
    assert resolved["acceptance"]["confirmed_candidate_resource_limits"] == 1
    assert resolved["failure_causes"][0]["counts"]["verifier rejection"] == 1


def test_analysis_recognizes_current_candidate_timeout_shape(tmp_path: Path) -> None:
    path = tmp_path / "timeout.jsonl"
    record = _record("low", "cudaq", "core", passed=False)
    record["status"] = "failed"
    record["evaluation"] = {
        "compiled": True,
        "ran": False,
        "passed": False,
        "verified_status": "resource_limit",
        "error_type": "EvaluationTimeout",
        "error": "evaluation timed out after 180.000s",
    }
    path.write_text(
        json.dumps(record) + "\n" + json.dumps({"kind": "summary", "summary": {}}) + "\n",
        encoding="utf-8",
    )

    report = analyze([path], bootstrap_samples=10, require_complete_campaign=False)

    assert report["acceptance"]["unresolved_infrastructure_failures"] == 0
    assert report["failure_causes"][0]["counts"]["verifier rejection"] == 1


def test_analysis_allows_missing_duplicate_effort_metadata_when_route_identity_is_verified(tmp_path: Path) -> None:
    path = tmp_path / "imported.jsonl"
    first = _record("max", "qiskit", "core", passed=True)
    second = _record("max", "cirq", "core", passed=True)
    del first["provider_response"]["metadata"]["reasoning_effort"]
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n" + json.dumps({"kind": "summary", "summary": {}}) + "\n",
        encoding="utf-8",
    )

    report = analyze([path], bootstrap_samples=10, require_complete_campaign=False)

    assert report["configurations"][0]["reasoning_effort"] == "max"
