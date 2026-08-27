from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import scripts.score_submission as scorer
from scripts.score_submission import REQUIRED_METADATA_FIELDS, main, recompute_summary, validate_submission

from qceval.core.bench import DEFAULT_FRAMEWORKS, Adaptor
from qceval.core.runner.trusted import TrustedRegradeTimeout, evaluate_trusted_candidate
from qceval.models import QCEvalEvaluation, QCEvalTask


class _StubAdaptor:
    """Use real task identities but constant-time trusted outcomes."""

    def __init__(self, outcome: str = "passed") -> None:
        self._bundle = Adaptor()
        self.outcome = outcome

    def load_tasks(self, framework: Any, suite: Any = "core") -> list[QCEvalTask]:
        return self._bundle.load_tasks(framework, suite=suite)

    def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
        del task, code
        if self.outcome == "infrastructure_error":
            return QCEvalEvaluation(
                compiled=False,
                ran=False,
                passed=False,
                error_type="InfrastructureError",
            )
        passed = self.outcome == "passed"
        return QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=passed,
            verified_status="verified_pass" if passed else "verified_fail",
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "bundled-qceval",
            "package_version": "1.2.3",
            "commit": "a" * 40,
            "dirty": False,
        }


def _exec_candidate_evaluation(task: QCEvalTask, code: str) -> QCEvalEvaluation:
    """Execute tiny containment-test candidates without framework grading."""
    del task
    namespace: dict[str, Any] = {}
    exec(compile(code, "<trusted-regrade-test>", "exec"), namespace)
    return QCEvalEvaluation(
        compiled=True,
        ran=True,
        passed=True,
        verified_status="verified_pass",
    )


def _validate_official(
    payload: dict[str, Any],
    *,
    adapter: _StubAdaptor | None = None,
) -> list[str]:
    metadata = _valid_metadata()
    metadata["provider"] = str(payload.get("provider"))
    metadata["model"] = str(payload.get("model"))
    trusted_adapter = adapter or _StubAdaptor()
    return validate_submission(
        payload,
        track="core-all-single",
        strict=True,
        metadata=metadata,
        adapter=trusted_adapter,
        _evaluation_hook=trusted_adapter.evaluate,
    )


def test_validate_submission_accepts_official_core_track_with_trusted_regrade() -> None:
    payload = _official_core_payload()

    errors = _validate_official(payload)

    assert errors == []


def test_validate_submission_rejects_missing_official_task_without_regrading() -> None:
    payload = _official_core_payload()
    payload["results"] = payload["results"][:-1]
    payload["summary"]["total_tasks"] = len(payload["results"])
    payload["summary"]["passed"] = len(payload["results"])
    adapter = _StubAdaptor()

    errors = _validate_official(payload, adapter=adapter)

    assert any("task IDs do not match bundled order" in error for error in errors)


def test_validate_submission_official_track_requires_metadata() -> None:
    payload = _official_core_payload()

    errors = validate_submission(
        payload,
        track="core-all-single",
        strict=False,
        adapter=_StubAdaptor(),
    )

    assert errors == ["official tracks and --strict custom validation require --metadata"]


def test_score_submission_main_writes_trusted_leaderboard_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = tmp_path / "submission.json"
    out = tmp_path / "leaderboard-row.json"
    metadata = _valid_metadata()
    metadata_path = tmp_path / "metadata.json"
    submission.write_text(json.dumps(_official_core_payload()), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(scorer, "Adaptor", lambda: _StubAdaptor("passed"))

    exit_code = main(
        [
            str(submission),
            "--track",
            "core-all-single",
            "--strict",
            "--metadata",
            str(metadata_path),
            "--out",
            str(out),
        ],
        _evaluation_hook=_StubAdaptor("passed").evaluate,
    )

    assert exit_code == 0
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["track"] == "core-all-single"
    assert row["passed"] == 232
    assert row["trusted_regrade"] is True
    assert row["validation_mode"] == "trusted_local_regrade"
    assert row["trusted_local_adapter"] == {
        "source": "bundled-qceval",
        "package_version": "1.2.3",
        "commit": "a" * 40,
        "commit_status": "available",
        "dirty": False,
    }
    assert row["metadata"] == metadata


def test_trusted_outcomes_override_forged_pass_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = tmp_path / "submission.json"
    out = tmp_path / "leaderboard-row.json"
    metadata_path = tmp_path / "metadata.json"
    submission.write_text(json.dumps(_official_core_payload()), encoding="utf-8")
    metadata_path.write_text(json.dumps(_valid_metadata()), encoding="utf-8")
    monkeypatch.setattr(scorer, "Adaptor", lambda: _StubAdaptor("failed"))

    exit_code = main(
        [
            str(submission),
            "--track",
            "core-all-single",
            "--metadata",
            str(metadata_path),
            "--out",
            str(out),
        ],
        _evaluation_hook=_StubAdaptor("failed").evaluate,
    )

    assert exit_code == 0
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["passed"] == 0
    assert row["pass_rate"] == 0.0


def test_default_main_path_routes_candidate_through_isolated_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps(_custom_payload(status="passed")), encoding="utf-8")
    calls: list[tuple[str, float]] = []

    def isolated(task: QCEvalTask, code: str, *, timeout_seconds: float) -> dict[str, Any]:
        del task
        calls.append((code, timeout_seconds))
        return QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=True,
            verified_status="verified_pass",
        ).to_dict()

    monkeypatch.setattr(scorer, "evaluate_trusted_candidate", isolated)

    exit_code = main([str(submission), "--track", "custom", "--trusted-regrade-timeout", "7"])

    assert exit_code == 0
    assert calls == [("def grover_search_oracle_00(): pass", 7.0)]


def test_fresh_worker_prevents_candidate_from_mutating_parent_scorer_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Adaptor().load_tasks("qiskit", suite="core")[0]
    monkeypatch.setattr(scorer, "_INTEGRITY_SENTINEL", "parent", raising=False)
    code = "\n".join(
        [
            "import scripts.score_submission as scorer",
            "scorer._INTEGRITY_SENTINEL = 'candidate'",
        ]
    )

    evaluation = evaluate_trusted_candidate(
        task,
        code,
        timeout_seconds=10,
        _evaluation_hook=_exec_candidate_evaluation,
    )

    assert evaluation["passed"] is True
    assert scorer._INTEGRITY_SENTINEL == "parent"


def test_fresh_workers_do_not_share_candidate_patched_globals() -> None:
    task = Adaptor().load_tasks("qiskit", suite="core")[0]
    first = "\n".join(
        [
            "import builtins",
            "builtins._QCEVAL_CANDIDATE_PATCH = True",
        ]
    )
    second = "\n".join(
        [
            "import builtins",
            "if getattr(builtins, '_QCEVAL_CANDIDATE_PATCH', False):",
            "    raise RuntimeError('candidate global leaked between workers')",
        ]
    )

    first_evaluation = evaluate_trusted_candidate(
        task,
        first,
        timeout_seconds=10,
        _evaluation_hook=_exec_candidate_evaluation,
    )
    second_evaluation = evaluate_trusted_candidate(
        task,
        second,
        timeout_seconds=10,
        _evaluation_hook=_exec_candidate_evaluation,
    )

    assert first_evaluation["passed"] is True
    assert second_evaluation["passed"] is True


def test_hanging_candidate_times_out_and_clean_rerun_succeeds() -> None:
    task = Adaptor().load_tasks("qiskit", suite="core")[0]

    with pytest.raises(TrustedRegradeTimeout, match="timed out"):
        evaluate_trusted_candidate(
            task,
            "while True:\n    pass\n",
            timeout_seconds=0.5,
            _evaluation_hook=_exec_candidate_evaluation,
        )

    evaluation = evaluate_trusted_candidate(
        task,
        "rerun_completed = True",
        timeout_seconds=10,
        _evaluation_hook=_exec_candidate_evaluation,
    )
    assert evaluation["passed"] is True


def test_trusted_timeout_is_rejected_as_requiring_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _custom_payload(status="passed")

    def time_out(*args: object, **kwargs: object) -> Mapping[str, Any]:
        del args, kwargs
        raise TrustedRegradeTimeout("trusted worker timed out after 0.100s")

    monkeypatch.setattr(scorer, "evaluate_trusted_candidate", time_out)

    errors = validate_submission(payload, track="custom", strict=False, adapter=_StubAdaptor())

    assert any("timed out" in error and "rerun required" in error for error in errors)


def test_validate_submission_rejects_tampered_summary() -> None:
    payload = _official_core_payload()
    payload["results"][0]["status"] = "failed"

    errors = _validate_official(payload)

    assert any("summary.passed" in error for error in errors)
    assert any("summary.pass_rate" in error for error in errors)


def test_validate_submission_rejects_status_contradicting_evaluation() -> None:
    payload = _official_core_payload()
    payload["results"][0]["evaluation"] = {"compiled": True, "ran": True, "passed": False}

    errors = _validate_official(payload)

    assert any("claims status 'passed' but its evaluation implies 'failed'" in error for error in errors)


def test_validate_submission_rejects_unverified_pass_counted_as_passed() -> None:
    payload = _official_core_payload()
    payload["results"][0]["evaluation"] = {
        "compiled": True,
        "ran": True,
        "passed": True,
        "verified_status": "unverified_pass",
    }

    errors = _validate_official(payload)

    assert any("summary.passed" in error for error in errors)


def test_recompute_summary_keeps_infrastructure_in_headline_assigned_denominator() -> None:
    payload = _official_core_payload()
    payload["results"][0]["status"] = "infrastructure_error"

    recomputed = recompute_summary(payload["results"])

    assert recomputed["total_tasks"] == len(payload["results"])
    assert recomputed["assigned_tasks"] == len(payload["results"])
    assert recomputed["scoreable_tasks"] == len(payload["results"]) - 1
    assert recomputed["passed"] == len(payload["results"]) - 1
    assert recomputed["pass_rate"] == (len(payload["results"]) - 1) / len(payload["results"])
    assert recomputed["pass_rate_denominator"] == "assigned_tasks"


def test_originless_candidate_execution_error_is_not_infrastructure() -> None:
    evaluation = {
        "compiled": True,
        "ran": True,
        "passed": False,
        "semantic_result": {
            "status": "execution_error",
            "diagnostics": [{"name": "framework_lowering", "value": "unsupported"}],
        },
    }

    assert scorer._derived_status(evaluation) == "failed"


@pytest.mark.parametrize(
    ("temperature", "expected_error"),
    [
        ({"value": 0.2, "source": "explicit"}, "exposed temperature 0.0"),
        ({"value": 0.0, "source": "explicit"}, None),
        ({"value": None, "source": "not_exposed"}, None),
    ],
)
def test_official_temperature_protocol(
    temperature: dict[str, Any],
    expected_error: str | None,
) -> None:
    payload = _official_core_payload()
    payload["summary"]["run_protocol"]["generation_parameters"]["temperature"] = temperature

    errors = _validate_official(payload)

    if expected_error is None:
        assert errors == []
    else:
        assert any(expected_error in error for error in errors)


def test_openrouter_can_claim_temperature_is_not_exposed_only_for_a_pinned_endpoint() -> None:
    payload = _official_core_payload()
    payload["provider"] = "openrouter"
    for record in payload["results"]:
        record["provider"] = "openrouter"
    generation = payload["summary"]["run_protocol"]["generation_parameters"]
    generation["temperature"] = {
        "value": None,
        "source": "not_exposed",
    }

    errors = _validate_official(payload)

    assert any("explicitly pinned endpoint" in error for error in errors)

    generation["endpoint_tag"] = {"value": "author/region", "source": "explicit"}
    assert _validate_official(payload) == []


def test_official_temperature_accepts_mixed_per_route_exposure() -> None:
    payload = _official_core_payload()
    generation = payload["summary"]["run_protocol"]["generation_parameters"]
    generation["temperature"] = {"value": [None, 0.0], "source": "per_record_route_provenance"}
    generation["endpoint_tag"] = {
        "value": ["author/region-a", "author/region-b"],
        "source": "per_record_route_provenance",
    }

    assert _validate_official(payload) == []


def test_official_result_must_match_bundled_entry_point() -> None:
    payload = _official_core_payload()
    payload["results"][0]["entry_point"] = "attacker_selected_entry_point"

    errors = _validate_official(payload)

    assert any("entry_point does not match bundled task" in error for error in errors)


def test_official_submission_requires_bundled_source_identity() -> None:
    payload = _official_core_payload()
    payload["qceval"] = {"source": "custom-adapter"}

    errors = _validate_official(payload)

    assert any("qceval.source='bundled-qceval'" in error for error in errors)


@pytest.mark.parametrize("code", [None, "", " \n\t "])
def test_null_or_blank_candidate_code_is_provider_failed(code: str | None) -> None:
    payload = _custom_payload(status="provider_failed")
    payload["results"][0]["provider_response"]["code"] = code

    def unexpected_evaluation(task: QCEvalTask, candidate: str) -> QCEvalEvaluation:
        del task, candidate
        raise AssertionError("blank provider output must not be evaluated")

    errors, trusted_results = scorer._validate_and_regrade(
        payload,
        track="custom",
        strict=False,
        metadata=None,
        adapter=_StubAdaptor(),
        _evaluation_hook=unexpected_evaluation,
    )

    assert errors == []
    assert trusted_results is not None
    assert trusted_results[0]["status"] == "provider_failed"
    assert trusted_results[0]["evaluation"] is None


def test_official_submission_package_version_must_match_trusted_scorer() -> None:
    payload = _official_core_payload()
    payload["qceval"]["package_version"] = "9.9.9"

    errors = _validate_official(payload)

    assert any("qceval.package_version" in error and "trusted scorer" in error for error in errors)


def test_official_submission_commit_must_match_trusted_scorer() -> None:
    payload = _official_core_payload()
    payload["qceval"]["commit"] = "b" * 40

    errors = _validate_official(payload)

    assert any("qceval.commit" in error and "trusted scorer" in error for error in errors)


def test_official_scoring_rejects_dirty_trusted_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _official_core_payload()
    adapter = _StubAdaptor()
    clean_metadata = adapter.metadata()
    monkeypatch.setattr(adapter, "metadata", lambda: {**clean_metadata, "dirty": True})

    errors = _validate_official(payload, adapter=adapter)

    assert any("requires a clean trusted QCircuitEval source checkout" in error for error in errors)


def test_wheel_scorer_row_marks_commit_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = tmp_path / "submission.json"
    out = tmp_path / "leaderboard-row.json"
    metadata_path = tmp_path / "metadata.json"
    submission.write_text(json.dumps(_official_core_payload()), encoding="utf-8")
    metadata_path.write_text(json.dumps(_valid_metadata()), encoding="utf-8")
    adapter = _StubAdaptor("passed")
    local_metadata = adapter.metadata()
    monkeypatch.setattr(adapter, "metadata", lambda: {**local_metadata, "commit": None, "dirty": None})
    monkeypatch.setattr(scorer, "Adaptor", lambda: adapter)

    exit_code = main(
        [
            str(submission),
            "--track",
            "core-all-single",
            "--metadata",
            str(metadata_path),
            "--out",
            str(out),
        ],
        _evaluation_hook=adapter.evaluate,
    )

    assert exit_code == 0
    trusted_identity = json.loads(out.read_text(encoding="utf-8"))["trusted_local_adapter"]
    assert trusted_identity["package_version"] == "1.2.3"
    assert trusted_identity["commit"] is None
    assert trusted_identity["commit_status"] == "unavailable"


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("submitter", "  ", "non-empty strings"),
        ("provider", "other", "metadata.provider"),
        ("model", "other", "metadata.model"),
        ("qceval_commit", "main", "40- or 64-character"),
        ("image", "qceval:latest", "@sha256"),
    ],
)
def test_strict_metadata_enforces_content_and_identity(
    field: str,
    value: str,
    expected_error: str,
) -> None:
    payload = _official_core_payload()
    metadata = _valid_metadata()
    metadata[field] = value

    errors = validate_submission(
        payload,
        track="core-all-single",
        strict=True,
        metadata=metadata,
        adapter=_StubAdaptor(),
    )

    assert any(expected_error in error for error in errors)


def test_official_provider_and_model_must_match_records() -> None:
    payload = _official_core_payload()
    payload["results"][0]["provider"] = "other"
    payload["results"][1]["model"] = "other"

    errors = _validate_official(payload)

    assert any("result 0.provider" in error for error in errors)
    assert any("result 1.model" in error for error in errors)


def test_infrastructure_error_is_structurally_valid_for_unsafe_custom_track() -> None:
    payload = _custom_payload(status="infrastructure_error")

    errors = validate_submission(
        payload,
        track="custom",
        strict=False,
        unsafe_structural_only=True,
    )

    assert errors == []


def test_official_infrastructure_error_requires_rerun() -> None:
    payload = _official_core_payload()
    payload["results"][0]["status"] = "infrastructure_error"
    payload["results"][0]["evaluation"] = {
        "compiled": False,
        "ran": False,
        "passed": False,
        "error_type": "InfrastructureError",
    }
    payload["summary"]["passed"] -= 1
    payload["summary"]["pass_rate"] = payload["summary"]["passed"] / len(payload["results"])

    errors = _validate_official(payload)

    assert any("official final acceptance requires a clean rerun" in error for error in errors)


def test_trusted_regrade_infrastructure_error_requires_rerun() -> None:
    payload = _custom_payload(status="passed")

    errors = validate_submission(
        payload,
        track="custom",
        strict=False,
        adapter=_StubAdaptor("infrastructure_error"),
        _evaluation_hook=_StubAdaptor("infrastructure_error").evaluate,
    )

    assert any("trusted regrade hit infrastructure_error" in error for error in errors)


def test_unsafe_structural_only_cannot_accept_official_track() -> None:
    payload = _official_core_payload()

    errors = validate_submission(
        payload,
        track="core-all-single",
        strict=False,
        unsafe_structural_only=True,
    )

    assert errors == ["--unsafe-structural-only is restricted to the custom track"]


def test_custom_unsafe_main_labels_output(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.json"
    out = tmp_path / "leaderboard-row.json"
    submission.write_text(json.dumps(_custom_payload(status="passed")), encoding="utf-8")

    exit_code = main(
        [
            str(submission),
            "--track",
            "custom",
            "--unsafe-structural-only",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["trusted_regrade"] is False
    assert row["validation_mode"] == "unsafe_custom_structural_only"


def _valid_metadata() -> dict[str, str]:
    metadata = dict.fromkeys(REQUIRED_METADATA_FIELDS, "value")
    metadata.update(
        {
            "provider": "smoke",
            "model": "smoke-canonical",
            "qceval_commit": "a" * 40,
            "image": f"qceval:0.1.0@sha256:{'b' * 64}",
        }
    )
    return metadata


def _custom_payload(*, status: str) -> dict[str, Any]:
    passed = 1 if status == "passed" else 0
    record: dict[str, Any] = {
        "suite": "core",
        "framework": "qiskit",
        "task_id": "01",
        "sample_index": 0,
        "attempt_index": 0,
        "entry_point": "grover_search_oracle_00",
        "provider": "smoke",
        "model": "smoke-canonical",
        "status": status,
        "provider_response": {
            "code": "def grover_search_oracle_00(): pass",
            "model": "smoke-canonical",
        },
    }
    if status == "infrastructure_error":
        record["evaluation"] = {
            "compiled": False,
            "ran": False,
            "passed": False,
            "error_type": "InfrastructureError",
        }
    return {
        "schema_version": "qceval.run.v2",
        "provider": "smoke",
        "model": "smoke-canonical",
        "results": [record],
        "summary": {
            "total_tasks": 1,
            "passed": passed,
            "pass_rate": float(passed),
            "run_protocol": {},
        },
    }


def _official_core_payload() -> dict[str, Any]:
    adapter = Adaptor()
    results: list[dict[str, Any]] = []
    for framework in DEFAULT_FRAMEWORKS:
        for task in adapter.load_tasks(framework, suite="core"):
            results.append(
                {
                    "suite": "core",
                    "framework": framework,
                    "task_id": task.task_id,
                    "sample_index": 0,
                    "attempt_index": 0,
                    "entry_point": task.entry_point,
                    "provider": "smoke",
                    "model": "smoke-canonical",
                    "status": "passed",
                    "provider_response": {
                        "code": "arbitrary candidate code",
                        "model": "smoke-canonical",
                    },
                }
            )
    return {
        "schema_version": "qceval.run.v2",
        "provider": "smoke",
        "model": "smoke-canonical",
        "suites": ["core"],
        "qceval": {
            "source": "bundled-qceval",
            "package_version": "1.2.3",
            "commit": "a" * 40,
        },
        "results": results,
        "summary": {
            "total_tasks": len(results),
            "passed": len(results),
            "failed": 0,
            "provider_failures": 0,
            "compile_failures": 0,
            "run_failures": 0,
            "infrastructure_failures": 0,
            "pass_rate": 1.0,
            "run_protocol": {
                "samples_per_task": 1,
                "pass_k": 1,
                "max_attempts": 1,
                "feedback_enabled": False,
                "generation_parameters": {
                    "temperature": {"value": 0.0, "source": "explicit"},
                },
            },
        },
    }
