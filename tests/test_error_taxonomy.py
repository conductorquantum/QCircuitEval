from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from qceval.core.runner.records import _record_from_dict, _status
from qceval.error_taxonomy import ERROR_AXES, classify_error_taxonomy
from qceval.evals.evaluator import build_evaluator
from qceval.evals.models import ExecutionResult
from qceval.models import BenchmarkRecord, ProviderResponse, QCEvalEvaluation
from qceval.reporting.error_taxonomy import error_taxonomy_summary
from qceval.semantics.portfolio import _reconcile_parameter_cases
from qceval.semantics.verifiers.result import RESULT_SCHEMA_VERSION, EvidenceRecord, SemanticStatus, VerifierResult


def _semantic(
    status: str,
    reason: str,
    *,
    evidence: list[dict[str, Any]] | None = None,
    diagnostics: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason,
        "evidence": evidence or [],
        "diagnostics": diagnostics or [],
    }


def _evaluation(semantic: dict[str, Any]) -> dict[str, Any]:
    return {"semantic_result": semantic}


def _evidence(reason: str, *preconditions: str) -> dict[str, Any]:
    return {"reason_code": reason, "preconditions": list(preconditions)}


@pytest.mark.parametrize(
    ("reason", "axis"),
    [
        ("terminal_observation_mismatch", "interface_observation_validity"),
        ("requirement_failed:min_num_qubits", "construction_resource_fidelity"),
        ("requirement_failed:required_interactions", "interaction_lifecycle_fidelity"),
        ("requirement_failed:forbid_state_preparation", "shortcut_provenance_violation"),
        ("metric_exceeds_fail_bound", "behavioral_target_mismatch"),
    ],
)
def test_decisive_reasons_map_to_operational_axes(reason: str, axis: str) -> None:
    taxonomy = classify_error_taxonomy(
        "failed",
        _evaluation(_semantic("semantic_fail", reason, evidence=[_evidence(reason)])),
    )

    assert taxonomy["outcome"] == "observed_error"
    assert taxonomy["axes"] == [axis]
    assert taxonomy["unclassified_reason_codes"] == []


@pytest.mark.parametrize("status", ["provider_failed", "compile_failed", "run_failed"])
def test_generation_and_execution_failures_use_first_axis(status: str) -> None:
    taxonomy = classify_error_taxonomy(status, None)

    assert taxonomy["outcome"] == "observed_error"
    assert taxonomy["axes"] == ["generation_execution_reliability"]
    assert taxonomy["reason_codes"] == [f"benchmark_status:{status}"]


def test_dense_gate_violation_is_multi_label() -> None:
    reason = "forbidden_gate_family:unitarygate"
    taxonomy = classify_error_taxonomy(
        "failed",
        _evaluation(_semantic("semantic_fail", reason, evidence=[_evidence(reason)])),
    )

    assert taxonomy["axes"] == [
        "construction_resource_fidelity",
        "shortcut_provenance_violation",
    ]


def test_parameter_robustness_requires_pass_and_semantic_fail_cases() -> None:
    evidence = [
        _evidence("metric_within_pass_bound", "case_index=0", "case_status=verified_pass"),
        _evidence("metric_exceeds_fail_bound", "case_index=1", "case_status=semantic_fail"),
    ]
    diagnostics = [
        {"name": "parameter_case_count:verified_pass", "value": "1"},
        {"name": "parameter_case_count:semantic_fail", "value": "1"},
    ]
    taxonomy = classify_error_taxonomy(
        "failed",
        _evaluation(
            _semantic("semantic_fail", "parameter_domain_semantic_fail", evidence=evidence, diagnostics=diagnostics)
        ),
    )

    assert taxonomy["axes"] == ["behavioral_target_mismatch", "parameter_domain_robustness"]
    assert taxonomy["parameter_case_status_counts"] == {
        "verified_pass": 1,
        "semantic_fail": 1,
        "execution_error": 0,
        "resource_limit": 0,
    }


def test_parameter_robustness_does_not_label_all_failing_domain() -> None:
    evidence = [
        _evidence("metric_exceeds_fail_bound", "case_index=0", "case_status=semantic_fail"),
        _evidence("metric_exceeds_fail_bound", "case_index=1", "case_status=semantic_fail"),
    ]
    diagnostics = [
        {"name": "parameter_case_count:verified_pass", "value": "0"},
        {"name": "parameter_case_count:semantic_fail", "value": "2"},
    ]
    taxonomy = classify_error_taxonomy(
        "failed",
        _evaluation(
            _semantic("semantic_fail", "parameter_domain_semantic_fail", evidence=evidence, diagnostics=diagnostics)
        ),
    )

    assert taxonomy["axes"] == ["behavioral_target_mismatch"]
    assert taxonomy["parameter_case_status_counts"]["semantic_fail"] == 2


def test_execution_error_distinguishes_candidate_from_grader() -> None:
    grader = classify_error_taxonomy(
        "failed",
        _evaluation(
            _semantic(
                "execution_error",
                "semantic_verifier_exception:RuntimeError",
                diagnostics=[{"name": "failure_origin", "value": "grader_verification"}],
            )
        ),
    )
    candidate = classify_error_taxonomy(
        "failed",
        _evaluation(
            _semantic(
                "execution_error",
                "candidate_execution_exception:ValueError",
                diagnostics=[{"name": "failure_origin", "value": "candidate_execution"}],
            )
        ),
    )

    assert grader["outcome"] == "grader_nondecision"
    assert grader["axes"] == []
    assert candidate["outcome"] == "observed_error"
    assert candidate["axes"] == ["generation_execution_reliability"]


@pytest.mark.parametrize(
    ("semantic_status", "reason", "outcome"),
    [
        ("execution_error", "framework_inspection_failed", "observed_error"),
        ("execution_error", "symbolic_worker_failure", "observed_error"),
        ("resource_limit", "symbolic_worker_resource_limit", "resource_limit"),
    ],
)
def test_originless_post_execution_failures_are_model_attributable(
    semantic_status: str,
    reason: str,
    outcome: str,
) -> None:
    evaluation = QCEvalEvaluation(
        compiled=True,
        ran=True,
        passed=False,
        semantic_result=_semantic(semantic_status, reason),
    )
    record = _evaluated_record(evaluation)

    taxonomy = record.to_dict()["error_taxonomy"]

    assert record.status == "failed"
    assert taxonomy["outcome"] == outcome
    assert taxonomy["axes"] == ["generation_execution_reliability"]
    assert taxonomy["reason_codes"] == [reason]


@pytest.mark.parametrize(
    "evaluation",
    [
        QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=False,
            semantic_result=_semantic(
                "execution_error",
                "semantic_verifier_exception:RuntimeError",
                diagnostics=[{"name": "failure_origin", "value": "grader_verification"}],
            ),
        ),
        QCEvalEvaluation(
            compiled=False,
            ran=False,
            passed=False,
            error_type="InfrastructureError",
        ),
    ],
)
def test_explicit_grader_fault_record_and_taxonomy_remain_infrastructure(
    evaluation: QCEvalEvaluation,
) -> None:
    record = _evaluated_record(evaluation)

    taxonomy = record.to_dict()["error_taxonomy"]

    assert record.status == "infrastructure_error"
    assert taxonomy["outcome"] == "grader_nondecision"
    assert taxonomy["axes"] == []


def test_evaluator_records_grader_exception_origin() -> None:
    evaluator = build_evaluator("qiskit", semantic_verifier=_RaisingVerifier())

    details = evaluator.grade_execution(
        task_id="01",
        execution=ExecutionResult(probabilities=[], metadata={}),
        code="",
    )

    semantic = details["semantic_verification"]
    assert semantic["reason_code"] == "semantic_verifier_exception:RuntimeError"
    assert {item["name"]: item["value"] for item in semantic["diagnostics"]}["failure_origin"] == (
        "grader_verification"
    )


def test_evaluator_records_candidate_replay_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = build_evaluator("qiskit", semantic_verifier=_RaisingVerifier())

    def raise_candidate_error(*args: Any) -> None:
        del args
        raise RuntimeError("candidate detail must not enter the record")

    monkeypatch.setattr(evaluator, "_semantic_execution_cases", raise_candidate_error)
    details = evaluator.grade_execution(
        task_id="01",
        execution=ExecutionResult(probabilities=[], metadata={}),
        code="",
    )

    semantic = details["semantic_verification"]
    assert semantic["reason_code"] == "candidate_execution_exception:RuntimeError"
    assert {item["name"]: item["value"] for item in semantic["diagnostics"]}["failure_origin"] == (
        "candidate_execution"
    )


def test_unknown_decisive_reason_remains_visible() -> None:
    reason = "new_decisive_failure"
    taxonomy = classify_error_taxonomy(
        "failed",
        _evaluation(_semantic("semantic_fail", reason, evidence=[_evidence(reason)])),
    )

    assert taxonomy["outcome"] == "observed_error"
    assert taxonomy["axes"] == []
    assert taxonomy["unclassified_reason_codes"] == [reason]


def test_invalid_contract_reason_is_not_an_observed_error() -> None:
    reason = "requirement_failed:invalid_interaction_contract"
    taxonomy = classify_error_taxonomy(
        "failed",
        _evaluation(_semantic("semantic_fail", reason, evidence=[_evidence(reason)])),
    )

    assert taxonomy["outcome"] == "grader_nondecision"
    assert taxonomy["axes"] == []
    assert taxonomy["grader_reason_codes"] == [reason]


def test_summary_uses_common_denominator_and_allows_overlap() -> None:
    records = [
        _record("compile_failed"),
        _record(
            "failed",
            _semantic(
                "semantic_fail",
                "forbidden_gate_family:unitarygate",
                evidence=[_evidence("forbidden_gate_family:unitarygate")],
            ),
        ),
        _record(
            "failed",
            _semantic(
                "semantic_fail",
                "metric_exceeds_fail_bound",
                evidence=[_evidence("metric_exceeds_fail_bound")],
            ),
        ),
    ]

    summary = error_taxonomy_summary(records)

    assert summary["assigned"] == 3
    assert summary["denominator"] == "all_assigned_records_in_stratum"
    assert summary["axis_labels"]["parameter_domain_robustness"] == "Parameter robustness"
    assert summary["axis_rates"]["generation_execution_reliability"] == 1 / 3
    assert summary["axis_rates"]["construction_resource_fidelity"] == 1 / 3
    assert summary["axis_rates"]["shortcut_provenance_violation"] == 1 / 3
    assert summary["axis_rates"]["behavioral_target_mismatch"] == 1 / 3
    assert sum(summary["axis_counts"].values()) == 4
    assert summary["observed_error_records"] == 3
    assert summary["by_suite"]["core"]["assigned"] == 3
    assert summary["by_suite_framework"]["core"]["qiskit"]["assigned"] == 3
    assert all(0 <= summary["axis_counts"][axis] <= summary["assigned"] for axis in ERROR_AXES)
    assert all(summary["axis_rates"][axis] == summary["axis_counts"][axis] / summary["assigned"] for axis in ERROR_AXES)
    assert summary["classification_coverage"] == 1.0


def test_record_serialization_and_resume_preserve_taxonomy() -> None:
    record = _record(
        "failed",
        _semantic(
            "semantic_fail",
            "metric_exceeds_fail_bound",
            evidence=[_evidence("metric_exceeds_fail_bound")],
        ),
    )

    payload = record.to_dict()
    resumed = _record_from_dict(payload)

    assert payload["error_taxonomy"]["axes"] == ["behavioral_target_mismatch"]
    assert resumed.error_taxonomy == payload["error_taxonomy"]
    assert resumed.to_dict()["error_taxonomy"] == payload["error_taxonomy"]


def test_summary_rejects_incompatible_taxonomy_versions() -> None:
    record = replace(
        _record("failed"),
        error_taxonomy={
            "taxonomy_version": "2",
            "multi_label": True,
            "outcome": "observed_error",
            "axes": [],
        },
    )

    with pytest.raises(ValueError, match="stratify records"):
        error_taxonomy_summary([record])


def test_parameter_reconciliation_records_each_case_status_once() -> None:
    passed = _verifier_result(SemanticStatus.VERIFIED_PASS, "metric_within_pass_bound")
    failed = _verifier_result(SemanticStatus.SEMANTIC_FAIL, "metric_exceeds_fail_bound")

    result = _reconcile_parameter_cases((((0,), passed), ((1,), failed)))

    assert result.status is SemanticStatus.SEMANTIC_FAIL
    assert result.diagnostics == (
        ("parameter_case_count:verified_pass", "1"),
        ("parameter_case_count:semantic_fail", "1"),
        ("parameter_case_count:execution_error", "0"),
        ("parameter_case_count:resource_limit", "0"),
    )
    assert "case_index=0" in result.evidence[0].preconditions
    assert "case_status=verified_pass" in result.evidence[0].preconditions
    assert "case_index=1" in result.evidence[1].preconditions
    assert "case_status=semantic_fail" in result.evidence[1].preconditions


def _record(status: str, semantic: dict[str, Any] | None = None) -> BenchmarkRecord:
    evaluation = None
    if status != "provider_failed":
        evaluation = QCEvalEvaluation(
            compiled=status != "compile_failed",
            ran=status not in {"compile_failed", "run_failed"},
            passed=status == "passed",
            semantic_result=semantic,
        )
    return BenchmarkRecord(
        framework="qiskit",
        task_id="01",
        entry_point="answer",
        category=None,
        provider="smoke",
        model="m",
        status=status,  # type: ignore[arg-type]
        provider_response=ProviderResponse(code="code", model="m"),
        evaluation=evaluation,
    )


def _evaluated_record(evaluation: QCEvalEvaluation) -> BenchmarkRecord:
    return BenchmarkRecord(
        framework="qiskit",
        task_id="01",
        entry_point="answer",
        category=None,
        provider="smoke",
        model="m",
        status=_status(evaluation),
        provider_response=ProviderResponse(code="code", model="m"),
        evaluation=evaluation,
    )


def _verifier_result(status: SemanticStatus, reason: str) -> VerifierResult:
    evidence = EvidenceRecord("test", "1", reason, "input", "target")
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        reason,
        "contract",
        "target",
        "1",
        (evidence,),
    )


class _RaisingVerifier:
    def verify(self, request: Any) -> VerifierResult:
        del request
        raise RuntimeError("grader detail must not enter the record")
