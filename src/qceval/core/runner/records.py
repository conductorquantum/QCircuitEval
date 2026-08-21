"""Runner result conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from qceval.core.runner.types import RunJob
from qceval.models import (
    BenchmarkRecord,
    Framework,
    OutcomeStatus,
    ProviderResponse,
    QCEvalEvaluation,
    Suite,
    TokenUsage,
)


def _status(evaluation: QCEvalEvaluation) -> OutcomeStatus:
    if _is_infrastructure_evaluation(evaluation):
        return "infrastructure_error"
    semantic = evaluation.semantic_result
    if isinstance(semantic, Mapping) and semantic.get("status") == "resource_limit":
        return "failed"
    if not evaluation.compiled:
        return "compile_failed"
    if not evaluation.ran:
        return "run_failed"
    return "passed" if evaluation.passed else "failed"


def _is_infrastructure_evaluation(evaluation: QCEvalEvaluation) -> bool:
    if evaluation.error_type == "InfrastructureError":
        return True
    semantic = evaluation.semantic_result
    if not isinstance(semantic, Mapping) or semantic.get("status") not in {"execution_error", "resource_limit"}:
        return False
    diagnostics = semantic.get("diagnostics")
    if not isinstance(diagnostics, list):
        return False
    for item in diagnostics:
        if isinstance(item, Mapping) and item.get("name") == "failure_origin":
            return item.get("value") == "grader_verification"
    return False


def _framework_from_str(value: str) -> Framework:
    if value not in {"qiskit", "cirq", "pennylane", "cudaq"}:
        raise ValueError(f"unknown framework in completed results: {value}")
    return cast(Framework, value)


def _suite_from_str(value: str) -> Suite:
    if value not in {"core", "qec"}:
        raise ValueError(f"unknown suite in completed results: {value}")
    return cast(Suite, value)


def _record_from_dict(payload: Mapping[str, Any]) -> BenchmarkRecord:
    evaluation_payload = payload.get("evaluation")
    evaluation = None if evaluation_payload is None else _evaluation_from_dict(evaluation_payload)
    taxonomy_payload = payload.get("error_taxonomy")
    error_taxonomy = taxonomy_payload if isinstance(taxonomy_payload, Mapping) else None
    return BenchmarkRecord(
        framework=payload["framework"],
        suite=payload.get("suite", "core"),
        task_id=str(payload["task_id"]),
        sample_index=int(payload.get("sample_index", 0)),
        attempt_index=int(payload.get("attempt_index", 0)),
        entry_point=str(payload["entry_point"]),
        category=None if payload.get("category") is None else str(payload.get("category")),
        provider=str(payload["provider"]),
        model=None if payload.get("model") is None else str(payload.get("model")),
        status=payload["status"],
        feedback=payload.get("feedback") or {},
        request_trace=payload.get("request_trace") or {},
        lineage=payload.get("lineage") or {},
        provider_response=_provider_response_from_dict(payload["provider_response"]),
        evaluation=evaluation,
        error_taxonomy=error_taxonomy,
    )


def _provider_response_from_dict(payload: Mapping[str, Any]) -> ProviderResponse:
    usage_payload = payload.get("usage")
    usage = None if usage_payload is None else TokenUsage(**usage_payload)
    return ProviderResponse(
        code=None if payload.get("code") is None else str(payload.get("code")),
        model=None if payload.get("model") is None else str(payload.get("model")),
        metadata=payload.get("metadata") or {},
        usage=usage,
        raw_response=payload.get("raw_response"),
        error=None if payload.get("error") is None else str(payload.get("error")),
    )


def _evaluation_from_dict(payload: Mapping[str, Any]) -> QCEvalEvaluation:
    grader_details = payload.get("grader_details") or {}
    if "verified_status" in payload:
        verified_status = payload.get("verified_status")
    else:
        verified_status = grader_details.get("verified_status")
    if verified_status is None and "verified_status" not in payload:
        verified_status = "unverified_pass" if bool(payload["passed"]) else "unverified_fail"
    return QCEvalEvaluation(
        compiled=bool(payload["compiled"]),
        ran=bool(payload["ran"]),
        passed=bool(payload["passed"]),
        metric=payload.get("metric"),
        metric_name=None if payload.get("metric_name") is None else str(payload["metric_name"]),
        probabilities=payload.get("probabilities"),
        execution_metadata=payload.get("execution_metadata") or {},
        grader_details=grader_details,
        verified_status=None if verified_status is None else str(verified_status),
        semantic_result=payload.get("semantic_result"),
        error=None if payload.get("error") is None else str(payload.get("error")),
        error_type=None if payload.get("error_type") is None else str(payload.get("error_type")),
    )


def _runtime_error_evaluation(error: str) -> QCEvalEvaluation:
    return QCEvalEvaluation(
        compiled=False,
        ran=False,
        passed=False,
        verified_status="execution_error",
        error=error,
        error_type="InfrastructureError",
    )


def _timeout_evaluation(timeout: float) -> QCEvalEvaluation:
    return QCEvalEvaluation(
        # The outer evaluator deadline can expire during framework execution
        # or semantic verification. It is not evidence of a syntax/compile
        # failure, so keep the record runtime-shaped.
        compiled=True,
        ran=False,
        passed=False,
        verified_status="resource_limit",
        semantic_result={
            "status": "resource_limit",
            "reason": "evaluation_timeout",
            "diagnostics": [
                {"name": "failure_origin", "value": "candidate_execution"},
                {"name": "timeout_seconds", "value": f"{timeout:.3f}"},
            ],
        },
        error=f"evaluation timed out after {timeout:.3f}s",
        error_type="EvaluationTimeout",
    )


def _job_key(job: RunJob) -> tuple[Suite, Framework, str, int, int]:
    return (job.task.suite, job.task.framework, job.task.task_id, job.sample_index, job.attempt_index)


def _chain_key(job: RunJob) -> tuple[Suite, Framework, str, int]:
    return (job.task.suite, job.task.framework, job.task.task_id, job.sample_index)


def _completed_chain(
    completed: Mapping[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
    job: RunJob,
) -> dict[int, BenchmarkRecord]:
    suite, framework, task_id, sample_index = _chain_key(job)
    return {
        attempt_index: record
        for (
            record_suite,
            record_framework,
            record_task_id,
            record_sample_index,
            attempt_index,
        ), record in completed.items()
        if (
            record_suite,
            record_framework,
            record_task_id,
            record_sample_index,
        )
        == (suite, framework, task_id, sample_index)
    }


def _chain_records(
    records: Mapping[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
    job: RunJob,
) -> list[BenchmarkRecord]:
    return [record for _, record in sorted(_completed_chain(records, job).items())]


def _feedback_record_order(
    record: BenchmarkRecord,
    task_order: Mapping[tuple[Suite, Framework, str, int], int],
) -> tuple[int, int]:
    return (
        task_order.get((record.suite, record.framework, record.task_id, record.sample_index), 10**9),
        record.attempt_index,
    )
