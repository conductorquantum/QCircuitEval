"""Bounded semantic result writer and historical reader."""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from typing import Any

from qceval.semantics.contracts import Contract
from qceval.semantics.ir import IR_VERSION
from qceval.semantics.verifiers.result import SemanticStatus, VerifierResult

RESULT_RECORD_SCHEMA_VERSION = "3"
SUPPORTED_RESULT_RECORD_SCHEMAS = frozenset({"1", "2", RESULT_RECORD_SCHEMA_VERSION})
MAX_EVIDENCE_RECORDS = 64
MAX_TEXT = 500


def make_result_record(
    result: VerifierResult,
    contract: Contract,
    *,
    framework: str,
    authoritative: bool,
) -> dict[str, Any]:
    """Write one bounded newest-schema semantic result.

    Args:
        result: Internal verifier result.
        contract: Contract used for verification.
        framework: Candidate framework name.
        authoritative: Whether this result controlled the score.

    Returns:
        JSON-compatible result record.
    """
    evidence = result.evidence[:MAX_EVIDENCE_RECORDS]
    input_hash = evidence[0].input_hash if evidence else None
    metrics = [item for item in evidence if item.metric is not None]
    elapsed = sum(item.elapsed_seconds for item in evidence)
    peak_values = [item.peak_rss_mib for item in evidence if item.peak_rss_mib is not None]
    return {
        "result_schema_version": RESULT_RECORD_SCHEMA_VERSION,
        "status": result.status.value,
        "reason_code": _bounded(result.reason),
        "summary": _bounded(result.reason.replace("_", " ")),
        "passed": result.passed,
        "authoritative": authoritative,
        "contract": {
            "suite": contract.suite,
            "task_id": contract.task_id,
            "schema_version": contract.schema_version,
            "contract_version": contract.contract_version,
            "hash": result.contract_hash,
        },
        "target": {"version": contract.target.version, "hash": result.target_hash},
        "ir": {"version": IR_VERSION, "semantic_hash": input_hash},
        "verifier": {
            "release_version": result.verifier_version,
            "engines": [{"name": item.engine, "version": item.engine_version} for item in evidence],
            "metric": None if not metrics else metrics[0].metric,
            "tolerance": None if not metrics else metrics[0].tolerance,
        },
        "evidence": [_evidence_record(item) for item in evidence],
        "requirements": [],
        "diagnostics": [{"name": key, "value": _bounded(value)} for key, value in result.diagnostics[:64]],
        "resources": {
            "wall_seconds": elapsed,
            "peak_rss_mib": max(peak_values) if peak_values else None,
            "evidence_truncated": len(result.evidence) > MAX_EVIDENCE_RECORDS,
        },
        "environment": {
            "python": platform.python_version(),
            "framework": framework,
            "platform": sys.platform,
        },
    }


def make_execution_error_result_record(
    *,
    suite: str,
    task_id: str,
    framework: str,
    reason: str,
) -> dict[str, Any]:
    """Write an authoritative execution-error record for a contractless task.

    Args:
        suite: Benchmark suite name.
        task_id: Task identifier within the suite.
        framework: Candidate framework name.
        reason: Stable reason code.

    Returns:
        JSON-compatible result record.
    """
    bounded_reason = _bounded(reason)
    return {
        "result_schema_version": RESULT_RECORD_SCHEMA_VERSION,
        "status": SemanticStatus.EXECUTION_ERROR.value,
        "reason_code": bounded_reason,
        "summary": _bounded(reason.replace("_", " ")),
        "passed": False,
        "authoritative": True,
        "contract": {
            "suite": suite,
            "task_id": task_id,
            "schema_version": "unavailable",
            "contract_version": "unavailable",
            "hash": "unavailable",
        },
        "target": {"version": "unavailable", "hash": "unavailable"},
        "ir": {"version": IR_VERSION, "semantic_hash": None},
        "verifier": {
            "release_version": "integration-1.0.0",
            "engines": [],
            "metric": None,
            "tolerance": None,
        },
        "evidence": [],
        "requirements": [],
        "diagnostics": [{"name": "contract_registry", "value": "unavailable"}],
        "resources": {"wall_seconds": 0.0, "peak_rss_mib": None, "evidence_truncated": False},
        "environment": {
            "python": platform.python_version(),
            "framework": framework,
            "platform": sys.platform,
        },
    }


def read_result_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read a supported result record into the newest schema.

    Args:
        payload: Historical or current result object.

    Returns:
        Validated result dictionary.

    Raises:
        ValueError: If the schema or semantic status is unsupported.
    """
    version = str(payload.get("result_schema_version", payload.get("schema_version", "")))
    if version not in SUPPORTED_RESULT_RECORD_SCHEMAS:
        raise ValueError(f"unsupported semantic result schema {version!r}")
    try:
        value = _migrate_v1(payload) if version == "1" else dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("semantic historical result is malformed") from exc
    value = _normalize_historical_status(value)
    value["result_schema_version"] = RESULT_RECORD_SCHEMA_VERSION
    value.pop("migration_state", None)
    try:
        status = SemanticStatus(value["status"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("semantic result has invalid status") from exc
    if bool(value.get("passed")) is not (status is SemanticStatus.VERIFIED_PASS):
        raise ValueError("semantic passed projection disagrees with status")
    _validate_v3(value)
    return value


def _migrate_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_status = str(payload["status"])
    status = _current_status(raw_status)
    reason = _bounded(str(payload.get("reason", "historical_result")))
    return {
        "result_schema_version": RESULT_RECORD_SCHEMA_VERSION,
        "status": status.value,
        "reason_code": reason,
        "summary": reason.replace("_", " "),
        "passed": status is SemanticStatus.VERIFIED_PASS,
        "authoritative": False,
        "contract": {
            "suite": str(payload.get("suite", "unknown")),
            "task_id": str(payload.get("task_id", "unknown")),
            "schema_version": str(payload.get("contract_schema_version", "unknown")),
            "contract_version": str(payload.get("contract_version", "unknown")),
            "hash": str(payload.get("contract_hash", "unknown")),
        },
        "target": {
            "version": str(payload.get("target_version", "unknown")),
            "hash": str(payload.get("target_hash", "unknown")),
        },
        "ir": {"version": str(payload.get("ir_version", "unknown")), "semantic_hash": payload.get("input_hash")},
        "verifier": {
            "release_version": str(payload.get("verifier_version", "unknown")),
            "engines": [],
            "metric": None,
            "tolerance": None,
        },
        "evidence": [],
        "requirements": [],
        "diagnostics": [
            {"name": "migrated_from", "value": "1"},
            *([{"name": "historical_status", "value": raw_status}] if raw_status != status.value else []),
        ],
        "resources": {"wall_seconds": 0.0, "peak_rss_mib": None, "evidence_truncated": False},
        "environment": {"python": "unknown", "framework": "unknown", "platform": "unknown"},
    }


def _normalize_historical_status(value: dict[str, Any]) -> dict[str, Any]:
    raw_status = str(value.get("status", ""))
    status = _current_status(raw_status)
    value["status"] = status.value
    value["passed"] = status is SemanticStatus.VERIFIED_PASS
    if raw_status != status.value:
        diagnostics = value.setdefault("diagnostics", [])
        if not isinstance(diagnostics, list):
            raise ValueError("semantic diagnostics must be an array")
        diagnostics.append({"name": "historical_status", "value": raw_status})
    return value


def _current_status(raw_status: str) -> SemanticStatus:
    if raw_status in {"unsupported", "inconclusive"}:
        return SemanticStatus.EXECUTION_ERROR
    return SemanticStatus(raw_status)


def _validate_v3(value: Mapping[str, Any]) -> None:
    required = {
        "result_schema_version",
        "status",
        "reason_code",
        "summary",
        "passed",
        "authoritative",
        "contract",
        "target",
        "ir",
        "verifier",
        "evidence",
        "requirements",
        "diagnostics",
        "resources",
        "environment",
    }
    if set(value) != required:
        raise ValueError("semantic result v3 keys differ from schema")
    if not isinstance(value["authoritative"], bool):
        raise ValueError("semantic authoritative must be boolean")
    for key in ("contract", "target", "ir", "verifier", "resources", "environment"):
        if not isinstance(value[key], Mapping):
            raise ValueError(f"semantic result {key} must be an object")
    for key in ("evidence", "requirements", "diagnostics"):
        if not isinstance(value[key], list):
            raise ValueError(f"semantic result {key} must be an array")
    if len(value["evidence"]) > MAX_EVIDENCE_RECORDS:
        raise ValueError("semantic evidence exceeds record limit")


def _evidence_record(item: Any) -> dict[str, Any]:
    return {
        "engine": item.engine,
        "engine_version": item.engine_version,
        "reason_code": _bounded(item.reason),
        "input_hash": item.input_hash,
        "target_hash": item.target_hash,
        "metric": item.metric,
        "value": item.value,
        "tolerance": item.tolerance,
        "uncertainty": item.uncertainty,
        "cases_checked": item.cases_checked,
        "elapsed_seconds": item.elapsed_seconds,
        "peak_rss_mib": item.peak_rss_mib,
        "preconditions": [_bounded(value) for value in item.preconditions[:32]],
    }


def _bounded(value: str) -> str:
    return value[:MAX_TEXT]
