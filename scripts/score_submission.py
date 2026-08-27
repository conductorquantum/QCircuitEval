#!/usr/bin/env python3
"""Validate and score QCircuitEval leaderboard submissions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from qceval.core.bench import DEFAULT_FRAMEWORKS, Adaptor
from qceval.core.runner.trusted import (
    DEFAULT_TRUSTED_REGRADE_TIMEOUT_SECONDS,
    TrustedRegradeError,
    TrustedRegradeTimeout,
    evaluate_trusted_candidate,
)
from qceval.models import Framework, QCEvalEvaluation, QCEvalTask, Suite
from qceval.reports import format_run_summary
from qceval.typing import TaskAdapter

Track = Literal["core-all-single", "qec-all-single", "full-all-single", "custom"]
_EvaluationHook = Callable[[QCEvalTask, str], QCEvalEvaluation | Mapping[str, Any]]

TRACKS: dict[Track, tuple[tuple[Suite, ...], tuple[Framework, ...]]] = {
    "core-all-single": (("core",), DEFAULT_FRAMEWORKS),
    "qec-all-single": (("qec",), DEFAULT_FRAMEWORKS),
    "full-all-single": (("core", "qec"), DEFAULT_FRAMEWORKS),
    "custom": ((), ()),
}

VALID_STATUSES = {
    "passed",
    "failed",
    "compile_failed",
    "run_failed",
    "provider_failed",
    "infrastructure_error",
}
REQUIRED_RESULT_FIELDS = {
    "suite",
    "framework",
    "task_id",
    "sample_index",
    "attempt_index",
    "entry_point",
    "provider",
    "model",
    "status",
    "provider_response",
}
REQUIRED_METADATA_FIELDS = {
    "submitter",
    "provider",
    "model",
    "model_version_or_date",
    "endpoint",
    "date_utc",
    "qceval_commit",
    "image",
    "command",
    "allowed_tools",
    "retry_policy",
    "cache_policy",
    "disclosure",
}
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_IMAGE_DIGEST_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


def main(
    argv: Sequence[str] | None = None,
    *,
    _evaluation_hook: _EvaluationHook | None = None,
) -> int:
    """Validate a run output and print leaderboard-ready score.

    Args:
        argv: Optional command-line argument vector.

    Returns:
        Process exit code. ``0`` means valid, ``1`` means validation failed.
    """
    args = _parse_args(argv)
    payload = _load_payload(args.submission)
    metadata = None if args.metadata is None else _load_metadata(args.metadata)
    trusted_adapter = None if args.unsafe_structural_only else Adaptor()
    errors, trusted_results = _validate_and_regrade(
        payload,
        track=args.track,
        strict=args.strict,
        metadata=metadata,
        adapter=trusted_adapter,
        unsafe_structural_only=args.unsafe_structural_only,
        _evaluation_hook=_evaluation_hook,
        _trusted_regrade_timeout=args.trusted_regrade_timeout,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    assert trusted_results is not None
    trusted_summary = recompute_summary(trusted_results)
    print(format_run_summary(trusted_summary))
    leaderboard = _leaderboard_record(
        payload,
        track=args.track,
        metadata=metadata,
        results=trusted_results,
        trusted_regrade=not args.unsafe_structural_only,
        trusted_adapter_metadata=None if trusted_adapter is None else trusted_adapter.metadata(),
    )
    if args.out is not None:
        args.out.write_text(json.dumps(leaderboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(leaderboard, sort_keys=True))
    return 0


def validate_submission(
    payload: Mapping[str, Any],
    *,
    track: Track,
    strict: bool,
    metadata: Mapping[str, Any] | None = None,
    adapter: TaskAdapter | None = None,
    unsafe_structural_only: bool = False,
    _evaluation_hook: _EvaluationHook | None = None,
    _trusted_regrade_timeout: float = DEFAULT_TRUSTED_REGRADE_TIMEOUT_SECONDS,
) -> list[str]:
    """Return validation errors for one leaderboard submission.

    Args:
        payload: Parsed QCircuitEval JSON or JSONL payload.
        track: Leaderboard track to validate against.
        strict: Whether to require disclosure metadata for custom validation.
            Official tracks always require it.
        metadata: Optional disclosure metadata.
        adapter: Trusted local task adapter. Defaults to the bundled
            :class:`Adaptor`.
        unsafe_structural_only: Skip local grading for a custom-track
            experiment. This mode is never accepted for official tracks.

    Returns:
        List of validation error strings. Empty means valid.
    """
    errors, _ = _validate_and_regrade(
        payload,
        track=track,
        strict=strict,
        metadata=metadata,
        adapter=adapter,
        unsafe_structural_only=unsafe_structural_only,
        _evaluation_hook=_evaluation_hook,
        _trusted_regrade_timeout=_trusted_regrade_timeout,
    )
    return errors


def _validate_and_regrade(
    payload: Mapping[str, Any],
    *,
    track: Track,
    strict: bool,
    metadata: Mapping[str, Any] | None,
    adapter: TaskAdapter | None = None,
    unsafe_structural_only: bool = False,
    _evaluation_hook: _EvaluationHook | None = None,
    _trusted_regrade_timeout: float = DEFAULT_TRUSTED_REGRADE_TIMEOUT_SECONDS,
) -> tuple[list[str], list[Mapping[str, Any]] | None]:
    """Validate a submission and return score-authoritative records."""
    errors: list[str] = []
    if unsafe_structural_only and track != "custom":
        errors.append("--unsafe-structural-only is restricted to the custom track")
        return errors, None

    _validate_schema(payload, errors)
    if errors:
        return errors, None

    if strict or track != "custom":
        _validate_metadata(metadata, payload, errors)
    _validate_protocol(
        cast(Mapping[str, Any], payload["summary"]),
        track,
        payload.get("provider"),
        errors,
    )
    results = cast(Sequence[Mapping[str, Any]], payload["results"])
    _validate_record_statuses(results, errors)
    _validate_summary_consistency(payload, errors)
    trusted_adapter = None if unsafe_structural_only else adapter or Adaptor()
    if track != "custom":
        assert trusted_adapter is not None
        _validate_official_identity(payload, track, errors, trusted_adapter)
        _validate_official_provenance(payload, metadata, errors, trusted_adapter)
        _validate_official_final_statuses(results, errors)
    if errors:
        return errors, None
    if unsafe_structural_only:
        return errors, list(results)

    assert trusted_adapter is not None
    trusted_results = _trusted_regrade(
        results,
        trusted_adapter,
        errors,
        _evaluation_hook=_evaluation_hook,
        timeout_seconds=_trusted_regrade_timeout,
    )
    if errors:
        return errors, None
    return errors, trusted_results


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="QCircuitEval JSON or JSONL output file")
    parser.add_argument("--track", choices=sorted(TRACKS), default="core-all-single")
    parser.add_argument("--metadata", type=Path, help="Public submission metadata JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require disclosure metadata for custom runs (official tracks always require it)",
    )
    parser.add_argument(
        "--unsafe-structural-only",
        action="store_true",
        help="Skip trusted regrading (custom track only; never valid for official acceptance)",
    )
    parser.add_argument(
        "--trusted-regrade-timeout",
        type=_positive_float,
        default=DEFAULT_TRUSTED_REGRADE_TIMEOUT_SECONDS,
        help="Hard wall-clock seconds allowed for each fresh trusted regrade worker",
    )
    parser.add_argument("--out", type=Path, help="Optional compact leaderboard JSON output path")
    return parser.parse_args(argv)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _load_jsonl(path: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        kind = payload.get("kind")
        if kind == "result":
            results.append({key: value for key, value in payload.items() if key != "kind"})
        elif kind == "summary":
            summary = {key: value for key, value in payload.items() if key != "kind"}
    if summary is None:
        raise ValueError(f"{path} does not contain a summary line")
    summary["results"] = results
    return summary


def _load_metadata(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _validate_schema(payload: Mapping[str, Any], errors: list[str]) -> None:
    if payload.get("schema_version") != "qceval.run.v2":
        errors.append("schema_version must be qceval.run.v2")
    results = payload.get("results")
    summary = payload.get("summary")
    if not isinstance(results, list):
        errors.append("results must be a list")
    if not isinstance(summary, Mapping):
        errors.append("summary must be an object")
    if errors:
        return
    if len(results) != summary.get("total_tasks"):
        errors.append("results length must equal summary.total_tasks")
    _validate_result_records(results, errors)


def _validate_result_records(results: Sequence[Any], errors: list[str]) -> None:
    for index, record in enumerate(results):
        if not isinstance(record, Mapping):
            errors.append(f"result {index} must be an object")
            continue
        _validate_result_record(index, record, errors)


def _validate_result_record(index: int, record: Mapping[str, Any], errors: list[str]) -> None:
    """Validate the schema and primitive field types of one result record."""
    missing = sorted(REQUIRED_RESULT_FIELDS - set(record))
    if missing:
        errors.append(f"result {index} missing fields: {', '.join(missing)}")
    if record.get("status") not in VALID_STATUSES:
        errors.append(f"result {index} has invalid status {record.get('status')!r}")
    for field in ("suite", "framework", "task_id", "entry_point", "provider", "model"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"result {index}.{field} must be a non-empty string")
    for field in ("sample_index", "attempt_index"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"result {index}.{field} must be a non-negative integer")
    provider_response = record.get("provider_response")
    if not isinstance(provider_response, Mapping):
        errors.append(f"result {index}.provider_response must be an object")
        return
    code = provider_response.get("code")
    if code is not None and not isinstance(code, str):
        errors.append(f"result {index}.provider_response.code must be a string or null")


def _record_counts_as_passed(record: Mapping[str, Any]) -> bool:
    """Return whether one submitted record independently counts as a pass."""
    if record.get("status") != "passed":
        return False
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return True
    if evaluation.get("passed") is False:
        return False
    verified_status = evaluation.get("verified_status")
    if verified_status is None and isinstance(evaluation.get("grader_details"), Mapping):
        verified_status = evaluation["grader_details"].get("verified_status")
    return verified_status is None or verified_status == "verified_pass"


def _derived_status(evaluation: Mapping[str, Any]) -> str | None:
    """Derive the outcome status implied by an embedded evaluation payload."""
    if _is_infrastructure_evaluation(evaluation):
        return "infrastructure_error"
    compiled = evaluation.get("compiled")
    ran = evaluation.get("ran")
    passed = evaluation.get("passed")
    if not all(isinstance(flag, bool) for flag in (compiled, ran, passed)):
        return None
    if not compiled:
        return "compile_failed"
    if not ran:
        return "run_failed"
    return "passed" if passed else "failed"


def _validate_record_statuses(results: Sequence[Mapping[str, Any]], errors: list[str]) -> None:
    """Reject records whose claimed status disagrees with their evaluation."""
    for index, record in enumerate(results):
        if not isinstance(record, Mapping):
            continue
        evaluation = record.get("evaluation")
        claimed = record.get("status")
        if not isinstance(evaluation, Mapping):
            provider_response = record.get("provider_response")
            code = provider_response.get("code") if isinstance(provider_response, Mapping) else None
            if claimed == "passed" and code is None:
                errors.append(f"result {index} claims status 'passed' without candidate code")
            continue
        derived = _derived_status(evaluation)
        if derived is not None and claimed != derived:
            errors.append(f"result {index} claims status {claimed!r} but its evaluation implies {derived!r}")


def recompute_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute leaderboard-relevant summary statistics from records.

    Args:
        results: Submitted result records.

    Returns:
        Recomputed totals, pass counts, pass rate, and per-suite/framework
        statistics derived only from the records themselves.
    """
    total = len(results)
    counts = Counter(record.get("status") for record in results)
    assigned = total - counts["generated"]
    passed = sum(1 for record in results if _record_counts_as_passed(record))
    scoreable = assigned - counts["infrastructure_error"]
    grouped = _records_by_suite_framework(results)
    by_suite_framework: dict[str, dict[str, dict[str, Any]]] = {}
    for (suite, framework), records in sorted(grouped.items()):
        by_suite_framework.setdefault(suite, {})[framework] = _summary_stats(records)
    return {
        "total_tasks": total,
        "assigned_tasks": assigned,
        "passed": passed,
        "failed": counts["failed"],
        "generated": counts["generated"],
        "provider_failures": counts["provider_failed"],
        "compile_failures": counts["compile_failed"],
        "run_failures": counts["run_failed"],
        "infrastructure_failures": counts["infrastructure_error"],
        "rerun_required_tasks": counts["infrastructure_error"],
        "scoreable_tasks": scoreable,
        "pass_rate": passed / assigned if assigned else 0.0,
        "pass_rate_denominator": "assigned_tasks",
        "by_suite_framework": by_suite_framework,
    }


def _summary_stats(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return score and terminal-status counts for one result group."""
    counts = Counter(record.get("status") for record in results)
    total = len(results)
    assigned = total - counts["generated"]
    scoreable = assigned - counts["infrastructure_error"]
    passed = sum(1 for record in results if _record_counts_as_passed(record))
    return {
        "total_tasks": total,
        "assigned_tasks": assigned,
        "passed": passed,
        "failed": counts["failed"],
        "generated": counts["generated"],
        "provider_failures": counts["provider_failed"],
        "compile_failures": counts["compile_failed"],
        "run_failures": counts["run_failed"],
        "infrastructure_failures": counts["infrastructure_error"],
        "rerun_required_tasks": counts["infrastructure_error"],
        "scoreable_tasks": scoreable,
        "pass_rate": passed / assigned if assigned else 0.0,
        "pass_rate_denominator": "assigned_tasks",
    }


def _validate_summary_consistency(payload: Mapping[str, Any], errors: list[str]) -> None:
    """Reject submissions whose embedded summary disagrees with the records."""
    summary = cast(Mapping[str, Any], payload["summary"])
    recomputed = recompute_summary(cast(Sequence[Mapping[str, Any]], payload["results"]))
    for key in ("total_tasks", "passed"):
        if summary.get(key) != recomputed[key]:
            errors.append(
                f"summary.{key} is {summary.get(key)!r} but recomputed value from records is {recomputed[key]!r}"
            )
    claimed_rate = summary.get("pass_rate")
    if not isinstance(claimed_rate, (int, float)) or abs(float(claimed_rate) - recomputed["pass_rate"]) > 1e-9:
        errors.append(
            f"summary.pass_rate is {claimed_rate!r} but recomputed value from records is {recomputed['pass_rate']!r}"
        )
    _validate_group_consistency(summary, recomputed, errors)


def _validate_group_consistency(summary: Mapping[str, Any], recomputed: Mapping[str, Any], errors: list[str]) -> None:
    claimed_groups = summary.get("by_suite_framework")
    if not isinstance(claimed_groups, Mapping):
        return
    for suite, frameworks in cast(Mapping[str, Any], recomputed["by_suite_framework"]).items():
        for framework, stats in frameworks.items():
            claimed_suite = claimed_groups.get(suite)
            claimed = claimed_suite.get(framework) if isinstance(claimed_suite, Mapping) else None
            if not isinstance(claimed, Mapping):
                errors.append(f"summary.by_suite_framework missing recomputed group {suite}/{framework}")
                continue
            for key in ("total_tasks", "passed"):
                if claimed.get(key) != stats[key]:
                    errors.append(
                        f"summary.by_suite_framework.{suite}.{framework}.{key} is {claimed.get(key)!r} "
                        f"but recomputed value from records is {stats[key]!r}"
                    )
            claimed_rate = claimed.get("pass_rate")
            if not isinstance(claimed_rate, (int, float)) or abs(float(claimed_rate) - stats["pass_rate"]) > 1e-9:
                errors.append(
                    f"summary.by_suite_framework.{suite}.{framework}.pass_rate is {claimed_rate!r} "
                    f"but recomputed value from records is {stats['pass_rate']!r}"
                )


def _validate_metadata(
    metadata: Mapping[str, Any] | None,
    payload: Mapping[str, Any],
    errors: list[str],
) -> None:
    if metadata is None:
        errors.append("official tracks and --strict custom validation require --metadata")
        return
    missing = sorted(REQUIRED_METADATA_FIELDS - set(metadata))
    if missing:
        errors.append(f"metadata missing fields: {', '.join(missing)}")
    empty = sorted(
        field
        for field in REQUIRED_METADATA_FIELDS & set(metadata)
        if not isinstance(metadata[field], str) or not metadata[field].strip()
    )
    if empty:
        errors.append(f"metadata fields must be non-empty strings: {', '.join(empty)}")
    if metadata.get("provider") != payload.get("provider"):
        errors.append("metadata.provider must match submission provider")
    if metadata.get("model") != payload.get("model"):
        errors.append("metadata.model must match submission model")
    commit = metadata.get("qceval_commit")
    if isinstance(commit, str) and commit.strip() and _COMMIT_RE.fullmatch(commit) is None:
        errors.append("metadata.qceval_commit must be a full 40- or 64-character hexadecimal commit")
    image = metadata.get("image")
    if isinstance(image, str) and image.strip() and _IMAGE_DIGEST_RE.fullmatch(image) is None:
        errors.append("metadata.image must be an image reference pinned by @sha256:<64 lowercase hex>")


def _validate_protocol(
    summary: Mapping[str, Any],
    track: Track,
    provider: Any,
    errors: list[str],
) -> None:
    protocol = summary.get("run_protocol")
    if not isinstance(protocol, Mapping):
        errors.append("summary.run_protocol must be present")
        return
    if track == "custom":
        return
    expected = {"samples_per_task": 1, "pass_k": 1, "max_attempts": 1, "feedback_enabled": False}
    for key, value in expected.items():
        if protocol.get(key) != value:
            errors.append(f"official tracks require run_protocol.{key}={value!r}")
    _validate_temperature_protocol(protocol, provider, errors)


def _validate_temperature_protocol(  # noqa: C901 - mutually exclusive protocol evidence gates
    protocol: Mapping[str, Any], provider: Any, errors: list[str]
) -> None:
    """Validate the official generation temperature declaration."""
    generation = protocol.get("generation_parameters")
    if not isinstance(generation, Mapping):
        errors.append("official tracks require run_protocol.generation_parameters")
        return
    temperature = generation.get("temperature")
    if not isinstance(temperature, Mapping):
        errors.append("official tracks require run_protocol.generation_parameters.temperature")
        return
    value = temperature.get("value")
    source = temperature.get("source")
    if source == "not_exposed":
        if value is not None:
            errors.append("non-exposed temperature must have value null")
        endpoint = generation.get("endpoint_tag")
        if provider == "openrouter" and (
            not isinstance(endpoint, Mapping) or endpoint.get("source") != "explicit" or not endpoint.get("value")
        ):
            errors.append("OpenRouter may declare temperature not_exposed only for an explicitly pinned endpoint")
    elif source == "per_record_route_provenance":
        values = value if isinstance(value, list) else [value]
        if any(isinstance(item, bool) or (item is not None and item != 0.0) for item in values):
            errors.append("official per-route temperatures must be 0.0 when exposed and null otherwise")
        endpoint = generation.get("endpoint_tag")
        if not isinstance(endpoint, Mapping) or endpoint.get("source") != "per_record_route_provenance":
            errors.append("per-route temperature requires per-route endpoint provenance")
    elif source not in {"explicit", "provider_default"}:
        errors.append(
            "official temperature source must be explicit, provider_default, not_exposed, "
            "or per_record_route_provenance"
        )
    elif isinstance(value, bool) or value != 0.0:
        errors.append("official tracks require exposed temperature 0.0")


def _validate_official_identity(
    payload: Mapping[str, Any],
    track: Track,
    errors: list[str],
    adapter: TaskAdapter,
) -> None:
    """Bind official records to one bundled task and provider/model identity."""
    qceval = payload.get("qceval")
    if not isinstance(qceval, Mapping) or qceval.get("source") != "bundled-qceval":
        errors.append("official tracks require qceval.source='bundled-qceval'")
    suites, _ = TRACKS[track]
    if payload.get("suites") != list(suites):
        errors.append(f"official track {track} requires top-level suites={list(suites)!r}")

    provider = payload.get("provider")
    model = payload.get("model")
    if not isinstance(provider, str) or not provider.strip():
        errors.append("official tracks require a non-empty top-level provider")
    if not isinstance(model, str) or not model.strip():
        errors.append("official tracks require a non-empty top-level model")
    for index, record in enumerate(cast(Sequence[Mapping[str, Any]], payload["results"])):
        if record.get("provider") != provider:
            errors.append(f"result {index}.provider must match submission provider")
        if record.get("model") != model:
            errors.append(f"result {index}.model must match submission model")
        response = record.get("provider_response")
        response_model = response.get("model") if isinstance(response, Mapping) else None
        if response_model is not None and response_model != record.get("model"):
            errors.append(f"result {index}.provider_response.model must match result model")

    _validate_track_tasks(payload, track, errors, adapter)


def _validate_official_provenance(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    errors: list[str],
    adapter: TaskAdapter,
) -> None:
    """Compare submitted package identity with the local trusted grader."""
    try:
        trusted = _trusted_local_adapter_metadata(adapter.metadata())
    except Exception as exc:
        errors.append(f"could not read trusted local adapter metadata: {type(exc).__name__}: {exc}")
        return
    if trusted["dirty"] is True:
        errors.append("official scoring requires a clean trusted QCircuitEval source checkout")

    submitted = payload.get("qceval")
    if not isinstance(submitted, Mapping):
        return
    trusted_version = trusted["package_version"]
    if trusted_version != "unknown" and submitted.get("package_version") != trusted_version:
        errors.append(
            "submission qceval.package_version "
            f"{submitted.get('package_version')!r} does not match trusted scorer {trusted_version!r}"
        )
    trusted_commit = trusted["commit"]
    if trusted_commit is None:
        return
    if submitted.get("commit") != trusted_commit:
        errors.append(
            f"submission qceval.commit {submitted.get('commit')!r} does not match trusted scorer {trusted_commit!r}"
        )
    if metadata is not None and metadata.get("qceval_commit") != trusted_commit:
        errors.append("metadata.qceval_commit does not match the trusted scorer commit")


def _validate_track_tasks(
    payload: Mapping[str, Any],
    track: Track,
    errors: list[str],
    adapter: TaskAdapter,
) -> None:
    suites, frameworks = TRACKS[track]
    results = cast(list[Mapping[str, Any]], payload["results"])
    actual = _records_by_suite_framework(results)
    expected_keys = {(suite, framework) for suite in suites for framework in frameworks}
    if set(actual) != expected_keys:
        errors.append(f"{track} requires exactly these suite/framework pairs: {_format_keys(expected_keys)}")
        return
    for suite, framework in sorted(expected_keys):
        try:
            tasks = adapter.load_tasks(framework, suite=suite)
        except Exception as exc:
            errors.append(f"could not load bundled {suite}/{framework} tasks: {type(exc).__name__}: {exc}")
            continue
        expected_ids = [task.task_id for task in tasks]
        records = actual[(suite, framework)]
        observed_ids = [str(record.get("task_id")) for record in records]
        if observed_ids != expected_ids:
            errors.append(f"{suite}/{framework} task IDs do not match bundled order")
        for record, task in zip(records, tasks, strict=False):
            if record.get("sample_index") != 0 or record.get("attempt_index") != 0:
                errors.append(f"{suite}/{framework}/{record.get('task_id')} must be single-shot sample 0 attempt 0")
            if record.get("task_id") == task.task_id and record.get("entry_point") != task.entry_point:
                errors.append(f"{suite}/{framework}/{task.task_id} entry_point does not match bundled task")


def _validate_official_final_statuses(results: Sequence[Mapping[str, Any]], errors: list[str]) -> None:
    """Require operational failures to be rerun before official acceptance."""
    for index, record in enumerate(results):
        if record.get("status") == "infrastructure_error":
            errors.append(f"result {index} has infrastructure_error; official final acceptance requires a clean rerun")


def _trusted_regrade(
    results: Sequence[Mapping[str, Any]],
    adapter: TaskAdapter,
    errors: list[str],
    *,
    _evaluation_hook: _EvaluationHook | None = None,
    timeout_seconds: float = DEFAULT_TRUSTED_REGRADE_TIMEOUT_SECONDS,
) -> list[Mapping[str, Any]]:
    """Regrade every submitted candidate against trusted local bundled tasks."""
    tasks: dict[tuple[Suite, Framework, str], QCEvalTask] = {}
    trusted: list[Mapping[str, Any]] = []
    for index, record in enumerate(results):
        trusted_record = _trusted_regrade_record(
            index,
            record,
            adapter,
            tasks,
            errors,
            _evaluation_hook=_evaluation_hook,
            timeout_seconds=timeout_seconds,
        )
        if trusted_record is not None:
            trusted.append(trusted_record)
    return trusted


def _trusted_regrade_record(
    index: int,
    record: Mapping[str, Any],
    adapter: TaskAdapter,
    tasks: dict[tuple[Suite, Framework, str], QCEvalTask],
    errors: list[str],
    *,
    _evaluation_hook: _EvaluationHook | None,
    timeout_seconds: float,
) -> Mapping[str, Any] | None:
    """Return one locally regraded record, or record a trusted-rerun error."""
    suite = cast(Suite, record.get("suite"))
    framework = cast(Framework, record.get("framework"))
    task_id = str(record.get("task_id"))
    task = _trusted_task(index, suite, framework, task_id, adapter, tasks, errors)
    if task is None:
        return None
    if record.get("entry_point") != task.entry_point:
        errors.append(f"result {index} entry_point does not match bundled task")
        return None

    response = cast(Mapping[str, Any], record["provider_response"])
    code = response.get("code")
    trusted_record = dict(record)
    if not isinstance(code, str) or not code.strip() or response.get("error") is not None:
        trusted_record["status"] = "provider_failed"
        trusted_record["evaluation"] = None
        return trusted_record
    try:
        evaluation_payload = _evaluate_candidate(
            task,
            code,
            _evaluation_hook=_evaluation_hook,
            timeout_seconds=timeout_seconds,
        )
    except TrustedRegradeTimeout as exc:
        errors.append(f"trusted regrade timed out for {suite}/{framework}/{task_id}; rerun required ({exc})")
        return None
    except TrustedRegradeError as exc:
        errors.append(f"trusted regrade failed for {suite}/{framework}/{task_id}; rerun required ({exc})")
        return None
    except Exception as exc:
        errors.append(
            f"trusted regrade failed for {suite}/{framework}/{task_id}; rerun required ({type(exc).__name__}: {exc})"
        )
        return None
    trusted_status = _derived_status(evaluation_payload)
    if trusted_status is None:
        errors.append(f"trusted regrade returned an invalid evaluation for {suite}/{framework}/{task_id}")
        return None
    if trusted_status == "infrastructure_error":
        errors.append(f"trusted regrade hit infrastructure_error for {suite}/{framework}/{task_id}; rerun required")
        return None
    trusted_record["status"] = trusted_status
    trusted_record["evaluation"] = evaluation_payload
    return trusted_record


def _evaluate_candidate(
    task: QCEvalTask,
    code: str,
    *,
    _evaluation_hook: _EvaluationHook | None,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """Return evaluation data, using an in-process hook only in unit tests."""
    if _evaluation_hook is None:
        return evaluate_trusted_candidate(task, code, timeout_seconds=timeout_seconds)
    evaluation = _evaluation_hook(task, code)
    if isinstance(evaluation, QCEvalEvaluation):
        return evaluation.to_dict()
    if isinstance(evaluation, Mapping):
        return dict(evaluation)
    raise TypeError("private evaluation hook must return an evaluation mapping")


def _trusted_task(
    index: int,
    suite: Suite,
    framework: Framework,
    task_id: str,
    adapter: TaskAdapter,
    tasks: dict[tuple[Suite, Framework, str], QCEvalTask],
    errors: list[str],
) -> QCEvalTask | None:
    """Resolve one submitted task against the trusted bundled inventory."""
    key = (suite, framework, task_id)
    if key not in tasks:
        try:
            loaded = adapter.load_tasks(framework, suite=suite)
        except Exception as exc:
            errors.append(f"trusted regrade could not load {suite}/{framework}: {type(exc).__name__}: {exc}")
            return None
        tasks.update({(task.suite, task.framework, task.task_id): task for task in loaded})
    task = tasks.get(key)
    if task is None:
        errors.append(f"result {index} does not identify a bundled task")
    return task


def _is_infrastructure_evaluation(evaluation: Mapping[str, Any]) -> bool:
    """Return whether an evaluation represents harness/infrastructure failure."""
    if evaluation.get("error_type") == "InfrastructureError":
        return True
    semantic = evaluation.get("semantic_result")
    if not isinstance(semantic, Mapping) or semantic.get("status") not in {"execution_error", "resource_limit"}:
        return False
    diagnostics = semantic.get("diagnostics")
    if not isinstance(diagnostics, list):
        return False
    for item in diagnostics:
        if isinstance(item, Mapping) and item.get("name") == "failure_origin":
            return item.get("value") == "grader_verification"
    return False


def _records_by_suite_framework(
    results: Sequence[Mapping[str, Any]],
) -> dict[tuple[Suite, Framework], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Suite, Framework], list[Mapping[str, Any]]] = defaultdict(list)
    for record in results:
        grouped[(cast(Suite, record["suite"]), cast(Framework, record["framework"]))].append(record)
    return dict(grouped)


def _format_keys(keys: set[tuple[Suite, Framework]]) -> str:
    return ", ".join(f"{suite}/{framework}" for suite, framework in sorted(keys))


def _leaderboard_record(
    payload: Mapping[str, Any],
    *,
    track: Track,
    metadata: Mapping[str, Any] | None,
    results: Sequence[Mapping[str, Any]],
    trusted_regrade: bool,
    trusted_adapter_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    # The leaderboard row never trusts the submitted summary or evaluations.
    # Official rows are computed only from locally regraded candidate code.
    recomputed = recompute_summary(results)
    return {
        "track": track,
        "schema_version": payload["schema_version"],
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "validation_mode": "trusted_local_regrade" if trusted_regrade else "unsafe_custom_structural_only",
        "trusted_regrade": trusted_regrade,
        "trusted_local_adapter": (
            _trusted_local_adapter_metadata(trusted_adapter_metadata) if trusted_adapter_metadata is not None else None
        ),
        "pass_rate": recomputed["pass_rate"],
        "passed": recomputed["passed"],
        "total_tasks": recomputed["total_tasks"],
        "scoreable_tasks": recomputed["scoreable_tasks"],
        "by_suite_framework": recomputed["by_suite_framework"],
        "metadata": metadata or {},
    }


def _trusted_local_adapter_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded local identity recorded in a leaderboard row."""
    package_version = metadata.get("package_version")
    if not isinstance(package_version, str) or not package_version.strip():
        package_version = "unknown"
    commit = metadata.get("commit")
    if not isinstance(commit, str) or not commit.strip():
        commit = None
    dirty = metadata.get("dirty")
    if not isinstance(dirty, bool):
        dirty = None
    source = metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        source = "unknown"
    return {
        "source": source,
        "package_version": package_version,
        "commit": commit,
        "commit_status": "available" if commit is not None else "unavailable",
        "dirty": dirty,
    }


if __name__ == "__main__":
    raise SystemExit(main())
