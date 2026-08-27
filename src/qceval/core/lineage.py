"""Stable provenance helpers for benchmark request and repair lineages."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qceval.core.prompt_safety import assert_provider_messages_exclude_oracle
from qceval.models import ProviderRequest, RunConfig, RunOptions
from qceval.serialization import to_jsonable

REQUEST_TRACE_SCHEMA_VERSION = "qceval.request_trace.v1"
FEEDBACK_LINEAGE_SCHEMA_VERSION = "qceval.feedback_lineage.v1"
RUN_IDENTITY_SCHEMA_VERSION = "qceval.run_identity.v1"


def sha256_text(value: str | None) -> str | None:
    """Return a lowercase SHA-256 digest for text, preserving missing values.

    Args:
        value: Text to hash, or ``None``.

    Returns:
        Hexadecimal digest, or ``None`` when the input is missing.
    """
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_trace(request: ProviderRequest) -> dict[str, Any]:
    """Record the exact provider-visible transcript and stable content hashes.

    Args:
        request: Provider request whose effective messages are recorded.

    Returns:
        Versioned prompt, message transcript, and SHA-256 hashes.
    """
    messages = (
        [{"role": message.role, "content": message.content} for message in request.messages]
        if request.messages
        else [{"role": "user", "content": request.prompt}]
    )
    assert_provider_messages_exclude_oracle(messages)
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "schema_version": REQUEST_TRACE_SCHEMA_VERSION,
        "prompt": request.prompt,
        "prompt_sha256": sha256_text(request.prompt),
        "messages_sha256": sha256_text(canonical),
        "messages": messages,
    }


def build_run_identity(
    config: RunConfig,
    options: RunOptions,
    qceval_metadata: Mapping[str, Any],
    jobs: Sequence[Any],
) -> dict[str, Any]:
    """Return the complete, secret-safe identity of resumable benchmark work.

    Args:
        config: Score-authoritative benchmark configuration.
        options: Execution options that select prompt and regrade phases.
        qceval_metadata: Adapter and asset provenance.
        jobs: Concrete task jobs included in the run.

    Returns:
        Canonical run-identity payload with its SHA-256 digest.
    """
    provider_settings = json.dumps(
        to_jsonable(config.provider_config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    tasks = sorted(
        {
            (
                job.task.suite,
                job.task.framework,
                job.task.task_id,
                job.task.entry_point,
                sha256_text(job.task.prompt),
            )
            for job in jobs
        }
    )
    identity: dict[str, Any] = {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "provider": config.provider,
        "model": config.model,
        "frameworks": list(config.frameworks),
        "suites": list(config.suites),
        "max_tasks": config.max_tasks,
        "task_numbers": None if config.task_numbers is None else list(config.task_numbers),
        "samples_per_task": config.samples_per_task,
        "pass_k": config.pass_k,
        "max_attempts": config.max_attempts,
        "feedback_max_chars": config.feedback_max_chars,
        "feedback_policy": config.feedback_policy.to_dict(),
        "provider_config_sha256": sha256_text(provider_settings),
        "prompt_frameworks": None if options.prompt_frameworks is None else list(options.prompt_frameworks),
        "regrade_frameworks": None if options.regrade_frameworks is None else list(options.regrade_frameworks),
        "qceval": to_jsonable(qceval_metadata),
        "tasks": [
            {
                "suite": suite,
                "framework": framework,
                "task_id": task_id,
                "entry_point": entry_point,
                "prompt_sha256": prompt_hash,
            }
            for suite, framework, task_id, entry_point, prompt_hash in tasks
        ],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    identity["sha256"] = sha256_text(canonical)
    return identity


def chain_id(
    run_id: str,
    *,
    provider: str,
    model: str | None,
    suite: str,
    framework: str,
    task_id: str,
    sample_index: int,
) -> str:
    """Return a deterministic chain identifier scoped to one benchmark run.

    Args:
        run_id: Unique benchmark run identifier.
        provider: Configured provider name.
        model: Configured model identifier.
        suite: Benchmark suite.
        framework: Quantum framework.
        task_id: Suite-local task identifier.
        sample_index: Independent sample index.

    Returns:
        SHA-256 chain identifier.
    """
    identity = {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "suite": suite,
        "framework": framework,
        "task_id": task_id,
        "sample_index": sample_index,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_run_id(resume_from: Path | None) -> str:
    """Create a run ID or preserve the single run ID found in resume data.

    Legacy resume files have no run ID. They receive a deterministic UUID based
    on the file contents, so repeated resumes of the same artifact stay linked.

    Args:
        resume_from: Optional JSONL artifact being resumed.

    Returns:
        Existing, new, or legacy-derived run UUID.

    Raises:
        ValueError: If resume data contains multiple run identifiers.
    """
    if resume_from is None or not resume_from.exists():
        return str(uuid.uuid4())
    run_ids = _resume_run_ids(resume_from)
    if len(run_ids) > 1:
        raise ValueError("resume data contains records from multiple run IDs")
    if run_ids:
        return next(iter(run_ids))
    digest = hashlib.sha256(resume_from.read_bytes()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"qceval:legacy-resume:{digest}"))


def _resume_run_ids(path: Path) -> set[str]:
    run_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        lineage = payload.get("lineage")
        if isinstance(lineage, Mapping) and isinstance(lineage.get("run_id"), str):
            run_ids.add(lineage["run_id"])
        elif payload.get("kind") == "summary" and isinstance(payload.get("run_id"), str):
            run_ids.add(payload["run_id"])
    return run_ids
