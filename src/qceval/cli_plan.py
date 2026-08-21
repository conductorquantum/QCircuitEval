"""Expand ``qceval run`` into per-model, per-effort Pass@1 jobs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qceval.production.campaign import configuration_id

NAMED_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
ENABLED_REASONING = "enabled"
ALL_REASONING_TOKEN = "all"
ALLOWED_REASONING_EFFORTS = frozenset((*NAMED_REASONING_EFFORTS, ENABLED_REASONING))
EFFORT_SWEEP_SCHEMA_VERSION = "qceval.effort_sweep.v1"
_EFFORT_RANK = {effort: index for index, effort in enumerate((*NAMED_REASONING_EFFORTS, ENABLED_REASONING))}
_FILE_SUFFIXES = {".json", ".jsonl"}


@dataclass(frozen=True)
class ReasoningJob:
    """One provider configuration in a Pass@1 matrix."""

    model: str | None
    effort: str | None

    def reasoning_effort(self) -> str | None:
        """Return the named OpenRouter effort, if this job uses one."""
        if self.effort is None or self.effort == ENABLED_REASONING:
            return None
        return self.effort

    def reasoning_enabled(self) -> bool | None:
        """Return whether this job enables unnamed reasoning."""
        return True if self.effort == ENABLED_REASONING else None

    def configuration_id(self) -> str:
        """Return the stable configuration identifier for this job."""
        if self.effort is None:
            raise ValueError("configuration identity requires a reasoning effort")
        return configuration_id(self.model or "smoke-canonical", self.effort)


def expand_reasoning_jobs(args: argparse.Namespace) -> tuple[ReasoningJob, ...]:
    """Return the model/effort matrix selected by CLI flags.

    Args:
        args: Parsed ``qceval run`` arguments.

    Returns:
        Ordered jobs to execute for this invocation.
    """
    requested = args.reasoning_effort
    if args.registry:
        return jobs_from_registry(
            load_registry_efforts(args.registry),
            requested_effort=requested,
            model_filter=args.model,
        )
    if requested == ALL_REASONING_TOKEN:
        return tuple(ReasoningJob(model=args.model, effort=effort) for effort in NAMED_REASONING_EFFORTS)
    effort = ENABLED_REASONING if args.reasoning_enabled else requested
    return (ReasoningJob(model=args.model, effort=effort),)


def load_registry_efforts(paths: Sequence[Path]) -> dict[str, tuple[str, ...]]:
    """Load and merge model efforts, preserving first-seen model order.

    Args:
        paths: Capability registry files to merge.

    Returns:
        Mapping from model ID to its supported efforts, ordered by strength.

    Raises:
        ValueError: If a registry omits models, model IDs, or efforts.
    """
    merged: dict[str, set[str]] = {}
    for path in paths:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
        models = payload.get("models") if isinstance(payload, Mapping) else None
        if not isinstance(models, list):
            raise ValueError(f"registry must contain a models array: {path}")
        for index, item in enumerate(models):
            if not isinstance(item, Mapping):
                raise ValueError(f"registry model {index} must be an object: {path}")
            model = item.get("model_id")
            efforts = item.get("reasoning_efforts")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"registry model {index} has no model_id: {path}")
            if not isinstance(efforts, list) or not efforts:
                raise ValueError(f"registry model {model!r} has no reasoning_efforts: {path}")
            normalized = {_validate_effort(effort, model, path) for effort in efforts}
            merged.setdefault(model, set()).update(normalized)
    return {model: tuple(sorted(efforts, key=_EFFORT_RANK.__getitem__)) for model, efforts in merged.items()}


def jobs_from_registry(
    models: Mapping[str, Sequence[str]],
    *,
    requested_effort: str | None,
    model_filter: str | None,
) -> tuple[ReasoningJob, ...]:
    """Expand registry models into ordered jobs.

    Args:
        models: Mapping from model ID to supported efforts.
        requested_effort: Single effort to select, or ``None``/``"all"`` for the
            full matrix.
        model_filter: Restrict expansion to one model ID when set.

    Returns:
        Ordered jobs in registry order, strongest effort last per model.

    Raises:
        ValueError: If the selection matches no registry model or effort.
    """
    jobs: list[ReasoningJob] = []
    for model, efforts in models.items():
        if model_filter is not None and model != model_filter:
            continue
        selected = efforts if requested_effort in {None, ALL_REASONING_TOKEN} else (requested_effort,)
        jobs.extend(ReasoningJob(model=model, effort=effort) for effort in selected if effort in efforts)
    if not jobs:
        detail = f" for model {model_filter!r}" if model_filter is not None else ""
        raise ValueError(f"registry selection produced no jobs{detail}")
    return tuple(jobs)


def apply_reasoning_job(
    args: argparse.Namespace,
    job: ReasoningJob,
    *,
    out: Path,
    assign_configuration_id: bool,
) -> argparse.Namespace:
    """Clone parsed arguments for one expanded job.

    Args:
        args: Parsed ``qceval run`` arguments to clone.
        job: Job whose model and effort override the clone.
        out: Output destination for this job.
        assign_configuration_id: Whether to stamp the job's configuration ID.

    Returns:
        Arguments for a single-configuration run.
    """
    cloned = argparse.Namespace(**vars(args))
    cloned.model = job.model
    cloned.reasoning_effort = job.reasoning_effort()
    cloned.reasoning_enabled = job.reasoning_enabled()
    cloned.configuration_id = job.configuration_id() if assign_configuration_id else args.configuration_id
    cloned.out = out
    cloned.reasoning_jobs = (job,)
    return cloned


def is_multi_model(jobs: Sequence[ReasoningJob]) -> bool:
    """Return whether jobs span multiple model labels.

    Args:
        jobs: Expanded matrix jobs.
    """
    return len({job.model for job in jobs}) > 1


def sweep_output_suffix(job: ReasoningJob, *, multi_model: bool) -> str:
    """Return the filename suffix for one expanded job.

    Args:
        job: Job whose output is being named.
        multi_model: Whether the matrix spans more than one model.
    """
    return job.configuration_id() if multi_model else f"effort-{job.effort}"


def sweep_output_path(base: Path, job: ReasoningJob, *, multi_model: bool) -> Path:
    """Return one job output path for a sweep destination.

    Args:
        base: Destination requested with ``--out``; a file stem or a directory.
        job: Job whose output is being placed.
        multi_model: Whether the matrix spans more than one model.
    """
    if base.suffix.lower() in _FILE_SUFFIXES:
        suffix = sweep_output_suffix(job, multi_model=multi_model)
        return base.with_name(f"{base.stem}.{suffix}{base.suffix}")
    return base / f"{job.configuration_id()}.json"


def sweep_manifest_path(base: Path) -> Path:
    """Return the sweep manifest path associated with an output destination.

    Args:
        base: Destination requested with ``--out``; a file stem or a directory.
    """
    if base.suffix.lower() in _FILE_SUFFIXES:
        return base.with_name(f"{base.stem}.efforts.json")
    return base / "manifest.json"


def write_sweep_manifest(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic matrix manifest.

    Args:
        path: Manifest destination; parent directories are created.
        entries: One job record per executed configuration.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EFFORT_SWEEP_SCHEMA_VERSION,
        "jobs": list(entries),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_effort(value: Any, model: str, path: Path) -> str:
    if not isinstance(value, str) or value not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning effort {value!r} for {model!r} in {path}")
    return value
