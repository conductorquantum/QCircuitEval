"""Internal runner work item models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from qceval.models import ProviderRequest, ProviderResponse, QCEvalTask


@dataclass(frozen=True)
class RunJob:
    """Internal work item pairing a task with its provider request.

    Attributes:
        index: Stable result order assigned before any concurrent work starts.
        task: Benchmark task to evaluate.
        request: Provider request derived from ``task`` and run config.
        sample_index: Repeated-sample index for Pass@K runs.
        attempt_index: Feedback-repair attempt index.
        feedback: Feedback metadata attached to this attempt.
        parent_attempt_index: Previous attempt, when this is a repair.
        parent_code_sha256: Hash of the code repaired by this attempt.
        prompt: Whether this job should request a fresh provider response.
        regrade: Whether this job should run the current grader.
        source_response: Stored provider response used for a regrade-only job.
        source_provider: Provider label paired with ``source_response``.
    """

    index: int
    task: QCEvalTask
    request: ProviderRequest
    sample_index: int = 0
    attempt_index: int = 0
    feedback: Mapping[str, Any] = field(default_factory=dict)
    parent_attempt_index: int | None = None
    parent_code_sha256: str | None = None
    prompt: bool = True
    regrade: bool = True
    source_response: ProviderResponse | None = None
    source_provider: str | None = None


@dataclass
class _ActiveIsolatedJob:
    """Running task-level worker process."""

    job: RunJob
    process: Any
    queue: Any
    started_at: float
