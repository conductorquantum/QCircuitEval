"""Run configuration and execution option models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from qceval.models.types import Framework, Suite

FeedbackMode = Literal["execution_trace"]
FeedbackHistoryMode = Literal["full_transcript"]
FeedbackStopRule = Literal["first_verified_pass"]


@dataclass(frozen=True)
class FeedbackPolicy:
    """Versioned model-facing repair protocol.

    The policy is intentionally narrower than the grader.  It exposes bounded
    execution observations while keeping contract targets, verifier evidence,
    and the derived error taxonomy out of provider messages.

    Attributes:
        version: Stable policy version recorded in run and attempt metadata.
        mode: Kind of observation returned after a failed attempt.
        history_mode: Conversation history retained for each repair request.
        stop_rule: Success condition that terminates a repair chain.
    """

    version: str = "feedback.execution_trace.v1"
    mode: FeedbackMode = "execution_trace"
    history_mode: FeedbackHistoryMode = "full_transcript"
    stop_rule: FeedbackStopRule = "first_verified_pass"

    def __post_init__(self) -> None:
        if self.version != "feedback.execution_trace.v1":
            raise ValueError(f"unsupported feedback policy version: {self.version}")
        if self.mode != "execution_trace":
            raise ValueError(f"unsupported feedback mode: {self.mode}")
        if self.history_mode != "full_transcript":
            raise ValueError(f"unsupported feedback history mode: {self.history_mode}")
        if self.stop_rule != "first_verified_pass":
            raise ValueError(f"unsupported feedback stop rule: {self.stop_rule}")

    def to_dict(self) -> dict[str, str]:
        """Return JSON-compatible policy metadata."""
        return {
            "version": self.version,
            "mode": self.mode,
            "history_mode": self.history_mode,
            "stop_rule": self.stop_rule,
        }


@dataclass(frozen=True)
class RunConfig:
    """Stable benchmark inputs that define what is evaluated.

    ``RunConfig`` captures semantic inputs that should appear in output and
    cache keys.  Operational controls such as worker counts live in
    :class:`RunOptions` so changing parallelism does not change run identity.

    Attributes:
        provider: Provider name used for labeling and registry lookup.
        frameworks: Frameworks to evaluate in order.
        source_hint: Optional provenance path recorded in metadata.
        model: Model identifier passed to providers.
        max_tasks: Optional per-framework task limit.
        task_numbers: Optional suite-local task numbers to evaluate.
        provider_config: Provider-specific settings.
        suites: Benchmark suites evaluated in order.
        samples_per_task: Number of independent samples generated per task.
        pass_k: Pass@K cutoff used when ``samples_per_task`` is greater than
            one.
        max_attempts: Maximum feedback-repair attempts per task.
        feedback_max_chars: Maximum feedback text sent back to providers.
        feedback_policy: Versioned repair-feedback and history policy.
    """

    provider: str
    frameworks: tuple[Framework, ...]
    source_hint: Path | None
    model: str | None
    max_tasks: int | None = None
    task_numbers: tuple[int, ...] | None = None
    provider_config: Mapping[str, Any] = field(default_factory=dict)
    suites: tuple[Suite, ...] = ("core",)
    samples_per_task: int = 1
    pass_k: int = 1
    max_attempts: int = 1
    feedback_max_chars: int = 2000
    feedback_policy: FeedbackPolicy = field(default_factory=FeedbackPolicy)

    def __post_init__(self) -> None:
        _validate_sampling_config(self)
        _validate_task_numbers(self.task_numbers)


def _validate_sampling_config(config: RunConfig) -> None:
    if config.samples_per_task < 1:
        raise ValueError("samples_per_task must be >= 1")
    if config.pass_k < 1:
        raise ValueError("pass_k must be >= 1")
    if config.pass_k > config.samples_per_task:
        raise ValueError(f"pass_k ({config.pass_k}) must be <= samples_per_task ({config.samples_per_task})")
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if config.feedback_max_chars < 1:
        raise ValueError("feedback_max_chars must be >= 1")
    if config.samples_per_task > 1 and config.max_attempts > 1:
        raise ValueError(
            "samples_per_task > 1 and max_attempts > 1 cannot be combined; "
            "Pass@K and feedback repair answer different evaluation questions"
        )


def _validate_task_numbers(task_numbers: tuple[int, ...] | None) -> None:
    if task_numbers is None:
        return
    if not task_numbers:
        raise ValueError("task_numbers must not be empty")
    if any(task_number < 1 for task_number in task_numbers):
        raise ValueError("task_numbers must be positive")
    if len(set(task_numbers)) != len(task_numbers):
        raise ValueError("task_numbers must not contain duplicates")


@dataclass(frozen=True)
class RunOptions:
    """Operational controls for a benchmark run.

    Options affect execution mechanics, not benchmark semantics.  Defaults keep
    execution serial and deterministic, matching a minimal local run.

    Attributes:
        generation_concurrency: Number of concurrent provider calls.
        evaluation_workers: Number of worker processes for grading.
        cache_dir: Optional root for cached provider responses.
        resume_from: Optional JSONL run file to reuse completed records from.
        stream_to: Optional JSONL path that receives records as they finish.
        task_timeout: Optional per-task wall-clock timeout in seconds for
            isolated generate-and-evaluate execution.
        eval_timeout: Optional per-task evaluation timeout in seconds.
        fail_fast: Whether to stop after the first non-passing record.
        progress: Whether to write progress updates to stderr.
        prompt_frameworks: Frameworks for which to request fresh candidate
            code. ``None`` means every configured framework.
        regrade_frameworks: Frameworks for which to execute the current
            grader. ``None`` means every configured framework.
        input_from: Optional JSONL artifact supplying stored candidate code
            for regrade-only frameworks.
        stop_on_infrastructure_error: Stop scheduling later generation chunks
            after draining the chunk containing an infrastructure failure.
    """

    generation_concurrency: int = 1
    evaluation_workers: int = 1
    cache_dir: Path | None = None
    resume_from: Path | None = None
    stream_to: Path | None = None
    task_timeout: float | None = None
    eval_timeout: float | None = None
    fail_fast: bool = False
    progress: bool = False
    prompt_frameworks: tuple[Framework, ...] | None = None
    regrade_frameworks: tuple[Framework, ...] | None = None
    input_from: Path | None = None
    stop_on_infrastructure_error: bool = False
