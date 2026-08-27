"""Task, evaluation, and record models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from qceval.error_taxonomy import classify_error_taxonomy
from qceval.models.providers import ProviderResponse
from qceval.models.types import Framework, OutcomeStatus, Suite
from qceval.serialization import to_jsonable


@dataclass(frozen=True)
class QCEvalTask:
    """Bundled benchmark task and grading metadata.

    Instances wrap raw task assets in a provider-friendly shape.  The raw asset
    remains available for evaluators and for advanced callers that need fields
    not promoted to first-class attributes.

    Attributes:
        task_id: Zero-padded task identifier within ``framework``.
        framework: Quantum framework targeted by the task.
        prompt: Prompt text sent to providers.
        entry_point: Function name candidate code must define.
        category: Optional task category from the bundled asset.
        canonical_class: Grader specification for the task.
        suite: Benchmark suite that owns the task.
        raw: Original JSONL task payload.
    """

    task_id: str
    framework: Framework
    prompt: str
    entry_point: str
    category: str | None
    canonical_class: Mapping[str, Any] | None
    suite: Suite = "core"
    raw: Mapping[str, Any] = field(default_factory=dict)

    def provider_metadata(self) -> dict[str, Any]:
        """Return request metadata for runners and smoke providers.

        Built-in LLM providers (OpenRouter, Coda) must never forward this
        metadata into model prompts. It may include grader class and canonical
        solution text for local smoke providers only.

        Returns:
            Dictionary containing category and grader class.  When a bundled
            canonical solution exists, it is included so smoke providers can
            return deterministic reference code without rereading assets.
        """
        metadata = {
            "category": self.category,
            "canonical_class": self.canonical_class,
        }
        if "canonical_solution" in self.raw:
            metadata["canonical_solution"] = self.raw["canonical_solution"]
        return metadata


@dataclass(frozen=True)
class QCEvalEvaluation:
    """Evaluation result for generated candidate code.

    Evaluation separates provider success from benchmark success.  ``compiled``
    and ``ran`` describe execution, while ``passed`` and ``metric`` describe the
    grader result.  This distinction lets reports split syntax errors, runtime
    errors, and semantic benchmark failures.

    Attributes:
        compiled: Whether candidate source compiled.
        ran: Whether compiled code ran to a framework result.
        passed: Whether grader accepted the result.
        metric: Primary grader metric, if available.
        metric_name: Name of the primary grader metric, if available.
        probabilities: Probability vector produced by the candidate.
        execution_metadata: Framework-specific execution metadata.
        grader_details: Full grader output.
        verified_status: Four-state verifier verdict.
        semantic_result: Versioned behavior-first result, when run.
        error: Captured error or failure detail.
        error_type: Stable error category used by reports.
    """

    compiled: bool
    ran: bool
    passed: bool
    metric: float | int | str | None = None
    metric_name: str | None = None
    probabilities: list[float] | None = None
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)
    grader_details: Mapping[str, Any] = field(default_factory=dict)
    verified_status: str | None = None
    semantic_result: Mapping[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evaluation data."""
        return to_jsonable(self)


@dataclass(frozen=True)
class BenchmarkRecord:
    """Complete provider and evaluator result for one task.

    A record is the atomic result written to JSONL streams and stored in final
    JSON output.  It combines immutable task identity, provider metadata,
    provider response, and optional evaluation data.

    Attributes:
        framework: Quantum framework targeted by the task.
        task_id: Zero-padded task identifier.
        entry_point: Function name the candidate was expected to define.
        category: Optional task category.
        provider: Provider implementation name.
        model: Model identifier used for generation.
        status: Stable outcome category for summaries.
        provider_response: Raw provider outcome.
        evaluation: Grader outcome, or ``None`` when provider generation failed.
        suite: Benchmark suite that owns the task.
        sample_index: Repeated-sample index for Pass@K runs.
        attempt_index: Feedback-repair attempt index.
        feedback: Feedback metadata used for repair attempts.
        request_trace: Exact provider-message transcript and stable hashes.
        lineage: Versioned chain and parent-attempt provenance.
        error_taxonomy: Versioned multi-label error classification. Historical
            records may omit it and derive it from their persisted evidence.
    """

    framework: Framework
    task_id: str
    entry_point: str
    category: str | None
    provider: str
    model: str | None
    status: OutcomeStatus
    provider_response: ProviderResponse
    evaluation: QCEvalEvaluation | None
    suite: Suite = "core"
    sample_index: int = 0
    attempt_index: int = 0
    feedback: Mapping[str, Any] = field(default_factory=dict)
    request_trace: Mapping[str, Any] = field(default_factory=dict)
    lineage: Mapping[str, Any] = field(default_factory=dict)
    error_taxonomy: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible benchmark record data."""
        evaluation = None if self.evaluation is None else self.evaluation.to_dict()
        taxonomy = self.error_taxonomy
        if taxonomy is None:
            taxonomy = classify_error_taxonomy(self.status, evaluation)
        return {
            "framework": self.framework,
            "suite": self.suite,
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "attempt_index": self.attempt_index,
            "entry_point": self.entry_point,
            "category": self.category,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "feedback": to_jsonable(self.feedback),
            "request_trace": to_jsonable(self.request_trace),
            "lineage": to_jsonable(self.lineage),
            "provider_response": self.provider_response.to_dict(),
            "evaluation": evaluation,
            "error_taxonomy": to_jsonable(taxonomy),
        }
