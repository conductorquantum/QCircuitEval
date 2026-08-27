"""Structural protocols for pluggable providers and task adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeGuard

from qceval.models import Framework, ProviderRequest, ProviderResponse, QCEvalEvaluation, QCEvalTask, Suite


class Provider(Protocol):
    """Code-generation provider used by the benchmark runner.

    A provider receives one :class:`qceval.models.ProviderRequest` and returns a
    serializable :class:`qceval.models.ProviderResponse`.  Implementations should
    encode remote API errors in the response rather than raising provider
    exceptions, which lets the runner produce complete output.

    Attributes:
        name: Stable provider name used in output and cache keys.
    """

    name: str

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate candidate code for one request.

        Args:
            request: Task prompt and metadata.

        Returns:
            Provider response containing code or an error message.
        """
        ...


class BatchProvider(Provider, Protocol):
    """Optional provider protocol for ordered batch generation.

    Batch providers can exploit remote batching while preserving runner result
    order.  ``generate_many`` must return one response per input request in the
    same order.
    """

    def generate_many(self, requests: Sequence[ProviderRequest]) -> list[ProviderResponse]:
        """Generate candidate code for ordered requests.

        Args:
            requests: Ordered provider requests.

        Returns:
            Ordered provider responses with the same length as ``requests``.
        """
        ...


class TaskAdapter(Protocol):
    """Task source and evaluator used by the benchmark runner.

    Adapters make runner execution independent from bundled assets.  A custom
    adapter can load a different task set or use a different grader while
    preserving provider and output contracts.
    """

    def load_tasks(self, framework: Framework, suite: Suite = "core") -> Sequence[QCEvalTask]:
        """Load tasks for one framework.

        Args:
            framework: Framework literal to load.
            suite: Benchmark suite to load.

        Returns:
            Ordered sequence of tasks for the framework.
        """
        ...

    def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
        """Evaluate candidate source for a task.

        Args:
            task: Task to grade.
            code: Candidate Python source.

        Returns:
            Evaluation result.
        """
        ...

    def metadata(self) -> Mapping[str, Any]:
        """Return adapter provenance metadata for run output."""
        ...


def is_batch_provider(provider: Provider) -> TypeGuard[BatchProvider]:
    """Return whether provider implements ordered batch generation.

    Args:
        provider: Provider object to inspect.

    Returns:
        ``True`` when ``provider`` exposes a callable ``generate_many`` method.
    """
    return callable(getattr(provider, "generate_many", None))
