"""Worker-process entry points for runner evaluation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout, suppress
from pathlib import Path
from typing import Any

from qceval.core.bench import Adaptor
from qceval.core.cache import ResponseCache
from qceval.core.runner.records import _runtime_error_evaluation, _status
from qceval.models import (
    BenchmarkRecord,
    Framework,
    OutcomeStatus,
    ProviderRequest,
    ProviderResponse,
    QCEvalTask,
    Suite,
)
from qceval.providers.base import Provider

_WORKER_ADAPTER: Adaptor | None = None
MAX_CANDIDATE_OUTPUT_CHARS = 65_536


class _BoundedWriter:
    def __init__(self, limit: int = MAX_CANDIDATE_OUTPUT_CHARS) -> None:
        self.limit = limit
        self.written = 0

    def write(self, value: str) -> int:
        admitted = max(0, min(len(value), self.limit - self.written))
        self.written += admitted
        return len(value)

    def flush(self) -> None:
        return None


def _worker_init(source_hint: str | None) -> None:
    global _WORKER_ADAPTER
    _start_process_group()
    # Build once per worker so package asset reads and canonical caches are
    # local to that process instead of pickled for every submitted task.
    _WORKER_ADAPTER = Adaptor(source_hint)


def _worker_evaluate(suite: Suite, framework: Framework, task_id: str, entry_point: str, code: str) -> dict[str, Any]:
    if _WORKER_ADAPTER is None:
        raise RuntimeError("qceval worker adapter is not initialized")
    task = next(task for task in _WORKER_ADAPTER.load_tasks(framework, suite=suite) if task.task_id == task_id)
    stdout = _BoundedWriter()
    stderr = _BoundedWriter()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return _WORKER_ADAPTER.evaluate(task, code).to_dict()


def _worker_run_job(
    provider: Provider,
    provider_name: str,
    task: QCEvalTask,
    request: ProviderRequest,
    model: str | None,
    provider_config: Mapping[str, Any],
    cache_dir: str | None,
    source_hint: str | None,
    queue: Any,
) -> None:
    _start_process_group()
    provider_response: ProviderResponse | None = None
    try:
        provider_response = _worker_generate(provider, request, provider_name, provider_config, cache_dir)
        evaluation = None
        status: OutcomeStatus = "provider_failed"
        if provider_response.ok:
            evaluation = Adaptor(source_hint).evaluate(task, provider_response.code or "")
            status = _status(evaluation)
        record = BenchmarkRecord(
            framework=task.framework,
            suite=task.suite,
            task_id=task.task_id,
            sample_index=_request_index(request, "sample_index"),
            attempt_index=_request_index(request, "attempt_index"),
            entry_point=task.entry_point,
            category=task.category,
            provider=provider_name,
            model=provider_response.model or model,
            status=status,
            feedback=request.metadata.get("feedback") or {},
            provider_response=provider_response,
            evaluation=evaluation,
        )
    except Exception as exc:
        infrastructure = provider_response is not None
        response = provider_response or ProviderResponse(
            code=None,
            model=model,
            error=f"{type(exc).__name__}: {exc}",
        )
        record = BenchmarkRecord(
            framework=task.framework,
            suite=task.suite,
            task_id=task.task_id,
            sample_index=_request_index(request, "sample_index"),
            attempt_index=_request_index(request, "attempt_index"),
            entry_point=task.entry_point,
            category=task.category,
            provider=provider_name,
            model=model,
            status="infrastructure_error" if infrastructure else "provider_failed",
            feedback=request.metadata.get("feedback") or {},
            provider_response=response,
            evaluation=_runtime_error_evaluation(f"{type(exc).__name__}: {exc}") if infrastructure else None,
        )
    queue.put(record.to_dict())


def _start_process_group() -> None:
    if hasattr(os, "setsid"):
        with suppress(OSError):
            os.setsid()


def _request_index(request: ProviderRequest, name: str) -> int:
    field_value = getattr(request, name)
    if isinstance(field_value, int):
        return field_value
    value = request.metadata.get(name, 0)
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _worker_generate(
    provider: Provider,
    request: ProviderRequest,
    provider_name: str,
    provider_config: Mapping[str, Any],
    cache_dir: str | None,
) -> ProviderResponse:
    if cache_dir is None:
        return provider.generate(request)
    cache = ResponseCache(Path(cache_dir))
    cache_key = cache.key_for(request, provider=provider_name, settings=provider_config)
    response = cache.get(cache_key)
    if response is not None:
        return response
    response = provider.generate(request)
    cache.put(cache_key, response)
    return response
