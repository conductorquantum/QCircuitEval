"""Process-pool evaluation helpers for benchmark runs."""

from __future__ import annotations

import time
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from multiprocessing import get_context
from typing import Any

from qceval.core.bench import Adaptor
from qceval.core.runner.processes import terminate_worker_process
from qceval.core.runner.records import _evaluation_from_dict, _runtime_error_evaluation, _timeout_evaluation
from qceval.core.runner.types import RunJob
from qceval.core.runner.workers import _worker_evaluate, _worker_init
from qceval.models import ProviderResponse, QCEvalEvaluation, RunConfig, RunOptions
from qceval.typing import TaskAdapter


class EvaluationMixin:
    """Evaluation worker behavior shared by runner implementations."""

    adapter: TaskAdapter
    config: RunConfig
    options: RunOptions

    def _evaluate(
        self, jobs: list[tuple[RunJob, ProviderResponse]]
    ) -> Iterable[tuple[RunJob, ProviderResponse, QCEvalEvaluation]]:
        if not jobs:
            return []
        out: list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]] = []
        with self._evaluation_scheduler() as scheduler:
            for job, provider_response in jobs:
                out.extend(scheduler.submit(job, provider_response))
            out.extend(scheduler.drain_all())
        return sorted(out, key=lambda item: item[0].index)

    def _evaluation_scheduler(self) -> EvaluationScheduler:
        return EvaluationScheduler(adapter=self.adapter, config=self.config, options=self.options)


class EvaluationScheduler:
    """Incremental evaluator for streaming runner paths.

    Serial mode evaluates inline.  Concurrent or timeout mode keeps one process
    pool open for the full scheduling window so ``evaluation_workers`` avoids
    per-record pool startup overhead.
    """

    def __init__(self, *, adapter: TaskAdapter, config: RunConfig, options: RunOptions) -> None:
        self._adapter = adapter
        self._config = config
        self._options = options
        self._capacity = max(1, options.evaluation_workers)
        self._source_hint = None if config.source_hint is None else str(config.source_hint)
        self._executor: ProcessPoolExecutor | None = None
        self._pending: dict[Future[dict[str, Any]], tuple[RunJob, ProviderResponse, float]] = {}
        if isinstance(adapter, Adaptor) or options.evaluation_workers > 1 or options.eval_timeout is not None:
            self._executor = self._new_executor()

    def __enter__(self) -> EvaluationScheduler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def submit(
        self, job: RunJob, provider_response: ProviderResponse
    ) -> list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]]:
        """Submit one provider response for evaluation.

        Args:
            job: Runner job associated with the provider response.
            provider_response: Generated code response to evaluate.

        Returns:
            Any evaluations already complete after submission.
        """
        if self._executor is None:
            return [(job, provider_response, self._adapter.evaluate(job.task, provider_response.code or ""))]
        out = self._wait_for_capacity()
        future = self._submit_future(job, provider_response)
        self._pending[future] = (job, provider_response, time.monotonic())
        out.extend(self.drain_ready())
        return sorted(out, key=lambda item: item[0].index)

    def _new_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._capacity,
            mp_context=get_context("spawn"),
            initializer=_worker_init,
            initargs=(self._source_hint,),
        )

    def _submit_future(self, job: RunJob, provider_response: ProviderResponse) -> Future[dict[str, Any]]:
        assert self._executor is not None
        return self._executor.submit(
            _worker_evaluate,
            job.task.suite,
            job.task.framework,
            job.task.task_id,
            job.task.entry_point,
            provider_response.code or "",
        )

    def _wait_for_capacity(self) -> list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]]:
        """Wait for a worker slot before starting another timeout clock.

        ``ProcessPoolExecutor.submit`` accepts an unbounded queue. Starting the
        timeout when work enters that queue incorrectly charges queued jobs for
        earlier evaluations. Keep at most one pending future per worker so the
        submission timestamp closely tracks the start of execution.
        """
        out: list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]] = []
        while len(self._pending) >= self._capacity:
            done, _ = wait(set(self._pending), timeout=0.05, return_when=FIRST_COMPLETED)
            out.extend(self._collect_done(done))
            out.extend(self._collect_timeouts())
        return out

    def drain_ready(self) -> list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]]:
        """Return completed or timed-out evaluations without blocking."""
        done = [future for future in self._pending if future.done()]
        out = self._collect_done(done)
        out.extend(self._collect_timeouts())
        return sorted(out, key=lambda item: item[0].index)

    def drain_all(self) -> list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]]:
        """Block until all submitted evaluations have terminal records.

        Returns:
            Completed evaluations sorted by job index.
        """
        out: list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]] = []
        while self._pending:
            done, _ = wait(set(self._pending), timeout=0.05, return_when=FIRST_COMPLETED)
            out.extend(self._collect_done(done))
            out.extend(self._collect_timeouts())
        return sorted(out, key=lambda item: item[0].index)

    def close(self) -> None:
        """Close the process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _collect_done(
        self, done: Iterable[Future[dict[str, Any]]]
    ) -> list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]]:
        out: list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]] = []
        for future in done:
            item = self._pending.pop(future, None)
            if item is None:
                continue
            job, provider_response, _ = item
            try:
                evaluation = _evaluation_from_dict(future.result())
            except Exception as exc:
                evaluation = _runtime_error_evaluation(f"{type(exc).__name__}: {exc}")
            out.append((job, provider_response, evaluation))
        return out

    def _collect_timeouts(self) -> list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]]:
        timeout = self._options.eval_timeout
        if timeout is None:
            return []
        now = time.monotonic()
        out: list[tuple[RunJob, ProviderResponse, QCEvalEvaluation]] = []
        expired = [future for future, (_, _, submitted_at) in self._pending.items() if now - submitted_at >= timeout]
        if not expired:
            return out
        for future in expired:
            job, provider_response, _ = self._pending.pop(future)
            future.cancel()
            out.append((job, provider_response, _timeout_evaluation(timeout)))
        self._recycle_executor()
        return out

    def _recycle_executor(self) -> None:
        """Kill timed-out workers and restart unfinished evaluations.

        A running ``ProcessPoolExecutor`` future cannot be cancelled. Leaving
        its worker alive would make later jobs queue behind the hung candidate
        and inherit false timeouts. Terminate the contaminated pool, then
        resubmit only the unfinished, non-expired jobs with fresh start clocks.
        """
        executor = self._executor
        if executor is None:
            return
        survivors = [item[:2] for item in self._pending.values()]
        self._pending.clear()
        processes = tuple(getattr(executor, "_processes", {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            terminate_worker_process(process)
        self._executor = self._new_executor()
        for job, provider_response in survivors:
            future = self._submit_future(job, provider_response)
            self._pending[future] = (job, provider_response, time.monotonic())
