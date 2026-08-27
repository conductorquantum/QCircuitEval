"""Task-level process isolation helpers for benchmark runs."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import suppress
from multiprocessing import get_context
from queue import Empty
from typing import Any, cast

from qceval.core.bench import Adaptor
from qceval.core.io import JsonlRunWriter
from qceval.core.runner.processes import terminate_worker_process
from qceval.core.runner.records import (
    _job_key,
    _record_from_dict,
    _runtime_error_evaluation,
    _timeout_evaluation,
)
from qceval.core.runner.types import RunJob, _ActiveIsolatedJob
from qceval.core.runner.workers import _worker_run_job
from qceval.models import BenchmarkRecord, Framework, ProviderResponse, RunConfig, RunOptions, Suite
from qceval.providers.base import Provider
from qceval.typing import TaskAdapter


class IsolationMixin:
    """Task timeout behavior shared by runner implementations."""

    adapter: TaskAdapter
    config: RunConfig
    options: RunOptions
    provider: Provider

    def _run_jobs_isolated(self, jobs: list[RunJob]) -> list[BenchmarkRecord]:
        if not isinstance(self.adapter, Adaptor):
            raise ValueError("task_timeout requires qceval.core.bench.Adaptor")
        runner = cast(Any, self)
        # Read resume records before opening the writer so a same-path resume
        # source is never truncated before it is consumed.
        completed = runner._completed_records()
        writer = None if self.options.stream_to is None else JsonlRunWriter(self.options.stream_to)
        records: list[BenchmarkRecord | None] = [None] * len(jobs)
        progress_bar = runner._progress_bar(len(jobs))
        try:
            pending = self._restore_completed(jobs, completed, records, writer, progress_bar)
            self._run_pending_isolated(pending, records, writer, progress_bar)
        finally:
            runner._close_progress(progress_bar)
            if writer is not None:
                writer.close()
        return [record for record in records if record is not None]

    def _restore_completed(
        self,
        jobs: list[RunJob],
        completed: Mapping[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        records: list[BenchmarkRecord | None],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> list[RunJob]:
        runner = cast(Any, self)
        pending: list[RunJob] = []
        for job in jobs:
            record = completed.get(_job_key(job))
            if record is None:
                pending.append(job)
                continue
            runner._store_record(records, job, record, writer, progress_bar)
        return pending

    def _run_pending_isolated(
        self,
        pending: list[RunJob],
        records: list[BenchmarkRecord | None],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        pending = self._record_harness_error_jobs(pending, records, writer, progress_bar)
        context = get_context("spawn")
        active: list[_ActiveIsolatedJob] = []
        next_index = 0
        while next_index < len(pending) or active:
            while next_index < len(pending) and len(active) < self.options.generation_concurrency:
                active.append(self._start_isolated_job(context, pending[next_index]))
                next_index += 1
            active, completed_any = self._drain_isolated_jobs(active, records, writer, progress_bar)
            if active and not completed_any:
                time.sleep(0.05)

    def _record_harness_error_jobs(
        self,
        pending: list[RunJob],
        records: list[BenchmarkRecord | None],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> list[RunJob]:
        """Record typed harness-error attempts without spawning workers.

        Jobs flagged with a harness error (for example, an oracle-safety
        containment during feedback construction) fail only that attempt and
        must never reach a provider.
        """
        runner = cast(Any, self)
        runnable: list[RunJob] = []
        for job in pending:
            if not job.feedback.get("harness_error"):
                runnable.append(job)
                continue
            response = runner._harness_error_response(job)
            record = runner._record(
                job,
                response,
                _runtime_error_evaluation("oracle isolation failure"),
                "infrastructure_error",
            )
            runner._store_record(records, job, record, writer, progress_bar)
        return runnable

    def _start_isolated_job(self, context: Any, job: RunJob) -> _ActiveIsolatedJob:
        queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_worker_run_job,
            args=(
                self.provider,
                self.provider.name,
                job.task,
                job.request,
                self.config.model,
                dict(self.config.provider_config),
                None if self.options.cache_dir is None else str(self.options.cache_dir),
                None if self.config.source_hint is None else str(self.config.source_hint),
                queue,
            ),
        )
        process.start()
        return _ActiveIsolatedJob(job=job, process=process, queue=queue, started_at=time.monotonic())

    def _drain_isolated_jobs(
        self,
        active_jobs: list[_ActiveIsolatedJob],
        records: list[BenchmarkRecord | None],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> tuple[list[_ActiveIsolatedJob], bool]:
        completed_any = False
        still_active: list[_ActiveIsolatedJob] = []
        for active in active_jobs:
            record = self._isolated_result(active)
            if record is None and self._isolated_timed_out(active):
                record = self._timeout_record(active.job, self.options.task_timeout or 0.0)
                self._stop_isolated_process(active)
            if record is None:
                still_active.append(active)
                continue
            self._finish_isolated_job(active, record, records, writer, progress_bar)
            completed_any = True
        return still_active, completed_any

    def _isolated_result(self, active: _ActiveIsolatedJob) -> BenchmarkRecord | None:
        try:
            payload = active.queue.get_nowait()
        except Empty:
            if active.process.is_alive():
                return None
            return self._result_after_process_exit(active)
        except Exception as exc:
            return self._worker_exit_record(active, f"{type(exc).__name__}: {exc}")
        return _record_from_dict(payload)

    def _result_after_process_exit(self, active: _ActiveIsolatedJob) -> BenchmarkRecord:
        active.process.join(timeout=0.1)
        try:
            payload = active.queue.get(timeout=0.1)
        except Empty:
            return self._worker_exit_record(active)
        except Exception as exc:
            return self._worker_exit_record(active, f"{type(exc).__name__}: {exc}")
        return _record_from_dict(payload)

    def _isolated_timed_out(self, active: _ActiveIsolatedJob) -> bool:
        timeout = self.options.task_timeout
        return timeout is not None and time.monotonic() - active.started_at >= timeout

    def _finish_isolated_job(
        self,
        active: _ActiveIsolatedJob,
        record: BenchmarkRecord,
        records: list[BenchmarkRecord | None],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        runner = cast(Any, self)
        active.process.join(timeout=0.1)
        self._close_isolated_queue(active)
        record = runner._record(
            active.job,
            record.provider_response,
            record.evaluation,
            record.status,
        )
        runner._store_record(records, active.job, record, writer, progress_bar)

    def _stop_isolated_process(self, active: _ActiveIsolatedJob) -> None:
        terminate_worker_process(active.process)
        self._close_isolated_queue(active)

    def _close_isolated_queue(self, active: _ActiveIsolatedJob) -> None:
        for method in ("close", "join_thread"):
            with suppress(Exception):
                getattr(active.queue, method)()

    def _worker_exit_record(self, active: _ActiveIsolatedJob, error: str | None = None) -> BenchmarkRecord:
        exitcode = active.process.exitcode
        message = error or f"task worker exited without result, exitcode={exitcode}"
        runner = cast(Any, self)
        return runner._record(
            active.job,
            ProviderResponse(code=None, model=self.config.model, metadata={"exitcode": exitcode}, error=message),
            _runtime_error_evaluation(message),
            "infrastructure_error",
        )

    def _timeout_record(self, job: RunJob, timeout: float) -> BenchmarkRecord:
        response = ProviderResponse(
            code=None,
            model=self.config.model,
            metadata={"timeout_seconds": timeout},
            error=f"task timed out after {timeout:.3f}s",
        )
        runner = cast(Any, self)
        return runner._record(job, response, _timeout_evaluation(timeout), "infrastructure_error")
