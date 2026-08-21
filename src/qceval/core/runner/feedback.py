"""Feedback-repair runner loops."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast

from qceval.core.cache import ResponseCache
from qceval.core.feedback import feedback_stop_reason
from qceval.core.io import JsonlRunWriter
from qceval.core.runner.attempts import AttemptMixin
from qceval.core.runner.records import _chain_key, _feedback_record_order, _job_key, _status
from qceval.core.runner.types import RunJob
from qceval.models import BenchmarkRecord, Framework, ProviderResponse, RunConfig, RunOptions, Suite
from qceval.typing import TaskAdapter


class FeedbackMixin(AttemptMixin):
    """Feedback repair behavior shared by runner implementations."""

    adapter: TaskAdapter
    config: RunConfig
    options: RunOptions

    def _run_jobs_feedback(self, jobs: list[RunJob]) -> list[BenchmarkRecord]:
        runner = cast(Any, self)
        if self.options.task_timeout is not None:
            return self._run_jobs_feedback_leveled(jobs)

        completed = runner._completed_records()
        writer = None if self.options.stream_to is None else JsonlRunWriter(self.options.stream_to)
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord] = {}
        task_order = {_chain_key(job): position for position, job in enumerate(jobs)}
        cache = None if self.options.cache_dir is None else ResponseCache(self.options.cache_dir)

        gen_queue: deque[RunJob] = deque(self._restore_feedback_jobs(jobs, completed, records_by_key, writer))
        progress_bar = runner._progress_bar(max(self._total_expected(jobs, records_by_key), len(records_by_key)))

        try:
            if self.options.generation_concurrency <= 1:
                self._feedback_loop_serial(gen_queue, cache, records_by_key, writer, progress_bar)
            else:
                self._feedback_loop_concurrent(gen_queue, cache, records_by_key, writer, progress_bar)
        finally:
            runner._close_progress(progress_bar)
            if writer is not None:
                writer.close()

        return sorted(records_by_key.values(), key=lambda record: _feedback_record_order(record, task_order))

    def _feedback_loop_serial(
        self,
        gen_queue: deque[RunJob],
        cache: ResponseCache | None,
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        runner = cast(Any, self)
        while gen_queue:
            job = gen_queue.popleft()
            response = runner._generate_one(job, cache)
            if response.ok:
                evaluation = self.adapter.evaluate(job.task, response.code or "")
                record = runner._record(job, response, evaluation, _status(evaluation))
            else:
                record = runner._provider_failure_record(job, response)
            self._handle_feedback_completion(job, record, records_by_key, gen_queue, writer, progress_bar)

    def _feedback_loop_concurrent(
        self,
        gen_queue: deque[RunJob],
        cache: ResponseCache | None,
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        eval_queue: list[tuple[RunJob, ProviderResponse]] = []
        with ThreadPoolExecutor(max_workers=self.options.generation_concurrency) as pool:
            gen_futures: dict[Future[ProviderResponse], RunJob] = {}
            while gen_queue or gen_futures or eval_queue:
                self._fill_feedback_generation(pool, gen_queue, gen_futures, cache)
                done_futures = [future for future in gen_futures if future.done()]
                self._collect_feedback_generation(
                    done_futures,
                    gen_futures,
                    eval_queue,
                    records_by_key,
                    gen_queue,
                    writer,
                    progress_bar,
                )
                self._drain_feedback_evaluation(eval_queue, records_by_key, gen_queue, writer, progress_bar)
                if gen_futures and not done_futures and not eval_queue:
                    time.sleep(0.01)

    def _fill_feedback_generation(
        self,
        pool: ThreadPoolExecutor,
        gen_queue: deque[RunJob],
        gen_futures: dict[Future[ProviderResponse], RunJob],
        cache: ResponseCache | None,
    ) -> None:
        while gen_queue and len(gen_futures) < self.options.generation_concurrency:
            job = gen_queue.popleft()
            gen_futures[pool.submit(cast(Any, self)._generate_one, job, cache)] = job

    def _collect_feedback_generation(
        self,
        done_futures: list[Future[ProviderResponse]],
        gen_futures: dict[Future[ProviderResponse], RunJob],
        eval_queue: list[tuple[RunJob, ProviderResponse]],
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        gen_queue: deque[RunJob],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        runner = cast(Any, self)
        for future in done_futures:
            job = gen_futures.pop(future)
            response = future.result()
            if response.ok:
                eval_queue.append((job, response))
                continue
            record = runner._provider_failure_record(job, response)
            self._handle_feedback_completion(job, record, records_by_key, gen_queue, writer, progress_bar)

    def _drain_feedback_evaluation(
        self,
        eval_queue: list[tuple[RunJob, ProviderResponse]],
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        gen_queue: deque[RunJob],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        if not eval_queue:
            return
        runner = cast(Any, self)
        batch = list(eval_queue)
        eval_queue.clear()
        for job, response, evaluation in runner._evaluate(batch):
            record = runner._record(job, response, evaluation, _status(evaluation))
            self._handle_feedback_completion(job, record, records_by_key, gen_queue, writer, progress_bar)

    def _handle_feedback_completion(
        self,
        job: RunJob,
        record: BenchmarkRecord,
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        gen_queue: deque[RunJob],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        records_by_key[_job_key(job)] = record
        if writer is not None:
            writer.append(record)
        cast(Any, self)._emit_progress(record, len(records_by_key), len(records_by_key), progress_bar)
        if feedback_stop_reason(record) is not None or job.attempt_index >= self.config.max_attempts - 1:
            return
        gen_queue.append(self._next_feedback_job(job, records_by_key, index=len(gen_queue)))

    def _total_expected(
        self,
        jobs: list[RunJob],
        records_by_key: Mapping[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
    ) -> int:
        return sum(
            1
            if records_by_key.get(_job_key(job)) is not None
            and feedback_stop_reason(records_by_key[_job_key(job)]) is not None
            else self.config.max_attempts
            for job in jobs
        )
