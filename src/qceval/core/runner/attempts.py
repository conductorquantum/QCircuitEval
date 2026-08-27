"""Feedback-attempt helpers for benchmark runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from qceval.core.cache import ResponseCache
from qceval.core.feedback import build_feedback, feedback_stop_reason
from qceval.core.io import JsonlRunWriter
from qceval.core.lineage import sha256_text
from qceval.core.runner.messages import _feedback_messages
from qceval.core.runner.records import (
    _chain_records,
    _completed_chain,
    _feedback_record_order,
    _job_key,
    _status,
)
from qceval.core.runner.types import RunJob
from qceval.models import (
    BenchmarkRecord,
    Framework,
    ProviderRequest,
    ProviderResponse,
    QCEvalTask,
    RunConfig,
    RunOptions,
    Suite,
)


class AttemptMixin:
    """Feedback attempt scheduling behavior."""

    config: RunConfig
    options: RunOptions

    def _run_jobs_feedback_leveled(self, jobs: list[RunJob]) -> list[BenchmarkRecord]:
        runner = cast(Any, self)
        completed = runner._completed_records()
        writer = None if self.options.stream_to is None else JsonlRunWriter(self.options.stream_to)
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord] = {}
        task_order = {
            (job.task.suite, job.task.framework, job.task.task_id, job.sample_index): i for i, job in enumerate(jobs)
        }
        pending = self._restore_feedback_jobs(jobs, completed, records_by_key, writer)
        cache = None if self.options.cache_dir is None else ResponseCache(self.options.cache_dir)
        try:
            while pending:
                attempt_index = min(job.attempt_index for job in pending)
                level_jobs = _dense_jobs([job for job in pending if job.attempt_index == attempt_index])
                pending = [job for job in pending if job.attempt_index != attempt_index]
                next_jobs = self._run_feedback_level(level_jobs, cache, writer, records_by_key)
                pending.extend(next_jobs)
        finally:
            if writer is not None:
                writer.close()
        return sorted(records_by_key.values(), key=lambda record: _feedback_record_order(record, task_order))

    def _run_feedback_level(
        self,
        level_jobs: list[RunJob],
        cache: ResponseCache | None,
        writer: JsonlRunWriter | None,
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
    ) -> list[RunJob]:
        runner = cast(Any, self)
        next_jobs: list[RunJob] = []
        progress_bar = runner._progress_bar(len(level_jobs))
        try:
            level_results = self._run_attempt_jobs(level_jobs, cache, writer, progress_bar)
        finally:
            runner._close_progress(progress_bar)
        for job, record in level_results:
            records_by_key[_job_key(job)] = record
            if feedback_stop_reason(record) is not None or job.attempt_index >= self.config.max_attempts - 1:
                continue
            next_jobs.append(self._next_feedback_job(job, records_by_key, index=len(next_jobs)))
        return next_jobs

    def _restore_feedback_jobs(
        self,
        jobs: list[RunJob],
        completed: Mapping[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        writer: JsonlRunWriter | None,
    ) -> list[RunJob]:
        pending: list[RunJob] = []
        for base_job in jobs:
            chain_records = _completed_chain(completed, base_job)
            if not chain_records:
                pending.append(base_job)
                continue
            self._restore_feedback_chain(base_job, chain_records, records_by_key, writer, pending)
        return pending

    def _restore_feedback_chain(
        self,
        base_job: RunJob,
        chain_records: Mapping[int, BenchmarkRecord],
        records_by_key: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        writer: JsonlRunWriter | None,
        pending: list[RunJob],
    ) -> None:
        self._reject_feedback_gaps(base_job, chain_records)
        self._reject_post_terminal_attempts(base_job, chain_records)
        for attempt_index, record in sorted(chain_records.items()):
            records_by_key[
                (
                    base_job.task.suite,
                    base_job.task.framework,
                    base_job.task.task_id,
                    base_job.sample_index,
                    attempt_index,
                )
            ] = record
            if writer is not None:
                writer.append(record)
        last_attempt = max(chain_records)
        last_record = chain_records[last_attempt]
        if feedback_stop_reason(last_record) is None and last_attempt < self.config.max_attempts - 1:
            pending.append(self._next_feedback_job(base_job, records_by_key, index=len(pending)))

    def _reject_feedback_gaps(self, base_job: RunJob, chain_records: Mapping[int, BenchmarkRecord]) -> None:
        last_attempt = max(chain_records)
        for attempt_index in range(last_attempt + 1):
            if attempt_index not in chain_records:
                task = base_job.task
                raise ValueError(
                    "resume data has a feedback gap for "
                    f"{task.suite}:{task.framework}:{task.task_id} sample {base_job.sample_index}: "
                    f"missing attempt {attempt_index} before attempt {last_attempt}"
                )

    def _reject_post_terminal_attempts(
        self,
        base_job: RunJob,
        chain_records: Mapping[int, BenchmarkRecord],
    ) -> None:
        last_attempt = max(chain_records)
        for attempt_index, record in sorted(chain_records.items()):
            if attempt_index < last_attempt and feedback_stop_reason(record) is not None:
                task = base_job.task
                raise ValueError(
                    "resume data continues after a terminal feedback outcome for "
                    f"{task.suite}:{task.framework}:{task.task_id} sample {base_job.sample_index}: "
                    f"attempt {attempt_index} is terminal"
                )

    def _next_feedback_job(
        self,
        base_job: RunJob,
        records_by_key: Mapping[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        *,
        index: int,
    ) -> RunJob:
        previous_records = _chain_records(records_by_key, base_job)
        previous = previous_records[-1]
        feedback = build_feedback(
            previous,
            max_chars=self.config.feedback_max_chars,
            policy=self.config.feedback_policy,
        )
        attempt_index = previous.attempt_index + 1
        request = self._feedback_request(
            base_job.task,
            sample_index=base_job.sample_index,
            attempt_index=attempt_index,
            previous_records=previous_records,
            feedback_message=feedback.message_to_model,
            feedback_metadata=feedback.metadata,
        )
        return RunJob(
            index=index,
            task=base_job.task,
            request=request,
            sample_index=base_job.sample_index,
            attempt_index=attempt_index,
            feedback=feedback.metadata,
            parent_attempt_index=previous.attempt_index,
            parent_code_sha256=sha256_text(previous.provider_response.code),
        )

    def _feedback_request(
        self,
        task: QCEvalTask,
        *,
        sample_index: int,
        attempt_index: int,
        previous_records: Sequence[BenchmarkRecord],
        feedback_message: str,
        feedback_metadata: Mapping[str, Any],
    ) -> ProviderRequest:
        metadata = cast(Any, self)._provider_metadata(task)
        metadata.update({"suite": task.suite, "sample_index": sample_index, "attempt_index": attempt_index})
        metadata["feedback"] = dict(feedback_metadata)
        return ProviderRequest(
            task_id=task.task_id,
            framework=task.framework,
            prompt=task.prompt,
            entry_point=task.entry_point,
            model=self.config.model,
            metadata=metadata,
            sample_index=sample_index,
            attempt_index=attempt_index,
            messages=_feedback_messages(task.prompt, previous_records, feedback_message),
        )

    def _run_attempt_jobs(
        self,
        jobs: list[RunJob],
        cache: ResponseCache | None,
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> list[tuple[RunJob, BenchmarkRecord]]:
        if self.options.task_timeout is not None:
            return self._run_attempt_jobs_isolated(jobs, writer, progress_bar)
        runner = cast(Any, self)
        out: list[tuple[RunJob, BenchmarkRecord]] = []
        pending_evaluation: list[tuple[RunJob, ProviderResponse]] = []
        for job, provider_response in runner._generate(jobs, cache):
            if provider_response.ok:
                pending_evaluation.append((job, provider_response))
            else:
                record = runner._provider_failure_record(job, provider_response)
                self._append_attempt_record(out, job, record, writer, progress_bar, total=len(jobs))
        for job, provider_response, evaluation in runner._evaluate(pending_evaluation):
            record = runner._record(job, provider_response, evaluation, _status(evaluation))
            self._append_attempt_record(out, job, record, writer, progress_bar, total=len(jobs))
        return sorted(out, key=lambda item: item[0].index)

    def _run_attempt_jobs_isolated(
        self,
        jobs: list[RunJob],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> list[tuple[RunJob, BenchmarkRecord]]:
        runner = cast(Any, self)
        records: list[BenchmarkRecord | None] = [None] * len(jobs)
        runner._run_pending_isolated(jobs, records, writer, progress_bar)
        return [(job, record) for job, record in zip(jobs, records, strict=True) if record is not None]

    def _append_attempt_record(
        self,
        out: list[tuple[RunJob, BenchmarkRecord]],
        job: RunJob,
        record: BenchmarkRecord,
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
        *,
        total: int,
    ) -> None:
        out.append((job, record))
        if writer is not None:
            writer.append(record)
        cast(Any, self)._emit_progress(record, len(out), total, progress_bar)


def _dense_jobs(jobs: list[RunJob]) -> list[RunJob]:
    return [replace(job, index=index) for index, job in enumerate(jobs)]
