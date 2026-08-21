"""Benchmark runner public implementation."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import replace
from itertools import chain
from typing import Any

from qceval.core.cache import ResponseCache
from qceval.core.feedback import terminal_reason
from qceval.core.io import JsonlRunWriter, read_completed, read_run_identity
from qceval.core.lineage import (
    FEEDBACK_LINEAGE_SCHEMA_VERSION,
    build_run_identity,
    chain_id,
    request_trace,
    resolve_run_id,
    sha256_text,
)
from qceval.core.prompt_safety import OracleLeakError
from qceval.core.runner.evaluation import EvaluationMixin
from qceval.core.runner.feedback import FeedbackMixin
from qceval.core.runner.generation import GenerationMixin
from qceval.core.runner.isolation import IsolationMixin
from qceval.core.runner.options import _completed_count, _normalized_options
from qceval.core.runner.progress import ProgressMixin
from qceval.core.runner.records import (
    _chain_records,
    _framework_from_str,
    _job_key,
    _record_from_dict,
    _runtime_error_evaluation,
    _status,
    _suite_from_str,
)
from qceval.core.runner.types import RunJob
from qceval.models import (
    BenchmarkRecord,
    Framework,
    OutcomeStatus,
    ProviderRequest,
    ProviderResponse,
    QCEvalEvaluation,
    QCEvalTask,
    RunConfig,
    RunOptions,
    Suite,
)
from qceval.providers.base import Provider
from qceval.reports import summarize
from qceval.typing import TaskAdapter


class BenchmarkRunner(IsolationMixin, GenerationMixin, EvaluationMixin, FeedbackMixin, ProgressMixin):
    """Run QCircuitEval tasks through a provider and evaluator.

    Args:
        config: Stable run configuration.
        provider: Code-generation provider implementation.
        adapter: Task source and evaluator implementation.
        options: Optional execution controls. Defaults preserve serial
            execution, no cache, no resume, and no streaming output.

    Attributes:
        config: Run configuration used for task selection and metadata.
        provider: Provider used for code generation.
        adapter: Task adapter used for loading and evaluating tasks.
        options: Normalized execution controls.
    """

    def __init__(
        self,
        *,
        config: RunConfig,
        provider: Provider,
        adapter: TaskAdapter,
        options: RunOptions | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.adapter = adapter
        self.options = _normalized_options(RunOptions() if options is None else options)
        # Resume records are read (and the run ID resolved) before any output
        # writer can be constructed, so the resume source is never truncated
        # before it is consumed.
        self.run_id = resolve_run_id(self.options.resume_from)
        self.run_identity: dict[str, Any] | None = None
        self._final_stream_path = self.options.stream_to
        if self._resumes_into_stream():
            # Same-path resume: stream into a sibling temporary file and
            # atomically replace the resume source only after the run
            # finishes, so the original file survives a crash intact.
            final_stream_path = self._final_stream_path
            assert final_stream_path is not None
            temporary = final_stream_path.with_name(final_stream_path.name + ".resume-tmp")
            self.options = replace(self.options, stream_to=temporary)

    def _resumes_into_stream(self) -> bool:
        stream_to = self.options.stream_to
        resume_from = self.options.resume_from
        if stream_to is None or resume_from is None:
            return False
        return stream_to.expanduser().resolve() == resume_from.expanduser().resolve()

    def run(self) -> dict[str, Any]:
        """Execute benchmark run and return serialized payload.

        Returns:
            Serialized run payload containing records and aggregate summary.
        """
        jobs = self._jobs()
        self.run_identity = build_run_identity(self.config, self.options, self.adapter.metadata(), jobs)
        self._validate_resume_identity()
        if self.config.max_attempts > 1:
            records = self._run_jobs_feedback(jobs)
        else:
            records = self._run_jobs_fail_fast(jobs) if self.options.fail_fast else self._run_jobs(jobs)
        payload = self._payload(records)
        if self.options.stream_to is not None:
            writer = JsonlRunWriter(self.options.stream_to, truncate=False)
            try:
                writer.finalize(payload)
            finally:
                writer.close()
            if self._final_stream_path is not None and self.options.stream_to != self._final_stream_path:
                os.replace(self.options.stream_to, self._final_stream_path)
        return payload

    def _payload(self, records: list[BenchmarkRecord]) -> dict[str, Any]:
        return {
            "schema_version": "qceval.run.v2",
            "run_id": self.run_id,
            "run_identity": self.run_identity,
            "provider": self.config.provider,
            "model": self.config.model,
            "suites": list(self.config.suites),
            "qceval": self.adapter.metadata(),
            "results": [record.to_dict() for record in records],
            "summary": summarize(records, run_config=self.config),
        }

    def _jobs(self) -> list[RunJob]:
        jobs: list[RunJob] = []
        requested_numbers = set(self.config.task_numbers or ())
        found_numbers: set[int] = set()
        prompt_frameworks, regrade_frameworks = self._phase_frameworks()
        source_records = self._source_records()
        for suite in self.config.suites:
            for framework in self.config.frameworks:
                tasks = self._selected_tasks(framework, suite, requested_numbers, found_numbers)
                jobs.extend(
                    self._jobs_for_tasks(
                        tasks,
                        suite,
                        framework,
                        len(jobs),
                        prompt_frameworks,
                        regrade_frameworks,
                        source_records,
                    )
                )
        missing = requested_numbers - found_numbers
        if missing:
            joined = ", ".join(str(number) for number in sorted(missing))
            raise ValueError(f"selected task numbers are not present in the selected suite(s): {joined}")
        return jobs

    def _phase_frameworks(self) -> tuple[tuple[Framework, ...], tuple[Framework, ...]]:
        prompt_frameworks = (
            self.config.frameworks if self.options.prompt_frameworks is None else self.options.prompt_frameworks
        )
        regrade_frameworks = (
            self.config.frameworks if self.options.regrade_frameworks is None else self.options.regrade_frameworks
        )
        return prompt_frameworks, regrade_frameworks

    def _selected_tasks(
        self,
        framework: Framework,
        suite: Suite,
        requested_numbers: set[int],
        found_numbers: set[int],
    ) -> list[QCEvalTask]:
        tasks = self.adapter.load_tasks(framework, suite=suite)
        if requested_numbers:
            tasks = [task for task in tasks if _task_number(task.task_id) in requested_numbers]
            found_numbers.update(_task_number(task.task_id) for task in tasks)
        if self.config.max_tasks is not None:
            tasks = tasks[: self.config.max_tasks]
        return list(tasks)

    def _jobs_for_tasks(
        self,
        tasks: list[QCEvalTask],
        suite: Suite,
        framework: Framework,
        start_index: int,
        prompt_frameworks: tuple[Framework, ...],
        regrade_frameworks: tuple[Framework, ...],
        source_records: dict[tuple[Suite, Framework, str, int], BenchmarkRecord],
    ) -> list[RunJob]:
        jobs: list[RunJob] = []
        for task in tasks:
            for sample_index in range(self.config.samples_per_task):
                jobs.append(
                    self._job_for_task(
                        task,
                        suite,
                        framework,
                        start_index + len(jobs),
                        sample_index,
                        prompt_frameworks,
                        regrade_frameworks,
                        source_records,
                    )
                )
        return jobs

    def _job_for_task(
        self,
        task: QCEvalTask,
        suite: Suite,
        framework: Framework,
        index: int,
        sample_index: int,
        prompt_frameworks: tuple[Framework, ...],
        regrade_frameworks: tuple[Framework, ...],
        source_records: dict[tuple[Suite, Framework, str, int], BenchmarkRecord],
    ) -> RunJob:
        prompt = framework in prompt_frameworks
        source = None if prompt else source_records.get((suite, framework, task.task_id, sample_index))
        if source is None and not prompt:
            raise ValueError(
                "no stored response for "
                f"{suite}/{framework}/{task.task_id} sample {sample_index}; provide it with --input"
            )
        return RunJob(
            index=index,
            task=task,
            request=self._request(task, sample_index=sample_index, attempt_index=0),
            sample_index=sample_index,
            prompt=prompt,
            regrade=framework in regrade_frameworks,
            source_response=None if source is None else source.provider_response,
            source_provider=None if source is None else source.provider,
        )

    def _source_records(self) -> dict[tuple[Suite, Framework, str, int], BenchmarkRecord]:
        if self.options.input_from is None:
            return {}
        records: dict[tuple[Suite, Framework, str, int], BenchmarkRecord] = {}
        for key, payload in read_completed(self.options.input_from).items():
            compact_key = (_suite_from_str(key[0]), _framework_from_str(key[1]), key[2], key[3])
            record = _record_from_dict(payload)
            existing = records.get(compact_key)
            if existing is None or record.attempt_index > existing.attempt_index:
                records[compact_key] = record
        return records

    def _request(self, task: QCEvalTask, *, sample_index: int, attempt_index: int) -> ProviderRequest:
        metadata = self._provider_metadata(task)
        metadata["suite"] = task.suite
        metadata["sample_index"] = sample_index
        metadata["attempt_index"] = attempt_index
        return ProviderRequest(
            task_id=task.task_id,
            framework=task.framework,
            prompt=task.prompt,
            entry_point=task.entry_point,
            model=self.config.model,
            metadata=metadata,
            sample_index=sample_index,
            attempt_index=attempt_index,
        )

    def _provider_metadata(self, task: QCEvalTask) -> dict[str, Any]:
        """Return metadata safe for this provider's trust boundary."""
        if bool(getattr(self.provider, "trusted_metadata", False)):
            return task.provider_metadata()
        return {"category": task.category}

    def _run_jobs(self, jobs: list[RunJob]) -> list[BenchmarkRecord]:
        if self.options.task_timeout is not None:
            return self._run_jobs_isolated(jobs)
        # Read resume records before opening the writer so a same-path resume
        # source is never truncated before it is consumed.
        completed = self._completed_records()
        writer = None if self.options.stream_to is None else JsonlRunWriter(self.options.stream_to)
        records: list[BenchmarkRecord | None] = [None] * len(jobs)
        progress_bar = self._progress_bar(len(jobs))
        try:
            pending_generation, stored_responses = self._partition_run_jobs(
                jobs, completed, records, writer, progress_bar
            )
            cache = None if self.options.cache_dir is None else ResponseCache(self.options.cache_dir)
            generated = (
                self._generate_streaming(pending_generation, cache)
                if writer is not None
                else self._generate(pending_generation, cache)
            )
            with self._evaluation_scheduler() as evaluator:
                self._process_responses(chain(stored_responses, generated), evaluator, records, writer, progress_bar)
                self._store_evaluations(evaluator.drain_all(), records, writer, progress_bar)
        finally:
            self._close_progress(progress_bar)
            if writer is not None:
                writer.close()
        return [record for record in records if record is not None]

    def _partition_run_jobs(
        self,
        jobs: list[RunJob],
        completed: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        records: list[BenchmarkRecord | None],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> tuple[list[RunJob], list[tuple[RunJob, ProviderResponse]]]:
        pending_generation: list[RunJob] = []
        stored_responses: list[tuple[RunJob, ProviderResponse]] = []
        for job in jobs:
            record = completed.get(_job_key(job))
            if record is not None and self._uses_default_phases(job):
                self._store_record(records, job, record, writer, progress_bar)
            elif job.prompt:
                pending_generation.append(job)
            elif job.source_response is not None:
                stored_responses.append((job, job.source_response))
            else:
                raise ValueError(f"missing provider response for {job.task.framework}/{job.task.task_id}")
        return pending_generation, stored_responses

    def _process_responses(
        self,
        responses: Iterable[tuple[RunJob, ProviderResponse]],
        evaluator: Any,
        records: list[BenchmarkRecord | None],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        for job, provider_response in responses:
            if not provider_response.ok:
                record = self._provider_failure_record(job, provider_response)
                self._store_record(records, job, record, writer, progress_bar)
            elif job.regrade:
                self._store_evaluations(evaluator.submit(job, provider_response), records, writer, progress_bar)
            else:
                self._store_response(records, job, provider_response, writer, progress_bar)
            self._store_evaluations(evaluator.drain_ready(), records, writer, progress_bar)

    def _uses_default_phases(self, job: RunJob) -> bool:
        return (
            self.options.prompt_frameworks is None
            and self.options.regrade_frameworks is None
            and job.prompt
            and job.regrade
        )

    def _store_response(
        self,
        records: list[BenchmarkRecord | None],
        job: RunJob,
        provider_response: ProviderResponse | None,
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        if provider_response is None:
            raise ValueError(f"missing provider response for {job.task.framework}/{job.task.task_id}")
        status: OutcomeStatus = "generated" if provider_response.ok else "provider_failed"
        record = (
            self._record(job, provider_response, None, status)
            if provider_response.ok
            else self._provider_failure_record(job, provider_response)
        )
        self._store_record(records, job, record, writer, progress_bar)

    def _run_jobs_fail_fast(self, jobs: list[RunJob]) -> list[BenchmarkRecord]:
        completed = self._completed_records()
        writer = None if self.options.stream_to is None else JsonlRunWriter(self.options.stream_to)
        cache = None if self.options.cache_dir is None else ResponseCache(self.options.cache_dir)
        records: list[BenchmarkRecord] = []
        progress_bar = self._progress_bar(len(jobs))
        try:
            for job in jobs:
                record = completed.get(_job_key(job))
                if record is None:
                    record = self._run_one_job(job, cache)
                records.append(record)
                if writer is not None:
                    writer.append(record)
                self._emit_progress(record, len(records), len(jobs), progress_bar)
                if record.status != "passed":
                    break
        finally:
            self._close_progress(progress_bar)
            if writer is not None:
                writer.close()
        return records

    def _completed_records(self) -> dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord]:
        if self.options.resume_from is None:
            return {}
        completed: dict[tuple[Suite, Framework, str, int, int], BenchmarkRecord] = {}
        for key, payload in read_completed(self.options.resume_from).items():
            completed[(_suite_from_str(key[0]), _framework_from_str(key[1]), key[2], key[3], key[4])] = (
                _record_from_dict(payload)
            )
        return completed

    def _validate_resume_identity(self) -> None:
        path = self.options.resume_from
        if path is None:
            return
        completed = read_completed(path)
        if not completed:
            return
        identity, digests = read_run_identity(path)
        expected = None if self.run_identity is None else self.run_identity.get("sha256")
        if not isinstance(expected, str):
            raise ValueError("current run identity is unavailable")
        if identity is not None and identity != self.run_identity:
            raise ValueError("resume run identity does not match current configuration")
        if digests and digests != {expected}:
            raise ValueError("resume records contain a different run identity")
        if identity is None and not digests:
            raise ValueError("resume data has no run identity and cannot be adopted safely")

    def _next_feedback_job(
        self,
        base_job: RunJob,
        records_by_key: Mapping[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        *,
        index: int,
    ) -> RunJob:
        """Build the next repair job, containing oracle-safety failures.

        A residual oracle-safety assertion (for example, diagnostic text that
        echoes deny-listed fragments) fails only this attempt: the returned job
        carries a typed harness error and is short-circuited by
        :meth:`_generate_one` instead of crashing the run.
        """
        try:
            return super()._next_feedback_job(base_job, records_by_key, index=index)
        except OracleLeakError as exc:
            return self._harness_safety_job(base_job, records_by_key, index=index, error=exc)

    def _harness_safety_job(
        self,
        base_job: RunJob,
        records_by_key: Mapping[tuple[Suite, Framework, str, int, int], BenchmarkRecord],
        *,
        index: int,
        error: OracleLeakError,
    ) -> RunJob:
        previous = _chain_records(records_by_key, base_job)[-1]
        attempt_index = previous.attempt_index + 1
        return RunJob(
            index=index,
            task=base_job.task,
            request=self._request(base_job.task, sample_index=base_job.sample_index, attempt_index=attempt_index),
            sample_index=base_job.sample_index,
            attempt_index=attempt_index,
            feedback={
                "reason": "harness_safety_violation",
                "harness_error": "oracle_isolation",
                "detail": str(error),
                "source_attempt_index": previous.attempt_index,
                "message_to_model": "",
                "truncated": False,
            },
            parent_attempt_index=previous.attempt_index,
            parent_code_sha256=sha256_text(previous.provider_response.code),
        )

    def _generate_one(self, job: RunJob, cache: ResponseCache | None) -> ProviderResponse:
        if job.feedback.get("harness_error"):
            return self._harness_error_response(job)
        return super()._generate_one(job, cache)

    def _harness_error_response(self, job: RunJob) -> ProviderResponse:
        """Return the typed harness-error response for a contained attempt."""
        return ProviderResponse(
            code=None,
            model=self.config.model,
            metadata={"harness_error": str(job.feedback.get("harness_error"))},
            error=f"harness safety violation: {job.feedback.get('detail')}",
        )

    def _provider_failure_record(self, job: RunJob, response: ProviderResponse) -> BenchmarkRecord:
        """Separate remote provider failures from contained harness failures."""
        resolved_provider_failure = _is_resolved_provider_failure(response.metadata)
        if response.metadata.get("harness_error") or (
            response.metadata.get("infrastructure_error") and not resolved_provider_failure
        ):
            evaluation = _runtime_error_evaluation("opaque harness safety failure")
            if response.metadata.get("infrastructure_error"):
                evaluation = _runtime_error_evaluation(response.error or "provider infrastructure failure")
            return self._record(job, response, evaluation, "infrastructure_error")
        return self._record(job, response, None, "provider_failed")

    def _store_record(
        self,
        records: list[BenchmarkRecord | None],
        job: RunJob,
        record: BenchmarkRecord,
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        records[job.index] = record
        if writer is not None:
            writer.append(record)
        self._emit_progress(record, _completed_count(records), len(records), progress_bar)

    def _store_evaluations(
        self,
        evaluations: Iterable[tuple[RunJob, ProviderResponse, QCEvalEvaluation]],
        records: list[BenchmarkRecord | None],
        writer: JsonlRunWriter | None,
        progress_bar: Any | None,
    ) -> None:
        for job, provider_response, evaluation in evaluations:
            record = self._record(job, provider_response, evaluation, _status(evaluation))
            self._store_record(records, job, record, writer, progress_bar)

    def _record(
        self,
        job: RunJob,
        provider_response: ProviderResponse,
        evaluation: QCEvalEvaluation | None,
        status: OutcomeStatus,
    ) -> BenchmarkRecord:
        task = job.task
        record = BenchmarkRecord(
            framework=task.framework,
            suite=task.suite,
            task_id=task.task_id,
            sample_index=job.sample_index,
            attempt_index=job.attempt_index,
            entry_point=task.entry_point,
            category=task.category,
            provider=job.source_provider or self.provider.name,
            model=provider_response.model or self.config.model,
            status=status,
            feedback=job.feedback,
            provider_response=provider_response,
            evaluation=evaluation,
        )
        stop_reason = (
            "generated_ungraded"
            if status == "generated"
            else terminal_reason(record, max_attempts=self.config.max_attempts)
        )
        return replace(
            record,
            request_trace=request_trace(job.request),
            lineage={
                "schema_version": FEEDBACK_LINEAGE_SCHEMA_VERSION,
                "run_id": self.run_id,
                "run_identity_sha256": None if self.run_identity is None else self.run_identity.get("sha256"),
                "chain_id": chain_id(
                    self.run_id,
                    provider=self.config.provider,
                    model=self.config.model,
                    suite=task.suite,
                    framework=task.framework,
                    task_id=task.task_id,
                    sample_index=job.sample_index,
                ),
                "attempt_index": job.attempt_index,
                "parent_attempt_index": job.parent_attempt_index,
                "parent_code_sha256": job.parent_code_sha256,
                "code_sha256": sha256_text(provider_response.code),
                "feedback_source_attempt_index": job.feedback.get("source_attempt_index"),
                "feedback_policy_version": self.config.feedback_policy.version,
                "terminal": stop_reason is not None,
                "stop_reason": stop_reason,
            },
        )


def _is_resolved_provider_failure(metadata: Mapping[str, Any]) -> bool:
    """Return whether campaign evidence reclassified a provider outcome out of infrastructure."""
    resolution = metadata.get("campaign_resolution")
    return (
        metadata.get("failure_classification") == "provider_policy_refusal"
        and isinstance(resolution, Mapping)
        and resolution.get("schema_version") == "qceval.policy_refusal_resolution.v1"
        and resolution.get("disposition") == "candidate_less_provider_failure"
    )


def _task_number(task_id: str) -> int:
    """Return the numeric portion of a core or QEC suite-local task ID."""
    numeric = task_id.removeprefix("qec")
    try:
        return int(numeric)
    except ValueError as exc:
        raise ValueError(f"task ID has no numeric suffix: {task_id}") from exc
