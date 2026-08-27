"""Provider generation helpers for benchmark runs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast

from qceval.core.cache import ResponseCache
from qceval.core.runner.records import _status
from qceval.core.runner.types import RunJob
from qceval.models import BenchmarkRecord, ProviderResponse, RunConfig, RunOptions
from qceval.providers.base import BatchProvider, Provider, fan_out_generate, fan_out_generate_stream, is_batch_provider


class GenerationMixin:
    """Provider generation behavior shared by runner implementations."""

    config: RunConfig
    options: RunOptions
    provider: Provider

    def _run_one_job(self, job: RunJob, cache: ResponseCache | None) -> BenchmarkRecord:
        runner = cast(Any, self)
        provider_response = self._generate_one(job, cache)
        if not provider_response.ok:
            return runner._provider_failure_record(job, provider_response)
        evaluated = list(runner._evaluate([(job, provider_response)]))
        _, _, evaluation = evaluated[0]
        return runner._record(job, provider_response, evaluation, _status(evaluation))

    def _generate_one(self, job: RunJob, cache: ResponseCache | None) -> ProviderResponse:
        if cache is None:
            return self.provider.generate(job.request)
        cache_key = cache.key_for(job.request, provider=self.provider.name, settings=self.config.provider_config)
        response = cache.get(cache_key)
        if response is not None:
            return response
        response = self.provider.generate(job.request)
        cache.put(cache_key, response)
        return response

    def _generate(self, jobs: list[RunJob], cache: ResponseCache | None) -> Iterable[tuple[RunJob, ProviderResponse]]:
        misses: list[RunJob] = []
        cached: dict[int, ProviderResponse] = {}
        for job in jobs:
            if cache is None:
                misses.append(job)
                continue
            cache_key = cache.key_for(job.request, provider=self.provider.name, settings=self.config.provider_config)
            response = cache.get(cache_key)
            if response is None:
                misses.append(job)
            else:
                cached[job.index] = response

        # Generate only cache misses, then merge by original job index so cached
        # and uncached responses preserve serial runner output order.
        generated = self._generate_uncached(misses)
        generated_by_index = {job.index: response for job, response in generated}
        if cache is not None:
            for job in misses:
                response = generated_by_index[job.index]
                cache_key = cache.key_for(
                    job.request,
                    provider=self.provider.name,
                    settings=self.config.provider_config,
                )
                cache.put(cache_key, response)

        for job in jobs:
            yield job, cached.get(job.index) or generated_by_index[job.index]

    def _generate_chunks(
        self, jobs: list[RunJob], cache: ResponseCache | None
    ) -> Iterator[tuple[list[RunJob], list[ProviderResponse]]]:
        """Generate responses and yield singleton chunks as each job completes.

        This path intentionally avoids ``generate_many`` because ordered batch
        APIs cannot expose per-task completions for JSONL streaming.
        """
        for job, response in self._generate_streaming(jobs, cache):
            yield [job], [response]

    def _generate_streaming(
        self, jobs: list[RunJob], cache: ResponseCache | None
    ) -> Iterator[tuple[RunJob, ProviderResponse]]:
        """Generate responses and yield each cache hit or completion."""
        misses, cached_jobs, cached_responses = self._partition_cached_jobs(jobs, cache)
        for job, response in zip(cached_jobs, cached_responses, strict=True):
            yield job, response

        chunk_size = max(1, self.options.generation_concurrency)
        for chunk in _chunks(misses, chunk_size):
            requests = [job.request for job in chunk]
            stop_after_chunk = False
            for request_index, response in fan_out_generate_stream(
                self.provider, requests, self.options.generation_concurrency
            ):
                job = chunk[request_index]
                self._cache_response(job, response, cache)
                if response.metadata.get("infrastructure_error"):
                    stop_after_chunk = True
                yield job, response
            if stop_after_chunk and self.options.stop_on_infrastructure_error:
                return

    def _partition_cached_jobs(
        self, jobs: list[RunJob], cache: ResponseCache | None
    ) -> tuple[list[RunJob], list[RunJob], list[ProviderResponse]]:
        misses: list[RunJob] = []
        cached_jobs: list[RunJob] = []
        cached_responses: list[ProviderResponse] = []
        if cache is None:
            return list(jobs), cached_jobs, cached_responses

        for job in jobs:
            cache_key = cache.key_for(job.request, provider=self.provider.name, settings=self.config.provider_config)
            response = cache.get(cache_key)
            if response is None:
                misses.append(job)
            else:
                cached_jobs.append(job)
                cached_responses.append(response)
        return misses, cached_jobs, cached_responses

    def _cache_response(self, job: RunJob, response: ProviderResponse, cache: ResponseCache | None) -> None:
        if cache is None:
            return
        cache_key = cache.key_for(job.request, provider=self.provider.name, settings=self.config.provider_config)
        cache.put(cache_key, response)

    def _generate_uncached(self, jobs: Sequence[RunJob]) -> list[tuple[RunJob, ProviderResponse]]:
        if not jobs:
            return []
        if self.options.generation_concurrency <= 1:
            return [(job, self.provider.generate(job.request)) for job in jobs]
        if is_batch_provider(self.provider):
            return self._generate_many(jobs, self.provider)
        responses = fan_out_generate(self.provider, [job.request for job in jobs], self.options.generation_concurrency)
        return list(zip(jobs, responses, strict=True))

    def _generate_many(self, jobs: Sequence[RunJob], provider: BatchProvider) -> list[tuple[RunJob, ProviderResponse]]:
        out: list[tuple[RunJob, ProviderResponse]] = []
        for chunk in _chunks(list(jobs), self.options.generation_concurrency):
            responses = provider.generate_many([job.request for job in chunk])
            out.extend(zip(chunk, responses, strict=True))
        return out


def _chunks(items: list[RunJob], size: int) -> Iterable[list[RunJob]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]
