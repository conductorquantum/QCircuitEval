"""Tests for runner generation, streaming, cache, and resume behavior."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from qceval.core.cache import ResponseCache
from qceval.core.runner import BenchmarkRunner
from qceval.models import ProviderRequest, ProviderResponse, QCEvalTask, RunConfig, RunOptions
from tests.runner_support import StubAdapter, StubFailingProvider, StubProvider


class _BlockingProvider(StubProvider):
    def __init__(self) -> None:
        self.release_first = threading.Event()

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if request.task_id == "01":
            self.release_first.wait(timeout=5.0)
        return super().generate(request)


class _TwoTaskAdapter(StubAdapter):
    def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
        return [
            QCEvalTask(
                task_id=str(index).zfill(2),
                framework="qiskit",
                prompt=f"p{index}",
                entry_point="answer",
                category="cat",
                canonical_class={"type": "exact_distribution"},
                suite=suite,  # type: ignore[arg-type]
                raw={"canonical_solution": f"code-{index}"},
            )
            for index in range(1, 3)
        ]


def test_runner_concurrent_generation_preserves_task_order() -> None:
    class ManyTaskAdapter(StubAdapter):
        def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
            return [
                QCEvalTask(
                    task_id=str(index).zfill(2),
                    framework="qiskit",
                    prompt=f"p{index}",
                    entry_point="answer",
                    category="cat",
                    canonical_class={"type": "exact_distribution"},
                    suite=suite,  # type: ignore[arg-type]
                    raw={"canonical_solution": f"code-{index}"},
                )
                for index in range(1, 6)
            ]

    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=ManyTaskAdapter(),  # type: ignore[arg-type]
        options=RunOptions(generation_concurrency=3),
    )
    payload = runner.run()
    assert [record["task_id"] for record in payload["results"]] == ["01", "02", "03", "04", "05"]
    assert payload["summary"]["passed"] == 5


def test_runner_streams_and_resumes_completed_results(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    first = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    ).run()
    runner = BenchmarkRunner(
        config=config,
        provider=StubFailingProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(resume_from=path, stream_to=tmp_path / "resumed.jsonl"),
    )
    resumed = runner.run()
    assert first["results"] == resumed["results"]
    assert resumed["summary"]["passed"] == 1


def test_runner_resume_rejects_mismatched_model_identity(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    first_config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="model-a")
    BenchmarkRunner(
        config=first_config,
        provider=StubProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    ).run()

    mismatched = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="model-b")
    with pytest.raises(ValueError, match="run identity"):
        BenchmarkRunner(
            config=mismatched,
            provider=StubProvider(),
            adapter=StubAdapter(),  # type: ignore[arg-type]
            options=RunOptions(resume_from=path),
        ).run()


def test_runner_withholds_oracle_metadata_from_untrusted_provider() -> None:
    class CapturingProvider:
        name = "remote"

        def __init__(self) -> None:
            self.metadata = None

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            self.metadata = dict(request.metadata)
            return ProviderResponse(code=None, model=request.model, error="stop")

    provider = CapturingProvider()
    config = RunConfig(provider="remote", frameworks=("qiskit",), source_hint=None, model="m")
    BenchmarkRunner(config=config, provider=provider, adapter=StubAdapter()).run()  # type: ignore[arg-type]

    assert provider.metadata is not None
    assert "canonical_solution" not in provider.metadata
    assert "canonical_class" not in provider.metadata
    assert provider.metadata["category"] == "cat"


def test_runner_resume_into_same_path_preserves_completed_records(tmp_path: Path) -> None:
    """Regression (H10): --out X --resume-from X must not truncate the resume
    source before it is read, must not re-call the provider, and must leave a
    complete output file behind."""

    class ExplodingProvider:
        name = "stub"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise AssertionError("provider must not be called for restored records")

    path = tmp_path / "results.jsonl"
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    first = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    ).run()

    resumed = BenchmarkRunner(
        config=config,
        provider=ExplodingProvider(),  # type: ignore[arg-type]
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(resume_from=path, stream_to=path),
    ).run()

    assert resumed["results"] == first["results"]
    assert resumed["summary"]["passed"] == 1
    assert resumed["run_id"] == first["run_id"]
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["kind"] for line in lines] == ["result", "summary"]
    assert lines[0]["task_id"] == "01"
    assert not (tmp_path / "results.jsonl.resume-tmp").exists()


def test_runner_same_path_resume_preserves_source_when_run_fails(tmp_path: Path) -> None:
    """Regression (H10): if a same-path resume crashes mid-run, the original
    resume source must survive untouched."""

    path = tmp_path / "results.jsonl"
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=_TwoTaskAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    ).run()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    partial = [line for line in lines if line.get("kind") == "result" and line.get("task_id") == "01"]
    partial.append(next(line for line in lines if line.get("kind") == "summary"))
    path.write_text("\n".join(json.dumps(line, sort_keys=True) for line in partial) + "\n", encoding="utf-8")
    original = path.read_text(encoding="utf-8")

    class CrashOnSecondTaskProvider(StubProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            if request.task_id == "02":
                raise RuntimeError("boom")
            return super().generate(request)

    with pytest.raises(RuntimeError, match="boom"):
        BenchmarkRunner(
            config=config,
            provider=CrashOnSecondTaskProvider(),
            adapter=_TwoTaskAdapter(),  # type: ignore[arg-type]
            options=RunOptions(resume_from=path, stream_to=path),
        ).run()

    assert path.read_text(encoding="utf-8") == original


def test_runner_response_cache_reuses_provider_response(tmp_path: Path) -> None:
    class CountingProvider(StubProvider):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            self.calls += 1
            return super().generate(request)

    provider = CountingProvider()
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    options = RunOptions(cache_dir=tmp_path / "cache")
    first = BenchmarkRunner(
        config=config,
        provider=provider,
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=options,
    ).run()
    second = BenchmarkRunner(
        config=config,
        provider=provider,
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=options,
    ).run()
    assert provider.calls == 1
    assert first["summary"] == second["summary"]
    assert second["results"][0]["provider_response"]["raw_response"] is None


def test_runner_uses_batch_provider_generate_many() -> None:
    class BatchStubProvider(StubProvider):
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def generate_many(self, requests: Sequence[ProviderRequest]) -> list[ProviderResponse]:
            self.batches.append([request.task_id for request in requests])
            return [self.generate(request) for request in requests]

    class ManyTaskAdapter(StubAdapter):
        def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
            return [
                QCEvalTask(
                    task_id=str(index).zfill(2),
                    framework="qiskit",
                    prompt=f"p{index}",
                    entry_point="answer",
                    category="cat",
                    canonical_class={"type": "exact_distribution"},
                    suite=suite,  # type: ignore[arg-type]
                    raw={"canonical_solution": f"code-{index}"},
                )
                for index in range(1, 5)
            ]

    provider = BatchStubProvider()
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    BenchmarkRunner(
        config=config,
        provider=provider,
        adapter=ManyTaskAdapter(),  # type: ignore[arg-type]
        options=RunOptions(generation_concurrency=2),
    ).run()
    assert provider.batches == [["01", "02"], ["03", "04"]]


def test_runner_streams_completed_generation_before_slow_peer(tmp_path: Path) -> None:
    provider = _BlockingProvider()
    output_path = tmp_path / "stream.jsonl"
    thread, payload, errors = _start_streaming_runner(provider, output_path)
    streamed_task_ids = _wait_for_streamed_task(output_path, "02")
    provider.release_first.set()
    thread.join(timeout=5.0)
    _raise_thread_error(thread, errors)
    assert "02" in streamed_task_ids
    assert "01" not in streamed_task_ids
    assert [record["task_id"] for record in payload["results"]] == ["01", "02"]  # type: ignore[index]


def test_streaming_generation_drains_active_chunk_then_stops_after_infrastructure_failure(tmp_path: Path) -> None:
    class FiveTaskAdapter(StubAdapter):
        def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
            return [
                QCEvalTask(
                    task_id=str(index).zfill(2),
                    framework="qiskit",
                    prompt=f"p{index}",
                    entry_point="answer",
                    category="cat",
                    canonical_class={"type": "exact_distribution"},
                    suite=suite,  # type: ignore[arg-type]
                    raw={"canonical_solution": f"code-{index}"},
                )
                for index in range(1, 6)
            ]

    class OutageProvider(StubProvider):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            self.calls.append(request.task_id)
            if request.task_id == "02":
                return ProviderResponse(
                    code=None,
                    model=request.model,
                    metadata={"infrastructure_error": True},
                    error="endpoint unavailable",
                )
            return super().generate(request)

    provider = OutageProvider()
    payload = BenchmarkRunner(
        config=RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m"),
        provider=provider,
        adapter=FiveTaskAdapter(),  # type: ignore[arg-type]
        options=RunOptions(
            generation_concurrency=2,
            stream_to=tmp_path / "segment.jsonl",
            stop_on_infrastructure_error=True,
            prompt_frameworks=("qiskit",),
            regrade_frameworks=(),
        ),
    ).run()

    assert set(provider.calls) == {"01", "02"}
    assert len(payload["results"]) == 2
    assert {record["status"] for record in payload["results"]} == {"generated", "infrastructure_error"}


def test_generation_mixin_generate_merges_cache_hits_and_misses(tmp_path: Path) -> None:
    class CountingProvider(StubProvider):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            self.calls.append(request.task_id)
            return super().generate(request)

    class ManyTaskAdapter(StubAdapter):
        def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
            return [
                QCEvalTask(
                    task_id=str(index).zfill(2),
                    framework="qiskit",
                    prompt=f"p{index}",
                    entry_point="answer",
                    category="cat",
                    canonical_class={"type": "exact_distribution"},
                    suite=suite,  # type: ignore[arg-type]
                    raw={"canonical_solution": f"code-{index}"},
                )
                for index in range(1, 4)
            ]

    provider = CountingProvider()
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(config=config, provider=provider, adapter=ManyTaskAdapter())  # type: ignore[arg-type]
    jobs = runner._jobs()
    cache = ResponseCache(tmp_path / "cache")
    first = list(runner._generate(jobs[:2], cache))
    second = list(runner._generate(jobs, cache))
    assert [job.task.task_id for job, _ in first] == ["01", "02"]
    assert [job.task.task_id for job, _ in second] == ["01", "02", "03"]
    assert provider.calls == ["01", "02", "03"]


def test_generation_mixin_generate_many_chunks_batch_provider() -> None:
    class BatchStubProvider(StubProvider):
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def generate_many(self, requests: Sequence[ProviderRequest]) -> list[ProviderResponse]:
            self.batches.append([request.task_id for request in requests])
            return [self.generate(request) for request in requests]

    class ManyTaskAdapter(StubAdapter):
        def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
            return [
                QCEvalTask(
                    task_id=str(index).zfill(2),
                    framework="qiskit",
                    prompt=f"p{index}",
                    entry_point="answer",
                    category="cat",
                    canonical_class={"type": "exact_distribution"},
                    suite=suite,  # type: ignore[arg-type]
                    raw={"canonical_solution": f"code-{index}"},
                )
                for index in range(1, 5)
            ]

    provider = BatchStubProvider()
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(
        config=config,
        provider=provider,
        adapter=ManyTaskAdapter(),  # type: ignore[arg-type]
        options=RunOptions(generation_concurrency=2),
    )
    generated = runner._generate_uncached(runner._jobs())
    assert [job.task.task_id for job, _ in generated] == ["01", "02", "03", "04"]
    assert provider.batches == [["01", "02"], ["03", "04"]]


def _streamed_task_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    task_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("kind") == "result":
            task_ids.append(str(payload["task_id"]))
    return task_ids


def _start_streaming_runner(
    provider: _BlockingProvider, output_path: Path
) -> tuple[threading.Thread, dict[str, object], list[BaseException]]:
    payload: dict[str, object] = {}
    errors: list[BaseException] = []

    def run_benchmark() -> None:
        try:
            payload.update(
                BenchmarkRunner(
                    config=RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m"),
                    provider=provider,
                    adapter=_TwoTaskAdapter(),  # type: ignore[arg-type]
                    options=RunOptions(generation_concurrency=2, stream_to=output_path),
                ).run()
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_benchmark)
    thread.start()
    return thread, payload, errors


def _wait_for_streamed_task(path: Path, task_id: str) -> list[str]:
    deadline = time.monotonic() + 5.0
    streamed_task_ids: list[str] = []
    while time.monotonic() < deadline:
        streamed_task_ids = _streamed_task_ids(path)
        if task_id in streamed_task_ids:
            break
        time.sleep(0.01)
    return streamed_task_ids


def _raise_thread_error(thread: threading.Thread, errors: list[BaseException]) -> None:
    if errors:
        raise errors[0]
    assert thread.is_alive() is False
