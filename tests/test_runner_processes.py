from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import Future
from contextlib import suppress
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from qceval.core.bench import Adaptor
from qceval.core.runner import BenchmarkRunner
from qceval.core.runner.evaluation import EvaluationScheduler
from qceval.core.runner.isolation import IsolationMixin
from qceval.core.runner.processes import terminate_worker_process
from qceval.core.runner.workers import _start_process_group
from qceval.models import QCEvalEvaluation, RunConfig, RunOptions
from qceval.providers.smoke import SmokeProvider
from tests.runner_support import SlowProvider, StubAdapter

_POSIX_PROCESS_GROUPS = all(hasattr(os, name) for name in ("getpgid", "getpgrp", "killpg", "setsid"))


class _StubbornProcess:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.alive = True
        self.terminated = False
        self.killed = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.alive = False


def _spawn_sleeping_descendant(pid_path: str) -> None:
    descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])

    def handle_termination(*_: object) -> None:
        with suppress(subprocess.TimeoutExpired):
            descendant.wait(timeout=0.5)
        os._exit(0)

    signal.signal(signal.SIGTERM, handle_termination)
    Path(pid_path).write_text(str(descendant.pid), encoding="utf-8")
    while True:
        time.sleep(0.1)


def _isolated_worker_with_descendant(pid_path: str) -> None:
    _start_process_group()
    _spawn_sleeping_descendant(pid_path)


def _exited_isolated_worker_with_descendant(pid_path: str) -> None:
    _start_process_group()
    descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    Path(pid_path).write_text(str(descendant.pid), encoding="utf-8")
    os._exit(0)


def _wait_for_descendant_pid(pid_path: Path, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with suppress(FileNotFoundError, ValueError):
            return int(pid_path.read_text(encoding="utf-8"))
        time.sleep(0.01)
    raise AssertionError("worker did not report its descendant PID")


def _process_is_running(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _wait_for_process_exit(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_is_running(pid):
            return True
        time.sleep(0.01)
    return not _process_is_running(pid)


def _cleanup_test_processes(group_id: int | None, descendant_pid: int | None, process: Any = None) -> None:
    if group_id is not None and group_id not in {os.getpid(), os.getpgrp()}:
        with suppress(ProcessLookupError):
            os.killpg(group_id, signal.SIGTERM)
    if process is not None:
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
    if descendant_pid is not None and _process_is_running(descendant_pid):
        if group_id is not None and group_id not in {os.getpid(), os.getpgrp()}:
            with suppress(ProcessLookupError):
                os.killpg(group_id, signal.SIGKILL)
        with suppress(ProcessLookupError):
            os.kill(descendant_pid, signal.SIGKILL)
        _wait_for_process_exit(descendant_pid)


def test_process_cleanup_falls_back_without_signaling_parent_group(monkeypatch) -> None:
    # Arrange
    process = _StubbornProcess()
    group_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda group_id, signum: group_signals.append((group_id, signum)), raising=False)

    # Act
    terminate_worker_process(process, grace_period=0)

    # Assert
    assert group_signals == []
    assert process.terminated
    assert process.killed


@pytest.mark.skipif(not _POSIX_PROCESS_GROUPS, reason="POSIX process groups are required")
def test_isolated_timeout_cleanup_terminates_descendant(tmp_path) -> None:
    # Arrange
    context = get_context("spawn")
    pid_path = tmp_path / "isolated-descendant.pid"
    process = context.Process(target=_isolated_worker_with_descendant, args=(str(pid_path),))
    process.start()
    descendant_pid: int | None = None
    group_id: int | None = process.pid
    queue = context.Queue(maxsize=1)
    try:
        descendant_pid = _wait_for_descendant_pid(pid_path)
        assert group_id is not None
        assert os.getpgid(group_id) == group_id
        active = SimpleNamespace(process=process, queue=queue)

        # Act
        IsolationMixin()._stop_isolated_process(active)  # type: ignore[arg-type]

        # Assert
        assert not process.is_alive()
        assert _wait_for_process_exit(descendant_pid)
    finally:
        _cleanup_test_processes(group_id, descendant_pid, process)
        for method in ("close", "join_thread"):
            with suppress(Exception):
                getattr(queue, method)()


@pytest.mark.skipif(not _POSIX_PROCESS_GROUPS, reason="POSIX process groups are required")
def test_verified_process_group_cleanup_survives_exited_worker_leader(tmp_path) -> None:
    context = get_context("spawn")
    pid_path = tmp_path / "exited-worker-descendant.pid"
    process = context.Process(target=_exited_isolated_worker_with_descendant, args=(str(pid_path),))
    process.start()
    group_id = process.pid
    descendant_pid: int | None = None
    try:
        descendant_pid = _wait_for_descendant_pid(pid_path)
        process.join(timeout=2)
        assert not process.is_alive()
        assert group_id is not None

        terminate_worker_process(
            process,
            grace_period=0.1,
            verified_process_group_id=group_id,
        )

        assert _wait_for_process_exit(descendant_pid)
    finally:
        _cleanup_test_processes(group_id, descendant_pid, process)


@pytest.mark.skipif(not _POSIX_PROCESS_GROUPS, reason="POSIX process groups are required")
def test_process_pool_timeout_cleanup_terminates_descendant(tmp_path) -> None:
    # Arrange
    scheduler = EvaluationScheduler(
        adapter=StubAdapter(),
        config=RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model=None),
        options=RunOptions(eval_timeout=0.01),
    )
    executor = scheduler._executor
    assert executor is not None
    pid_path = tmp_path / "pool-descendant.pid"
    future = executor.submit(_spawn_sleeping_descendant, str(pid_path))
    descendant_pid: int | None = None
    group_id: int | None = None
    worker_process: Any = None
    try:
        descendant_pid = _wait_for_descendant_pid(pid_path)
        processes = tuple(executor._processes.values())
        assert len(processes) == 1
        worker_process = processes[0]
        group_id = worker_process.pid
        assert group_id is not None
        assert os.getpgid(group_id) == group_id
        scheduler._pending[future] = (None, None, time.monotonic() - 1)  # type: ignore[assignment]

        # Act
        assert scheduler._collect_timeouts()

        # Assert
        assert not worker_process.is_alive()
        assert _wait_for_process_exit(descendant_pid)
        scheduler.close()

    finally:
        with suppress(Exception):
            executor.shutdown(wait=False, cancel_futures=True)
        _cleanup_test_processes(group_id, descendant_pid, worker_process)


def test_evaluation_scheduler_drains_before_exceeding_worker_capacity() -> None:
    scheduler = EvaluationScheduler(
        adapter=StubAdapter(),
        config=RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model=None),
        options=RunOptions(evaluation_workers=2, eval_timeout=60),
    )
    assert scheduler._executor is not None
    scheduler._executor.shutdown(wait=False, cancel_futures=True)
    completed: Future[dict[str, Any]] = Future()
    completed.set_result(
        {
            "compiled": True,
            "ran": True,
            "passed": True,
            "grader_details": {},
        }
    )
    running: Future[dict[str, Any]] = Future()
    first_job = SimpleNamespace(index=0)
    second_job = SimpleNamespace(index=1)
    first_response = SimpleNamespace()
    second_response = SimpleNamespace()
    scheduler._pending = {
        completed: (first_job, first_response, time.monotonic()),
        running: (second_job, second_response, time.monotonic()),
    }  # type: ignore[assignment]

    out = scheduler._wait_for_capacity()

    assert [(job.index, evaluation.passed) for job, _, evaluation in out] == [(0, True)]
    assert list(scheduler._pending) == [running]
    scheduler._executor = None


def test_runner_task_timeout_streams_provider_failure(tmp_path) -> None:
    # Arrange
    path = tmp_path / "results.jsonl"
    config = RunConfig(
        provider="slow",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        max_tasks=1,
    )
    runner = BenchmarkRunner(
        config=config,
        provider=SlowProvider(),
        adapter=Adaptor(),
        options=RunOptions(task_timeout=0.05, stream_to=path),
    )

    # Act
    payload = runner.run()
    first_line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    # Assert
    assert payload["summary"]["infrastructure_failures"] == 1
    assert payload["results"][0]["status"] == "infrastructure_error"
    assert "task timed out after" in payload["results"][0]["provider_response"]["error"]
    assert first_line["status"] == "infrastructure_error"


def test_timeout_record_terminates_feedback_chain() -> None:
    # Arrange
    pytest.importorskip("qiskit")
    config = RunConfig(
        provider="slow",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        max_tasks=1,
        max_attempts=2,
    )
    runner = BenchmarkRunner(
        config=config,
        provider=SlowProvider(),
        adapter=Adaptor(),
        options=RunOptions(task_timeout=0.05),
    )

    # Act
    payload = runner.run()

    # Assert
    assert [(record["sample_index"], record["attempt_index"]) for record in payload["results"]] == [(0, 0)]
    assert payload["results"][0]["status"] == "infrastructure_error"
    assert payload["results"][0]["lineage"]["stop_reason"] == "grader_nondecision"


def test_isolated_worker_result_receives_request_provenance() -> None:
    pytest.importorskip("qiskit")
    config = RunConfig(
        provider="smoke",
        frameworks=("qiskit",),
        source_hint=None,
        model="smoke-canonical",
        max_tasks=1,
    )

    payload = BenchmarkRunner(
        config=config,
        provider=SmokeProvider(),
        adapter=Adaptor(),
        options=RunOptions(task_timeout=30),
    ).run()

    record = payload["results"][0]
    assert record["request_trace"]["schema_version"] == "qceval.request_trace.v1"
    assert record["lineage"]["run_id"] == payload["run_id"]


def test_evaluation_round_trip_preserves_metric_name(tmp_path) -> None:
    # Arrange
    from qceval.core.runner.records import _evaluation_from_dict

    evaluation = QCEvalEvaluation(
        compiled=True,
        ran=True,
        passed=True,
        metric=0.5,
        metric_name="hellinger_infidelity",
    )

    # Act
    payload = evaluation.to_dict()
    restored = _evaluation_from_dict(payload)

    # Assert
    assert tmp_path.exists()
    assert restored.metric == 0.5
    assert restored.metric_name == "hellinger_infidelity"
