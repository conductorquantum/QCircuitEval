from __future__ import annotations

import json
from dataclasses import replace

import pytest

from qceval.core.lineage import build_run_identity
from qceval.core.runner import BenchmarkRunner
from qceval.models import BenchmarkRecord, ProviderResponse, QCEvalEvaluation, QCEvalTask, RunConfig, RunOptions
from tests.runner_support import RepairProvider, StubAdapter, StubFailingProvider


class RepairAdapter(StubAdapter):
    def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
        return QCEvalEvaluation(compiled=True, ran=True, passed=code == "good", metric=0.0)


def _write_resume_record(path, record, config, adapter, provider) -> None:
    runner = BenchmarkRunner(config=config, provider=provider, adapter=adapter)
    identity = build_run_identity(config, runner.options, adapter.metadata(), runner._jobs())
    record = replace(record, lineage={"run_identity_sha256": identity["sha256"]})
    lines = [
        {"kind": "result", **record.to_dict()},
        {"kind": "summary", "run_identity": identity},
    ]
    path.write_text("\n".join(json.dumps(line, sort_keys=True) for line in lines) + "\n", encoding="utf-8")


def test_feedback_runner_passes_on_second_attempt(tmp_path) -> None:
    # Arrange
    path = tmp_path / "feedback.jsonl"
    config = RunConfig(provider="repair", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=2)
    runner = BenchmarkRunner(
        config=config,
        provider=RepairProvider(),
        adapter=RepairAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    )

    # Act
    payload = runner.run()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    # Assert
    assert [record["attempt_index"] for record in payload["results"]] == [0, 1]
    assert payload["results"][1]["status"] == "passed"
    assert payload["results"][1]["feedback"]["reason"] == "failed"
    assert payload["results"][0]["request_trace"]["messages"] == [{"role": "user", "content": "p"}]
    assert [message["role"] for message in payload["results"][1]["request_trace"]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert payload["results"][1]["lineage"]["parent_attempt_index"] == 0
    assert payload["results"][1]["lineage"]["parent_code_sha256"] == payload["results"][0]["lineage"]["code_sha256"]
    assert payload["results"][1]["lineage"]["stop_reason"] == "verified_pass"
    assert payload["summary"]["feedback_lineage"]["assigned_chains"] == 1
    assert payload["summary"]["feedback_lineage"]["provenance_coverage"] == 1.0
    assert lines[-1]["run_id"] == payload["run_id"]
    assert [line["kind"] for line in lines] == ["result", "result", "summary"]


def test_feedback_per_chain_concurrent_produces_correct_results() -> None:
    # Arrange
    config = RunConfig(provider="repair", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=2)
    runner = BenchmarkRunner(
        config=config,
        provider=RepairProvider(),
        adapter=RepairAdapter(),  # type: ignore[arg-type]
        options=RunOptions(generation_concurrency=2),
    )

    # Act
    payload = runner.run()

    # Assert
    assert [record["attempt_index"] for record in payload["results"]] == [0, 1]
    assert payload["results"][1]["status"] == "passed"


def test_feedback_streams_incrementally_for_independent_chains(tmp_path) -> None:
    # Arrange
    path = tmp_path / "feedback.jsonl"

    class TwoTaskAdapter(StubAdapter):
        def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
            return [_task("01", suite), _task("02", suite)]

        def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
            passed = task.task_id == "01" or code == "good"
            return QCEvalEvaluation(compiled=True, ran=True, passed=passed, metric=0.0)

    config = RunConfig(provider="repair", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=2)

    # Act
    BenchmarkRunner(
        config=config,
        provider=RepairProvider(),
        adapter=TwoTaskAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    ).run()

    # Assert
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len([line for line in lines if line["kind"] == "result"]) == 3


def test_feedback_resume_continues_from_next_attempt(tmp_path) -> None:
    # Arrange
    path = tmp_path / "partial.jsonl"
    failed = BenchmarkRecord(
        framework="qiskit",
        task_id="01",
        entry_point="answer",
        category="cat",
        provider="repair",
        model="m",
        status="failed",
        provider_response=ProviderResponse(code="bad", model="m"),
        evaluation=QCEvalEvaluation(compiled=True, ran=True, passed=False, metric=1.0),
    )
    config = RunConfig(provider="repair", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=2)
    _write_resume_record(path, failed, config, RepairAdapter(), RepairProvider())

    # Act
    payload = BenchmarkRunner(
        config=config,
        provider=RepairProvider(),
        adapter=RepairAdapter(),  # type: ignore[arg-type]
        options=RunOptions(resume_from=path),
    ).run()

    # Assert
    assert [record["attempt_index"] for record in payload["results"]] == [0, 1]
    assert payload["results"][1]["status"] == "passed"


def test_feedback_resume_preserves_run_id(tmp_path) -> None:
    path = tmp_path / "complete.jsonl"
    config = RunConfig(provider="repair", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=2)
    first = BenchmarkRunner(
        config=config,
        provider=RepairProvider(),
        adapter=RepairAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    ).run()

    resumed = BenchmarkRunner(
        config=config,
        provider=RepairProvider(),
        adapter=RepairAdapter(),  # type: ignore[arg-type]
        options=RunOptions(resume_from=path),
    ).run()

    assert resumed["run_id"] == first["run_id"]
    assert {record["lineage"]["run_id"] for record in resumed["results"]} == {first["run_id"]}


def test_feedback_resume_rejects_gap(tmp_path) -> None:
    # Arrange
    path = tmp_path / "gap.jsonl"
    attempt_two = BenchmarkRecord(
        framework="qiskit",
        task_id="01",
        attempt_index=2,
        entry_point="answer",
        category="cat",
        provider="repair",
        model="m",
        status="failed",
        provider_response=ProviderResponse(code="bad", model="m"),
        evaluation=QCEvalEvaluation(compiled=True, ran=True, passed=False),
    )
    config = RunConfig(provider="repair", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=3)
    _write_resume_record(path, attempt_two, config, StubAdapter(), RepairProvider())
    runner = BenchmarkRunner(
        config=config,
        provider=RepairProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(resume_from=path),
    )

    # Act / Assert
    with pytest.raises(ValueError, match="feedback gap"):
        runner.run()


def test_feedback_stops_passed_tasks() -> None:
    # Arrange
    class TwoTaskAdapter(StubAdapter):
        def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
            return [_task("01", suite), _task("02", suite)]

        def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
            passed = task.task_id == "01" or code == "good"
            return QCEvalEvaluation(compiled=True, ran=True, passed=passed, metric=0.0)

    config = RunConfig(provider="repair", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=2)

    # Act
    payload = BenchmarkRunner(
        config=config,
        provider=RepairProvider(),
        adapter=TwoTaskAdapter(),  # type: ignore[arg-type]
    ).run()

    # Assert
    assert [(record["task_id"], record["attempt_index"]) for record in payload["results"]] == [
        ("01", 0),
        ("02", 0),
        ("02", 1),
    ]


def test_feedback_stops_provider_failure_without_repair() -> None:
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=3)

    payload = BenchmarkRunner(
        config=config,
        provider=StubFailingProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
    ).run()

    assert len(payload["results"]) == 1
    assert payload["results"][0]["lineage"]["stop_reason"] == "provider_failure"
    assert payload["summary"]["feedback_lineage"]["terminal_stop_reason_counts"] == {"provider_failure": 1}


def test_feedback_repair_transcript_with_benign_target_dict_does_not_crash(tmp_path) -> None:
    """Regression (H9): assistant-echoed candidate code containing raw oracle
    substrings like ``{"target": 2}`` must not crash the run."""

    class TargetDictRepairProvider:
        name = "repair"

        def generate(self, request):
            if request.attempt_index == 0:
                return ProviderResponse(code='spec = {"target": 2, "control": 0}\n', model=request.model)
            return ProviderResponse(code="good", model=request.model)

    path = tmp_path / "target.jsonl"
    config = RunConfig(provider="repair", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=2)
    runner = BenchmarkRunner(
        config=config,
        provider=TargetDictRepairProvider(),  # type: ignore[arg-type]
        adapter=RepairAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    )

    payload = runner.run()

    assert [record["attempt_index"] for record in payload["results"]] == [0, 1]
    assert [record["status"] for record in payload["results"]] == ["failed", "passed"]
    repair_messages = payload["results"][1]["request_trace"]["messages"]
    assert repair_messages[1]["role"] == "assistant"
    assert '"target": 2' in repair_messages[1]["content"]


def test_feedback_oracle_safety_violation_fails_only_that_attempt() -> None:
    """Regression (H9): a residual deny-list hit in harness-authored feedback
    text fails the single attempt with a typed harness error, not the run."""

    class CountingProvider:
        name = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            return ProviderResponse(code="bad", model=request.model)

    class LeakyErrorAdapter(StubAdapter):
        def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
            return QCEvalEvaluation(
                compiled=True,
                ran=False,
                passed=False,
                error='candidate raised on line: spec = {"target": 2}',
                error_type="RuntimeError",
            )

    provider = CountingProvider()
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=3)

    payload = BenchmarkRunner(
        config=config,
        provider=provider,  # type: ignore[arg-type]
        adapter=LeakyErrorAdapter(),  # type: ignore[arg-type]
    ).run()

    assert provider.calls == 1
    assert [record["attempt_index"] for record in payload["results"]] == [0, 1]
    assert payload["results"][0]["status"] == "run_failed"
    harness_record = payload["results"][1]
    assert harness_record["status"] == "infrastructure_error"
    assert harness_record["feedback"]["reason"] == "harness_safety_violation"
    assert harness_record["feedback"]["harness_error"] == "oracle_isolation"
    assert harness_record["provider_response"]["metadata"]["harness_error"] == "oracle_isolation"
    assert "harness safety violation" in harness_record["provider_response"]["error"]
    assert harness_record["lineage"]["stop_reason"] == "grader_nondecision"


def _task(task_id: str, suite: str) -> QCEvalTask:
    return QCEvalTask(
        task_id=task_id,
        framework="qiskit",
        prompt=f"p{task_id}",
        entry_point="answer",
        category="cat",
        canonical_class={"type": "exact_distribution"},
        suite=suite,  # type: ignore[arg-type]
        raw={"canonical_solution": "code"},
    )
