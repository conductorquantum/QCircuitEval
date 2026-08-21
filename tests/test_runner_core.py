from __future__ import annotations

import pytest

from qceval.core.io import OutputFormat, write_output
from qceval.core.runner import BenchmarkRunner
from qceval.models import ProviderRequest, ProviderResponse, QCEvalEvaluation, QCEvalTask, RunConfig, RunOptions
from tests.runner_support import StubAdapter, StubFailingProvider, StubProvider


def test_runner_records_passed_task() -> None:
    # Arrange
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(config=config, provider=StubProvider(), adapter=StubAdapter())  # type: ignore[arg-type]

    # Act
    payload = runner.run()

    # Assert
    assert payload["summary"]["passed"] == 1
    assert payload["results"][0]["status"] == "passed"


def test_runner_can_store_prompt_only_response() -> None:
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(prompt_frameworks=("qiskit",), regrade_frameworks=()),
    )

    payload = runner.run()

    assert payload["results"][0]["status"] == "generated"
    assert payload["results"][0]["evaluation"] is None
    assert payload["summary"]["generated"] == 1
    assert payload["summary"]["failed"] == 0


@pytest.mark.parametrize("output_format", ["jsonl", "json"])
def test_runner_regrades_stored_response_without_prompting(tmp_path, output_format: OutputFormat) -> None:
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    source_payload = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
    ).run()
    source = tmp_path / f"source.{output_format}"
    write_output(source, source_payload, output_format)
    runner = BenchmarkRunner(
        config=config,
        provider=StubFailingProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(prompt_frameworks=(), regrade_frameworks=("qiskit",), input_from=source),
    )

    payload = runner.run()

    assert payload["results"][0]["status"] == "passed"
    assert payload["results"][0]["provider_response"]["code"] == "code"


def test_runner_regrade_preserves_resolved_policy_refusal_as_provider_failure(tmp_path) -> None:
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    source_payload = BenchmarkRunner(
        config=config,
        provider=StubFailingProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
    ).run()
    metadata = source_payload["results"][0]["provider_response"]["metadata"]
    metadata.update(
        {
            "infrastructure_error": True,
            "failure_classification": "provider_policy_refusal",
            "campaign_resolution": {
                "schema_version": "qceval.policy_refusal_resolution.v1",
                "disposition": "candidate_less_provider_failure",
            },
        }
    )
    source = tmp_path / "source.jsonl"
    write_output(source, source_payload, "jsonl")
    runner = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(prompt_frameworks=(), regrade_frameworks=("qiskit",), input_from=source),
    )

    payload = runner.run()

    assert payload["results"][0]["status"] == "provider_failed"
    assert payload["results"][0]["evaluation"] is None
    assert payload["summary"]["infrastructure_failures"] == 0


def test_runner_regrade_keeps_unresolved_provider_infrastructure_failure(tmp_path) -> None:
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    source_payload = BenchmarkRunner(
        config=config,
        provider=StubFailingProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
    ).run()
    source_payload["results"][0]["provider_response"]["metadata"].update(
        {
            "infrastructure_error": True,
            "failure_classification": "provider_policy_refusal",
            "campaign_resolution": {
                "schema_version": "qceval.policy_refusal_resolution.v1",
                "disposition": "unresolved",
            },
        }
    )
    source = tmp_path / "source.jsonl"
    write_output(source, source_payload, "jsonl")
    runner = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(prompt_frameworks=(), regrade_frameworks=("qiskit",), input_from=source),
    )

    payload = runner.run()

    assert payload["results"][0]["status"] == "infrastructure_error"
    assert payload["summary"]["infrastructure_failures"] == 1


def test_runner_expands_samples_per_task() -> None:
    # Arrange
    config = RunConfig(
        provider="stub",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        samples_per_task=3,
        pass_k=1,
    )
    runner = BenchmarkRunner(config=config, provider=StubProvider(), adapter=StubAdapter())  # type: ignore[arg-type]

    # Act
    payload = runner.run()

    # Assert
    assert len(payload["results"]) == 3
    assert [record["sample_index"] for record in payload["results"]] == [0, 1, 2]
    assert [record["attempt_index"] for record in payload["results"]] == [0, 0, 0]


def test_runner_pass_at_k_summary_groups_samples() -> None:
    # Arrange
    class SampleProvider:
        name = "sample"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(code=f"sample={request.metadata['sample_index']}", model=request.model)

    class SampleAdapter(StubAdapter):
        def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
            return QCEvalEvaluation(compiled=True, ran=True, passed=code == "sample=1")

    config = RunConfig(
        provider="sample",
        frameworks=("qiskit",),
        source_hint=None,
        model="m",
        samples_per_task=3,
        pass_k=2,
    )
    runner = BenchmarkRunner(config=config, provider=SampleProvider(), adapter=SampleAdapter())  # type: ignore[arg-type]

    # Act
    payload = runner.run()

    # Assert
    assert payload["summary"]["task_totals"] == {"unique_tasks": 1, "record_count": 3}
    assert payload["summary"]["pass_at_k"]["tasks"][0]["n"] == 3
    assert payload["summary"]["pass_at_k"]["tasks"][0]["c"] == 1
    assert payload["summary"]["pass_at_k"]["pass_at_k"] == pytest.approx(2 / 3)


def test_runconfig_rejects_invalid_passk_and_feedback_settings() -> None:
    with pytest.raises(ValueError, match="pass_k.*must be <= samples_per_task"):
        RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m", samples_per_task=2, pass_k=3)
    with pytest.raises(ValueError, match="samples_per_task must be >= 1"):
        RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m", samples_per_task=0)
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m", max_attempts=0)
    with pytest.raises(ValueError, match="feedback_max_chars must be >= 1"):
        RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m", feedback_max_chars=0)
    with pytest.raises(ValueError, match="cannot be combined"):
        RunConfig(
            provider="stub",
            frameworks=("qiskit",),
            source_hint=None,
            model="m",
            samples_per_task=2,
            max_attempts=2,
        )


def test_runner_records_provider_failure() -> None:
    # Arrange
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(config=config, provider=StubFailingProvider(), adapter=StubAdapter())  # type: ignore[arg-type]

    # Act
    payload = runner.run()

    # Assert
    assert payload["summary"]["provider_failures"] == 1
    assert payload["results"][0]["status"] == "provider_failed"


def test_runner_records_compile_failure() -> None:
    # Arrange
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    adapter = StubAdapter()
    adapter.evaluation = QCEvalEvaluation(compiled=False, ran=False, passed=False)
    runner = BenchmarkRunner(config=config, provider=StubProvider(), adapter=adapter)  # type: ignore[arg-type]

    # Act
    payload = runner.run()

    # Assert
    assert payload["summary"]["compile_failures"] == 1
    assert payload["results"][0]["status"] == "compile_failed"


def test_runner_records_run_failure() -> None:
    # Arrange
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    adapter = StubAdapter()
    adapter.evaluation = QCEvalEvaluation(compiled=True, ran=False, passed=False)
    runner = BenchmarkRunner(config=config, provider=StubProvider(), adapter=adapter)  # type: ignore[arg-type]

    # Act
    payload = runner.run()

    # Assert
    assert payload["summary"]["run_failures"] == 1
    assert payload["results"][0]["status"] == "run_failed"


def test_runner_records_whitespace_only_code_as_provider_failure() -> None:
    # Arrange
    class EmptyProvider:
        name = "empty"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(code=" \n\t", model=request.model)

    class EmptyCodeAdapter(StubAdapter):
        def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
            raise AssertionError("empty provider output must not reach evaluation")

    config = RunConfig(provider="empty", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(config=config, provider=EmptyProvider(), adapter=EmptyCodeAdapter())  # type: ignore[arg-type]

    # Act
    payload = runner.run()

    # Assert
    assert payload["summary"]["provider_failures"] == 1
    assert payload["results"][0]["status"] == "provider_failed"
    assert payload["results"][0]["evaluation"] is None


def test_runner_fail_fast_stops_after_first_failure() -> None:
    # Arrange
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
                    raw={"canonical_solution": "code"},
                )
                for index in range(1, 4)
            ]

        def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
            return QCEvalEvaluation(compiled=True, ran=True, passed=task.task_id != "02", metric=0.0)

    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=ManyTaskAdapter(),  # type: ignore[arg-type]
        options=RunOptions(fail_fast=True),
    )

    # Act
    payload = runner.run()

    # Assert
    assert [record["task_id"] for record in payload["results"]] == ["01", "02"]
    assert payload["summary"]["total_tasks"] == 2
    assert payload["summary"]["failed"] == 1


def test_runner_progress_reports_to_stderr(capsys) -> None:
    # Arrange
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")
    runner = BenchmarkRunner(
        config=config,
        provider=StubProvider(),
        adapter=StubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(progress=True),
    )

    # Act
    runner.run()

    # Assert
    assert "[1/1] core:qiskit:01 passed" in capsys.readouterr().err


def test_runner_rejects_invalid_options() -> None:
    # Arrange
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m")

    # Act / Assert
    with pytest.raises(ValueError, match="generation_concurrency"):
        BenchmarkRunner(
            config=config,
            provider=StubProvider(),
            adapter=StubAdapter(),  # type: ignore[arg-type]
            options=RunOptions(generation_concurrency=0),
        )
    with pytest.raises(ValueError, match="evaluation_workers"):
        BenchmarkRunner(
            config=config,
            provider=StubProvider(),
            adapter=StubAdapter(),  # type: ignore[arg-type]
            options=RunOptions(evaluation_workers=0),
        )
    with pytest.raises(ValueError, match="eval_timeout"):
        BenchmarkRunner(
            config=config,
            provider=StubProvider(),
            adapter=StubAdapter(),  # type: ignore[arg-type]
            options=RunOptions(eval_timeout=0),
        )
    with pytest.raises(ValueError, match="task_timeout"):
        BenchmarkRunner(
            config=config,
            provider=StubProvider(),
            adapter=StubAdapter(),  # type: ignore[arg-type]
            options=RunOptions(task_timeout=0),
        )
