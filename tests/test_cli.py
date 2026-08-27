from __future__ import annotations

from pathlib import Path

import pytest

import qceval.cli as cli_module
from qceval.cli import (
    _frameworks,
    _provider_config,
    _stream_path,
    _suites,
    main,
    parse_args,
)
from qceval.core.io import infer_format


def test_parse_run_args_accepts_required_command() -> None:
    # Arrange
    argv = ["run", "--provider", "smoke", "--framework", "qiskit", "--out", "results.json"]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.command == "run"
    assert args.provider == "smoke"
    assert args.framework == ["qiskit"]
    assert args.suite == "core"


def test_parse_run_args_selects_task_subset_and_phase_frameworks(tmp_path: Path) -> None:
    source = tmp_path / "prior.jsonl"
    source.write_text("", encoding="utf-8")

    args = parse_args(
        [
            "run",
            "--out",
            "results.jsonl",
            "--tasks",
            "1",
            "7",
            "--rerun",
            "qiskit",
            "cirq",
            "--regrade",
            "cirq",
            "cudaq",
            "--input",
            str(source),
        ]
    )

    assert args.tasks == [1, 7]
    assert _frameworks(args.prompt) == ("qiskit", "cirq")
    assert _frameworks(args.regrade) == ("cirq", "cudaq")


def test_parse_run_args_accepts_json_regrade_input(tmp_path: Path) -> None:
    source = tmp_path / "prior.json"
    source.write_text("{}", encoding="utf-8")

    args = parse_args(
        [
            "run",
            "--regrade",
            "all",
            "--suite",
            "all",
            "--input",
            str(source),
            "--out",
            "regraded.jsonl",
        ]
    )

    assert args.input == source
    assert args.suite == "all"


def test_parse_run_args_rejects_non_json_regrade_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "prior.txt"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        parse_args(["run", "--regrade", "all", "--input", str(source), "--out", "regraded.jsonl"])

    assert exc.value.code == 2
    assert "--input only supports JSON or JSONL" in capsys.readouterr().err


def test_parse_run_args_accepts_throughput_flags(tmp_path: Path) -> None:
    # Arrange
    resume = tmp_path / "results.jsonl"
    resume.write_text("", encoding="utf-8")
    argv = [
        "run",
        "--provider",
        "smoke",
        "--out",
        "results.jsonl",
        "--generation-concurrency",
        "4",
        "--evaluation-workers",
        "2",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--resume-from",
        str(resume),
        "--task-timeout",
        "90",
        "--eval-timeout",
        "1",
        "--fail-fast",
        "--progress",
    ]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.generation_concurrency == 4
    assert args.evaluation_workers == 2
    assert args.task_timeout == 90
    assert args.fail_fast is True
    assert _stream_path(args) == Path("results.jsonl")


def test_cli_accepts_pass_k_flags() -> None:
    # Arrange
    argv = [
        "run",
        "--provider",
        "smoke",
        "--framework",
        "qiskit",
        "--out",
        "results.jsonl",
        "--samples-per-task",
        "3",
        "--pass-k",
        "2",
        "--max-attempts",
        "1",
        "--feedback-max-chars",
        "1000",
    ]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.samples_per_task == 3
    assert args.pass_k == 2
    assert args.max_attempts == 1
    assert args.feedback_max_chars == 1000
    assert _stream_path(args) == Path("results.jsonl")


def test_cli_accepts_openrouter_reasoning_effort() -> None:
    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--openrouter-api-key",
            "key",
            "--model",
            "z-ai/glm-5.2",
            "--reasoning-effort",
            "xhigh",
            "--out",
            "results.jsonl",
        ]
    )

    assert args.reasoning_effort == "xhigh"
    assert _provider_config(args)["reasoning_effort"] == "xhigh"


def test_cli_accepts_all_reasoning_efforts() -> None:
    args = parse_args(
        [
            "run",
            "--provider",
            "smoke",
            "--reasoning-effort",
            "all",
            "--out",
            "results.json",
        ]
    )

    assert [job.effort for job in args.reasoning_jobs] == [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


def test_cli_registry_forwards_multiple_registry_files() -> None:
    args = parse_args(
        [
            "run",
            "--provider",
            "smoke",
            "--registry",
            "production/models.prompt-effort-sweep.json",
            "production/models.max-reasoning.json",
            "--reasoning-effort",
            "all",
            "--out",
            "results",
        ]
    )

    models = {job.model for job in args.reasoning_jobs}
    assert "openai/gpt-5.6-luna" in models
    assert "x-ai/grok-4.6" in models


def test_cli_openrouter_registry_does_not_require_model() -> None:
    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--openrouter-api-key",
            "key",
            "--registry",
            "production/models.max-reasoning.json",
            "--out",
            "results",
        ]
    )

    assert args.model is None
    assert len(args.reasoning_jobs) == 8


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--provider", "coda", "--coda-api-key", "key"], "not supported with --provider coda"),
        (["--resume-from", "prior.jsonl"], "--resume-from cannot be combined"),
        (["--rerun", "qiskit"], "--rerun/--regrade cannot be combined"),
        (["--configuration-id", "manual"], "assigned automatically"),
    ],
)
def test_cli_rejects_incompatible_multi_job_flags(
    extra: list[str],
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prior = tmp_path / "prior.jsonl"
    prior.write_text("", encoding="utf-8")
    resolved = [str(prior) if value == "prior.jsonl" else value for value in extra]

    with pytest.raises(SystemExit) as exc:
        parse_args(
            [
                "run",
                "--reasoning-effort",
                "all",
                "--out",
                "results.json",
                *resolved,
            ]
        )

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_rejects_reasoning_enabled_with_registry(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(
            [
                "run",
                "--provider",
                "smoke",
                "--registry",
                "production/models.max-reasoning.json",
                "--reasoning-enabled",
                "--out",
                "results",
            ]
        )

    assert exc.value.code == 2
    assert "--reasoning-enabled cannot be combined" in capsys.readouterr().err


def test_unpinned_openrouter_sweep_omits_provider_configuration_id() -> None:
    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--openrouter-api-key",
            "key",
            "--model",
            "model/a",
            "--reasoning-effort",
            "all",
            "--out",
            "results.json",
        ]
    )
    args.configuration_id = "model-a__effort-max"

    assert "configuration_id" not in _provider_config(args)


def test_cli_accepts_complete_endpoint_cap_exception_provenance() -> None:
    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--openrouter-api-key",
            "key",
            "--model",
            "x-ai/grok-4.6",
            "--openrouter-endpoint-tag",
            "xai",
            "--openrouter-max-output-tokens",
            "128000",
            "--openrouter-output-limit-source",
            "benchmark_floor",
            "--openrouter-endpoint-cap-status",
            "undisclosed_first_party_exception",
            "--openrouter-output-token-parameter",
            "max_tokens",
            "--openrouter-route-revision",
            "route-grok",
            "--configuration-id",
            "x-ai-grok-4-6__effort-xhigh",
            "--reasoning-effort",
            "xhigh",
            "--out",
            "results.jsonl",
        ]
    )

    config = _provider_config(args)
    assert config["openrouter_endpoint_cap_status"] == "undisclosed_first_party_exception"
    assert config["configuration_id"] == "x-ai-grok-4-6__effort-xhigh"


def test_cli_reads_openrouter_api_key_file(tmp_path: Path) -> None:
    key_file = tmp_path / "openrouter.key"
    key_file.write_text("secret-key\n", encoding="utf-8")
    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--openrouter-api-key-file",
            str(key_file),
            "--model",
            "z-ai/glm-5.2",
            "--out",
            "results.jsonl",
        ]
    )

    assert _provider_config(args)["openrouter_api_key"] == "secret-key"


def test_cli_reads_openrouter_api_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")

    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--model",
            "poolside/laguna-xs-2.1",
            "--out",
            "results.jsonl",
        ]
    )

    assert _provider_config(args)["openrouter_api_key"] == "environment-key"


def test_cli_reads_openrouter_api_key_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "# Local credentials\nexport OPENROUTER_API_KEY='dotenv-key'\n",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--model",
            "poolside/laguna-xs-2.1",
            "--out",
            "results.jsonl",
        ]
    )

    assert _provider_config(args)["openrouter_api_key"] == "dotenv-key"


def test_cli_environment_credentials_override_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=dotenv-key\n", encoding="utf-8")

    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--model",
            "poolside/laguna-xs-2.1",
            "--out",
            "results.jsonl",
        ]
    )

    assert _provider_config(args)["openrouter_api_key"] == "environment-key"


def test_cli_explicit_credentials_override_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")

    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--openrouter-api-key",
            "explicit-key",
            "--model",
            "poolside/laguna-xs-2.1",
            "--out",
            "results.jsonl",
        ]
    )

    assert _provider_config(args)["openrouter_api_key"] == "explicit-key"


def test_cli_reads_coda_api_key_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CODA_API_KEY", raising=False)
    (tmp_path / ".env").write_text('CODA_API_KEY="coda-dotenv-key"\n', encoding="utf-8")

    args = parse_args(
        [
            "run",
            "--provider",
            "coda",
            "--out",
            "results.jsonl",
        ]
    )

    assert _provider_config(args)["coda_api_key"] == "coda-dotenv-key"


def test_cli_accepts_reasoning_enabled() -> None:
    args = parse_args(
        [
            "run",
            "--provider",
            "openrouter",
            "--openrouter-api-key",
            "key",
            "--model",
            "google/gemma-4-31b-it",
            "--reasoning-enabled",
            "--out",
            "results.jsonl",
        ]
    )

    assert _provider_config(args)["reasoning_enabled"] is True


def test_cli_rejects_pass_k_greater_than_samples(capsys) -> None:
    # Arrange
    argv = [
        "run",
        "--provider",
        "smoke",
        "--out",
        "results.json",
        "--samples-per-task",
        "3",
        "--pass-k",
        "5",
    ]

    # Act
    code = main(argv)

    # Assert
    assert code == 2
    assert "pass_k" in capsys.readouterr().err


def test_fail_fast_rejected_with_pass_k() -> None:
    # Arrange
    argv = [
        "run",
        "--provider",
        "smoke",
        "--out",
        "results.json",
        "--samples-per-task",
        "2",
        "--fail-fast",
    ]

    # Act / Assert
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_frameworks_expands_all() -> None:
    # Arrange
    choice = "all"

    # Act
    frameworks = _frameworks(choice)

    # Assert
    assert frameworks == ("qiskit", "cirq", "pennylane", "cudaq")


def test_cli_accepts_cudaq_framework_choice() -> None:
    # Arrange
    argv = ["run", "--provider", "smoke", "--framework", "cudaq", "--out", "results.json"]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.framework == ["cudaq"]


def test_cli_accepts_suite_qec() -> None:
    # Arrange
    argv = ["run", "--provider", "smoke", "--framework", "qiskit", "--suite", "qec", "--out", "results.json"]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.suite == "qec"


def test_suites_expands_all() -> None:
    # Arrange
    choice = "all"

    # Act
    suites = _suites(choice)

    # Assert
    assert suites == ("core", "qec")


def test_stream_path_uses_task_timeout_for_jsonl() -> None:
    # Arrange
    args = parse_args(["run", "--provider", "smoke", "--out", "results.jsonl", "--task-timeout", "90"])

    # Act / Assert
    assert _stream_path(args) == Path("results.jsonl")


def test_provider_timeout_does_not_set_whole_task_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class CapturingRunner:
        def __init__(self, *, options: object, **_: object) -> None:
            captured["options"] = options

        def run(self) -> dict[str, object]:
            return {"summary": {}}

    monkeypatch.setattr(cli_module, "BenchmarkRunner", CapturingRunner)
    monkeypatch.setattr(cli_module, "format_run_summary", lambda _summary: "")
    args = parse_args(
        [
            "run",
            "--provider",
            "smoke",
            "--out",
            str(tmp_path / "results.jsonl"),
            "--timeout",
            "3",
        ]
    )

    assert cli_module._run(args) == 0
    assert captured["options"].task_timeout is None  # type: ignore[union-attr]


def test_cli_accepts_zero_temperature() -> None:
    # Arrange
    argv = ["run", "--provider", "smoke", "--framework", "qiskit", "--temperature", "0.0", "--out", "results.json"]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.temperature == 0.0


def test_cli_accepts_coda_without_model() -> None:
    # Arrange
    argv = [
        "run",
        "--provider",
        "coda",
        "--coda-api-key",
        "key",
        "--coda-mode",
        "learn",
        "--coda-fast",
        "--framework",
        "qiskit",
        "--out",
        "results.json",
    ]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.model == "coda/learn-fast"
    assert args.provider == "coda"


def test_cli_rejects_coda_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CODA_API_KEY", raising=False)
    argv = ["run", "--provider", "coda", "--framework", "qiskit", "--out", "results.json"]

    # Act / Assert
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_cli_warns_when_coda_temperature_is_set(capsys) -> None:
    # Arrange
    argv = [
        "run",
        "--provider",
        "coda",
        "--coda-api-key",
        "key",
        "--temperature",
        "0.8",
        "--out",
        "results.json",
    ]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.temperature == 0.8
    assert "temperature is ignored" in capsys.readouterr().err


def test_cli_accepts_coda_provider_flags() -> None:
    # Arrange
    argv = [
        "run",
        "--provider",
        "coda",
        "--coda-api-key",
        "key",
        "--coda-agents-url",
        "http://localhost:8000/v0/coda/agents",
        "--coda-prefer-structured-response",
        "--retry-max-delay",
        "9",
        "--out",
        "results.json",
    ]

    # Act
    args = parse_args(argv)
    config = _provider_config(args)

    # Assert
    assert config["coda_api_key"] == "key"
    assert config["coda_agents_url"] == "http://localhost:8000/v0/coda/agents"
    assert config["coda_prefer_structured_response"] is True
    assert config["retry_max_delay"] == 9
    assert "temperature" not in config


def test_infer_format_uses_jsonl_suffix() -> None:
    # Arrange
    path = Path("results.jsonl")

    # Act
    output_format = infer_format(path, "auto")

    # Assert
    assert output_format == "jsonl"
