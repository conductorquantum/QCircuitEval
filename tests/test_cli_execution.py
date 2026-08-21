from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import pytest

from qceval.cli import main, non_negative_float, parse_args, positive_float, positive_int
from qceval.core.lineage import build_run_identity


def test_cli_main_writes_output(monkeypatch, tmp_path: Path) -> None:
    # Arrange
    class StubRunner:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self) -> dict:
            return {
                "schema_version": "qceval.run.v2",
                "provider": "smoke",
                "model": None,
                "suites": ["core"],
                "qceval": {"path": None},
                "results": [],
                "summary": {"pass_rate": 0.0, "passed": 0, "total_tasks": 0},
            }

    monkeypatch.setattr("qceval.cli.build_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr("qceval.cli.Adaptor", lambda path, **kwargs: object())
    monkeypatch.setattr("qceval.cli.BenchmarkRunner", StubRunner)
    out = tmp_path / "results.json"

    # Act
    code = main(["run", "--provider", "smoke", "--framework", "qiskit", "--out", str(out)])

    # Assert
    assert code == 0
    assert out.exists()


def test_cli_smoke_qiskit_reasoning_effort_all(tmp_path: Path) -> None:
    out = tmp_path / "matrix.json"

    code = main(
        [
            "run",
            "--provider",
            "smoke",
            "--framework",
            "qiskit",
            "--reasoning-effort",
            "all",
            "--max-tasks",
            "1",
            "--eval-timeout",
            "10",
            "--fail-fast",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    manifest = json.loads((tmp_path / "matrix.efforts.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "qceval.effort_sweep.v1"
    assert len(manifest["jobs"]) == 7
    assert all(job["exit_code"] == 0 for job in manifest["jobs"])
    assert all(Path(job["out"]).exists() for job in manifest["jobs"])


def test_cli_config_validation_shows_clean_error(capsys) -> None:
    # Act
    code = main(
        [
            "run",
            "--provider",
            "smoke",
            "--framework",
            "qiskit",
            "--samples-per-task",
            "2",
            "--max-attempts",
            "2",
            "--out",
            "results.json",
        ]
    )

    # Assert
    assert code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_cli_smoke_integration_pass_k(tmp_path: Path) -> None:
    # Arrange
    out = tmp_path / "passk.jsonl"

    # Act
    code = main(
        [
            "run",
            "--provider",
            "smoke",
            "--framework",
            "qiskit",
            "--max-tasks",
            "2",
            "--samples-per-task",
            "2",
            "--pass-k",
            "2",
            "--out",
            str(out),
        ]
    )

    # Assert
    assert code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    summary = json.loads(lines[-1])
    assert summary["qceval"]["package_version"] == importlib.metadata.version("qceval")
    assert summary["qceval"]["path"] is None
    commit = summary["qceval"]["commit"]
    assert commit is None or len(commit) == 40


def test_cli_rejects_incomplete_openrouter_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    argv = ["run", "--provider", "openrouter", "--model", "model", "--out", "results.json"]

    # Act
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)

    # Assert
    assert exc.value.code == 2


def test_regrade_only_ignores_credentials_and_dotenv_for_run_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("", encoding="utf-8")
    captured_identities: list[dict] = []
    captured_configs: list[dict] = []

    class StubRunner:
        def __init__(self, **kwargs) -> None:
            captured_configs.append(dict(kwargs["config"].provider_config))
            captured_identities.append(build_run_identity(kwargs["config"], kwargs["options"], {"path": None}, []))

        def run(self) -> dict:
            return {
                "schema_version": "qceval.run.v2",
                "provider": "openrouter",
                "model": "model",
                "suites": ["core"],
                "qceval": {"path": None},
                "results": [],
                "summary": {"pass_rate": 0.0, "passed": 0, "total_tasks": 0},
            }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("qceval.cli.BenchmarkRunner", StubRunner)
    (tmp_path / ".env").write_text('OPENROUTER_API_KEY="unterminated\n', encoding="utf-8")

    base_args = [
        "run",
        "--provider",
        "openrouter",
        "--model",
        "model",
        "--regrade",
        "qiskit",
        "--input",
        str(source),
    ]
    assert main([*base_args, "--openrouter-api-key", "flag-key", "--out", str(tmp_path / "flag.json")]) == 0

    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")
    assert main([*base_args, "--out", str(tmp_path / "environment.json")]) == 0

    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert main([*base_args, "--out", str(tmp_path / "malformed-dotenv.json")]) == 0

    assert len(captured_configs) == 3
    assert all("openrouter_api_key" not in config for config in captured_configs)
    assert captured_configs[0] == captured_configs[1] == captured_configs[2]
    assert captured_identities[0] == captured_identities[1] == captured_identities[2]


def test_cli_rejects_missing_resume_file() -> None:
    # Arrange
    argv = ["run", "--provider", "smoke", "--out", "results.jsonl", "--resume-from", "missing.jsonl"]

    # Act
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)

    # Assert
    assert exc.value.code == 2


def test_positive_parsers_reject_non_positive_values() -> None:
    # Arrange
    values = ("0", "-1")

    # Act / Assert
    for value in values:
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(value)
        with pytest.raises(argparse.ArgumentTypeError):
            positive_float(value)
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_float("-0.1")
