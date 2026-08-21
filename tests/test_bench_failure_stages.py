"""Candidate failure-stage regressions for the bundled adapter."""

from __future__ import annotations

import importlib.metadata
from dataclasses import replace
from pathlib import Path

import pytest

import qceval.core.bench as bench_module
from qceval.core.bench import Adaptor, _validate_runtime_prompt_hash
from qceval.semantics.contracts import ContractRegistry


def test_bundled_metadata_embeds_reproducibility_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench_module, "_package_version", lambda: "1.2.3")
    monkeypatch.setattr(bench_module, "_checkout_commit", lambda: "a" * 40)
    monkeypatch.setattr(bench_module, "_checkout_dirty", lambda: False)

    adapter = Adaptor(tmp_path / "hint")
    metadata = adapter.metadata()

    assert metadata == {
        "source": "bundled-qceval",
        "package_version": "1.2.3",
        "source_hint": str(tmp_path / "hint"),
        "path": None,
        "branch": None,
        "commit": "a" * 40,
        "commit_status": "available",
        "dirty": False,
    }
    monkeypatch.setattr(bench_module, "_checkout_commit", lambda: "b" * 40)
    assert adapter.metadata() == metadata


def test_package_version_uses_installed_distribution() -> None:
    assert bench_module._package_version() == importlib.metadata.version("qceval")


def test_checkout_commit_is_omitted_outside_source_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wheel_module = tmp_path / "site-packages" / "qceval" / "core" / "bench.py"
    monkeypatch.setattr(bench_module, "__file__", str(wheel_module))

    assert bench_module._checkout_commit() is None


def test_checkout_commit_reads_source_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_module = tmp_path / "src" / "qceval" / "core" / "bench.py"
    source_module.parent.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(bench_module, "__file__", str(source_module))

    def run_git(command: list[str], **kwargs: object) -> bench_module.subprocess.CompletedProcess[str]:
        assert command == ["git", "rev-parse", "--verify", "HEAD"]
        assert kwargs["cwd"] == tmp_path
        return bench_module.subprocess.CompletedProcess(command, 0, stdout="b" * 40 + "\n")

    monkeypatch.setattr(bench_module.subprocess, "run", run_git)

    assert bench_module._checkout_commit() == "b" * 40


def test_checkout_dirty_reads_source_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_module = tmp_path / "src" / "qceval" / "core" / "bench.py"
    source_module.parent.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(bench_module, "__file__", str(source_module))

    def run_git(command: list[str], **kwargs: object) -> bench_module.subprocess.CompletedProcess[str]:
        assert command == ["git", "status", "--porcelain", "--untracked-files=normal"]
        assert kwargs["cwd"] == tmp_path
        return bench_module.subprocess.CompletedProcess(command, 0, stdout=" M src/qceval/core/bench.py\n")

    monkeypatch.setattr(bench_module.subprocess, "run", run_git)

    assert bench_module._checkout_dirty() is True


def test_checkout_dirty_is_unknown_outside_source_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wheel_module = tmp_path / "site-packages" / "qceval" / "core" / "bench.py"
    monkeypatch.setattr(bench_module, "__file__", str(wheel_module))

    assert bench_module._checkout_dirty() is None


def test_candidate_syntax_error_is_reported_as_compile_failure() -> None:
    adapter = Adaptor()
    task = adapter.load_tasks("qiskit", "core")[0]

    evaluation = adapter.evaluate(task, "def broken(:\n    pass\n")

    assert evaluation.compiled is False
    assert evaluation.ran is False
    assert evaluation.error_type == "SyntaxError"
    assert evaluation.error is not None and "SyntaxError" in evaluation.error
    assert "Traceback" not in evaluation.error


def test_candidate_exception_is_reported_as_runtime_failure() -> None:
    adapter = Adaptor()
    task = adapter.load_tasks("qiskit", "core")[0]
    code = "def grover_search_oracle_00():\n    raise ValueError('candidate boom')\n"

    evaluation = adapter.evaluate(task, code)

    assert evaluation.compiled is True
    assert evaluation.ran is False
    assert evaluation.error_type == "ValueError"
    assert evaluation.error == "ValueError: candidate boom"


def test_runtime_prompt_hash_mismatch_fails_closed() -> None:
    adapter = Adaptor()
    task = adapter.load_tasks("qiskit", "core")[0]
    changed = replace(task, prompt=task.prompt + "\nchanged")

    try:
        _validate_runtime_prompt_hash(changed, ContractRegistry.from_package("core"))
    except ValueError as exc:
        assert "runtime prompt identity mismatch" in str(exc)
    else:
        raise AssertionError("changed prompt must not pass runtime identity validation")
