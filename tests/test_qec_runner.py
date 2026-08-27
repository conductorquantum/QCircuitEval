from __future__ import annotations

import json
from pathlib import Path

from qceval.core.runner import BenchmarkRunner
from qceval.models import RunConfig, RunOptions
from tests.qec_support import SuiteFailingProvider, SuiteStubAdapter, SuiteStubProvider


def test_runner_records_suite_and_summary_by_suite() -> None:
    # Arrange
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m", suites=("core", "qec"))
    runner = BenchmarkRunner(config=config, provider=SuiteStubProvider(), adapter=SuiteStubAdapter())  # type: ignore[arg-type]

    # Act
    payload = runner.run()

    # Assert
    assert [record["suite"] for record in payload["results"]] == ["core", "qec"]
    assert payload["suites"] == ["core", "qec"]
    assert payload["summary"]["by_suite"]["qec"]["passed"] == 1


def test_resume_uses_suite_in_key(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "results.jsonl"
    config = RunConfig(provider="stub", frameworks=("qiskit",), source_hint=None, model="m", suites=("core", "qec"))
    first = BenchmarkRunner(
        config=config,
        provider=SuiteStubProvider(),
        adapter=SuiteStubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(stream_to=path),
    ).run()

    # Act
    resumed = BenchmarkRunner(
        config=config,
        provider=SuiteFailingProvider(),
        adapter=SuiteStubAdapter(),  # type: ignore[arg-type]
        options=RunOptions(resume_from=path, stream_to=tmp_path / "resumed.jsonl"),
    ).run()

    # Assert
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["suite"] for line in lines if line["kind"] == "result"] == [
        "core",
        "qec",
    ]
    assert first["results"] == resumed["results"]
    assert resumed["summary"]["passed"] == 2
