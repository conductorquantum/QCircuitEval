from __future__ import annotations

import json
from pathlib import Path

from qceval.core.io import JsonlRunWriter, read_completed
from qceval.models import BenchmarkRecord, ProviderResponse, QCEvalEvaluation


def test_jsonl_run_writer_appends_results_and_summary(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "results.jsonl"
    writer = JsonlRunWriter(path)
    record = _record("01")
    payload = {
        "schema_version": "qceval.run.v1",
        "provider": "smoke",
        "model": "smoke",
        "configuration_id": "smoke__effort-max",
        "suites": ["core"],
        "qceval": {},
        "results": [record.to_dict()],
        "summary": {"passed": 1},
    }

    # Act
    writer.append(record)
    writer.finalize(payload)
    writer.close()

    # Assert
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["kind"] == "result"
    assert lines[1]["kind"] == "summary"
    assert lines[1]["configuration_id"] == "smoke__effort-max"


def test_jsonl_writer_persistent_handle(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    writer = JsonlRunWriter(path)
    for _ in range(10):
        writer.append(_record("01"))
    writer.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10


def test_jsonl_writer_sync_interval(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    writer = JsonlRunWriter(path, sync_interval=5)
    for _ in range(3):
        writer.append(_record("01"))
    writer.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_jsonl_writer_close_flushes(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    writer = JsonlRunWriter(path, sync_interval=100)
    writer.append(_record("01"))
    writer.append(_record("02"))
    writer.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_jsonl_writer_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    with JsonlRunWriter(path) as writer:
        writer.append(_record("01"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_read_completed_skips_partial_lines(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"kind": "result", **_record("01").to_dict()}) + "\n" + "{partial\n",
        encoding="utf-8",
    )

    # Act
    completed = read_completed(path)

    # Assert
    assert completed[("core", "qiskit", "01", 0, 0)]["status"] == "passed"


def test_read_completed_skips_malformed_resume_indices(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    payload = {
        "kind": "result",
        "suite": "core",
        "framework": "qiskit",
        "task_id": "01",
        "sample_index": "not-an-integer",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert read_completed(path) == {}


def test_read_completed_loads_json_run_envelope(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    record = _record("01").to_dict()
    path.write_text(
        json.dumps({"schema_version": "qceval.run.v2", "results": [record]}, indent=2) + "\n",
        encoding="utf-8",
    )

    completed = read_completed(path)

    assert completed[("core", "qiskit", "01", 0, 0)]["status"] == "passed"
    assert "kind" not in completed[("core", "qiskit", "01", 0, 0)]


def test_read_completed_distinguishes_sample_index(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "results.jsonl"
    first = _record("01")
    second = BenchmarkRecord(
        framework="qiskit",
        task_id="01",
        sample_index=1,
        entry_point="answer",
        category="cat",
        provider="smoke",
        model="smoke",
        status="passed",
        provider_response=ProviderResponse(code="code", model="smoke"),
        evaluation=QCEvalEvaluation(compiled=True, ran=True, passed=True),
    )
    path.write_text(
        "\n".join(
            [
                json.dumps({"kind": "result", **first.to_dict()}),
                json.dumps({"kind": "result", **second.to_dict()}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Act
    completed = read_completed(path)

    # Assert
    assert set(completed) == {
        ("core", "qiskit", "01", 0, 0),
        ("core", "qiskit", "01", 1, 0),
    }


def _record(task_id: str) -> BenchmarkRecord:
    return BenchmarkRecord(
        framework="qiskit",
        task_id=task_id,
        entry_point="answer",
        category="cat",
        provider="smoke",
        model="smoke",
        status="passed",
        provider_response=ProviderResponse(code="code", model="smoke"),
        evaluation=QCEvalEvaluation(compiled=True, ran=True, passed=True),
    )
