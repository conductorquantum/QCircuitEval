from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_run_records import merge_run_records

from qceval.cli import main as qceval_main


def test_merge_run_records_recomputes_framework_summary(tmp_path: Path) -> None:
    qiskit = tmp_path / "qiskit.jsonl"
    cirq = tmp_path / "cirq.jsonl"
    merged = tmp_path / "merged.jsonl"
    assert (
        qceval_main(["run", "--provider", "smoke", "--framework", "qiskit", "--tasks", "1", "--out", str(qiskit)]) == 0
    )
    assert qceval_main(["run", "--provider", "smoke", "--framework", "cirq", "--tasks", "1", "--out", str(cirq)]) == 0

    merge_run_records([qiskit, cirq], merged)

    lines = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines()]
    assert [line["kind"] for line in lines] == ["result", "result", "summary"]
    assert [line["framework"] for line in lines[:-1]] == ["qiskit", "cirq"]
    assert lines[-1]["summary"]["total_tasks"] == 2
    assert set(lines[-1]["summary"]["by_framework"]) == {"qiskit", "cirq"}


def test_merge_run_records_rejects_incomplete_shard(tmp_path: Path) -> None:
    shard = tmp_path / "partial.jsonl"
    shard.write_text('{"kind":"result"}\n', encoding="utf-8")

    try:
        merge_run_records([shard], tmp_path / "merged.jsonl")
    except ValueError as exc:
        assert "no summary" in str(exc)
    else:
        raise AssertionError("incomplete shard was accepted")
