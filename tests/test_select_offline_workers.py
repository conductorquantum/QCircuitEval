from __future__ import annotations

import json
from pathlib import Path

from scripts.select_offline_workers import select_offline_workers


def _write_result(path: Path, *, passed: bool = True) -> None:
    payload = {
        "kind": "result",
        "suite": "core",
        "framework": "qiskit",
        "task_id": "task-1",
        "status": "passed" if passed else "failed",
        "evaluation": {"passed": passed},
        "error_taxonomy": None,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_selects_fastest_stable_worker_setting(tmp_path: Path) -> None:
    (tmp_path / "attempts.tsv").write_text("2\t3000\tpassed\n4\t1000\tpassed\n8\t2000\tpassed\n", encoding="utf-8")
    for workers in (2, 4, 8):
        _write_result(tmp_path / f"workers-{workers}.jsonl")

    result = select_offline_workers(tmp_path)

    assert result["selected_evaluation_workers"] == 4
    assert all(attempt["stable"] for attempt in result["attempts"])


def test_excludes_worker_setting_with_grader_variance(tmp_path: Path) -> None:
    (tmp_path / "attempts.tsv").write_text("2\t3000\tpassed\n4\t2000\tpassed\n8\t1000\tpassed\n", encoding="utf-8")
    _write_result(tmp_path / "workers-2.jsonl")
    _write_result(tmp_path / "workers-4.jsonl")
    _write_result(tmp_path / "workers-8.jsonl", passed=False)

    result = select_offline_workers(tmp_path)

    assert result["selected_evaluation_workers"] == 4
    assert result["attempts"][2]["stable"] is False
