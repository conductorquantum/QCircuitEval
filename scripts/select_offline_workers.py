#!/usr/bin/env python3
"""Select the fastest stable offline-grader worker setting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def select_offline_workers(root: Path) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    baseline: str | None = None
    for raw in (root / "attempts.tsv").read_text(encoding="utf-8").splitlines():
        workers, elapsed, status = raw.split("\t")
        row: dict[str, Any] = {
            "workers": int(workers),
            "elapsed_ms": int(elapsed),
            "status": status,
            "stable": False,
        }
        path = root / f"workers-{workers}.jsonl"
        if status == "passed":
            keyed: dict[str, dict[str, Any]] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                if payload.get("kind") != "result":
                    continue
                key = json.dumps(
                    [payload.get("suite"), payload.get("framework"), payload.get("task_id")],
                    separators=(",", ":"),
                )
                keyed[key] = {name: payload.get(name) for name in ("status", "evaluation", "error_taxonomy")}
            digest = hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()
            row["evaluation_sha256"] = digest
            if baseline is None:
                baseline = digest
            row["stable"] = digest == baseline
        attempts.append(row)
    eligible = [row for row in attempts if row["status"] == "passed" and row["stable"]]
    if not eligible:
        raise ValueError("no calibration setting completed without failure or grader variance")
    selected = min(eligible, key=lambda row: (row["elapsed_ms"], row["workers"]))
    payload = {
        "schema_version": "1",
        "attempts": attempts,
        "selected_evaluation_workers": selected["workers"],
    }
    (root / "calibration.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_dir", type=Path)
    args = parser.parse_args()
    print(select_offline_workers(args.calibration_dir)["selected_evaluation_workers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
