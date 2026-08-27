#!/usr/bin/env python3
"""Prove benchmark content is byte-identical to a frozen Git baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTENT_PATHS = (
    "src/qceval/assets",
    "src/qceval/evals",
    "src/qceval/frameworks",
    "src/qceval/semantics",
    "src/qceval/core/bench.py",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = _git("rev-parse", "--verify", f"{args.baseline}^{{commit}}")
    paths = _git_lines("ls-tree", "-r", "--name-only", baseline, "--", *CONTENT_PATHS)
    if not paths:
        parser.error("baseline contains no benchmark-content paths")
    entries = []
    mismatches = []
    for relative in paths:
        baseline_bytes = subprocess.run(
            ["git", "show", f"{baseline}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        worktree_path = Path(relative)
        worktree_bytes = worktree_path.read_bytes() if worktree_path.is_file() else None
        baseline_sha = hashlib.sha256(baseline_bytes).hexdigest()
        worktree_sha = None if worktree_bytes is None else hashlib.sha256(worktree_bytes).hexdigest()
        status = "identical" if baseline_sha == worktree_sha else "mismatch"
        entry = {
            "path": relative,
            "baseline_sha256": baseline_sha,
            "worktree_sha256": worktree_sha,
            "status": status,
        }
        entries.append(entry)
        if status != "identical":
            mismatches.append(entry)

    payload: dict[str, Any] = {
        "schema_version": "1",
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_content_commit": baseline,
        "content_roots": list(CONTENT_PATHS),
        "file_count": len(entries),
        "byte_identical": not mismatches,
        "files": entries,
        "mismatches": mismatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"benchmark content files={len(entries)} byte_identical={not mismatches}")
    return 0 if not mismatches else 2


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _git_lines(*args: str) -> list[str]:
    return [line for line in _git(*args).splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())
