"""Bundled task asset loading."""

from __future__ import annotations

import json
from typing import Any

from qceval.assets._resources import task_resource
from qceval.models import Framework, Suite


def load_tasks(framework: Framework, suite: Suite = "core") -> dict[str, dict[str, Any]]:
    """Load raw bundled task payloads for a framework.

    Args:
        framework: Framework whose package asset should be read.
        suite: Benchmark suite whose package asset should be read.

    Returns:
        Dictionary keyed by zero-padded task id.

    Raises:
        ValueError: If ``suite`` is not a packaged suite name.
        FileNotFoundError: If the package asset is missing.
        json.JSONDecodeError: If a JSONL row is malformed.
    """

    lines = task_resource(suite, framework).read_text(encoding="utf-8").splitlines()
    return {str(task["task_id"]).zfill(2): task for task in (json.loads(line) for line in lines if line.strip())}
