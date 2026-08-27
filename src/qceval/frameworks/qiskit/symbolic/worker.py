"""Bounded subprocess entry point for symbolic family proofs."""

from __future__ import annotations

import json
import sys

from qceval.frameworks.qiskit.symbolic.proof import _prove


def main() -> int:
    """Read one request and emit one bounded JSON proof response.

    Returns:
        Process exit code.
    """
    try:
        payload = json.loads(sys.stdin.read())
        result = _prove(payload)
    except Exception as exc:  # noqa: BLE001 - never emit a worker traceback.
        result = {
            "outcome": "inconclusive",
            "reason": f"symbolic_worker_protocol:{type(exc).__name__}",
            "certified_error_bound": None,
            "gate_families": [],
            "peak_expression_nodes": 0,
            "residuals": [],
        }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0
