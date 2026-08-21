"""Shared helpers for generated smoke-provider source."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _target_qubits(target: str) -> int:
    if target == "ccx":
        return 3
    if target in {"controlled_h", "cx"}:
        return 2
    return 1


def _normalize(probabilities: Sequence[float]) -> list[float]:
    arr = np.asarray(probabilities, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return [float(probability) for probability in (arr / total)]


def _num_bits(probabilities: Sequence[float]) -> int:
    return int(math.log2(len(probabilities)))
