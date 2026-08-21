"""Shared helpers for structural circuit metadata."""

from __future__ import annotations

from collections.abc import Sequence

OperationSignature = tuple[str, tuple[int, ...]]


def detect_repeated_blocks(ops: Sequence[OperationSignature]) -> int:
    """Return largest adjacent repetition count in an operation stream.

    Args:
        ops: Ordered operation family and qubit-index signatures.

    Returns:
        Largest number of adjacent identical operation blocks.
    """
    if not ops:
        return 0
    max_count = 1
    for width in range(1, len(ops) // 2 + 1):
        for start in range(0, len(ops) - (2 * width) + 1):
            block = ops[start : start + width]
            count = 1
            cursor = start + width
            while cursor + width <= len(ops) and ops[cursor : cursor + width] == block:
                count += 1
                cursor += width
            max_count = max(max_count, count)
    return max_count
