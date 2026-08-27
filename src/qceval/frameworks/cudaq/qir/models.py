"""Dataclasses, parse limits, and fail-closed QIR errors."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from qceval.semantics.ir import Operation

_MAX_QIR_BYTES = 32 * 1024 * 1024


class QirParseError(ValueError):
    """Stable fail-closed adaptive-QIR parsing failure."""


@dataclass(frozen=True)
class QirParseLimits:
    """Deterministic resource limits for compiler-IR parsing."""

    max_text_bytes: int = _MAX_QIR_BYTES
    max_blocks: int = 4096
    max_instructions: int = 100_000
    max_branch_depth: int = 64


@dataclass(frozen=True)
class _Terminator:
    condition: str | bool | None
    targets: tuple[str, ...]


@dataclass(frozen=True)
class _Block:
    label: str
    instructions: tuple[str, ...]
    terminator: _Terminator


@dataclass(frozen=True)
class _BitPredicate:
    bit: int
    inverted: bool = False


@dataclass(frozen=True)
class _QubitRef:
    wire: int


@dataclass(frozen=True)
class _ComplexArray:
    values: tuple[complex, ...]


@dataclass
class _Memory:
    values: list[Any]


@dataclass(frozen=True)
class _MemoryRef:
    memory: _Memory
    index: int


@dataclass
class _QubitArray:
    values: list[int | None]


@dataclass(frozen=True)
class _QubitArraySlot:
    array: _QubitArray
    index: int


@dataclass
class _State:
    values: dict[str, Any] = field(default_factory=dict)
    predecessor: str | None = None

    def clone(self) -> _State:
        """Return an independent branch state."""
        return copy.deepcopy(self)


@dataclass
class _ParseContext:
    blocks: dict[str, _Block]
    state: _State
    operations: list[Operation]
    required_qubits: int
    required_results: int
    limits: QirParseLimits
    visited_steps: int = 0
