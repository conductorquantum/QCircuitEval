"""CUDA-Q source/entry-point carrier used across the grading pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CudaqProgram:
    """Carrier for CUDA-Q source, concrete arguments, and compiled kernel.

    ``kernel`` is the native object produced by the already-sandboxed candidate
    execution. Keeping it on this internal carrier lets semantic lowering ask
    CUDA-Q for the same compiler IR that was executed, instead of reinterpreting
    candidate Python syntax.
    """

    code: str
    entry_point: str
    call_args: tuple[Any, ...] = ()
    kernel: Any | None = field(default=None, repr=False, compare=False)
