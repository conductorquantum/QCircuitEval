"""Bounded adaptive-QIR reader for CUDA-Q semantic lowering.

The reader intentionally supports only compiler forms observed in CUDA-Q's
adaptive profile: static qubit/result pointers, intrinsic QIS calls, bounded
control arrays, constant classical arithmetic, and structured branches over
measurement results. Unknown instructions fail closed.

This module is a compatibility facade over ``qceval.frameworks.cudaq.qir``.
"""

from __future__ import annotations

from qceval.frameworks.cudaq.qir.cfg import parse_adaptive_qir
from qceval.frameworks.cudaq.qir.models import QirParseError, QirParseLimits

__all__ = ["QirParseError", "QirParseLimits", "parse_adaptive_qir"]
