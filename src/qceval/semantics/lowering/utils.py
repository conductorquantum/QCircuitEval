"""Shared deterministic lowering helpers."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from qceval.semantics.ir import Parameter, ParameterKind
from qceval.semantics.lowering.base import LoweringError, LoweringResult, LoweringStatus


def normalize_parameter(value: Any) -> Parameter:
    """Normalize one finite numeric, symbolic, or textual parameter.

    Args:
        value: Framework-native parameter value.

    Returns:
        Deterministic IR parameter.
    """
    if isinstance(value, bool):
        return Parameter(ParameterKind.TEXT, str(value).lower())
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("operation parameter must be finite")
        return Parameter(ParameterKind.NUMBER, "0" if numeric == 0 else format(numeric, ".17g").lower())
    free_symbols = getattr(value, "free_symbols", ()) or getattr(value, "parameters", ())
    if free_symbols:
        return Parameter(ParameterKind.SYMBOL, str(value))
    try:
        complex_value = complex(value)
    except (TypeError, ValueError):
        return Parameter(ParameterKind.TEXT, str(value))
    if not (math.isfinite(complex_value.real) and math.isfinite(complex_value.imag)):
        raise ValueError("operation parameter must be finite")
    if abs(complex_value.imag) <= 1e-15:
        text = "0" if complex_value.real == 0 else format(complex_value.real, ".17g").lower()
        return Parameter(ParameterKind.NUMBER, text)
    return Parameter(ParameterKind.TEXT, f"{complex_value.real:.17g}{complex_value.imag:+.17g}j")


def matrix_sha256(matrix: Any) -> str:
    """Hash a finite complex matrix with pinned complex128 bytes.

    Args:
        matrix: Square framework-native matrix.

    Returns:
        Shape- and content-addressed SHA-256 digest.
    """
    value = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError("semantic matrix must be finite and square")
    payload = f"{value.shape[0]}x{value.shape[1]}:".encode() + value.tobytes()
    return hashlib.sha256(payload).hexdigest()


def statevector_sha256(statevector: Any) -> str:
    """Hash a finite complex statevector with pinned complex128 bytes.

    Args:
        statevector: One-dimensional framework-native statevector.

    Returns:
        Dimension- and content-addressed SHA-256 digest.
    """
    value = np.ascontiguousarray(np.asarray(statevector, dtype=np.complex128))
    if value.ndim != 1 or value.size < 1 or not np.all(np.isfinite(value)):
        raise ValueError("semantic statevector must be finite and one-dimensional")
    return hashlib.sha256(f"{value.size}:".encode() + value.tobytes()).hexdigest()


def matrix_sha256_from_bytes(raw: bytes, dimension: int) -> str:
    """Hash raw complex128 matrix bytes with an explicit dimension.

    Args:
        raw: Contiguous complex128 matrix payload.
        dimension: Square matrix dimension.

    Returns:
        Shape- and content-addressed SHA-256 digest.
    """
    return hashlib.sha256(f"{dimension}x{dimension}:".encode() + raw).hexdigest()


def statevector_sha256_from_bytes(raw: bytes, dimension: int) -> str:
    """Hash raw complex128 statevector bytes with an explicit dimension.

    Args:
        raw: Contiguous complex128 statevector payload.
        dimension: Vector dimension.

    Returns:
        Dimension- and content-addressed SHA-256 digest.
    """
    return hashlib.sha256(f"{dimension}:".encode() + raw).hexdigest()


def bounded_matrix_semantic_data(
    matrix: Any,
    *,
    max_dimension: int = 16,
    wire_order: str = "big_endian",
) -> tuple[tuple[str, str], ...]:
    """Encode one small exact matrix for framework-neutral simulation.

    Args:
        matrix: Square framework-native matrix.
        max_dimension: Maximum admitted row/column dimension.
        wire_order: Whether the first declared wire is the local matrix's
            most- or least-significant subsystem.

    Returns:
        Content hash and canonical complex128 byte payload.

    Raises:
        ValueError: If the matrix is invalid or exceeds the bound.
    """
    value = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] > max_dimension:
        raise ValueError("semantic matrix exceeds bounded dense lowering")
    if not np.all(np.isfinite(value)):
        raise ValueError("semantic matrix must be finite")
    if wire_order not in {"big_endian", "little_endian"}:
        raise ValueError("semantic matrix wire order is invalid")
    return (
        ("matrix_sha256", matrix_sha256(value)),
        ("matrix_complex128_hex", value.tobytes().hex()),
        ("matrix_wire_order", wire_order),
    )


def bounded_statevector_semantic_data(
    statevector: Any,
    *,
    max_dimension: int = 4096,
    wire_order: str = "little_endian",
) -> tuple[tuple[str, str], ...]:
    """Encode one normalized finite statevector for exact IR simulation.

    Args:
        statevector: One-dimensional framework-native statevector.
        max_dimension: Maximum admitted vector dimension.
        wire_order: Whether the first declared wire is the local vector's
            most- or least-significant subsystem.

    Returns:
        Content hash and canonical complex128 byte payload.

    Raises:
        ValueError: If the statevector is invalid or exceeds the bound.
    """

    value = np.ascontiguousarray(np.asarray(statevector, dtype=np.complex128))
    if value.ndim != 1 or value.size < 1 or value.size > max_dimension or value.size & (value.size - 1):
        raise ValueError("semantic statevector has invalid bounded dimension")
    if not np.all(np.isfinite(value)) or abs(float(np.linalg.norm(value)) - 1.0) > 1e-10:
        raise ValueError("semantic statevector must be finite and normalized")
    if wire_order not in {"big_endian", "little_endian"}:
        raise ValueError("semantic statevector wire order is invalid")
    raw = value.tobytes()
    digest = statevector_sha256(value)
    return (
        ("statevector_sha256", digest),
        ("statevector_complex128_hex", raw.hex()),
        ("statevector_wire_order", wire_order),
    )


def lowering_failure(
    status: LoweringStatus,
    reason: str,
    *,
    node_kind: str | None = None,
    source_location: str | None = None,
    detail: str | None = None,
) -> LoweringResult:
    """Build a typed non-verdict lowering failure.

    Args:
        status: Unsupported, execution-error, or resource-limit status.
        reason: Stable reason code.
        node_kind: Optional unsupported node kind.
        source_location: Optional diagnostic location.
        detail: Optional non-semantic detail.

    Returns:
        Validated failure result.
    """
    return LoweringResult(
        status,
        error=LoweringError(reason, node_kind=node_kind, source_location=source_location, detail=detail),
    )
