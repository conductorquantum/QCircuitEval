"""CUDA-Q result normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from qceval.evals.probabilities import ProbabilityVector, as_prob_array, bitstring_index, num_bits


def _cudaq_counts_to_probabilities(counts: Mapping[Any, Any], *, from_sample: bool = False) -> np.ndarray:
    """Normalize CUDA-Q counts into QCircuitEval probability order.

    Iterates the dict once and scatters into a zero-initialized vector
    (O(k) for k unique bitstrings). When ``from_sample`` is True, reverses
    each bitstring at parse time to map CUDA-Q's left-to-right qubit order
    into QCircuitEval's integer-index order (qubit 0 = LSB).
    """
    cleaned = _clean_counts(dict(counts.items()))
    n_bits = _counts_n_bits(cleaned)
    out = np.zeros(1 << n_bits, dtype=float)
    for bitstring, value in cleaned.items():
        out[bitstring_index(bitstring[::-1] if from_sample else bitstring)] += value
    return as_prob_array(out)


def _clean_counts(counts: Mapping[Any, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in counts.items():
        clean_key = str(key).split()[0]
        cleaned[clean_key] = cleaned.get(clean_key, 0.0) + float(value)
    if not cleaned:
        raise ValueError("counts dictionary is empty")
    return cleaned


def _counts_n_bits(counts: Mapping[str, float]) -> int:
    n_bits = len(next(iter(counts)))
    if any(len(bitstring) != n_bits for bitstring in counts):
        raise ValueError("counts bitstrings must have equal length")
    return n_bits


def _probabilities_from_array(values: Any) -> np.ndarray | None:
    if isinstance(values, str | bytes | bytearray) or not isinstance(values, np.ndarray | Sequence):
        return None
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1:
        return None
    try:
        return as_prob_array(arr)
    except ValueError:
        return None


def _unitary_from_array(values: Any) -> np.ndarray | None:
    if isinstance(values, str | bytes | bytearray) or not isinstance(values, np.ndarray | Sequence):
        return None
    try:
        arr = np.asarray(values, dtype=complex)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    dimension = arr.shape[0]
    if dimension == 0 or dimension & (dimension - 1):
        return None
    return arr


def _array_metadata(probabilities: np.ndarray, *, method: str) -> dict[str, Any]:
    return {
        "probability_method": method,
        "num_qubits": _num_qubits(probabilities),
        "measurement_count": 0,
        "non_measurement_operation_count": 0,
        "entangling_gate_count": 0,
        "circuit_depth": 0,
        "repeated_block_count": 0,
        "measurement_pairs": [],
        "has_measurements": False,
    }


def _project_measured_probabilities(probabilities: np.ndarray, measured_qubits: Sequence[int]) -> np.ndarray:
    if not measured_qubits:
        return probabilities
    out = np.zeros(1 << len(measured_qubits), dtype=float)
    for basis_index, probability in enumerate(probabilities):
        out[_measured_index(basis_index, measured_qubits)] += float(probability)
    return as_prob_array(out)


def _measured_index(basis_index: int, measured_qubits: Sequence[int]) -> int:
    # QCircuitEval integer-index order keeps qubit 0 as the least-significant
    # bit, so the first listed measured qubit maps to output bit 0.
    measured_index = 0
    for position, qubit_index in enumerate(measured_qubits):
        measured_index |= ((basis_index >> qubit_index) & 1) << position
    return measured_index


def _num_qubits(values: ProbabilityVector) -> int:
    length = len(values)
    if length == 0 or length & (length - 1):
        raise ValueError(f"state/probability length must be a power of 2, got {length}")
    return num_bits(values)
