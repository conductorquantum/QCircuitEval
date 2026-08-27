"""Probability-vector and bitstring helpers for graders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import permutations
from math import log2
from typing import Any, TypeAlias

import numpy as np

ProbabilityVector: TypeAlias = Sequence[float] | np.ndarray[Any, Any]


def as_prob_array(values: ProbabilityVector) -> np.ndarray:
    """Normalize values as a one-dimensional probability vector.

    Args:
        values: Sequence or NumPy array of nonnegative probabilities.

    Returns:
        Float NumPy array normalized to sum to one.

    Raises:
        ValueError: If the input is not one-dimensional, has non-power-of-two
            length, or has non-positive total mass.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("probabilities must be a 1-D vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError("probabilities must be finite")
    if np.any(arr < 0):
        raise ValueError("probabilities must be nonnegative")
    if len(arr) == 0 or len(arr) & (len(arr) - 1):
        raise ValueError(f"probability length must be a power of 2, got {len(arr)}")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return arr / total


def num_bits(values: ProbabilityVector) -> int:
    """Return bit width represented by a probability vector.

    Args:
        values: Probability vector whose length is a power of two.

    Returns:
        Number of bits represented by the vector length.
    """
    return int(log2(len(values)))


def bitstrings_for_probs(values: ProbabilityVector) -> list[str]:
    """Return integer-ordered bitstrings for a probability vector.

    Args:
        values: Probability vector whose length defines bit width.

    Returns:
        Bitstrings from ``0`` to ``len(values) - 1`` padded to the vector bit
        width.
    """
    n_bits = num_bits(values)
    return [format(i, f"0{n_bits}b") for i in range(len(values))]


def bit_reverse_probs(values: ProbabilityVector) -> np.ndarray:
    """Return the probability vector with the measured register's bit order reversed.

    The amplitude at basis index ``i`` (bits ``b_{n-1}...b_0``) moves to the index
    whose bit order is reversed (``b_0...b_{n-1}``). This is the distribution-level
    analogue of the state oracle's bit-reversal quotient: it reconciles a candidate
    measured in the opposite (equally admissible) endianness with the reference,
    without introducing arbitrary qubit permutations.

    Args:
        values: Probability vector whose length is a power of two.

    Returns:
        Bit-reversed probability vector (a permutation of the input).
    """
    arr = as_prob_array(values)
    n_bits = num_bits(arr)
    if n_bits <= 1:
        return arr
    out = np.zeros_like(arr)
    for index in range(len(arr)):
        reversed_index = int(format(index, f"0{n_bits}b")[::-1], 2)
        out[reversed_index] = arr[index]
    return out


def bitstring_index(bitstring: str) -> int:
    """Return integer index for a binary bitstring.

    Args:
        bitstring: Binary string such as ``"010"``.

    Returns:
        Integer index represented by ``bitstring``.
    """
    return int(bitstring, 2)


def distribution_from_support(
    support: Iterable[str], n_bits: int, weights: Mapping[str, float] | None = None
) -> np.ndarray:
    """Build a probability vector from support bitstrings.

    Args:
        support: Bitstrings with nonzero probability.
        n_bits: Number of bits in the full distribution.
        weights: Optional per-bitstring weights.  When omitted, support mass is
            uniform.

    Returns:
        Normalized probability vector of length ``2 ** n_bits``.

    Raises:
        ValueError: If support is empty or weighted support has no mass.
    """
    out = np.zeros(2**n_bits, dtype=float)
    support_list = list(support)
    if not support_list:
        raise ValueError("support cannot be empty")
    if weights is None:
        for bitstring in support_list:
            out[bitstring_index(bitstring)] = 1.0 / len(support_list)
        return out
    for bitstring in support_list:
        out[bitstring_index(bitstring)] = float(weights[bitstring])
    total = float(out.sum())
    if total <= 0:
        raise ValueError("support weights sum to zero")
    return out / total


def observed_support(probs: np.ndarray, tau: float) -> frozenset[str]:
    """Return bitstrings whose probability is greater than threshold.

    Args:
        probs: Probability vector.
        tau: Strict probability threshold.

    Returns:
        Frozen set of observed bitstrings.
    """
    return frozenset(
        bit for bit, probability in zip(bitstrings_for_probs(probs), probs, strict=True) if probability > tau
    )


def support_matches(
    observed: frozenset[str],
    expected: frozenset[str],
    *,
    permutation_invariant: bool,
) -> tuple[bool, tuple[int, ...] | None]:
    """Compare observed support to expected support.

    Args:
        observed: Bitstrings observed above threshold.
        expected: Accepted bitstrings.
        permutation_invariant: Whether qubit-order permutations may match.

    Returns:
        Pair ``(matched, permutation)``.  ``permutation`` is ``None`` unless a
        permutation-invariant match was required.
    """
    if observed == expected:
        return True, None
    if not permutation_invariant or not expected:
        return False, None
    return _permutation_match(observed, expected)


def top_k_bitstrings(probs: np.ndarray, top_k: int, *, sig_digits: int = 10) -> list[str]:
    """Return most likely bitstrings with deterministic tie ordering.

    Probabilities within floating-point noise of each other are treated as
    tied (rounded to *sig_digits* significant figures before sorting), so that
    statevector simulation noise does not produce non-deterministic peak sets.

    Args:
        probs: Probability vector.
        top_k: Number of bitstrings to return.
        sig_digits: Significant figures for tie-breaking quantization.

    Returns:
        Top bitstrings sorted by probability descending, then bitstring.

    Raises:
        ValueError: If ``top_k`` is not positive.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    bitstrings = bitstrings_for_probs(probs)

    def _quantize(p: float) -> float:
        if p <= 0:
            return 0.0
        mag = 10 ** (sig_digits - 1 - int(np.floor(np.log10(p))))
        return round(p * mag) / mag

    order = sorted(range(len(probs)), key=lambda i: (-_quantize(float(probs[i])), bitstrings[i]))
    return [bitstrings[i] for i in order[:top_k]]


def project_and_normalize(probs: np.ndarray, bitstrings: Iterable[str]) -> np.ndarray:
    """Project probabilities onto bitstrings and renormalize.

    Args:
        probs: Full probability vector.
        bitstrings: Ordered bitstrings to project.

    Returns:
        Projected probability vector, normalized when projected mass is
        positive.
    """
    out = np.asarray([probs[bitstring_index(bit)] for bit in bitstrings], dtype=float)
    total = float(out.sum())
    return out / total if total > 0 else out


def hellinger_fidelity(probs: ProbabilityVector, expected: ProbabilityVector) -> float:
    """Compute classical Hellinger fidelity.

    The returned value is ``(sum_i sqrt(p_i * q_i)) ** 2`` and is clamped to
    ``[0.0, 1.0]`` to absorb floating-point noise.

    Args:
        probs: Candidate probability vector.
        expected: Reference probability vector.

    Returns:
        Hellinger fidelity between normalized vectors.

    Raises:
        ValueError: If normalized vectors have different lengths.
    """
    p = as_prob_array(probs)
    q = as_prob_array(expected)
    if len(p) != len(q):
        raise ValueError(f"shape mismatch: model len {len(p)}, expected len {len(q)}")
    value = float(np.square(np.sum(np.sqrt(p * q))))
    return float(np.clip(value, 0.0, 1.0))


def hellinger_infidelity(probs: ProbabilityVector, expected: ProbabilityVector) -> float:
    """Compute classical Hellinger infidelity.

    Args:
        probs: Candidate probability vector.
        expected: Reference probability vector.

    Returns:
        ``1 - hellinger_fidelity(probs, expected)``, clamped to ``[0.0, 1.0]``.

    Raises:
        ValueError: If normalized vectors have different lengths.
    """
    return float(np.clip(1.0 - hellinger_fidelity(probs, expected), 0.0, 1.0))


def kl_divergence(reference: ProbabilityVector, candidate: ProbabilityVector, *, epsilon: float = 1e-12) -> float:
    """Compute the Kullback-Leibler divergence ``D_KL(reference || candidate)``.

    This is the acceptance metric specified by the QCircuitEval grading protocol
    for probabilistic tasks: both distributions are additively smoothed by
    ``epsilon`` and renormalized before the divergence is computed, so a
    candidate that assigns zero mass to a state the reference supports yields a
    large but finite divergence rather than ``+inf``.

    Args:
        reference: Reference (canonical) probability vector ``P``.
        candidate: Candidate probability vector ``Q``.
        epsilon: Additive smoothing constant applied to both vectors before
            renormalization.

    Returns:
        Nonnegative divergence ``sum_x P(x) log(P(x) / Q(x))`` in nats.

    Raises:
        ValueError: If the normalized vectors have different lengths.
    """
    p = as_prob_array(reference)
    q = as_prob_array(candidate)
    if len(p) != len(q):
        raise ValueError(f"shape mismatch: reference len {len(p)}, candidate len {len(q)}")
    p = (p + epsilon) / (1.0 + epsilon * len(p))
    q = (q + epsilon) / (1.0 + epsilon * len(q))
    return float(np.sum(p * np.log(p / q)))


def _permutation_match(observed: frozenset[str], expected: frozenset[str]) -> tuple[bool, tuple[int, ...] | None]:
    n_bits = len(next(iter(expected)))
    for perm in permutations(range(n_bits)):
        permuted = frozenset("".join(bit[i] for i in perm) for bit in expected)
        if observed == permuted:
            return True, perm
    return False, None
