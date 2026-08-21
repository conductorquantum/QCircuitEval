"""Unitary matrix comparison utilities used by exact graders."""

from __future__ import annotations

from typing import Any

import numpy as np


def unitaries_equivalent(
    candidate: Any,
    expected: Any,
    *,
    tolerance: float,
    ignore_global_phase: bool,
) -> tuple[bool, float]:
    """Compare two unitary matrices with optional global-phase invariance.

    Args:
        candidate: Candidate unitary-like object.
        expected: Reference unitary-like object.
        tolerance: Maximum normalized Frobenius distance accepted.
        ignore_global_phase: Whether to align global phase before measuring
            distance.

    Returns:
        Pair ``(passed, distance)``.  Missing matrices or shape mismatches return
        ``False`` and infinite distance.
    """
    if candidate is None or expected is None:
        return False, float("inf")
    cand = np.asarray(candidate, dtype=complex)
    exp = np.asarray(expected, dtype=complex)
    if cand.shape != exp.shape:
        return False, float("inf")
    distance = _global_phase_distance(cand, exp) if ignore_global_phase else _distance(cand, exp)
    return distance <= tolerance, distance


def unitary_is_entangling(unitary: Any, *, tolerance: float = 1e-6, wires: list[int] | None = None) -> bool | None:
    """Return whether a net unitary is nonlocal on the given wires.

    "Nonlocal" here means *not a tensor product of single-qubit unitaries*
    (operator Schmidt rank greater than 1 across a single-qubit-vs-rest cut),
    which is the property the benchmark's entangling-gate floor is a syntactic
    proxy for. Note this is operator nonlocality, not state-entangling power:
    a ``SWAP`` is accepted (rank 4) even though it maps products to products,
    consistent with the syntactic floor counting ``swap`` as an entangler.

    Because the net unitary is invariant under padding, a hardcoded circuit
    wrapped in canceling entangler pairs (``CX(a,b); CX(a,b)``) clears the
    syntactic gate-count floors but stays a local product here. Restricting
    ``wires`` to the measured register closes the complementary exploit of
    parking a live entangler on never-excited ancillas (e.g. ``CZ`` between
    two ancillas that remain in the unreachable ``|11>`` control subspace):
    nonlocality on unmeasured wires alone then no longer satisfies the floor.
    A residual limitation remains by construction -- an entangler that touches
    a measured wire but acts trivially on the reachable subspace still counts
    -- so this check is an anti-shortcut floor, not an equivalence proof;
    operator-class tasks get the latter from the full-domain case-table oracle.

    Args:
        unitary: Candidate net unitary in canonical little-endian order (wire
            ``w`` at bit ``w``), or ``None``.
        tolerance: Relative singular-value threshold for rank counting.
        wires: Wires whose cuts must witness the nonlocality (typically the
            measured register). ``None`` tests every wire.

    Returns:
        ``True`` when at least one tested single-qubit cut has operator
        Schmidt rank greater than 1, ``False`` when every tested cut is a
        local factor, and ``None`` when no well-formed square power-of-two
        matrix is available (the caller falls back to the syntactic counts).
    """
    if unitary is None:
        return None
    mat = np.asarray(unitary, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 2:
        return None
    dim = mat.shape[0]
    num_qubits = dim.bit_length() - 1
    if (1 << num_qubits) != dim:
        return None
    if num_qubits < 2:
        return False
    tested = range(num_qubits) if wires is None else [w for w in wires if 0 <= w < num_qubits]
    # Axes 0..n-1 are row bits (big-endian), axes n..2n-1 are column bits.
    tensor = mat.reshape([2] * (2 * num_qubits))
    for qubit in tested:
        row_axis = num_qubits - 1 - qubit
        col_axis = 2 * num_qubits - 1 - qubit
        # Matricize as B(H_qubit) x B(H_rest): rows indexed by the qubit's
        # (row bit, col bit) pair, columns by the remaining bits.
        cut = np.moveaxis(tensor, (row_axis, col_axis), (0, 1)).reshape(4, -1)
        singular_values = np.linalg.svd(cut, compute_uv=False)
        if singular_values[1] > tolerance * singular_values[0]:
            return True
    return False


def bit_reverse_unitary(unitary: Any) -> np.ndarray | None:
    """Reindex a unitary between big-endian and little-endian qubit order.

    Cirq and PennyLane executors expose the circuit unitary in big-endian
    order (wire 0 most significant); the wire-indexed cut tests expect
    canonical little-endian (wire ``w`` at bit ``w``). Bit-reversing both the
    row and column indices converts between the two conventions (it is an
    involution).

    Args:
        unitary: Square power-of-two unitary, or ``None``.

    Returns:
        The reindexed unitary, or ``None`` when the input is malformed.
    """
    if unitary is None:
        return None
    mat = np.asarray(unitary, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 2:
        return None
    dim = mat.shape[0]
    num_qubits = dim.bit_length() - 1
    if (1 << num_qubits) != dim:
        return None
    perm = [int(format(index, f"0{num_qubits}b")[::-1], 2) for index in range(dim)]
    return mat[np.ix_(perm, perm)]


def u_gate_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Return single-qubit U-gate matrix.

    Args:
        theta: Polar rotation angle.
        phi: First phase angle.
        lam: Second phase angle.

    Returns:
        Complex 2x2 unitary matrix matching Qiskit's ``UGate`` convention.
    """
    return np.asarray(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ],
        dtype=complex,
    )


def _global_phase_distance(candidate: np.ndarray, expected: np.ndarray) -> float:
    inner = np.vdot(expected.reshape(-1), candidate.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 0 else 1.0
    return _distance(candidate, phase * expected)


def _distance(candidate: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - expected) / np.sqrt(candidate.size))
