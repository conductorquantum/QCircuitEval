"""Semantic entangling floor on the net unitary.

The syntactic ``min_entangling_gate_count`` floor counts gate names, so a
candidate can hardcode the answer with local gates and clear the floor with a
canceling entangler pair (``CX(a,b); CX(a,b)``): the interaction graph gains an
edge while the net computation is unchanged. The net unitary is invariant under
that padding, so when it is available the entangling floor is also enforced
semantically: the net unitary must not factor into single-qubit unitaries.
"""

from __future__ import annotations

import numpy as np

from qceval.evals.unitaries import bit_reverse_unitary, unitary_is_entangling

CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


class TestUnitaryIsEntangling:
    def test_local_product_is_not_entangling(self) -> None:
        assert unitary_is_entangling(np.kron(np.kron(X, H), X)) is False

    def test_identity_is_not_entangling(self) -> None:
        assert unitary_is_entangling(np.eye(8, dtype=complex)) is False

    def test_cx_is_entangling(self) -> None:
        assert unitary_is_entangling(CX) is True

    def test_swap_is_entangling(self) -> None:
        assert unitary_is_entangling(SWAP) is True

    def test_canceling_cx_pair_is_not_entangling(self) -> None:
        assert unitary_is_entangling(CX @ CX) is False

    def test_entangler_on_subregister_is_entangling(self) -> None:
        assert unitary_is_entangling(np.kron(CX, np.eye(2, dtype=complex))) is True

    def test_missing_or_malformed_is_none(self) -> None:
        assert unitary_is_entangling(None) is None
        assert unitary_is_entangling(np.zeros((3, 3), dtype=complex)) is None
        assert unitary_is_entangling(np.zeros(4, dtype=complex)) is None

    def test_single_qubit_unitary_is_not_entangling(self) -> None:
        assert unitary_is_entangling(H) is False

    def test_wires_restriction_ignores_ancilla_only_nonlocality(self) -> None:
        # CZ on wires 1,2 with wire 0 local: nonlocality exists, but not on
        # wire 0's cut, so a measured-register restriction to [0] rejects it.
        cz = np.diag([1, 1, 1, -1]).astype(complex)
        unitary = np.kron(cz, X)  # little-endian: X on wire 0, CZ on wires 1-2
        assert unitary_is_entangling(unitary) is True
        assert unitary_is_entangling(unitary, wires=[0]) is False
        assert unitary_is_entangling(unitary, wires=[0, 1]) is True

    def test_bit_reverse_unitary_reindexes_endianness(self) -> None:
        # Big-endian CX (wire 0 = MSB control) reindexes to little-endian CX
        # with the control on bit 0; reversal is an involution.
        big_endian_cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        little = bit_reverse_unitary(big_endian_cx)
        expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
        assert np.allclose(little, expected)
        assert np.allclose(bit_reverse_unitary(little), big_endian_cx)
        assert bit_reverse_unitary(None) is None
