from __future__ import annotations

import numpy as np
import pytest

from qceval.evals.ir import Circuit, Control, Gate, full_unitary

X = np.asarray([[0, 1], [1, 0]], dtype=complex)
TOFFOLI = np.asarray(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    dtype=complex,
)


def test_controlled_base_is_not_double_controlled() -> None:
    circuit = Circuit(2, (Gate.controlled(X, targets=(1,), controls=(0,), name="cx"),))

    unitary = full_unitary(circuit)

    assert np.allclose(unitary[:, 1], np.eye(4, dtype=complex)[:, 3])
    assert np.allclose(unitary[:, 3], np.eye(4, dtype=complex)[:, 1])


def test_open_control_is_supported() -> None:
    circuit = Circuit(2, (Gate.controlled(X, targets=(1,), controls=(Control(0, 0),), name="open-cx"),))

    unitary = full_unitary(circuit)

    assert np.allclose(unitary[:, 0], np.eye(4, dtype=complex)[:, 2])
    assert np.allclose(unitary[:, 2], np.eye(4, dtype=complex)[:, 0])
    assert np.allclose(unitary[:, 1], np.eye(4, dtype=complex)[:, 1])


def test_multi_control_gate_matches_toffoli() -> None:
    circuit = Circuit(3, (Gate.controlled(X, targets=(2,), controls=(0, 1), name="ccx"),))

    assert np.allclose(full_unitary(circuit), TOFFOLI)


def test_target_control_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="target/control overlap"):
        Gate.controlled(X, targets=(0,), controls=(0,), name="bad")


def test_non_unitary_and_non_finite_matrices_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be unitary"):
        Gate.full(np.asarray([[1, 1], [0, 1]], dtype=complex), (0,), name="non-unitary")
    with pytest.raises(ValueError, match="finite"):
        Gate.full(np.asarray([[1, 0], [0, np.nan]], dtype=complex), (0,), name="nan")


def test_duplicate_and_out_of_range_wires_are_rejected() -> None:
    with pytest.raises(ValueError, match="active wires must be unique"):
        Gate.full(np.eye(4, dtype=complex), (0, 0), name="duplicate")
    with pytest.raises(ValueError, match="outside"):
        Circuit(1, (Gate.full(X, (1,), name="outside"),))
