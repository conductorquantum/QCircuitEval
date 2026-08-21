from __future__ import annotations

import numpy as np
import pytest

from qceval.evals.probabilities import (
    as_prob_array,
    bitstring_index,
    bitstrings_for_probs,
    distribution_from_support,
    hellinger_fidelity,
    hellinger_infidelity,
    num_bits,
    observed_support,
    project_and_normalize,
    support_matches,
    top_k_bitstrings,
)


def test_probability_helpers_cover_core_paths() -> None:
    # Arrange
    probs = [0.0, 2.0, 0.0, 2.0]

    # Act
    normalized = as_prob_array(probs)
    support = observed_support(normalized, 0.1)
    projected = project_and_normalize(normalized, ["01", "11"])

    # Assert
    assert normalized.tolist() == [0.0, 0.5, 0.0, 0.5]
    assert num_bits(normalized) == 2
    assert bitstrings_for_probs(normalized) == ["00", "01", "10", "11"]
    assert bitstring_index("11") == 3
    assert support == frozenset({"01", "11"})
    assert top_k_bitstrings(normalized, 2) == ["01", "11"]
    assert projected.tolist() == [0.5, 0.5]


def test_probability_helpers_raise_on_invalid_inputs() -> None:
    # Arrange
    cases = [[], [0.0, 0.0], [[1.0]], [float("nan"), 1.0], [-1.0, 2.0], [1.0, 0.0, 0.0]]

    # Act
    errors = []
    for case in cases:
        with pytest.raises(ValueError) as exc:
            as_prob_array(case)  # type: ignore[arg-type]
        errors.append(str(exc.value))

    # Assert
    assert errors == [
        "probability length must be a power of 2, got 0",
        "probabilities sum to zero",
        "probabilities must be a 1-D vector",
        "probabilities must be finite",
        "probabilities must be nonnegative",
        "probability length must be a power of 2, got 3",
    ]


def test_distribution_and_support_matching_paths() -> None:
    # Arrange
    observed = frozenset({"01", "10"})
    expected = frozenset({"10", "01"})

    # Act
    distribution = distribution_from_support(["01", "11"], 2)
    weighted = distribution_from_support(["01", "11"], 2, {"01": 3.0, "11": 1.0})
    direct_match = support_matches(observed, expected, permutation_invariant=False)
    permuted_match = support_matches(frozenset({"10"}), frozenset({"01"}), permutation_invariant=True)

    # Assert
    assert distribution.tolist() == [0.0, 0.5, 0.0, 0.5]
    assert weighted.tolist() == [0.0, 0.75, 0.0, 0.25]
    assert direct_match == (True, None)
    assert permuted_match[0] is True


def test_support_matching_rejects_non_matches() -> None:
    # Arrange
    observed = frozenset({"00"})
    expected = frozenset({"11"})

    # Act
    direct = support_matches(observed, expected, permutation_invariant=False)
    permuted = support_matches(observed, expected, permutation_invariant=True)

    # Assert
    assert direct == (False, None)
    assert permuted == (False, None)


def test_distribution_and_top_k_raise_on_invalid_inputs() -> None:
    # Act
    with pytest.raises(ValueError) as support_error:
        distribution_from_support([], 2)
    with pytest.raises(ValueError) as weights_error:
        distribution_from_support(["00"], 2, {"00": 0.0})
    with pytest.raises(ValueError) as top_error:
        top_k_bitstrings(np.array([1.0, 0.0]), 0)

    # Assert
    assert str(support_error.value) == "support cannot be empty"
    assert str(weights_error.value) == "support weights sum to zero"
    assert str(top_error.value) == "top_k must be positive"


def test_hellinger_fidelity_perfect_and_disjoint() -> None:
    # Arrange
    matching = ([0.5, 0.5], [0.5, 0.5])
    disjoint = ([1.0, 0.0], [0.0, 1.0])

    # Act
    matching_fidelity = hellinger_fidelity(*matching)
    matching_infidelity = hellinger_infidelity(*matching)
    disjoint_fidelity = hellinger_fidelity(*disjoint)
    disjoint_infidelity = hellinger_infidelity(*disjoint)

    # Assert
    assert matching_fidelity == 1.0
    assert matching_infidelity == 0.0
    assert disjoint_fidelity == 0.0
    assert disjoint_infidelity == 1.0


def test_hellinger_is_symmetric() -> None:
    # Arrange
    p = [0.2, 0.8]
    q = [0.75, 0.25]

    # Act
    forward = hellinger_fidelity(p, q)
    backward = hellinger_fidelity(q, p)

    # Assert
    assert forward == pytest.approx(backward)


def test_hellinger_shape_mismatch_raises() -> None:
    # Arrange
    probs = [1.0, 0.0]
    expected = [1.0, 0.0, 0.0, 0.0]

    # Act
    with pytest.raises(ValueError) as exc:
        hellinger_fidelity(probs, expected)

    # Assert
    assert "shape mismatch" in str(exc.value)


def test_hellinger_normalizes_unnormalized_inputs() -> None:
    # Arrange
    probs = [0.5, 0.5]
    expected = [1.0, 1.0]

    # Act
    fidelity = hellinger_fidelity(probs, expected)

    # Assert
    assert fidelity == 1.0
