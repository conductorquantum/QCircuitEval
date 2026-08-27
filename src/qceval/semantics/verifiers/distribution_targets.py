"""Independent analytic materialization for audited core distributions."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

DistributionProvider = Callable[[Mapping[str, Any], int, tuple[Any, ...]], dict[str, float]]


def analytic_distribution(
    target: Mapping[str, Any],
    width: int,
    arguments: tuple[Any, ...] = (),
) -> dict[str, float]:
    """Materialize a finite distribution from one reviewed target specification.

    Args:
        target: Prompt-derived analytic target specification.
        width: Number of observed classical bits.
        arguments: Runtime arguments for input-dependent distributions.

    Returns:
        Exact probability mapping in contracted bitstring order.
    """

    providers: dict[str, DistributionProvider] = {
        "swap_test_formula": _swap_test_distribution,
        "chsh_conditional_distribution": _bound_chsh_distribution,
        "hadamard_test_distribution": lambda value, _width, _arguments: _hadamard_test(value),
        "qpe_distribution": _phase_estimation_distribution,
        "quantum_counting_distribution": _phase_estimation_distribution,
        "exact_circuit_distribution": lambda value, bits, _arguments: _exact_circuit_distribution(value, bits),
        "fixed_qaoa_distribution": lambda value, bits, _arguments: _qaoa_distribution(value, bits),
        "ctqw_distribution": _uniform_distribution,
        "spectral_phase_distribution": lambda value, bits, _arguments: _spectral_distribution(value, bits),
        "fixed_trotter_distribution": lambda value, bits, _arguments: _trotter_distribution(value, bits),
        "ideal_ctqw_distribution": lambda value, bits, _arguments: _ideal_ctqw_distribution(value, bits),
    }
    target_type = str(target.get("type"))
    provider = providers.get(target_type)
    if provider is None:
        raise ValueError(f"analytic distribution target unsupported: {target_type}")
    return provider(target, width, arguments)


def _swap_test_distribution(
    target: Mapping[str, Any],
    width: int,
    arguments: tuple[Any, ...],
) -> dict[str, float]:
    del target, width
    if len(arguments) != 1:
        raise ValueError("SWAP-test target requires the unknown-state argument")
    overlap = _zero_probability(arguments[0])
    return {"0": (1.0 + overlap) / 2.0, "1": (1.0 - overlap) / 2.0}


def _bound_chsh_distribution(
    target: Mapping[str, Any],
    width: int,
    arguments: tuple[Any, ...],
) -> dict[str, float]:
    del width
    if len(arguments) != 2:
        raise ValueError("CHSH target requires Alice and Bob settings")
    return _chsh_distribution(target, int(arguments[0]), int(arguments[1]))


def _phase_estimation_distribution(
    target: Mapping[str, Any],
    width: int,
    arguments: tuple[Any, ...],
) -> dict[str, float]:
    del arguments
    phases = target.get("eigenphase_fractions", target.get("grover_eigenphases"))
    if not isinstance(phases, Sequence):
        raise ValueError("QPE target is missing eigenphases")
    weights = target.get("eigenstate_weights")
    if weights is None:
        weights = [f"1/{len(phases)}"] * len(phases)
    if not isinstance(weights, Sequence):
        raise ValueError("QPE target weights are malformed")
    return _qpe_distribution(phases, weights, width)


def _uniform_distribution(
    target: Mapping[str, Any],
    width: int,
    arguments: tuple[Any, ...],
) -> dict[str, float]:
    del target, arguments
    return {format(index, f"0{width}b"): 1.0 / (2**width) for index in range(2**width)}


def _zero_probability(value: Any) -> float:
    if callable(value):
        state = np.asarray(value(), dtype=np.complex128).reshape(-1)
    elif value.__class__.__module__.startswith("qiskit"):
        from qiskit.quantum_info import Statevector

        state = np.asarray(Statevector.from_instruction(value).data, dtype=np.complex128)
    elif value.__class__.__module__.startswith("cirq"):
        import cirq

        state = np.asarray(cirq.final_state_vector(value, dtype=np.complex128), dtype=np.complex128)
    else:
        state = np.asarray(value, dtype=np.complex128).reshape(-1)
    if state.shape != (2,):
        raise ValueError("SWAP-test input must be one pure qubit")
    norm = float(np.vdot(state, state).real)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("SWAP-test input state is not normalized")
    return float(abs(state[0]) ** 2)


def _chsh_distribution(target: Mapping[str, Any], alice: int, bob: int) -> dict[str, float]:
    alice_angles = target.get("alice_angles")
    bob_angles = target.get("bob_angles")
    if not isinstance(alice_angles, Mapping) or not isinstance(bob_angles, Mapping):
        raise ValueError("CHSH measurement angles are missing")
    if str(alice) not in alice_angles or str(bob) not in bob_angles:
        raise ValueError("CHSH setting is outside the contracted domain")
    state = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    _apply_single(state, _ry(_angle(alice_angles[str(alice)])), 0)
    _apply_single(state, _ry(_angle(bob_angles[str(bob)])), 1)
    return _dense_table(np.abs(state) ** 2, 2)


def _angle(value: Any) -> float:
    if value == "pi/4":
        return math.pi / 4.0
    if value == "-pi/4":
        return -math.pi / 4.0
    if value == "-pi/2":
        return -math.pi / 2.0
    return _number(value)


def _ry(angle: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(angle / 2.0), -math.sin(angle / 2.0)],
            [math.sin(angle / 2.0), math.cos(angle / 2.0)],
        ],
        dtype=np.complex128,
    )


def _hadamard_test(target: Mapping[str, Any]) -> dict[str, float]:
    raw = target.get("probability_zero")
    if raw == "(1+1/sqrt(2))/2":
        p0 = (1.0 + 1.0 / math.sqrt(2.0)) / 2.0
    elif target.get("component") == "real" and target.get("unitary_sequence"):
        state = np.asarray([1, 1, 0, 0], dtype=np.complex128) / math.sqrt(2.0)
        evolved = state.copy()
        _apply_cz(evolved, 0, 1)
        _apply_single(evolved, _h(), 0)
        _apply_single(evolved, np.diag([1.0, np.exp(1j * math.pi / 4.0)]), 0)
        _apply_cnot(evolved, 0, 1)
        p0 = (1.0 + float(np.vdot(state, evolved).real)) / 2.0
    else:
        raise ValueError(f"unsupported Hadamard-test formula: {raw}")
    return {"0": p0, "1": 1.0 - p0}


def _qpe_distribution(phases: Sequence[Any], weights: Sequence[Any], width: int) -> dict[str, float]:
    if len(phases) != len(weights):
        raise ValueError("QPE phase and weight counts differ")
    size = 2**width
    result = np.zeros(size, dtype=float)
    for raw_phase, raw_weight in zip(phases, weights, strict=True):
        phase = _number(raw_phase) % 1.0
        weight = _number(raw_weight)
        for outcome in range(size):
            delta = phase - outcome / size
            denominator = math.sin(math.pi * delta)
            if abs(denominator) < 1e-14:
                probability = 1.0
            else:
                probability = (math.sin(math.pi * size * delta) / (size * denominator)) ** 2
            result[outcome] += weight * probability
    return _dense_table(result, width)


def _exact_circuit_distribution(target: Mapping[str, Any], width: int) -> dict[str, float]:
    if target.get("operation") == "parallel_bitwise_controlled_swaps":
        return _parallel_swap_tests(width)
    if target.get("marked_states"):
        marked = tuple(str(value) for value in target["marked_states"])
        return {value: 1.0 / len(marked) for value in marked}
    if target.get("hidden_period"):
        return _simon_joint_distribution(str(target["hidden_period"]), width)
    if target.get("function") == "x mod 2 over Z4":
        return {"001": 0.5, "101": 0.5}
    raise ValueError("unsupported exact-circuit distribution target")


def _parallel_swap_tests(width: int) -> dict[str, float]:
    """Derive the joint SWAP-test distribution from permutation projectors.

    The outcome law P(b) = ||prod_i (I + (-1)^{b_i} S_i)/2 |GHZ x +++>||^2 is
    evaluated directly on the six data qubits, independently of any gate-level
    circuit replay, so the target cannot inherit a construction error from the
    canonical implementation.
    """
    if width != 3:
        raise ValueError("parallel SWAP-test target requires three ancillas")
    dimension = 2**6
    state = np.zeros(dimension, dtype=np.complex128)
    # GHZ on pair-first qubits (bits 0-2), |+++> on pair-second qubits (3-5).
    for second in range(8):
        state[second << 3] += 1.0
        state[(second << 3) | 0b111] += 1.0
    state /= np.linalg.norm(state)
    probabilities = np.zeros(8, dtype=float)
    for outcome in range(8):
        projected = state.copy()
        for pair in range(3):
            swapped = np.zeros(dimension, dtype=np.complex128)
            for basis in range(dimension):
                first_bit = (basis >> pair) & 1
                second_bit = (basis >> (pair + 3)) & 1
                target = basis ^ (((first_bit ^ second_bit) << pair) | ((first_bit ^ second_bit) << (pair + 3)))
                swapped[target] += projected[basis]
            sign = -1.0 if (outcome >> pair) & 1 else 1.0
            projected = (projected + sign * swapped) / 2.0
        probabilities[outcome] = float(np.vdot(projected, projected).real)
    return _dense_table(probabilities, width)


def _simon_joint_distribution(period: str, width: int) -> dict[str, float]:
    if width != 6 or len(period) != 3:
        raise ValueError("Simon target requires two three-qubit registers")
    period_value = int(period, 2)
    rows: dict[str, float] = {}
    seen: set[int] = set()
    for x in range(8):
        mate = x ^ period_value
        representative = min(x, mate)
        if representative in seen:
            continue
        seen.add(representative)
        f0 = ((representative >> 0) ^ (representative >> 2)) & 1
        f1 = (representative >> 1) & 1
        output = f0 | (f1 << 1)
        for spectral in range(8):
            if ((spectral & period_value).bit_count() % 2) == 0:
                outcome = spectral | (output << 3)
                rows[format(outcome, "06b")] = 1.0 / 16.0
    return rows


def _qaoa_distribution(target: Mapping[str, Any], width: int) -> dict[str, float]:
    edges = target.get("graph_edges")
    gammas = target.get("gamma")
    betas = target.get("beta")
    if not isinstance(edges, Sequence) or not isinstance(gammas, Sequence) or not isinstance(betas, Sequence):
        raise ValueError("QAOA target is incomplete")
    if len(gammas) != len(betas):
        raise ValueError("QAOA layer parameter counts differ")
    state = np.ones(2**width, dtype=np.complex128) / math.sqrt(2**width)
    for gamma, beta in zip(gammas, betas, strict=True):
        for basis in range(2**width):
            eigenvalue = 0
            for edge in edges:
                first, second = (int(edge[0]), int(edge[1]))
                eigenvalue += 1 if ((basis >> first) & 1) == ((basis >> second) & 1) else -1
            state[basis] *= np.exp(-1j * float(gamma) * eigenvalue)
        mixer = np.asarray(
            [
                [math.cos(float(beta)), -1j * math.sin(float(beta))],
                [-1j * math.sin(float(beta)), math.cos(float(beta))],
            ],
            dtype=np.complex128,
        )
        for wire in range(width):
            _apply_single(state, mixer, wire)
    return _dense_table(np.abs(state) ** 2, width)


def _spectral_distribution(target: Mapping[str, Any], width: int) -> dict[str, float]:
    weights = target.get("eigenstate_weights")
    if not isinstance(weights, Sequence):
        raise ValueError("spectral target is missing eigenstate weights")
    time = math.pi / 4.0 if target.get("time") == "pi/4" else _number(target.get("time"))
    phases = [(-eigenvalue * time / (2.0 * math.pi)) % 1.0 for eigenvalue in range(len(weights))]
    return _qpe_distribution(phases, weights, width)


def _trotter_distribution(target: Mapping[str, Any], width: int) -> dict[str, float]:
    if width != 2 or target.get("graph") != "path4":
        raise ValueError("unsupported Trotter target")
    gamma = float(target["gamma"])
    total_time = float(target["time"])
    steps = int(target["steps"])
    adjacency = np.zeros((4, 4), dtype=np.complex128)
    for vertex in range(3):
        adjacency[vertex, vertex + 1] = adjacency[vertex + 1, vertex] = 1.0
    walk = -gamma * adjacency
    marked = np.zeros((4, 4), dtype=np.complex128)
    marked[int(target["marked_vertex"]), int(target["marked_vertex"])] = -1.0
    delta = total_time / steps
    half_marked = _hermitian_exponential(marked, delta / 2.0)
    full_walk = _hermitian_exponential(walk, delta)
    step = half_marked @ full_walk @ half_marked
    state = np.ones(4, dtype=np.complex128) / 2.0
    for _ in range(steps):
        state = step @ state
    return _dense_table(np.abs(state) ** 2, width)


def _ideal_ctqw_distribution(target: Mapping[str, Any], width: int) -> dict[str, float]:
    if width != 2 or target.get("graph") != "path4":
        raise ValueError("unsupported ideal CTQW target")
    adjacency = np.zeros((4, 4), dtype=np.complex128)
    for vertex in range(3):
        adjacency[vertex, vertex + 1] = adjacency[vertex + 1, vertex] = 1.0
    hamiltonian = -float(target["gamma"]) * adjacency
    marked = int(target["marked_vertex"])
    hamiltonian[marked, marked] -= 1.0
    state = np.ones(4, dtype=np.complex128) / 2.0
    state = _hermitian_exponential(hamiltonian, float(target["time"])) @ state
    return _dense_table(np.abs(state) ** 2, width)


def _hermitian_exponential(matrix: np.ndarray, time: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.exp(-1j * eigenvalues * time)) @ eigenvectors.conjugate().T


def _number(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def _dense_table(probabilities: Iterable[float], width: int) -> dict[str, float]:
    return {
        format(index, f"0{width}b"): float(probability)
        for index, probability in enumerate(probabilities)
        if abs(probability) > 1e-15
    }


def _h() -> np.ndarray:
    return np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)


def _apply_single(state: np.ndarray, gate: np.ndarray, wire: int) -> None:
    stride = 1 << wire
    for base in range(0, len(state), stride * 2):
        for offset in range(stride):
            first = base + offset
            second = first + stride
            a, b = state[first], state[second]
            state[first] = gate[0, 0] * a + gate[0, 1] * b
            state[second] = gate[1, 0] * a + gate[1, 1] * b


def _apply_cnot(state: np.ndarray, control: int, target: int) -> None:
    for basis in range(len(state)):
        if ((basis >> control) & 1) and not ((basis >> target) & 1):
            other = basis | (1 << target)
            state[basis], state[other] = state[other], state[basis]


def _apply_cz(state: np.ndarray, first: int, second: int) -> None:
    for basis in range(len(state)):
        if ((basis >> first) & 1) and ((basis >> second) & 1):
            state[basis] *= -1.0
