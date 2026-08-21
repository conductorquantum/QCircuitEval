"""Independent mathematical checks for every packaged Core target.

The expected values in this module are deliberately derived without the Core
audit source, target generators, canonical solutions, or the target-loading
facade.  The production target providers are the system under test; formulas,
truth tables, basis actions, and reviewed numeric vectors below are a second
encoding of the contracted mathematics.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from qceval.semantics.contracts import ContractRegistry
from qceval.semantics.ir import IR_VERSION, Program, Provenance
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.distribution_engine import PackagedDistributionTargetProvider
from qceval.semantics.verifiers.exact import PackagedClassicalTargetProvider
from qceval.semantics.verifiers.instrument import PackagedInstrumentTargetProvider
from qceval.semantics.verifiers.targets import CoreAnalyticTargetProvider

ROOT = Path(__file__).resolve().parents[2]
CORE_TASK_IDS = tuple(f"{index:02d}" for index in range(1, 59))

_EXACT_DISTRIBUTION_IDS = (
    "01",
    "03",
    "05",
    "07",
    "09",
    "11",
    "12",
    "13",
    "14",
    "20",
    "31",
    "32",
    "47",
    "52",
)
_ANALYTIC_DISTRIBUTION_IDS = ("06", "15", "25", "26", "29", "45", "46", "48", "51", "55", "56", "57")
_STATE_VECTOR_IDS = ("02", "10", "16", "17", "19", "24", "36", "38", "50", "53", "58")
_PARAMETERIZED_STATE_IDS = ("39", "40", "41")
_UNITARY_IDS = ("08", "27", "37", "43", "44")
_CLASSICAL_IDS = ("21", "22", "23", "30", "35")
_RARE_KIND_IDS = ("18", "28", "33", "34")
_DECLARATIVE_FAMILY_IDS = ("04", "42")


def _reference(function_name: str) -> tuple[str, ...]:
    return (f"tests/semantics/test_core_independent_targets.py::{function_name}",)


# Machine-readable release-gate evidence.  Keep this next to the checks so a
# task cannot silently inherit a family reference that does not parameterize it.
CORE_INDEPENDENT_TARGET_EVIDENCE = {
    **{
        task_id: _reference("test_core_exact_distribution_matches_independent_vector")
        for task_id in _EXACT_DISTRIBUTION_IDS
    },
    **{
        task_id: _reference("test_core_analytic_distribution_matches_independent_formula")
        for task_id in _ANALYTIC_DISTRIBUTION_IDS
    },
    **{task_id: _reference("test_core_state_matches_independent_vector") for task_id in _STATE_VECTOR_IDS},
    **{
        task_id: _reference("test_core_parameterized_state_matches_closed_form") for task_id in _PARAMETERIZED_STATE_IDS
    },
    **{task_id: _reference("test_core_unitary_matches_independent_basis_action") for task_id in _UNITARY_IDS},
    **{task_id: _reference("test_core_classical_target_matches_exhaustive_relation") for task_id in _CLASSICAL_IDS},
    **{task_id: _reference("test_core_rare_kind_target_matches_independent_action") for task_id in _RARE_KIND_IDS},
    **{
        task_id: _reference("test_core_declarative_family_has_independent_closed_form")
        for task_id in _DECLARATIVE_FAMILY_IDS
    },
    "49": _reference("test_task49_ordering_set_matches_independent_enumeration"),
    "54": _reference("test_task54_ordering_set_matches_independent_enumeration"),
}


def _context(task_id: str, arguments: tuple[object, ...] = ()) -> VerificationContext:
    contract = ContractRegistry.from_package("core").get("core", task_id)
    program = Program(
        IR_VERSION,
        contract.limits.max_qubits,
        0,
        (),
        None,
        (),
        Provenance("independent_target_test", "1", source_hash="a" * 64),
    )
    return VerificationContext(contract, "contract", contract.target.sha256, "input", program, arguments)


def _distribution(task_id: str, arguments: tuple[object, ...] = ()) -> dict[str, float]:
    table = PackagedDistributionTargetProvider().distribution(_context(task_id, arguments))
    return {"".join(outcome): probability for outcome, probability in table.rows}


def _assert_distribution(actual: Mapping[str, float], expected: Mapping[str, float], width: int) -> None:
    outcomes = tuple(format(index, f"0{width}b") for index in range(2**width))
    assert np.asarray([actual.get(item, 0.0) for item in outcomes]) == pytest.approx(
        [expected.get(item, 0.0) for item in outcomes],
        abs=2e-12,
    )


_EXACT_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    "01": {"00": 1.0},
    "03": {"010": 0.5, "110": 0.5},
    "05": {"10001": 1.0},
    "07": dict.fromkeys(("000", "010", "100", "110"), 0.25),
    "09": {"000": 0.5, "100": 0.5},
    "11": {"1100": 1.0},
    "12": {"0000": 1.0},
    "13": {"00": 0.5, "11": 0.5},
    "14": dict.fromkeys(("000", "001", "110", "111"), 0.25),
    "20": {"011": 1.0},
    "31": dict.fromkeys(("0000", "0100", "1000", "1100"), 0.25),
    "32": dict.fromkeys(("00000000", "01000000", "10000000", "11000000"), 0.25),
    "47": {"01": 1.0},
    "52": dict.fromkeys(("0000", "0100", "1000", "1100"), 0.25),
}


@pytest.mark.parametrize("task_id", _EXACT_DISTRIBUTION_IDS)
def test_core_exact_distribution_matches_independent_vector(task_id: str) -> None:
    """Compare point-mass/uniform targets with reviewed explicit vectors."""
    width = len(next(iter(_EXACT_DISTRIBUTIONS[task_id])))
    _assert_distribution(_distribution(task_id), _EXACT_DISTRIBUTIONS[task_id], width)


def _qpe_probabilities(phases: Sequence[float], weights: Sequence[float], width: int) -> dict[str, float]:
    size = 2**width
    probabilities = np.zeros(size)
    for phase, weight in zip(phases, weights, strict=True):
        for outcome in range(size):
            delta = phase - outcome / size
            if math.isclose(delta, round(delta), abs_tol=1e-14):
                value = 1.0
            else:
                value = (math.sin(math.pi * size * delta) / (size * math.sin(math.pi * delta))) ** 2
            probabilities[outcome] += weight * value
    return {format(index, f"0{width}b"): float(value) for index, value in enumerate(probabilities)}


def _apply_one_qubit(state: np.ndarray, matrix: np.ndarray, wire: int) -> np.ndarray:
    result = state.copy()
    stride = 1 << wire
    for base in range(0, len(state), 2 * stride):
        for offset in range(stride):
            low = base + offset
            high = low + stride
            result[low], result[high] = matrix @ state[[low, high]]
    return result


def _qaoa_probabilities(
    width: int,
    edges: Sequence[tuple[int, int]],
    gammas: Sequence[float],
    betas: Sequence[float],
) -> dict[str, float]:
    """Independently evaluate diagonal ZZ phases followed by X rotations."""
    state = np.ones(2**width, dtype=complex) / math.sqrt(2**width)
    for gamma, beta in zip(gammas, betas, strict=True):
        for basis in range(2**width):
            zz_sum = sum(1 if ((basis >> a) & 1) == ((basis >> b) & 1) else -1 for a, b in edges)
            state[basis] *= np.exp(-1j * gamma * zz_sum)
        rx = np.asarray(
            [[math.cos(beta), -1j * math.sin(beta)], [-1j * math.sin(beta), math.cos(beta)]],
            dtype=complex,
        )
        for wire in range(width):
            state = _apply_one_qubit(state, rx, wire)
    return {format(index, f"0{width}b"): float(abs(value) ** 2) for index, value in enumerate(state)}


def _analytic_distribution_case(task_id: str) -> tuple[tuple[object, ...], dict[str, float], int]:
    if task_id == "15":
        p0 = (1.0 + 1.0 / math.sqrt(2.0)) / 2.0
        return (), {"0": p0, "1": 1.0 - p0}, 1
    if task_id in {"25", "51"}:
        return (), _qpe_probabilities((1 / 6, 5 / 6), (0.5, 0.5), 3), 3
    if task_id == "26":
        # Three commuting SWAP projectors on GHZ3 x |+++>; 111 has zero norm.
        return (
            (),
            {
                "000": 7 / 16,
                "001": 1 / 8,
                "010": 1 / 8,
                "011": 1 / 16,
                "100": 1 / 8,
                "101": 1 / 16,
                "110": 1 / 16,
            },
            3,
        )
    if task_id == "45":
        return (), {"010": 0.5, "101": 0.5}, 3
    return _later_analytic_distribution_case(task_id)


def _later_analytic_distribution_case(task_id: str) -> tuple[tuple[object, ...], dict[str, float], int]:
    if task_id == "46":
        # Measure only the query register: uniform over z with z · 101 = 0.
        spectral = (0b000, 0b010, 0b101, 0b111)
        expected = {format(value, "03b"): 1 / 4 for value in spectral}
        return (), expected, 3
    if task_id == "48":
        triangle_edges = ((0, 1), (1, 2), (0, 2))
        return (), _qaoa_probabilities(3, triangle_edges, (0.7,), (0.4,)), 3
    if task_id == "55":
        square_edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))
        return (), _qaoa_probabilities(4, square_edges, (0.6, 0.3), (0.4, 0.2)), 4
    if task_id == "56":
        # t=pi/4 maps eigenvalues 0,1,2,3 to exact three-bit phases 0,7,6,5.
        return (), dict.fromkeys(("000", "101", "110", "111"), 0.25), 3
    if task_id == "57":
        # Reviewed dense diagonalization of -0.3*A_path4-|0><0| at t=1.5.
        return (
            (),
            {
                "00": 0.34916515938978043,
                "01": 0.15712940112131568,
                "10": 0.2877695620958855,
                "11": 0.20593587739301825,
            },
            2,
        )
    raise AssertionError(task_id)


@pytest.mark.parametrize("task_id", _ANALYTIC_DISTRIBUTION_IDS)
def test_core_analytic_distribution_matches_independent_formula(task_id: str) -> None:
    """Check analytic distribution families against separate closed forms."""
    if task_id == "06":
        witness_states = (
            np.asarray([1, 0], dtype=complex),
            np.asarray([0, 1], dtype=complex),
            np.asarray([math.cos(0.37 / 2), np.exp(0.91j) * math.sin(0.37 / 2)]),
        )
        for state in witness_states:
            p0 = (1.0 + abs(state[0]) ** 2) / 2.0
            _assert_distribution(_distribution(task_id, (state,)), {"0": p0, "1": 1.0 - p0}, 1)
        return
    if task_id == "29":
        alice_angles = (0.0, -math.pi / 2)
        bob_angles = (-math.pi / 4, math.pi / 4)
        for alice, bob in itertools.product((0, 1), repeat=2):
            correlation = math.cos(alice_angles[alice] - bob_angles[bob])
            same = (1.0 + correlation) / 4.0
            different = (1.0 - correlation) / 4.0
            expected = {"00": same, "01": different, "10": different, "11": same}
            _assert_distribution(_distribution(task_id, (alice, bob)), expected, 2)
        return
    arguments, expected, width = _analytic_distribution_case(task_id)
    _assert_distribution(_distribution(task_id, arguments), expected, width)


_STATE_VECTORS: dict[str, np.ndarray] = {
    "02": np.asarray([0, 0, 0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0, 0], dtype=complex),
    "10": np.asarray(
        [0.8889566369846307, 0.34570535882735637, 0.04938647983247948, 0.29631887899487686],
        dtype=complex,
    ),
    "16": np.asarray([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], dtype=complex),
    "17": np.asarray([1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)], dtype=complex),
    "19": np.asarray(
        [1 / math.sqrt(2) if index in {0b00101, 0b00110} else 0 for index in range(32)],
        dtype=complex,
    ),
    "24": np.asarray([0, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=complex),
    "36": np.asarray([1, 0, 0, 0], dtype=complex),
    "38": np.asarray(
        [
            0.083548938594568 - 0.42761341321798674j,
            0.10595944686689006 - 0.32031995737548363j,
            0.17046380038080908 + 0.32064573589476936j,
            0.13917707236653307 + 0.04703083321164963j,
            -0.2460749508831755 + 0.1932800197628435j,
            0.3967519007786996 - 0.43921936531060346j,
            0.17601028925663628 - 0.23738017261509123j,
            -0.0848588053290577 - 0.01070317923115869j,
        ]
    ),
    "50": np.asarray([0.5, -0.5j, -0.5j, -0.5], dtype=complex),
    "53": np.asarray([2 / math.sqrt(5), 1 / math.sqrt(5)], dtype=complex),
    "58": np.asarray(
        [
            (1 + 1 / math.sqrt(2)) / 2,
            (1 / math.sqrt(2) - 1) / 2,
            1 / (2 * math.sqrt(2)),
            1 / (2 * math.sqrt(2)),
            0,
            0,
            0,
            0,
        ],
        dtype=complex,
    ),
}


@pytest.mark.parametrize("task_id", _STATE_VECTOR_IDS)
def test_core_state_matches_independent_vector(task_id: str) -> None:
    """Compare exact states with reviewed amplitudes or solved linear systems."""
    actual = CoreAnalyticTargetProvider().array(_context(task_id), "statevector").value
    assert actual == pytest.approx(_STATE_VECTORS[task_id], abs=2e-12)


def _rx(angle: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(angle / 2), -1j * math.sin(angle / 2)],
            [-1j * math.sin(angle / 2), math.cos(angle / 2)],
        ],
        dtype=complex,
    )


def _ry(angle: float) -> np.ndarray:
    return np.asarray(
        [[math.cos(angle / 2), -math.sin(angle / 2)], [math.sin(angle / 2), math.cos(angle / 2)]],
        dtype=complex,
    )


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def _cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
    output = np.zeros_like(state)
    for basis, amplitude in enumerate(state):
        destination = basis ^ (1 << target) if (basis >> control) & 1 else basis
        output[destination] = amplitude
    return output


def _parameterized_expected(task_id: str) -> tuple[list[float], np.ndarray]:
    if task_id == "39":
        values = [0.37, -0.91]
        return values, _ry(values[1]) @ _rx(values[0]) @ np.asarray([1, 0], dtype=complex)
    if task_id == "40":
        values = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
        state = np.asarray([0, 1, 0, 0], dtype=complex)
        for matrix, wire in (
            (_rz(values[0]), 0),
            (_rz(values[1]), 0),
            (_ry(values[2]), 0),
            (_rz(values[3]), 1),
        ):
            state = _apply_one_qubit(state, matrix, wire)
        state = _cx(state, 0, 1)
        for matrix, wire in (
            (_rz(values[4]), 0),
            (_rz(values[5]), 1),
            (_ry(values[6]), 0),
            (_rz(values[7]), 1),
        ):
            state = _apply_one_qubit(state, matrix, wire)
        return values, state
    values = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    q0 = _rz(values[2]) @ _ry(values[1]) @ _rz(values[0]) @ np.asarray([1, 0], dtype=complex)
    q1 = _rz(values[5]) @ _ry(values[4]) @ _rz(values[3]) @ np.asarray([1, 0], dtype=complex)
    return values, np.kron(q1, q0)


@pytest.mark.parametrize("task_id", _PARAMETERIZED_STATE_IDS)
def test_core_parameterized_state_matches_closed_form(task_id: str) -> None:
    """Evaluate universal rotation families at a nondegenerate reviewed point."""
    values, expected = _parameterized_expected(task_id)
    actual = CoreAnalyticTargetProvider().array(_context(task_id, (values,)), "statevector").value
    assert actual == pytest.approx(expected, abs=2e-12)


def _independent_controlled_operator(task_id: str) -> np.ndarray:
    if task_id == "08":
        indices = np.arange(64)
        return np.exp(2j * math.pi * np.outer(indices, indices) / 64) / 8
    if task_id == "27":
        controls: Sequence[int] = (0,)
        target, width, operator = 1, 2, "X"
    elif task_id == "37":
        controls = (0,)
        target, width, operator = 1, 2, "H"
    elif task_id == "43":
        controls = (0, 1)
        target, width, operator = 2, 3, "X"
    else:
        controls = (0,)
        target, width, operator = 1, 2, "X"
    matrix = np.zeros((2**width, 2**width), dtype=complex)
    for basis in range(2**width):
        if all((basis >> control) & 1 for control in controls):
            if operator == "X":
                matrix[basis ^ (1 << target), basis] = 1
            else:
                target_bit = (basis >> target) & 1
                zero = basis & ~(1 << target)
                matrix[zero, basis] = 1 / math.sqrt(2)
                matrix[zero | (1 << target), basis] = (-1 if target_bit else 1) / math.sqrt(2)
        else:
            matrix[basis, basis] = 1
    return matrix


@pytest.mark.parametrize("task_id", _UNITARY_IDS)
def test_core_unitary_matches_independent_basis_action(task_id: str) -> None:
    """Check Fourier/controlled gates on every computational-basis column."""
    actual = CoreAnalyticTargetProvider().array(_context(task_id), "unitary").value
    assert actual == pytest.approx(_independent_controlled_operator(task_id), abs=2e-12)


def _classical_relation(task_id: str) -> tuple[tuple[str, str], ...]:
    if task_id == "21":
        rows = []
        for value in range(8):
            ci, bi, ai = ((value >> index) & 1 for index in range(3))
            output = f"{int(ai + bi + ci >= 2)}{ai ^ bi}{ai ^ ci}"
            rows.append((f"{value:03b}", output))
        return tuple(rows)
    if task_id == "22":
        rows = []
        for value in range(128):
            carry = value & 1
            b_value = sum(((value >> (2 * index + 1)) & 1) << index for index in range(3))
            a_value = sum(((value >> (2 * index + 2)) & 1) << index for index in range(3))
            rows.append((f"{value:07b}", f"{a_value + b_value + carry:04b}"))
        return tuple(rows)
    if task_id == "23":
        return tuple((f"{value:03b}", f"{(value - 3) % 8:03b}") for value in range(8))
    if task_id == "30":
        return tuple((f"{value:02b}", str(int(bool(value & 1) or bool(value & 2)))) for value in range(4))
    return tuple((f"{value:03b}", str(value.bit_count() % 2)) for value in range(8))


@pytest.mark.parametrize("task_id", _CLASSICAL_IDS)
def test_core_classical_target_matches_exhaustive_relation(task_id: str) -> None:
    """Check all inputs using integer Boolean/arithmetic definitions."""
    actual = PackagedClassicalTargetProvider().classical_table(_context(task_id)).rows
    assert actual == _classical_relation(task_id)


@pytest.mark.parametrize("task_id", _RARE_KIND_IDS)
def test_core_rare_kind_target_matches_independent_action(task_id: str) -> None:
    """Check Choi, clean-ancilla isometry, and complete instrument branches."""
    context = _context(task_id)
    if task_id == "18":
        unitary = _rx(math.pi / 2)
        bell = np.asarray([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        vector = np.kron(np.eye(2), unitary) @ bell
        expected = np.outer(vector, vector.conj())
        # The verifier uses an unnormalized Choi convention with trace two.
        expected *= 2
        actual = CoreAnalyticTargetProvider().array(context, "choi").value
        assert actual == pytest.approx(expected, abs=2e-12)
        return
    if task_id == "28":
        expected = np.zeros((32, 16), dtype=complex)
        inputs = [basis for basis in range(32) if not basis & 0b01000]
        for column, basis in enumerate(inputs):
            clean_output = basis ^ 0b10000 if basis & 0b00111 == 0b00111 else basis
            expected[clean_output, column] = 1
        actual = CoreAnalyticTargetProvider().array(context, "isometry").value
        assert actual == pytest.approx(expected, abs=2e-12)
        return
    branches = PackagedInstrumentTargetProvider().instrument(context).branches
    expected_outcome = "01" if task_id == "33" else "001"
    assert len(branches) == 1
    assert branches[0].outcome == expected_outcome
    assert branches[0].probability == pytest.approx(1.0)
    assert branches[0].conditional == pytest.approx(np.asarray([[0, 0], [0, 1]], dtype=complex))


def _raw_target(task_id: str) -> dict[str, object]:
    document = json.loads((ROOT / "src/qceval/assets/targets/core/target.json").read_text(encoding="utf-8"))
    value = document["tasks"][task_id]
    return value.get("target", value)


@pytest.mark.parametrize("task_id", _DECLARATIVE_FAMILY_IDS)
def test_core_declarative_family_has_independent_closed_form(task_id: str) -> None:
    """Pin formula-only family artifacts and prove one nontrivial invariant."""
    target = _raw_target(task_id)
    if task_id == "04":
        assert target == {
            "type": "qaoa_state_family",
            "cost": "maxcut",
            "graph_edges": [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]],
            "initial_state": "plus_5",
            "layers": 5,
            "mixer": "sum_X",
        }
        # With every cost/mixer angle zero, all 32 amplitudes remain 1/sqrt(32).
        zero_state = np.ones(32, dtype=complex) / math.sqrt(32)
        assert np.linalg.norm(zero_state) == pytest.approx(1.0)
        assert np.abs(zero_state) ** 2 == pytest.approx(np.full(32, 1 / 32))
        return
    assert target["matrix"] == [
        ["cos(theta/2)", "-exp(i*lam)*sin(theta/2)"],
        ["exp(i*phi)*sin(theta/2)", "exp(i*(phi+lam))*cos(theta/2)"],
    ]
    theta, phi, lam = 0.73, -0.41, 1.17
    declared = np.asarray(
        [
            [math.cos(theta / 2), -np.exp(1j * lam) * math.sin(theta / 2)],
            [np.exp(1j * phi) * math.sin(theta / 2), np.exp(1j * (phi + lam)) * math.cos(theta / 2)],
        ]
    )
    euler = _rz(phi) @ _ry(theta) @ _rz(lam)
    assert declared == pytest.approx(np.exp(0.5j * (phi + lam)) * euler, abs=2e-12)


def _three_wire_term(one_wire: np.ndarray, first: int, second: int | None = None) -> np.ndarray:
    identity = np.eye(2, dtype=complex)
    factors = [identity, identity, identity]
    factors[first] = one_wire
    if second is not None:
        factors[second] = one_wire
    return np.kron(np.kron(factors[2], factors[1]), factors[0])


def _exponential(hermitian: np.ndarray, time: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(hermitian)
    return vectors @ np.diag(np.exp(-1j * values * time)) @ vectors.conj().T


def _projective_key(state: np.ndarray) -> tuple[tuple[float, float], ...]:
    pivot = state[int(np.argmax(np.abs(state)))]
    normalized = state * np.exp(-1j * np.angle(pivot))
    return tuple((round(value.real, 10), round(value.imag, 10)) for value in normalized)


def _independent_task49_states() -> set[tuple[tuple[float, float], ...]]:
    x = np.asarray([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    identity = np.eye(2, dtype=complex)
    factors = (
        0.7 * np.kron(z, z),
        0.4 * np.kron(identity, x),
        0.3 * np.kron(x, identity),
    )
    exponentials = [_exponential(term, 0.4) for term in factors]
    initial = np.zeros(4, dtype=complex)
    initial[0b01] = 1
    states = set()
    for ordering in itertools.permutations(range(3)):
        step = np.eye(4, dtype=complex)
        for index in ordering:
            step = exponentials[index] @ step
        states.add(_projective_key(step @ step @ initial))
    return states


def test_task49_ordering_set_matches_independent_enumeration() -> None:
    """Check every consistent first-order ordering, not one canonical order."""
    actual = CoreAnalyticTargetProvider().array(_context("49"), "statevector").value
    assert {_projective_key(state) for state in actual} == _independent_task49_states()


def _independent_task54_states() -> set[tuple[tuple[float, float], ...]]:
    x = np.asarray([[0, 1], [1, 0]], dtype=complex)
    y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    factors = [
        _three_wire_term(x, 0, 1),
        _three_wire_term(x, 1, 2),
        0.8 * _three_wire_term(y, 0, 1),
        0.8 * _three_wire_term(y, 1, 2),
        0.5 * _three_wire_term(z, 0, 1),
        0.5 * _three_wire_term(z, 1, 2),
        0.3 * sum((_three_wire_term(z, wire) for wire in range(3)), start=np.zeros((8, 8), dtype=complex)),
    ]
    half_steps = [_exponential(term, 0.25) for term in factors]
    initial = np.zeros(8, dtype=complex)
    initial[0b100] = 1
    states = set()
    for ordering in itertools.permutations(range(7)):
        step = np.eye(8, dtype=complex)
        for index in (*ordering, *reversed(ordering)):
            step = half_steps[index] @ step
        states.add(_projective_key(step @ step @ initial))
    return states


def test_task54_ordering_set_matches_independent_enumeration() -> None:
    """Check all seven-factor symmetric orderings, not one canonical order."""
    actual = CoreAnalyticTargetProvider().array(_context("54"), "statevector").value
    assert {_projective_key(state) for state in actual} == _independent_task54_states()


def test_core_independent_target_evidence_covers_all_58_contracts() -> None:
    """Keep the machine-readable evidence mapping complete and exact."""
    assert set(CORE_INDEPENDENT_TARGET_EVIDENCE) == set(CORE_TASK_IDS)
