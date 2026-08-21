"""Analytic target provider independent of candidate materialization."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from qceval.semantics.targets import load_contract_target_document
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.materialize import ArrayMaterialization, ClassicalTableMaterialization


class CoreAnalyticTargetProvider:
    """Small analytic targets used by the first exact-engine packets."""

    def array(self, context: VerificationContext, representation: str) -> ArrayMaterialization:
        """Return one independently derived core target array.

        Args:
            context: Task contract and version context.
            representation: Requested state/operator/isometry/Choi form.

        Returns:
            Exact dense target.
        """
        key = (context.contract.task_id, representation)
        providers = {
            ("02", "statevector"): _task02_state,
            ("27", "unitary"): _task27_unitary,
            ("28", "isometry"): _task28_isometry,
            ("18", "choi"): _task18_witness_channel_choi,
        }
        provider = providers.get(key)
        value = (
            provider()
            if provider is not None
            else _array_from_target(_packaged_target(context), representation, context.arguments)
        )
        cases = value.shape[1] if value.ndim == 2 else 1
        return ArrayMaterialization(value, representation, cases)

    def classical_table(self, context: VerificationContext) -> ClassicalTableMaterialization:
        """Return one exhaustive Boolean target table from the packaged artifact.

        Args:
            context: Task contract and version context.

        Returns:
            Complete deterministic relation.
        """
        from qceval.semantics.verifiers.exact import PackagedClassicalTargetProvider

        return PackagedClassicalTargetProvider().classical_table(context)


def _task02_state() -> np.ndarray:
    state = np.zeros(8, dtype=np.complex128)
    state[0b011] = 1 / math.sqrt(2)
    state[0b100] = -1 / math.sqrt(2)
    return state


def _task27_unitary() -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    for basis in range(4):
        output = basis ^ 0b10 if basis & 0b01 else basis
        matrix[output, basis] = 1
    return matrix


def _task28_isometry() -> np.ndarray:
    inputs = [basis for basis in range(32) if not basis & (1 << 3)]
    matrix = np.zeros((32, 16), dtype=np.complex128)
    for column, basis in enumerate(inputs):
        output = basis ^ (1 << 4) if basis & 0b111 == 0b111 else basis
        matrix[output, column] = 1
    return matrix


def _task18_witness_channel_choi() -> np.ndarray:
    # The prompt requires the RX(pi/2) preparation inside the returned circuit,
    # so a faithful teleportation induces the RX(pi/2) unitary channel from
    # Alice q0 to Bob q2 rather than the bare identity.
    unitary = np.asarray([[1, -1j], [-1j, 1]], dtype=np.complex128) / math.sqrt(2)
    choi = np.zeros((4, 4), dtype=np.complex128)
    for row in range(2):
        for column in range(2):
            block = np.zeros((2, 2), dtype=np.complex128)
            block[row, column] = 1.0
            choi[2 * row : 2 * row + 2, 2 * column : 2 * column + 2] = unitary @ block @ unitary.conjugate().T
    return choi


def _packaged_target(context: VerificationContext) -> dict[str, Any]:
    value = load_contract_target_document(context.contract)
    target = value.get("target")
    if not isinstance(target, dict):
        raise ValueError("target artifact is malformed")
    return target


def _array_from_target(
    target: dict[str, Any],
    representation: str,
    arguments: tuple[Any, ...] = (),
) -> np.ndarray:
    target = _argument_state_target(target, arguments)
    target_type = target.get("type")
    builder = _TARGET_ARRAY_BUILDERS.get((representation, str(target_type)))
    if builder is None:
        raise NotImplementedError(f"no analytic {representation} target for {target_type}")
    return builder(target, arguments)


def _fixed_trotter_state(target: dict[str, Any]) -> np.ndarray:
    if target.get("step_sequence"):
        return _task49_state(target)
    raise NotImplementedError("second-order Trotter sequence remains underspecified")


def _first_order_trotter_state_set(target: dict[str, Any]) -> np.ndarray:
    """Enumerate every consistent ordering of the task's three factors."""
    from itertools import permutations

    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    factors = (
        0.7 * np.kron(z, z),
        0.4 * np.kron(identity, x),
        0.3 * np.kron(x, identity),
    )
    steps = int(target["steps"])
    delta = float(target["time"]) / steps
    exponentials = [_matrix_exponential(factor, delta) for factor in factors]
    initial = np.zeros(4, dtype=np.complex128)
    initial[1] = 1.0
    states = []
    seen: set[bytes] = set()
    for ordering in permutations(range(3)):
        step = np.eye(4, dtype=np.complex128)
        for index in ordering:
            step = exponentials[index] @ step
        state = initial
        for _ in range(steps):
            state = step @ state
        fingerprint = np.round(state * np.exp(-1j * np.angle(state[np.argmax(np.abs(state))])), 12).tobytes()
        if fingerprint not in seen:
            seen.add(fingerprint)
            states.append(state)
    return np.asarray(states, dtype=np.complex128)


def _fourier_operator(target: dict[str, Any]) -> np.ndarray:
    dimension = int(target["dimension"])
    indices = np.arange(dimension)
    return np.exp(2j * np.pi * np.outer(indices, indices) / dimension) / math.sqrt(dimension)


_TARGET_ARRAY_BUILDERS: dict[tuple[str, str], Any] = {
    ("statevector", "sparse_exact_state"): lambda target, _: _exact_state(target),
    ("statevector", "exact_state"): lambda target, _: _exact_state(target),
    ("statevector", "fixed_trotter_state"): lambda target, _: _fixed_trotter_state(target),
    ("statevector", "first_order_trotter_state_set"): lambda target, _: _first_order_trotter_state_set(target),
    ("statevector", "postselected_linear_system_state"): lambda target, _: _linear_system_state(target),
    ("statevector", "ideal_heisenberg_state"): lambda target, _: _ideal_heisenberg_state(target),
    ("statevector", "trotter_ordering_state_set"): lambda target, _: _trotter_ordering_state_set(target),
    ("statevector", "parameterized_state_family"): lambda target, arguments: _parameterized_family_state(
        target, arguments
    ),
    ("unitary", "fourier_operator"): lambda target, _: _fourier_operator(target),
    ("unitary", "controlled_operator"): lambda target, _: _controlled_operator(target),
    ("unitary", "controlled_controlled_operator"): lambda target, _: _controlled_operator(target),
}


def _argument_state_target(target: dict[str, Any], arguments: tuple[Any, ...]) -> dict[str, Any]:
    if target.get("type") != "argument_state_cases":
        return target
    cases = target.get("cases")
    if not isinstance(cases, list):
        raise ValueError("argument-state target cases are missing")
    for case in cases:
        if not isinstance(case, dict) or case.get("arguments") != list(arguments):
            continue
        state = case.get("state")
        if not isinstance(state, dict):
            raise ValueError("argument-state target case is malformed")
        return state
    raise ValueError(f"no state target for arguments {arguments!r}")


def _exact_state(target: dict[str, Any]) -> np.ndarray:
    amplitudes = target.get("amplitudes")
    if not isinstance(amplitudes, dict) or not amplitudes:
        raise ValueError("exact state amplitudes are missing")
    width = len(next(iter(amplitudes)))
    state = np.zeros(2**width, dtype=np.complex128)
    for bitstring, amplitude in amplitudes.items():
        state[int(bitstring, 2)] = _amplitude(amplitude)
    return state


def _linear_system_state(target: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(
        [[_real_scalar(value) for value in row] for row in target["matrix"]],
        dtype=np.complex128,
    )
    rhs = np.asarray([_real_scalar(value) for value in target["rhs"]], dtype=np.complex128)
    solution = np.linalg.solve(matrix, rhs)
    return solution / np.linalg.norm(solution)


def _amplitude(value: Any) -> complex:
    if isinstance(value, int | float):
        return complex(value)
    if isinstance(value, str):
        square_root = re.fullmatch(r"(-?1)/sqrt\((\d+)\)", value)
        if square_root is not None:
            sign = -1 if square_root.group(1).startswith("-") else 1
            return complex(sign / math.sqrt(int(square_root.group(2))))
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return complex(float(numerator) / float(denominator))
        return complex(value)
    raise ValueError(f"unsupported exact amplitude: {value!r}")


def _real_scalar(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if value == "1/sqrt(2)":
        return 1 / math.sqrt(2)
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def _controlled_operator(target: dict[str, Any]) -> np.ndarray:
    controls = tuple(int(value) for value in target.get("controls", [target.get("control")]))
    output = int(target["target"])
    width = max((*controls, output)) + 1
    dimension = 2**width
    operator = str(target["operator"])
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis in range(dimension):
        if all((basis >> control) & 1 for control in controls):
            if operator == "X":
                matrix[basis ^ (1 << output), basis] = 1.0
            elif operator == "H":
                bit = (basis >> output) & 1
                base = basis & ~(1 << output)
                matrix[base, basis] = 1 / math.sqrt(2)
                matrix[base | (1 << output), basis] = (1 if bit == 0 else -1) / math.sqrt(2)
            else:
                raise ValueError(f"unsupported controlled operator: {operator}")
        else:
            matrix[basis, basis] = 1.0
    return matrix


def _task49_state(target: dict[str, Any]) -> np.ndarray:
    state = np.zeros(4, dtype=np.complex128)
    state[1] = 1.0
    steps = int(target["steps"])
    delta = float(target["time"]) / steps
    for _ in range(steps):
        for basis in range(4):
            parity = 1 if ((basis >> 0) & 1) == ((basis >> 1) & 1) else -1
            state[basis] *= np.exp(-1j * 0.7 * delta * parity)
        _apply_x_evolution(state, 0.4 * delta, 0)
        _apply_x_evolution(state, 0.3 * delta, 1)
    return state


def _ideal_heisenberg_state(target: dict[str, Any]) -> np.ndarray:
    couplings = target["couplings"]
    hamiltonian = np.zeros((8, 8), dtype=np.complex128)
    for basis in range(8):
        for first, second in ((0, 1), (1, 2)):
            hamiltonian[basis ^ (1 << first) ^ (1 << second), basis] += float(couplings["Jx"])
            y_phase = (1j if not (basis >> first) & 1 else -1j) * (1j if not (basis >> second) & 1 else -1j)
            hamiltonian[basis ^ (1 << first) ^ (1 << second), basis] += float(couplings["Jy"]) * y_phase
            parity = 1 if ((basis >> first) & 1) == ((basis >> second) & 1) else -1
            hamiltonian[basis, basis] += float(couplings["Jz"]) * parity
        field = sum(1 if not (basis >> wire) & 1 else -1 for wire in range(3))
        hamiltonian[basis, basis] += float(couplings["h"]) * field
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    initial = np.zeros(8, dtype=np.complex128)
    initial[int(str(target["initial_state"]).split("_", maxsplit=1)[0], 2)] = 1.0
    phases = np.exp(-1j * eigenvalues * float(target["time"]))
    return eigenvectors @ (phases * (eigenvectors.conjugate().T @ initial))


def _trotter_ordering_state_set(target: dict[str, Any]) -> np.ndarray:
    """Enumerate every symmetric-ordering second-order Trotter state.

    The prompt fixes couplings, time, and step count but not the ordering of
    the exponential factors inside each symmetric half-step. Membership in
    this exhaustive set is the strongest check that admits every valid
    ordering while rejecting wrong-Hamiltonian circuits that merely land
    inside a distance ball around the ideal evolution.
    """
    from itertools import permutations

    couplings = target["couplings"]
    steps = int(target["steps"])
    delta = float(target["time"]) / steps
    factors = _heisenberg_factors(
        float(couplings["Jx"]),
        float(couplings["Jy"]),
        float(couplings["Jz"]),
        float(couplings["h"]),
    )
    half_steps = [_matrix_exponential(factor, delta / 2) for factor in factors]
    initial = np.zeros(8, dtype=np.complex128)
    initial[int(str(target["initial_state"]).split("_", maxsplit=1)[0], 2)] = 1.0
    states: list[np.ndarray] = []
    seen: set[bytes] = set()
    for ordering in permutations(range(len(half_steps))):
        step = _symmetric_step(half_steps, ordering)
        state = initial
        for _ in range(steps):
            state = step @ state
        fingerprint = np.round(state * np.exp(-1j * np.angle(state[np.argmax(np.abs(state))])), 12).tobytes()
        if fingerprint not in seen:
            seen.add(fingerprint)
            states.append(state)
    return np.asarray(states, dtype=np.complex128)


def _symmetric_step(half_steps: list[np.ndarray], ordering: tuple[int, ...]) -> np.ndarray:
    step = np.eye(8, dtype=np.complex128)
    for index in ordering:
        step = half_steps[index] @ step
    for index in reversed(ordering):
        step = half_steps[index] @ step
    return step


def _heisenberg_factors(jx: float, jy: float, jz: float, field: float) -> list[np.ndarray]:
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)

    def two_site(op: np.ndarray, first: int, second: int) -> np.ndarray:
        matrices = [identity, identity, identity]
        matrices[first] = op
        matrices[second] = op
        # Little-endian composition: wire w occupies tensor slot w.
        return np.kron(np.kron(matrices[2], matrices[1]), matrices[0])

    factors: list[np.ndarray] = [
        jx * two_site(x, 0, 1),
        jx * two_site(x, 1, 2),
        jy * two_site(y, 0, 1),
        jy * two_site(y, 1, 2),
        jz * two_site(z, 0, 1),
        jz * two_site(z, 1, 2),
    ]
    field_term = np.zeros((8, 8), dtype=np.complex128)
    for wire in range(3):
        matrices = [identity, identity, identity]
        matrices[wire] = z
        field_term += np.kron(np.kron(matrices[2], matrices[1]), matrices[0])
    factors.append(field * field_term)
    return factors


def _matrix_exponential(hermitian: np.ndarray, scale: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    return eigenvectors @ np.diag(np.exp(-1j * eigenvalues * scale)) @ eigenvectors.conjugate().T


_FAMILY_OPERATION = re.compile(r"^(RX|RY|RZ|CX|X)(\d*)\(([^)]*)\)$")


def _parameterized_family_state(target: dict[str, Any], arguments: tuple[object, ...]) -> np.ndarray:
    """Evaluate a declared rotation-family target at the executed arguments.

    This backs the numeric fallback for candidates whose source spelling is
    not provably matched by the structured AST verifier: the family target is
    bound at each executed diagnostic point and compared state-exactly.
    """
    values = _family_parameter_values(arguments)
    order = target.get("operation_order")
    if not isinstance(order, list) or not order:
        raise NotImplementedError("parameterized state family target is missing operation_order")
    initial = str(target.get("initial_state", "0")).split("_", maxsplit=1)[0]
    width = len(initial)
    state = np.zeros(2**width, dtype=np.complex128)
    state[int(initial, 2)] = 1.0
    for item in order:
        state = _apply_family_operation(state, str(item), values, width)
    return state


def _family_parameter_values(arguments: tuple[object, ...]) -> list[float]:
    if len(arguments) == 1 and isinstance(arguments[0], list | tuple):
        return [float(value) for value in arguments[0]]
    return [float(value) for value in arguments]  # type: ignore[arg-type]


def _apply_family_operation(
    state: np.ndarray,
    operation: str,
    values: list[float],
    width: int,
) -> np.ndarray:
    match = _FAMILY_OPERATION.fullmatch(operation.replace(" ", ""))
    if match is None:
        raise NotImplementedError(f"unsupported family operation: {operation}")
    name, wire_suffix, argument = match.groups()
    if name == "CX":
        control, out = (int(part) for part in argument.split(","))
        return _apply_family_cx(state, control, out, width)
    wire = int(wire_suffix) if wire_suffix else 0
    if name == "X":
        return _apply_family_gate(state, np.asarray([[0, 1], [1, 0]], dtype=np.complex128), wire, width)
    angle = values[int(argument.removeprefix("p"))] if argument.startswith("p") else values[_family_index(argument)]
    half = angle / 2
    if name == "RX":
        gate = np.asarray(
            [[math.cos(half), -1j * math.sin(half)], [-1j * math.sin(half), math.cos(half)]],
            dtype=np.complex128,
        )
    elif name == "RY":
        gate = np.asarray(
            [[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]],
            dtype=np.complex128,
        )
    else:
        gate = np.asarray(
            [[np.exp(-1j * half), 0], [0, np.exp(1j * half)]],
            dtype=np.complex128,
        )
    return _apply_family_gate(state, gate, wire, width)


_FAMILY_BINDINGS = {"rx_angle": 0, "ry_angle": 1}


def _family_index(argument: str) -> int:
    if argument in _FAMILY_BINDINGS:
        return _FAMILY_BINDINGS[argument]
    raise NotImplementedError(f"unsupported family binding: {argument}")


def _apply_family_gate(state: np.ndarray, gate: np.ndarray, wire: int, width: int) -> np.ndarray:
    result = state.copy()
    stride = 1 << wire
    for base in range(0, 1 << width, 2 * stride):
        for offset in range(stride):
            low = base + offset
            high = low + stride
            a, b = state[low], state[high]
            result[low] = gate[0, 0] * a + gate[0, 1] * b
            result[high] = gate[1, 0] * a + gate[1, 1] * b
    return result


def _apply_family_cx(state: np.ndarray, control: int, target_wire: int, width: int) -> np.ndarray:
    result = state.copy()
    for basis in range(1 << width):
        if (basis >> control) & 1:
            result[basis ^ (1 << target_wire)] = state[basis]
    return result


def _apply_x_evolution(state: np.ndarray, angle: float, wire: int) -> None:
    stride = 1 << wire
    cosine = math.cos(angle)
    sine = -1j * math.sin(angle)
    for base in range(0, len(state), 2 * stride):
        for offset in range(stride):
            first = base + offset
            second = first + stride
            a, b = state[first], state[second]
            state[first] = cosine * a + sine * b
            state[second] = sine * a + cosine * b
