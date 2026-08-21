"""Resource-bounded exact branch simulation for dynamic Program IR."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qceval.semantics.ir import Operation, OperationKind, ParameterKind, Program
from qceval.semantics.verifiers.result import SemanticStatus


@dataclass(frozen=True)
class DynamicBranch:
    """One exact classical branch with a normalized pure state."""

    probability: float
    statevector: np.ndarray
    classical_bits: tuple[int, ...]


class DynamicSimulationError(RuntimeError):
    """Typed dynamic-simulation capability or resource failure."""

    def __init__(self, status: SemanticStatus, reason: str) -> None:
        """Initialize a stable dynamic failure.

        Args:
            status: Unsupported or resource-limit status.
            reason: Stable machine-readable reason.
        """
        self.status = status
        self.reason = reason
        super().__init__(reason)


class ExactBranchSimulator:
    """Enumerate ideal measurement/reset/feed-forward branches exactly."""

    def run(
        self,
        program: Program,
        *,
        initial_state: np.ndarray | None = None,
        max_branches: int,
    ) -> tuple[DynamicBranch, ...]:
        """Execute supported Program IR with a deterministic branch cap.

        Args:
            program: Framework-neutral program.
            initial_state: Optional normalized full-register statevector.
            max_branches: Maximum live branches after any operation.

        Returns:
            Exact nonzero terminal branches.

        Raises:
            DynamicSimulationError: On unsupported behavior or branch excess.
        """
        if max_branches < 1:
            raise DynamicSimulationError(SemanticStatus.RESOURCE_LIMIT, "invalid_branch_limit")
        dimension = 2**program.num_qubits
        state = (
            np.zeros(dimension, dtype=np.complex128)
            if initial_state is None
            else np.asarray(initial_state, dtype=np.complex128)
        )
        if initial_state is None:
            state[0] = 1.0
        if (
            state.shape != (dimension,)
            or not np.all(np.isfinite(state))
            or abs(float(np.linalg.norm(state)) - 1.0) > 1e-10
        ):
            raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "invalid_initial_state")
        state = _apply_global_phase(state, program)
        branches: tuple[DynamicBranch, ...] = (DynamicBranch(1.0, state.copy(), (0,) * program.num_clbits),)
        for operation in program.operations:
            branches = self._operation(branches, operation, program.num_qubits)
            if len(branches) > max_branches:
                raise DynamicSimulationError(SemanticStatus.RESOURCE_LIMIT, "dynamic_branch_limit")
        return branches

    def _operation(
        self,
        branches: tuple[DynamicBranch, ...],
        operation: Operation,
        num_qubits: int,
    ) -> tuple[DynamicBranch, ...]:
        if operation.kind is OperationKind.BARRIER:
            return branches
        if operation.kind is OperationKind.MEASUREMENT:
            return self._measurement(branches, operation)
        if operation.kind is OperationKind.RESET:
            return self._reset(branches, operation)
        if operation.kind is OperationKind.STATE_PREPARATION:
            return self._state_preparation(branches, operation, num_qubits)
        if operation.kind is not OperationKind.GATE:
            raise DynamicSimulationError(
                SemanticStatus.EXECUTION_ERROR,
                f"dynamic_node_unsupported:{operation.kind.value}",
            )
        return self._gate(branches, operation, num_qubits)

    def _state_preparation(
        self,
        branches: tuple[DynamicBranch, ...],
        operation: Operation,
        num_qubits: int,
    ) -> tuple[DynamicBranch, ...]:
        from qceval.semantics.verifiers.dynamic.payload import _semantic_statevector

        if operation.name != "statevector":
            raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "state_preparation_unsupported")
        local = _semantic_statevector(operation)
        values = []
        for branch in branches:
            if not _condition_matches(branch, operation):
                values.append(branch)
                continue
            nonzero = np.flatnonzero(np.abs(branch.statevector) > 1e-14)
            if nonzero.size != 1 or int(nonzero[0]) != 0:
                raise DynamicSimulationError(
                    SemanticStatus.EXECUTION_ERROR,
                    "noninitial_state_preparation_unsupported",
                )
            state = np.zeros(2**num_qubits, dtype=np.complex128)
            first_target_msb = (
                dict(operation.semantic_data).get("statevector_wire_order", "little_endian") == "big_endian"
            )
            for local_basis, amplitude in enumerate(local):
                basis = sum(
                    (
                        (
                            local_basis >> (len(operation.quantum_wires) - index - 1)
                            if first_target_msb
                            else local_basis >> index
                        )
                        & 1
                    )
                    << wire
                    for index, wire in enumerate(operation.quantum_wires)
                )
                state[basis] = amplitude
            values.append(DynamicBranch(branch.probability, state, branch.classical_bits))
        return tuple(values)

    def _measurement(
        self,
        branches: tuple[DynamicBranch, ...],
        operation: Operation,
    ) -> tuple[DynamicBranch, ...]:
        if len(operation.quantum_wires) != len(operation.classical_bits):
            raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "measurement_width_mismatch")
        # Cirq-style invert masks flip the *recorded* classical bit without
        # touching the collapsed quantum state.
        mask = dict(operation.semantic_data).get("invert_mask", "")
        values: list[DynamicBranch] = []
        for branch in branches:
            if not _condition_matches(branch, operation):
                # Skipped conditional measurement: the classical bit retains
                # its prior value, matching Qiskit/hardware semantics for an
                # untaken branch.
                values.append(branch)
                continue
            measured: tuple[DynamicBranch, ...] = (branch,)
            for index, (wire, bit) in enumerate(zip(operation.quantum_wires, operation.classical_bits, strict=True)):
                invert = index < len(mask) and mask[index] == "1"
                measured = tuple(child for parent in measured for child in _measure(parent, wire, bit, invert=invert))
            values.extend(measured)
        return tuple(values)

    def _reset(
        self,
        branches: tuple[DynamicBranch, ...],
        operation: Operation,
    ) -> tuple[DynamicBranch, ...]:
        values: list[DynamicBranch] = []
        for branch in branches:
            if not _condition_matches(branch, operation):
                values.append(branch)
                continue
            reset: tuple[DynamicBranch, ...] = (branch,)
            for wire in operation.quantum_wires:
                reset = tuple(child for parent in reset for child in _reset(parent, wire))
            values.extend(reset)
        return tuple(values)

    def _gate(
        self,
        branches: tuple[DynamicBranch, ...],
        operation: Operation,
        num_qubits: int,
    ) -> tuple[DynamicBranch, ...]:
        from qceval.semantics.verifiers.dynamic.apply import _apply_operation

        values = []
        for branch in branches:
            if operation.condition is not None and not _condition_matches(branch, operation):
                values.append(branch)
                continue
            state = _apply_operation(branch.statevector, operation, num_qubits)
            values.append(DynamicBranch(branch.probability, state, branch.classical_bits))
        return tuple(values)


def reduced_density_matrix(statevector: np.ndarray, kept_wires: tuple[int, ...], num_qubits: int) -> np.ndarray:
    """Trace all but selected wires from a pure state.

    Args:
        statevector: Normalized full-register statevector in little-endian index order.
        kept_wires: Output wires in semantic order, least-significant first.
        num_qubits: Full register width.

    Returns:
        Reduced density matrix on ``kept_wires``.
    """
    if len(set(kept_wires)) != len(kept_wires) or any(wire < 0 or wire >= num_qubits for wire in kept_wires):
        raise ValueError("invalid reduced-state wires")
    output_dimension = 2 ** len(kept_wires)
    density = np.zeros((output_dimension, output_dimension), dtype=np.complex128)
    discarded = tuple(wire for wire in range(num_qubits) if wire not in kept_wires)
    for row in range(2**num_qubits):
        output_row = _extract_bits(row, kept_wires)
        discarded_row = _extract_bits(row, discarded)
        for column in range(2**num_qubits):
            if _extract_bits(column, discarded) != discarded_row:
                continue
            output_column = _extract_bits(column, kept_wires)
            density[output_row, output_column] += statevector[row] * np.conjugate(statevector[column])
    return density


def _measure(
    branch: DynamicBranch,
    wire: int,
    classical_bit: int,
    *,
    invert: bool = False,
) -> tuple[DynamicBranch, ...]:
    values = []
    for outcome, probability, state in _collapse(branch.statevector, wire):
        bits = list(branch.classical_bits)
        bits[classical_bit] = outcome ^ 1 if invert else outcome
        values.append(DynamicBranch(branch.probability * probability, state, tuple(bits)))
    return tuple(values)


def _reset(branch: DynamicBranch, wire: int) -> tuple[DynamicBranch, ...]:
    from qceval.semantics.verifiers.dynamic.apply import _X, _apply_single

    values = []
    for outcome, probability, collapsed in _collapse(branch.statevector, wire):
        state = collapsed if outcome == 0 else _apply_single(collapsed, _X, wire, ())
        values.append(DynamicBranch(branch.probability * probability, state, branch.classical_bits))
    return tuple(values)


# Branches below this probability are indistinguishable from accumulated
# float64 rounding noise (amplitude error ~1e-13 over long circuits squares to
# ~1e-26) and sit far below every contract decision tolerance.
_BRANCH_PROBABILITY_FLOOR = 1e-24


def _collapse(statevector: np.ndarray, wire: int) -> tuple[tuple[int, float, np.ndarray], ...]:
    values = []
    for outcome in (0, 1):
        mask = np.asarray([((basis >> wire) & 1) == outcome for basis in range(statevector.size)])
        probability = float(np.sum(np.abs(statevector[mask]) ** 2))
        if probability <= _BRANCH_PROBABILITY_FLOOR:
            continue
        state = np.where(mask, statevector, 0.0) / math.sqrt(probability)
        values.append((outcome, probability, state))
    return tuple(values)


def _condition_matches(branch: DynamicBranch, operation: Operation) -> bool:
    condition = operation.condition
    if condition is None:
        return True
    value = sum(branch.classical_bits[bit] << index for index, bit in enumerate(condition.bits))
    return value == condition.value


def _apply_global_phase(state: np.ndarray, program: Program) -> np.ndarray:
    phase = program.global_phase
    if phase is None:
        return state
    if phase.kind is not ParameterKind.NUMBER:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "symbolic_global_phase")
    try:
        value = float(phase.value)
    except ValueError as exc:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "invalid_global_phase") from exc
    if not math.isfinite(value):
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "invalid_global_phase")
    return state * np.exp(1j * value)


def _extract_bits(value: int, wires: tuple[int, ...]) -> int:
    return sum(((value >> wire) & 1) << index for index, wire in enumerate(wires))
