"""Framework-neutral dense materialization from supported Program IR."""

from __future__ import annotations

import math

import numpy as np

from qceval.semantics.contracts.kinds import SystemRole
from qceval.semantics.ir import OperationKind, Program
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.classical_wires import terminal_wire_permutation
from qceval.semantics.verifiers.dynamic import ExactBranchSimulator, reduced_density_matrix
from qceval.semantics.verifiers.materialize import (
    ArrayMaterialization,
    CandidateSemanticError,
    ClassicalTableMaterialization,
)


class ProgramIRMaterializer:
    """Materialize supported static dense objects from Program IR."""

    def __init__(self, simulator: ExactBranchSimulator | None = None) -> None:
        """Initialize a Program IR materializer.

        Args:
            simulator: Optional exact statevector branch simulator.
        """
        self._simulator = simulator or ExactBranchSimulator()

    def array(self, context: VerificationContext, representation: str) -> ArrayMaterialization:
        """Materialize a statevector, unitary, logical isometry, or Choi matrix.

        Args:
            context: Contract and candidate Program IR.
            representation: Requested dense representation.

        Returns:
            Exact complex128 semantic object.

        Raises:
            NotImplementedError: If the representation or IR is unsupported.
        """
        program = _without_terminal_measurements(
            context.program,
            allow_mid_measurement=(context.contract.observation.postselection is not None or representation == "choi"),
        )
        if representation == "statevector":
            return ArrayMaterialization(self._state(program, context), representation)
        if representation == "unitary":
            value = self._columns(program, tuple(range(program.num_qubits)))
            return ArrayMaterialization(value, representation, value.shape[1])
        if representation == "isometry":
            logical_wires = tuple(
                index
                for system in context.contract.systems.items
                if system.role in {SystemRole.LOGICAL_INPUT, SystemRole.LOGICAL_IO}
                for index in system.indices
            )
            if not logical_wires:
                raise NotImplementedError("logical input subspace is not declared")
            value = self._columns(program, logical_wires)
            return ArrayMaterialization(value, representation, value.shape[1])
        if representation == "choi":
            value = self._choi(program, context)
            return ArrayMaterialization(value, representation, value.shape[0])
        raise NotImplementedError(f"Program IR materialization does not support {representation!r}")

    def classical_table(self, context: VerificationContext) -> ClassicalTableMaterialization:
        """Reject source-replay classical domains at the Program-only seam.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            This method never returns.

        Raises:
            NotImplementedError: Always; one Program instance is not an
                exhaustive public-argument family.
        """
        del context
        raise NotImplementedError("classical I/O requires exhaustive source argument replay")

    def _state(self, program: Program, context: VerificationContext) -> np.ndarray:
        wire_map = terminal_wire_permutation(context.contract, context.program) or {}
        observed_wires = tuple(
            wire_map.get(index, index)
            for system in context.contract.systems.items
            if system.name in context.contract.observation.quantum
            for index in system.indices
        )
        branches = self._simulator.run(program, max_branches=context.contract.limits.max_branches)
        if len(branches) != 1 or not math.isclose(branches[0].probability, 1.0, abs_tol=1e-12):
            raise NotImplementedError("statevector route requires one deterministic pure branch")
        state = _postselect(branches[0].statevector, context, wire_map)
        if not observed_wires or set(observed_wires) == set(range(program.num_qubits)):
            return state
        density = reduced_density_matrix(state, observed_wires, program.num_qubits)
        eigenvalues, eigenvectors = np.linalg.eigh(density)
        dominant = int(np.argmax(eigenvalues))
        if float(eigenvalues[dominant]) < 1.0 - max(1e-10, context.contract.approximation.uncertainty):
            raise CandidateSemanticError("logical_output_not_pure")
        return eigenvectors[:, dominant]

    def _columns(self, program: Program, logical_wires: tuple[int, ...]) -> np.ndarray:
        dimension = 2**program.num_qubits
        columns = np.zeros((dimension, 2 ** len(logical_wires)), dtype=np.complex128)
        for logical_basis in range(columns.shape[1]):
            physical_basis = sum(
                ((logical_basis >> position) & 1) << wire for position, wire in enumerate(logical_wires)
            )
            initial = np.zeros(dimension, dtype=np.complex128)
            initial[physical_basis] = 1.0
            branches = self._simulator.run(program, initial_state=initial, max_branches=1)
            if len(branches) != 1 or not math.isclose(branches[0].probability, 1.0, abs_tol=1e-12):
                raise NotImplementedError("operator route encountered dynamic branching")
            columns[:, logical_basis] = branches[0].statevector
        return columns

    def _choi(self, program: Program, context: VerificationContext) -> np.ndarray:
        input_wires = tuple(
            index
            for system in context.contract.systems.items
            if system.role in {SystemRole.LOGICAL_INPUT, SystemRole.LOGICAL_IO}
            for index in system.indices
        )
        output_wires = tuple(
            index
            for system in context.contract.systems.items
            if system.name in context.contract.observation.quantum
            for index in system.indices
        )
        if not input_wires or not output_wires:
            raise NotImplementedError("channel route requires logical input and quantum output systems")
        input_dim = 2 ** len(input_wires)
        output_dim = 2 ** len(output_wires)
        diagonal = [
            self._channel_image(
                program,
                input_wires,
                output_wires,
                _basis_vector(input_dim, index),
                context.contract.limits.max_branches,
            )
            for index in range(input_dim)
        ]
        images: dict[tuple[int, int], np.ndarray] = {(index, index): diagonal[index] for index in range(input_dim)}
        for row in range(input_dim):
            for column in range(row + 1, input_dim):
                plus = (_basis_vector(input_dim, row) + _basis_vector(input_dim, column)) / math.sqrt(2)
                plus_i = (_basis_vector(input_dim, row) + 1j * _basis_vector(input_dim, column)) / math.sqrt(2)
                sum_image = (
                    2
                    * self._channel_image(
                        program,
                        input_wires,
                        output_wires,
                        plus,
                        context.contract.limits.max_branches,
                    )
                    - diagonal[row]
                    - diagonal[column]
                )
                skew_image = (
                    2
                    * self._channel_image(
                        program,
                        input_wires,
                        output_wires,
                        plus_i,
                        context.contract.limits.max_branches,
                    )
                    - diagonal[row]
                    - diagonal[column]
                )
                image = (sum_image + 1j * skew_image) / 2
                images[(row, column)] = image
                images[(column, row)] = image.conjugate().T
        choi = np.zeros((input_dim * output_dim, input_dim * output_dim), dtype=np.complex128)
        for input_row in range(input_dim):
            for input_column in range(input_dim):
                block = images[(input_row, input_column)]
                row_start = input_row * output_dim
                column_start = input_column * output_dim
                choi[row_start : row_start + output_dim, column_start : column_start + output_dim] = block
        return choi

    def _channel_image(
        self,
        program: Program,
        input_wires: tuple[int, ...],
        output_wires: tuple[int, ...],
        logical_state: np.ndarray,
        max_branches: int,
    ) -> np.ndarray:
        full_state = _embed_logical_state(logical_state, program.num_qubits, input_wires)
        branches = self._simulator.run(
            program,
            initial_state=full_state,
            max_branches=max_branches,
        )
        density = np.zeros((2 ** len(output_wires), 2 ** len(output_wires)), dtype=np.complex128)
        for branch in branches:
            density += branch.probability * reduced_density_matrix(branch.statevector, output_wires, program.num_qubits)
        return density


def _without_terminal_measurements(program: Program, *, allow_mid_measurement: bool = False) -> Program:
    terminal = set()
    for index, operation in enumerate(program.operations):
        if operation.kind is not OperationKind.MEASUREMENT:
            continue
        measured_wires = set(operation.quantum_wires)
        written_bits = set(operation.classical_bits)
        # Barriers are semantic no-ops.  In particular, a barrier appended
        # after terminal measurements must not turn those measurements into
        # dynamic, mid-circuit observations.
        later = tuple(item for item in program.operations[index + 1 :] if item.kind is not OperationKind.BARRIER)
        if all(
            not measured_wires.intersection((*item.quantum_wires, *(control.wire for control in item.controls)))
            and (item.condition is None or not written_bits.intersection(item.condition.bits))
            for item in later
        ):
            terminal.add(index)
    operations = tuple(operation for index, operation in enumerate(program.operations) if index not in terminal)
    has_mid_measurement = any(operation.kind is OperationKind.MEASUREMENT for operation in operations)
    if has_mid_measurement and not allow_mid_measurement:
        raise NotImplementedError("mid-circuit measurement requires a dynamic semantic route")
    return Program(
        program.ir_version,
        program.num_qubits,
        program.num_clbits if has_mid_measurement else 0,
        operations,
        program.global_phase,
        (),
        program.provenance,
        program.diagnostics,
    )


def _postselect(state: np.ndarray, context: VerificationContext, wire_map: dict[int, int] | None = None) -> np.ndarray:
    postselection = context.contract.observation.postselection
    if postselection is None:
        return state
    systems = {system.name: system for system in context.contract.systems.items}
    system = systems[postselection.system]
    wires = tuple((wire_map or {}).get(index, index) for index in system.indices)
    allowed = {int(value, 2) for value in postselection.values}
    mask = np.asarray(
        [_extract_selected_bits(basis, wires) in allowed for basis in range(state.size)],
        dtype=bool,
    )
    probability = float(np.sum(np.abs(state[mask]) ** 2))
    if probability < postselection.min_probability:
        raise CandidateSemanticError("postselection_probability_below_minimum")
    return np.where(mask, state, 0.0) / math.sqrt(probability)


def _extract_selected_bits(value: int, wires: tuple[int, ...]) -> int:
    return sum(((value >> wire) & 1) << position for position, wire in enumerate(wires))


def _basis_vector(dimension: int, index: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.complex128)
    vector[index] = 1.0
    return vector


def _embed_logical_state(logical_state: np.ndarray, num_qubits: int, logical_wires: tuple[int, ...]) -> np.ndarray:
    logical_state = np.asarray(logical_state, dtype=np.complex128)
    expected = 2 ** len(logical_wires)
    if logical_state.shape != (expected,):
        raise NotImplementedError("logical channel input has wrong dimension")
    norm = float(np.linalg.norm(logical_state))
    if not math.isclose(norm, 1.0, abs_tol=1e-10):
        raise NotImplementedError("logical channel input must be normalized")
    full_state = np.zeros(2**num_qubits, dtype=np.complex128)
    for logical_basis, amplitude in enumerate(logical_state):
        physical_basis = sum(((logical_basis >> position) & 1) << wire for position, wire in enumerate(logical_wires))
        full_state[physical_basis] = amplitude
    return full_state
