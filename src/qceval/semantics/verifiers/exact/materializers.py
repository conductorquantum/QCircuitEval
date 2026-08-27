"""Program IR materializers for exact classical I/O verification."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from qceval.semantics.ir import OperationKind, Program
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.dynamic import ExactBranchSimulator
from qceval.semantics.verifiers.exact.classical import (
    _classical_input_wires,
    _classical_output_bits,
    _packaged_target,
    _strip_prefix_x_wires,
)
from qceval.semantics.verifiers.materialize import ArrayMaterialization, ClassicalTableMaterialization


class ProgramClassicalIOMaterializer:
    """Exhaustively replay finite deterministic classical relations from Program IR."""

    def __init__(self, simulator: ExactBranchSimulator | None = None) -> None:
        """Initialize a classical I/O materializer."""
        self._simulator = simulator or ExactBranchSimulator()

    def classical_table(self, context: VerificationContext) -> ClassicalTableMaterialization:
        """Enumerate declared finite inputs and read declared classical outputs.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Complete deterministic input/output table.
        """
        input_wires = _classical_input_wires(context)
        program = _without_baked_classical_witness(context.program, context)
        output_bits = _classical_output_bits(context)
        rows: list[tuple[str, str]] = []
        for value in range(2 ** len(input_wires)):
            initial = _basis_state(program.num_qubits, input_wires, value)
            outcomes: dict[str, float] = {}
            for branch in self._simulator.run(
                program,
                initial_state=initial,
                max_branches=context.contract.limits.max_branches,
            ):
                output = "".join(str(branch.classical_bits[index]) for index in output_bits)
                outcomes[output] = outcomes.get(output, 0.0) + branch.probability
            deterministic = [key for key, probability in outcomes.items() if probability > 1.0 - 1e-12]
            output = deterministic[0] if len(deterministic) == 1 else "__nondeterministic__"
            rows.append((f"{value:0{len(input_wires)}b}", output))
        return ClassicalTableMaterialization(tuple(rows))

    def array(self, context: VerificationContext, representation: str) -> ArrayMaterialization:
        """Reject array requests at the classical-only materializer seam.

        Args:
            context: Verification context, unused by this route.
            representation: Requested dense representation.

        Returns:
            This method never returns a value.
        """
        del context
        raise NotImplementedError(f"classical I/O materializer does not support {representation!r}")


def _basis_state(num_qubits: int, input_wires: tuple[int, ...], value: int) -> np.ndarray:
    basis = sum(((value >> position) & 1) << wire for position, wire in enumerate(input_wires))
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    state[basis] = 1.0
    return state


def _without_baked_classical_witness(program: Program, context: VerificationContext) -> Program:
    removable = _strip_prefix_x_wires(_packaged_target(context))
    if not removable:
        return program
    prefix = True
    operations = []
    for operation in program.operations:
        if prefix and operation.kind is OperationKind.BARRIER:
            operations.append(operation)
            continue
        if (
            prefix
            and operation.kind is OperationKind.GATE
            and operation.name.lower() == "x"
            and not operation.controls
            and not operation.parameters
        ):
            if len(operation.quantum_wires) == 1 and operation.quantum_wires[0] in removable:
                continue
            operations.append(operation)
            continue
        prefix = False
        operations.append(operation)
    return replace(program, operations=tuple(operations))
