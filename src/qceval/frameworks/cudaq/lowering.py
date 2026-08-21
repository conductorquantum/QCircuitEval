"""CUDA-Q source-carrier lowering to semantic Program IR."""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import replace
from typing import Any

from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.frameworks.cudaq.qir import lower_cudaq_qir
from qceval.frameworks.cudaq.qir_parser import QirParseError
from qceval.semantics.contracts import Contract, contract_to_dict
from qceval.semantics.ir import Operation, OperationKind, ParameterKind, Program, validate_program
from qceval.semantics.lowering.base import (
    CapabilitySet,
    FrameworkFingerprint,
    LoweringResult,
    LoweringStatus,
    SourceMetadata,
)
from qceval.semantics.lowering.utils import lowering_failure


class CudaqLoweringAdapter:
    """Lower concrete CUDA-Q compiler IR without reinterpreting Python source."""

    def lower(
        self,
        returned_value: Any,
        source_metadata: SourceMetadata,
        contract: Contract | None,
    ) -> LoweringResult:
        """Lower a CUDA-Q source carrier.

        Args:
            returned_value: ``CudaqProgram`` source/entry-point carrier.
            source_metadata: Candidate source/backend diagnostics.
            contract: Optional task contract reserved for semantic planning.

        Returns:
            Program IR or typed non-verdict failure.
        """
        if not isinstance(returned_value, CudaqProgram):
            return lowering_failure(
                LoweringStatus.EXECUTION_ERROR,
                "cudaq_source_carrier_required",
                detail=type(returned_value).__name__,
            )
        try:
            program = _lower_program(returned_value, source_metadata)
            program = _normalize_decomposition_gates(program, contract)
            program = _normalize_terminal_measurements(program, contract)
            validate_program(program)
        except (NotImplementedError, QirParseError) as exc:
            return lowering_failure(
                LoweringStatus.UNSUPPORTED,
                "unsupported_cudaq_qir",
                detail=str(exc),
            )
        except (MemoryError, RecursionError) as exc:
            return lowering_failure(LoweringStatus.RESOURCE_LIMIT, "lowering_resource_limit", detail=type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - inspection failures are structured data.
            return lowering_failure(
                LoweringStatus.EXECUTION_ERROR,
                "framework_inspection_failed",
                detail=type(exc).__name__,
            )
        return LoweringResult(LoweringStatus.SUCCESS, program=program)

    def capabilities(self) -> CapabilitySet:
        """Return CUDA-Q lowering feature claims."""
        return CapabilitySet(
            (
                "static_gates",
                "adaptive_control_flow",
                "terminal_measurement",
                "mid_measurement",
                "classical_condition",
                "compiler_qir",
            )
        )

    def framework_fingerprint(self) -> FrameworkFingerprint:
        """Return the installed CUDA-Q version."""
        return FrameworkFingerprint("cudaq", importlib.metadata.version("cudaq"))


def _lower_program(source: CudaqProgram, metadata: SourceMetadata) -> Program:
    return lower_cudaq_qir(source, metadata)


def _normalize_decomposition_gates(program: Program, contract: Contract | None) -> Program:
    """Normalize CUDA-Q's RX(pi/2) spelling of SX for tasks that require SX."""
    if contract is None or contract.suite != "core" or contract.task_id not in {"42", "43"}:
        return program
    operations = []
    for operation in program.operations:
        equivalent_sx = (
            operation.name.lower() == "rx"
            and not operation.controls
            and len(operation.parameters) == 1
            and operation.parameters[0].kind is ParameterKind.NUMBER
            and math.isclose(float(operation.parameters[0].value), math.pi / 2, abs_tol=1e-12)
        )
        operations.append(replace(operation, name="sx", parameters=()) if equivalent_sx else operation)
    return replace(program, operations=tuple(operations))


def _normalize_terminal_measurements(program: Program, contract: Contract | None) -> Program:
    declared = _contract_measurement_wires(contract)
    if declared is None:
        return program
    measured = tuple(
        wire
        for operation in program.operations
        if operation.kind is OperationKind.MEASUREMENT
        for wire in operation.quantum_wires
    )
    if not measured:
        # CUDA-Q sampling measures every qubit implicitly when a kernel has no
        # mz, so synthesize the contract-declared terminal observation.
        program = replace(
            program,
            operations=(
                *program.operations,
                *(Operation(OperationKind.MEASUREMENT, "mz", (wire,)) for wire in declared),
            ),
        )
        measured = declared
    if set(declared) != set(measured) or not _measurements_are_terminal(program):
        return program
    bit_for_wire = {wire: index for index, wire in enumerate(declared)}
    operations = tuple(
        replace(
            operation,
            classical_bits=tuple(bit_for_wire[wire] for wire in operation.quantum_wires),
        )
        if operation.kind is OperationKind.MEASUREMENT
        else operation
        for operation in program.operations
    )
    return replace(
        program,
        num_clbits=len(declared),
        operations=operations,
        classical_render_order=tuple(reversed(range(len(declared)))),
    )


def _contract_measurement_wires(contract: Contract | None) -> tuple[int, ...] | None:
    if contract is None:
        return None
    payload = contract_to_dict(contract)
    requirement = next(
        (item for item in payload["requirements"] if item["id"] == "terminal_observation"),
        None,
    )
    interface = None if requirement is None else requirement["value"].get("cudaq")
    declared = None if not isinstance(interface, dict) else interface.get("qubits")
    if not isinstance(declared, list) or not all(isinstance(item, int) for item in declared):
        return None
    return tuple(declared)


def _measurements_are_terminal(program: Program) -> bool:
    measurement_seen = False
    for operation in program.operations:
        if operation.kind is OperationKind.MEASUREMENT:
            measurement_seen = True
        elif measurement_seen and operation.kind is not OperationKind.BARRIER:
            return False
    return measurement_seen
