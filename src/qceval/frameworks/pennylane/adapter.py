"""PennyLane lowering adapter that builds Program IR without a verdict."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from qceval.frameworks.pennylane.operations import (
    _lower_tape_operation,
    _lower_terminal_measurement,
    _measurement_value_bits,
)
from qceval.frameworks.pennylane.wire_map import (
    _contract_output_wires,
    _tape_wire_labels,
    _Unsupported,
)
from qceval.semantics.contracts import Contract
from qceval.semantics.ir import (
    IR_VERSION,
    Operation,
    Program,
    Provenance,
    validate_program,
)
from qceval.semantics.lowering.base import (
    CapabilitySet,
    FrameworkFingerprint,
    LoweringResult,
    LoweringStatus,
    SourceMetadata,
)
from qceval.semantics.lowering.utils import lowering_failure


class PennyLaneLoweringAdapter:
    """Lower supported PennyLane tapes without producing a verdict."""

    def lower(
        self,
        returned_value: Any,
        source_metadata: SourceMetadata,
        contract: Contract | None,
    ) -> LoweringResult:
        """Lower a PennyLane quantum script/tape.

        Args:
            returned_value: Captured PennyLane tape or ``QuantumScript``.
            source_metadata: Candidate source/backend diagnostics.
            contract: Optional task contract reserved for semantic planning.

        Returns:
            Program IR or typed non-verdict failure.
        """
        if not _is_tape(returned_value):
            return lowering_failure(
                LoweringStatus.EXECUTION_ERROR,
                "captured_tape_required",
                detail=type(returned_value).__name__,
            )
        try:
            program = _lower_tape(returned_value, source_metadata, contract)
            validate_program(program)
        except _Unsupported as exc:
            return lowering_failure(
                LoweringStatus.UNSUPPORTED,
                exc.reason,
                node_kind=exc.node_kind,
                source_location=exc.location,
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
        """Return PennyLane lowering feature claims."""
        return CapabilitySet(("static_gates", "state_preparation", "terminal_measurement", "captured_tape"))

    def framework_fingerprint(self) -> FrameworkFingerprint:
        """Return the installed PennyLane version."""
        return FrameworkFingerprint("pennylane", importlib.metadata.version("pennylane"))


def _lower_tape(tape: Any, metadata: SourceMetadata, contract: Contract | None) -> Program:
    labels = _tape_wire_labels(tape, contract)
    num_qubits = max(labels, default=-1) + 1
    operations: list[Operation] = []
    measured_bits: dict[int, int] = {}
    measurement_ids: dict[int, int] = {}
    next_clbit = 0
    for index, native in enumerate(tape.operations):
        location = f"operations[{index}]"
        lowered, next_clbit = _lower_tape_operation(native, location, measured_bits, next_clbit, measurement_ids)
        operations.extend(lowered)
    render_override: tuple[int, ...] | None = None
    for index, measurement in enumerate(tape.measurements):
        value_bits = _measurement_value_bits(measurement, measurement_ids, index)
        if value_bits is not None:
            if render_override is not None:
                raise _Unsupported(
                    "unsupported_measurement_process", "multiple_value_statistics", f"measurements[{index}]"
                )
            render_override = value_bits
            continue
        terminal_operation, next_clbit = _lower_terminal_measurement(
            measurement,
            index,
            labels,
            next_clbit,
            _contract_output_wires(contract),
        )
        if terminal_operation is not None:
            operations.append(terminal_operation)
    return Program(
        IR_VERSION,
        num_qubits,
        next_clbit,
        tuple(operations),
        None,
        render_override if render_override is not None else tuple(range(next_clbit)),
        Provenance(
            "pennylane",
            importlib.metadata.version("pennylane"),
            source_hash=metadata.source_hash,
            backend=metadata.backend,
        ),
    )


def _is_tape(value: Any) -> bool:
    return (
        value.__class__.__module__.startswith("pennylane")
        and hasattr(value, "operations")
        and hasattr(value, "measurements")
    )
