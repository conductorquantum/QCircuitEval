"""Cirq lowering adapter that builds Program IR without a verdict."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from qceval.frameworks.cirq.operations import _lower_native
from qceval.frameworks.cirq.wire_map import _Unsupported, _wire_map
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
from qceval.semantics.lowering.utils import lowering_failure, normalize_parameter


class CirqLoweringAdapter:
    """Lower supported Cirq circuits without producing a verdict."""

    def lower(
        self,
        returned_value: Any,
        source_metadata: SourceMetadata,
        contract: Contract | None,
    ) -> LoweringResult:
        """Lower a Cirq circuit.

        Args:
            returned_value: Expected ``cirq.Circuit``.
            source_metadata: Candidate source/backend diagnostics.
            contract: Optional task contract reserved for semantic planning.

        Returns:
            Program IR or typed non-verdict failure.
        """
        if not _is_circuit(returned_value):
            return lowering_failure(
                LoweringStatus.EXECUTION_ERROR,
                "invalid_return_type",
                detail=type(returned_value).__name__,
            )
        try:
            program = _lower_circuit(returned_value, source_metadata, contract)
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
        """Return Cirq lowering feature claims."""
        return CapabilitySet(("static_gates", "terminal_measurement", "mid_measurement", "reset", "global_phase"))

    def framework_fingerprint(self) -> FrameworkFingerprint:
        """Return the installed Cirq version."""
        return FrameworkFingerprint("cirq", importlib.metadata.version("cirq"))


def _lower_circuit(circuit: Any, metadata: SourceMetadata, contract: Contract | None) -> Program:
    qubits = sorted(circuit.all_qubits())
    mapped = _wire_map(qubits, contract)
    num_qubits = max(mapped.values(), default=-1) + 1
    operations: list[Operation] = []
    key_bits: dict[str, tuple[int, ...]] = {}
    global_phase = 0.0
    next_clbit = 0
    for index, native in enumerate(circuit.all_operations()):
        lowered, next_clbit, operation_phase = _lower_native(native, index, mapped, key_bits, next_clbit)
        operations.extend(lowered)
        global_phase += operation_phase
    normalized_phase = normalize_parameter(global_phase) if abs(global_phase) > 0 else None
    return Program(
        IR_VERSION,
        num_qubits,
        next_clbit,
        tuple(operations),
        normalized_phase,
        _classical_render_order(key_bits, next_clbit),
        Provenance(
            "cirq",
            importlib.metadata.version("cirq"),
            source_hash=metadata.source_hash,
            backend=metadata.backend,
        ),
    )


def _classical_render_order(key_bits: dict[str, tuple[int, ...]], num_clbits: int) -> tuple[int, ...]:
    """Return the rendered classical-bit order for the measured keys.

    A single joint measurement key follows Cirq's own convention: the listed
    qubit order is the rendered string order. Sequential single-bit keys (the
    dynamic-circuit style) follow the benchmark convention instead: the first
    measured bit is the least-significant bit, matching Qiskit's classical
    register packing for iterative protocols.

    Args:
        key_bits: Measurement key to allocated classical-bit mapping.
        num_clbits: Total classical bits allocated for the circuit.

    Returns:
        Classical-bit render order used by Program IR.
    """
    if len(key_bits) > 1 and all(len(bits) == 1 for bits in key_bits.values()):
        return tuple(reversed(range(num_clbits)))
    return tuple(range(num_clbits))


def _is_circuit(value: Any) -> bool:
    return value.__class__.__module__.startswith("cirq") and value.__class__.__name__ in {
        "Circuit",
        "FrozenCircuit",
    }
