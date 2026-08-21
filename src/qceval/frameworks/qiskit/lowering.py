"""Qiskit QuantumCircuit lowering to semantic Program IR."""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import replace
from typing import Any

from qceval.semantics.contracts import Contract
from qceval.semantics.ir import (
    IR_VERSION,
    ClassicalCondition,
    Control,
    Operation,
    OperationKind,
    Parameter,
    ParameterKind,
    Program,
    Provenance,
    validate_program,
)
from qceval.semantics.lowering.base import (
    CapabilitySet,
    FrameworkFingerprint,
    LoweringError,
    LoweringResult,
    LoweringStatus,
    SourceMetadata,
)
from qceval.semantics.lowering.utils import bounded_matrix_semantic_data, bounded_statevector_semantic_data

_CONTROL_FLOW = frozenset({"if_else", "while_loop", "for_loop", "switch_case", "break_loop", "continue_loop", "box"})
_STATE_PREPARATION = frozenset({"initialize", "state_preparation", "set_statevector", "set_density_matrix"})
_OPAQUE_RUNTIME = frozenset({"delay", "snapshot", "save_state", "save_statevector", "save_density_matrix"})


class QiskitLoweringAdapter:
    """Lower supported Qiskit circuits without producing a verdict."""

    def lower(
        self,
        returned_value: Any,
        source_metadata: SourceMetadata,
        contract: Contract | None,
    ) -> LoweringResult:
        """Lower a Qiskit circuit into Program IR.

        Args:
            returned_value: Expected Qiskit ``QuantumCircuit``.
            source_metadata: Candidate source/backend diagnostics.
            contract: Optional task contract used only for declared limits.

        Returns:
            Program IR or typed non-verdict failure.
        """
        del contract
        if not _is_quantum_circuit(returned_value):
            return _failure(LoweringStatus.EXECUTION_ERROR, "invalid_return_type", detail=type(returned_value).__name__)
        try:
            program = _lower_circuit(returned_value, source_metadata)
            validate_program(program)
        except _Unsupported as exc:
            return _failure(
                LoweringStatus.UNSUPPORTED,
                exc.reason,
                node_kind=exc.node_kind,
                source_location=exc.location,
            )
        except (MemoryError, RecursionError) as exc:
            return _failure(LoweringStatus.RESOURCE_LIMIT, "lowering_resource_limit", detail=type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - inspection failures are structured result data.
            return _failure(LoweringStatus.EXECUTION_ERROR, "framework_inspection_failed", detail=type(exc).__name__)
        return LoweringResult(LoweringStatus.SUCCESS, program=program)

    def capabilities(self) -> CapabilitySet:
        """Return Qiskit lowering feature claims."""
        return CapabilitySet(
            (
                "static_gates",
                "custom_definitions",
                "terminal_measurement",
                "mid_measurement",
                "reset",
                "classical_condition",
                "global_phase",
            )
        )

    def framework_fingerprint(self) -> FrameworkFingerprint:
        """Return the installed Qiskit version."""
        return FrameworkFingerprint("qiskit", importlib.metadata.version("qiskit"))


def _lower_circuit(circuit: Any, metadata: SourceMetadata) -> Program:
    operations = tuple(
        operation
        for index, instruction in enumerate(circuit.data)
        for operation in _lower_instructions(circuit, instruction, index)
    )
    render_order = tuple(reversed(range(circuit.num_clbits)))
    return Program(
        ir_version=IR_VERSION,
        num_qubits=int(circuit.num_qubits),
        num_clbits=int(circuit.num_clbits),
        operations=operations,
        global_phase=_parameter(circuit.global_phase),
        classical_render_order=render_order,
        provenance=Provenance(
            framework="qiskit",
            framework_version=importlib.metadata.version("qiskit"),
            source_hash=metadata.source_hash,
            backend=metadata.backend,
        ),
    )


def _lower_instructions(circuit: Any, instruction: Any, index: int) -> tuple[Operation, ...]:
    operation = instruction.operation
    name = str(operation.name).lower()
    location = f"circuit.data[{index}]"
    if name != "if_else":
        return (_lower_instruction(circuit, instruction, index),)
    blocks = tuple(getattr(operation, "blocks", ()))
    if len(blocks) != 1:
        raise _Unsupported("unsupported_if_else_shape", name, location)
    condition = _condition(circuit, operation)
    if condition is None:
        raise _Unsupported("unsupported_if_else_condition", name, location)
    parent_qubits = tuple(circuit.find_bit(bit).index for bit in instruction.qubits)
    parent_clbits = tuple(circuit.find_bit(bit).index for bit in instruction.clbits)
    block = blocks[0]
    if len(block.qubits) != len(parent_qubits) or len(block.clbits) != len(parent_clbits):
        raise _Unsupported("unsupported_if_else_mapping", name, location)
    values = []
    for child_index, child in enumerate(block.data):
        lowered = _lower_instruction(block, child, child_index)
        if lowered.condition is not None:
            raise _Unsupported("nested_classical_condition", name, location)
        values.append(
            replace(
                lowered,
                quantum_wires=tuple(parent_qubits[wire] for wire in lowered.quantum_wires),
                classical_bits=tuple(parent_clbits[bit] for bit in lowered.classical_bits),
                controls=tuple(replace(control, wire=parent_qubits[control.wire]) for control in lowered.controls),
                condition=condition,
                source_location=f"{location}.true_body[{child_index}]",
            )
        )
    return tuple(values)


def _lower_instruction(circuit: Any, instruction: Any, index: int) -> Operation:
    operation = instruction.operation
    name = _operation_name(operation)
    location = f"circuit.data[{index}]"
    if name in _CONTROL_FLOW:
        raise _Unsupported("unsupported_control_flow", name, location)
    if name in _OPAQUE_RUNTIME:
        raise _Unsupported("unsupported_opaque_runtime", name, location)
    qubits = tuple(circuit.find_bit(bit).index for bit in instruction.qubits)
    clbits = tuple(circuit.find_bit(bit).index for bit in instruction.clbits)
    special = _lower_special_instruction(operation, name, qubits, clbits, location)
    if special is not None:
        return special
    return _lower_gate_instruction(circuit, operation, name, qubits, clbits, location)


def _lower_special_instruction(
    operation: Any,
    name: str,
    qubits: tuple[int, ...],
    clbits: tuple[int, ...],
    location: str,
) -> Operation | None:
    if name == "measure":
        return Operation(OperationKind.MEASUREMENT, name, qubits, clbits, source_location=location)
    if name == "reset":
        return Operation(OperationKind.RESET, name, qubits, source_location=location)
    if name == "barrier":
        return Operation(OperationKind.BARRIER, name, qubits, source_location=location)
    if name in {"initialize", "state_preparation", "set_statevector"}:
        return Operation(
            OperationKind.STATE_PREPARATION,
            "statevector",
            quantum_wires=qubits,
            semantic_data=bounded_statevector_semantic_data(operation.params),
            source_location=location,
        )
    return None


def _lower_gate_instruction(
    circuit: Any,
    operation: Any,
    name: str,
    qubits: tuple[int, ...],
    clbits: tuple[int, ...],
    location: str,
) -> Operation:
    kind = OperationKind.STATE_PREPARATION if name in _STATE_PREPARATION else OperationKind.GATE
    dense = _bounded_custom_matrix(operation, location)
    if dense is not None:
        return Operation(
            kind,
            "dense_unitary",
            quantum_wires=qubits,
            semantic_data=dense,
            source_location=location,
        )
    controls, targets = _controls(operation, qubits)
    definition = _definition(operation)
    if name == "qft":
        # The shared named-QFT primitive uses the first listed wire as the most
        # significant DFT bit, while Qiskit's QFT treats q0 as least
        # significant.  Reverse only genuine library QFTGate wires.  Other
        # operations named qft must lower through an inspectable definition.
        if _is_trusted_qft_gate(operation) and not controls:
            return Operation(
                kind=kind,
                name="qft",
                quantum_wires=tuple(reversed(targets)),
                classical_bits=clbits,
                condition=_condition(circuit, operation),
                source_location=location,
            )
        if not definition:
            raise _Unsupported("unsupported_named_qft", type(operation).__name__, location)
        name = "qft_definition"
    return Operation(
        kind=kind,
        name=name,
        quantum_wires=targets,
        classical_bits=clbits,
        parameters=tuple(_parameter(value) for value in operation.params),
        controls=controls,
        condition=_condition(circuit, operation),
        definition=definition,
        source_location=location,
    )


def _bounded_custom_matrix(operation: Any, location: str) -> tuple[tuple[str, str], ...] | None:
    base_gate = getattr(operation, "base_gate", None)
    # Qiskit exposes an internal StandardGate identity for built-in gates.  A
    # user-defined Gate may choose any display name (including "cx"), so its
    # name and definition are not an authority for behavior.  Lower every
    # non-standard unitary from Qiskit's independent Operator semantics.
    built_in_standard = str(type(operation).__module__).startswith("qiskit.circuit.library.standard_gates.")
    custom = (
        (getattr(operation, "_standard_gate", None) is None and not built_in_standard)
        or type(operation).__name__ == "Instruction"
        or type(base_gate).__name__ == "UnitaryGate"
        or str(operation.name).lower() in {"rxx", "ryy", "rzz", "rzx"}
    )
    if not custom:
        return None
    if int(operation.num_qubits) > 6:
        definition = getattr(operation, "definition", None)
        # Large library composites such as an eight-qubit inverse QFT cannot
        # be represented within the bounded dense payload. Preserve their
        # inspectable definition instead; genuinely opaque large gates remain
        # unsupported.
        if definition is not None and tuple(getattr(definition, "data", ())):
            return None
        raise _Unsupported(
            "custom_unitary_exceeds_dense_limit",
            type(operation).__name__,
            location,
        )
    try:
        from qiskit.quantum_info import Operator

        matrix = Operator(operation).data
        payload = bounded_matrix_semantic_data(
            matrix,
            max_dimension=64,
            wire_order="little_endian",
        )
        if _has_matching_gate_definition(operation, matrix):
            payload += (("matrix_origin", "qiskit_gate_definition"),)
        return payload
    except Exception as exc:  # noqa: BLE001 - custom behavior must fail closed.
        raise _Unsupported("unsupported_custom_unitary", type(operation).__name__, location) from exc


def _has_matching_gate_definition(operation: Any, matrix: Any) -> bool:
    """Return whether a custom matrix is backed by the same gate-built circuit."""
    if not _has_inspectable_gate_definition(operation, set()):
        return False
    definition = operation.definition
    try:
        import numpy as np
        from qiskit.quantum_info import Operator

        return bool(np.allclose(Operator(definition).data, matrix, atol=1e-12, rtol=0.0))
    except Exception:  # noqa: BLE001 - provenance is optional and must fail closed.
        return False


def _has_inspectable_gate_definition(operation: Any, active: set[int]) -> bool:
    """Return whether a custom gate recursively decomposes to standard gates."""
    base_gate = getattr(operation, "base_gate", None)
    if type(operation).__name__ == "UnitaryGate" or type(base_gate).__name__ == "UnitaryGate":
        return False
    identity = id(operation)
    if identity in active:
        return False
    definition = getattr(operation, "definition", None)
    instructions = tuple(getattr(definition, "data", ())) if definition is not None else ()
    if not instructions:
        return False
    active.add(identity)
    try:
        for instruction in instructions:
            child = instruction.operation
            child_base = getattr(child, "base_gate", None)
            if type(child).__name__ == "UnitaryGate" or type(child_base).__name__ == "UnitaryGate":
                return False
            built_in_standard = str(type(child).__module__).startswith("qiskit.circuit.library.standard_gates.")
            if getattr(child, "_standard_gate", None) is not None or built_in_standard:
                continue
            if not _has_inspectable_gate_definition(child, active):
                return False
        return True
    except Exception:  # noqa: BLE001 - provenance is optional and must fail closed.
        return False
    finally:
        active.remove(identity)


def _is_trusted_qft_gate(operation: Any) -> bool:
    """Return whether an operation is Qiskit's own library ``QFTGate``."""
    return (
        type(operation).__name__ == "QFTGate"
        and str(type(operation).__module__).startswith("qiskit.circuit.library")
        and int(getattr(operation, "num_ctrl_qubits", 0) or 0) == 0
    )


def _operation_name(operation: Any) -> str:
    """Return a behavior-derived name for trusted Qiskit standard gates."""
    standard = getattr(operation, "_standard_gate", None)
    standard_name = getattr(standard, "name", None)
    return str(standard_name if standard_name is not None else operation.name).lower()


def _controls(operation: Any, qubits: tuple[int, ...]) -> tuple[tuple[Control, ...], tuple[int, ...]]:
    count = int(getattr(operation, "num_ctrl_qubits", 0) or 0)
    if count <= 0:
        return (), qubits
    state = int(getattr(operation, "ctrl_state", (1 << count) - 1))
    controls = tuple(Control(wire, (state >> index) & 1) for index, wire in enumerate(qubits[:count]))
    return controls, qubits[count:]


def _definition(operation: Any) -> tuple[Operation, ...]:
    definition = getattr(operation, "definition", None)
    if definition is None or str(operation.name).lower() in {"u", "u3", "cx", "ccx", "cz", "swap"}:
        return ()
    return tuple(
        _lower_instruction(definition, instruction, index) for index, instruction in enumerate(definition.data)
    )


def _condition(circuit: Any, operation: Any) -> ClassicalCondition | None:
    condition = getattr(operation, "condition", None)
    if condition is None:
        return None
    register_or_bit, value = condition
    try:
        bits = tuple(circuit.find_bit(bit).index for bit in register_or_bit)
    except TypeError:
        bits = (circuit.find_bit(register_or_bit).index,)
    return ClassicalCondition(bits, int(value))


def _parameter(value: Any) -> Parameter:
    if isinstance(value, bool):
        return Parameter(ParameterKind.TEXT, str(value).lower())
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("operation parameter must be finite")
        text = "0" if numeric == 0 else format(numeric, ".17g").lower()
        return Parameter(ParameterKind.NUMBER, text)
    parameters = getattr(value, "parameters", ())
    if parameters:
        return Parameter(ParameterKind.SYMBOL, str(value))
    try:
        complex_value = complex(value)
    except (TypeError, ValueError):
        return Parameter(ParameterKind.TEXT, str(value))
    if abs(complex_value.imag) <= 1e-15:
        text = "0" if complex_value.real == 0 else format(complex_value.real, ".17g").lower()
        return Parameter(ParameterKind.NUMBER, text)
    return Parameter(ParameterKind.TEXT, f"{complex_value.real:.17g}{complex_value.imag:+.17g}j")


def _is_quantum_circuit(value: Any) -> bool:
    return value.__class__.__module__.startswith("qiskit") and value.__class__.__name__ == "QuantumCircuit"


def _failure(
    status: LoweringStatus,
    reason: str,
    *,
    node_kind: str | None = None,
    source_location: str | None = None,
    detail: str | None = None,
) -> LoweringResult:
    return LoweringResult(
        status,
        error=LoweringError(reason, node_kind=node_kind, source_location=source_location, detail=detail),
    )


class _Unsupported(Exception):
    def __init__(self, reason: str, node_kind: str, location: str) -> None:
        self.reason = reason
        self.node_kind = node_kind
        self.location = location
        super().__init__(reason)
