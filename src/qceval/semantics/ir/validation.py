"""Dimensional, wire, control, and lifecycle validation for Program IR."""

from __future__ import annotations

import re
from dataclasses import dataclass

from qceval.semantics.ir.model import IR_VERSION, Operation, OperationKind, ParameterKind, Program

_FINITE_NUMBER = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[+-]?\d+)?$")


class IRValidationError(ValueError):
    """A stable path-addressed Program IR validation failure."""

    def __init__(self, path: str, reason: str) -> None:
        """Initialize a validation failure.

        Args:
            path: Program node path.
            reason: Stable failure reason.
        """
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class IRValidationLimits:
    """Pre-materialization Program IR resource limits."""

    max_qubits: int = 64
    max_clbits: int = 4096
    max_operations: int = 100000
    max_definition_depth: int = 32


def validate_program(program: Program, limits: IRValidationLimits | None = None) -> None:
    """Validate one complete Program IR or raise.

    Args:
        program: Program to validate.
        limits: Deterministic structural resource limits.

    Raises:
        IRValidationError: If an invariant or limit is violated.
    """
    limits = limits or IRValidationLimits()
    if program.ir_version != IR_VERSION:
        raise IRValidationError("$.ir_version", f"unsupported version {program.ir_version!r}")
    if program.num_qubits < 0 or program.num_qubits > limits.max_qubits:
        raise IRValidationError("$.num_qubits", "outside configured limit")
    if program.num_clbits < 0 or program.num_clbits > limits.max_clbits:
        raise IRValidationError("$.num_clbits", "outside configured limit")
    if len(program.operations) > limits.max_operations:
        raise IRValidationError("$.operations", "operation limit exceeded")
    if sorted(program.classical_render_order) != list(range(program.num_clbits)):
        raise IRValidationError("$.classical_render_order", "must be a permutation of all classical bits")
    if program.global_phase is not None:
        _validate_parameter(program.global_phase, "$.global_phase")
    for index, operation in enumerate(program.operations):
        _validate_operation(operation, f"$.operations[{index}]", program, limits, depth=0)


def _validate_operation(
    operation: Operation,
    path: str,
    program: Program,
    limits: IRValidationLimits,
    *,
    depth: int,
) -> None:
    if not operation.name:
        raise IRValidationError(f"{path}.name", "must not be empty")
    _indices(operation.quantum_wires, program.num_qubits, f"{path}.quantum_wires")
    _indices(operation.classical_bits, program.num_clbits, f"{path}.classical_bits")
    control_wires = tuple(control.wire for control in operation.controls)
    _indices(control_wires, program.num_qubits, f"{path}.controls")
    if set(control_wires) & set(operation.quantum_wires):
        raise IRValidationError(f"{path}.controls", "controls overlap target wires")
    if any(control.value not in (0, 1) for control in operation.controls):
        raise IRValidationError(f"{path}.controls", "control values must be zero or one")
    _validate_condition(operation, path, program)
    for index, parameter in enumerate(operation.parameters):
        _validate_parameter(parameter, f"{path}.parameters[{index}]")
    if operation.power is not None:
        _validate_parameter(operation.power, f"{path}.power")
    _validate_kind_shape(operation, path)
    _validate_definition(operation, path, program, limits, depth)


def _validate_condition(operation: Operation, path: str, program: Program) -> None:
    if operation.condition is None:
        return
    _indices(operation.condition.bits, program.num_clbits, f"{path}.condition.bits")
    if operation.condition.value < 0:
        raise IRValidationError(f"{path}.condition.value", "must be non-negative")


def _validate_definition(
    operation: Operation,
    path: str,
    program: Program,
    limits: IRValidationLimits,
    depth: int,
) -> None:
    if depth >= limits.max_definition_depth and operation.definition:
        raise IRValidationError(f"{path}.definition", "definition depth limit exceeded")
    for index, nested in enumerate(operation.definition):
        _validate_operation(nested, f"{path}.definition[{index}]", program, limits, depth=depth + 1)


def _validate_kind_shape(operation: Operation, path: str) -> None:
    if operation.kind is OperationKind.GATE and not (*operation.controls, *operation.quantum_wires):
        raise IRValidationError(f"{path}.quantum_wires", "gate requires a quantum wire")
    if operation.kind is OperationKind.MEASUREMENT and (
        not operation.quantum_wires or len(operation.quantum_wires) != len(operation.classical_bits)
    ):
        raise IRValidationError(path, "measurement requires equal nonempty quantum/classical mappings")
    if operation.kind is OperationKind.RESET and (len(operation.quantum_wires) != 1 or operation.classical_bits):
        raise IRValidationError(path, "reset requires exactly one quantum wire")
    if operation.kind is OperationKind.OPAQUE and not operation.semantic_data:
        raise IRValidationError(f"{path}.semantic_data", "opaque operation requires a semantic provider")


def _validate_parameter(parameter: object, path: str) -> None:
    if not hasattr(parameter, "kind") or not hasattr(parameter, "value"):
        raise IRValidationError(path, "must be a Parameter")
    value = str(parameter.value)
    if not value:
        raise IRValidationError(path, "value must not be empty")
    if parameter.kind is ParameterKind.NUMBER and not _FINITE_NUMBER.fullmatch(value):
        raise IRValidationError(path, "number must be finite canonical decimal text")


def _indices(values: tuple[int, ...], width: int, path: str) -> None:
    if len(set(values)) != len(values):
        raise IRValidationError(path, "indices must be unique")
    if any(isinstance(value, bool) or value < 0 or value >= width for value in values):
        raise IRValidationError(path, "index outside register")
