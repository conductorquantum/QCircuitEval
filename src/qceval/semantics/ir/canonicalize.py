"""Canonical finite JSON representation for Program IR."""

from __future__ import annotations

from typing import Any

from qceval.semantics.ir.model import Operation, OperationKind, Parameter, Program
from qceval.semantics.ir.validation import validate_program

_GATE_ALIASES = {
    "cnot": "cx",
    "controlled_x": "cx",
    "controlled_not": "cx",
    "toffoli": "ccx",
    "phase_shift": "p",
}


def canonical_program_dict(program: Program) -> dict[str, Any]:
    """Return semantic Program IR data with diagnostics removed.

    Args:
        program: Validated Program IR.

    Returns:
        JSON-compatible canonical semantic mapping.
    """
    validate_program(program)
    return {
        "ir_version": program.ir_version,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "global_phase": _parameter(program.global_phase),
        "classical_render_order": list(program.classical_render_order),
        "operations": [
            _operation(operation) for operation in program.operations if operation.kind is not OperationKind.BARRIER
        ],
    }


def _operation(operation: Operation) -> dict[str, Any]:
    name = _GATE_ALIASES.get(operation.name.lower(), operation.name.lower())
    controls = sorted(operation.controls, key=lambda control: control.wire)
    return {
        "kind": operation.kind.value,
        "name": name,
        "quantum_wires": list(operation.quantum_wires),
        "classical_bits": list(operation.classical_bits),
        "parameters": [_parameter(item) for item in operation.parameters],
        "controls": [{"wire": item.wire, "value": item.value} for item in controls],
        "condition": None
        if operation.condition is None
        else {"bits": list(operation.condition.bits), "value": operation.condition.value},
        "inverse": operation.inverse,
        "power": _parameter(operation.power),
        "definition": [_operation(item) for item in operation.definition],
        "semantic_data": dict(sorted(operation.semantic_data)),
    }


def _parameter(parameter: Parameter | None) -> dict[str, str] | None:
    if parameter is None:
        return None
    return {"kind": parameter.kind.value, "value": parameter.value}
