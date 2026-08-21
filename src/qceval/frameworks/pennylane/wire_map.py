"""PennyLane tape wire layout and contract output-wire helpers."""

from __future__ import annotations

from typing import Any

from qceval.semantics.contracts import Contract
from qceval.semantics.contracts.kinds import FrozenArray, FrozenObject


class _Unsupported(Exception):
    """Typed PennyLane lowering limitation that becomes a non-verdict failure."""

    def __init__(self, reason: str, node_kind: str, location: str) -> None:
        self.reason = reason
        self.node_kind = node_kind
        self.location = location
        super().__init__(reason)


def _tape_wire_labels(tape: Any, contract: Contract | None) -> list[int]:
    """Collect and validate explicit integer wire labels from a tape.

    Args:
        tape: Captured PennyLane tape or ``QuantumScript``.
        contract: Optional contract whose qubit limit applies.

    Returns:
        Sorted unique wire labels used by operations and measurements.

    Raises:
        _Unsupported: When wires are non-integral or exceed the contract.
    """
    try:
        labels = sorted({int(wire) for item in (*tape.operations, *tape.measurements) for wire in item.wires})
    except (TypeError, ValueError) as exc:
        raise _Unsupported("explicit_wire_layout_required", "wire", "tape.wires") from exc
    maximum = contract.limits.max_qubits if contract is not None else None
    if labels and (min(labels) < 0 or maximum is not None and max(labels) >= maximum):
        raise _Unsupported("wire_layout_exceeds_contract", "wire", "tape.wires")
    return labels


def _contract_output_wires(contract: Contract | None) -> tuple[int, ...] | None:
    """Return PennyLane output wires declared by a contract interface.

    Args:
        contract: Optional semantic contract.

    Returns:
        Declared PennyLane output wires, or ``None`` when unspecified.
    """
    if contract is None:
        return None
    for requirement in contract.requirements:
        if requirement.requirement_id != "terminal_observation":
            continue
        value = _plain_requirement(requirement.value)
        if not isinstance(value, dict):
            return None
        interface = value.get("pennylane")
        if not isinstance(interface, dict):
            return None
        wires = interface.get("wires", interface.get("qubits"))
        if isinstance(wires, list) and all(isinstance(wire, int) for wire in wires):
            return tuple(wires)
    return None


def _plain_requirement(value: Any) -> Any:
    """Convert immutable contract containers into ordinary Python values.

    Args:
        value: Frozen or plain contract requirement payload.

    Returns:
        Nested dict/list form suitable for ordinary Python lookups.
    """
    if isinstance(value, FrozenObject):
        return {key: _plain_requirement(item) for key, item in value.items}
    if isinstance(value, FrozenArray):
        return [_plain_requirement(item) for item in value.items]
    return value
