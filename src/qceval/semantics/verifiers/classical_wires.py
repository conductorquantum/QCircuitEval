"""Classical output-wire selection shared by exact semantic materializers."""

from __future__ import annotations

from typing import Any, TypeGuard

from qceval.semantics.contracts.kinds import BitOrder, Contract, FrozenArray, FrozenObject
from qceval.semantics.ir import OperationKind, Program


def measured_render_order(program: Program) -> tuple[int, ...]:
    """Return rendered classical bits that are actually written by measurement.

    Framework programs may allocate unused classical bits. Those storage slots
    are not part of the observed result and must not widen or shift the public
    bitstring.

    Args:
        program: Lowered framework-neutral program.

    Returns:
        Actually measured classical bits in public render order.
    """
    measured = {
        bit
        for operation in program.operations
        if operation.kind is OperationKind.MEASUREMENT
        for bit in operation.classical_bits
    }
    return tuple(bit for bit in program.classical_render_order if bit in measured)


def rendered_quantum_wires(program: Program) -> tuple[int, ...]:
    """Return measured quantum wires in the public classical render order.

    Args:
        program: Lowered framework-neutral program.

    Returns:
        Rendered quantum wires, or an empty tuple for ambiguous mappings.
    """
    bit_to_wire: dict[int, int] = {}
    for operation in program.operations:
        if operation.kind is not OperationKind.MEASUREMENT:
            continue
        if len(operation.quantum_wires) != len(operation.classical_bits):
            return ()
        for wire, bit in zip(operation.quantum_wires, operation.classical_bits, strict=True):
            if bit in bit_to_wire and bit_to_wire[bit] != wire:
                return ()
            bit_to_wire[bit] = wire
    return tuple(bit_to_wire[bit] for bit in program.classical_render_order if bit in bit_to_wire)


def terminal_wire_permutation(contract: Contract, program: Program) -> dict[int, int] | None:
    """Map contract-declared output wires to the candidate's role-equivalent wires.

    ``register_roles`` interfaces pin register roles and render order but not
    physical qubit placement, so the candidate wire rendered at each position
    plays the role the contract assigns to the declared wire at that position.

    Args:
        contract: Behavior contract with a terminal-observation interface.
        program: Lowered candidate program.

    Returns:
        Declared-to-candidate wire mapping, or ``None`` when the interface
        pins the physical layout or no unambiguous mapping exists.
    """
    framework = program.provenance.framework
    interface = _terminal_interface(contract, framework)
    if not isinstance(interface, dict) or interface.get("layout") != "register_roles":
        return None
    declared = _terminal_render_wires(contract, framework)
    actual = rendered_quantum_wires(program)
    if declared is None or len(actual) != len(declared) or len(set(actual)) != len(actual):
        return None
    return dict(zip(declared, actual, strict=True))


def measurement_clbits_by_qubit(program: Program) -> dict[int, int]:
    """Return each measured quantum wire's final classical destination.

    A later measurement of the same quantum wire is the score-authoritative
    write, matching the final-record semantics used by framework executors.

    Args:
        program: Lowered framework-neutral program.

    Returns:
        Mapping from measured quantum wires to final classical destinations.
    """
    bindings: dict[int, int] = {}
    for operation in program.operations:
        if operation.kind is not OperationKind.MEASUREMENT:
            continue
        bindings.update(zip(operation.quantum_wires, operation.classical_bits, strict=True))
    return bindings


def contracted_classical_variables(
    contract: Contract,
    program: Program,
    names: tuple[str, ...],
    *,
    require_all: bool,
) -> tuple[tuple[str, int], ...]:
    """Bind contract-named classical variables to measured Program-IR bits.

    The prompt-derived terminal interface identifies physical output wires;
    contracts without one fall back to system indices. The contract's
    bit-order policy determines public order, independent of candidate
    classical storage or render order.

    Args:
        contract: Behavior contract defining logical classical variables.
        program: Lowered candidate program.
        names: Contract system names to bind.
        require_all: Whether missing measured outputs are an error.

    Returns:
        Logical variable names paired with candidate classical-bit indices.
    """
    systems = {item.name: item for item in contract.systems.items}
    measured = measurement_clbits_by_qubit(program)
    interface_bindings = _interface_quantum_wires(contract, program)
    values: list[tuple[str, int]] = []
    missing: list[str] = []
    for name in names:
        system = systems[name]
        indices = system.indices
        if contract.observation.bit_order is BitOrder.LITTLE_ENDIAN:
            indices = tuple(reversed(indices))
        for index in indices:
            variable = f"{name}[{index}]"
            quantum_wire = interface_bindings.get(variable, index)
            if quantum_wire not in measured:
                missing.append(variable)
                continue
            values.append((variable, measured[quantum_wire]))
    if require_all and missing:
        raise NotImplementedError("contracted classical outputs are absent from Program IR")
    return tuple(values)


def _interface_quantum_wires(contract: Contract, program: Program) -> dict[str, int]:
    """Map logical contract variables to prompt-declared physical output wires."""
    variables = _contract_variables(contract, contract.observation.classical)
    wires = _terminal_render_wires(contract, program.provenance.framework)
    if wires is None or len(wires) != len(variables):
        return {}
    return dict(zip(variables, wires, strict=True))


def _terminal_interface(contract: Contract, framework: str) -> dict[str, Any] | None:
    for requirement in contract.requirements:
        if requirement.requirement_id != "terminal_observation":
            continue
        value = _plain(requirement.value)
        if isinstance(value, dict) and isinstance(value.get(framework), dict):
            return value[framework]
        return None
    return None


def _terminal_render_wires(contract: Contract, framework: str) -> tuple[int, ...] | None:
    interface = _terminal_interface(contract, framework)
    if interface is None:
        return None
    render_order = interface.get("render_order")
    if _integer_list(render_order):
        return tuple(render_order)
    wires = interface.get("wires")
    if _integer_list(wires):
        return tuple(wires)
    qubits = interface.get("qubits")
    if not _integer_list(qubits):
        return None
    classical_bits = interface.get("classical_bits")
    if framework == "qiskit" and _integer_list(classical_bits) and len(classical_bits) == len(qubits):
        pairs = sorted(zip(classical_bits, qubits, strict=True), reverse=True)
        return tuple(qubit for _, qubit in pairs)
    if contract.observation.bit_order is BitOrder.LITTLE_ENDIAN:
        return tuple(reversed(qubits))
    return tuple(qubits)


def _contract_variables(contract: Contract, names: tuple[str, ...]) -> tuple[str, ...]:
    systems = {item.name: item for item in contract.systems.items}
    variables: list[str] = []
    for name in names:
        indices = systems[name].indices
        if contract.observation.bit_order is BitOrder.LITTLE_ENDIAN:
            indices = tuple(reversed(indices))
        variables.extend(f"{name}[{index}]" for index in indices)
    return tuple(variables)


def _plain(value: Any) -> Any:
    if isinstance(value, FrozenArray):
        return [_plain(item) for item in value.items]
    if isinstance(value, FrozenObject):
        return {key: _plain(item) for key, item in value.items}
    return value


def _integer_list(value: Any) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value)


__all__ = [
    "contracted_classical_variables",
    "measured_render_order",
    "measurement_clbits_by_qubit",
    "rendered_quantum_wires",
    "terminal_wire_permutation",
]
