"""Cirq native operation lowering onto Program IR operations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from qceval.frameworks.cirq.wire_map import _Unsupported
from qceval.semantics.ir import (
    ClassicalCondition,
    Control,
    Operation,
    OperationKind,
    Parameter,
)
from qceval.semantics.lowering.utils import (
    bounded_matrix_semantic_data,
    matrix_sha256,
    normalize_parameter,
)

_DENSE_TWO_QUBIT_GATES = frozenset({"XXPowGate", "YYPowGate", "ZZPowGate", "ISwapPowGate", "PhasedISwapPowGate"})

_ALIASES = {
    "HPowGate": "h",
    "XPowGate": "x",
    "YPowGate": "y",
    "ZPowGate": "z",
    "CXPowGate": "cx",
    "CZPowGate": "cz",
    "CCXPowGate": "ccx",
    "CCZPowGate": "z",
    "SwapPowGate": "swap",
    "CSwapGate": "cswap",
}


def _classical_condition(
    native: Any,
    key_bits: dict[str, tuple[int, ...]],
    location: str,
) -> ClassicalCondition:
    """Map Cirq key conditions onto an equality over normalized classical bits.

    Only conjunctions of single-bit ``KeyCondition`` truthiness tests are
    expressible as one IR equality; sympy conditions and multi-bit keys stay
    typed-unsupported.

    Args:
        native: Cirq classically controlled operation.
        key_bits: Measurement key to allocated classical-bit mapping.
        location: Source location string for unsupported diagnostics.

    Returns:
        Classical equality condition over previously measured bits.
    """
    import cirq

    bits: list[int] = []
    for control in native.classical_controls:
        if not isinstance(control, cirq.KeyCondition):
            raise _Unsupported("unsupported_classical_condition", type(control).__name__, location)
        key = str(control.key)
        measured = key_bits.get(key)
        if measured is None:
            raise _Unsupported("unsupported_classical_condition", f"unmeasured_key:{key}", location)
        if len(measured) != 1:
            raise _Unsupported("unsupported_classical_condition", f"multi_bit_key:{key}", location)
        bits.append(measured[0])
    if not bits:
        raise _Unsupported("unsupported_classical_condition", "empty_condition", location)
    # KeyCondition truthiness on a single bit means the bit equals 1; a
    # conjunction requires every listed bit to be 1.
    return ClassicalCondition(tuple(bits), (1 << len(bits)) - 1)


def _lower_native(
    native: Any,
    index: int,
    wire_map: dict[Any, int],
    key_bits: dict[str, tuple[int, ...]],
    next_clbit: int,
) -> tuple[list[Operation], int, float]:
    """Lower one Cirq operation into Program IR operations.

    Args:
        native: Cirq operation from ``circuit.all_operations()``.
        index: Operation index used for source locations.
        wire_map: Qubit object to IR wire mapping.
        key_bits: Mutable measurement-key classical-bit registry.
        next_clbit: Next unused classical-bit index.

    Returns:
        Lowered operations, updated classical-bit cursor, and global phase.
    """
    import cirq

    location = f"operation[{index}]"
    if isinstance(native.gate, cirq.GlobalPhaseGate):
        return [], next_clbit, float(np.angle(complex(native.gate.coefficient)))
    if isinstance(native, cirq.ClassicallyControlledOperation):
        condition = _classical_condition(native, key_bits, location)
        inner, next_clbit, phase = _lower_native(
            native.without_classical_controls(),
            index,
            wire_map,
            key_bits,
            next_clbit,
        )
        conditioned = []
        for operation in inner:
            if operation.condition is not None or operation.kind is not OperationKind.GATE:
                raise _Unsupported("unsupported_classical_condition", "nested_or_non_gate", location)
            conditioned.append(replace(operation, condition=condition))
        if phase:
            raise _Unsupported("unsupported_classical_condition", "conditioned_global_phase", location)
        return conditioned, next_clbit, 0.0
    if isinstance(native.gate, cirq.MeasurementGate):
        key = cirq.measurement_key_name(native)
        wires = tuple(wire_map[item] for item in native.qubits)
        bits = tuple(range(next_clbit, next_clbit + len(wires)))
        key_bits[key] = (*key_bits.get(key, ()), *bits)
        invert = "".join("1" if value else "0" for value in native.gate.full_invert_mask())
        operation = Operation(
            OperationKind.MEASUREMENT,
            "measure",
            quantum_wires=wires,
            classical_bits=bits,
            semantic_data=(("invert_mask", invert), ("key", key)),
            source_location=location,
        )
        return [operation], next_clbit + len(wires), 0.0
    if isinstance(native.gate, cirq.ResetChannel):
        resets = [
            Operation(
                OperationKind.RESET,
                "reset",
                quantum_wires=(wire_map[qubit],),
                source_location=location,
            )
            for qubit in native.qubits
        ]
        return resets, next_clbit, 0.0
    if not cirq.has_unitary(native):
        raise _Unsupported("unsupported_channel_or_mixture", type(native.gate).__name__, location)
    return [_gate_operation(native, wire_map, location)], next_clbit, 0.0


def _gate_operation(native: Any, wire_map: dict[Any, int], location: str) -> Operation:
    import cirq

    gate = native.gate
    qubits = tuple(wire_map[item] for item in native.qubits)
    type_name = type(gate).__name__
    matrix = cirq.unitary(native)
    # Custom Gate implementations control their __str__ output.  Never let a
    # user-supplied label such as "cx" select built-in IR semantics; preserve
    # the matrix Cirq actually executed instead.
    if not type(gate).__module__.startswith("cirq."):
        return Operation(
            OperationKind.GATE,
            "dense_unitary",
            quantum_wires=qubits,
            semantic_data=bounded_matrix_semantic_data(matrix),
            source_location=location,
        )
    if type_name == "ControlledGate":
        controlled = _controlled_gate_operation(gate, qubits, matrix, location)
        if controlled is not None:
            return controlled
    if type_name in {"ControlledGate", "MatrixGate", *_DENSE_TWO_QUBIT_GATES}:
        return Operation(
            OperationKind.GATE,
            "dense_unitary",
            quantum_wires=qubits,
            semantic_data=bounded_matrix_semantic_data(matrix),
            source_location=location,
        )
    name = _ALIASES.get(type_name, str(gate).split("(", maxsplit=1)[0].lower())
    inverse = type_name == "_InverseCompositeGate" and name == "qft†"
    if inverse:
        name = "qft"
    control_count = _control_count(type_name, gate)
    controls = tuple(Control(wire) for wire in qubits[:control_count])
    targets = qubits[control_count:]
    exponent = getattr(gate, "exponent", 1)
    parameters: tuple[Parameter, ...]
    if type_name in {"Rx", "Ry", "Rz"}:
        parameters = (normalize_parameter(float(exponent) * np.pi),)
    elif exponent == 1:
        parameters = ()
    else:
        parameters = (normalize_parameter(exponent),)
        name = {
            "XPowGate": "xpow",
            "YPowGate": "ypow",
            "ZPowGate": "zpow",
            "HPowGate": "hpow",
            "CXPowGate": "xpow",
            "CZPowGate": "zpow",
        }.get(type_name, name)
    return Operation(
        OperationKind.GATE,
        name,
        quantum_wires=targets,
        parameters=parameters,
        controls=controls,
        semantic_data=(("matrix_sha256", matrix_sha256(matrix)),),
        inverse=inverse,
        source_location=location,
    )


def _controlled_gate_operation(
    gate: Any,
    qubits: tuple[int, ...],
    matrix: np.ndarray,
    location: str,
) -> Operation | None:
    import cirq

    control_count = int(gate.num_controls())
    values = tuple(tuple(item) for item in gate.control_values)
    if len(values) != control_count or any(len(item) != 1 or item[0] not in {0, 1} for item in values):
        return None
    base = gate.sub_gate
    controls = tuple(Control(wire, value[0]) for wire, value in zip(qubits[:control_count], values, strict=True))
    named = _named_controlled_base(base)
    if named is None:
        if type(base).__name__ != "MatrixGate":
            return None
        return Operation(
            OperationKind.GATE,
            "dense_unitary",
            quantum_wires=qubits[control_count:],
            controls=controls,
            semantic_data=bounded_matrix_semantic_data(cirq.unitary(base)),
            source_location=location,
        )
    name, parameters = named
    return Operation(
        OperationKind.GATE,
        name,
        quantum_wires=qubits[control_count:],
        parameters=parameters,
        controls=controls,
        semantic_data=(("matrix_sha256", matrix_sha256(matrix)),),
        source_location=location,
    )


def _named_controlled_base(base: Any) -> tuple[str, tuple[Parameter, ...]] | None:
    """Return the named IR lowering of a supported controlled sub-gate.

    Pauli/Hadamard powers keep their exact exponent so multi-controlled phase
    constructions (for example phase-matched Grover oracles) lower onto the
    named ``*pow`` gates instead of one full-register dense matrix.
    """
    import cirq

    exponent = float(getattr(base, "exponent", 1.0))
    if type(base).__name__ in {"Rx", "Ry", "Rz"}:
        return type(base).__name__.lower(), (normalize_parameter(exponent * np.pi),)
    pow_names = (
        (cirq.XPowGate, "x", "xpow"),
        (cirq.YPowGate, "y", "ypow"),
        (cirq.ZPowGate, "z", "zpow"),
        (cirq.HPowGate, "h", "hpow"),
    )
    if float(getattr(base, "global_shift", 0.0)) != 0.0:
        return None
    for gate_class, plain, powered in pow_names:
        if isinstance(base, gate_class):
            return (plain, ()) if exponent == 1.0 else (powered, (normalize_parameter(exponent),))
    return None


def _control_count(type_name: str, gate: Any) -> int:
    if type_name in {"CXPowGate", "CZPowGate", "CSwapGate"}:
        return 1
    if type_name in {"CCXPowGate", "CCZPowGate"}:
        return 2
    return int(getattr(gate, "num_controls", 0) or 0)
