"""PennyLane tape operation and measurement lowering onto Program IR."""

from __future__ import annotations

from typing import Any

from qceval.frameworks.pennylane.wire_map import _Unsupported
from qceval.semantics.ir import (
    ClassicalCondition,
    Control,
    Operation,
    OperationKind,
)
from qceval.semantics.lowering.utils import (
    bounded_matrix_semantic_data,
    bounded_statevector_semantic_data,
    matrix_sha256,
    normalize_parameter,
)

_ALIASES = {
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "CNOT": "cx",
    "CZ": "cz",
    "Toffoli": "ccx",
    "SWAP": "swap",
    "CSWAP": "cswap",
    "PhaseShift": "p",
    "ControlledPhaseShift": "p",
    "QFT": "qft",
}
_STATE_PREPARATION = frozenset({"BasisState", "StatePrep", "QubitStateVector"})
_ISING_NAMES = {"IsingXX": "rxx", "IsingYY": "ryy", "IsingZZ": "rzz"}
_NAMED_CONTROLLED_BASES = frozenset({"x", "y", "z", "h", "s", "t", "sx", "rx", "ry", "rz", "p"})


def _measurement_value_bits(
    measurement: Any,
    measurement_ids: dict[int, int],
    index: int,
) -> tuple[int, ...] | None:
    """Resolve statistics over recorded measurement values onto their bits.

    ``qml.probs(op=[m1, m0])`` and ``qml.sample([m1, m0])`` report the bits
    already produced by mid-circuit measurements; they record no new
    measurement, only the classical output order.

    Args:
        measurement: Terminal PennyLane measurement process.
        measurement_ids: Object-id to classical-bit mapping for mid-measures.
        index: Measurement index used for source locations.

    Returns:
        Classical bits already recorded by mid-circuit measurements, or
        ``None`` when this measurement allocates new bits.
    """
    value = getattr(measurement, "mv", None)
    if value is None:
        return None
    values = value if isinstance(value, list | tuple) else [value]
    bits: list[int] = []
    for item in values:
        constituents = tuple(getattr(item, "measurements", ()))
        if len(constituents) != 1:
            raise _Unsupported(
                "unsupported_measurement_process", "composite_measurement_value", f"measurements[{index}]"
            )
        bit = measurement_ids.get(id(constituents[0]))
        if bit is None:
            raise _Unsupported(
                "unsupported_measurement_process", "unrecorded_measurement_value", f"measurements[{index}]"
            )
        bits.append(bit)
    return tuple(bits)


def _lower_tape_operation(
    native: Any,
    location: str,
    measured_bits: dict[int, int],
    next_clbit: int,
    measurement_ids: dict[int, int] | None = None,
) -> tuple[list[Operation], int]:
    """Lower one tape operation, including mid-circuit measurement.

    Args:
        native: PennyLane operation from ``tape.operations``.
        location: Source location string for diagnostics.
        measured_bits: Wire to classical-bit mapping for conditionals.
        next_clbit: Next unused classical-bit index.
        measurement_ids: Optional object-id registry for measurement values.

    Returns:
        Lowered operations and the updated classical-bit cursor.
    """
    type_name = type(native).__name__
    if "Conditional" in type_name:
        return [_conditional_operation(native, type_name, location, measured_bits)], next_clbit
    if "MidMeasure" not in type_name:
        return [_operation(native, location)], next_clbit
    wires = tuple(int(wire) for wire in native.wires)
    bits = tuple(range(next_clbit, next_clbit + len(wires)))
    measured_bits.update(dict(zip(wires, bits, strict=True)))
    if measurement_ids is not None and len(bits) == 1:
        measurement_ids[id(native)] = bits[0]
    lowered = [
        Operation(
            OperationKind.MEASUREMENT,
            "mid_measure",
            quantum_wires=wires,
            classical_bits=bits,
            source_location=location,
        )
    ]
    if getattr(native, "reset", False):
        lowered.extend(
            Operation(OperationKind.RESET, "reset", quantum_wires=(wire,), source_location=location) for wire in wires
        )
    return lowered, next_clbit + len(wires)


def _conditional_operation(
    native: Any,
    type_name: str,
    location: str,
    measured_bits: dict[int, int],
) -> Operation:
    measurement_value = native.hyperparameters.get("meas_val")
    measurements: Any = getattr(measurement_value, "measurements", ())
    if len(measurements) != 1 or len(measurements[0].wires) != 1:
        raise _Unsupported("unsupported_classical_condition", type_name, location)
    measured_wire = int(measurements[0].wires[0])
    if measured_wire not in measured_bits:
        raise _Unsupported("unknown_classical_condition", type_name, location)
    base = native.hyperparameters.get("base")
    raw_name = str(getattr(base, "name", ""))
    base_name = _ALIASES.get(raw_name, raw_name.lower())
    if base_name not in {"x", "y", "z", "h", "s", "t", "sx", "rx", "ry", "rz", "p", "phaseshift"}:
        raise _Unsupported("unsupported_conditional_gate", type_name, location)
    return Operation(
        OperationKind.GATE,
        base_name,
        quantum_wires=tuple(int(wire) for wire in native.wires),
        parameters=tuple(normalize_parameter(value) for value in getattr(base, "data", ())),
        condition=ClassicalCondition((measured_bits[measured_wire],), 1),
        source_location=location,
    )


def _lower_terminal_measurement(
    measurement: Any,
    index: int,
    labels: list[int],
    next_clbit: int,
    output_wires: tuple[int, ...] | None,
) -> tuple[Operation | None, int]:
    """Lower a terminal measurement process onto a Program IR measurement.

    Args:
        measurement: PennyLane terminal measurement process.
        index: Measurement index used for source locations.
        labels: Sorted tape wire labels used as a fallback wire order.
        next_clbit: Next unused classical-bit index.
        output_wires: Optional contract-declared output wires.

    Returns:
        Optional measurement operation and the updated classical-bit cursor.
    """
    measurement_type = type(measurement).__name__
    if measurement_type in {"StateMP", "DensityMatrixMP"}:
        return None, next_clbit
    if measurement_type not in {"ProbabilityMP", "ProbsMP", "SampleMP", "CountsMP"}:
        raise _Unsupported("unsupported_measurement_process", measurement_type, f"measurements[{index}]")
    # Preserve an explicit qml.probs/sample wire order exactly.  Contract
    # normalization belongs in the semantic materializer; lowering must not
    # silently rewrite the candidate's public return order.
    wires = tuple(int(wire) for wire in measurement.wires) or output_wires or tuple(reversed(labels))
    if not wires:
        return None, next_clbit
    bits = tuple(range(next_clbit, next_clbit + len(wires)))
    return (
        Operation(
            OperationKind.MEASUREMENT,
            _measurement_name(measurement),
            quantum_wires=wires,
            classical_bits=bits,
            semantic_data=(("process", measurement_type),),
            source_location=f"measurements[{index}]",
        ),
        next_clbit + len(wires),
    )


def _state_preparation_operation(native: Any, type_name: str, wires: tuple[int, ...], location: str) -> Operation:
    if type_name == "BasisState":
        bits = tuple(int(value) for value in native.data[0])
        state = [0.0j] * (2 ** len(bits))
        state[int("".join(str(value) for value in bits), 2)] = 1.0
    else:
        state = native.data[0]
    return Operation(
        OperationKind.STATE_PREPARATION,
        "statevector",
        quantum_wires=wires,
        semantic_data=bounded_statevector_semantic_data(state, wire_order="big_endian"),
        source_location=location,
    )


def _operation(native: Any, location: str) -> Operation:
    import pennylane as qml

    type_name = type(native).__name__
    inverse = type_name.startswith("Adjoint")
    base = getattr(native, "base", native)
    base_name = str(base.name)
    name = _ALIASES.get(base_name, base_name.lower())
    wires = tuple(int(wire) for wire in native.wires)
    control_wires = tuple(int(wire) for wire in getattr(native, "control_wires", ()))
    targets = tuple(wire for wire in wires if wire not in control_wires)
    control_values = tuple(getattr(native, "control_values", None) or (True,) * len(control_wires))
    controls = tuple(Control(wire, int(bool(value))) for wire, value in zip(control_wires, control_values, strict=True))
    kind = OperationKind.GATE
    if type_name in _STATE_PREPARATION:
        return _state_preparation_operation(native, type_name, wires, location)
    try:
        matrix = qml.matrix(native)
        digest = matrix_sha256(matrix)
    except Exception as exc:  # noqa: BLE001 - absence of a matrix is a capability result.
        raise _Unsupported("unsupported_nonunitary_operation", type_name, location) from exc
    # Operation.name is user-overridable.  A custom class named CNOT must not
    # acquire CNOT semantics when PennyLane actually executed another matrix.
    if not type(native).__module__.startswith("pennylane."):
        return Operation(
            kind,
            "dense_unitary",
            quantum_wires=wires,
            semantic_data=bounded_matrix_semantic_data(matrix),
            source_location=location,
        )
    if type_name in {"ControlledOp", "ControlledQubitUnitary"} and type(base).__name__ == "QubitUnitary":
        return Operation(
            kind,
            "dense_unitary",
            quantum_wires=targets,
            controls=controls,
            semantic_data=bounded_matrix_semantic_data(qml.matrix(base)),
            source_location=location,
        )
    if (
        type_name == "ControlledOp"
        and name in _NAMED_CONTROLLED_BASES
        and type(base).__module__.startswith("pennylane.")
    ):
        # Multi-controlled named single-qubit gates (for example the
        # multi-controlled PhaseShift of a phase-matched Grover oracle) keep
        # their named lowering instead of one full-register dense matrix.
        return Operation(
            kind,
            name,
            quantum_wires=targets,
            parameters=tuple(normalize_parameter(value) for value in base.data),
            controls=controls,
            semantic_data=(("matrix_sha256", digest),),
            source_location=location,
        )
    compressed = _controlled_block_matrix(matrix, wires, kind, location) if type_name == "QubitUnitary" else None
    if compressed is not None:
        return compressed
    if type_name in _ISING_NAMES:
        # Native two-qubit Ising rotations are trusted library gates, not
        # opaque candidate-supplied matrices; keep their named lowering with
        # the exact dense payload so the simulator replays PennyLane's
        # convention.
        return Operation(
            kind,
            _ISING_NAMES[type_name],
            quantum_wires=wires,
            parameters=tuple(normalize_parameter(value) for value in native.data),
            semantic_data=bounded_matrix_semantic_data(matrix),
            source_location=location,
        )
    if type_name in {"ControlledOp", "ControlledQubitUnitary", "QubitUnitary", "PauliRot"} or base_name in {
        "ISWAP",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
    }:
        return Operation(
            kind,
            "dense_unitary",
            quantum_wires=wires,
            semantic_data=bounded_matrix_semantic_data(matrix),
            source_location=location,
        )
    return Operation(
        kind,
        name,
        quantum_wires=targets,
        parameters=tuple(normalize_parameter(value) for value in native.data),
        controls=controls,
        inverse=inverse,
        semantic_data=(("matrix_sha256", digest),),
        source_location=location,
    )


def _controlled_block_matrix(
    matrix: Any,
    wires: tuple[int, ...],
    kind: OperationKind,
    location: str,
) -> Operation | None:
    import numpy as np

    value = np.asarray(matrix, dtype=np.complex128)
    if len(wires) < 2 or value.shape[0] > 32 or value.shape[0] != value.shape[1]:
        return None
    half = value.shape[0] // 2
    identity = np.eye(half, dtype=np.complex128)
    zero = np.zeros((half, half), dtype=np.complex128)
    if (
        np.allclose(value[:half, :half], identity)
        and np.allclose(value[:half, half:], zero)
        and np.allclose(value[half:, :half], zero)
    ):
        control_value, base = 1, value[half:, half:]
    elif (
        np.allclose(value[half:, half:], identity)
        and np.allclose(value[:half, half:], zero)
        and np.allclose(value[half:, :half], zero)
    ):
        control_value, base = 0, value[:half, :half]
    else:
        return None
    return Operation(
        kind,
        "dense_unitary",
        quantum_wires=wires[1:],
        controls=(Control(wires[0], control_value),),
        semantic_data=bounded_matrix_semantic_data(base),
        source_location=location,
    )


def _measurement_name(measurement: Any) -> str:
    type_name = type(measurement).__name__
    if type_name in {"ProbabilityMP", "ProbsMP"}:
        return "probabilities"
    if type_name == "SampleMP":
        return "sample"
    if type_name == "CountsMP":
        return "counts"
    return type_name.lower()
