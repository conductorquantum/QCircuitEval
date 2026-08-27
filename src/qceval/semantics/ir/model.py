"""Immutable semantic Program IR nodes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

IR_VERSION = "1"


class OperationKind(StrEnum):
    """Kinds of source behavior preserved before semantic planning."""

    GATE = "gate"
    STATE_PREPARATION = "state_preparation"
    MEASUREMENT = "measurement"
    RESET = "reset"
    DISCARD = "discard"
    BARRIER = "barrier"
    OPAQUE = "opaque"


class ParameterKind(StrEnum):
    """Whether a normalized operation parameter is numeric or symbolic."""

    NUMBER = "number"
    SYMBOL = "symbol"
    TEXT = "text"


@dataclass(frozen=True)
class Parameter:
    """One deterministic normalized operation parameter."""

    kind: ParameterKind
    value: str


@dataclass(frozen=True)
class Control:
    """One positive or open quantum control."""

    wire: int
    value: int = 1


@dataclass(frozen=True)
class ClassicalCondition:
    """Equality condition over normalized classical bits."""

    bits: tuple[int, ...]
    value: int


@dataclass(frozen=True)
class Provenance:
    """Non-semantic framework/source diagnostics excluded from IR hashes."""

    framework: str
    framework_version: str
    source_hash: str | None = None
    backend: str | None = None


@dataclass(frozen=True)
class Operation:
    """One ordered Program IR operation.

    Attributes:
        kind: Semantic node kind.
        name: Canonical operation name.
        quantum_wires: Target or measured quantum wires in logical order.
        classical_bits: Classical inputs/outputs in normalized order.
        parameters: Exact parser-normalized parameters.
        controls: Quantum controls distinct from ``quantum_wires``.
        condition: Optional classical equality condition.
        inverse: Whether the declared operation is adjointed.
        power: Optional exact normalized exponent.
        definition: Optional nested local operation definition.
        semantic_data: Stable provider data needed to interpret an opaque or
            custom operation.
        source_location: Non-semantic diagnostic location.
    """

    kind: OperationKind
    name: str
    quantum_wires: tuple[int, ...] = ()
    classical_bits: tuple[int, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    controls: tuple[Control, ...] = ()
    condition: ClassicalCondition | None = None
    inverse: bool = False
    power: Parameter | None = None
    definition: tuple[Operation, ...] = ()
    semantic_data: tuple[tuple[str, str], ...] = ()
    source_location: str | None = None


@dataclass(frozen=True)
class Program:
    """Complete framework-neutral ordered quantum/classical program."""

    ir_version: str
    num_qubits: int
    num_clbits: int
    operations: tuple[Operation, ...]
    global_phase: Parameter | None
    classical_render_order: tuple[int, ...]
    provenance: Provenance
    diagnostics: tuple[str, ...] = ()
