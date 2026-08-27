"""Immutable models and enums for behavior-first task contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class ContractValidationError(ValueError):
    """A stable, path-addressed contract validation failure."""

    def __init__(self, path: str, reason: str) -> None:
        """Initialize a validation error.

        Args:
            path: Dotted JSON path at which validation failed.
            reason: Stable human-readable failure reason.
        """
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class BehaviorKind(StrEnum):
    """Primary semantic object a contract verifies."""

    STATE = "state"
    TOTAL_UNITARY = "total_unitary"
    ISOMETRY = "isometry"
    CHANNEL = "channel"
    INSTRUMENT = "instrument"
    DISTRIBUTION = "distribution"
    CLASSICAL_IO = "classical_io"
    OBJECTIVE = "objective"


class AuditStatus(StrEnum):
    """Review state of one task contract."""

    PROVISIONAL = "provisional"
    REVIEWED = "reviewed"
    BLOCKED = "blocked"


class SystemKind(StrEnum):
    """Physical type of a named system."""

    QUANTUM = "quantum"
    CLASSICAL = "classical"


class SystemRole(StrEnum):
    """Contract role assigned to a named system."""

    LOGICAL_INPUT = "logical_input"
    LOGICAL_OUTPUT = "logical_output"
    LOGICAL_IO = "logical_io"
    CLASSICAL_INPUT = "classical_input"
    CLASSICAL_OUTPUT = "classical_output"
    CLASSICAL_IO = "classical_io"
    ANCILLA = "ancilla"
    WORK = "work"
    ENVIRONMENT = "environment"


class BitOrder(StrEnum):
    """Normalized rendering order for observed classical bits."""

    LITTLE_ENDIAN = "little_endian"
    BIG_ENDIAN = "big_endian"
    PROMPT = "prompt"
    NOT_APPLICABLE = "not_applicable"


class RelativePhase(StrEnum):
    """Whether relative phase is part of the contracted observation."""

    PRESERVE = "preserve"
    UNCONSTRAINED = "unconstrained"
    NOT_APPLICABLE = "not_applicable"


class AncillaInitial(StrEnum):
    """Allowed ancilla initialization policy."""

    ZERO = "zero"
    DIRTY = "dirty"
    UNCONSTRAINED = "unconstrained"


class AncillaFinal(StrEnum):
    """Required ancilla final-state policy."""

    RESTORE = "restore"
    DISCARD = "discard"
    UNCONSTRAINED = "unconstrained"


class ParameterQuantifier(StrEnum):
    """Completeness quantifier for program parameters."""

    NONE = "none"
    ALL = "all"
    EXHAUSTIVE = "exhaustive"
    BOUNDED = "bounded"


class ApproximationMode(StrEnum):
    """Whether target behavior is exact or intentionally approximate."""

    EXACT = "exact"
    APPROXIMATE = "approximate"


JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class FrozenArray:
    """Immutable JSON array used by requirement values."""

    items: tuple[JsonValue, ...]


@dataclass(frozen=True)
class FrozenObject:
    """Immutable, key-sorted JSON object used by requirement values."""

    items: tuple[tuple[str, JsonValue], ...]


JsonValue: TypeAlias = JsonScalar | FrozenArray | FrozenObject
ParameterPointValue: TypeAlias = JsonScalar


@dataclass(frozen=True)
class ArgumentSpec:
    """One public entry-point argument."""

    name: str
    value_type: str
    domain: str
    required: bool


@dataclass(frozen=True)
class SignatureSpec:
    """Candidate entry-point and return interface."""

    entry_point: str
    arguments: tuple[ArgumentSpec, ...]
    return_type: str


@dataclass(frozen=True)
class SystemSpec:
    """One named quantum or classical input/output system."""

    name: str
    kind: SystemKind
    role: SystemRole
    indices: tuple[int, ...]
    dimension: int


@dataclass(frozen=True)
class SystemsSpec:
    """Named systems and their logical roles."""

    items: tuple[SystemSpec, ...]


@dataclass(frozen=True)
class PostselectionSpec:
    """A declared postselection event and probability floor."""

    system: str
    values: tuple[str, ...]
    min_probability: float


@dataclass(frozen=True)
class ObservationSpec:
    """Quantum/classical outputs that semantic verification observes."""

    quantum: tuple[str, ...]
    classical: tuple[str, ...]
    ignored: tuple[str, ...]
    marginalize: tuple[str, ...]
    bit_order: BitOrder
    postselection: PostselectionSpec | None


@dataclass(frozen=True)
class PhaseSpec:
    """Global- and relative-phase policy."""

    global_phase_irrelevant: bool
    relative_phase: RelativePhase


@dataclass(frozen=True)
class AncillaPolicy:
    """Initialization, restoration, and discard policy for one ancilla."""

    system: str
    initial: AncillaInitial
    final: AncillaFinal


@dataclass(frozen=True)
class AncillasSpec:
    """Complete policy for every named ancilla system."""

    items: tuple[AncillaPolicy, ...]


@dataclass(frozen=True)
class ParameterSpec:
    """One program parameter and its declared domain."""

    name: str
    value_type: str
    domain: str
    units: str
    periodicity: float | None
    excluded: tuple[float, ...]
    binding: str


@dataclass(frozen=True)
class ParametersSpec:
    """Parameter family, quantifier, and completeness strategy."""

    items: tuple[ParameterSpec, ...]
    quantifier: ParameterQuantifier
    completeness: str | None
    diagnostic_points: tuple[tuple[ParameterPointValue, ...], ...]


@dataclass(frozen=True)
class ApproximationSpec:
    """Metric, tolerance, uncertainty, and algorithmic error budget."""

    mode: ApproximationMode
    metric: str
    tolerance: float
    uncertainty: float
    error_budget: float


@dataclass(frozen=True)
class TargetSpec:
    """Independent target artifact reference and provenance."""

    target_id: str
    version: str
    sha256: str
    source: str
    manifest: str
    independent_derivations: int


@dataclass(frozen=True)
class RouteSpec:
    """One ordered verifier route and its required capabilities."""

    engine: str
    capabilities: tuple[str, ...]
    cross_check: bool


@dataclass(frozen=True)
class RoutingSpec:
    """Ordered primary and fallback verifier routes."""

    primary: tuple[RouteSpec, ...]
    fallback: tuple[RouteSpec, ...]


@dataclass(frozen=True)
class LimitsSpec:
    """Deterministic resource and materialization limits."""

    wall_seconds: float
    cpu_seconds: float
    memory_mib: int
    max_qubits: int
    max_dimension: int
    max_cases: int
    max_branches: int
    max_expression_nodes: int


@dataclass(frozen=True)
class RequirementSpec:
    """One prompt-derived hard API or structural requirement."""

    requirement_id: str
    kind: str
    source: str
    value: JsonValue


@dataclass(frozen=True)
class DiagnosticSpec:
    """One non-authoritative diagnostic observation."""

    diagnostic_id: str
    kind: str
    enabled: bool


@dataclass(frozen=True)
class Contract:
    """Complete immutable semantic contract for one benchmark task."""

    schema_version: str
    suite: str
    task_id: str
    contract_version: str
    kind: BehaviorKind
    shadow_only: bool
    audit_status: AuditStatus
    signature: SignatureSpec
    systems: SystemsSpec
    observation: ObservationSpec
    phase: PhaseSpec
    ancillas: AncillasSpec
    parameters: ParametersSpec
    approximation: ApproximationSpec
    target: TargetSpec
    routing: RoutingSpec
    limits: LimitsSpec
    requirements: tuple[RequirementSpec, ...]
    diagnostics: tuple[DiagnosticSpec, ...]

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable ``(suite, task_id)`` registry key."""

        return (self.suite, self.task_id)
