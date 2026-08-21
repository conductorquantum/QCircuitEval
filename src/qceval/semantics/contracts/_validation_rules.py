"""Cross-field invariants for validated contract sections."""

from __future__ import annotations

from collections.abc import Mapping

from qceval.semantics.contracts._validation_primitives import fail
from qceval.semantics.contracts.kinds import (
    ApproximationMode,
    BehaviorKind,
    BitOrder,
    Contract,
    ParameterQuantifier,
    RelativePhase,
    SystemKind,
    SystemRole,
    SystemSpec,
)


def validate_cross_fields(contract: Contract) -> None:
    """Validate invariants spanning multiple contract sections.

    Args:
        contract: Individually parsed contract sections to cross-check.

    Raises:
        ContractValidationError: If references, policies, or semantic
            constraints conflict across sections.
    """
    systems = {item.name: item for item in contract.systems.items}
    referenced = {
        *contract.observation.quantum,
        *contract.observation.classical,
        *contract.observation.ignored,
        *contract.observation.marginalize,
    }
    unknown = sorted(referenced - systems.keys())
    if unknown:
        fail("$.observation", f"references unknown systems: {unknown}")
    _validate_observation_sets(contract, systems)
    _validate_observation_kinds(contract, systems)
    _validate_system_roles(contract)
    _validate_ancilla_policies(contract, systems)
    _validate_parameters(contract)
    _validate_approximation(contract)
    _validate_behavior_kind(contract)


def _validate_observation_sets(
    contract: Contract,
    systems: Mapping[str, SystemSpec],
) -> None:
    observed = set(contract.observation.quantum) | set(contract.observation.classical)
    ignored = set(contract.observation.ignored)
    marginalized = set(contract.observation.marginalize)
    if observed & ignored:
        fail("$.observation", "a system cannot be both observed and ignored")
    if marginalized - observed:
        fail("$.observation.marginalize", "marginalized systems must be observed")
    postselection = contract.observation.postselection
    if postselection is None:
        return
    if postselection.system not in systems:
        fail("$.observation.postselection.system", "must name a declared system")
    system = systems[postselection.system]
    width = len(system.indices)
    if any(len(value) != width or set(value) - {"0", "1"} for value in postselection.values):
        fail(
            "$.observation.postselection.values",
            f"must contain {width}-bit binary values for the selected system",
        )
    if system.kind is SystemKind.CLASSICAL and postselection.system not in contract.observation.classical:
        fail(
            "$.observation.postselection.system",
            "classical postselection must name an observed system",
        )


def _validate_observation_kinds(
    contract: Contract,
    systems: Mapping[str, SystemSpec],
) -> None:
    if any(systems[name].kind is not SystemKind.QUANTUM for name in contract.observation.quantum):
        fail("$.observation.quantum", "must reference only quantum systems")
    if any(systems[name].kind is not SystemKind.CLASSICAL for name in contract.observation.classical):
        fail("$.observation.classical", "must reference only classical systems")
    if contract.observation.classical and contract.observation.bit_order is BitOrder.NOT_APPLICABLE:
        fail(
            "$.observation.bit_order",
            "classical observation requires an explicit bit order",
        )
    if not contract.observation.classical and contract.observation.bit_order is not BitOrder.NOT_APPLICABLE:
        fail(
            "$.observation.bit_order",
            "must be not_applicable without classical observation",
        )


def _validate_ancilla_policies(
    contract: Contract,
    systems: Mapping[str, SystemSpec],
) -> None:
    ancilla_systems = {item.name for item in systems.values() if item.role is SystemRole.ANCILLA}
    policy_systems = {item.system for item in contract.ancillas.items}
    if ancilla_systems != policy_systems:
        fail(
            "$.ancillas.items",
            "must provide exactly one policy for every ancilla system",
        )
    if any(systems[item.system].kind is not SystemKind.QUANTUM for item in contract.ancillas.items):
        fail("$.ancillas.items", "ancilla policies require quantum systems")


def _validate_system_roles(contract: Contract) -> None:
    classical_roles = {
        SystemRole.CLASSICAL_INPUT,
        SystemRole.CLASSICAL_OUTPUT,
        SystemRole.CLASSICAL_IO,
    }
    for system in contract.systems.items:
        if system.kind is SystemKind.QUANTUM and system.role in classical_roles:
            fail(
                "$.systems.items",
                f"quantum system {system.name!r} has a classical role",
            )
        if system.kind is SystemKind.CLASSICAL and system.role not in classical_roles:
            fail(
                "$.systems.items",
                f"classical system {system.name!r} has a quantum role",
            )


def _validate_parameters(contract: Contract) -> None:
    parameters = contract.parameters
    if not parameters.items:
        if parameters.quantifier is not ParameterQuantifier.NONE or parameters.completeness is not None:
            fail(
                "$.parameters",
                "empty parameter lists require quantifier none and null completeness",
            )
        if parameters.diagnostic_points:
            fail(
                "$.parameters.diagnostic_points",
                "cannot contain points without parameters",
            )
        return
    if parameters.quantifier is ParameterQuantifier.NONE:
        fail(
            "$.parameters.quantifier",
            "parameterized contracts cannot use quantifier none",
        )
    if (
        parameters.quantifier in {ParameterQuantifier.ALL, ParameterQuantifier.EXHAUSTIVE}
        and not parameters.completeness
    ):
        fail(
            "$.parameters.completeness",
            "universal/exhaustive verification requires a completeness method",
        )
    expected_width = len(parameters.items)
    if any(len(point) != expected_width for point in parameters.diagnostic_points):
        fail(
            "$.parameters.diagnostic_points",
            "every point must bind every parameter",
        )


def _validate_approximation(contract: Contract) -> None:
    approximation = contract.approximation
    if approximation.uncertainty > approximation.tolerance:
        fail(
            "$.approximation.uncertainty",
            "cannot exceed the acceptance tolerance",
        )
    if approximation.mode is ApproximationMode.EXACT and approximation.error_budget != 0.0:
        fail(
            "$.approximation.error_budget",
            "exact targets require a zero algorithmic error budget",
        )
    if approximation.mode is ApproximationMode.APPROXIMATE and approximation.error_budget <= 0.0:
        fail(
            "$.approximation.error_budget",
            "approximate targets require a positive error budget",
        )


def _validate_behavior_kind(contract: Contract) -> None:
    quantum = bool(contract.observation.quantum)
    classical = bool(contract.observation.classical)
    if not quantum and contract.kind in {
        BehaviorKind.STATE,
        BehaviorKind.TOTAL_UNITARY,
        BehaviorKind.ISOMETRY,
        BehaviorKind.CHANNEL,
    }:
        fail(
            "$.observation.quantum",
            f"{contract.kind.value} requires quantum output",
        )
    if contract.kind is BehaviorKind.TOTAL_UNITARY and classical:
        fail(
            "$.observation.classical",
            "total_unitary cannot contract a classical measurement",
        )
    if contract.kind in {BehaviorKind.ISOMETRY, BehaviorKind.CHANNEL} and classical:
        fail(
            "$.observation.classical",
            f"{contract.kind.value} requires quantum-only output",
        )
    if (
        contract.kind
        in {
            BehaviorKind.DISTRIBUTION,
            BehaviorKind.CLASSICAL_IO,
        }
        and not classical
    ):
        fail(
            "$.observation.classical",
            f"{contract.kind.value} requires classical output",
        )
    if (
        contract.kind
        in {
            BehaviorKind.DISTRIBUTION,
            BehaviorKind.CLASSICAL_IO,
        }
        and quantum
    ):
        fail(
            "$.observation.quantum",
            f"{contract.kind.value} cannot contract quantum output",
        )
    if contract.kind is BehaviorKind.INSTRUMENT and not (quantum and classical):
        fail("$.observation", "instrument requires quantum and classical outputs")
    if not quantum and contract.phase.global_phase_irrelevant:
        fail(
            "$.phase.global_phase_irrelevant",
            "global phase is meaningless without quantum output",
        )
    if not quantum and contract.phase.relative_phase is not RelativePhase.NOT_APPLICABLE:
        fail(
            "$.phase.relative_phase",
            "must be not_applicable without quantum output",
        )
