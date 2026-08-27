"""Field-level parsers for strict semantic contracts."""

from __future__ import annotations

from typing import Any

from qceval.semantics.contracts._validation_primitives import (
    array_value,
    boolean_value,
    enum_value,
    fail,
    integer_tuple,
    integer_value,
    nonempty_string,
    number_value,
    object_value,
    string_tuple,
    unique,
)
from qceval.semantics.contracts.kinds import (
    AncillaFinal,
    AncillaInitial,
    AncillaPolicy,
    AncillasSpec,
    ArgumentSpec,
    BitOrder,
    ObservationSpec,
    PhaseSpec,
    PostselectionSpec,
    RelativePhase,
    SignatureSpec,
    SystemKind,
    SystemRole,
    SystemSpec,
    SystemsSpec,
)


def parse_signature(value: Any) -> SignatureSpec:
    """Parse the callable signature section.

    Args:
        value: Untrusted JSON-compatible signature value.

    Returns:
        Validated callable signature specification.

    Raises:
        ContractValidationError: If the section is malformed.
    """
    raw = object_value(
        value,
        "$.signature",
        required={"entry_point", "arguments", "return_type"},
    )
    arguments = tuple(
        _parse_argument(item, f"$.signature.arguments[{index}]")
        for index, item in enumerate(array_value(raw["arguments"], "$.signature.arguments"))
    )
    unique(
        [item.name for item in arguments],
        "$.signature.arguments",
        "argument names",
    )
    return SignatureSpec(
        entry_point=nonempty_string(
            raw["entry_point"],
            "$.signature.entry_point",
        ),
        arguments=arguments,
        return_type=nonempty_string(
            raw["return_type"],
            "$.signature.return_type",
        ),
    )


def _parse_argument(value: Any, path: str) -> ArgumentSpec:
    raw = object_value(
        value,
        path,
        required={"name", "type", "domain", "required"},
    )
    return ArgumentSpec(
        name=nonempty_string(raw["name"], f"{path}.name"),
        value_type=nonempty_string(raw["type"], f"{path}.type"),
        domain=nonempty_string(raw["domain"], f"{path}.domain"),
        required=boolean_value(raw["required"], f"{path}.required"),
    )


def parse_systems(value: Any) -> SystemsSpec:
    """Parse declared quantum and classical systems.

    Args:
        value: Untrusted JSON-compatible systems value.

    Returns:
        Validated named-system specification.

    Raises:
        ContractValidationError: If a system is malformed or duplicated.
    """
    raw = object_value(value, "$.systems", required={"items"})
    items = tuple(
        _parse_system(item, f"$.systems.items[{index}]")
        for index, item in enumerate(array_value(raw["items"], "$.systems.items"))
    )
    if not items:
        fail("$.systems.items", "must contain at least one system")
    unique([item.name for item in items], "$.systems.items", "system names")
    unique(
        [(item.kind, index) for item in items for index in item.indices],
        "$.systems.items",
        "physical indices within one system kind",
    )
    return SystemsSpec(items)


def _parse_system(value: Any, path: str) -> SystemSpec:
    raw = object_value(
        value,
        path,
        required={"name", "kind", "role", "indices", "dimension"},
    )
    indices = integer_tuple(raw["indices"], f"{path}.indices", minimum=0)
    if not indices:
        fail(f"{path}.indices", "must not be empty")
    unique(list(indices), f"{path}.indices", "indices")
    return SystemSpec(
        name=nonempty_string(raw["name"], f"{path}.name"),
        kind=enum_value(SystemKind, raw["kind"], f"{path}.kind"),
        role=enum_value(SystemRole, raw["role"], f"{path}.role"),
        indices=indices,
        dimension=integer_value(raw["dimension"], f"{path}.dimension", minimum=2),
    )


def parse_observation(value: Any) -> ObservationSpec:
    """Parse observation and postselection policy.

    Args:
        value: Untrusted JSON-compatible observation value.

    Returns:
        Validated observation specification.

    Raises:
        ContractValidationError: If observation or postselection data is
            malformed.
    """
    raw = object_value(
        value,
        "$.observation",
        required={
            "quantum",
            "classical",
            "ignored",
            "marginalize",
            "bit_order",
            "postselection",
        },
    )
    return ObservationSpec(
        quantum=string_tuple(raw["quantum"], "$.observation.quantum"),
        classical=string_tuple(raw["classical"], "$.observation.classical"),
        ignored=string_tuple(raw["ignored"], "$.observation.ignored"),
        marginalize=string_tuple(
            raw["marginalize"],
            "$.observation.marginalize",
        ),
        bit_order=enum_value(
            BitOrder,
            raw["bit_order"],
            "$.observation.bit_order",
        ),
        postselection=_parse_postselection(raw["postselection"]),
    )


def _parse_postselection(value: Any) -> PostselectionSpec | None:
    if value is None:
        return None
    raw = object_value(
        value,
        "$.observation.postselection",
        required={"system", "values", "min_probability"},
    )
    values = string_tuple(raw["values"], "$.observation.postselection.values")
    if not values:
        fail("$.observation.postselection.values", "must not be empty")
    probability = number_value(
        raw["min_probability"],
        "$.observation.postselection.min_probability",
        minimum=0.0,
    )
    if probability <= 0.0 or probability > 1.0:
        fail("$.observation.postselection.min_probability", "must be in (0, 1]")
    return PostselectionSpec(
        system=nonempty_string(
            raw["system"],
            "$.observation.postselection.system",
        ),
        values=values,
        min_probability=probability,
    )


def parse_phase(value: Any) -> PhaseSpec:
    """Parse phase-equivalence policy.

    Args:
        value: Untrusted JSON-compatible phase value.

    Returns:
        Validated phase specification.

    Raises:
        ContractValidationError: If the phase policy is malformed.
    """
    raw = object_value(
        value,
        "$.phase",
        required={"global_phase_irrelevant", "relative_phase"},
    )
    return PhaseSpec(
        global_phase_irrelevant=boolean_value(
            raw["global_phase_irrelevant"],
            "$.phase.global_phase_irrelevant",
        ),
        relative_phase=enum_value(
            RelativePhase,
            raw["relative_phase"],
            "$.phase.relative_phase",
        ),
    )


def parse_ancillas(value: Any) -> AncillasSpec:
    """Parse ancilla lifecycle policies.

    Args:
        value: Untrusted JSON-compatible ancilla value.

    Returns:
        Validated ancilla policy specification.

    Raises:
        ContractValidationError: If a policy is malformed or duplicated.
    """
    raw = object_value(value, "$.ancillas", required={"items"})
    items = tuple(
        _parse_ancilla(item, f"$.ancillas.items[{index}]")
        for index, item in enumerate(array_value(raw["items"], "$.ancillas.items"))
    )
    unique(
        [item.system for item in items],
        "$.ancillas.items",
        "ancilla systems",
    )
    return AncillasSpec(items)


def _parse_ancilla(value: Any, path: str) -> AncillaPolicy:
    raw = object_value(
        value,
        path,
        required={"system", "initial", "final"},
    )
    return AncillaPolicy(
        system=nonempty_string(raw["system"], f"{path}.system"),
        initial=enum_value(AncillaInitial, raw["initial"], f"{path}.initial"),
        final=enum_value(AncillaFinal, raw["final"], f"{path}.final"),
    )
