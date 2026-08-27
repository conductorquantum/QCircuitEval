"""Strict parsing and cross-field validation for task contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qceval.semantics.contracts._validation_fields import (
    parse_ancillas,
    parse_observation,
    parse_phase,
    parse_signature,
    parse_systems,
)
from qceval.semantics.contracts._validation_policy import (
    parse_approximation,
    parse_diagnostics,
    parse_limits,
    parse_parameters,
    parse_requirements,
    parse_routing,
    parse_target,
)
from qceval.semantics.contracts._validation_primitives import (
    boolean_value,
    enum_value,
    fail,
    nonempty_string,
    object_value,
    semantic_version,
    string_value,
)
from qceval.semantics.contracts._validation_rules import validate_cross_fields
from qceval.semantics.contracts.kinds import (
    AuditStatus,
    BehaviorKind,
    Contract,
)

CONTRACT_SCHEMA_VERSION = "2"
SUPPORTED_CONTRACT_SCHEMA_VERSIONS = frozenset({"1", CONTRACT_SCHEMA_VERSION})


def parse_contract(payload: Mapping[str, Any]) -> Contract:
    """Parse and validate a strict contract mapping.

    Args:
        payload: Raw JSON-compatible contract object.

    Returns:
        Immutable validated contract.

    Raises:
        ContractValidationError: If any field, type, or cross-field invariant
            is invalid.
    """
    raw = object_value(
        payload,
        "$",
        required={
            "schema_version",
            "suite",
            "task_id",
            "contract_version",
            "kind",
            "shadow_only",
            "audit_status",
            "signature",
            "systems",
            "observation",
            "phase",
            "ancillas",
            "parameters",
            "approximation",
            "target",
            "routing",
            "limits",
            "requirements",
            "diagnostics",
        },
    )
    schema_version = string_value(raw["schema_version"], "$.schema_version")
    if schema_version not in SUPPORTED_CONTRACT_SCHEMA_VERSIONS:
        fail("$.schema_version", f"unsupported version {schema_version!r}")
    contract = Contract(
        schema_version=schema_version,
        suite=nonempty_string(raw["suite"], "$.suite"),
        task_id=nonempty_string(raw["task_id"], "$.task_id"),
        contract_version=semantic_version(
            raw["contract_version"],
            "$.contract_version",
        ),
        kind=enum_value(BehaviorKind, raw["kind"], "$.kind"),
        shadow_only=boolean_value(raw["shadow_only"], "$.shadow_only"),
        audit_status=enum_value(AuditStatus, raw["audit_status"], "$.audit_status"),
        signature=parse_signature(raw["signature"]),
        systems=parse_systems(raw["systems"]),
        observation=parse_observation(raw["observation"]),
        phase=parse_phase(raw["phase"]),
        ancillas=parse_ancillas(raw["ancillas"]),
        parameters=parse_parameters(raw["parameters"]),
        approximation=parse_approximation(raw["approximation"]),
        target=parse_target(raw["target"]),
        routing=parse_routing(raw["routing"]),
        limits=parse_limits(raw["limits"]),
        requirements=parse_requirements(raw["requirements"]),
        diagnostics=parse_diagnostics(raw["diagnostics"]),
    )
    if schema_version == "1" and any(
        not isinstance(value, int | float) or isinstance(value, bool)
        for point in contract.parameters.diagnostic_points
        for value in point
    ):
        fail("$.parameters.diagnostic_points", "schema version 1 supports only numeric points")
    validate_cross_fields(contract)
    return contract


__all__ = ["CONTRACT_SCHEMA_VERSION", "SUPPORTED_CONTRACT_SCHEMA_VERSIONS", "parse_contract"]
