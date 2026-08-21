"""Canonical JSON serialization and hashing for task contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from qceval.semantics.contracts.kinds import Contract, ContractValidationError, FrozenArray, FrozenObject
from qceval.semantics.contracts.validation import parse_contract


def parse_contract_json(payload: str | bytes | bytearray) -> Contract:
    """Parse strict JSON into a validated immutable contract.

    Args:
        payload: UTF-8 JSON text or bytes.

    Returns:
        Validated contract.

    Raises:
        ContractValidationError: If JSON syntax, duplicate keys, constants, or
            contract fields are invalid.
    """

    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractValidationError("$", f"invalid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ContractValidationError("$", "must be an object")
    return parse_contract(raw)


def contract_to_dict(contract: Contract) -> dict[str, Any]:
    """Return the newest JSON-compatible contract representation.

    Args:
        contract: Validated immutable contract.

    Returns:
        JSON-compatible mapping.
    """

    return {
        "schema_version": contract.schema_version,
        "suite": contract.suite,
        "task_id": contract.task_id,
        "contract_version": contract.contract_version,
        "kind": contract.kind.value,
        "shadow_only": contract.shadow_only,
        "audit_status": contract.audit_status.value,
        "signature": {
            "entry_point": contract.signature.entry_point,
            "arguments": [
                {
                    "name": item.name,
                    "type": item.value_type,
                    "domain": item.domain,
                    "required": item.required,
                }
                for item in contract.signature.arguments
            ],
            "return_type": contract.signature.return_type,
        },
        "systems": {
            "items": [
                {
                    "name": item.name,
                    "kind": item.kind.value,
                    "role": item.role.value,
                    "indices": list(item.indices),
                    "dimension": item.dimension,
                }
                for item in contract.systems.items
            ]
        },
        "observation": {
            "quantum": list(contract.observation.quantum),
            "classical": list(contract.observation.classical),
            "ignored": list(contract.observation.ignored),
            "marginalize": list(contract.observation.marginalize),
            "bit_order": contract.observation.bit_order.value,
            "postselection": _postselection(contract),
        },
        "phase": {
            "global_phase_irrelevant": contract.phase.global_phase_irrelevant,
            "relative_phase": contract.phase.relative_phase.value,
        },
        "ancillas": {
            "items": [
                {"system": item.system, "initial": item.initial.value, "final": item.final.value}
                for item in contract.ancillas.items
            ]
        },
        "parameters": {
            "items": [
                {
                    "name": item.name,
                    "type": item.value_type,
                    "domain": item.domain,
                    "units": item.units,
                    "periodicity": item.periodicity,
                    "excluded": list(item.excluded),
                    "binding": item.binding,
                }
                for item in contract.parameters.items
            ],
            "quantifier": contract.parameters.quantifier.value,
            "completeness": contract.parameters.completeness,
            "diagnostic_points": [list(point) for point in contract.parameters.diagnostic_points],
        },
        "approximation": {
            "mode": contract.approximation.mode.value,
            "metric": contract.approximation.metric,
            "tolerance": contract.approximation.tolerance,
            "uncertainty": contract.approximation.uncertainty,
            "error_budget": contract.approximation.error_budget,
        },
        "target": {
            "id": contract.target.target_id,
            "version": contract.target.version,
            "sha256": contract.target.sha256,
            "source": contract.target.source,
            "manifest": contract.target.manifest,
            "independent_derivations": contract.target.independent_derivations,
        },
        "routing": {
            "primary": [_route(item) for item in contract.routing.primary],
            "fallback": [_route(item) for item in contract.routing.fallback],
        },
        "limits": {
            "wall_seconds": contract.limits.wall_seconds,
            "cpu_seconds": contract.limits.cpu_seconds,
            "memory_mib": contract.limits.memory_mib,
            "max_qubits": contract.limits.max_qubits,
            "max_dimension": contract.limits.max_dimension,
            "max_cases": contract.limits.max_cases,
            "max_branches": contract.limits.max_branches,
            "max_expression_nodes": contract.limits.max_expression_nodes,
        },
        "requirements": [
            {
                "id": item.requirement_id,
                "kind": item.kind,
                "source": item.source,
                "value": _plain_json(item.value),
            }
            for item in contract.requirements
        ],
        "diagnostics": [
            {"id": item.diagnostic_id, "kind": item.kind, "enabled": item.enabled} for item in contract.diagnostics
        ],
    }


def canonical_contract_json(contract: Contract) -> str:
    """Return stable finite JSON without insignificant whitespace.

    Args:
        contract: Validated immutable contract.

    Returns:
        Canonical JSON text.
    """

    return json.dumps(
        contract_to_dict(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_contract_bytes(contract: Contract) -> bytes:
    """Return UTF-8 canonical contract bytes.

    Args:
        contract: Validated immutable contract.

    Returns:
        Canonical UTF-8 bytes.
    """

    return canonical_contract_json(contract).encode("utf-8")


def contract_hash(contract: Contract) -> str:
    """Return the content-addressed SHA-256 contract digest.

    Args:
        contract: Validated immutable contract.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(canonical_contract_bytes(contract)).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError("$", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ContractValidationError("$", f"non-finite JSON constant {value!r}")


def _postselection(contract: Contract) -> dict[str, Any] | None:
    value = contract.observation.postselection
    if value is None:
        return None
    return {"system": value.system, "values": list(value.values), "min_probability": value.min_probability}


def _route(value: Any) -> dict[str, Any]:
    return {
        "engine": value.engine,
        "capabilities": list(value.capabilities),
        "cross_check": value.cross_check,
    }


def _plain_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, FrozenArray):
        return [_plain_json(item) for item in value.items]
    if isinstance(value, FrozenObject):
        return {key: _plain_json(item) for key, item in value.items}
    return value
