"""Strict, versioned task contracts for semantic grading."""

from qceval.semantics.contracts.binding import (
    call_args_from_arity,
    call_args_from_code,
    call_args_from_signature,
    required_arity_from_code,
)
from qceval.semantics.contracts.kinds import (
    AuditStatus,
    BehaviorKind,
    Contract,
    ContractValidationError,
    RequirementSpec,
    RouteSpec,
    RoutingSpec,
)
from qceval.semantics.contracts.registry import ContractChange, ContractRegistry
from qceval.semantics.contracts.serialization import (
    canonical_contract_bytes,
    canonical_contract_json,
    contract_hash,
    contract_to_dict,
    parse_contract_json,
)
from qceval.semantics.contracts.validation import (
    CONTRACT_SCHEMA_VERSION,
    SUPPORTED_CONTRACT_SCHEMA_VERSIONS,
    parse_contract,
)

__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "SUPPORTED_CONTRACT_SCHEMA_VERSIONS",
    "AuditStatus",
    "BehaviorKind",
    "Contract",
    "ContractChange",
    "ContractRegistry",
    "ContractValidationError",
    "RequirementSpec",
    "RouteSpec",
    "RoutingSpec",
    "call_args_from_arity",
    "call_args_from_code",
    "call_args_from_signature",
    "canonical_contract_bytes",
    "canonical_contract_json",
    "contract_hash",
    "contract_to_dict",
    "parse_contract",
    "parse_contract_json",
    "required_arity_from_code",
]
