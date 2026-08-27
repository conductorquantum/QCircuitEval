"""Strict parsing, serialization, validation, and registry tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from qceval.semantics.contracts import (
    ContractRegistry,
    ContractValidationError,
    canonical_contract_json,
    contract_hash,
    contract_to_dict,
    parse_contract,
    parse_contract_json,
)


def _valid_contract() -> dict[str, Any]:
    target_hash = "a" * 64
    return {
        "schema_version": "1",
        "suite": "core",
        "task_id": "02",
        "contract_version": "1.0.0",
        "kind": "state",
        "shadow_only": True,
        "audit_status": "provisional",
        "signature": {"entry_point": "prepare", "arguments": [], "return_type": "quantum_program"},
        "systems": {
            "items": [
                {"name": "output", "kind": "quantum", "role": "logical_output", "indices": [0], "dimension": 2},
                {"name": "readout", "kind": "classical", "role": "classical_output", "indices": [0], "dimension": 2},
            ]
        },
        "observation": {
            "quantum": ["output"],
            "classical": ["readout"],
            "ignored": [],
            "marginalize": [],
            "bit_order": "prompt",
            "postselection": None,
        },
        "phase": {"global_phase_irrelevant": True, "relative_phase": "preserve"},
        "ancillas": {"items": []},
        "parameters": {"items": [], "quantifier": "none", "completeness": None, "diagnostic_points": []},
        "approximation": {
            "mode": "exact",
            "metric": "trace_distance",
            "tolerance": 1e-9,
            "uncertainty": 1e-12,
            "error_budget": 0.0,
        },
        "target": {
            "id": "task_02_state",
            "version": "1.0.0",
            "sha256": target_hash,
            "source": "analytic_spec",
            "manifest": "targets/core/02/manifest.json",
            "independent_derivations": 1,
        },
        "routing": {
            "primary": [{"engine": "state_exact", "capabilities": ["static"], "cross_check": False}],
            "fallback": [],
        },
        "limits": {
            "wall_seconds": 10.0,
            "cpu_seconds": 10.0,
            "memory_mib": 2048,
            "max_qubits": 8,
            "max_dimension": 256,
            "max_cases": 256,
            "max_branches": 256,
            "max_expression_nodes": 10000,
        },
        "requirements": [{"id": "return", "kind": "return_type", "source": "prompt", "value": {"type": "circuit"}}],
        "diagnostics": [{"id": "gate_counts", "kind": "gate_counts", "enabled": True}],
    }


def test_contract_round_trips_canonically() -> None:
    contract = parse_contract(_valid_contract())
    canonical = canonical_contract_json(contract)
    reparsed = parse_contract_json(canonical)

    assert reparsed == contract
    assert contract_to_dict(reparsed) == _valid_contract()
    assert contract_hash(reparsed) == contract_hash(contract)
    assert " " not in canonical


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (lambda value: value.__setitem__("unknown", True), "$"),
        (lambda value: value["signature"].__setitem__("unknown", True), "$.signature"),
        (lambda value: value.pop("target"), "$"),
        (lambda value: value.__setitem__("shadow_only", 1), "$.shadow_only"),
        (lambda value: value["limits"].__setitem__("memory_mib", True), "$.limits.memory_mib"),
        (lambda value: value.__setitem__("schema_version", "3"), "$.schema_version"),
        (lambda value: value.__setitem__("contract_version", "v1"), "$.contract_version"),
        (lambda value: value["target"].__setitem__("sha256", "bad"), "$.target.sha256"),
    ],
)
def test_strict_parser_rejects_invalid_fields(mutate: Any, path: str) -> None:
    payload = _valid_contract()
    mutate(payload)

    with pytest.raises(ContractValidationError) as exc:
        parse_contract(payload)

    assert exc.value.path == path


def test_json_parser_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(ContractValidationError, match="duplicate JSON key"):
        parse_contract_json('{"schema_version":"1","schema_version":"1"}')
    with pytest.raises(ContractValidationError, match="non-finite JSON constant"):
        parse_contract_json('{"value":NaN}')


def test_requirement_values_are_immutable_and_canonical() -> None:
    payload = _valid_contract()
    payload["requirements"][0]["value"] = {"z": [2, 1], "a": {"enabled": True}}
    contract = parse_contract(payload)
    before = canonical_contract_json(contract)
    payload["requirements"][0]["value"]["z"].append(3)

    assert canonical_contract_json(contract) == before
    assert contract_to_dict(contract)["requirements"][0]["value"] == {
        "a": {"enabled": True},
        "z": [2, 1],
    }


def test_parameter_family_requires_completeness_and_full_points() -> None:
    payload = _valid_contract()
    payload["signature"]["arguments"] = [{"name": "theta", "type": "float", "domain": "all_real", "required": True}]
    payload["parameters"] = {
        "items": [
            {
                "name": "theta",
                "type": "real",
                "domain": "all_real",
                "units": "radian",
                "periodicity": 6.283185307179586,
                "excluded": [],
                "binding": "theta",
            }
        ],
        "quantifier": "all",
        "completeness": None,
        "diagnostic_points": [[0.0]],
    }

    with pytest.raises(ContractValidationError, match="completeness method"):
        parse_contract(payload)

    payload["parameters"]["completeness"] = "symbolic_identity"
    payload["parameters"]["diagnostic_points"] = [[]]
    with pytest.raises(ContractValidationError, match="bind every parameter"):
        parse_contract(payload)


def test_schema_two_preserves_nullable_exhaustive_argument_points() -> None:
    payload = _valid_contract()
    payload["schema_version"] = "2"
    payload["signature"]["arguments"] = [
        {"name": "error_qubit", "type": "int_or_none", "domain": "none_or_0_to_2", "required": True}
    ]
    payload["parameters"] = {
        "items": [
            {
                "name": "error_qubit",
                "type": "int_or_none",
                "domain": "none_or_0_to_2",
                "units": "discrete",
                "periodicity": None,
                "excluded": [],
                "binding": "error_qubit",
            }
        ],
        "quantifier": "exhaustive",
        "completeness": "finite_prompt_domain_exhaustive",
        "diagnostic_points": [[None], [0], [1], [2]],
    }

    contract = parse_contract(payload)

    assert contract.parameters.diagnostic_points == ((None,), (0,), (1,), (2,))


def test_distribution_rejects_quantum_phase_policy() -> None:
    payload = _valid_contract()
    payload["kind"] = "distribution"
    payload["observation"]["quantum"] = []
    payload["phase"] = {"global_phase_irrelevant": True, "relative_phase": "preserve"}

    with pytest.raises(ContractValidationError, match="global phase is meaningless"):
        parse_contract(payload)


def test_ancilla_policies_must_cover_exactly_ancilla_systems() -> None:
    payload = _valid_contract()
    payload["systems"]["items"].append(
        {"name": "work", "kind": "quantum", "role": "ancilla", "indices": [1], "dimension": 2}
    )

    with pytest.raises(ContractValidationError, match="exactly one policy"):
        parse_contract(payload)


def test_systems_reject_kind_role_mismatch_and_duplicate_indices() -> None:
    payload = _valid_contract()
    payload["systems"]["items"][0]["role"] = "classical_output"
    with pytest.raises(ContractValidationError, match="has a classical role"):
        parse_contract(payload)

    payload = _valid_contract()
    payload["systems"]["items"].append(
        {"name": "alias", "kind": "quantum", "role": "work", "indices": [0], "dimension": 2}
    )
    with pytest.raises(ContractValidationError, match="duplicate physical indices"):
        parse_contract(payload)


def test_registry_rejects_duplicates_and_normalizes_lookup() -> None:
    contract = parse_contract(_valid_contract())
    with pytest.raises(ContractValidationError, match="duplicate contract key"):
        ContractRegistry([contract, contract])

    registry = ContractRegistry([contract])
    assert registry.get("core", "2") == contract
    assert registry.hashes()[contract.key] == contract_hash(contract)


def test_registry_jsonl_reports_source_line() -> None:
    valid = json.dumps(_valid_contract())
    invalid = copy.deepcopy(_valid_contract())
    invalid["unknown"] = True

    with pytest.raises(ContractValidationError) as exc:
        ContractRegistry.from_jsonl(valid + "\n" + json.dumps(invalid))

    assert exc.value.path.startswith("line[2]")


def test_registry_diff_is_stable(tmp_path: Path) -> None:
    old_payload = _valid_contract()
    new_payload = copy.deepcopy(old_payload)
    new_payload["contract_version"] = "1.1.0"
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    old_path.write_text(json.dumps(old_payload) + "\n", encoding="utf-8")
    new_path.write_text(json.dumps(new_payload) + "\n", encoding="utf-8")

    changes = ContractRegistry.from_path(old_path).diff(ContractRegistry.from_path(new_path))

    assert len(changes) == 1
    assert changes[0].kind == "modified"
    assert changes[0].old_version == "1.0.0"
    assert changes[0].new_version == "1.1.0"
    assert changes[0].old_hash != changes[0].new_hash
