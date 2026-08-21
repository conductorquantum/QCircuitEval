"""Deterministic generation helpers for the core semantic audit.

The source audit is deliberately declarative and separate from framework
canonical programs. Generated contracts and targets are checked into the
package so normal grading never needs to regenerate them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qceval.assets._resources import read_text, target_resource, task_resource
from qceval.semantics._core_contracts import CORE_CONTRACT_VERSION, FRAMEWORKS, contract_payload
from qceval.semantics.contracts import (
    ContractRegistry,
    canonical_contract_json,
    contract_to_dict,
    parse_contract,
)
from qceval.semantics.targets import PILOT_TASK_IDS, canonical_target_bytes

SOURCE_RESOURCE = "core-audit-source.json"
_STRUCTURAL_CHECK_FIELDS = frozenset(
    {
        "forbid_returned_counts",
        "forbid_returned_probabilities",
        "forbid_returned_unitary",
        "forbid_full_register_dense_unitary",
        "forbidden_gate_family_counts",
        "forbidden_measurement_qubits",
        "max_measurement_count",
        "min_any_gate_family_counts",
        "min_entangling_gate_count",
        "min_gate_family_counts",
        "min_gate_family_group_counts",
        "min_measurement_count",
        "min_non_measurement_operation_count",
        "min_num_qubits",
        "require_net_unitary_entangling",
        "required_any_interaction_groups",
        "required_interaction_groups",
        "required_interactions",
        "required_measurement_qubits",
    }
)


def load_core_audit_source() -> dict[str, dict[str, Any]]:
    """Load the declarative non-pilot core audit source.

    Returns:
        Mapping keyed by normalized task id.
    """
    payload = json.loads(read_text("contracts", SOURCE_RESOURCE))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "tasks",
    }:
        raise ValueError(
            "core audit source must contain schema_version and tasks",
        )
    if payload["schema_version"] != "1":
        raise ValueError("unsupported core audit source version")
    tasks = payload["tasks"]
    if not isinstance(tasks, dict):
        raise ValueError("core audit tasks must be an object")
    expected = set(_asset_records("qiskit")) - set(PILOT_TASK_IDS)
    if set(tasks) != expected:
        raise ValueError(f"core audit task coverage mismatch: {sorted(set(tasks) ^ expected)}")
    return tasks


def generated_target_payload(task_id: str) -> dict[str, Any]:
    """Return the prompt-derived target payload for one non-pilot task.

    Args:
        task_id: Core task id.

    Returns:
        JSON-compatible target artifact.
    """
    normalized = str(task_id).zfill(2)
    spec = load_core_audit_source()[normalized]
    return {
        "format": "semantic_target_spec_v1",
        "kind": spec["kind"],
        "suite": "core",
        "target": spec["target"],
        "task_id": normalized,
    }


def generated_core_assets() -> dict[Path, bytes]:
    """Generate every non-source contract, target, and manifest asset.

    Returns:
        Mapping of repository-relative asset paths to expected bytes.
    """
    specs = load_core_audit_source()
    files: dict[Path, bytes] = {}
    parity = prompt_parity_report()
    contracts = _pilot_contract_payloads(parity)
    # Pilot entries in the suite-level manifest and target files are curated
    # rather than generated, so start from the packaged documents and refresh
    # every audit-sourced task in place.
    manifest_doc = _packaged_targets_json("manifest.json")
    target_doc = _packaged_targets_json("target.json")
    for manifest in manifest_doc["tasks"].values():
        manifest.pop("review", None)
        manifest["artifact"] = "target.json"
    for task_id, spec in sorted(specs.items()):
        payload = generated_target_payload(task_id)
        artifact_hash = hashlib.sha256(canonical_target_bytes(payload)).hexdigest()
        target_doc["tasks"][task_id] = payload
        manifest_doc["tasks"][task_id] = _manifest_payload(
            task_id,
            spec,
            "target.json",
            artifact_hash,
        )
        contracts.append(
            contract_payload(
                task_id,
                spec,
                artifact_hash,
                parity["tasks"][task_id],
            )
        )
    files[Path("src/qceval/assets/targets/core/manifest.json")] = _pretty_json_bytes(manifest_doc)
    files[Path("src/qceval/assets/targets/core/target.json")] = _pretty_json_bytes(target_doc)
    validated = [parse_contract(payload) for payload in contracts]
    registry = ContractRegistry(validated)
    contract_lines = [canonical_contract_json(contract) for contract in registry]
    files[Path("src/qceval/assets/contracts/core.jsonl")] = ("\n".join(contract_lines) + "\n").encode()
    return files


def _packaged_targets_json(name: str) -> dict[str, Any]:
    payload = json.loads(target_resource("core", name).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
        raise ValueError(f"consolidated core {name} is malformed")
    return payload


def prompt_parity_report() -> dict[str, Any]:
    """Return stable cross-framework prompt and entry-point evidence.

    Returns:
        Report with task coverage, entry-point parity, and prompt hashes.
    """
    records = {framework: _asset_records(framework) for framework in FRAMEWORKS}
    task_ids = sorted(records["qiskit"])
    if any(sorted(records[framework]) != task_ids for framework in FRAMEWORKS):
        raise ValueError("core framework task coverage differs")
    tasks: dict[str, Any] = {}
    for task_id in task_ids:
        entry_points = {framework: records[framework][task_id]["entry_point"] for framework in FRAMEWORKS}
        if len(set(entry_points.values())) != 1:
            raise ValueError(f"task {task_id}: entry points differ")
        tasks[task_id] = {
            "entry_point": next(iter(entry_points.values())),
            "prompt_sha256": {
                framework: hashlib.sha256(records[framework][task_id]["prompt"].encode()).hexdigest()
                for framework in FRAMEWORKS
            },
            "structural_constraints": _task_structural_constraints(records, task_id),
        }
    return {
        "frameworks": list(FRAMEWORKS),
        "task_count": len(task_ids),
        "tasks": tasks,
    }


def _manifest_payload(
    task_id: str,
    spec: dict[str, Any],
    artifact_name: str,
    artifact_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "suite": "core",
        "task_id": task_id,
        "target_id": (f"core-{task_id}-{spec['target']['type'].replace('_', '-')}"),
        "target_version": "1.0.0",
        "artifact": artifact_name,
        "artifact_sha256": artifact_hash,
        "artifact_format": "semantic_target_spec_v1",
        "dimensions": spec.get(
            "target_dimensions",
            [
                2
                ** len(
                    spec.get(
                        "logical_qubits",
                        spec.get("output_qubits", []),
                    )
                )
            ],
        ),
        "source": "Natural-language prompts in all four core framework assets.",
        "normalization": spec.get(
            "normalization",
            "QCircuitEval q[n-1]...q[0] bit order.",
        ),
        "generator_command": ("uv run python ci/generate_semantic_assets.py --check"),
        "derivations": [
            {
                "id": "prompt-formalization",
                "method": "equation_or_truth_table",
                "narrative": ("Formalize the prompt as the checked semantic target artifact."),
                "evidence": str(spec["derivation"]),
            },
            {
                "id": "independent-invariants",
                "method": "analytic_cross_check",
                "narrative": ("Challenge the target using a separately stated invariant or construction."),
                "evidence": str(spec["crosscheck"]),
            },
        ],
        "invariants": list(spec.get("invariants", [spec["crosscheck"]])),
        "applicable_frameworks": list(FRAMEWORKS),
        "known_ambiguities": list(spec.get("ambiguities", [])),
        "resolution_adrs": [],
    }


def _pilot_contract_payloads(parity: dict[str, Any]) -> list[dict[str, Any]]:
    """Return pilot contracts with prompt hashes refreshed from task assets.

    Args:
        parity: Current cross-framework prompt parity report.

    Returns:
        Pilot contract payloads with complete prompt-hash requirements.
    """
    registry = ContractRegistry.from_package("core")
    payloads = []
    for task_id in PILOT_TASK_IDS:
        payload = contract_to_dict(registry.get("core", task_id))
        payload["contract_version"] = CORE_CONTRACT_VERSION
        primary = payload["routing"]["primary"]
        if not primary:
            raise ValueError(f"pilot task {task_id} has no existing primary route")
        route = dict(primary[0])
        route["cross_check"] = False
        payload["routing"] = {"primary": [route], "fallback": []}
        requirement = {
            "id": "framework_prompts",
            "kind": "prompt_hashes",
            "source": "four_framework_prompt_audit",
            "value": parity["tasks"][task_id]["prompt_sha256"],
        }
        structural = {
            "id": "structural_constraints",
            "kind": "structural_constraints",
            "source": "historical_structural_audit_calibrated_against_program_ir",
            "value": parity["tasks"][task_id]["structural_constraints"],
        }
        requirements = [
            item
            for item in payload["requirements"]
            if item["id"] not in {"framework_prompts", "structural_constraints"}
        ]
        payload["requirements"] = [requirement, structural, *requirements]
        payloads.append(payload)
    return payloads


def _asset_records(framework: str) -> dict[str, dict[str, Any]]:
    resource = task_resource("core", framework)
    records = [json.loads(line) for line in resource.read_text(encoding="utf-8").splitlines() if line]
    return {str(record["task_id"]).zfill(2): record for record in records}


def _structural_policy(record: dict[str, Any]) -> dict[str, Any]:
    canonical = record.get("canonical_class", {})
    if not isinstance(canonical, dict):
        return {}
    checks = canonical.get("structure_checks", canonical.get("metadata_checks", {}))
    policy = (
        {key: value for key, value in checks.items() if key in _STRUCTURAL_CHECK_FIELDS}
        if isinstance(checks, dict)
        else {}
    )
    for key in ("forbidden_imports", "forbidden_calls"):
        if key in canonical:
            policy[key] = canonical[key]
    return policy


def _task_structural_constraints(
    records: dict[str, dict[str, dict[str, Any]]],
    task_id: str,
) -> dict[str, Any]:
    """Return calibrated per-framework policy with IR-safe shared floors."""
    policies = {framework: _structural_policy(records[framework][task_id]) for framework in FRAMEWORKS}
    # CUDA-Q sampling may add terminal observation implicitly to an unmeasured
    # kernel, so source-level measurement totals are not authoritative.
    for key in (
        "min_measurement_count",
        "max_measurement_count",
        "required_measurement_qubits",
        "forbidden_measurement_qubits",
    ):
        policies["cudaq"].pop(key, None)
    # Historical zeroes often reflected framework metadata blind spots rather
    # than task intent. Program IR represents controls and multi-wire gates
    # uniformly, so the strongest audited entangling floor is now portable.
    entangling_floor = max(
        (int(policy.get("min_entangling_gate_count", 0)) for policy in policies.values()),
        default=0,
    )
    if entangling_floor:
        for policy in policies.values():
            policy["min_entangling_gate_count"] = entangling_floor
    # Every task must be built from individual gates rather than one opaque
    # matrix spanning the complete program. The shared grading note in every
    # prompt states this rule, so the ban is uniform across the suite.
    for policy in policies.values():
        policy["forbid_full_register_dense_unitary"] = True
    return {"frameworks": policies}


def _pretty_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


__all__ = [
    "FRAMEWORKS",
    "SOURCE_RESOURCE",
    "generated_core_assets",
    "generated_target_payload",
    "load_core_audit_source",
    "prompt_parity_report",
]
