"""Contract payload construction for the core semantic audit."""

from __future__ import annotations

from typing import Any

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")
CORE_CONTRACT_VERSION = "1.5.0"

_METRIC = {
    "state": "trace_distance",
    "total_unitary": "operator_norm",
    "channel": "normalized_choi_frobenius",
    "instrument": "max_branch_choi_distance",
    "distribution": "hellinger_infidelity",
    "classical_io": "max_case_error",
    "objective": "objective_gap",
}
_ENGINE = {
    "state": "state_exact",
    "total_unitary": "unitary_exact",
    "channel": "channel_exact",
    "instrument": "instrument_exact",
    "distribution": "distribution_exact",
    "classical_io": "classical_io_exhaustive",
    "objective": "objective_exact",
}


def contract_payload(
    task_id: str,
    spec: dict[str, Any],
    artifact_hash: str,
    prompt_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build one strict core contract payload.

    Args:
        task_id: Normalized core task identifier.
        spec: Audited semantic task specification.
        artifact_hash: SHA-256 digest of the independent target artifact.
        prompt_evidence: Entry-point and prompt-hash evidence.

    Returns:
        JSON-compatible strict semantic contract payload.

    Raises:
        KeyError: If required task or prompt evidence is absent.
    """
    kind = str(spec["kind"])
    systems, observation = _systems_and_observation(kind, spec)
    parameters = _parameters(spec)
    approximation = _approximation(kind, spec)
    audit_status = str(spec.get("audit_status", "provisional"))
    requirements = [
        _requirement(
            "framework_prompts",
            "prompt_hashes",
            prompt_evidence["prompt_sha256"],
        ),
        _requirement(
            "terminal_observation",
            "framework_interface",
            _terminal_observation(spec),
        ),
        _requirement(
            "semantic_requirements",
            "prompt_semantics",
            spec.get("hard", {}),
        ),
    ]
    structural = prompt_evidence.get("structural_constraints")
    if structural:
        requirements.append(
            _requirement(
                "structural_constraints",
                "structural_constraints",
                structural,
                source="historical_structural_audit_calibrated_against_program_ir",
            )
        )
    if blocker := spec.get("blocker"):
        requirements.append(
            _requirement(
                "audit_blocker",
                "blocking_semantic_question",
                blocker,
            )
        )
    target_id = f"core-{task_id}-{spec['target']['type'].replace('_', '-')}"
    return {
        "schema_version": "1",
        "suite": "core",
        "task_id": task_id,
        "contract_version": CORE_CONTRACT_VERSION,
        "kind": kind,
        "shadow_only": audit_status != "reviewed",
        "audit_status": audit_status,
        "signature": {
            "entry_point": prompt_evidence["entry_point"],
            "arguments": spec.get("arguments", []),
            "return_type": "framework_quantum_program",
        },
        "systems": {"items": systems},
        "observation": observation,
        "phase": _phase(observation),
        "ancillas": {"items": _ancilla_policies(spec)},
        "parameters": parameters,
        "approximation": approximation,
        "target": {
            "id": target_id,
            "version": "1.0.0",
            "sha256": artifact_hash,
            "source": "four_framework_prompts_and_independent_mathematical_spec",
            "manifest": "targets/core/manifest.json",
            "independent_derivations": 2,
        },
        "routing": {
            "primary": [_route(kind, approximation)],
            "fallback": [],
        },
        "limits": _limits(spec),
        "requirements": requirements,
        "diagnostics": [
            {
                "id": "legacy_verdict",
                "kind": "legacy_comparison",
                "enabled": True,
            },
            {
                "id": "resource_usage",
                "kind": "resource_usage",
                "enabled": True,
            },
        ],
    }


def _route(
    kind: str,
    approximation: dict[str, Any],
) -> dict[str, Any]:
    del approximation
    return {
        "engine": _ENGINE[kind],
        "capabilities": [kind, "framework_neutral_ir"],
        "cross_check": False,
    }


def _systems_and_observation(
    kind: str,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quantum: list[str] = []
    classical: list[str] = []
    systems: list[dict[str, Any]] = []
    logical = list(spec.get("logical_qubits", []))
    output = list(spec.get("output_qubits", []))
    if kind == "channel":
        inputs = list(spec["input_qubits"])
        systems.extend(
            [
                _system("logical_input", "quantum", "logical_input", inputs),
                _system("logical_output", "quantum", "logical_output", logical),
            ]
        )
        quantum.append("logical_output")
    elif kind == "instrument":
        systems.append(
            _system(
                "conditional_output",
                "quantum",
                "logical_output",
                logical,
            )
        )
        systems.append(
            _system(
                "classical_output",
                "classical",
                "classical_output",
                range(len(output)),
            )
        )
        quantum.append("conditional_output")
        classical.append("classical_output")
    elif kind in {"state", "total_unitary", "objective"}:
        role = "logical_io" if kind == "total_unitary" else "logical_output"
        systems.append(_system("logical_output", "quantum", role, logical))
        quantum.append("logical_output")
    else:
        systems.append(
            _system(
                "classical_output",
                "classical",
                "classical_output",
                range(len(output)),
            )
        )
        classical.append("classical_output")
    for index, ancilla in enumerate(spec.get("ancillas", [])):
        systems.append(
            _system(
                f"ancilla_{index}",
                "quantum",
                "ancilla",
                ancilla["indices"],
            )
        )
    return systems, {
        "quantum": quantum,
        "classical": classical,
        "ignored": [],
        "marginalize": [],
        "bit_order": "little_endian" if classical else "not_applicable",
        "postselection": spec.get("postselection"),
    }


def _system(
    name: str,
    kind: str,
    role: str,
    indices: Any,
) -> dict[str, Any]:
    values = list(indices)
    return {
        "name": name,
        "kind": kind,
        "role": role,
        "indices": values,
        "dimension": 2 ** len(values),
    }


def _ancilla_policies(spec: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "system": f"ancilla_{index}",
            "initial": item["initial"],
            "final": item["final"],
        }
        for index, item in enumerate(spec.get("ancillas", []))
    ]


def _parameters(spec: dict[str, Any]) -> dict[str, Any]:
    items = spec.get("parameters", [])
    if names := spec.get("parameter_names"):
        items = [
            {
                "name": name,
                "type": spec.get("parameter_type", "real"),
                "domain": spec.get("parameter_domain", "all_real"),
                "units": spec.get("parameter_units", "radian"),
                "periodicity": spec.get(
                    "parameter_periodicity",
                    6.283185307179586,
                ),
                "excluded": [],
                "binding": name,
            }
            for name in names
        ]
    if not items:
        return {
            "items": [],
            "quantifier": "none",
            "completeness": None,
            "diagnostic_points": [],
        }
    return {
        "items": items,
        "quantifier": spec.get("parameter_quantifier", "all"),
        "completeness": spec.get(
            "parameter_completeness",
            "analytic_family_identity",
        ),
        "diagnostic_points": spec.get("diagnostic_points", []),
    }


def _approximation(
    kind: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    if value := spec.get("approximation"):
        return value
    return {
        "mode": "exact",
        "metric": _METRIC[kind],
        "tolerance": 1e-9,
        "uncertainty": 1e-12,
        "error_budget": 0.0,
    }


def _phase(observation: dict[str, Any]) -> dict[str, Any]:
    quantum = bool(observation["quantum"])
    return {
        "global_phase_irrelevant": quantum,
        "relative_phase": "preserve" if quantum else "not_applicable",
    }


def _terminal_observation(spec: dict[str, Any]) -> dict[str, Any]:
    if interfaces := spec.get("interfaces"):
        return dict(interfaces)
    output = list(spec.get("output_qubits", []))
    mode = spec.get("measurement", "terminal")
    return {
        "qiskit": {
            "mode": mode,
            "qubits": output,
            "render_order": list(reversed(output)),
        },
        "cirq": {
            "mode": mode,
            "qubits": output,
            "render_order": list(reversed(output)),
            "key": "result",
        },
        "pennylane": {
            "mode": "probabilities" if mode == "terminal" else mode,
            "wires": list(reversed(output)),
        },
        "cudaq": {
            "mode": mode,
            "qubits": output,
            "render_order": list(reversed(output)),
        },
    }


def _limits(spec: dict[str, Any]) -> dict[str, Any]:
    qubits = int(spec["total_qubits"])
    max_qubits = int(spec.get("max_qubits", qubits))
    max_dimension = int(spec.get("max_dimension", 2 ** min(max_qubits, 12)))
    max_cases = int(spec.get("max_cases", max(2 ** min(qubits, 12), 1)))
    return {
        "wall_seconds": 10.0,
        "cpu_seconds": 10.0,
        "memory_mib": 2048,
        "max_qubits": max_qubits,
        "max_dimension": max_dimension,
        "max_cases": max_cases,
        "max_branches": 4096,
        "max_expression_nodes": 10000,
    }


def _requirement(
    requirement_id: str,
    kind: str,
    value: Any,
    *,
    source: str = "four_framework_prompt_audit",
) -> dict[str, Any]:
    return {
        "id": requirement_id,
        "kind": kind,
        "source": source,
        "value": value,
    }
