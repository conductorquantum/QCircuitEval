#!/usr/bin/env python3
"""Generate shared, framework-neutral QEC contracts and semantic targets."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")
QEC_IDS = tuple(f"qec{index:02d}" for index in range(1, 13))


@dataclass(frozen=True)
class QECSpec:
    """Prompt-derived semantic definition for one QEC task."""

    entry_point: str
    arguments: tuple[tuple[str, str, str], ...]
    points: tuple[tuple[Any, ...], ...]
    kind: str
    output_qubits: tuple[int, ...]
    min_qubits: int
    target: dict[str, Any]
    error_gate: tuple[int, tuple[str, ...], tuple[int, ...]] | None = None
    case_input_gates: tuple[tuple[int, tuple[str, ...], tuple[int, ...]], ...] = ()
    required_interactions: tuple[tuple[int, int], ...] = ()
    controlled_correction: tuple[tuple[int, ...], tuple[int, ...], int] | None = None
    connected_groups: tuple[tuple[int, ...], ...] = ()
    inter_before_intra_groups: tuple[tuple[int, ...], ...] = ()
    encoder_state_support: tuple[str, ...] = ()
    encoder_state_data_wires: tuple[int, ...] = ()
    encoder_state_ancilla_wires: tuple[int, ...] = ()
    encoder_state_reference_arguments: tuple[Any, ...] = ()


def _syndrome(error: int | None, supports: tuple[tuple[int, ...], ...]) -> str:
    bits = ["0" if error is None or error not in support else "1" for support in supports]
    return "".join(reversed(bits))


def _distribution_cases(
    points: tuple[tuple[Any, ...], ...],
    outcome: Callable[[tuple[Any, ...]], str],
) -> dict[str, Any]:
    return {
        "type": "argument_distribution_cases",
        "cases": [
            {
                "arguments": list(point),
                "distribution": {"type": "exact_distribution", "probabilities": {outcome(point): "1"}},
            }
            for point in points
        ],
    }


def _state(amplitudes: dict[str, str]) -> dict[str, Any]:
    return {"type": "exact_state", "amplitudes": amplitudes}


def _qec_specs() -> dict[str, QECSpec]:
    binary = ((0,), (1,))
    errors3 = ((None,), (0,), (1,), (2,))
    logical_errors3 = tuple((logical, error) for logical in (0, 1) for error in (None, 0, 1, 2))
    logical_pairs = tuple(itertools.product((0, 1), repeat=2))
    errors7 = tuple((error,) for error in (None, *range(7)))
    logical_errors7 = tuple((logical, error) for logical in (0, 1) for error in (None, *range(7)))
    errors9 = tuple((error,) for error in (None, *range(9)))
    logical_errors5 = tuple((logical, error) for logical in (0, 1) for error in (None, *range(5)))

    shor_support = tuple("".join(blocks) for blocks in itertools.product(("000", "111"), repeat=3))
    steane_zero = (
        "0000000",
        "0011110",
        "0101101",
        "0110011",
        "1001011",
        "1010101",
        "1100110",
        "1111000",
    )
    steane_one = tuple("".join("1" if bit == "0" else "0" for bit in value) for value in steane_zero)

    specs = {
        "qec01": QECSpec(
            "bit_flip_encode_decode",
            (("logical_bit", "int", "zero_or_one"),),
            binary,
            "distribution",
            (0,),
            3,
            _distribution_cases(binary, lambda point: str(point[0])),
            case_input_gates=((0, ("x", "paulix"), (0,)),),
            required_interactions=((0, 1), (0, 2)),
        ),
        "qec02": QECSpec(
            "bit_flip_syndrome",
            (("error_qubit", "int_or_none", "none_or_0_to_2"),),
            errors3,
            "distribution",
            (3, 4),
            5,
            _distribution_cases(errors3, lambda point: _syndrome(point[0], ((0, 1), (1, 2)))),
            (0, ("x", "paulix"), (0, 1, 2)),
            (),
            ((0, 3), (1, 3), (1, 4), (2, 4)),
        ),
        "qec03": QECSpec(
            "bit_flip_correct",
            (("logical_bit", "int", "zero_or_one"), ("error_qubit", "int_or_none", "none_or_0_to_2")),
            logical_errors3,
            "distribution",
            (0,),
            3,
            _distribution_cases(logical_errors3, lambda point: str(point[0])),
            (1, ("x", "paulix"), (0, 1, 2)),
            ((0, ("x", "paulix"), (0,)),),
            ((0, 1), (0, 2)),
            controlled_correction=(tuple(range(3)), tuple(range(5)), 2),
        ),
        "qec04": QECSpec(
            "phase_flip_correct",
            (("logical_bit", "int", "zero_or_one"), ("error_qubit", "int_or_none", "none_or_0_to_2")),
            logical_errors3,
            "distribution",
            (0,),
            3,
            _distribution_cases(logical_errors3, lambda point: str(point[0])),
            (1, ("z", "pauliz"), (0, 1, 2)),
            ((0, ("x", "paulix"), (0,)),),
            ((0, 1), (0, 2)),
            controlled_correction=(tuple(range(3)), tuple(range(5)), 2),
        ),
        "qec05": QECSpec(
            "repetition_logical_cnot",
            (("control_bit", "int", "zero_or_one"), ("target_bit", "int", "zero_or_one")),
            logical_pairs,
            "distribution",
            (0, 3),
            6,
            _distribution_cases(logical_pairs, lambda point: f"{point[0] ^ point[1]}{point[0]}"),
            case_input_gates=(
                (0, ("x", "paulix"), (0,)),
                (1, ("x", "paulix"), (3,)),
            ),
            required_interactions=((0, 3), (1, 4), (2, 5)),
        ),
        "qec06": QECSpec(
            "shor_encode_zero",
            (),
            (),
            "state",
            tuple(range(9)),
            9,
            _state(dict.fromkeys(shor_support, "1/sqrt(8)")),
            connected_groups=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        ),
        "qec07": QECSpec(
            "steane_encode",
            (("logical_bit", "int", "zero_or_one"),),
            binary,
            "state",
            tuple(range(7)),
            7,
            {
                "type": "argument_state_cases",
                "cases": [
                    {"arguments": [0], "state": _state(dict.fromkeys(steane_zero, "1/sqrt(8)"))},
                    {"arguments": [1], "state": _state(dict.fromkeys(steane_one, "1/sqrt(8)"))},
                ],
            },
            case_input_gates=((0, ("x", "paulix"), tuple(range(7))),),
        ),
        "qec08": QECSpec(
            "steane_z_syndrome",
            (("error_qubit", "int_or_none", "none_or_0_to_6"),),
            errors7,
            "distribution",
            (7, 8, 9),
            10,
            _distribution_cases(
                errors7,
                lambda point: _syndrome(point[0], ((0, 2, 4, 6), (1, 2, 5, 6), (3, 4, 5, 6))),
            ),
            (0, ("x", "paulix"), tuple(range(7))),
            (),
            (
                (0, 7),
                (2, 7),
                (4, 7),
                (6, 7),
                (1, 8),
                (2, 8),
                (5, 8),
                (6, 8),
                (3, 9),
                (4, 9),
                (5, 9),
                (6, 9),
            ),
            encoder_state_support=steane_zero,
            encoder_state_data_wires=tuple(range(7)),
            encoder_state_ancilla_wires=(7, 8, 9),
            encoder_state_reference_arguments=(None,),
        ),
        "qec09": QECSpec(
            "steane_x_correct",
            (("logical_bit", "int", "zero_or_one"), ("error_qubit", "int_or_none", "none_or_0_to_6")),
            logical_errors7,
            "distribution",
            tuple(range(7)),
            10,
            # Observe every decoded data qubit so a missing or mistargeted
            # correction leaves a visible residual on qubits 1 through 6.
            _distribution_cases(logical_errors7, lambda point: "000000" + str(point[0])),
            (1, ("x", "paulix"), tuple(range(7))),
            ((0, ("x", "paulix"), tuple(range(7))),),
            (
                (0, 7),
                (2, 7),
                (4, 7),
                (6, 7),
                (1, 8),
                (2, 8),
                (5, 8),
                (6, 8),
                (3, 9),
                (4, 9),
                (5, 9),
                (6, 9),
            ),
            controlled_correction=(tuple(range(7)), tuple(range(7, 11)), 2),
            encoder_state_support=steane_zero,
            encoder_state_data_wires=tuple(range(7)),
            encoder_state_ancilla_wires=(7, 8, 9),
            encoder_state_reference_arguments=(0, None),
        ),
        "qec10": QECSpec(
            "shor_z_syndrome",
            (("error_qubit", "int_or_none", "none_or_0_to_8"),),
            errors9,
            "distribution",
            tuple(range(9, 15)),
            15,
            _distribution_cases(
                errors9,
                lambda point: _syndrome(
                    point[0],
                    ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)),
                ),
            ),
            (0, ("x", "paulix"), tuple(range(9))),
            (),
            (
                (0, 9),
                (1, 9),
                (1, 10),
                (2, 10),
                (3, 11),
                (4, 11),
                (4, 12),
                (5, 12),
                (6, 13),
                (7, 13),
                (7, 14),
                (8, 14),
            ),
            # The prompt defines |0_L> directly as GHZ x GHZ x GHZ with no
            # construction-order mandate; the encoder-state requirement below
            # covers the anti-shortcut concern without over-constraining the
            # inter-/intra-block gate ordering.
            encoder_state_support=shor_support,
            encoder_state_data_wires=tuple(range(9)),
            encoder_state_ancilla_wires=tuple(range(9, 15)),
            encoder_state_reference_arguments=(None,),
        ),
        "qec11": QECSpec(
            "shor_x_syndrome",
            (("error_qubit", "int_or_none", "none_or_0_to_8"),),
            errors9,
            "distribution",
            (9, 10),
            11,
            _distribution_cases(
                errors9,
                lambda point: _syndrome(point[0], (tuple(range(6)), tuple(range(3, 9)))),
            ),
            (0, ("z", "pauliz"), tuple(range(9))),
            (),
            tuple((data, 9) for data in range(6)) + tuple((data, 10) for data in range(3, 9)),
            encoder_state_support=shor_support,
            encoder_state_data_wires=tuple(range(9)),
            encoder_state_ancilla_wires=(9, 10),
            encoder_state_reference_arguments=(None,),
        ),
        "qec12": QECSpec(
            "rep5_correct",
            (("logical_bit", "int", "zero_or_one"), ("error_qubit", "int_or_none", "none_or_0_to_4")),
            logical_errors5,
            "distribution",
            (0,),
            9,
            _distribution_cases(logical_errors5, lambda point: str(point[0])),
            (1, ("x", "paulix"), tuple(range(5))),
            ((0, ("x", "paulix"), (0,)),),
            (
                (0, 5),
                (1, 5),
                (1, 6),
                (2, 6),
                (2, 7),
                (3, 7),
                (3, 8),
                (4, 8),
            ),
            controlled_correction=(tuple(range(5)), tuple(range(5, 11)), 2),
        ),
    }
    if set(specs) != set(QEC_IDS):
        raise AssertionError("QEC specification set is incomplete")
    return specs


def _load_framework_tasks(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    tasks: dict[str, dict[str, dict[str, Any]]] = {}
    for framework in FRAMEWORKS:
        path = root / "src/qceval/assets/qec" / f"{framework}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        tasks[framework] = {str(record["task_id"]): record for record in records}
    if any(set(records) != set(QEC_IDS) for records in tasks.values()):
        raise ValueError("every framework must define exactly qec01 through qec12")
    return tasks


def _prompt_hashes(tasks: dict[str, dict[str, dict[str, Any]]], task_id: str) -> dict[str, str]:
    return {
        framework: hashlib.sha256(tasks[framework][task_id]["prompt"].encode()).hexdigest() for framework in FRAMEWORKS
    }


def _terminal_interfaces(spec: QECSpec) -> dict[str, Any]:
    qubits = list(spec.output_qubits)
    classical_bits = list(range(len(qubits)))
    return {
        "qiskit": {"kind": "measurement", "qubits": qubits, "classical_bits": classical_bits},
        "cirq": {"kind": "measurement", "qubits": qubits},
        "pennylane": {"mode": "probabilities", "wires": list(reversed(qubits))},
        "cudaq": {"kind": "measurement", "qubits": qubits},
    }


def _interaction_requirements(spec: QECSpec) -> dict[str, Any]:
    if not spec.required_interactions:
        return {}
    value: dict[str, Any] = {"required_interactions": [list(pair) for pair in spec.required_interactions]}
    if spec.entry_point in {
        "steane_z_syndrome",
        "steane_x_correct",
        "shor_z_syndrome",
        "rep5_correct",
    }:
        value["required_parity_interactions"] = [list(pair) for pair in spec.required_interactions]
    elif spec.entry_point != "shor_x_syndrome":
        # shor_x_syndrome admits both the ancilla-controlled extraction and the
        # equally correct H-conjugated extraction (H on the data support, CX
        # data->ancilla, H back), so only the undirected required_interactions
        # pairs are enforced for it.
        value["required_controlled_x_interactions"] = [list(pair) for pair in spec.required_interactions]
    if spec.entry_point == "bit_flip_encode_decode":
        value["required_any_interaction_sequences"] = [
            [[0, 1], [0, 2], [0, 2], [0, 1]],
            [[0, 1], [0, 2], [0, 1], [0, 2]],
            [[0, 2], [0, 1], [0, 1], [0, 2]],
            [[0, 2], [0, 1], [0, 2], [0, 1]],
        ]
    else:
        value["reject_canceling_interaction_padding"] = True
    return value


def _semantic_requirements(spec: QECSpec) -> dict[str, Any]:
    value: dict[str, Any] = {
        "min_num_qubits": spec.min_qubits,
        "forbid_state_preparation": True,
        "forbidden_gate_families": [
            "unitary",
            "unitarygate",
            "matrixgate",
            "qubitunitary",
            "dense_unitary",
        ],
        "forbidden_probability_methods": {
            "qiskit": ["returned_counts", "returned_probabilities", "returned_unitary"],
            "cirq": ["returned_counts", "returned_probabilities", "returned_unitary"],
            "pennylane": ["returned_probabilities"],
            "cudaq": ["returned_counts", "returned_probabilities", "returned_unitary"],
        },
        "forbidden_imports": ["stim", "pymatching", "ldpc", "flamingpy", "qiskit_qec"],
        "forbidden_calls": [
            "NoiseModel",
            "noise_model",
            "AerSimulator",
            "Stim",
            "Sampler",
            "sample",
            "minimize",
            "optimizer",
            "decoder",
        ],
    }
    value.update(_interaction_requirements(spec))
    if spec.error_gate is not None:
        argument_index, gates, wires = spec.error_gate
        value["argument_conditioned_gate"] = {
            "argument_index": argument_index,
            "gate_names": list(gates),
            "wires": list(wires),
            "verification": "program_ir_concrete_case",
        }
    case_rules = [
        {
            "activation": "equals_one",
            "argument_index": argument_index,
            "gate_names": list(gates),
            "wires": list(wires),
        }
        for argument_index, gates, wires in spec.case_input_gates
    ]
    if spec.error_gate is not None:
        argument_index, gates, wires = spec.error_gate
        case_rules.append(
            {
                "activation": "selected_wire",
                "argument_index": argument_index,
                "gate_names": list(gates),
                "wires": list(wires),
            }
        )
    if case_rules:
        value["case_program_invariance"] = {
            "reference_arguments": list(spec.points[0]),
            "allowed_case_deltas": case_rules,
        }
    if spec.controlled_correction is not None:
        targets, controls, minimum = spec.controlled_correction
        value["required_controlled_correction"] = {
            "gate_family": "x",
            "target_wires": list(targets),
            "control_wires": list(controls),
            "min_controls": minimum,
        }
    if spec.connected_groups:
        value["required_connected_interaction_groups"] = [list(group) for group in spec.connected_groups]
    if spec.inter_before_intra_groups:
        value["required_inter_group_before_intra_group"] = [list(group) for group in spec.inter_before_intra_groups]
    if spec.encoder_state_support:
        value["required_encoder_state_before_ancilla_use"] = {
            "reference_arguments": list(spec.encoder_state_reference_arguments),
            "data_wires": list(spec.encoder_state_data_wires),
            "ancilla_wires": list(spec.encoder_state_ancilla_wires),
            "positive_uniform_support": list(spec.encoder_state_support),
        }
    return value


def _contract(
    task_id: str,
    spec: QECSpec,
    tasks: dict[str, dict[str, dict[str, Any]]],
    artifact_hash: str,
) -> dict[str, Any]:
    state_kind = spec.kind == "state"
    output_width = len(spec.output_qubits)
    max_qubits = {
        "qec01": 5,
        "qec02": 7,
        "qec03": 5,
        "qec04": 5,
        "qec05": 8,
        "qec09": 11,
        "qec12": 11,
    }.get(task_id, spec.min_qubits)
    parameters = [
        {
            "name": name,
            "type": value_type,
            "domain": domain,
            "units": "discrete",
            "periodicity": None,
            "excluded": [],
            "binding": name,
        }
        for name, value_type, domain in spec.arguments
    ]
    systems = [
        {
            "name": "logical_output" if state_kind else "classical_output",
            "kind": "quantum" if state_kind else "classical",
            "role": "logical_output" if state_kind else "classical_output",
            "indices": list(spec.output_qubits) if state_kind else list(range(output_width)),
            "dimension": 2**output_width,
        }
    ]
    requirements = [
        {
            "id": "framework_prompts",
            "kind": "prompt_hashes",
            "source": "four_framework_qec_prompt_audit",
            "value": _prompt_hashes(tasks, task_id),
        },
        {
            "id": "terminal_observation",
            "kind": "framework_interface",
            "source": "four_framework_qec_prompt_audit",
            "value": _terminal_interfaces(spec),
        },
        {
            "id": "semantic_requirements",
            "kind": "prompt_semantics",
            "source": "qec_prompt_and_stabilizer_specification",
            "value": _semantic_requirements(spec),
        },
    ]
    return {
        "schema_version": "2",
        "suite": "qec",
        "task_id": task_id,
        "contract_version": "1.7.0",
        "kind": spec.kind,
        "shadow_only": False,
        "audit_status": "reviewed",
        "signature": {
            "entry_point": spec.entry_point,
            "arguments": [
                {"name": name, "type": value_type, "domain": domain, "required": True}
                for name, value_type, domain in spec.arguments
            ],
            "return_type": "framework_quantum_program",
        },
        "systems": {"items": systems},
        "observation": {
            "quantum": ["logical_output"] if state_kind else [],
            "classical": [] if state_kind else ["classical_output"],
            "ignored": [],
            "marginalize": [],
            "bit_order": "not_applicable" if state_kind else "little_endian",
            "postselection": None,
        },
        "phase": {
            "global_phase_irrelevant": state_kind,
            "relative_phase": "preserve" if state_kind else "not_applicable",
        },
        "ancillas": {"items": []},
        "parameters": {
            "items": parameters,
            "quantifier": "none" if not parameters else "exhaustive",
            "completeness": None if not parameters else "finite_prompt_domain_exhaustive",
            "diagnostic_points": [list(point) for point in spec.points],
        },
        "approximation": {
            "mode": "exact",
            "metric": "trace_distance" if state_kind else "hellinger_infidelity",
            "tolerance": 1e-9,
            "uncertainty": 1e-12,
            "error_budget": 0.0,
        },
        "target": {
            "id": f"{task_id}-{'exact-state' if state_kind else 'exact-distribution-family'}",
            "version": "1.0.0",
            "sha256": artifact_hash,
            "source": "prompt_stabilizers_and_independent_gf2_derivation",
            "manifest": "targets/qec/manifest.json",
            "independent_derivations": 2,
        },
        "routing": {
            "primary": [
                {
                    "engine": "state_exact" if state_kind else "distribution_exact",
                    "capabilities": (
                        ["state", "static", "pure_state", "terminal_measurement_removal", "framework_neutral_ir"]
                        if state_kind
                        else ["distribution", "exact", "partial_observation", "framework_neutral_ir"]
                    ),
                    "cross_check": False,
                }
            ],
            "fallback": [],
        },
        "limits": {
            "wall_seconds": 30.0,
            "cpu_seconds": 30.0,
            "memory_mib": 4096,
            "max_qubits": max_qubits,
            "max_dimension": max(2**max_qubits, 2**output_width),
            "max_cases": max(1, len(spec.points), 2**output_width),
            "max_branches": 4096,
            "max_expression_nodes": 50000,
        },
        "requirements": requirements,
        "diagnostics": [
            {"id": "legacy_case_table", "kind": "legacy_comparison", "enabled": True},
            {"id": "gate_counts", "kind": "gate_counts", "enabled": True},
            {"id": "resource_usage", "kind": "resource_usage", "enabled": True},
        ],
    }


def _artifact(task_id: str, spec: QECSpec) -> dict[str, Any]:
    return {
        "format": "semantic_target_spec_v1",
        "suite": "qec",
        "task_id": task_id,
        "kind": spec.kind,
        "target": spec.target,
    }


def _manifest(task_id: str, spec: QECSpec, artifact_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "suite": "qec",
        "task_id": task_id,
        "target_id": f"{task_id}-{'exact-state' if spec.kind == 'state' else 'exact-distribution-family'}",
        "target_version": "1.0.0",
        "artifact": "target.json",
        "artifact_sha256": artifact_hash,
        "artifact_format": "semantic_target_spec_v1",
        "dimensions": [2 ** len(spec.output_qubits)],
        "source": "Natural-language QEC prompts shared across all four framework assets.",
        "normalization": "QCircuitEval little-endian classical output; state basis q[n-1]...q[0].",
        "generator_command": "uv run python ci/generate_qec_semantic_assets.py --check",
        "derivations": [
            {
                "id": "stabilizer-or-truth-table",
                "method": "analytic_gf2_or_stabilizer_derivation",
                "narrative": "Derive the contracted state or syndrome relation directly from the prompt.",
                "evidence": "The generated target follows the declared codewords, stabilizers, and logical map.",
            },
            {
                "id": "independent-invariants",
                "method": "exhaustive_domain_and_code_invariants",
                "narrative": "Cross-check every declared finite input and code-space invariant independently.",
                "evidence": "All no-error and single-error points are enumerated; state targets satisfy unit norm.",
            },
        ],
        "invariants": [
            "Every finite prompt argument is covered exactly once.",
            "Exact distributions are normalized point masses.",
            "Exact codeword states preserve relative phase and have unit norm.",
        ],
        "applicable_frameworks": list(FRAMEWORKS),
        "known_ambiguities": [],
        "resolution_adrs": [],
    }


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _pretty_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def generated_assets(root: Path) -> dict[Path, bytes]:
    """Return every deterministic QEC contract and target asset."""

    tasks = _load_framework_tasks(root)
    specs = _qec_specs()
    contracts: list[dict[str, Any]] = []
    manifest_tasks: dict[str, Any] = {}
    target_tasks: dict[str, Any] = {}
    for task_id in QEC_IDS:
        spec = specs[task_id]
        entry_points = {tasks[framework][task_id]["entry_point"] for framework in FRAMEWORKS}
        if entry_points != {spec.entry_point}:
            raise ValueError(f"{task_id}: framework entry points differ from the shared specification")
        artifact = _artifact(task_id, spec)
        # Hashes cover the canonical bytes of the per-task document, so grouping
        # documents into one suite-level artifact leaves every hash unchanged.
        artifact_hash = hashlib.sha256(_canonical_bytes(artifact)).hexdigest()
        contracts.append(_contract(task_id, spec, tasks, artifact_hash))
        manifest_tasks[task_id] = _manifest(task_id, spec, artifact_hash)
        target_tasks[task_id] = artifact
    target_root = Path("src/qceval/assets/targets/qec")
    return {
        target_root / "manifest.json": _pretty_json_bytes(
            {"schema_version": "1", "suite": "qec", "tasks": manifest_tasks}
        ),
        target_root / "target.json": _pretty_json_bytes({"schema_version": "1", "suite": "qec", "tasks": target_tasks}),
        Path("src/qceval/assets/contracts/qec.jsonl"): b"".join(_canonical_bytes(value) for value in contracts),
    }


def main() -> None:
    """Write generated assets or verify the checked-in copies."""

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check generated assets (default)")
    mode.add_argument("--write", action="store_true", help="replace generated assets")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    mismatches: list[str] = []
    for relative, expected in generated_assets(root).items():
        path = root / relative
        actual = path.read_bytes() if path.exists() else None
        if actual == expected:
            continue
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
        else:
            mismatches.append(str(relative))
    if mismatches:
        parser.error("generated assets differ: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()
