"""Authoritative anti-shortcut requirement tests."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from qceval.evals.parser.family import _prove_rotation_family
from qceval.evals.tasks import load_tasks
from qceval.semantics.contracts import parse_contract
from qceval.semantics.core_audit import _structural_policy, prompt_parity_report
from qceval.semantics.lowering import default_lowering_registry
from qceval.semantics.lowering.base import SourceMetadata
from qceval.semantics.verifiers.requirements import verify_program_requirements
from qceval.semantics.verifiers.result import SemanticStatus


def _contract(
    structural: dict[str, Any] | None = None,
    semantic: dict[str, Any] | None = None,
) -> Any:
    requirements = []
    if structural is not None:
        requirements.append(
            {
                "id": "structural_constraints",
                "kind": "structural_constraints",
                "source": "test",
                "value": structural,
            }
        )
    if semantic is not None:
        requirements.append(
            {
                "id": "semantic_requirements",
                "kind": "prompt_semantics",
                "source": "test",
                "value": semantic,
            }
        )
    return parse_contract(
        {
            "schema_version": "1",
            "suite": "core",
            "task_id": "99",
            "contract_version": "1.0.0",
            "kind": "state",
            "shadow_only": False,
            "audit_status": "reviewed",
            "signature": {
                "entry_point": "answer",
                "arguments": [],
                "return_type": "quantum_program",
            },
            "systems": {
                "items": [
                    {
                        "name": "output",
                        "kind": "quantum",
                        "role": "logical_output",
                        "indices": [0],
                        "dimension": 2,
                    }
                ]
            },
            "observation": {
                "quantum": ["output"],
                "classical": [],
                "ignored": [],
                "marginalize": [],
                "bit_order": "not_applicable",
                "postselection": None,
            },
            "phase": {
                "global_phase_irrelevant": True,
                "relative_phase": "preserve",
            },
            "ancillas": {"items": []},
            "parameters": {
                "items": [],
                "quantifier": "none",
                "completeness": None,
                "diagnostic_points": [],
            },
            "approximation": {
                "mode": "exact",
                "metric": "trace_distance",
                "tolerance": 1e-9,
                "uncertainty": 1e-12,
                "error_budget": 0.0,
            },
            "target": {
                "id": "task_99",
                "version": "1.0.0",
                "sha256": "a" * 64,
                "source": "test",
                "manifest": "targets/core/99/manifest.json",
                "independent_derivations": 1,
            },
            "routing": {
                "primary": [
                    {
                        "engine": "state_exact",
                        "capabilities": ["state", "framework_neutral_ir"],
                        "cross_check": False,
                    }
                ],
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
            "requirements": requirements,
            "diagnostics": [],
        }
    )


def _program(circuit: QuantumCircuit) -> Any:
    result = (
        default_lowering_registry()
        .get("qiskit")
        .lower(
            circuit,
            SourceMetadata("qiskit", None, None),
            None,
        )
    )
    assert result.program is not None
    return result.program


def _reason(
    circuit: QuantumCircuit,
    structural: dict[str, Any] | None = None,
    semantic: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    unitary: Any | None = None,
    source_code: str | None = None,
    arguments: tuple[Any, ...] = (),
) -> str | None:
    contract = _contract(structural, semantic)
    result = verify_program_requirements(
        contract,
        _program(circuit),
        framework="qiskit",
        execution_metadata=metadata or {},
        source_code=source_code,
        candidate_unitary=unitary,
        arguments=arguments,
    )
    return None if result is None else result.reason


def test_minimum_qubit_and_entangling_floors_are_hard() -> None:
    local = QuantumCircuit(1)
    local.x(0)
    assert _reason(local, {"min_num_qubits": 2}) == "requirement_failed:min_num_qubits"

    two_qubit_local = QuantumCircuit(2)
    two_qubit_local.x(0)
    assert _reason(two_qubit_local, {"min_entangling_gate_count": 1}) == "requirement_failed:min_entangling_gate_count"


def test_net_unitary_rejects_canceling_entangler_padding() -> None:
    padded = QuantumCircuit(2)
    padded.x(0)
    padded.cx(0, 1)
    padded.cx(0, 1)
    unitary = np.asarray(Operator(padded).data)
    assert (
        _reason(
            padded,
            {"min_entangling_gate_count": 1},
            unitary=unitary,
        )
        == "requirement_failed:net_unitary_nonlocal"
    )

    entangling = QuantumCircuit(2)
    entangling.cx(0, 1)
    assert (
        _reason(
            entangling,
            {"min_entangling_gate_count": 1},
            unitary=np.asarray(Operator(entangling).data),
        )
        is None
    )


def test_native_unitary_must_agree_with_lowered_program_semantics() -> None:
    represented = QuantumCircuit(2)
    represented.h(0)
    represented.cx(0, 1)
    native = QuantumCircuit(2)
    native.h(0)
    native.cz(0, 1)

    result = verify_program_requirements(
        _contract(),
        _program(represented),
        framework="qiskit",
        execution_metadata={},
        candidate_unitary=np.asarray(Operator(native).data),
    )

    assert result is not None
    assert result.reason == "requirement_failed:native_ir_semantic_disagreement"


def test_net_unitary_check_falls_back_and_supports_uncompute_exception() -> None:
    padded = QuantumCircuit(2)
    padded.cx(0, 1)
    padded.cx(0, 1)
    assert _reason(padded, {"min_entangling_gate_count": 1}, unitary=None) is None
    assert (
        _reason(
            padded,
            {
                "min_entangling_gate_count": 1,
                "require_net_unitary_entangling": False,
            },
            unitary=np.asarray(Operator(padded).data),
        )
        is None
    )


def test_argument_conditioned_error_gate_is_checked_before_cancellation() -> None:
    circuit = QuantumCircuit(5)
    circuit.x(4)
    circuit.x(4)
    assert (
        _reason(
            circuit,
            semantic={
                "argument_conditioned_gate": {
                    "argument_index": 0,
                    "gate_names": ["x"],
                    "wires": [0, 1, 2, 3, 4],
                }
            },
            arguments=(4,),
        )
        is None
    )


def test_net_nonlocality_must_reach_an_observed_wire() -> None:
    parked = QuantumCircuit(3, 1)
    parked.x(0)
    parked.cz(1, 2)
    parked.measure(0, 0)
    stripped = parked.remove_final_measurements(inplace=False)
    assert (
        _reason(
            parked,
            {"min_entangling_gate_count": 1},
            unitary=np.asarray(Operator(stripped).data),
        )
        == "requirement_failed:net_unitary_nonlocal"
    )


def test_semantic_family_and_interaction_requirements_use_program_ir() -> None:
    direct = QuantumCircuit(2)
    direct.cx(0, 1)
    assert (
        _reason(
            direct,
            {
                "forbidden_gate_family_counts": {"controlled_not": 0},
            },
        )
        == "forbidden_gate_family:controlled_not"
    )

    decomposed = QuantumCircuit(2)
    decomposed.h(1)
    decomposed.cz(0, 1)
    decomposed.h(1)
    assert (
        _reason(
            decomposed,
            {
                "forbidden_gate_family_counts": {"controlled_not": 0},
                "required_interactions": [[0, 1]],
            },
        )
        is None
    )


def test_qec_interactions_ignore_adjacent_canceling_padding() -> None:
    padded = QuantumCircuit(2)
    padded.cx(0, 1)
    padded.barrier()
    padded.cx(0, 1)
    assert (
        _reason(
            padded,
            semantic={
                "required_interactions": [[0, 1]],
                "reject_canceling_interaction_padding": True,
            },
        )
        == "requirement_failed:missing_interaction:0-1"
    )

    effective = QuantumCircuit(2)
    effective.cx(0, 1)
    assert (
        _reason(
            effective,
            semantic={
                "required_interactions": [[0, 1]],
                "reject_canceling_interaction_padding": True,
            },
        )
        is None
    )


def test_qec_reverse_order_uncomputation_is_not_treated_as_padding() -> None:
    semantics = {
        "required_interactions": [[0, 1], [0, 2]],
        "required_controlled_x_interactions": [[0, 1], [0, 2]],
        "reject_canceling_interaction_padding": True,
    }

    # Encode CX(0,1);CX(0,2) followed by the reverse-order decoder
    # CX(0,2);CX(0,1): in the mandatory no-error case the inverse encoding
    # gates sit next to each other, but this nested bracket structure is
    # legitimate uncomputation, not padding.
    reverse_decoder = QuantumCircuit(3)
    reverse_decoder.cx(0, 1)
    reverse_decoder.cx(0, 2)
    reverse_decoder.cx(0, 2)
    reverse_decoder.cx(0, 1)
    assert _reason(reverse_decoder, semantic=semantics) is None

    same_order_decoder = QuantumCircuit(3)
    same_order_decoder.cx(0, 1)
    same_order_decoder.cx(0, 2)
    same_order_decoder.cx(0, 1)
    same_order_decoder.cx(0, 2)
    assert _reason(same_order_decoder, semantic=semantics) is None

    palindromic_hadamard = QuantumCircuit(3)
    palindromic_hadamard.cx(0, 1)
    palindromic_hadamard.cx(0, 2)
    palindromic_hadamard.h(0)
    palindromic_hadamard.h(1)
    palindromic_hadamard.h(2)
    palindromic_hadamard.h(2)
    palindromic_hadamard.h(1)
    palindromic_hadamard.h(0)
    palindromic_hadamard.cx(0, 2)
    palindromic_hadamard.cx(0, 1)
    assert _reason(palindromic_hadamard, semantic=semantics) is None

    # Gratuitous adjacent canceling pairs that do not bracket other canceling
    # structure remain rejected padding even when repeated per interaction.
    padded = QuantumCircuit(3)
    padded.cx(0, 1)
    padded.cx(0, 1)
    padded.cx(0, 2)
    padded.cx(0, 2)
    assert _reason(padded, semantic=semantics) == "requirement_failed:missing_interaction:0-1"


def test_qec_interaction_topology_checks_gate_family_and_direction() -> None:
    directed = {"required_controlled_x_interactions": [[0, 1]]}
    correct = QuantumCircuit(2)
    correct.cx(0, 1)
    assert _reason(correct, semantic=directed) is None

    reversed_cx = QuantumCircuit(2)
    reversed_cx.cx(1, 0)
    assert _reason(reversed_cx, semantic=directed) == "requirement_failed:missing_controlled_x_interaction:0-1"

    parity = {"required_parity_interactions": [[0, 1]]}
    phase_parity = QuantumCircuit(2)
    phase_parity.cz(0, 1)
    assert _reason(phase_parity, semantic=parity) is None

    unrelated = QuantumCircuit(2)
    unrelated.swap(0, 1)
    assert _reason(unrelated, semantic=parity) == "requirement_failed:missing_parity_interaction:0-1"


def test_qec_argument_gate_uses_concrete_program_ir() -> None:
    requirement = {
        "argument_conditioned_gate": {
            "argument_index": 0,
            "gate_names": ["x", "paulix"],
            "wires": [0, 1, 2],
            "verification": "program_ir_concrete_case",
        }
    }
    correct = QuantumCircuit(3)
    correct.x(1)
    assert _reason(correct, semantic=requirement, arguments=(1,)) is None

    wrong_wire = QuantumCircuit(3)
    wrong_wire.x(0)
    assert _reason(wrong_wire, semantic=requirement, arguments=(1,)) == "requirement_failed:argument_conditioned_gate"

    canceled = QuantumCircuit(3)
    canceled.x(1)
    canceled.x(1)
    # The exhaustive-family verifier separately compares this concrete case
    # with its no-error baseline to reject argument-independent padding.
    assert _reason(canceled, semantic=requirement, arguments=(1,)) is None
    assert _reason(QuantumCircuit(3), semantic=requirement, arguments=(None,)) is None


def test_qec_state_preparation_shortcut_is_hard_failure() -> None:
    injected = QuantumCircuit(1)
    injected.initialize([1.0, 0.0], 0)
    assert (
        _reason(injected, semantic={"forbid_state_preparation": True}) == "requirement_failed:forbid_state_preparation"
    )


@pytest.mark.parametrize(
    "source",
    (
        """
def helper(circuit, amplitudes):
    prepare = circuit.prepare_state
    prepare(amplitudes, range(9))

def answer():
    pass
""",
        """
def helper(circuit, amplitudes):
    getattr(circuit, "prepare_" + "state")(amplitudes, range(9))

def answer():
    pass
""",
        """
def helper(circuit, amplitudes):
    getattr(circuit, "".join(("prepare", "_state")))(amplitudes, range(9))

def answer():
    pass
""",
    ),
)
def test_state_preparation_source_aliases_cannot_be_laundered(source: str) -> None:
    injected = QuantumCircuit(1)
    injected.initialize([1.0, 0.0], 0)
    decomposed = injected.decompose(reps=10)

    assert (
        _reason(
            decomposed,
            semantic={"forbid_state_preparation": True},
            source_code=source,
        )
        == "requirement_failed:forbid_state_preparation_source"
    )


def test_unresolved_reflection_fails_closed_for_state_preparation_policy() -> None:
    source = """
def helper(circuit, amplitudes, method_name):
    getattr(circuit, method_name)(amplitudes, range(9))
"""
    assert (
        _reason(
            QuantumCircuit(1),
            semantic={"forbid_state_preparation": True},
            source_code=source,
        )
        == "requirement_failed:forbid_dynamic_reflection"
    )


def test_qec_controlled_correction_requires_multicontrolled_x_on_data() -> None:
    policy = {
        "required_controlled_correction": {
            "gate_family": "x",
            "target_wires": [0, 1, 2],
            "control_wires": [3, 4],
            "min_controls": 2,
        }
    }
    direct = QuantumCircuit(5)
    direct.x(0)
    assert _reason(direct, semantic=policy) == "requirement_failed:required_controlled_correction"

    coherent = QuantumCircuit(5)
    coherent.ccx(3, 4, 0)
    assert _reason(coherent, semantic=policy) is None


def test_qec_connected_interaction_groups_accept_chains_not_disconnected_blocks() -> None:
    policy = {"required_connected_interaction_groups": [[0, 1, 2]]}
    chain = QuantumCircuit(3)
    chain.cx(0, 1)
    chain.cx(1, 2)
    assert _reason(chain, semantic=policy) is None

    disconnected = QuantumCircuit(3)
    disconnected.cx(0, 1)
    assert _reason(disconnected, semantic=policy) == "requirement_failed:disconnected_interaction_group"


def test_forbidden_calls_ignore_innocent_variable_names() -> None:
    circuit = QuantumCircuit(1)
    policy = {"forbidden_calls": ["sample", "decoder", "optimizer"]}
    harmless = """
def answer():
    sample = 2
    decoder = "majority vote"
    optimizer = None
"""
    assert _reason(circuit, semantic=policy, source_code=harmless) is None

    invoked = """
def answer():
    sample()
"""
    assert _reason(circuit, semantic=policy, source_code=invoked) == "requirement_failed:forbidden_calls"


def test_forbidden_calls_resolve_import_aliases() -> None:
    circuit = QuantumCircuit(1)
    source = """
from qiskit_aer import AerSimulator as Backend

def answer():
    return Backend()
"""
    assert (
        _reason(circuit, semantic={"forbidden_calls": ["AerSimulator"]}, source_code=source)
        == "requirement_failed:forbidden_calls"
    )


def test_forbidden_imports_detect_static_dynamic_imports() -> None:
    circuit = QuantumCircuit(1)
    source = """
import importlib

def answer():
    return importlib.import_module("sti" + "m")
"""
    assert (
        _reason(circuit, semantic={"forbidden_imports": ["stim"]}, source_code=source)
        == "requirement_failed:forbidden_imports"
    )


def test_forbidden_imports_resolve_module_aliases() -> None:
    """``import qiskit.circuit.library as library; library.QFT(...)`` is a QFT shortcut."""
    circuit = QuantumCircuit(6)
    source = """
from qiskit import QuantumCircuit
import qiskit.circuit.library as library

def qft_6():
    qc = QuantumCircuit(6)
    qc.append(library.QFT(num_qubits=6), range(6))
    return qc
"""
    assert (
        _reason(circuit, semantic={"forbidden_imports": ["QFT"]}, source_code=source)
        == "requirement_failed:forbidden_imports"
    )


def test_forbidden_imports_resolve_renamed_imports() -> None:
    """``from ... import QFT as myqft`` is still a banned prebuilt constructor."""
    circuit = QuantumCircuit(6)
    source = """
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT as myqft

def qft_6():
    qc = QuantumCircuit(6)
    qc.append(myqft(num_qubits=6), range(6))
    return qc
"""
    assert (
        _reason(circuit, semantic={"forbidden_imports": ["QFT"]}, source_code=source)
        == "requirement_failed:forbidden_imports"
    )


def test_forbidden_imports_block_newer_constructor_names() -> None:
    """Newer/alternative constructor names like ``QFTGate`` must be rejected."""
    circuit = QuantumCircuit(6)
    source = """
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate

def qft_6():
    qc = QuantumCircuit(6)
    qc.append(QFTGate(num_qubits=6), range(6))
    return qc
"""
    assert (
        _reason(circuit, semantic={"forbidden_imports": ["QFT", "QFTGate"]}, source_code=source)
        == "requirement_failed:forbidden_imports"
    )


def test_forbidden_imports_allows_local_qft_helper() -> None:
    """A candidate-defined helper named ``qft`` is not a prebuilt algorithm import."""
    circuit = QuantumCircuit(6)
    source = """
from qiskit import QuantumCircuit
from math import pi

def qft_6():
    qc = QuantumCircuit(6)
    def qft(qubits):
        n = len(qubits)
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
        for i in range(n):
            for j in range(i):
                qc.cp(pi / (2 ** (i - j)), qubits[j], qubits[i])
            qc.h(qubits[i])
    qft(range(6))
    return qc
"""
    assert _reason(circuit, semantic={"forbidden_imports": ["QFT"]}, source_code=source) is None


def test_allowed_rotation_family_admits_special_angles() -> None:
    rotations = QuantumCircuit(1)
    rotations.rx(np.pi / 2, 0)
    rotations.ry(np.pi / 4, 0)
    assert _reason(rotations, semantic={"allowed_gate_families": ["rx", "ry"]}) is None
    # The specialized family is still visible: rx(pi/2) trips a forbidden sx.
    assert _reason(rotations, semantic={"forbidden_gate_families": ["sx"]}) == "forbidden_gate_family:sx"

    off_basis = QuantumCircuit(1)
    off_basis.h(0)
    assert _reason(off_basis, semantic={"allowed_gate_families": ["rx", "ry"]}) == "forbidden_gate_family:h"


def test_returned_probability_shortcut_is_rejected() -> None:
    circuit = QuantumCircuit(1)
    assert (
        _reason(
            circuit,
            {"forbid_returned_probabilities": True},
            metadata={"probability_method": "returned_probabilities"},
        )
        == "requirement_failed:forbid_returned_probabilities"
    )


def test_clifford_gate_class_blocks_non_clifford_and_matrix_shortcuts() -> None:
    direct_matrix = QuantumCircuit(2)
    direct_matrix.append(UnitaryGate(np.eye(4)), [0, 1])
    semantic = {
        "allowed_gate_class": "Clifford including an entangling Clifford primitive",
        "forbidden_gate_families": ["cx", "cnot"],
    }
    assert _reason(direct_matrix, semantic=semantic) == "forbidden_gate_family:dense_unitary"

    non_clifford = QuantumCircuit(2)
    non_clifford.t(0)
    non_clifford.cz(0, 1)
    assert _reason(non_clifford, semantic=semantic) == "forbidden_gate_family:t"

    valid = QuantumCircuit(2)
    valid.h(1)
    valid.cz(0, 1)
    valid.h(1)
    assert _reason(valid, semantic=semantic) is None


def test_decomposition_requirement_blocks_matrix_payload_shortcuts() -> None:
    circuit = QuantumCircuit(2)
    circuit.append(UnitaryGate(np.eye(4)), [0, 1])
    assert (
        _reason(
            circuit,
            semantic={
                "decomposition_required": True,
                "forbidden_gate_families": ["controlled_h"],
            },
        )
        == "forbidden_gate_family:dense_unitary"
    )


def test_dense_matrix_shortcuts_are_rejected_after_cirq_and_pennylane_lowering() -> None:
    import cirq
    import pennylane as qml

    qubit = cirq.LineQubit(0)
    cirq_program = cirq.Circuit(cirq.MatrixGate(np.eye(2)).on(qubit))
    pennylane_program = qml.tape.QuantumScript(
        [qml.QubitUnitary(np.eye(2), wires=0)],
        [qml.state()],
    )
    contract = _contract(semantic={"forbidden_gate_families": ["dense_unitary"]})
    candidates = (
        ("cirq", cirq_program),
        ("pennylane", pennylane_program),
    )
    for framework, candidate in candidates:
        lowered = (
            default_lowering_registry()
            .get(framework)
            .lower(
                candidate,
                SourceMetadata(framework, None, None),
                contract,
            )
        )
        assert lowered.program is not None
        result = verify_program_requirements(
            contract,
            lowered.program,
            framework=framework,
            execution_metadata={},
        )
        assert result is not None
        assert result.reason == "forbidden_gate_family:dense_unitary"


def test_structural_policy_preserves_calibrated_task_metadata() -> None:
    record = {
        "canonical_class": {
            "metadata_checks": {
                "min_num_qubits": 5,
                "min_entangling_gate_count": 1,
                "unknown_legacy_field": 7,
            },
            "forbidden_imports": ["Shor"],
            "forbidden_calls": ["sample("],
        }
    }
    assert _structural_policy(record) == {
        "min_num_qubits": 5,
        "min_entangling_gate_count": 1,
        "forbidden_imports": ["Shor"],
        "forbidden_calls": ["sample("],
    }


def test_prompt_audit_emits_per_framework_structural_requirements() -> None:
    report = prompt_parity_report()["tasks"]
    requirements = report["14"]["structural_constraints"]["frameworks"]
    assert requirements["qiskit"]["min_num_qubits"] == 5
    assert requirements["cirq"]["min_num_qubits"] == 5
    assert requirements["pennylane"]["min_entangling_gate_count"] == 1
    assert requirements["cudaq"]["forbid_returned_unitary"] is True
    assert "min_measurement_count" not in requirements["cudaq"]
    assert "required_measurement_qubits" not in report["51"]["structural_constraints"]["frameworks"]["cudaq"]
    assert report["15"]["structural_constraints"]["frameworks"]["cirq"]["min_entangling_gate_count"] == 1


def test_rotation_family_accepts_plain_unpack_and_equivalent_task_40_placement() -> None:
    plain_unpack = ast.parse(
        """
def quantum_state_preparation(parameters):
    theta, phi = parameters
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.ry(phi, 0)
    return qc
"""
    ).body[0]
    assert isinstance(plain_unpack, ast.FunctionDef)
    assert _prove_rotation_family(plain_unpack, "39")[0] is SemanticStatus.VERIFIED_PASS

    prompt_literal = ast.parse(
        """
def VQE_2(parameters):
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.rz(parameters[0], 0)
    qc.rz(parameters[1], 0)
    qc.ry(parameters[2], 0)
    qc.rz(parameters[3], 1)
    qc.cx(0, 1)
    qc.rz(parameters[4], 0)
    qc.rz(parameters[5], 0)
    qc.ry(parameters[6], 0)
    qc.rz(parameters[7], 1)
    return qc
"""
    ).body[0]
    assert isinstance(prompt_literal, ast.FunctionDef)
    assert _prove_rotation_family(prompt_literal, "40")[0] is SemanticStatus.VERIFIED_PASS


def test_task_40_prompts_match_the_canonical_parameter_placement() -> None:
    for framework in ("qiskit", "cirq", "pennylane", "cudaq"):
        prompt = load_tasks(framework, "core")["40"]["prompt"]
        assert "parameters[5]" in prompt
        assert (
            "parameters[5]:    RZ rotation on qubit 1" in prompt
            or "parameters[5] as an RZ rotation on qubit 1" in prompt
            or "parameters[5] as RZ on q1" in prompt
        )


def test_task_55_prompts_require_the_canonical_initial_superposition() -> None:
    for framework in ("qiskit", "cirq", "pennylane", "cudaq"):
        task = load_tasks(framework, "core")["55"]
        assert "starting in |++++>" in task["prompt"]
        assert "by applying H to all four qubits" not in task["prompt"]


def test_pennylane_decomposition_canonicals_do_not_use_forbidden_gates() -> None:
    tasks = load_tasks("pennylane", "core")
    assert "qml.CH(" not in tasks["37"]["canonical_solution"]
    assert "qml.CNOT(" not in tasks["44"]["canonical_solution"]
