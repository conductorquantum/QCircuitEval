"""Structural, measurement, interaction, and terminal-observation checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from qceval.evals.parser.source import source_call_names, source_dynamic_features, source_import_references
from qceval.evals.unitaries import bit_reverse_unitary, unitaries_equivalent, unitary_is_entangling
from qceval.semantics.contracts import Contract
from qceval.semantics.ir import OperationKind, Program
from qceval.semantics.verifiers.classical_wires import rendered_quantum_wires as _rendered_quantum_wires
from qceval.semantics.verifiers.requirements.gate_family import (
    _family_count_violation,
    _operation_wires,
)
from qceval.semantics.verifiers.requirements.interactions import (
    _declared_pair,
    _interaction_pair,
    _interaction_pairs,
)


def _framework_structural_policy(value: Mapping[str, Any], framework: str) -> dict[str, Any]:
    """Select common and framework-specific anti-shortcut requirements."""
    common = value.get("common", {})
    frameworks = value.get("frameworks")
    if not isinstance(frameworks, Mapping):
        return dict(value)
    selected = frameworks.get(framework, {})
    merged = dict(common) if isinstance(common, Mapping) else {}
    if isinstance(selected, Mapping):
        merged.update(selected)
    return merged


def _structural_violation(
    contract: Contract,
    program: Program,
    policy: Mapping[str, Any],
    framework: str,
    execution_metadata: Mapping[str, Any],
    source_code: str | None,
    candidate_unitary: Any | None,
) -> str | None:
    """Return the first violated framework-neutral anti-shortcut constraint."""
    reason = _returned_value_violation(policy, execution_metadata)
    if reason is not None:
        return reason
    reason = _full_register_dense_unitary_violation(policy, program)
    if reason is not None:
        return reason
    reason = _state_preparation_violation(policy, program, contract, source_code)
    if reason is not None:
        return reason
    reason = _numeric_structure_violation(policy, program)
    if reason is not None:
        return reason
    reason = _family_count_violation(policy, program)
    if reason is not None:
        return reason
    reason = _structural_measurement_violation(policy, program)
    if reason is not None:
        return reason
    reason = _interaction_violation(policy, program)
    if reason is not None:
        return reason
    reason = _source_policy_violation(policy, source_code)
    if reason is not None:
        return reason
    return _net_unitary_violation(
        contract,
        program,
        policy,
        framework,
        candidate_unitary,
    )


def _returned_value_violation(
    policy: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str | None:
    method = metadata.get("probability_method")
    if policy.get("forbid_returned_counts") is True and (
        metadata.get("returned_counts") is True or method == "returned_counts"
    ):
        return "requirement_failed:forbid_returned_counts"
    if policy.get("forbid_returned_probabilities") is True and method == "returned_probabilities":
        return "requirement_failed:forbid_returned_probabilities"
    if policy.get("forbid_returned_unitary") is True and method == "returned_unitary":
        return "requirement_failed:forbid_returned_unitary"
    return None


def _full_register_dense_unitary_violation(
    policy: Mapping[str, Any],
    program: Program,
) -> str | None:
    """Reject a nontrivial opaque matrix spanning the complete register."""
    if policy.get("forbid_full_register_dense_unitary") is not True:
        return None
    all_wires = frozenset(range(program.num_qubits))
    for operation in program.operations:
        if (
            operation.kind is OperationKind.GATE
            and operation.name == "dense_unitary"
            and _operation_wires(operation) == all_wires
            and not _dense_unitary_is_identity(operation)
            and dict(operation.semantic_data).get("matrix_origin") != "qiskit_gate_definition"
        ):
            return "requirement_failed:forbid_full_register_dense_unitary"
    return None


def _dense_unitary_is_identity(operation: Any) -> bool:
    payload = dict(operation.semantic_data).get("matrix_complex128_hex")
    if not isinstance(payload, str):
        return False
    try:
        import numpy as np

        values = np.frombuffer(bytes.fromhex(payload), dtype=np.complex128)
        dimension = int(round(values.size**0.5))
        matrix = values.reshape(dimension, dimension)
    except (ValueError, OverflowError):
        return False
    return bool(np.allclose(matrix, np.eye(dimension, dtype=np.complex128), atol=1e-12, rtol=0.0))


def _state_preparation_violation(
    policy: Mapping[str, Any],
    program: Program,
    contract: Contract,
    source_code: str | None = None,
) -> str | None:
    """Reject direct amplitude injection where a circuit construction is required.

    ``initialize``/``StatePrep``-style operations are simulated faithfully by
    the behavioral engines, so without this gate a candidate can pass a state
    or distribution contract by injecting the answer amplitudes instead of
    building the requested circuit.
    """
    if policy.get("forbid_state_preparation", True) is False:
        return None
    preparations = tuple(
        operation for operation in program.operations if operation.kind is OperationKind.STATE_PREPARATION
    )
    if preparations and _prepares_supplied_input_off_register(contract, program, preparations):
        return None
    if preparations:
        return "requirement_failed:forbid_state_preparation"
    # Decomposition can erase an initialize/StatePrep node before lowering.
    # Source provenance is therefore part of this hard requirement, not merely
    # an IR gate-family check.
    state_preparation_calls = {
        "amplitudeembedding",
        "basisstate",
        "initialize",
        "mottonenstatepreparation",
        "prepare_state",
        "qubitstatevector",
        "set_state",
        "set_statevector",
        "stateprep",
        "statepreparation",
    }
    if source_call_names(source_code) & state_preparation_calls:
        return "requirement_failed:forbid_state_preparation_source"
    if source_dynamic_features(source_code):
        return "requirement_failed:forbid_dynamic_reflection"
    return None


def _prepares_supplied_input_off_register(
    contract: Contract,
    program: Program,
    preparations: tuple[Any, ...],
) -> bool:
    """Permit loading a task-supplied input state away from the observed register.

    A ``single_qubit_program`` signature argument supplies a quantum state the
    candidate must prepare somewhere; only amplitude injection onto the
    measured register can smuggle in the answer distribution.
    """
    if not any(argument.value_type == "single_qubit_program" for argument in contract.signature.arguments):
        return False
    measured = set(_measurement_wires(program))
    return all(measured.isdisjoint(operation.quantum_wires) for operation in preparations)


def _native_ir_semantic_violation(
    contract: Contract,
    program: Program,
    framework: str,
    candidate_unitary: Any | None,
) -> str | None:
    """Cross-check lowered IR against independently inspected native behavior.

    The native framework and Program IR are separate semantic paths.  Exact
    agreement makes parser aliases tamper-evident: even if an unfamiliar gate
    is accidentally labeled as a built-in operation, its native unitary will
    disagree with the unitary materialized from that label and grading fails
    closed.
    """
    if candidate_unitary is None or program.num_qubits > 8:
        return None
    expected_dimension = 2**program.num_qubits
    if getattr(candidate_unitary, "shape", None) != (expected_dimension, expected_dimension):
        # Cirq and PennyLane omit declared-but-unused wire labels from their
        # native unitary. The behavior engine still verifies the full Program
        # IR; a differently sized native object cannot soundly cross-check it.
        return None
    if any(
        operation.kind in {OperationKind.RESET, OperationKind.STATE_PREPARATION} or operation.condition is not None
        for operation in program.operations
    ):
        return None
    try:
        from qceval.semantics.verifiers.base import VerificationContext
        from qceval.semantics.verifiers.dynamic import DynamicSimulationError
        from qceval.semantics.verifiers.program_materializer import ProgramIRMaterializer

        context = VerificationContext(
            contract=contract,
            contract_hash="",
            target_hash="",
            input_hash="",
            program=program,
        )
        lowered = ProgramIRMaterializer().array(context, "unitary").value
    except (DynamicSimulationError, MemoryError, NotImplementedError, ValueError):
        return None
    native = bit_reverse_unitary(candidate_unitary) if framework in {"cirq", "pennylane"} else candidate_unitary
    equivalent, _ = unitaries_equivalent(
        lowered,
        native,
        tolerance=1e-8,
        ignore_global_phase=True,
    )
    if not equivalent:
        return "requirement_failed:native_ir_semantic_disagreement"
    return None


def _numeric_structure_violation(
    policy: Mapping[str, Any],
    program: Program,
) -> str | None:
    thresholds = {
        "min_num_qubits": program.num_qubits,
        "min_measurement_count": len(_measurement_wires(program)),
        "min_non_measurement_operation_count": sum(
            operation.kind not in {OperationKind.MEASUREMENT, OperationKind.BARRIER} for operation in program.operations
        ),
        "min_entangling_gate_count": len(_entangling_operations(program)),
    }
    for name, observed in thresholds.items():
        minimum = _integer_policy(policy.get(name))
        if minimum is not None and observed < minimum:
            return f"requirement_failed:{name}"
    maximum = _integer_policy(policy.get("max_measurement_count"))
    if maximum is not None and len(_measurement_wires(program)) > maximum:
        return "requirement_failed:max_measurement_count"
    return None


def _integer_policy(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _entangling_operations(program: Program) -> tuple[Any, ...]:
    return tuple(
        operation
        for operation in program.operations
        if operation.kind is OperationKind.GATE and len(_operation_wires(operation)) >= 2
    )


def _measurement_wires(program: Program) -> tuple[int, ...]:
    return tuple(
        wire
        for operation in program.operations
        if operation.kind is OperationKind.MEASUREMENT
        for wire in operation.quantum_wires
    )


def _structural_measurement_violation(
    policy: Mapping[str, Any],
    program: Program,
) -> str | None:
    measured = set(_measurement_wires(program))
    required = policy.get("required_measurement_qubits")
    if (
        isinstance(required, Sequence)
        and not isinstance(required, str | bytes)
        and not all(isinstance(wire, int) and wire in measured for wire in required)
    ):
        return "requirement_failed:required_measurement_qubits"
    forbidden = policy.get("forbidden_measurement_qubits")
    if (
        isinstance(forbidden, Sequence)
        and not isinstance(forbidden, str | bytes)
        and any(isinstance(wire, int) and wire in measured for wire in forbidden)
    ):
        return "requirement_failed:forbidden_measurement_qubits"
    return None


def _interaction_violation(
    policy: Mapping[str, Any],
    program: Program,
) -> str | None:
    observed = _interaction_pairs(program)
    required = policy.get("required_interactions")
    if (
        isinstance(required, Sequence)
        and not isinstance(required, str | bytes)
        and any(_declared_pair(item) not in observed for item in required)
    ):
        return "requirement_failed:required_interactions"
    groups = policy.get("required_interaction_groups")
    if isinstance(groups, Sequence) and not isinstance(groups, str | bytes):
        for group in groups:
            if not isinstance(group, Sequence) or isinstance(group, str | bytes) or len(group) < 2:
                return "requirement_failed:required_interaction_groups"
            if any(_interaction_pair(int(a), int(b)) not in observed for a, b in combinations(group, 2)):
                return "requirement_failed:required_interaction_groups"
    alternatives = policy.get("required_any_interaction_groups")
    if (
        isinstance(alternatives, Sequence)
        and not isinstance(alternatives, str | bytes)
        and alternatives
        and not any(
            isinstance(group, Sequence)
            and not isinstance(group, str | bytes)
            and all(_declared_pair(item) in observed for item in group)
            for group in alternatives
        )
    ):
        return "requirement_failed:required_any_interaction_groups"
    return None


def _source_policy_violation(
    policy: Mapping[str, Any],
    source_code: str | None,
) -> str | None:
    policies = {
        "forbidden_imports": source_import_references(source_code),
        "forbidden_calls": source_call_names(source_code),
    }
    for key, observed in policies.items():
        blocked = policy.get(key)
        if not isinstance(blocked, Sequence) or isinstance(blocked, str | bytes):
            continue
        normalized = {str(item).lower().replace("(", "").split(".")[-1] for item in blocked}
        if observed & normalized:
            return f"requirement_failed:{key}"
    return None


def _net_unitary_violation(
    contract: Contract,
    program: Program,
    policy: Mapping[str, Any],
    framework: str,
    candidate_unitary: Any | None,
) -> str | None:
    minimum = _integer_policy(policy.get("min_entangling_gate_count"))
    if not minimum or contract.parameters.items or policy.get("require_net_unitary_entangling", True) is False:
        return None
    unitary = candidate_unitary
    if framework in {"cirq", "pennylane"}:
        unitary = bit_reverse_unitary(unitary)
    wires = sorted(set(_measurement_wires(program)))
    if not wires:
        observed = set(contract.observation.quantum)
        wires = sorted(wire for system in contract.systems.items if system.name in observed for wire in system.indices)
    verdict = unitary_is_entangling(unitary, wires=wires or None)
    if verdict is False:
        return "requirement_failed:net_unitary_nonlocal"
    return None


def _terminal_violation(
    program: Program,
    interface: Mapping[str, Any],
    execution_metadata: Mapping[str, Any],
) -> str | None:
    selected = _selected_interface(program, interface)
    if selected is None:
        return "terminal_observation_mismatch"
    has_measurement = any(operation.kind is OperationKind.MEASUREMENT for operation in program.operations)
    if selected.get("kind") == "none":
        return "terminal_observation_mismatch" if has_measurement else None
    if selected.get("kind") == "measurement":
        return _measurement_register_violation(program, selected)
    required = selected.get("render_order", selected.get("wires", selected.get("qubits")))
    if required is None:
        return None
    if not isinstance(required, list | tuple) or not all(isinstance(item, int) for item in required):
        return "terminal_observation_contract_invalid"
    if selected.get("mode") == "statevector":
        return _statevector_mode_violation(required, has_measurement, execution_metadata)
    if not has_measurement and _statevector_projection_matches(selected, execution_metadata):
        return None
    return _terminal_render_order_violation(program, selected, required)


def _terminal_render_order_violation(
    program: Program,
    selected: Mapping[str, Any],
    required: list[Any] | tuple[Any, ...],
) -> str | None:
    """Validate the terminal register and framework-specific order policy."""
    actual = _rendered_quantum_wires(program)
    if program.provenance.framework == "pennylane" and selected.get("mode") == "probabilities":
        # PennyLane prompts explicitly permit qml.probs wires in any order.
        # Preserve the candidate order in IR, validate the declared register
        # here, and let distribution materialization reorder by the contract.
        return None if sorted(actual) == sorted(required) else "terminal_observation_mismatch"
    if actual != tuple(required):
        # A register-role interface pins register sizes and render order, not
        # physical placement; any unambiguous same-width register is the same
        # observation with its roles read positionally.
        if (
            selected.get("layout") == "register_roles"
            and len(actual) == len(required)
            and len(set(actual)) == len(actual)
        ):
            return None
        return "terminal_observation_mismatch"
    return None


def _statevector_mode_violation(
    required: list[Any] | tuple[Any, ...],
    has_measurement: bool,
    execution_metadata: Mapping[str, Any],
) -> str | None:
    observed = execution_metadata.get("measurement_qubits")
    if (
        has_measurement
        or execution_metadata.get("probability_method") != "statevector"
        or not isinstance(observed, list | tuple)
        or not all(isinstance(item, int) for item in observed)
        or tuple(observed) != tuple(required)
    ):
        return "terminal_observation_mismatch"
    return None


def _statevector_projection_matches(selected: Mapping[str, Any], execution_metadata: Mapping[str, Any]) -> bool:
    """Accept unmeasured programs whose executor projected the declared register.

    CUDA-Q kernels follow a return-unmeasured convention; the executor derives
    exact probabilities from the statevector over the task's declared output
    qubits. That projection observes exactly the contracted register.
    """

    declared = selected.get("qubits", selected.get("wires"))
    if not isinstance(declared, list | tuple) or not all(isinstance(item, int) for item in declared):
        return False
    observed = execution_metadata.get("measurement_qubits")
    if execution_metadata.get("probability_method") != "statevector":
        return False
    if not isinstance(observed, list | tuple) or not all(isinstance(item, int) for item in observed):
        return False
    if not observed:
        # An unmeasured kernel with no projection observes the full register;
        # that matches only a contract that declares every wire.
        return sorted(declared) == list(range(len(declared))) and execution_metadata.get("num_qubits") == len(declared)
    return sorted(observed) == sorted(declared)


def _measurement_register_violation(program: Program, selected: Mapping[str, Any]) -> str | None:
    qubits = selected.get("qubits")
    if not isinstance(qubits, list | tuple) or not all(isinstance(item, int) for item in qubits):
        return "terminal_observation_contract_invalid"
    bits = selected.get("classical_bits")
    actual: dict[int, int] = {}
    for operation in program.operations:
        if operation.kind is not OperationKind.MEASUREMENT:
            continue
        if len(operation.quantum_wires) != len(operation.classical_bits):
            return "terminal_observation_mismatch"
        for wire, bit in zip(operation.quantum_wires, operation.classical_bits, strict=True):
            if bit in actual and actual[bit] != wire:
                return "terminal_observation_mismatch"
            actual[bit] = wire
    if isinstance(bits, list | tuple) and len(bits) == len(qubits) and all(isinstance(item, int) for item in bits):
        expected = dict(zip(bits, qubits, strict=True))
        if actual == expected:
            return None
        # The framework grading note also documents the identity packing
        # (measure qubit i into classical bit i); accept it for the same
        # measured register.
        identity = {qubit: qubit for qubit in qubits}
        return None if actual == identity else "terminal_observation_mismatch"
    # Without an explicit classical map the declared register is an unordered
    # set; bit ordering is validated by the downstream semantic comparison.
    return None if sorted(actual.values()) == sorted(qubits) else "terminal_observation_mismatch"


def _measurement_exclusion_violation(program: Program, exclusion: Mapping[str, Any]) -> str | None:
    excluded = exclusion.get("qubits")
    if not isinstance(excluded, list | tuple) or not all(isinstance(item, int) for item in excluded):
        return None
    blocked = set(excluded)
    for operation in program.operations:
        if operation.kind is not OperationKind.MEASUREMENT:
            continue
        touched = sorted(blocked.intersection(operation.quantum_wires))
        if touched:
            return f"forbidden_measured_qubit:{touched[0]}"
    return None


def _selected_interface(program: Program, interface: Mapping[str, Any]) -> Mapping[str, Any] | None:
    alternatives = interface.get("alternatives")
    if not isinstance(alternatives, list):
        return interface
    actual = _rendered_quantum_wires(program)
    for alternative in alternatives:
        if not isinstance(alternative, Mapping):
            continue
        required = alternative.get("render_order", alternative.get("wires"))
        if isinstance(required, list | tuple) and actual == tuple(required):
            return alternative
    return None
