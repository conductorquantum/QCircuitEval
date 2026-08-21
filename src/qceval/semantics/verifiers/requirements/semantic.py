"""Semantic, source, QEC, and algorithm-level requirement checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from qceval.evals.parser.source import source_call_names, source_dynamic_features
from qceval.semantics.contracts import Contract
from qceval.semantics.ir import OperationKind, Program
from qceval.semantics.verifiers.requirements.gate_family import (
    _clifford_gate_class_violation,
    _gate_basis_violation,
    _operation_family,
)
from qceval.semantics.verifiers.requirements.interactions import (
    _argument_conditioned_gate_violation,
    _connected_interaction_groups_violation,
    _controlled_correction_violation,
    _encoder_state_before_ancilla_use_violation,
    _inter_group_before_intra_group_violation,
    _noncanceling_gate_operations,
    _qec_state_preparation_violation,
    _required_interaction_violation,
)
from qceval.semantics.verifiers.requirements.structural import _source_policy_violation

_PHASE_FAMILIES = {"z", "p", "phase", "rz", "r1", "cp", "cphase", "mcp", "mcphase", "zpow"}


def _semantic_violation(
    contract: Contract,
    program: Program,
    semantics: Mapping[str, Any],
    framework: str,
    execution_metadata: Mapping[str, Any],
    source_code: str | None,
    arguments: tuple[Any, ...],
) -> str | None:
    qec_violation = _qec_requirement_violation(
        contract,
        program,
        semantics,
        framework,
        execution_metadata,
        source_code,
        arguments,
    )
    if qec_violation is not None:
        return qec_violation
    gate_violation = _semantic_gate_violation(program, semantics)
    if gate_violation is not None:
        return gate_violation

    parsed_names = source_call_names(source_code)
    policies = {
        # Prompts permit low-level composed QFT construction (including a
        # candidate helper merely named ``qft``); only end-to-end
        # phase-estimation algorithm classes are shortcuts.
        "forbid_library_shortcuts": {
            "phaseestimation",
            "quantumphaseestimation",
            "quantumcounting",
            "amplitudeestimation",
        },
        "forbid_dense_evolution_shortcuts": {
            "expm",
            "matrixgate",
            "paulievolutiongate",
            "qubitunitary",
            "unitary",
            "unitarygate",
        },
        "forbid_optimizer": {"minimize", "optimizer", "optimize"},
        "forbid_eigensolver_shortcuts": {"eig", "eigh", "eigvals", "eigvalsh"},
        "forbid_unitary_shortcuts": {"matrixgate", "qubitunitary", "unitary", "unitarygate"},
    }
    if semantics.get("decomposition_required") is True and parsed_names & {
        "matrixgate",
        "qubitunitary",
        "unitary",
        "unitarygate",
    }:
        return "requirement_failed:forbid_unitary_shortcuts"
    for policy, blocked in policies.items():
        if semantics.get(policy) is True and parsed_names & blocked:
            return f"requirement_failed:{policy}"
    if semantics.get("must_include_entangling_uncompute") is True and not _has_bell_prepare_uncompute(program):
        return "requirement_failed:must_include_entangling_uncompute"
    recipe_violation = _recipe_violation(program, semantics)
    if recipe_violation is not None:
        return recipe_violation
    steps = semantics.get("steps")
    if (
        isinstance(steps, int)
        and not isinstance(steps, bool)
        and ("trotter" in str(semantics.get("algorithm", "")) or semantics.get("algorithm") == "ctqw_spatial_search")
        and not _has_exact_repeated_steps(program, steps)
    ):
        return "requirement_failed:trotter_step_count"
    return None


def _recipe_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    return _query_algorithm_recipe_violation(program, semantics) or _algorithm_recipe_violation(program, semantics)


def _query_algorithm_recipe_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    """Validate query-algorithm structure without prescribing one oracle encoding.

    Their fixed expected bitstrings are otherwise especially easy to synthesize
    directly. Deutsch-Jozsa requires the prompt's phase-kickback construction;
    Bernstein-Vazirani accepts either that construction or the equivalent
    query-register phase oracle. Simon accepts compact output registers while
    still requiring a genuine query/oracle/interference recipe.
    """
    algorithm = str(semantics.get("algorithm", "")).lower()
    if algorithm == "simon":
        return _simon_recipe_violation(program, semantics)
    if algorithm == "grover":
        return _grover_recipe_violation(program, semantics)
    if algorithm not in {"deutsch_jozsa", "bernstein_vazirani"}:
        return None
    return _dj_bv_recipe_violation(program, semantics, algorithm)


def _grover_recipe_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    """Require the Grover / amplitude-amplification skeleton on the search register.

    The check accepts every phase convention (standard pi-phase Grover and
    phase-matched exact search) while rejecting direct output synthesis:

    - every search wire carries at least three Hadamards (the superposition
      layer plus the diffuser sandwich of at least one iteration); and
    - at least ``2 * grover_iterations`` multi-wire gates touch the search
      register after the initial Hadamard layer and act non-trivially on the
      reached state (each iteration contributes an oracle reflection and a
      diffuser reflection).

    Adjacent self-canceling gate pairs are removed first so identity padding
    cannot fake the Hadamard requirement. State-trivial multi-wire padding
    (for example diagonal gates whose controls are always |0>) does not count
    toward the reflection quota.
    """
    wires = semantics.get("search_qubits")
    iterations = semantics.get("grover_iterations")
    if (
        not isinstance(wires, list | tuple)
        or not wires
        or not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 1
    ):
        return "requirement_failed:algorithm_recipe_contract"
    search = {int(wire) for wire in wires}
    gates = list(_noncanceling_gate_operations(program))
    first_h: dict[int, int] = {}
    for wire in sorted(search):
        positions = _gate_positions(gates, family=("h", 0), target=wire)
        if len(positions) < 3:
            return "requirement_failed:grover_hadamard_layers"
        first_h[wire] = positions[0]
    if semantics.get("ancillas_restore_to_zero") is True and not _ancillas_restored_to_zero(
        program.num_qubits,
        gates,
        search,
    ):
        return "requirement_failed:grover_ancillas_not_restored"
    start = max(first_h.values())
    reflections = _nontrivial_search_reflections(program.num_qubits, gates, start, search)
    if reflections is None or reflections < 2 * iterations:
        # Prompts that mandate elementary-gate decompositions produce zero
        # countable multi-wire phase gates; recognize whole reflection blocks
        # instead before rejecting.
        segmented = _segmented_search_reflections(program.num_qubits, gates, search)
        if segmented < 2 * iterations:
            return "requirement_failed:grover_reflections"
    return None


def _ancillas_restored_to_zero(
    num_qubits: int,
    gates: list[Any],
    search: set[int],
) -> bool:
    """Return whether every non-search wire ends in the zero state."""
    ancillas = set(range(num_qubits)) - search
    if not ancillas:
        return True
    try:
        import numpy as np

        from qceval.semantics.verifiers.dynamic.apply import _apply_operation

        state = np.zeros(2**num_qubits, dtype=np.complex128)
        state[0] = 1.0
        for operation in gates:
            state = _apply_operation(state, operation, num_qubits)
    except Exception:  # noqa: BLE001 - a hard uncomputation requirement fails closed.
        return False
    leaked_probability = sum(
        abs(amplitude) ** 2 for basis, amplitude in enumerate(state) if any((basis >> wire) & 1 for wire in ancillas)
    )
    return float(leaked_probability) <= 1e-8


def _nontrivial_search_reflections(
    num_qubits: int,
    gates: list[Any],
    start: int,
    search: set[int],
) -> int | None:
    """Count phase-like multi-wire search gates that change the reached state.

    Clean-ancilla decompositions are also accepted: a phase-family gate whose
    wires all lie off the search register counts as a search reflection when,
    at the reached state, those ancilla wires hold a deterministic classical
    function of the search-basis bits (so the gate applies a diagonal phase to
    the search components), the gate changes the reached ray, and every
    ancilla wire is uncomputed back to zero by the end of the circuit.
    """
    try:
        import numpy as np

        from qceval.semantics.verifiers.dynamic.apply import _apply_operation
    except Exception:  # noqa: BLE001 - recipe verification must fail closed.
        return None
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    state[0] = 1.0
    reflections = 0
    ancillas_clean: bool | None = None
    try:
        for index, operation in enumerate(gates):
            nxt = _apply_operation(state, operation, num_qubits)
            wires = _semantic_operation_wires(operation)
            direct = search & wires and _is_phase_reflection_operation(gates, index)
            if not direct and not (search & wires) and _operation_family(operation)[0] in _PHASE_FAMILIES:
                if ancillas_clean is None:
                    ancillas_clean = _ancillas_restored_to_zero(num_qubits, gates, search)
                direct = ancillas_clean and _ancilla_wires_search_determined(state, wires, search)
            if index > start and direct and not _same_state_ray(state, nxt):
                reflections += 1
            state = nxt
    except Exception:  # noqa: BLE001 - unsupported gates fail closed here.
        return None
    return reflections


def _segmented_search_reflections(num_qubits: int, gates: list[Any], search: set[int]) -> int:
    """Count decomposed reflection blocks between full-register Hadamard layers.

    Prompt-mandated elementary-gate decompositions build each oracle/diffuser
    phase from CX/Toffoli ladders plus single-qubit phase gates, so no single
    gate is a countable multi-wire phase reflection. This counter instead
    treats each maximal gate segment delimited by full-search-register
    Hadamard layers as one candidate reflection and counts it when its net
    action, at the reached ancilla configuration, is a search-diagonal phase
    whose phase pattern genuinely couples search wires (it is not a tensor
    product of single-qubit phases) and it changes the reached ray. Fake
    reflections made of per-wire phases are product patterns and do not
    count, so direct output synthesis cannot inflate this counter.
    """
    try:
        import numpy as np

        from qceval.semantics.verifiers.dynamic.apply import _apply_operation
    except Exception:  # noqa: BLE001 - recipe verification must fail closed.
        return 0
    segments = _hadamard_delimited_segments(gates, search)
    if not segments:
        return 0
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    state[0] = 1.0
    reflections = 0
    position = 0
    try:
        for segment_start, segment_end in segments:
            for operation in gates[position:segment_start]:
                state = _apply_operation(state, operation, num_qubits)
            position = segment_start
            segment = gates[segment_start:segment_end]
            if segment and _is_composite_search_reflection(state, segment, search, num_qubits):
                reflections += 1
            for operation in segment:
                state = _apply_operation(state, operation, num_qubits)
            position = segment_end
    except Exception:  # noqa: BLE001 - unsupported gates fail closed here.
        return 0
    return reflections


def _hadamard_delimited_segments(gates: list[Any], search: set[int]) -> list[tuple[int, int]]:
    """Return gate-index spans between full-search-register Hadamard layers."""
    layers: list[tuple[int, int]] = []
    index = 0
    while index < len(gates):
        operation = gates[index]
        if not (_operation_family(operation) == ("h", 0) and set(operation.quantum_wires) <= search):
            index += 1
            continue
        run_start = index
        covered: set[int] = set()
        while (
            index < len(gates)
            and _operation_family(gates[index]) == ("h", 0)
            and set(gates[index].quantum_wires) <= search
        ):
            covered.update(gates[index].quantum_wires)
            index += 1
        if covered == search:
            layers.append((run_start, index))
    return [(first_end, second_start) for (_, first_end), (second_start, _) in zip(layers, layers[1:], strict=False)]


def _is_composite_search_reflection(
    state: Any,
    segment: list[Any],
    search: set[int],
    num_qubits: int,
    *,
    atol: float = 1e-6,
) -> bool:
    """Return whether one gate segment is a genuine search-register reflection."""
    import numpy as np

    from qceval.semantics.verifiers.dynamic.apply import _apply_operation
    from qceval.semantics.verifiers.dynamic.simulator import reduced_density_matrix

    search_wires = tuple(sorted(search))
    ancilla_wires = tuple(sorted(set(range(num_qubits)) - search))
    if ancilla_wires:
        reduced = reduced_density_matrix(state, ancilla_wires, num_qubits)
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        dominant = int(np.argmax(eigenvalues))
        if float(eigenvalues[dominant]) < 1.0 - atol:
            return False
        ancilla_state = eigenvectors[:, dominant]
    else:
        ancilla_state = np.ones(1, dtype=np.complex128)
    phases = _segment_diagonal_phases(segment, search_wires, ancilla_wires, ancilla_state, num_qubits, atol)
    if phases is None or _is_product_phase_pattern(phases, len(search_wires), atol):
        return False
    after = state
    for operation in segment:
        after = _apply_operation(after, operation, num_qubits)
    return not _same_state_ray(state, after)


def _segment_diagonal_phases(
    segment: list[Any],
    search_wires: tuple[int, ...],
    ancilla_wires: tuple[int, ...],
    ancilla_state: Any,
    num_qubits: int,
    atol: float,
) -> list[complex] | None:
    """Return per-search-basis phases when the segment is search-diagonal.

    The segment is probed on every search basis state tensored with the
    reached ancilla state. It qualifies when each probe returns the same
    basis state with the ancillas left in one common (segment-defined) state,
    so kickback ancillas prepared or rotated inside the segment still verify.
    """
    import numpy as np

    from qceval.semantics.verifiers.dynamic.apply import _apply_operation

    dimension = 2**num_qubits
    phases: list[complex] = []
    reference: np.ndarray | None = None
    for basis in range(2 ** len(search_wires)):
        probe = np.zeros(dimension, dtype=np.complex128)
        for ancilla_basis, amplitude in enumerate(np.asarray(ancilla_state)):
            if abs(amplitude) <= 1e-14:
                continue
            index = sum(((basis >> position) & 1) << wire for position, wire in enumerate(search_wires))
            index += sum(((ancilla_basis >> position) & 1) << wire for position, wire in enumerate(ancilla_wires))
            probe[index] = amplitude
        out = probe
        for operation in segment:
            out = _apply_operation(out, operation, num_qubits)
        block = _search_basis_block(out, basis, search_wires, ancilla_wires, num_qubits)
        if block is None:
            return None
        if reference is None:
            reference = block
            phases.append(1.0 + 0.0j)
            continue
        overlap = complex(np.vdot(reference, block))
        if abs(abs(overlap) - 1.0) > atol:
            return None
        phases.append(overlap / abs(overlap))
    return phases


def _search_basis_block(
    out: Any,
    basis: int,
    search_wires: tuple[int, ...],
    ancilla_wires: tuple[int, ...],
    num_qubits: int,
    *,
    atol: float = 1e-6,
) -> Any | None:
    """Extract the normalized ancilla block when output stays on one search basis."""
    import numpy as np

    block = np.zeros(2 ** len(ancilla_wires), dtype=np.complex128)
    leaked = 0.0
    for index, amplitude in enumerate(np.asarray(out)):
        if abs(amplitude) <= 1e-14:
            continue
        search_bits = sum(((index >> wire) & 1) << position for position, wire in enumerate(search_wires))
        if search_bits != basis:
            leaked += abs(amplitude) ** 2
            continue
        ancilla_bits = sum(((index >> wire) & 1) << position for position, wire in enumerate(ancilla_wires))
        block[ancilla_bits] = amplitude
    if leaked > atol:
        return None
    norm = float(np.linalg.norm(block))
    if abs(norm - 1.0) > atol:
        return None
    return block / norm


def _is_product_phase_pattern(phases: list[complex], width: int, atol: float) -> bool:
    """Return whether basis phases factor into independent per-wire phases."""
    base = phases[0]
    singles = [phases[1 << position] / base for position in range(width)]
    for basis, phase in enumerate(phases):
        expected = base
        for position in range(width):
            if (basis >> position) & 1:
                expected *= singles[position]
        if abs(phase - expected) > atol:
            return False
    return True


def _ancilla_wires_search_determined(
    state: Any,
    wires: set[int],
    search: set[int],
    *,
    atol: float = 1e-8,
) -> bool:
    """Return whether the ancilla wires are a classical function of the search bits."""
    observed: dict[int, tuple[int, ...]] = {}
    for basis, amplitude in enumerate(state):
        if abs(amplitude) <= atol:
            continue
        key = sum(((basis >> wire) & 1) << position for position, wire in enumerate(sorted(search)))
        value = tuple((basis >> wire) & 1 for wire in sorted(wires))
        if observed.setdefault(key, value) != value:
            return False
    return True


def _is_phase_reflection_operation(gates: list[Any], index: int) -> bool:
    """Return whether a gate is an oracle/diffuser phase reflection.

    Plain CX state-preparation edges are excluded. Multi-controlled X counts
    only when sandwiched by Hadamards on its target, which is the standard
    multi-controlled-Z construction used by several canonical solutions.
    """
    operation = gates[index]
    wires = _semantic_operation_wires(operation)
    if len(wires) < 2:
        return False
    family = _operation_family(operation)
    if family[0] in _PHASE_FAMILIES:
        return True
    if family[0] != "x" or family[1] < 1 or len(operation.quantum_wires) != 1:
        return False
    target = operation.quantum_wires[0]
    before = any(
        _operation_family(gates[prior]) == ("h", 0) and gates[prior].quantum_wires == (target,)
        for prior in range(index - 1, max(-1, index - 8), -1)
    )
    after = any(
        _operation_family(gates[later]) == ("h", 0) and gates[later].quantum_wires == (target,)
        for later in range(index + 1, min(len(gates), index + 8))
    )
    return before and after


def _same_state_ray(left: Any, right: Any, *, atol: float = 1e-8) -> bool:
    """Return whether two statevectors agree up to a global phase."""
    import numpy as np

    if left.shape != right.shape:
        return False
    if float(np.linalg.norm(left)) <= atol and float(np.linalg.norm(right)) <= atol:
        return True
    overlap = complex(np.vdot(left, right))
    return abs(abs(overlap) - 1.0) <= 1e-6


def _dj_bv_recipe_violation(
    program: Program,
    semantics: Mapping[str, Any],
    algorithm: str,
) -> str | None:
    resolved = _query_secret(semantics, algorithm)
    if resolved is None:
        return "requirement_failed:algorithm_recipe_contract"
    query_count, secret = resolved
    query_wires = tuple(range(query_count))
    gates = [operation for operation in program.operations if operation.kind is OperationKind.GATE]
    bounds = _hadamard_layer_bounds(gates, query_wires)
    if bounds is None:
        return "requirement_failed:query_hadamard_layers"
    first_h, last_h = bounds
    if _has_direct_query_output_synthesis(gates, query_wires):
        return "requirement_failed:direct_output_synthesis"
    expected_controls = {index for index, bit in enumerate(reversed(secret)) if bit == "1"}
    if algorithm == "deutsch_jozsa" and semantics.get("oracle_class") == "constant_zero":
        # The constant-zero phase oracle is the identity, so a separate
        # kickback target is optional. The surrounding Hadamard layers are
        # sufficient algorithmic evidence and work on a four-qubit circuit.
        return None
    if algorithm == "bernstein_vazirani" and _has_phase_oracle(
        gates,
        expected_controls,
        first_h,
        last_h,
    ):
        return None
    return _phase_kickback_violation(gates, query_wires, expected_controls, first_h, last_h)


def _query_secret(semantics: Mapping[str, Any], algorithm: str) -> tuple[int, str] | None:
    secret = str(semantics.get("secret", ""))
    if algorithm == "deutsch_jozsa" and semantics.get("oracle_class") == "constant_zero":
        return 4, "0000"
    return (len(secret), secret) if secret and set(secret) <= {"0", "1"} else None


def _hadamard_layer_bounds(
    gates: list[Any],
    query_wires: tuple[int, ...],
) -> tuple[list[int], list[int]] | None:
    first_h: list[int] = []
    last_h: list[int] = []
    for wire in query_wires:
        positions = _gate_positions(gates, family=("h", 0), target=wire)
        if len(positions) < 2:
            return None
        first_h.append(positions[0])
        last_h.append(positions[-1])
    return first_h, last_h


def _has_direct_query_output_synthesis(gates: list[Any], query_wires: tuple[int, ...]) -> bool:
    return any(
        _operation_family(operation) == ("x", 0) and operation.quantum_wires == (wire,)
        for operation in gates
        for wire in query_wires
    )


def _phase_kickback_violation(
    gates: list[Any],
    query_wires: tuple[int, ...],
    expected_controls: set[int],
    first_h: list[int],
    last_h: list[int],
) -> str | None:
    ancilla = len(query_wires)
    minus_prep, one_prep = _ancilla_kickback_preparations(gates, ancilla)
    if not minus_prep and not one_prep:
        return "requirement_failed:phase_kickback_preparation"
    ancilla_x = _gate_positions(gates, family=("x", 0), target=ancilla)
    ancilla_h = _gate_positions(gates, family=("h", 0), target=ancilla)
    prep_end = ancilla_h[0] if minus_prep and ancilla_h else ancilla_x[0]

    def in_query_window(wire: int, index: int) -> bool:
        return wire in query_wires and prep_end < index and first_h[wire] < index < last_h[wire]

    cx_controls = {
        control.wire
        for index, operation in enumerate(gates)
        if _operation_family(operation) == ("x", 1)
        and operation.quantum_wires == (ancilla,)
        and len(operation.controls) == 1
        for control in operation.controls
        if in_query_window(control.wire, index)
    }
    if minus_prep and cx_controls == expected_controls:
        return None
    cz_partners = {
        wire
        for index, operation in enumerate(gates)
        if _operation_family(operation) == ("z", 1) and ancilla in _semantic_operation_wires(operation)
        for wire in _semantic_operation_wires(operation) - {ancilla}
        if in_query_window(wire, index)
    }
    if one_prep and not ancilla_h and cz_partners == expected_controls:
        return None
    return "requirement_failed:oracle_interactions"


def _has_phase_oracle(
    gates: list[Any],
    expected_wires: set[int],
    first_h: list[int],
    last_h: list[int],
) -> bool:
    """Return whether exact Z phases encode a Bernstein-Vazirani secret."""
    observed = {
        operation.quantum_wires[0]
        for index, operation in enumerate(gates)
        if _operation_family(operation) == ("z", 0)
        and len(operation.quantum_wires) == 1
        and not operation.controls
        and operation.quantum_wires[0] < len(first_h)
        and first_h[operation.quantum_wires[0]] < index < last_h[operation.quantum_wires[0]]
    }
    return observed == expected_wires


def _simon_recipe_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    """Require Simon query layers and a nontrivial query-to-output oracle."""
    period = str(semantics.get("hidden_period", ""))
    if not period or set(period) - {"0", "1"}:
        return "requirement_failed:algorithm_recipe_contract"
    query_count = len(period)
    if program.num_qubits < query_count + max(1, query_count - 1):
        return "requirement_failed:simon_output_register"
    query_wires = set(range(query_count))
    output_wires = set(range(query_count, program.num_qubits))
    gates = [operation for operation in program.operations if operation.kind is OperationKind.GATE]
    first_h: dict[int, int] = {}
    last_h: dict[int, int] = {}
    for wire in query_wires:
        positions = _gate_positions(gates, family=("h", 0), target=wire)
        if len(positions) < 2:
            return "requirement_failed:query_hadamard_layers"
        first_h[wire], last_h[wire] = positions[0], positions[-1]
    interacted: set[int] = set()
    touched_outputs: set[int] = set()
    for index, operation in enumerate(gates):
        if _operation_family(operation) != ("x", 1) or len(operation.controls) != 1:
            continue
        control = operation.controls[0].wire
        targets = set(operation.quantum_wires)
        if control in query_wires and targets <= output_wires and first_h[control] < index < last_h[control]:
            interacted.add(control)
            touched_outputs.update(targets)
    if interacted != query_wires or len(touched_outputs) < max(1, query_count - 1):
        return "requirement_failed:simon_oracle_interactions"
    return None


def _algorithm_recipe_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    """Validate prompt-required algorithm scaffolds against direct answer synthesis."""
    recipe = str(semantics.get("algorithm_recipe", ""))
    if recipe == "phase_estimation_scaffold":
        return _phase_estimation_scaffold_violation(program, semantics)
    if recipe == "hhl_diagonal_2x2":
        return _hhl_diagonal_2x2_violation(program)
    if recipe == "teleportation_q0_q1_q2":
        return _teleportation_recipe_violation(program)
    operation_recipe = semantics.get("operation_recipe")
    algorithm = str(semantics.get("algorithm", ""))
    if operation_recipe == "prompt_exact" and algorithm == "single_simon_query":
        return _exact_simon_query_violation(program)
    if operation_recipe == "prompt_exact" and algorithm == "phase_kickback_period_finding":
        return _period_finding_phase_kickback_violation(program)
    return None


def _phase_estimation_scaffold_violation(
    program: Program,
    semantics: Mapping[str, Any],
) -> str | None:
    phase_wires = set(_program_measurement_wires(program))
    work_wires = set(range(program.num_qubits)) - phase_wires
    if not phase_wires or not work_wires:
        return "requirement_failed:phase_estimation_registers"
    gates = _gate_operations(program)
    cross_positions = [
        index
        for index, operation in enumerate(gates)
        if phase_wires & _semantic_operation_wires(operation) and work_wires & _semantic_operation_wires(operation)
    ]
    active_phase_wires = {
        wire for index in cross_positions for wire in phase_wires & _semantic_operation_wires(gates[index])
    }
    # For an order-two unitary such as Pauli X, every power after U**1 is the
    # identity. Requiring interactions from two clock wires would reject the
    # exact minimal QPE circuit while adding no algorithmic evidence.
    minimum_active_wires = 1 if semantics.get("unitary") == "pauli_x" else 2
    if len(active_phase_wires) < min(minimum_active_wires, len(phase_wires)):
        return "requirement_failed:controlled_evolution_register"
    first_evolution = min(cross_positions)
    if any(
        not any(
            index < first_evolution and _operation_family(operation) == ("h", 0) and operation.quantum_wires == (wire,)
            for index, operation in enumerate(gates)
        )
        for wire in phase_wires
    ):
        return "requirement_failed:phase_estimation_preparation"
    last_evolution = max(cross_positions)
    if not any(
        index > last_evolution
        and _semantic_operation_wires(operation) <= phase_wires
        and (_operation_family(operation) == ("h", 0) or len(_semantic_operation_wires(operation)) >= 2)
        for index, operation in enumerate(gates)
    ):
        return "requirement_failed:inverse_phase_transform"
    return None


def _hhl_diagonal_2x2_violation(program: Program) -> str | None:
    if program.num_qubits < 4:
        return "requirement_failed:hhl_registers"
    gates = _gate_operations(program)
    clock_wires = {0, 1}
    system_wire = 2
    success_wire = 3
    for clock in clock_wires:
        if len(_gate_positions(gates, family=("h", 0), target=clock)) < 2:
            return "requirement_failed:hhl_clock_transforms"
    success_positions = [
        index
        for index, operation in enumerate(gates)
        if success_wire in _semantic_operation_wires(operation)
        and bool(clock_wires & _semantic_operation_wires(operation))
    ]
    if not success_positions:
        return "requirement_failed:hhl_reciprocal_rotation"
    first_success = min(success_positions)
    last_success = max(success_positions)
    for clock in clock_wires:
        interactions = [
            index
            for index, operation in enumerate(gates)
            if {clock, system_wire} <= _semantic_operation_wires(operation)
        ]
        if not any(index < first_success for index in interactions) or not any(
            index > last_success for index in interactions
        ):
            return "requirement_failed:hhl_evolution_uncompute"
    return None


def _teleportation_recipe_violation(program: Program) -> str | None:
    """Require the standard q0 -> q2 teleportation construction.

    The protocol evidence, in order on padding-free gates, is:

    1. a Bell pair between qubits 1 and 2 (a Hadamard on either qubit
       followed by an entangling gate on the pair);
    2. the Bell-basis interaction between Alice's qubits 0 and 1; and
    3. Pauli corrections onto Bob's qubit 2: an X-family correction linked to
       qubit 1 and a Z-family correction linked to qubit 0, either coherent
       (deferred measurement) or classically conditioned.

    A direct SWAP or any relabeling shortcut has no {1,2} Bell preparation
    followed by a {0,1} Bell measurement, so it fails here even though it
    matches the identity channel numerically. Identity gadgets that only
    tick the structural boxes are rejected by requiring a live Bell pair
    immediately before Alice's Bell interaction and by rejecting circuits
    whose net effect leaves Alice's qubits in |00> (the SWAP signature).
    """
    gates = list(_noncanceling_gate_operations(program))
    bell = next(
        (index for index, operation in enumerate(gates) if _semantic_operation_wires(operation) == {1, 2}),
        None,
    )
    if bell is None or not any(
        _operation_family(operation) == ("h", 0) and operation.quantum_wires[0] in (1, 2) for operation in gates[:bell]
    ):
        return "requirement_failed:teleportation_bell_pair"
    bell_measurement = next(
        (
            index
            for index, operation in enumerate(gates)
            if index > bell and _semantic_operation_wires(operation) == {0, 1}
        ),
        None,
    )
    if bell_measurement is None:
        return "requirement_failed:teleportation_bell_measurement"
    if not _teleportation_bell_pair_present(program.num_qubits, gates, bell_measurement):
        return "requirement_failed:teleportation_bell_pair"
    if not _teleportation_correction_present(gates, bell_measurement, "x", 1):
        return "requirement_failed:teleportation_x_correction"
    if not _teleportation_correction_present(gates, bell_measurement, "z", 0):
        return "requirement_failed:teleportation_z_correction"
    if _teleportation_is_swap_shortcut(program.num_qubits, gates):
        return "requirement_failed:teleportation_swap_shortcut"
    return None


def _teleportation_bell_pair_present(num_qubits: int, gates: list[Any], bell_measurement: int) -> bool:
    """Return whether qubits 1 and 2 remain entangled before Alice's interaction."""
    try:
        import numpy as np

        from qceval.semantics.verifiers.dynamic.apply import _apply_operation
        from qceval.semantics.verifiers.dynamic.simulator import reduced_density_matrix
    except Exception:  # noqa: BLE001 - structural evidence remains if simulation fails.
        return True
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    state[0] = 1.0
    try:
        for operation in gates[:bell_measurement]:
            state = _apply_operation(state, operation, num_qubits)
        purities = []
        for wire in (1, 2):
            reduced = reduced_density_matrix(state, (wire,), num_qubits)
            purities.append(float(np.real(np.trace(reduced @ reduced))))
    except Exception:  # noqa: BLE001 - structural evidence remains if simulation fails.
        return True
    return all(purity <= 0.5 + 1e-6 for purity in purities)


def _teleportation_is_swap_shortcut(num_qubits: int, gates: list[Any]) -> bool:
    """Return whether Alice's qubits are left in |00>, the direct-SWAP signature."""
    try:
        import numpy as np

        from qceval.semantics.verifiers.dynamic.apply import _apply_operation
    except Exception:  # noqa: BLE001 - do not reject when simulation is unavailable.
        return False
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    state[0] = 1.0
    try:
        for operation in gates:
            state = _apply_operation(state, operation, num_qubits)
    except Exception:  # noqa: BLE001 - do not reject when simulation is unavailable.
        return False
    prob00 = float(sum(abs(amplitude) ** 2 for index, amplitude in enumerate(state) if (index & 0b11) == 0))
    return prob00 >= 1.0 - 1e-6


def _teleportation_correction_present(
    gates: list[Any],
    after: int,
    base_family: str,
    linked_wire: int,
) -> bool:
    for index, operation in enumerate(gates):
        if index <= after or _operation_family(operation)[0] != base_family:
            continue
        wires = _semantic_operation_wires(operation)
        if 2 not in wires:
            continue
        if linked_wire in wires or operation.condition is not None:
            return True
    return False


def _exact_simon_query_violation(program: Program) -> str | None:
    gates = _gate_operations(program)
    required_pairs = {(0, 3), (2, 3), (1, 4)}
    for input_wire in range(3):
        hadamards = _gate_positions(gates, family=("h", 0), target=input_wire)
        if len(hadamards) < 2:
            return "requirement_failed:simon_hadamard_layers"
        for control, target in required_pairs:
            if control != input_wire:
                continue
            if not any(
                hadamards[0] < index < hadamards[-1]
                and _operation_family(operation) == ("x", 1)
                and operation.quantum_wires == (target,)
                and tuple(item.wire for item in operation.controls) == (control,)
                for index, operation in enumerate(gates)
            ):
                return "requirement_failed:simon_oracle_interactions"
    return None


def _period_finding_phase_kickback_violation(program: Program) -> str | None:
    """Require a genuine phase-kickback query on qubit 0 with ancilla qubit 2.

    Every behaviorally exact convention is accepted:

    - a ``|->`` ancilla (prepared X-then-H or H-then-Z) queried through a
      CX(0 -> 2) oracle; or
    - a ``|1>`` ancilla (prepared by X alone) queried through a CZ on
      qubits {0, 2}, which induces the identical kickback phase.
    """
    gates = _gate_operations(program)
    phase_h = _gate_positions(gates, family=("h", 0), target=0)
    spectator_h = _gate_positions(gates, family=("h", 0), target=1)
    if len(phase_h) < 2 or len(spectator_h) < 2:
        return "requirement_failed:phase_kickback_preparation"
    minus_prep, one_prep = _ancilla_kickback_preparations(gates, 2)
    if not minus_prep and not one_prep:
        return "requirement_failed:phase_kickback_preparation"
    cx_oracle = any(
        phase_h[0] < index < phase_h[-1]
        and _operation_family(operation) == ("x", 1)
        and operation.quantum_wires == (2,)
        and tuple(item.wire for item in operation.controls) == (0,)
        for index, operation in enumerate(gates)
    )
    cz_oracle = any(
        phase_h[0] < index < phase_h[-1]
        and _operation_family(operation) == ("z", 1)
        and _semantic_operation_wires(operation) == {0, 2}
        for index, operation in enumerate(gates)
    )
    if (minus_prep and cx_oracle) or (one_prep and cz_oracle):
        return None
    return "requirement_failed:phase_kickback_oracle"


def _ancilla_kickback_preparations(gates: list[Any], ancilla: int) -> tuple[bool, bool]:
    """Classify how the kickback ancilla is prepared.

    Returns:
        ``(minus_prep, one_prep)`` where ``minus_prep`` means a ``|->``
        preparation (X then H, or H then Z) and ``one_prep`` means a bare X
        preparation into ``|1>``.
    """
    ancilla_x = _gate_positions(gates, family=("x", 0), target=ancilla)
    ancilla_h = _gate_positions(gates, family=("h", 0), target=ancilla)
    ancilla_z = _gate_positions(gates, family=("z", 0), target=ancilla)
    minus_prep = bool(ancilla_x and ancilla_h and ancilla_x[0] < ancilla_h[0]) or bool(
        ancilla_h and ancilla_z and ancilla_h[0] < ancilla_z[0]
    )
    one_prep = bool(ancilla_x)
    return minus_prep, one_prep


def _gate_operations(program: Program) -> list[Any]:
    return [operation for operation in program.operations if operation.kind is OperationKind.GATE]


def _semantic_operation_wires(operation: Any) -> set[int]:
    return set(operation.quantum_wires) | {control.wire for control in operation.controls}


def _program_measurement_wires(program: Program) -> tuple[int, ...]:
    return tuple(
        wire
        for operation in program.operations
        if operation.kind is OperationKind.MEASUREMENT
        for wire in operation.quantum_wires
    )


def _gate_positions(
    gates: list[Any],
    *,
    family: tuple[str, int],
    target: int,
) -> list[int]:
    return [
        index
        for index, operation in enumerate(gates)
        if _operation_family(operation) == family and operation.quantum_wires == (target,)
    ]


def _qec_requirement_violation(
    contract: Contract,
    program: Program,
    semantics: Mapping[str, Any],
    framework: str,
    execution_metadata: Mapping[str, Any],
    source_code: str | None,
    arguments: tuple[Any, ...],
) -> str | None:
    del contract  # QEC IR checks are program/argument driven after lowering.
    checks = (
        _minimum_qubits_violation(program, semantics),
        _execution_method_violation(framework, execution_metadata, semantics),
        _source_restriction_violation(source_code, semantics),
        _source_state_preparation_violation(source_code, semantics),
        _qec_state_preparation_violation(program, semantics),
        _encoder_state_before_ancilla_use_violation(program, arguments, semantics),
        _required_interaction_violation(program, semantics),
        _connected_interaction_groups_violation(program, semantics),
        _inter_group_before_intra_group_violation(program, semantics),
        _argument_conditioned_gate_violation(program, arguments, semantics),
        _controlled_correction_violation(program, semantics),
    )
    return next((reason for reason in checks if reason is not None), None)


def _minimum_qubits_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    minimum = semantics.get("min_num_qubits")
    if isinstance(minimum, int) and not isinstance(minimum, bool) and program.num_qubits < minimum:
        return "requirement_failed:min_num_qubits"
    return None


def _execution_method_violation(
    framework: str,
    metadata: Mapping[str, Any],
    semantics: Mapping[str, Any],
) -> str | None:
    by_framework = semantics.get("forbidden_probability_methods")
    if not isinstance(by_framework, Mapping):
        return None
    methods = by_framework.get(framework)
    if not isinstance(methods, list | tuple):
        return None
    if metadata.get("probability_method") in methods:
        return "requirement_failed:forbidden_probability_method"
    return None


def _source_restriction_violation(source_code: str | None, semantics: Mapping[str, Any]) -> str | None:
    return _source_policy_violation(semantics, source_code)


def _source_state_preparation_violation(
    source_code: str | None,
    semantics: Mapping[str, Any],
) -> str | None:
    if semantics.get("forbid_state_preparation") is not True:
        return None
    blocked = {
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
    if source_call_names(source_code) & blocked:
        return "requirement_failed:forbid_state_preparation_source"
    if source_dynamic_features(source_code):
        return "requirement_failed:forbid_dynamic_reflection"
    return None


def _semantic_gate_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    if semantics.get("decomposition_required") is True:
        reason = _gate_basis_violation(
            program,
            {
                "forbidden": [
                    "unitary",
                    "unitarygate",
                    "matrixgate",
                    "qubitunitary",
                    "dense_unitary",
                ]
            },
        )
        if reason is not None:
            return reason
    allowed = semantics.get("allowed_gate_families")
    if isinstance(allowed, list | tuple):
        reason = _gate_basis_violation(program, {"allowed": allowed})
        if reason is not None:
            return reason
    gate_class = semantics.get("allowed_gate_class")
    if isinstance(gate_class, str) and gate_class.lower().startswith("clifford"):
        reason = _clifford_gate_class_violation(program)
        if reason is not None:
            return reason
    forbidden = semantics.get("forbidden_gate_families")
    if not isinstance(forbidden, list | tuple):
        return None
    return _gate_basis_violation(program, {"forbidden": forbidden})


def _has_bell_prepare_uncompute(program: Program) -> bool:
    """Return whether an H-CX ... CX-H Bell round trip is present.

    Any equivalent Bell convention on qubits 0 and 1 is accepted: the Hadamard
    may sit on either qubit and the CX may take either orientation, provided
    the uncompute half exactly reverses the prepare half.
    """
    gates = [
        (operation.name.lower(), _operation_family(operation), operation.quantum_wires, operation.controls)
        for operation in program.operations
        if operation.kind is OperationKind.GATE
    ]
    for start in range(len(gates) - 3):
        first, prepare, uncompute, last = gates[start : start + 4]
        for hadamard_wire in (0, 1):
            if first[1] != ("h", 0) or first[2] != (hadamard_wire,):
                continue
            if last[1] != ("h", 0) or last[2] != (hadamard_wire,):
                continue
            if (
                _is_bell_cx(prepare, hadamard_wire)
                and prepare[2:] == uncompute[2:]
                and _is_bell_cx(uncompute, hadamard_wire)
            ):
                return True
    return False


def _is_bell_cx(gate: tuple[Any, tuple[str, int], tuple[int, ...], tuple[Any, ...]], control: int) -> bool:
    """Return whether a normalized gate is CX from ``control`` on qubits {0, 1}."""
    _, family, targets, controls = gate
    return family == ("x", 1) and targets == (1 - control,) and len(controls) == 1 and controls[0].wire == control


def _has_exact_repeated_steps(program: Program, steps: int) -> bool:
    if steps < 1:
        return False
    operations = tuple(
        _without_location(operation)
        for operation in program.operations
        if operation.kind not in {OperationKind.BARRIER, OperationKind.MEASUREMENT}
    )
    for prefix_length in range(min(program.num_qubits, len(operations)) + 1):
        body = operations[prefix_length:]
        if not body or len(body) % steps:
            continue
        block_length = len(body) // steps
        if block_length < 2:
            continue
        block = body[:block_length]
        if all(body[index * block_length : (index + 1) * block_length] == block for index in range(steps)):
            return True
    return False


def _without_location(operation: Any) -> Any:
    return replace(operation, source_location=None)
