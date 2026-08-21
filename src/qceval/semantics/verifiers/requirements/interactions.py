"""Program-IR interaction and QEC topology requirement checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import combinations
from typing import Any

from qceval.semantics.ir import Operation, OperationKind, Program
from qceval.semantics.verifiers.requirements.gate_family import (
    _gate_family,
    _operation_family,
    _operation_wires,
)

_SELF_INVERSE_GATE_NAMES = {
    "ccx",
    "ccz",
    "ch",
    "cnot",
    "controlled_h",
    "controlled_not",
    "cswap",
    "cx",
    "cy",
    "cz",
    "fredkin",
    "h",
    "mcx",
    "swap",
    "toffoli",
    "x",
    "y",
    "z",
}


def _interaction_pairs(program: Program) -> set[tuple[int, int]]:
    """Return unordered wire pairs that share a gate."""
    return _interaction_pairs_from_operations(program.operations)


def _effective_interaction_pairs(program: Program) -> set[tuple[int, int]]:
    """Return interactions after removing obvious canceling gate padding."""
    return _interaction_pairs_from_operations(_noncanceling_gate_operations(program))


def _interaction_pairs_from_operations(operations: Sequence[Any]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for operation in operations:
        if operation.kind is not OperationKind.GATE:
            continue
        for first, second in combinations(sorted(_operation_wires(operation)), 2):
            result.add(_interaction_pair(first, second))
    return result


def _interaction_pair(first: int, second: int) -> tuple[int, int]:
    return (min(first, second), max(first, second))


def _noncanceling_gate_operations(program: Program) -> tuple[Any, ...]:
    """Remove canceling gate padding from Program IR.

    Barriers do not prevent cancellation because they have no quantum
    semantics. Measurements, resets, and other non-gate operations delimit
    segments. This bounded peephole rule deliberately avoids algebraic
    commutation, so it cannot rewrite legitimately separated QEC operations.

    A canceling inverse or self-inverse pair only counts as padding when it
    is gratuitous: the two gates are immediately adjacent in the raw gate
    stream and the pair does not enclose (bracket) other canceling structure.
    Nested cancellations are the signature of legitimate reverse-order
    uncomputation (encode ... decode), where the mandatory no-error case
    places inverse encoding gates next to each other; those operations are
    kept so that they still witness required interactions.
    """
    effective: list[Any] = []
    segment: list[Any] = []
    for operation in program.operations:
        if operation.kind is OperationKind.BARRIER:
            continue
        if operation.kind is not OperationKind.GATE:
            effective.extend(_padding_free_segment(segment))
            segment.clear()
            continue
        segment.append(operation)
    effective.extend(_padding_free_segment(segment))
    return tuple(effective)


def _padding_free_segment(segment: Sequence[Any]) -> tuple[Any, ...]:
    """Drop gratuitous canceling pairs from one uninterrupted gate segment."""
    stack: list[int] = []
    removed: list[tuple[int, int]] = []
    for index, operation in enumerate(segment):
        if stack and _gate_pair_cancels(segment[stack[-1]], operation):
            removed.append((stack.pop(), index))
        else:
            stack.append(index)
    padding: set[int] = set()
    for first, second in removed:
        adjacent = second == first + 1
        enclosed = any(outer < first and second < inner for outer, inner in removed)
        if adjacent and not enclosed:
            padding.update((first, second))
    return tuple(operation for index, operation in enumerate(segment) if index not in padding)


def _gate_pair_cancels(first: Any, second: Any) -> bool:
    first_plain = replace(first, inverse=False, source_location=None)
    second_plain = replace(second, inverse=False, source_location=None)
    if first_plain != second_plain:
        return False
    if first.inverse != second.inverse:
        return True
    return not first.parameters and first.power is None and first.name.lower() in _SELF_INVERSE_GATE_NAMES


def _declared_pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        return (-1, -1)
    return _interaction_pair(int(value[0]), int(value[1]))


def _declared_directed_pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        return (-1, -1)
    return int(value[0]), int(value[1])


def _required_interaction_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    reject_padding = semantics.get("reject_canceling_interaction_padding") is True
    present = _effective_interaction_pairs(program) if reject_padding else _interaction_pairs(program)
    effective = (
        _noncanceling_gate_operations(program)
        if reject_padding
        else tuple(operation for operation in program.operations if operation.kind is OperationKind.GATE)
    )
    checks = (
        _required_pair_violation(semantics.get("required_interactions"), present),
        _required_sequence_violation(program, semantics.get("required_any_interaction_sequences")),
        _required_controlled_x_violation(effective, semantics.get("required_controlled_x_interactions")),
        _required_parity_violation(effective, semantics.get("required_parity_interactions")),
    )
    return next((reason for reason in checks if reason is not None), None)


def _required_pair_violation(required: Any, present: set[tuple[int, int]]) -> str | None:
    if isinstance(required, list | tuple):
        for raw_pair in required:
            if not isinstance(raw_pair, list | tuple) or len(raw_pair) != 2:
                return "requirement_failed:invalid_interaction_contract"
            pair = tuple(sorted((int(raw_pair[0]), int(raw_pair[1]))))
            if pair not in present:
                return f"requirement_failed:missing_interaction:{pair[0]}-{pair[1]}"
    return None


def _required_sequence_violation(program: Program, alternatives: Any) -> str | None:
    if isinstance(alternatives, list | tuple) and not any(
        _interaction_sequence_present(program, sequence) for sequence in alternatives
    ):
        return "requirement_failed:required_any_interaction_sequences"
    return None


def _required_controlled_x_violation(effective: Sequence[Any], directed: Any) -> str | None:
    if isinstance(directed, list | tuple):
        for raw_pair in directed:
            control, target = _declared_directed_pair(raw_pair)
            if control < 0 or not _has_controlled_x_interaction(effective, control, target):
                return f"requirement_failed:missing_controlled_x_interaction:{control}-{target}"
    return None


def _required_parity_violation(effective: Sequence[Any], parity: Any) -> str | None:
    if isinstance(parity, list | tuple):
        for raw_pair in parity:
            data, ancilla = _declared_directed_pair(raw_pair)
            if data < 0 or not _has_parity_interaction(effective, data, ancilla):
                return f"requirement_failed:missing_parity_interaction:{data}-{ancilla}"
    return None


def _has_controlled_x_interaction(operations: Sequence[Any], control: int, target: int) -> bool:
    return any(
        operation.kind is OperationKind.GATE
        and operation.quantum_wires == (target,)
        and _operation_family(operation) == ("x", 1)
        and len(operation.controls) == 1
        and operation.controls[0].wire == control
        for operation in operations
    )


def _has_parity_interaction(operations: Sequence[Any], data: int, ancilla: int) -> bool:
    if _has_controlled_x_interaction(operations, data, ancilla):
        return True
    pair = _interaction_pair(data, ancilla)
    return any(
        operation.kind is OperationKind.GATE
        and _operation_family(operation) == ("z", 1)
        and _operation_wires(operation) == frozenset(pair)
        for operation in operations
    )


def _interaction_sequence_present(program: Program, raw_sequence: Any) -> bool:
    if not isinstance(raw_sequence, list | tuple):
        return False
    sequence = tuple(_declared_pair(value) for value in raw_sequence)
    if not sequence or (-1, -1) in sequence:
        return False
    position = 0
    for operation in program.operations:
        if operation.kind is not OperationKind.GATE:
            continue
        pairs = {
            _interaction_pair(first, second) for first, second in combinations(sorted(_operation_wires(operation)), 2)
        }
        if sequence[position] in pairs:
            position += 1
            if position == len(sequence):
                return True
    return False


def _connected_interaction_groups_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    groups = semantics.get("required_connected_interaction_groups")
    if not isinstance(groups, list | tuple):
        return None
    interactions = _effective_interaction_pairs(program)
    for raw_group in groups:
        if not isinstance(raw_group, list | tuple) or len(raw_group) < 2:
            return "requirement_failed:invalid_connected_interaction_group"
        group = {int(wire) for wire in raw_group}
        reached = {next(iter(group))}
        while True:
            expanded = (
                reached
                | {second for first, second in interactions if first in reached and second in group}
                | {first for first, second in interactions if second in reached and first in group}
            )
            if expanded == reached:
                break
            reached = expanded
        if reached != group:
            return "requirement_failed:disconnected_interaction_group"
    return None


class _GroupComponents:
    """Union-find over interaction groups."""

    def __init__(self, count: int) -> None:
        self._component = list(range(count))

    def _root(self, index: int) -> int:
        while self._component[index] != index:
            self._component[index] = self._component[self._component[index]]
            index = self._component[index]
        return index

    def union(self, first: int, second: int) -> None:
        self._component[self._root(first)] = self._root(second)

    def connected(self) -> bool:
        return len({self._root(index) for index in range(len(self._component))}) == 1


def _operation_group_pairs(operation: Operation, group_of: Mapping[int, int]) -> list[tuple[int, int]]:
    return [
        (group_of[first], group_of[second])
        for first, second in combinations(sorted(_operation_wires(operation)), 2)
        if first in group_of and second in group_of
    ]


def _inter_group_before_intra_group_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    """Require inter-group interactions to connect every group before any intra-group interaction.

    This makes outer-then-inner concatenated encoders construction-sensitive:
    a Shor-code outer phase-flip encoder must link the three blocks before the
    inner bit-flip encoders act within a block, so omitting the outer stage or
    running the cross-block gates after the inner encoders is rejected even
    when the shortcut is state-trivial on the graded inputs.
    """
    groups = semantics.get("required_inter_group_before_intra_group")
    if not isinstance(groups, list | tuple):
        return None
    if any(not isinstance(group, list | tuple) or len(group) < 2 for group in groups):
        return "requirement_failed:invalid_inter_group_contract"
    group_of = {int(wire): index for index, group in enumerate(groups) for wire in group}
    components = _GroupComponents(len(groups))
    for operation in _noncanceling_gate_operations(program):
        pairs = _operation_group_pairs(operation, group_of)
        for first, second in pairs:
            if first != second:
                components.union(first, second)
        if not components.connected() and any(first == second for first, second in pairs):
            return "requirement_failed:intra_group_interaction_before_inter_group_connection"
    if not components.connected():
        return "requirement_failed:missing_inter_group_connection"
    return None


def _encoder_state_before_ancilla_use_violation(
    program: Program,
    arguments: tuple[Any, ...],
    semantics: Mapping[str, Any],
) -> str | None:
    """Require the declared codeword state on the data register before ancilla use.

    Syndrome-extraction tasks are otherwise satisfiable without any encoder:
    the target syndromes only witness parities, which trivial data states can
    reproduce. At the contract's reference arguments (the no-error case), the
    data register is simulated up to the first data-to-ancilla interaction
    and must equal the uniform positive superposition over the declared
    codeword support, which is the logical zero state any correct encoder
    prepares regardless of its gate ordering.
    """
    spec = semantics.get("required_encoder_state_before_ancilla_use")
    if not isinstance(spec, Mapping):
        return None
    reason = "requirement_failed:required_encoder_state_before_ancilla_use"
    data_raw = spec.get("data_wires")
    ancilla_raw = spec.get("ancilla_wires")
    support_raw = spec.get("positive_uniform_support")
    reference = spec.get("reference_arguments")
    if (
        not isinstance(data_raw, Sequence)
        or not isinstance(ancilla_raw, Sequence)
        or not isinstance(support_raw, Sequence)
        or isinstance(support_raw, str | bytes)
        or not support_raw
        or not isinstance(reference, Sequence)
        or isinstance(reference, str | bytes)
    ):
        return "requirement_failed:invalid_encoder_state_contract"
    if not _arguments_match_reference(arguments, tuple(reference)):
        return None
    data_wires = tuple(int(wire) for wire in data_raw)
    ancilla_wires = {int(wire) for wire in ancilla_raw}
    support = [str(value) for value in support_raw]
    if any(len(value) != len(data_wires) or set(value) - {"0", "1"} for value in support):
        return "requirement_failed:invalid_encoder_state_contract"
    if program.num_qubits < max((*data_wires, *ancilla_wires)) + 1:
        return reason
    state = _state_before_ancilla_use(program, set(data_wires), ancilla_wires)
    if state is None:
        return reason
    return None if _data_state_matches_support(state, data_wires, support, program.num_qubits) else reason


def _arguments_match_reference(arguments: tuple[Any, ...], reference: tuple[Any, ...]) -> bool:
    if len(arguments) != len(reference):
        return False
    for actual, expected in zip(arguments, reference, strict=True):
        if expected is None or actual is None:
            if actual is not expected:
                return False
        elif isinstance(actual, bool) or actual != expected:
            return False
    return True


def _state_before_ancilla_use(
    program: Program,
    data_wires: set[int],
    ancilla_wires: set[int],
) -> Any | None:
    """Simulate raw gates up to the first data-to-ancilla coupling."""
    try:
        import numpy as np

        from qceval.semantics.verifiers.dynamic.apply import _apply_operation
    except Exception:  # noqa: BLE001 - a hard encoder requirement fails closed.
        return None
    state = np.zeros(2**program.num_qubits, dtype=np.complex128)
    state[0] = 1.0
    for operation in program.operations:
        if operation.kind is OperationKind.BARRIER:
            continue
        wires = _operation_wires(operation)
        conditioned = operation.condition is not None
        if (wires & data_wires and (wires & ancilla_wires or conditioned)) or (
            operation.kind is not OperationKind.GATE and wires & data_wires
        ):
            return state
        if operation.kind is not OperationKind.GATE:
            continue
        try:
            state = _apply_operation(state, operation, program.num_qubits)
        except Exception:  # noqa: BLE001 - unsupported gates fail closed here.
            return None
    return state


def _data_state_matches_support(
    state: Any,
    data_wires: tuple[int, ...],
    support: list[str],
    num_qubits: int,
    *,
    atol: float = 1e-6,
) -> bool:
    import numpy as np

    target = np.zeros(2 ** len(data_wires), dtype=np.complex128)
    for bitstring in support:
        # Support strings render the most-significant data wire first,
        # matching the contract's observation rendering convention.
        index = sum((1 << position) for position, char in enumerate(reversed(bitstring)) if char == "1")
        target[index] = 1.0
    target /= np.linalg.norm(target)
    try:
        # <t|rho_data|t> = sum_rest |sum_data conj(t)[data] psi[data, rest]|^2,
        # computed as one tensor contraction instead of materializing the
        # reduced density matrix (the O(4^n) Python loop is intractable at
        # QEC register sizes).
        tensor = np.asarray(state, dtype=np.complex128).reshape([2] * num_qubits)
        # Statevector is little-endian (index bit w <-> wire w), so wire w
        # lives on tensor axis num_qubits - 1 - w. Target bit p corresponds
        # to data_wires[p]; ordering axes [p=d-1 .. 0] makes the C-order
        # flattened data index equal the target index.
        data_axes = [num_qubits - 1 - wire for wire in data_wires]
        rest_axes = [axis for axis in range(num_qubits) if axis not in set(data_axes)]
        ordered = np.transpose(tensor, list(reversed(data_axes)) + rest_axes)
        matrix = ordered.reshape(2 ** len(data_wires), -1)
        projected = np.conjugate(target) @ matrix
        fidelity = float(np.sum(np.abs(projected) ** 2))
    except Exception:  # noqa: BLE001 - a hard encoder requirement fails closed.
        return False
    return fidelity >= 1.0 - atol


def _qec_state_preparation_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    if semantics.get("forbid_state_preparation") is not True:
        return None
    if any(operation.kind is OperationKind.STATE_PREPARATION for operation in program.operations):
        return "requirement_failed:forbid_state_preparation"
    return None


def _argument_conditioned_gate_violation(
    program: Program,
    arguments: tuple[Any, ...],
    semantics: Mapping[str, Any],
) -> str | None:
    spec = semantics.get("argument_conditioned_gate")
    if not isinstance(spec, Mapping):
        return None
    index = spec.get("argument_index")
    gates = spec.get("gate_names")
    wires = spec.get("wires")
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index >= len(arguments)
        or not isinstance(gates, list | tuple)
        or not isinstance(wires, list | tuple)
    ):
        return "requirement_failed:invalid_argument_gate_contract"
    selected = arguments[index]
    if selected is None:
        return None
    if isinstance(selected, bool) or not isinstance(selected, int):
        return "requirement_failed:invalid_argument_gate_value"
    allowed_wires = {int(value) for value in wires}
    if selected not in allowed_wires:
        return "requirement_failed:invalid_argument_gate_value"
    allowed_families = {_gate_family(str(value), 0) for value in gates}
    # The prompt requires the requested error to appear as an actual gate.
    # Inspect raw Program IR here: a legitimate injected error can be adjacent
    # to an identical data-preparation or correction gate and therefore cancel
    # algebraically even though the requested injection is present.
    for operation in program.operations:
        if (
            operation.kind is OperationKind.GATE
            and selected in operation.quantum_wires
            and _operation_family(operation) in allowed_families
        ):
            return None
    return "requirement_failed:argument_conditioned_gate"


def _controlled_correction_violation(program: Program, semantics: Mapping[str, Any]) -> str | None:
    spec = semantics.get("required_controlled_correction")
    if not isinstance(spec, Mapping):
        return None
    targets = spec.get("target_wires")
    controls = spec.get("control_wires")
    minimum = spec.get("min_controls")
    family = spec.get("gate_family")
    if (
        not isinstance(targets, list | tuple)
        or not isinstance(controls, list | tuple)
        or not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or minimum < 1
        or not isinstance(family, str)
    ):
        return "requirement_failed:invalid_controlled_correction_contract"
    target_wires = {int(wire) for wire in targets}
    control_wires = {int(wire) for wire in controls}
    required_base = _gate_family(family, 0)[0]
    for operation in _noncanceling_gate_operations(program):
        if (
            operation.kind is OperationKind.GATE
            and target_wires.intersection(operation.quantum_wires)
            and _operation_family(operation)[0] == required_base
            and sum(control.wire in control_wires for control in operation.controls) >= minimum
        ):
            return None
    return "requirement_failed:required_controlled_correction"
