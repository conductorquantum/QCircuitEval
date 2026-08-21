"""Cross-case Program IR invariants for finite parameter domains."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import cache
from typing import Any

from qceval.semantics.contracts import Contract
from qceval.semantics.contracts.kinds import FrozenArray, FrozenObject
from qceval.semantics.ir import OperationKind, Program
from qceval.semantics.verifiers.requirements.gate_family import _gate_family, _operation_family


def case_program_invariance_required(contract: Contract) -> bool:
    """Return whether a contract declares a cross-case program invariant.

    Args:
        contract: Behavior contract to inspect.
    """
    return _case_policy(contract) is not None


def case_program_invariance_violation(
    contract: Contract,
    cases: Sequence[tuple[tuple[Any, ...], Program]],
) -> str | None:
    """Require exhaustive cases to differ only by declared input gates.

    A finite truth table alone is vulnerable when candidate source branches on
    its public arguments and synthesizes each expected answer directly. QEC
    contracts therefore identify the physical input/error gates that may vary
    between cases. After removing exactly those declared deltas, every case
    must lower to the same Program IR.

    Args:
        contract: Behavior contract declaring the permitted case deltas.
        cases: Bound arguments paired with their lowered programs.

    Returns:
        Stable requirement-failure reason, or ``None`` when all cases match.
    """
    policy = _case_policy(contract)
    if policy is None:
        return None
    reference_raw = policy.get("reference_arguments")
    rules = policy.get("allowed_case_deltas")
    if not isinstance(reference_raw, Sequence) or isinstance(reference_raw, str | bytes):
        return "requirement_failed:case_program_invariance_contract"
    if not isinstance(rules, Sequence) or isinstance(rules, str | bytes):
        return "requirement_failed:case_program_invariance_contract"
    reference_arguments = tuple(reference_raw)
    reference = next((program for arguments, program in cases if arguments == reference_arguments), None)
    if reference is None:
        return "requirement_failed:case_program_invariance_contract"
    for arguments, program in cases:
        permitted = _permitted_deltas(arguments, reference_arguments, rules)
        if permitted is None or not _program_matches_with_deltas(reference, program, permitted):
            return "requirement_failed:case_program_invariance"
    return None


def _case_policy(contract: Contract) -> Mapping[str, Any] | None:
    for requirement in contract.requirements:
        if requirement.requirement_id != "semantic_requirements":
            continue
        semantics = _plain(requirement.value)
        if not isinstance(semantics, Mapping):
            return None
        policy = semantics.get("case_program_invariance")
        return policy if isinstance(policy, Mapping) else None
    return None


def _permitted_deltas(
    arguments: tuple[Any, ...],
    reference: tuple[Any, ...],
    rules: Sequence[Any],
) -> Counter[tuple[str, int]] | None:
    permitted: Counter[tuple[str, int]] = Counter()
    for raw_rule in rules:
        parsed = _parse_delta_rule(raw_rule, len(arguments), len(reference))
        if parsed is None:
            return None
        activated = _activated_deltas(arguments, reference, parsed)
        if activated is None:
            return None
        permitted.update(activated)
    return permitted


def _parse_delta_rule(
    raw_rule: Any,
    argument_count: int,
    reference_count: int,
) -> tuple[int, str, str, set[int]] | None:
    if not isinstance(raw_rule, Mapping):
        return None
    index = raw_rule.get("argument_index")
    activation = raw_rule.get("activation")
    gates = raw_rule.get("gate_names")
    wires = raw_rule.get("wires")
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= argument_count
        or index >= reference_count
        or not isinstance(activation, str)
        or not isinstance(gates, Sequence)
        or isinstance(gates, str | bytes)
        or not isinstance(wires, Sequence)
        or isinstance(wires, str | bytes)
    ):
        return None
    aliases = {"paulix": "x", "pauliy": "y", "pauliz": "z"}
    families = {aliases.get(str(name).lower(), _gate_family(str(name), 0)[0]) for name in gates}
    if len(families) != 1:
        return None
    return index, activation, next(iter(families)), {int(wire) for wire in wires}


def _activated_deltas(
    arguments: tuple[Any, ...],
    reference: tuple[Any, ...],
    rule: tuple[int, str, str, set[int]],
) -> Counter[tuple[str, int]] | None:
    index, activation, family, declared_wires = rule
    value = arguments[index]
    if activation == "equals_one":
        if reference[index] != 0 or value not in {0, 1}:
            return None
        return Counter((family, wire) for wire in declared_wires) if value == 1 else Counter()
    if activation != "selected_wire" or reference[index] is not None:
        return None
    if value is None:
        return Counter()
    if isinstance(value, bool) or not isinstance(value, int) or value not in declared_wires:
        return None
    return Counter({(family, value): 1})


def _program_matches_with_deltas(
    reference: Program,
    candidate: Program,
    permitted: Counter[tuple[str, int]],
) -> bool:
    if (
        candidate.num_qubits != reference.num_qubits
        or candidate.num_clbits != reference.num_clbits
        or candidate.global_phase != reference.global_phase
        or candidate.classical_render_order != reference.classical_render_order
    ):
        return False
    expected = tuple(replace(operation, source_location=None) for operation in reference.operations)
    actual = tuple(replace(operation, source_location=None) for operation in candidate.operations)
    if len(actual) != len(expected) + sum(permitted.values()):
        return False
    # Framework schedulers may move operations on disjoint wires into earlier
    # moments when an argument-dependent input gate is inserted. Compare the
    # induced dependency trace on every quantum and classical resource rather
    # than requiring one arbitrary global topological ordering. Operations
    # sharing a resource remain strictly ordered, so this is equivalent to
    # comparing the circuit DAG after deleting exactly the declared deltas.
    for wire in range(reference.num_qubits):
        expected_trace = tuple(operation for operation in expected if wire in _quantum_resources(operation))
        actual_trace = tuple(operation for operation in actual if wire in _quantum_resources(operation))
        wire_deltas = Counter({label: count for label, count in permitted.items() if label[1] == wire})
        if not _trace_matches_with_deltas(expected_trace, actual_trace, wire_deltas):
            return False
    for bit in range(reference.num_clbits):
        expected_trace = tuple(operation for operation in expected if bit in _classical_resources(operation))
        actual_trace = tuple(operation for operation in actual if bit in _classical_resources(operation))
        if expected_trace != actual_trace:
            return False
    expected_unbound = tuple(
        operation for operation in expected if not _quantum_resources(operation) and not _classical_resources(operation)
    )
    actual_unbound = tuple(
        operation for operation in actual if not _quantum_resources(operation) and not _classical_resources(operation)
    )
    return expected_unbound == actual_unbound


def _trace_matches_with_deltas(
    expected: tuple[Any, ...],
    actual: tuple[Any, ...],
    permitted: Counter[tuple[str, int]],
) -> bool:
    labels = tuple(sorted(permitted))
    initial = tuple(permitted[label] for label in labels)

    @cache
    def matches(expected_index: int, actual_index: int, remaining: tuple[int, ...]) -> bool:
        if expected_index == len(expected) and actual_index == len(actual):
            return not any(remaining)
        if actual_index == len(actual):
            return False
        operation = actual[actual_index]
        if (
            expected_index < len(expected)
            and operation == expected[expected_index]
            and matches(expected_index + 1, actual_index + 1, remaining)
        ):
            return True
        for label_index, label in enumerate(labels):
            if remaining[label_index] <= 0 or not _matches_delta(operation, label):
                continue
            updated = list(remaining)
            updated[label_index] -= 1
            if matches(expected_index, actual_index + 1, tuple(updated)):
                return True
        return False

    return matches(0, 0, initial)


def _quantum_resources(operation: Any) -> set[int]:
    return set(operation.quantum_wires) | {control.wire for control in operation.controls}


def _classical_resources(operation: Any) -> set[int]:
    resources = set(operation.classical_bits)
    if operation.condition is not None:
        resources.update(operation.condition.bits)
    return resources


def _matches_delta(operation: Any, label: tuple[str, int]) -> bool:
    family, wire = label
    return (
        operation.kind is OperationKind.GATE
        and _operation_family(operation)[0] == family
        and operation.quantum_wires == (wire,)
        and not operation.controls
        and not operation.parameters
        and operation.condition is None
        and operation.power is None
    )


def _plain(value: Any) -> Any:
    if isinstance(value, FrozenArray):
        return [_plain(item) for item in value.items]
    if isinstance(value, FrozenObject):
        return {key: _plain(item) for key, item in value.items}
    return value


__all__ = ["case_program_invariance_required", "case_program_invariance_violation"]
