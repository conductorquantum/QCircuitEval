"""Block collection and adaptive-QIR control-flow graph walking."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import replace
from typing import Any

from qceval.frameworks.cudaq.qir.models import (
    QirParseError,
    QirParseLimits,
    _BitPredicate,
    _Block,
    _Memory,
    _ParseContext,
    _QubitArray,
    _State,
    _Terminator,
)
from qceval.frameworks.cudaq.qir.ssa import (
    _evaluate_instruction,
    _global_constants,
)
from qceval.frameworks.cudaq.qir.tokens import (
    _SSA_REFERENCE,
    _VALUE,
    _strip_comment,
)
from qceval.semantics.ir import (
    IR_VERSION,
    ClassicalCondition,
    Operation,
    Program,
    Provenance,
    validate_program,
)

_LABEL = re.compile(r"^\s*([-a-zA-Z$._0-9]+):(?:\s*;.*)?$")

_FUNCTION = re.compile(r"^define\s+void\s+@(?P<name>[^(]+)\([^)]*\)\s*(?P<attributes>#[0-9]+)?\s*\{")

_ATTRIBUTES = re.compile(r"^attributes\s+(#[0-9]+)\s*=\s*\{(.*)\}\s*$")

_WIDTH = re.compile(r'"required(?P<kind>Qubits|Results)"="(?P<value>[0-9]+)"')

_DIRECT_BRANCH = re.compile(r"^br\s+label\s+%?([-a-zA-Z$._0-9]+)$")

_CONDITIONAL_BRANCH = re.compile(
    rf"^br\s+i1\s+({_VALUE}|true|false),\s+label\s+%?([-a-zA-Z$._0-9]+),\s+label\s+%?([-a-zA-Z$._0-9]+)$"
)

_RETURN = re.compile(r"^(?:ret\s+void|unreachable)$")


def parse_adaptive_qir(
    text: str,
    *,
    provenance: Provenance,
    limits: QirParseLimits | None = None,
) -> Program:
    """Parse one CUDA-Q adaptive-QIR module into Program IR.

    Args:
        text: LLVM textual QIR emitted by ``cudaq.translate``.
        provenance: Candidate/framework identity for the resulting Program.
        limits: Optional deterministic parser limits.

    Returns:
        Validated Program IR.

    Raises:
        QirParseError: If QIR is malformed, unsupported, or exceeds bounds.
    """
    limits = limits or QirParseLimits()
    if len(text.encode("utf-8")) > limits.max_text_bytes:
        raise QirParseError("adaptive QIR exceeds the text-size limit")
    body, attribute_id, entry_label = _entry_function(text)
    required_qubits, required_results = _required_widths(text, attribute_id)
    blocks = _parse_blocks(body, limits, entry_label)
    entry = next(iter(blocks))
    context = _ParseContext(blocks, _State(_global_constants(text)), [], required_qubits, required_results, limits)
    _walk_cfg(context, entry, stop=None, conditions={}, stack=(), depth=0)
    num_qubits = required_qubits or _inferred_qubits(context.operations)
    num_clbits = required_results or _inferred_clbits(context.operations)
    program = Program(
        IR_VERSION,
        num_qubits,
        num_clbits,
        tuple(context.operations),
        None,
        tuple(reversed(range(num_clbits))),
        provenance,
        ("cudaq_qir_adaptive",),
    )
    validate_program(program)
    return program


def _entry_function(text: str) -> tuple[list[str], str | None, str]:
    lines = text.splitlines()
    start = None
    attribute_id = None
    entry_label = "0"
    for index, line in enumerate(lines):
        match = _FUNCTION.match(line.strip())
        if match is not None and "__nvqpp__mlirgen__" in match.group("name"):
            start = index + 1
            attribute_id = match.group("attributes")
            arguments = _SSA_REFERENCE.findall(line[: line.find(")")])
            entry_label = str(max((int(item[1:]) for item in arguments if item[1:].isdigit()), default=-1) + 1)
            break
    if start is None:
        raise QirParseError("adaptive QIR entry function was not found")
    body: list[str] = []
    for line in lines[start:]:
        if line.strip() == "}":
            return body, attribute_id, entry_label
        body.append(line)
    raise QirParseError("adaptive QIR entry function is unterminated")


def _required_widths(text: str, attribute_id: str | None) -> tuple[int, int]:
    if attribute_id is None:
        return 0, 0
    for line in text.splitlines():
        match = _ATTRIBUTES.match(line.strip())
        if match is None or match.group(1) != attribute_id:
            continue
        widths = {item.group("kind"): int(item.group("value")) for item in _WIDTH.finditer(match.group(2))}
        return widths.get("Qubits", 0), widths.get("Results", 0)
    return 0, 0


def _parse_blocks(lines: list[str], limits: QirParseLimits, entry_label: str) -> dict[str, _Block]:
    pending = _collect_block_lines(lines, limits, entry_label)
    blocks = {
        label: _Block(label, tuple(instructions[:-1]), _parse_terminator(instructions[-1]))
        for label, instructions in pending.items()
    }
    for block in blocks.values():
        missing = [target for target in block.terminator.targets if target not in blocks]
        if missing:
            raise QirParseError(f"QIR branch target {missing[0]!r} does not exist")
    return blocks


def _collect_block_lines(lines: list[str], limits: QirParseLimits, entry_label: str) -> dict[str, list[str]]:
    pending: dict[str, list[str]] = {entry_label: []}
    current = entry_label
    instruction_count = 0
    for raw in lines:
        line = _strip_comment(raw).strip()
        if not line:
            continue
        label_match = _LABEL.match(line)
        if label_match is not None:
            current = label_match.group(1)
            if current in pending:
                raise QirParseError(f"duplicate QIR block label {current!r}")
            pending[current] = []
            if len(pending) > limits.max_blocks:
                raise QirParseError("adaptive QIR exceeds the block limit")
            continue
        pending[current].append(line)
        instruction_count += 1
        if instruction_count > limits.max_instructions:
            raise QirParseError("adaptive QIR exceeds the instruction limit")
    for label, instructions in pending.items():
        if not instructions:
            raise QirParseError(f"QIR block {label!r} is empty")
    return pending


def _parse_terminator(line: str) -> _Terminator:
    direct = _DIRECT_BRANCH.match(line)
    if direct is not None:
        return _Terminator(None, (direct.group(1),))
    conditional = _CONDITIONAL_BRANCH.match(line)
    if conditional is not None:
        raw = conditional.group(1)
        condition: str | bool = raw == "true" if raw in {"true", "false"} else raw
        return _Terminator(condition, (conditional.group(2), conditional.group(3)))
    if _RETURN.match(line):
        return _Terminator(None, ())
    raise QirParseError(f"unsupported QIR terminator: {line[:160]}")


def _walk_cfg(
    context: _ParseContext,
    label: str,
    *,
    stop: str | None,
    conditions: dict[int, int],
    stack: tuple[str, ...],
    depth: int,
) -> _State:
    if depth > context.limits.max_branch_depth:
        raise QirParseError("adaptive QIR exceeds the branch-depth limit")
    current = label
    path = list(stack)
    predecessor = path[-1] if path else None
    while current != stop:
        block = context.blocks[current]
        _process_block(context, block, predecessor, conditions)
        decision = _terminator_decision(block.terminator, context.state)
        if decision is None:
            return context.state
        if isinstance(decision, _BitPredicate):
            return _walk_symbolic_branch(context, block, decision, stop, conditions, tuple(path), depth)
        path.append(current)
        predecessor, current = current, decision
    return context.state


def _process_block(
    context: _ParseContext,
    block: _Block,
    predecessor: str | None,
    conditions: dict[int, int],
) -> None:
    context.visited_steps += 1
    if context.visited_steps > context.limits.max_instructions:
        raise QirParseError("adaptive QIR CFG traversal exceeds the instruction limit")
    context.state.predecessor = predecessor
    for instruction in block.instructions:
        operation = _evaluate_instruction(instruction, context.state)
        if operation is not None:
            context.operations.append(_with_condition(operation, conditions))


def _terminator_decision(terminator: _Terminator, state: _State) -> str | _BitPredicate | None:
    if not terminator.targets:
        return None
    if terminator.condition is None:
        return terminator.targets[0]
    condition = _resolve_branch_condition(terminator.condition, state)
    if isinstance(condition, _BitPredicate):
        return condition
    return terminator.targets[0 if condition else 1]


def _walk_symbolic_branch(
    context: _ParseContext,
    block: _Block,
    condition: _BitPredicate,
    stop: str | None,
    conditions: dict[int, int],
    stack: tuple[str, ...],
    depth: int,
) -> _State:
    terminator = block.terminator
    merge = _nearest_common_successor(context.blocks, *terminator.targets)
    original_state = context.state
    branch_states: list[_State] = []
    for expected, target in ((1, terminator.targets[0]), (0, terminator.targets[1])):
        branch_conditions = _extend_conditions(conditions, condition, expected)
        if branch_conditions is None:
            continue
        context.state = original_state.clone()
        branch_states.append(
            _walk_cfg(
                context,
                target,
                stop=merge,
                conditions=branch_conditions,
                stack=(*stack, block.label),
                depth=depth + 1,
            )
        )
    context.state = _merge_states(original_state, branch_states)
    if merge is None:
        return context.state
    return _walk_cfg(
        context,
        merge,
        stop=stop,
        conditions=conditions,
        stack=(*stack, block.label),
        depth=depth + 1,
    )


def _resolve_branch_condition(value: str | bool, state: _State) -> bool | _BitPredicate:
    if isinstance(value, bool):
        return value
    resolved = state.values.get(value)
    if isinstance(resolved, bool | _BitPredicate):
        return resolved
    if isinstance(resolved, int):
        return bool(resolved)
    raise QirParseError(f"QIR branch condition {value!r} is not statically or semantically resolvable")


def _extend_conditions(
    conditions: dict[int, int],
    predicate: _BitPredicate,
    expected: int,
) -> dict[int, int] | None:
    value = expected ^ int(predicate.inverted)
    existing = conditions.get(predicate.bit)
    if existing is not None and existing != value:
        return None
    return {**conditions, predicate.bit: value}


def _with_condition(operation: Operation, conditions: dict[int, int]) -> Operation:
    if not conditions:
        return operation
    bits = tuple(sorted(conditions))
    value = sum(conditions[bit] << index for index, bit in enumerate(bits))
    condition = ClassicalCondition(bits, value)
    if operation.condition is not None and operation.condition != condition:
        raise QirParseError("QIR operation has overlapping incompatible classical conditions")
    return replace(operation, condition=condition)


def _nearest_common_successor(blocks: dict[str, _Block], left: str, right: str) -> str | None:
    left_distances = _reachable_distances(blocks, left)
    right_distances = _reachable_distances(blocks, right)
    common = set(left_distances) & set(right_distances)
    if not common:
        return None
    return min(
        common,
        key=lambda item: (
            max(left_distances[item], right_distances[item]),
            left_distances[item] + right_distances[item],
        ),
    )


def _reachable_distances(blocks: dict[str, _Block], start: str) -> dict[str, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        label = queue.popleft()
        for target in blocks[label].terminator.targets:
            if target not in distances:
                distances[target] = distances[label] + 1
                queue.append(target)
    return distances


def _merge_states(original: _State, branches: list[_State]) -> _State:
    if not branches:
        return original
    merged = original.clone()
    keys = set.intersection(*(set(branch.values) for branch in branches))
    for key in keys:
        values = [branch.values[key] for branch in branches]
        if all(_state_value_equal(values[0], value) for value in values[1:]):
            merged.values[key] = values[0]
    return merged


def _state_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (_Memory, _QubitArray)) and isinstance(right, type(left)):
        return left.values == right.values
    return left == right


def _inferred_qubits(operations: list[Operation]) -> int:
    wires = [
        wire
        for operation in operations
        for wire in (*operation.quantum_wires, *(control.wire for control in operation.controls))
    ]
    return max(wires, default=-1) + 1


def _inferred_clbits(operations: list[Operation]) -> int:
    bits = [
        bit
        for operation in operations
        for bit in (
            *operation.classical_bits,
            *((operation.condition.bits) if operation.condition is not None else ()),
        )
    ]
    return max(bits, default=-1) + 1
