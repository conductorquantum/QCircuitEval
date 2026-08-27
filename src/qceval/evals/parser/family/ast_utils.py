"""Shared AST helpers for structured family source proofs."""

from __future__ import annotations

import ast
from dataclasses import dataclass

# Contract signature names of family parameter vectors ("parameters" for
# tasks 39/40, "param" for task 41).
_PARAMETER_VECTOR_NAMES = frozenset({"parameters", "params", "param"})


@dataclass(frozen=True)
class _Gate:
    name: str
    parameters: tuple[int | str, ...]
    wires: tuple[int, ...]


def _parameter_vector_bindings(
    function: ast.FunctionDef,
    parameter_vector_names: frozenset[str] = _PARAMETER_VECTOR_NAMES,
    permitted_bindings: frozenset[ast.AST] = frozenset(),
) -> set[ast.AST]:
    """Return nodes that rebind, shadow, mutate, or delete a parameter vector.

    A structured-family proof reads ``parameters[i]``/``param[i]`` as the
    contract's universally quantified parameter vector. Any assignment to that
    name (including tuple/loop/with/walrus targets), element or attribute
    assignment, deletion, ``global``/``nonlocal`` declaration, import alias,
    exception alias, nested definition or argument shadowing, or attribute
    access (which may mutate the vector, e.g. ``param.insert``) breaks that
    reading, so callers must fail closed instead of proving the family.

    Args:
        function: Candidate entry-point function AST.

    The one exemption is the canonical dead-branch default
    ``if <vector> is None: <vector> = <default>``: the grader always binds a
    real parameter vector, so that assignment can never execute on a graded
    call and does not disturb the universal claim.

    Returns:
        Offending AST nodes; empty when the vector is only ever read.
    """
    own_arguments = set(ast.walk(function.args))
    exempt = _none_default_rebindings(function, parameter_vector_names) | set(permitted_bindings)
    return {
        node
        for node in ast.walk(function)
        if node not in exempt and _binds_parameter_vector(node, function, own_arguments, parameter_vector_names)
    }


def _binds_parameter_vector(
    node: ast.AST,
    function: ast.FunctionDef,
    own_arguments: set[ast.AST],
    parameter_vector_names: frozenset[str],
) -> bool:
    """Return whether one node rebinds, shadows, mutates, or deletes a vector."""
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Store | ast.Del) and node.id in parameter_vector_names
    if isinstance(node, ast.Subscript):
        return (
            isinstance(node.ctx, ast.Store | ast.Del)
            and isinstance(node.value, ast.Name)
            and node.value.id in parameter_vector_names
        )
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id in parameter_vector_names
    if isinstance(node, ast.arg):
        return node not in own_arguments and node.arg in parameter_vector_names
    if isinstance(node, ast.Global | ast.Nonlocal):
        return bool(set(node.names) & parameter_vector_names)
    if isinstance(node, ast.alias):
        return (node.asname or node.name) in parameter_vector_names
    if isinstance(node, ast.ExceptHandler):
        return node.name in parameter_vector_names
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node is not function and node.name in parameter_vector_names
    return False


def _none_default_rebindings(
    function: ast.FunctionDef,
    parameter_vector_names: frozenset[str] = _PARAMETER_VECTOR_NAMES,
) -> set[ast.AST]:
    """Return vector rebinding targets inside ``if <vector> is None`` bodies.

    Only a plain ``<vector> = <default>`` in the body of an ``If`` whose test
    is ``<that same vector> is None`` qualifies. Graded calls always supply a
    real vector, so the branch is dead and the assignment cannot shadow the
    contract's universally quantified parameters. Assignments in the
    ``orelse`` (taken on graded calls) are never exempt.
    """
    exempt: set[ast.AST] = set()
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.If)
            and _is_none_default(node)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id in parameter_vector_names
        ):
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == node.test.left.id
            ):
                exempt.add(statement.targets[0])
    return exempt


def _parameter_binding_nodes(function: ast.FunctionDef) -> set[ast.AST]:
    """Return AST nodes that belong to pure parameter-unpack bindings.

    ``p0, ..., pN = [float(value) for value in parameters]`` and
    ``angles = [float(value) for value in parameters]`` are plain renamings of
    the family parameters, not candidate control flow.
    """
    nodes: set[ast.AST] = set()
    for statement in ast.walk(function):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not (
            isinstance(target, ast.Name)
            or isinstance(target, ast.Tuple)
            and all(isinstance(item, ast.Name) for item in target.elts)
        ):
            continue
        value = statement.value
        if (
            isinstance(value, ast.Name)
            or _parameter_list_comprehension_source(value, _PARAMETER_VECTOR_NAMES) is not None
        ):
            nodes.update(ast.walk(value))
    return nodes


def _unpacked_parameter_indices(function: ast.FunctionDef) -> dict[str, tuple[str, int]]:
    """Map unpacked binding names to their (family, index) source."""
    indices: dict[str, tuple[str, int]] = {}
    for statement in ast.walk(function):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Tuple)
            and all(isinstance(item, ast.Name) for item in statement.targets[0].elts)
            and (
                isinstance(statement.value, ast.Name)
                or (
                    isinstance(statement.value, ast.ListComp)
                    and len(statement.value.generators) == 1
                    and isinstance(statement.value.generators[0].iter, ast.Name)
                )
            )
        ):
            iterated = statement.value if isinstance(statement.value, ast.Name) else statement.value.generators[0].iter
            assert isinstance(iterated, ast.Name)
            for position, item in enumerate(statement.targets[0].elts):
                if isinstance(item, ast.Name):
                    indices[item.id] = (iterated.id, position)
    return indices


def _decode_gate(
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    assignments: dict[str, ast.expr],
    unpacked: dict[str, tuple[str, int]] | None = None,
    parameter_vector_names: frozenset[str] = _PARAMETER_VECTOR_NAMES,
) -> _Gate | None:
    raw = _call_name(call.func)
    if raw == "ctrl" and isinstance(call.func, ast.Attribute) and _call_name(call.func.value) == "x":
        # CUDA-Q spells CNOT as ``x.ctrl(control, target)``.
        return _resolved_gate("cx", [], list(call.args[:2]), assignments, unpacked, parameter_vector_names)
    aliases = {
        "RX": "rx",
        "RY": "ry",
        "RZ": "rz",
        "PauliX": "x",
        "X": "x",
        "CNOT": "cx",
        "SWAP": "swap",
        "U3": "u3",
        "U3Gate": "u3",
    }
    name = aliases.get(raw, raw.lower() if raw in {"rx", "ry", "rz", "cx", "x"} else None)
    if name is None:
        return None
    decoded = _gate_nodes(call, raw, name, parents)
    if decoded is None:
        return None
    name, parameter_nodes, wire_nodes = decoded
    return _resolved_gate(name, parameter_nodes, wire_nodes, assignments, unpacked, parameter_vector_names)


def _gate_nodes(
    call: ast.Call,
    raw: str,
    name: str,
    parents: dict[ast.AST, ast.AST],
) -> tuple[str, list[ast.expr], list[ast.expr]] | None:
    wire_nodes: list[ast.expr] = []
    parameter_nodes: list[ast.expr] = []
    if raw in {"rx", "ry", "rz"} and _is_cirq_factory(call):
        outer = _cirq_application_call(call, parents)
        if outer is None:
            return None
        parameter_nodes = list(call.args[:1])
        wire_nodes = list(outer.args)
    elif name in {"rx", "ry", "rz"}:
        parameter_nodes = list(call.args[:1])
        wire_nodes = list(call.args[1:2]) or _keyword_wires(call)
    elif name in {"cx", "swap"}:
        wire_nodes = list(call.args[:2]) or _keyword_wires(call)
    elif name == "x":
        wire_nodes = list(call.args[:1]) or _keyword_wires(call)
    else:
        if not _u3_is_x(call):
            return "u3", [], []
        wire_nodes = _keyword_wires(call) or _append_wires(call, parents)
        name = "x"
    return name, parameter_nodes, wire_nodes


def _resolved_gate(
    name: str,
    parameter_nodes: list[ast.expr],
    wire_nodes: list[ast.expr],
    assignments: dict[str, ast.expr],
    unpacked: dict[str, tuple[str, int]] | None = None,
    parameter_vector_names: frozenset[str] = _PARAMETER_VECTOR_NAMES,
) -> _Gate:
    wires = tuple(_wire_index(node, assignments) for node in wire_nodes)
    if any(value is None for value in wires):
        return _Gate("unresolved", (), ())
    resolved_wires = tuple(value for value in wires if value is not None)
    parameters = tuple(
        _parameter_index(node, assignments, unpacked, parameter_vector_names) for node in parameter_nodes
    )
    if any(value is None for value in parameters):
        return _Gate("unresolved", (), resolved_wires)
    return _Gate(name, tuple(value for value in parameters if value is not None), resolved_wires)


def _parents(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)}


def _assignments(root: ast.AST) -> dict[str, ast.expr]:
    values: dict[str, ast.expr] = {}
    for node in ast.walk(root):
        assignment = _assignment_parts(node)
        if assignment is None:
            continue
        targets, value = assignment
        for target in targets:
            _record_assignment(values, target, value)
    return values


def _assignment_parts(node: ast.AST) -> tuple[list[ast.expr], ast.expr] | None:
    if isinstance(node, ast.Assign):
        return list(node.targets), node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return [node.target], node.value
    return None


def _record_assignment(values: dict[str, ast.expr], target: ast.expr, value: ast.expr) -> None:
    if isinstance(target, ast.Name):
        values[target.id] = value
        return
    if not isinstance(target, ast.Tuple):
        return
    indices = _line_qubit_range_indices(value, len(target.elts))
    if indices is None:
        return
    for item, index in zip(target.elts, indices, strict=True):
        if isinstance(item, ast.Name):
            values[item.id] = ast.Constant(index)


def _line_qubit_range_indices(value: ast.expr, width: int) -> tuple[int, ...] | None:
    """Resolve ``q0, ..., qN = cirq.LineQubit.range(N + 1)`` aliases."""
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "range"
        and _call_name(value.func.value) == "LineQubit"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Constant)
        and value.args[0].value == width
    ):
        return None
    return tuple(range(width))


def _resolve(node: ast.expr, assignments: dict[str, ast.expr]) -> ast.expr:
    seen: set[str] = set()
    while isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        seen.add(node.id)
        node = assignments[node.id]
    return node


def _depends_on(
    node: ast.AST,
    name: str,
    assignments: dict[str, ast.expr],
    seen: set[str] | None = None,
) -> bool:
    seen = set() if seen is None else seen
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == name:
            return True
        if not isinstance(child, ast.Name):
            continue
        if child.id == name:
            return True
        if child.id in assignments and child.id not in seen:
            seen.add(child.id)
            if _depends_on(assignments[child.id], name, assignments, seen):
                return True
    return False


def _target_names(node: ast.expr) -> set[str]:
    return {value.id for value in ast.walk(node) if isinstance(value, ast.Name)}


def _is_none_default(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _is_cirq_factory(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "cirq"
    )


def _cirq_application_call(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> ast.Call | None:
    direct = parents.get(call)
    if isinstance(direct, ast.Call) and direct.func is call:
        return direct
    attribute = parents.get(call)
    outer = (
        parents.get(attribute) if isinstance(attribute, ast.Attribute) and attribute.attr in {"on", "on_each"} else None
    )
    return outer if isinstance(outer, ast.Call) else None


def _keyword_wires(call: ast.Call) -> list[ast.expr]:
    value = next((item.value for item in call.keywords if item.arg in {"wire", "wires"}), None)
    if value is None:
        return []
    return list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]


def _append_wires(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> list[ast.expr]:
    node: ast.AST = call
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.Call) and _call_name(node.func) == "append" and len(node.args) >= 2:
            value = node.args[1]
            return list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return []


def _u3_is_x(call: ast.Call) -> bool:
    if len(call.args) < 3:
        return False
    return _is_pi(call.args[0]) and all(_is_zero(value) for value in call.args[1:2]) and _is_pi(call.args[2])


def _is_pi(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "pi" or isinstance(node, ast.Name) and node.id == "pi"


def _is_zero(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0


def _parameter_index(
    node: ast.expr,
    assignments: dict[str, ast.expr] | None = None,
    unpacked: dict[str, tuple[str, int]] | None = None,
    parameter_vector_names: frozenset[str] = _PARAMETER_VECTOR_NAMES,
) -> int | str | None:
    if isinstance(node, ast.Name):
        return _named_parameter_index(node.id, assignments, unpacked, parameter_vector_names)
    if isinstance(node, ast.Call) and _call_name(node.func) == "float" and len(node.args) == 1:
        return _parameter_index(node.args[0], assignments, unpacked, parameter_vector_names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _summed_parameter_index(node, assignments, unpacked, parameter_vector_names)
    return _subscript_parameter_index(node, assignments, parameter_vector_names)


def _named_parameter_index(
    name: str,
    assignments: dict[str, ast.expr] | None,
    unpacked: dict[str, tuple[str, int]] | None,
    parameter_vector_names: frozenset[str],
) -> int | str | None:
    if unpacked and name in unpacked:
        family, index = unpacked[name]
        return index if family in parameter_vector_names else None
    if assignments and name in assignments:
        return _parameter_index(assignments[name], None, None, parameter_vector_names)
    return None


def _summed_parameter_index(
    node: ast.BinOp,
    assignments: dict[str, ast.expr] | None,
    unpacked: dict[str, tuple[str, int]] | None,
    parameter_vector_names: frozenset[str],
) -> str | None:
    left = _parameter_index(node.left, assignments, unpacked, parameter_vector_names)
    right = _parameter_index(node.right, assignments, unpacked, parameter_vector_names)
    return f"{left}+{right}" if isinstance(left, int) and isinstance(right, int) else None


def _subscript_parameter_index(
    node: ast.expr,
    assignments: dict[str, ast.expr] | None,
    parameter_vector_names: frozenset[str],
) -> int | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    family = node.value.id
    if family not in parameter_vector_names and assignments and family in assignments:
        alias = _parameter_list_comprehension_source(assignments[family], parameter_vector_names)
        if alias is not None:
            family = alias
    if family not in parameter_vector_names:
        return None
    return node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int) else None


def _parameter_list_comprehension_source(
    node: ast.expr | None,
    parameter_vector_names: frozenset[str],
) -> str | None:
    """Return the vector copied by a pure identity/``float`` list comprehension."""
    if not isinstance(node, ast.ListComp) or len(node.generators) != 1:
        return None
    generator = node.generators[0]
    if (
        generator.ifs
        or generator.is_async
        or not isinstance(generator.target, ast.Name)
        or not isinstance(generator.iter, ast.Name)
        or generator.iter.id not in parameter_vector_names
    ):
        return None
    element = node.elt
    if isinstance(element, ast.Call) and _call_name(element.func) == "float" and len(element.args) == 1:
        element = element.args[0]
    if not isinstance(element, ast.Name) or element.id != generator.target.id:
        return None
    return generator.iter.id


def _wire_index(node: ast.expr, assignments: dict[str, ast.expr]) -> int | None:
    node = _resolve(node, assignments)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
        return node.slice.value
    if (
        isinstance(node, ast.Call)
        and _call_name(node.func) == "LineQubit"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, int)
    ):
        return node.args[0].value
    return None


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
