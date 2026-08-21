"""Prove structured rotation-parameter circuit families from source AST."""

from __future__ import annotations

import ast

from qceval.evals.parser.family.ast_utils import (
    _assignments,
    _call_name,
    _decode_gate,
    _Gate,
    _is_none_default,
    _parameter_binding_nodes,
    _parameter_list_comprehension_source,
    _parameter_vector_bindings,
    _parents,
    _unpacked_parameter_indices,
    _wire_index,
)
from qceval.semantics.verifiers.result import SemanticStatus


def _prove_rotation_family(function: ast.FunctionDef, task_id: str) -> tuple[SemanticStatus, str]:
    """Prove or refute a rotation-parameter family for one task id.

    Args:
        function: Candidate entry-point function AST.
        task_id: Contract task id selecting the expected gate sequence.

    Returns:
        Status and machine-readable reason code.
    """
    parameter_vector_names, permitted_bindings = _rotation_parameter_vector_context(function)
    rebindings = _parameter_vector_bindings(function, parameter_vector_names, permitted_bindings)
    if task_id == "39" and _is_enumerated_rx_ry_family(function):
        # The accepted enumerated spelling binds the loop variable ``param``
        # from ``enumerate(parameters)``; that loop target is the only
        # permitted binding of a parameter-vector name.
        loop = next(node for node in ast.walk(function) if isinstance(node, ast.For))
        if rebindings <= set(ast.walk(loop.target)):
            return SemanticStatus.VERIFIED_PASS, "structured_rotation_family_identity"
        return SemanticStatus.EXECUTION_ERROR, "family_parameter_rebinding_unsupported"
    if rebindings:
        # Rebinding, shadowing, mutating, or deleting the parameter vector
        # breaks the identification of ``parameters[i]`` with the contract's
        # universally quantified parameters; fail closed instead of proving.
        return SemanticStatus.EXECUTION_ERROR, "family_parameter_rebinding_unsupported"
    forbidden = (ast.For, ast.While, ast.Try, ast.With, ast.Match, ast.Lambda, ast.comprehension)
    binding_nodes = _parameter_binding_nodes(function)
    if any(isinstance(node, forbidden) for node in ast.walk(function) if node not in binding_nodes):
        return SemanticStatus.EXECUTION_ERROR, "family_control_flow_unsupported"
    if any(
        isinstance(node, ast.If) and not _is_none_default(node) and not _is_parameter_length_guard(node, task_id)
        for node in ast.walk(function)
    ):
        return SemanticStatus.EXECUTION_ERROR, "family_parameter_branch_unsupported"
    parents = _parents(function)
    assignments = _assignments(function)
    unpacked = _unpacked_parameter_indices(function)
    gates = tuple(
        gate
        for call in sorted(
            (node for node in ast.walk(function) if isinstance(node, ast.Call)),
            key=lambda node: (node.lineno, node.col_offset),
        )
        if (
            gate := _decode_gate(
                call,
                parents,
                assignments,
                unpacked,
                parameter_vector_names,
            )
        )
        is not None
    )
    expected = {
        "39": (
            (
                _Gate("rx", (0,), (0,)),
                _Gate("ry", (1,), (0,)),
            ),
        ),
        "40": (
            (
                _Gate("x", (), (0,)),
                _Gate("rz", (0,), (0,)),
                _Gate("rz", (1,), (0,)),
                _Gate("ry", (2,), (0,)),
                _Gate("rz", (3,), (1,)),
                _Gate("cx", (), (0, 1)),
                _Gate("rz", (4,), (0,)),
                _Gate("rz", (5,), (1,)),
                _Gate("ry", (6,), (0,)),
                _Gate("rz", (7,), (1,)),
            ),
            # Adjacent rotations about the same axis add exactly.
            (
                _Gate("x", (), (0,)),
                _Gate("rz", ("0+1",), (0,)),
                _Gate("ry", (2,), (0,)),
                _Gate("rz", (3,), (1,)),
                _Gate("cx", (), (0, 1)),
                _Gate("rz", (4,), (0,)),
                _Gate("rz", (5,), (1,)),
                _Gate("ry", (6,), (0,)),
                _Gate("rz", (7,), (1,)),
            ),
            # On the reachable span {|00>, |11>}, RZ(p5) on q0 and q1
            # are identical. Accept the prompt-literal placement as the same
            # parameterized state family instead of rejecting by syntax.
            (
                _Gate("x", (), (0,)),
                _Gate("rz", (0,), (0,)),
                _Gate("rz", (1,), (0,)),
                _Gate("ry", (2,), (0,)),
                _Gate("rz", (3,), (1,)),
                _Gate("cx", (), (0, 1)),
                _Gate("rz", (4,), (0,)),
                _Gate("rz", (5,), (0,)),
                _Gate("ry", (6,), (0,)),
                _Gate("rz", (7,), (1,)),
            ),
        ),
        # Task 41 applies the Euler triple RZ(p0), RY(p1), RZ(p2) on qubit 0
        # and RZ(p3), RY(p4), RZ(p5) on qubit 1. The two single-qubit strands
        # act on disjoint wires, so every source ordering that preserves each
        # per-qubit gate order prepares the identical parameterized state for
        # all real parameters. Accept exactly those interleavings.
        "41": _ordered_interleavings(
            (
                _Gate("rz", (0,), (0,)),
                _Gate("ry", (1,), (0,)),
                _Gate("rz", (2,), (0,)),
            ),
            (
                _Gate("rz", (3,), (1,)),
                _Gate("ry", (4,), (1,)),
                _Gate("rz", (5,), (1,)),
            ),
        ),
    }.get(task_id)
    if expected is None:
        return SemanticStatus.EXECUTION_ERROR, "family_target_unsupported"
    if gates in expected:
        return SemanticStatus.VERIFIED_PASS, "structured_rotation_family_identity"
    return SemanticStatus.SEMANTIC_FAIL, "structured_rotation_family_mismatch"


def _rotation_parameter_vector_context(
    function: ast.FunctionDef,
) -> tuple[frozenset[str], frozenset[ast.AST]]:
    """Return proven vector aliases and their permitted declaration nodes."""
    names = {"parameters", "params", "param"}
    permitted: set[ast.AST] = set()
    outer_vectors = [arg.arg for arg in function.args.args if arg.arg in names]
    if len(outer_vectors) != 1:
        return frozenset(names), frozenset()
    for node in ast.walk(function):
        if node is function or not isinstance(node, ast.FunctionDef):
            continue
        if len(node.args.args) == 1 and any(
            isinstance(decorator, ast.Attribute | ast.Name) and _call_name(decorator) == "kernel"
            for decorator in node.decorator_list
        ):
            names.add(node.args.args[0].arg)
            permitted.add(node.args.args[0])
    vector_names = frozenset(names)
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _parameter_list_comprehension_source(node.value, vector_names) is not None
        ):
            continue
        names.add(node.targets[0].id)
        permitted.add(node.targets[0])
    return frozenset(names), frozenset(permitted)


def _is_parameter_length_guard(node: ast.If, task_id: str) -> bool:
    """Accept a raise-only guard for the contract's fixed vector length."""
    expected = {"39": 2, "40": 8, "41": 6}.get(task_id)
    test = node.test
    return bool(
        expected is not None
        and not node.orelse
        and node.body
        and all(isinstance(statement, ast.Raise) for statement in node.body)
        and isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.NotEq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == expected
        and isinstance(test.left, ast.Call)
        and _call_name(test.left.func) == "len"
        and len(test.left.args) == 1
        and isinstance(test.left.args[0], ast.Name)
        and test.left.args[0].id in {"parameters", "params", "param"}
    )


def _ordered_interleavings(
    first: tuple[_Gate, ...],
    second: tuple[_Gate, ...],
) -> tuple[tuple[_Gate, ...], ...]:
    """Enumerate merges of two gate strands preserving each strand's order.

    Sound only when the strands act on disjoint wires, where all such merges
    are the same circuit family.
    """
    if not first:
        return (second,)
    if not second:
        return (first,)
    return tuple((first[0], *rest) for rest in _ordered_interleavings(first[1:], second)) + tuple(
        (second[0], *rest) for rest in _ordered_interleavings(first, second[1:])
    )


def _is_enumerated_rx_ry_family(function: ast.FunctionDef) -> bool:
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    if len(loops) != 1:
        return False
    loop = loops[0]
    if (
        not isinstance(loop.target, ast.Tuple)
        or [item.id for item in loop.target.elts if isinstance(item, ast.Name)] != ["i", "param"]
        or not isinstance(loop.iter, ast.Call)
        or _call_name(loop.iter.func) != "enumerate"
        or len(loop.iter.args) != 1
        or not isinstance(loop.iter.args[0], ast.Name)
        or loop.iter.args[0].id != "parameters"
        or len(loop.body) != 1
        or not isinstance(loop.body[0], ast.If)
    ):
        return False
    branch = loop.body[0]
    expected_test = "i % 2 == 0"
    if ast.unparse(branch.test).replace("(", "").replace(")", "") != expected_test:
        return False
    calls = [
        node
        for statements in (branch.body, branch.orelse)
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    ]
    if len(calls) != 2 or [_call_name(call.func).lower() for call in calls] != ["rx", "ry"]:
        return False
    return all(
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "param"
        and len(call.args) >= 2
        and _wire_index(call.args[1], {}) == 0
        for call in calls
    )
