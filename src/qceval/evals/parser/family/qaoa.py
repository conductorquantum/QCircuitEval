"""Prove structured QAOA MaxCut ansatz families from source AST."""

from __future__ import annotations

import ast
from typing import Any

from qceval.evals.parser.family.ast_utils import (
    _assignments,
    _call_name,
    _depends_on,
    _is_none_default,
    _resolve,
    _target_names,
    _unpacked_parameter_indices,
    _wire_index,
)
from qceval.semantics.verifiers.result import SemanticStatus

_QAOA_EDGES = frozenset({(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)})


def _prove_qaoa(function: ast.FunctionDef) -> tuple[SemanticStatus, str]:
    """Prove or refute the structured five-layer MaxCut QAOA family.

    Args:
        function: Candidate entry-point function AST.

    Returns:
        Status and machine-readable reason code.
    """
    if any(isinstance(node, (ast.While, ast.Try, ast.With, ast.Match, ast.Lambda)) for node in ast.walk(function)):
        return SemanticStatus.EXECUTION_ERROR, "family_control_flow_unsupported"
    if not any(isinstance(node, ast.For) for node in ast.walk(function)):
        return _prove_unrolled_qaoa(function)
    if any(
        isinstance(node, ast.If) and not (_is_none_default(node) or _is_graph_default(node))
        for node in ast.walk(function)
    ):
        return SemanticStatus.EXECUTION_ERROR, "family_parameter_branch_unsupported"
    assignments = _assignments(function)
    calls = sorted(
        (node for node in ast.walk(function) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    names = tuple(name for call in calls if (name := _qaoa_gate_name(call)) is not None)
    if names not in {
        ("h", "cx", "rz", "cx", "rx"),
        ("h", "rzz", "rx"),
    }:
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_gate_topology_mismatch"
    rz_call = next(call for call in calls if _qaoa_gate_name(call) in {"rz", "rzz"})
    rx_call = next(call for call in calls if _qaoa_gate_name(call) == "rx")
    rz_angle = _qaoa_angle_argument(rz_call)
    rx_angle = _qaoa_angle_argument(rx_call)
    if rz_angle is None or not _is_twice_family_reference(rz_angle, "gamma", assignments):
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_gamma_binding_mismatch"
    if rx_angle is None or not _is_twice_family_reference(rx_angle, "beta", assignments):
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_beta_binding_mismatch"
    return _qaoa_loop_domains(function, assignments)


def _qaoa_loop_domains(
    function: ast.FunctionDef,
    assignments: dict[str, ast.expr],
) -> tuple[SemanticStatus, str]:
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    if not any(
        _depends_on(node.iter, "beta", assignments) or _is_exact_range(node.iter, 5, assignments) for node in loops
    ):
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_layer_domain_mismatch"
    if not any(_depends_on(node.iter, "edges", assignments) for node in loops):
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_edge_domain_mismatch"
    if not any(_covers_qaoa_wires(node, assignments) for node in loops):
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_wire_domain_mismatch"
    return SemanticStatus.VERIFIED_PASS, "structured_qaoa_family_identity"


def _prove_unrolled_qaoa(function: ast.FunctionDef) -> tuple[SemanticStatus, str]:
    """Prove the fixed five-layer MaxCut ansatz from a fully unrolled body.

    CUDA-Q kernels cannot iterate captured edge lists, so faithful candidates
    unroll the layers; the unrolled sequence is checked gate by gate against
    the declared graph, layer count, and 2*gamma / 2*beta bindings.
    """
    unpacked = _unpacked_parameter_indices(function)
    sequence = _unrolled_qaoa_sequence(function, unpacked)
    if sequence is None:
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_gate_topology_mismatch"
    position = 0
    if [item[0] for item in sequence[:5]] != ["h"] * 5 or {item[1] for item in sequence[:5]} != {0, 1, 2, 3, 4}:
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_wire_domain_mismatch"
    position = 5
    for layer in range(5):
        edges = set()
        for _ in range(6):
            triple = sequence[position : position + 3]
            if (
                len(triple) != 3
                or triple[0][0] != "cx"
                or triple[2] != triple[0]
                or triple[1][0] != "rz"
                or triple[1][1] != triple[0][1][1]
                or triple[1][2] != ("gamma", layer)
            ):
                return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_gamma_binding_mismatch"
            edges.add(triple[0][1])
            position += 3
        if edges != set(_QAOA_EDGES):
            return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_edge_domain_mismatch"
        mixer = sequence[position : position + 5]
        if (
            len(mixer) != 5
            or any(item[0] != "rx" or item[2] != ("beta", layer) for item in mixer)
            or {item[1] for item in mixer} != {0, 1, 2, 3, 4}
        ):
            return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_beta_binding_mismatch"
        position += 5
    if position != len(sequence):
        return SemanticStatus.SEMANTIC_FAIL, "structured_qaoa_gate_topology_mismatch"
    return SemanticStatus.VERIFIED_PASS, "structured_qaoa_family_identity"


def _unrolled_qaoa_sequence(
    function: ast.FunctionDef,
    unpacked: dict[str, tuple[str, int]],
) -> list[tuple[str, Any, tuple[str, int] | None]] | None:
    sequence: list[tuple[str, Any, tuple[str, int] | None]] = []
    for call in sorted(
        (node for node in ast.walk(function) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    ):
        raw = _call_name(call.func)
        if raw == "ctrl" and isinstance(call.func, ast.Attribute) and _call_name(call.func.value) == "x":
            wires = tuple(_wire_index(node, {}) for node in call.args[:2])
            if None in wires or len(wires) != 2:
                return None
            sequence.append(("cx", (wires[0], wires[1]), None))
            continue
        if raw == "h" and call.args:
            wire = _wire_index(call.args[0], {})
            if wire is None:
                return None
            sequence.append(("h", wire, None))
            continue
        if raw in {"rz", "rx"} and len(call.args) >= 2:
            wire = _wire_index(call.args[1], {})
            binding = _doubled_family_binding(call.args[0], unpacked)
            if wire is None or binding is None:
                return None
            sequence.append((raw, wire, binding))
            continue
        if raw in {"qvector", "kernel", "float"} or raw == function.name:
            continue
        return None
    return sequence


def _doubled_family_binding(
    node: ast.expr,
    unpacked: dict[str, tuple[str, int]],
) -> tuple[str, int] | None:
    """Resolve ``2 * gamma[k]``-style mixer/cost angle bindings."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return None
    for scalar, expression in ((node.left, node.right), (node.right, node.left)):
        if isinstance(scalar, ast.Constant) and scalar.value in {2, 2.0}:
            if isinstance(expression, ast.Name) and expression.id in unpacked:
                return unpacked[expression.id]
            if (
                isinstance(expression, ast.Subscript)
                and isinstance(expression.value, ast.Name)
                and isinstance(expression.slice, ast.Constant)
                and isinstance(expression.slice.value, int)
            ):
                return expression.value.id, expression.slice.value
    return None


def _is_graph_default(node: ast.If) -> bool:
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "G"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    ):
        return False
    rendered = (
        "\n".join(ast.unparse(statement) for statement in node.body),
        "\n".join(ast.unparse(statement) for statement in node.orelse),
    )
    required_edges = {"[0, 3]", "[0, 4]", "[1, 3]", "[1, 4]", "[2, 3]", "[2, 4]"}
    return "G.edges" in rendered[0] and all(edge in rendered[1] for edge in required_edges)


def _qaoa_gate_name(call: ast.Call) -> str | None:
    raw = _call_name(call.func)
    aliases = {
        "H": "h",
        "Hadamard": "h",
        "CNOT": "cx",
        "RZ": "rz",
        "RX": "rx",
        "RY": "ry",
        "Rz": "rz",
        "Rx": "rx",
        "Ry": "ry",
        "IsingZZ": "rzz",
    }
    if raw == "on_each" and isinstance(call.func, ast.Attribute):
        value = call.func.value
        if isinstance(value, ast.Attribute) and value.attr == "H":
            return "h"
    if raw in {"h", "cx", "rz", "rx", "rzz"}:
        return raw
    return aliases.get(raw)


def _qaoa_angle_argument(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    return next((keyword.value for keyword in call.keywords if keyword.arg == "rads"), None)


def _is_exact_range(node: ast.expr, count: int, assignments: dict[str, ast.expr]) -> bool:
    node = _resolve(node, assignments)
    if not isinstance(node, ast.Call) or _call_name(node.func) != "range" or len(node.args) != 1:
        return False
    argument = _resolve(node.args[0], assignments)
    return isinstance(argument, ast.Constant) and isinstance(argument.value, int) and argument.value == count


def _covers_qaoa_wires(node: ast.For, assignments: dict[str, ast.expr]) -> bool:
    target_names = _target_names(node.target)
    if not target_names & {"i", "q", "qubit", "wire"}:
        return False
    if _is_exact_range(node.iter, 5, assignments) or _depends_on(node.iter, "G", assignments):
        return True
    resolved = _resolve(node.iter, assignments)
    if isinstance(resolved, ast.ListComp) and any(
        _is_exact_range(generator.iter, 5, assignments) for generator in resolved.generators
    ):
        return True
    if isinstance(resolved, (ast.List, ast.Tuple)) and len(resolved.elts) == 5:
        return True
    rendered = ast.unparse(resolved).lower()
    return any(name in rendered for name in ("qubits", "nodes", "wires")) and (
        "range(5)" in rendered or "range(n" in rendered or "[0, 1, 2, 3, 4]" in rendered
    )


def _is_twice_family_reference(node: ast.expr, family: str, assignments: dict[str, ast.expr]) -> bool:
    node = _resolve(node, assignments)
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return False
    sides = ((node.left, node.right), (node.right, node.left))
    for scalar, expression in sides:
        if isinstance(scalar, ast.Constant) and scalar.value == 2:
            expression = _resolve(expression, assignments)
            return family in ast.unparse(expression)
    return False
