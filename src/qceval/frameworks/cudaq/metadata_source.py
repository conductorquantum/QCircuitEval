"""AST-based CUDA-Q allocation and qubit-reference analysis."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from qceval.frameworks.cudaq.metadata_patterns import (
    _CTRL_PAIR_FAMILIES,
    _CTRL_ROTATION_FAMILIES,
    _INT_OPERATORS,
    _PARAMETRIC_GATES,
    _add_family,
    _all_pairs,
)


@dataclass
class _SourceFacts:
    """Facts statically resolved from CUDA-Q candidate source."""

    vector_sizes: dict[str, int] = field(default_factory=dict)
    vector_offsets: dict[str, int] = field(default_factory=dict)
    qubit_indices: dict[str, int] = field(default_factory=dict)
    aliases: dict[str, int] = field(default_factory=dict)
    measurement_indices: list[int] = field(default_factory=list)
    interaction_pairs: list[list[int]] = field(default_factory=list)
    gate_family_counts: dict[str, int] = field(default_factory=dict)
    total_allocated: int | None = None


def _source_facts(code: str) -> _SourceFacts:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _SourceFacts()
    visitor = _CudaqSourceVisitor()
    visitor.visit(tree)
    return visitor.facts


class _CudaqSourceVisitor(ast.NodeVisitor):
    """Collect statically resolvable CUDA-Q source facts."""

    def __init__(self) -> None:
        self.facts = _SourceFacts(total_allocated=0)
        self._scalars: dict[str, int] = {}
        self._next_qubit_index: int | None = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        positional = list(node.args.posonlyargs) + list(node.args.args)
        defaults = list(node.args.defaults)
        for arg, default in zip(
            positional[len(positional) - len(defaults) :],
            defaults,
            strict=False,
        ):
            value = _eval_int(default, self._scalars)
            if value is not None:
                self._scalars.setdefault(arg.arg, value)
        for arg, kw_default in zip(
            node.args.kwonlyargs,
            node.args.kw_defaults,
            strict=False,
        ):
            if kw_default is not None:
                value = _eval_int(kw_default, self._scalars)
                if value is not None:
                    self._scalars.setdefault(arg.arg, value)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        target = _single_name_target(node)
        if target is not None:
            self._record_assignment(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._record_assignment(node.target.id, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        family = _call_family(node.func)
        if family in {"mz", "measure"}:
            for arg in node.args:
                self.facts.measurement_indices.extend(_qubit_indices(arg, self.facts, self._scalars))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ctrl":
            self._record_controlled_gate(node, node.func)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        """Visit literal ``range`` loops with the loop index bound."""
        if not (
            isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and _call_family(node.iter.func) == "range"
        ):
            self.generic_visit(node)
            return
        arguments = [_eval_int(argument, self._scalars) for argument in node.iter.args]
        if any(argument is None for argument in arguments):
            self.generic_visit(node)
            return
        values = range(*(int(argument) for argument in arguments if argument is not None))
        prior = self._scalars.get(node.target.id)
        had_prior = node.target.id in self._scalars
        for value in values:
            self._scalars[node.target.id] = value
            for statement in node.body:
                self.visit(statement)
        if had_prior:
            assert prior is not None
            self._scalars[node.target.id] = prior
        else:
            self._scalars.pop(node.target.id, None)
        for statement in node.orelse:
            self.visit(statement)

    def _record_controlled_gate(
        self,
        node: ast.Call,
        func: ast.Attribute,
    ) -> None:
        base = _call_family(func.value)
        args = list(node.args)
        if base in _PARAMETRIC_GATES and args:
            args = args[1:]
        if len(args) < 2:
            return
        *control_args, target_arg = args
        control_indices: list[int] = []
        for control in control_args:
            control_indices.extend(_control_indices(control, self.facts, self._scalars))
        target_indices = _qubit_indices(
            target_arg,
            self.facts,
            self._scalars,
        )
        if not control_indices or not target_indices:
            return
        self.facts.interaction_pairs.extend(_all_pairs([*control_indices, *target_indices]))
        family = _controlled_family(base, len(control_indices))
        if family is not None:
            _add_family(self.facts.gate_family_counts, family)

    def _record_assignment(self, target: str, value: ast.AST) -> None:
        scalar = _eval_int(value, self._scalars)
        if scalar is not None:
            self._scalars[target] = scalar
        call_name = _call_family(value.func) if isinstance(value, ast.Call) else None
        if isinstance(value, ast.Call) and call_name == "qvector" and value.args:
            size = _eval_int(value.args[0], self._scalars)
            if size is None:
                self._next_qubit_index = None
                self.facts.total_allocated = None
                return
            self.facts.vector_sizes[target] = size
            if self._next_qubit_index is not None:
                self.facts.vector_offsets[target] = self._next_qubit_index
                self._next_qubit_index += size
                self.facts.total_allocated = self._next_qubit_index
            return
        if isinstance(value, ast.Call) and call_name == "qubit":
            if self._next_qubit_index is not None:
                self.facts.qubit_indices[target] = self._next_qubit_index
                self._next_qubit_index += 1
                self.facts.total_allocated = self._next_qubit_index
            else:
                self.facts.total_allocated = None
            return
        sliced = _subscript_qubit_slice(
            value,
            self.facts,
            self._scalars,
        )
        if sliced is not None:
            offset, size = sliced
            self.facts.vector_offsets[target] = offset
            self.facts.vector_sizes[target] = size
            return
        alias_index = _subscript_qubit_index(
            value,
            self.facts,
            self._scalars,
        )
        if alias_index is not None:
            self.facts.aliases[target] = alias_index


def _controlled_family(base: str | None, num_controls: int) -> str | None:
    """Map a controlled-gate base to its normalized family."""
    if base == "x":
        return "ccx" if num_controls >= 2 else "cx"
    if base == "z":
        return "ccz" if num_controls >= 2 else "cz"
    if base in _CTRL_PAIR_FAMILIES:
        return _CTRL_PAIR_FAMILIES[base]
    if base in _CTRL_ROTATION_FAMILIES:
        return _CTRL_ROTATION_FAMILIES[base]
    return None


def _single_name_target(node: ast.Assign) -> str | None:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    return node.targets[0].id


def _call_family(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        if node.attr in {"ctrl", "adj"}:
            return _call_family(node.value)
        return node.attr
    return None


def _eval_int(
    node: ast.AST,
    scalars: Mapping[str, int],
) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.Name):
        return scalars.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_int(node.operand, scalars)
        return None if value is None else -value
    if isinstance(node, ast.BinOp):
        left = _eval_int(node.left, scalars)
        right = _eval_int(node.right, scalars)
        operation = _INT_OPERATORS.get(type(node.op))
        if left is None or right is None or operation is None:
            return None
        if isinstance(node.op, ast.FloorDiv) and right == 0:
            return None
        return operation(left, right)
    return None


def _qubit_indices(
    node: ast.AST,
    facts: _SourceFacts,
    scalars: Mapping[str, int],
) -> list[int]:
    if isinstance(node, ast.Name):
        if node.id in facts.aliases:
            return [facts.aliases[node.id]]
        if node.id in facts.qubit_indices:
            return [facts.qubit_indices[node.id]]
        if node.id in facts.vector_sizes:
            offset = facts.vector_offsets.get(node.id, 0)
            return list(range(offset, offset + facts.vector_sizes[node.id]))
    subscript_index = _subscript_qubit_index(node, facts, scalars)
    if subscript_index is not None:
        return [subscript_index]
    return []


def _control_indices(
    node: ast.AST | None,
    facts: _SourceFacts,
    scalars: Mapping[str, int],
) -> list[int]:
    if node is None:
        return []
    if isinstance(node, ast.List | ast.Tuple):
        indices: list[int] = []
        for item in node.elts:
            indices.extend(_qubit_indices(item, facts, scalars))
        return indices
    return _qubit_indices(node, facts, scalars)


def _subscript_qubit_index(
    node: ast.AST,
    facts: _SourceFacts,
    scalars: Mapping[str, int],
) -> int | None:
    if not isinstance(node, ast.Subscript) or not isinstance(
        node.value,
        ast.Name,
    ):
        return None
    name = node.value.id
    index = _subscript_index(node, scalars)
    if index is None:
        return None
    if name not in facts.vector_sizes:
        return index
    size = facts.vector_sizes[name]
    if index < 0 or index >= size:
        return None
    return facts.vector_offsets.get(name, 0) + index


def _subscript_index(
    node: ast.AST,
    scalars: Mapping[str, int],
) -> int | None:
    if not isinstance(node, ast.Subscript):
        return None
    return _eval_int(node.slice, scalars)


def _subscript_qubit_slice(
    node: ast.AST,
    facts: _SourceFacts,
    scalars: Mapping[str, int],
) -> tuple[int, int] | None:
    """Resolve a known qvector slice to ``(offset, size)``."""
    if not isinstance(node, ast.Subscript) or not isinstance(
        node.value,
        ast.Name,
    ):
        return None
    if not isinstance(node.slice, ast.Slice) or node.slice.step is not None:
        return None
    name = node.value.id
    if name not in facts.vector_sizes:
        return None
    size = facts.vector_sizes[name]
    base = facts.vector_offsets.get(name, 0)
    lower = 0 if node.slice.lower is None else _eval_int(node.slice.lower, scalars)
    upper = size if node.slice.upper is None else _eval_int(node.slice.upper, scalars)
    if lower is None or upper is None:
        return None
    lower = max(0, lower)
    upper = min(size, upper)
    if upper <= lower:
        return None
    return base + lower, upper - lower


def _merge_gate_family_counts(
    target: dict[str, int],
    source: Mapping[str, int],
) -> None:
    for family, count in source.items():
        target[family] = max(target.get(family, 0), int(count))


def _current_target_name(cudaq: Any) -> str | None:
    try:
        return cudaq.get_target().name
    except Exception:
        return None
