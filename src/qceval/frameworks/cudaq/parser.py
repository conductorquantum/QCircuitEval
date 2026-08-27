"""Parse CUDA-Q source into the neutral circuit IR."""

from __future__ import annotations

import ast
import copy
from collections.abc import Iterable
from typing import Any

from qceval.evals.ir.core import Circuit, Gate
from qceval.frameworks.cudaq.gates import (
    _CUDAQ_IGNORED_CALLS,
    _CUDAQ_MEASUREMENT_CALLS,
    _cudaq_gate_from_call,
    _subscript_indices,
)
from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.frameworks.cudaq.values import (
    _attr_or_name,
    _const_int_expr,
    _cudaq_constant_bindings,
    _cudaq_registered_matrices,
)

_CUDAQ_UNSUPPORTED_CONTROL_FLOW = (
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.Match,
)


def from_cudaq(program: CudaqProgram) -> Circuit:
    """Convert supported CUDA-Q source into the neutral circuit IR.

    Args:
        program: CUDA-Q source and public entry-point name.

    Returns:
        The lowered framework-neutral circuit.

    Raises:
        SyntaxError: If the source is not valid Python.
        ValueError: If the entry point cannot be resolved.
        NotImplementedError: If the kernel uses unsupported allocation,
            control flow, measurement, or gate constructs.
    """
    tree = ast.parse(program.code)
    kernel = _find_cudaq_kernel(tree, program.entry_point)
    constants = _cudaq_constant_bindings(
        program.code,
        program.entry_point,
        kernel,
        call_args=program.call_args,
        include_kernel_assignments=False,
    )
    registered = _cudaq_registered_matrices(tree)
    register_name, num_qubits = _cudaq_register(kernel)
    gates: list[Gate] = []
    measured_wires: set[int] = set()
    for statement, constants_at_call in _kernel_calls_in_source_order(kernel, constants):
        name = _attr_or_name(statement.func)
        if name in _CUDAQ_MEASUREMENT_CALLS:
            measured_wires.update(
                _cudaq_measurement_wires(
                    statement,
                    register_name,
                    num_qubits,
                )
            )
            continue
        if name in _CUDAQ_IGNORED_CALLS:
            continue
        gate = _cudaq_gate_from_call(
            statement,
            constants=constants_at_call,
            registered=registered,
            register_name=register_name,
            num_qubits=num_qubits,
        )
        if gate is None:
            continue
        gate_name, matrix, targets, controls = gate
        used = set(targets) | set(controls)
        if measured_wires.intersection(used):
            raise NotImplementedError("CUDA-Q mid-circuit measurement is non-unitary and cannot be equivalence-checked")
        if not controls and matrix.shape == (2, 2) and len(targets) > 1:
            gates.extend(Gate.full(matrix, (target,), name=gate_name) for target in targets)
        elif controls:
            gates.append(
                Gate.controlled(
                    matrix,
                    targets=targets,
                    controls=controls,
                    name=gate_name,
                )
            )
        else:
            gates.append(Gate.full(matrix, targets, name=gate_name))
    return Circuit(num_qubits=num_qubits, gates=tuple(gates))


def _kernel_calls_in_source_order(
    kernel: ast.FunctionDef,
    initial_constants: dict[str, Any] | None = None,
) -> Iterable[tuple[ast.Call, dict[str, Any]]]:
    """Yield CUDA-Q calls with the constants visible at each statement."""
    constants = dict(initial_constants or {})
    yield from _kernel_calls_with_constants(kernel, constants)


def _kernel_calls_with_constants(
    kernel: ast.FunctionDef,
    constants: dict[str, Any],
) -> Iterable[tuple[ast.Call, dict[str, Any]]]:
    """Yield calls in source order while updating a mutable environment."""
    for statement in kernel.body:
        yield from _cudaq_calls_from_stmt(statement, constants)


def _cudaq_calls_from_stmt(
    stmt: ast.stmt,
    constants: dict[str, Any],
) -> Iterable[tuple[ast.Call, dict[str, Any]]]:
    if isinstance(stmt, ast.For):
        if not isinstance(stmt.target, ast.Name):
            raise NotImplementedError("CUDA-Q static loop target must be a name")
        values = _static_range_values(stmt.iter)
        for value in values:
            replacer = _NameConstantReplacer(
                stmt.target.id,
                value,
            )
            for body_statement in stmt.body:
                replaced = replacer.visit(copy.deepcopy(body_statement))
                ast.fix_missing_locations(replaced)
                yield from _cudaq_calls_from_stmt(replaced, constants)
        return
    if isinstance(stmt, _CUDAQ_UNSUPPORTED_CONTROL_FLOW):
        raise NotImplementedError("CUDA-Q kernels with non-static control flow are not supported")
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            yield node, dict(constants)
    if isinstance(stmt, ast.Assign):
        _update_constant_assignment(stmt, constants)


def _update_constant_assignment(stmt: ast.Assign, constants: dict[str, Any]) -> None:
    """Apply one assignment after its RHS calls have been evaluated."""
    from qceval.frameworks.cudaq.values import _collect_constant_assignments

    if len(stmt.targets) != 1:
        return
    target = stmt.targets[0]
    if isinstance(target, ast.Name):
        names = [target.id]
    elif isinstance(target, ast.Tuple):
        names = [item.id for item in target.elts if isinstance(item, ast.Name)]
    else:
        names = []
    for name in names:
        constants.pop(name, None)
    _collect_constant_assignments((stmt,), constants)


def _static_range_values(node: ast.AST) -> range:
    if not (isinstance(node, ast.Call) and _attr_or_name(node.func) == "range"):
        raise NotImplementedError("CUDA-Q loops must be literal range(...) loops")
    args = [_const_int_expr(argument) for argument in node.args]
    if any(argument is None for argument in args):
        raise NotImplementedError("CUDA-Q loop range bounds must be constant")
    values = [int(argument) for argument in args if argument is not None]
    return range(*values)


class _NameConstantReplacer(ast.NodeTransformer):
    """Replace one loop-variable name with a literal integer."""

    def __init__(self, name: str, value: int) -> None:
        self._name = name
        self._value = value

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id == self._name:
            return ast.copy_location(
                ast.Constant(self._value),
                node,
            )
        return node


def _find_cudaq_kernel(
    tree: ast.Module,
    entry_point: str,
) -> ast.FunctionDef:
    """Resolve the decorated kernel exposed by ``entry_point``."""
    top_level = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    entry = top_level.get(entry_point)
    if entry is None:
        raise ValueError(f"CUDA-Q entry point {entry_point!r} was not found")
    if _is_cudaq_kernel(entry):
        return entry
    nested = {
        node.name: node
        for node in ast.walk(entry)
        if isinstance(node, ast.FunctionDef) and node is not entry and _is_cudaq_kernel(node)
    }
    returned_names = {
        node.value.id for node in entry.body if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
    }
    resolved = [
        kernel
        for name in returned_names
        if (kernel := nested.get(name) or top_level.get(name)) is not None and _is_cudaq_kernel(kernel)
    ]
    if len(resolved) != 1:
        raise ValueError(f"CUDA-Q entry point {entry_point!r} must return exactly one decorated kernel")
    return resolved[0]


def _is_cudaq_kernel(function: ast.FunctionDef) -> bool:
    return any(_attr_or_name(decorator) == "kernel" for decorator in function.decorator_list)


def _cudaq_register(kernel: ast.FunctionDef) -> tuple[str, int]:
    """Return the sole statically sized register allocated by ``kernel``."""
    allocations: list[tuple[str, ast.Call]] = []
    for node in ast.walk(kernel):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _attr_or_name(node.value.func) in {"qvector", "qreg", "qalloc"}
        ):
            continue
        allocations.append((node.targets[0].id, node.value))
    if len(allocations) != 1:
        raise NotImplementedError("CUDA-Q equivalence requires exactly one statically sized qubit register")
    register_name, allocation = allocations[0]
    width = _const_int_expr(allocation.args[0]) if len(allocation.args) == 1 else None
    if width is None or width < 0:
        raise NotImplementedError("CUDA-Q qubit register size must be a non-negative integer constant")
    return register_name, width


def _cudaq_measurement_wires(
    call: ast.Call,
    register_name: str,
    num_qubits: int,
) -> list[int]:
    wires: list[int] = []
    for argument in call.args:
        if isinstance(argument, ast.Name) and argument.id == register_name:
            wires.extend(range(num_qubits))
            continue
        resolved = _subscript_indices(argument)
        if not resolved:
            raise NotImplementedError("CUDA-Q measurement wires must be statically resolvable")
        wires.extend(resolved)
    if not wires:
        raise NotImplementedError("CUDA-Q measurement must name at least one wire")
    return wires
