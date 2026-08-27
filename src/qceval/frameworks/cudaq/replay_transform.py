"""CUDA-Q source-replay AST transforms and static allocation scans."""

from __future__ import annotations

import ast
from dataclasses import dataclass

_MEASURE_NAMES = frozenset({"mz", "my", "mx", "measure"})
_KERNEL_DECORATORS = frozenset({"kernel"})
_VECTOR_ALLOC_NAMES = frozenset({"qvector", "qreg", "qalloc"})
_SINGLE_ALLOC_NAMES = frozenset({"qubit", "qalloc"})
_QUBIT_ALLOC_NAMES = _VECTOR_ALLOC_NAMES | _SINGLE_ALLOC_NAMES
_KERNEL_FACTORY_NAMES = frozenset({"make_kernel"})


def parsed_num_qubits(code: str) -> int:
    """Return the statically visible CUDA-Q allocation width.

    Args:
        code: CUDA-Q Python source.

    Returns:
        Total number of statically allocated qubits.
    """
    sized = 0
    singles = 0
    for node in ast.walk(ast.parse(code)):
        if not (isinstance(node, ast.Call) and _attr_or_name(node.func) in _QUBIT_ALLOC_NAMES):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
            sized += int(node.args[0].value)
        elif _attr_or_name(node.func) in _SINGLE_ALLOC_NAMES and not node.args:
            singles += 1
    return sized + singles


def parsed_measured_wires(code: str) -> list[int]:
    """Return statically resolved CUDA-Q measurement wires in source order.

    Args:
        code: CUDA-Q Python source.

    Returns:
        Measured wire indices in source order.
    """
    wires: list[int] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call) and _attr_or_name(node.func) in _MEASURE_NAMES:
            for arg in node.args:
                wires.extend(_subscript_indices(arg))
    return wires


@dataclass
class _KernelInfo:
    """Resolved structure of a CUDA-Q source-replay candidate."""

    container: ast.FunctionDef
    kernel_var: str | None
    qubit_var: str
    alloc_index: int
    subscriptable: bool


def _transform_source(
    code: str,
    *,
    prep: dict[int, int],
    strip_leading_x_on: set[int],
    entry_point: str | None = None,
) -> str:
    tree = ast.parse(code)
    info = _resolve_kernel(tree, entry_point)
    head = info.container.body[: info.alloc_index + 1]
    tail = info.container.body[info.alloc_index + 1 :]
    prep_stmts = [_x_gate(info, wire) for wire, value in sorted(prep.items()) if value]
    body_stmts = _filter_body(tail, info, strip_leading_x_on=strip_leading_x_on)
    info.container.body = head + prep_stmts + body_stmts
    _reject_residual_measurements(info.container)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _resolve_kernel(tree: ast.Module, entry_point: str | None) -> _KernelInfo:
    """Resolve the candidate's kernel, supporting decorated and builder styles."""
    kernel = _find_kernel(tree)
    if kernel is not None:
        qubit_var, alloc_index, subscriptable = _qubit_allocation(kernel.body, kernel_var=None)
        return _KernelInfo(kernel, None, qubit_var, alloc_index, subscriptable)
    entry = _find_builder_entry(tree, entry_point)
    if entry is None:
        raise ValueError("no @cudaq.kernel or make_kernel builder found in candidate source")
    kernel_var = _find_kernel_factory_var(entry)
    qubit_var, alloc_index, subscriptable = _qubit_allocation(entry.body, kernel_var=kernel_var)
    return _KernelInfo(entry, kernel_var, qubit_var, alloc_index, subscriptable)


def _filter_body(
    stmts: list[ast.stmt],
    info: _KernelInfo,
    *,
    strip_leading_x_on: set[int],
) -> list[ast.stmt]:
    """Drop measurements and the candidate's input-preparation ``X`` gates."""
    out: list[ast.stmt] = []
    stripping = True
    for stmt in stmts:
        if _is_measurement(stmt, info):
            continue
        wire = _bare_x_wire(stmt, info)
        if stripping and wire is not None and wire in strip_leading_x_on:
            continue
        if not (stripping and wire is not None):
            stripping = False
        stripped = _strip_nested_measurements(stmt, info)
        if stripped is not None:
            out.append(stripped)
    return out


def _strip_nested_measurements(stmt: ast.stmt, info: _KernelInfo) -> ast.stmt | None:
    """Remove discarded measurement statements nested in compound statements.

    Returns ``None`` when a loop body empties out entirely, so the caller can
    drop the loop instead of leaving a body CUDA-Q cannot compile.
    """
    if _is_measurement(stmt, info):
        return None
    for field in ("body", "orelse", "finalbody"):
        block = getattr(stmt, field, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            kept = [
                inner for inner in (_strip_nested_measurements(child, info) for child in block) if inner is not None
            ]
            setattr(stmt, field, kept if field != "body" else kept or [ast.Pass()])
    if isinstance(stmt, ast.For | ast.While) and all(isinstance(s, ast.Pass) for s in stmt.body) and not stmt.orelse:
        return None
    return stmt


def _reject_residual_measurements(container: ast.FunctionDef) -> None:
    """Fail closed when a measurement survives the replay transform.

    A measurement whose result feeds classical logic (an assignment, a branch
    condition, an expression operand) collapses the state; replaying such a
    kernel with ``cudaq.get_state`` would report one collapsed branch as the
    exact statevector.
    """
    for node in ast.walk(container):
        if isinstance(node, ast.Call) and _attr_or_name(node.func) in _MEASURE_NAMES:
            raise ValueError(
                "CUDA-Q replay cannot strip a measurement whose result is consumed; "
                "the kernel requires dynamic simulation"
            )


def _find_kernel(tree: ast.Module) -> ast.FunctionDef | None:
    decorated: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            _attr_or_name(dec) in _KERNEL_DECORATORS for dec in node.decorator_list
        ):
            decorated.append(node)
    if not decorated:
        return None
    for node in decorated:
        if _has_allocation(node.body):
            return node
    return decorated[0]


def _has_allocation(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and _attr_or_name(stmt.value.func) in _QUBIT_ALLOC_NAMES
        for stmt in body
    )


def _find_builder_entry(tree: ast.Module, entry_point: str | None) -> ast.FunctionDef | None:
    """Return the builder entry function that defines a kernel object."""
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if entry_point is not None:
        for node in functions:
            if node.name == entry_point and _find_kernel_factory_var(node) is not None:
                return node
    for node in functions:
        if _find_kernel_factory_var(node) is not None:
            return node
    return None


def _find_kernel_factory_var(fn: ast.FunctionDef) -> str | None:
    for stmt in fn.body:
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and _attr_or_name(stmt.value.func) in _KERNEL_FACTORY_NAMES
            and stmt.targets
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return stmt.targets[0].id
    return None


def _qubit_allocation(
    body: list[ast.stmt],
    *,
    kernel_var: str | None,
) -> tuple[str, int, bool]:
    """Locate the qubit-register allocation in ``body``."""
    for index, stmt in enumerate(body):
        if not (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and stmt.targets
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        name = _attr_or_name(stmt.value.func)
        if name not in _QUBIT_ALLOC_NAMES:
            continue
        if kernel_var is not None and not _is_method_on(stmt.value.func, kernel_var):
            continue
        has_size = bool(stmt.value.args) and isinstance(stmt.value.args[0], ast.Constant)
        subscriptable = has_size and name in _VECTOR_ALLOC_NAMES
        return stmt.targets[0].id, index, subscriptable
    raise ValueError("no qubit register allocation found in CUDA-Q candidate")


def _is_method_on(func: ast.AST, kernel_var: str) -> bool:
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == kernel_var


def _is_measurement(stmt: ast.stmt, info: _KernelInfo) -> bool:
    del info
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    return _attr_or_name(stmt.value.func) in _MEASURE_NAMES


def _bare_x_wire(stmt: ast.stmt, info: _KernelInfo) -> int | None:
    """Return the wire of a bare single-qubit ``X`` statement, else ``None``."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    call = stmt.value
    if info.kernel_var is None:
        if not (isinstance(call.func, ast.Name) and call.func.id == "x"):
            return None
    elif not (_is_method_on(call.func, info.kernel_var) and _attr_or_name(call.func) == "x"):
        return None
    if len(call.args) != 1:
        return None
    return _target_wire(call.args[0], info)


def _target_wire(arg: ast.AST, info: _KernelInfo) -> int | None:
    """Return the wire a single-qubit gate argument addresses, or ``None``."""
    if not info.subscriptable:
        return 0 if isinstance(arg, ast.Name) and arg.id == info.qubit_var else None
    indices = _subscript_indices(arg, qubit_var=info.qubit_var)
    return indices[0] if len(indices) == 1 else None


def _x_gate(info: _KernelInfo, wire: int) -> ast.Expr:
    if info.subscriptable:
        target: ast.expr = ast.Subscript(
            value=ast.Name(id=info.qubit_var, ctx=ast.Load()),
            slice=ast.Constant(value=wire),
            ctx=ast.Load(),
        )
    else:
        target = ast.Name(id=info.qubit_var, ctx=ast.Load())
    if info.kernel_var is None:
        function: ast.expr = ast.Name(id="x", ctx=ast.Load())
    else:
        function = ast.Attribute(
            value=ast.Name(id=info.kernel_var, ctx=ast.Load()),
            attr="x",
            ctx=ast.Load(),
        )
    return ast.Expr(value=ast.Call(func=function, args=[target], keywords=[]))


def _subscript_indices(
    node: ast.AST,
    *,
    qubit_var: str | None = None,
) -> list[int]:
    if isinstance(node, ast.Subscript):
        if qubit_var is not None and not (isinstance(node.value, ast.Name) and node.value.id == qubit_var):
            return []
        value = _const_int(node.slice)
        return [value] if value is not None else []
    if isinstance(node, ast.List | ast.Tuple):
        out: list[int] = []
        for element in node.elts:
            out.extend(_subscript_indices(element, qubit_var=qubit_var))
        return out
    return []


def _const_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def _attr_or_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""
