"""Restricted-source validation for the Qiskit symbolic worker."""

from __future__ import annotations

import ast

_ALLOWED_IMPORTS = {
    "math": frozenset({"pi"}),
    "numpy": frozenset({"pi"}),
    "qiskit": frozenset({"QuantumCircuit", "QuantumRegister"}),
    "qiskit.circuit": frozenset({"Parameter", "QuantumRegister"}),
    "qiskit.circuit.library": frozenset({"RZGate", "SXGate"}),
}
_ALLOWED_CALLS = frozenset({"QuantumCircuit", "QuantumRegister", "RZGate", "SXGate", "append", "rz", "sx", "sxdg"})
_FORBIDDEN_CONTROL = (
    ast.Assert,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.BoolOp,
    ast.Compare,
    ast.For,
    ast.If,
    ast.IfExp,
    ast.Lambda,
    ast.ListComp,
    ast.Match,
    ast.NamedExpr,
    ast.SetComp,
    ast.Try,
    ast.While,
    ast.With,
    ast.comprehension,
)


def _validate_source(code: str, entry_point: str) -> str | None:
    """Validate that source is within the restricted symbolic grammar.

    Args:
        code: Candidate Python source.
        entry_point: Required top-level function name.

    Returns:
        Machine-readable issue code, or ``None`` when valid.
    """
    if len(code.encode("utf-8")) > 100_000:
        return "symbolic_source_size_limit"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "symbolic_source_syntax_error"
    function, issue = _validate_module(tree, entry_point)
    if issue is not None or function is None:
        return issue
    return _validate_function(function)


def _validate_module(tree: ast.Module, entry_point: str) -> tuple[ast.FunctionDef | None, str | None]:
    function = None
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            issue = _validate_import(statement)
            if issue is not None:
                return None, issue
        elif isinstance(statement, ast.FunctionDef) and statement.name == entry_point:
            function = statement
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ) or (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and _call_name(statement.value.func) == entry_point
        ):
            continue
        else:
            return None, f"symbolic_module_statement_unsupported:{type(statement).__name__}"
    if function is None:
        return None, "symbolic_entry_point_missing"
    return function, None


def _validate_function(function: ast.FunctionDef) -> str | None:
    for node in ast.walk(function):
        if node is not function and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return "symbolic_nested_definition_unsupported"
        if isinstance(node, _FORBIDDEN_CONTROL):
            return f"symbolic_control_unsupported:{type(node).__name__}"
        if isinstance(node, ast.Call) and _call_name(node.func) not in _ALLOWED_CALLS:
            return f"symbolic_call_unsupported:{_call_name(node.func) or type(node.func).__name__}"
    return None


def _validate_import(statement: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(statement, ast.Import):
        unexpected = [alias.name for alias in statement.names if alias.name not in _ALLOWED_IMPORTS]
        return None if not unexpected else f"symbolic_import_unsupported:{unexpected[0]}"
    module = statement.module or ""
    allowed = _ALLOWED_IMPORTS.get(module)
    if module not in _ALLOWED_IMPORTS or allowed is None:
        return f"symbolic_import_unsupported:{module}"
    unexpected = sorted({alias.name for alias in statement.names} - allowed)
    return None if not unexpected else f"symbolic_import_unsupported:{module}:{unexpected[0]}"


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
