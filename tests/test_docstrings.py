from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "qceval"


def test_public_python_api_has_google_style_docstrings() -> None:
    missing: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(SRC_ROOT.parents[0])
        if ast.get_docstring(tree) is None:
            missing.append(f"{relative}: module docstring")
        for name, node in _public_nodes(tree):
            docstring = ast.get_docstring(node) or ""
            if not docstring:
                missing.append(f"{relative}: {name} docstring")
                continue
            if isinstance(node, ast.ClassDef):
                continue
            if _has_public_parameters(node) and "Args:" not in docstring:
                missing.append(f"{relative}: {name} Args section")
            if _returns_value(node) and "Returns:" not in docstring and not docstring.lstrip().startswith("Return"):
                missing.append(f"{relative}: {name} Returns section")

    assert not missing, "\n".join(missing)


def _public_nodes(tree: ast.Module) -> list[tuple[str, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]]:
    nodes: list[tuple[str, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            nodes.append((node.name, node))
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and not child.name.startswith("_"):
                    nodes.append((f"{node.name}.{child.name}", child))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
            nodes.append((node.name, node))
    return nodes


def _has_public_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    args = [arg.arg for arg in [*node.args.args, *node.args.kwonlyargs]]
    args = [arg for arg in args if arg not in {"self", "cls"}]
    return bool(args or node.args.vararg or node.args.kwarg)


def _returns_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.returns is None:
        return True
    return not (isinstance(node.returns, ast.Constant) and node.returns.value is None)
