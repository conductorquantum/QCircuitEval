"""Framework-neutral bounded source scans."""

from __future__ import annotations

import ast


def source_names(source_code: str | None) -> set[str]:
    """Collect normalized names used by explicit anti-shortcut policies.

    Args:
        source_code: Candidate Python source, or ``None`` when unavailable.

    Returns:
        Lowercase referenced names, attribute names, and import-path parts.
    """
    if not source_code:
        return set()
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        names.update(_node_names(node))
    return names


def source_import_names(source_code: str | None) -> set[str]:
    """Collect only names introduced by import statements.

    Args:
        source_code: Candidate Python source, or ``None``.

    Returns:
        Lowercase import-path components, imported symbols, and aliases.
    """
    tree = _source_tree(source_code)
    if tree is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.update(part.lower() for part in node.module.split("."))
            names.update(_imported_names(node.names))
        elif isinstance(node, ast.Import):
            names.update(_imported_names(node.names))
        elif isinstance(node, ast.Call):
            dynamic = _dynamic_import_name(node)
            if dynamic is not None:
                names.update(part.lower() for part in dynamic.split("."))
    return names


def _dynamic_import_name(call: ast.Call) -> str | None:
    family = _dotted_name(call.func).lower().split(".")[-1]
    if family not in {"__import__", "import_module"} or not call.args:
        return None
    return _static_string(call.args[0])


def _static_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        if left is not None and right is not None:
            return left + right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
    ):
        separator = _static_string(node.func.value)
        values = node.args[0]
        if separator is not None and isinstance(values, ast.List | ast.Tuple):
            items = [_static_string(item) for item in values.elts]
            if all(item is not None for item in items):
                return separator.join(item for item in items if item is not None)
    return None


def source_dynamic_features(source_code: str | None) -> set[str]:
    """Collect reflection and runtime-code features that evade static policy.

    Statically resolvable ``getattr`` and import calls remain admissible: their
    resolved target flows through the normal forbidden-call/import checks.
    Unresolved reflection is evidence that a blocked constructor cannot be
    ruled out and therefore fails closed when a contract requires provenance.

    Args:
        source_code: Candidate Python source, or ``None`` when unavailable.

    Returns:
        Normalized names of unresolved dynamic source features.
    """
    tree = _source_tree(source_code)
    if tree is None:
        return set()
    features: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted_name(node.func).lower().split(".")[-1]
            if name == "getattr" and (len(node.args) < 2 or _static_string(node.args[1]) is None):
                features.add("dynamic_getattr")
            elif name in {"eval", "exec", "compile", "globals", "locals", "vars", "setattr"}:
                features.add(name)
            elif name in {"__import__", "import_module"} and (not node.args or _static_string(node.args[0]) is None):
                features.add("dynamic_import")
        elif isinstance(node, ast.Attribute) and node.attr == "__dict__":
            features.add("dynamic_namespace")
    return features


def source_call_names(source_code: str | None) -> set[str]:
    """Collect names that are actually invoked by candidate source.

    Import aliases are resolved to the imported symbol as well as the spelling
    used at the call site. This avoids rejecting innocent local variables whose
    names happen to match a forbidden API while still catching common aliasing
    of prohibited constructors and functions.

    Args:
        source_code: Candidate Python source, or ``None``.

    Returns:
        Lowercase terminal call names and resolved imported symbols.
    """
    tree = _source_tree(source_code)
    if tree is None:
        return set()
    aliases = _import_aliases(tree)
    aliases.update(_callable_assignment_aliases(tree, aliases))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _callable_dotted_name(node.func).lower()
        if not dotted:
            continue
        names.add(dotted.split(".")[-1])
        root = dotted.split(".", 1)[0]
        resolved = aliases.get(root)
        if resolved is not None:
            suffix = dotted[len(root) :]
            names.add(f"{resolved}{suffix}".split(".")[-1])
    return names


def _callable_assignment_aliases(tree: ast.Module, imports: dict[str, str]) -> dict[str, str]:
    """Resolve simple local aliases of framework methods and constructors."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        dotted = _callable_dotted_name(value).lower()
        if not dotted:
            continue
        root = dotted.split(".", 1)[0]
        resolved = aliases.get(root, imports.get(root))
        aliases[target.id.lower()] = f"{resolved}{dotted[len(root) :]}" if resolved is not None else dotted
    return aliases


def _callable_dotted_name(node: ast.AST) -> str:
    """Return a direct or statically resolved ``getattr`` callable name."""
    direct = _dotted_name(node)
    if direct:
        return direct
    if not isinstance(node, ast.Call) or _dotted_name(node.func).lower().split(".")[-1] != "getattr":
        return ""
    if len(node.args) < 2:
        return ""
    base = _dotted_name(node.args[0])
    attribute = _static_string(node.args[1])
    return f"{base}.{attribute}" if base and attribute else ""


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname.lower()] = alias.name.lower()
                else:
                    # ``import a.b.c`` binds the top-level package ``a``.
                    local = alias.name.split(".", 1)[0]
                    aliases[local.lower()] = local.lower()
        elif isinstance(node, ast.ImportFrom):
            module = "" if node.module is None else f"{node.module}."
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local.lower()] = f"{module}{alias.name}".lower()
    return aliases


def _resolved_name_parts(dotted: str, aliases: dict[str, str]) -> set[str]:
    """Return lower-cased components of ``dotted`` resolved through ``aliases``.

    The longest matching prefix is replaced by its alias target so that
    module-qualified references (``library.QFT``) and renamed imports
    (``QFT as myqft``) are checked against the real symbol path.
    """
    if not dotted:
        return set()
    parts = dotted.split(".")
    for prefix_len in range(len(parts), 0, -1):
        prefix = ".".join(parts[:prefix_len])
        resolved = aliases.get(prefix)
        if resolved is not None:
            full = (*resolved.split("."), *parts[prefix_len:])
            return {part.lower() for part in full}
    return set()


def source_import_references(source_code: str | None) -> set[str]:
    """Collect names that refer to imported modules or symbols.

    Includes import-path components, imported symbols, and attribute or call
    references that resolve through an imported alias (e.g.
    ``qiskit.circuit.library.QFT`` or ``library.QFTGate``).

    Args:
        source_code: Candidate Python source, or ``None`` when unavailable.

    Returns:
        Lowercased import-path components, imported symbols, and resolved
        alias references.
    """
    tree = _source_tree(source_code)
    if tree is None:
        return set()
    aliases = _import_aliases(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        names.update(_import_reference_names(node, aliases))
    return names


def _import_reference_names(node: ast.AST, aliases: dict[str, str]) -> set[str]:
    if isinstance(node, ast.ImportFrom):
        names = set(_imported_names(node.names))
        if node.module is not None:
            names.update(part.lower() for part in node.module.split("."))
        return names
    if isinstance(node, ast.Import):
        return set(_imported_names(node.names))
    if isinstance(node, ast.Call):
        return _call_import_names(node, aliases)
    if isinstance(node, ast.Attribute):
        dotted = _dotted_name(node).lower()
        return _resolved_name_parts(dotted, aliases) if dotted else set()
    return set()


def _call_import_names(node: ast.Call, aliases: dict[str, str]) -> set[str]:
    names: set[str] = set()
    dynamic = _dynamic_import_name(node)
    if dynamic is not None:
        names.update(part.lower() for part in dynamic.split("."))
    dotted = _callable_dotted_name(node.func).lower()
    if dotted:
        names.update(_resolved_name_parts(dotted, aliases))
    return names


def _node_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id.lower()}
    if isinstance(node, ast.Attribute):
        return {node.attr.lower()}
    if isinstance(node, ast.ImportFrom):
        result = set() if node.module is None else {part.lower() for part in node.module.split(".")}
        return result | _imported_names(node.names)
    if isinstance(node, ast.Import):
        return _imported_names(node.names)
    return set()


def _imported_names(aliases: list[ast.alias]) -> set[str]:
    names = {part.lower() for alias in aliases for part in alias.name.split(".")}
    names.update(alias.asname.lower() for alias in aliases if alias.asname)
    return names


def _source_tree(source_code: str | None) -> ast.Module | None:
    if not source_code:
        return None
    try:
        return ast.parse(source_code)
    except SyntaxError:
        return None


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""
