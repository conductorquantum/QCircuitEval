"""Derive entry-point call bindings from signatures or source arity."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any

from qceval.semantics.contracts.kinds import SignatureSpec


def call_args_from_signature(signature: SignatureSpec, input_value: Any) -> tuple[Any, ...]:
    """Build positional call arguments from a contract signature and task input.

    Args:
        signature: Contract entry-point signature.
        input_value: Bundled task input for this entry point, or ``None`` when
            the task declares no runtime inputs.

    Returns:
        Positional arguments to pass to the candidate entry point.

    Raises:
        ValueError: If a multi-argument signature cannot be bound to
            ``input_value``.
    """
    if input_value is None and not any(argument.required for argument in signature.arguments):
        return ()
    return call_args_from_arity(len(signature.arguments), input_value, required=True)


def call_args_from_code(code: str, entry_point: str, input_value: Any) -> tuple[Any, ...]:
    """Build positional call arguments from source arity and task input.

    Used when a contract signature is unavailable (for example smoke canonical
    replay before packaged contracts are merged). Arity comes from the named
    function definition, never from task-id tables.

    Args:
        code: Candidate or canonical Python source.
        entry_point: Function name to bind.
        input_value: Bundled task input for this entry point, or ``None``.

    Returns:
        Positional arguments to pass to the entry point.
    """
    return call_args_from_arity(required_arity_from_code(code, entry_point), input_value, required=False)


def call_args_from_arity(arity: int, input_value: Any, *, required: bool = False) -> tuple[Any, ...]:
    """Bind a bundled task input to a fixed positional arity.

    Args:
        arity: Number of positional arguments expected by the entry point.
        input_value: Bundled task input, or ``None`` when absent.
        required: When true, missing inputs for non-zero arity raise.

    Returns:
        Positional call arguments.

    Raises:
        ValueError: If binding is impossible for the declared arity.
    """
    if arity <= 0:
        return ()
    if input_value is None:
        if required:
            raise ValueError(f"missing required entry-point arguments for arity {arity}")
        return ()
    if arity == 1:
        return (input_value,)
    if not isinstance(input_value, Sequence) or isinstance(input_value, str | bytes):
        raise ValueError(
            f"signature declares {arity} arguments but input value is not a sequence: {type(input_value).__name__}"
        )
    if len(input_value) != arity:
        raise ValueError(f"signature declares {arity} arguments but input sequence has length {len(input_value)}")
    return tuple(input_value)


def required_arity_from_code(code: str, entry_point: str) -> int:
    """Return the number of required positional parameters for ``entry_point``.

    Args:
        code: Python source containing the entry point.
        entry_point: Function name to inspect.

    Returns:
        Count of required positional-or-keyword parameters (no defaults).
        Returns ``0`` when the definition cannot be found or parsed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == entry_point:
            positional = list(node.args.posonlyargs) + list(node.args.args)
            return max(0, len(positional) - len(node.args.defaults))
    return 0
