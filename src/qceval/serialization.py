"""Serialization helpers for QCircuitEval models and scientific objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert common Python objects into JSON-compatible values.

    The conversion is recursive and intentionally conservative.  Dataclasses,
    mappings, paths, complex numbers, array-like objects, and sequences keep
    structured forms; unknown objects fall back to ``repr`` so output writing
    does not fail after a successful benchmark run.

    Args:
        value: Object to convert.

    Returns:
        JSON-compatible representation made from dictionaries, lists, strings,
        numbers, booleans, and ``None``.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [to_jsonable(item) for item in value]
    return repr(value)
