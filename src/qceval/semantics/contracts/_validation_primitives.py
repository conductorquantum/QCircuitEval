"""Primitive validators shared by contract parsing modules."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, TypeVar

from qceval.semantics.contracts.kinds import (
    ContractValidationError,
    FrozenArray,
    FrozenObject,
    JsonValue,
    ParameterPointValue,
)

SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EnumT = TypeVar("_EnumT")


def object_value(value: Any, path: str, *, required: set[str]) -> Mapping[str, Any]:
    """Validate a mapping with exactly the required keys.

    Args:
        value: Candidate mapping value.
        path: JSON path used in validation errors.
        required: Exact set of allowed and required keys.

    Returns:
        The validated mapping.

    Raises:
        ContractValidationError: If the value or its keys are invalid.
    """
    if not isinstance(value, Mapping):
        fail(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        fail(path, "object keys must be strings")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        fail(path, f"missing keys: {missing}")
    if unknown:
        fail(path, f"unknown keys: {unknown}")
    return value


def array_value(value: Any, path: str) -> Sequence[Any]:
    """Validate an array-like value.

    Args:
        value: Candidate sequence value.
        path: JSON path used in validation errors.

    Returns:
        The validated non-string sequence.

    Raises:
        ContractValidationError: If ``value`` is not an array.
    """
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        fail(path, "must be an array")
    return value


def string_value(value: Any, path: str) -> str:
    """Validate a string value.

    Args:
        value: Candidate string value.
        path: JSON path used in validation errors.

    Returns:
        The validated string.

    Raises:
        ContractValidationError: If ``value`` is not a string.
    """
    if not isinstance(value, str):
        fail(path, "must be a string")
    return value


def nonempty_string(value: Any, path: str) -> str:
    """Validate a nonempty string value.

    Args:
        value: Candidate string value.
        path: JSON path used in validation errors.

    Returns:
        The validated nonempty string.

    Raises:
        ContractValidationError: If ``value`` is not a nonempty string.
    """
    result = string_value(value, path)
    if not result:
        fail(path, "must not be empty")
    return result


def string_tuple(value: Any, path: str) -> tuple[str, ...]:
    """Validate an array of unique nonempty strings.

    Args:
        value: Candidate string-array value.
        path: JSON path used in validation errors.

    Returns:
        The validated strings as an immutable tuple.

    Raises:
        ContractValidationError: If the array or any string is invalid.
    """
    result = tuple(nonempty_string(item, f"{path}[{index}]") for index, item in enumerate(array_value(value, path)))
    unique(list(result), path, "values")
    return result


def boolean_value(value: Any, path: str) -> bool:
    """Validate a boolean value.

    Args:
        value: Candidate boolean value.
        path: JSON path used in validation errors.

    Returns:
        The validated boolean.

    Raises:
        ContractValidationError: If ``value`` is not a boolean.
    """
    if not isinstance(value, bool):
        fail(path, "must be a boolean")
    return value


def integer_value(value: Any, path: str, *, minimum: int) -> int:
    """Validate a lower-bounded integer.

    Args:
        value: Candidate integer value.
        path: JSON path used in validation errors.
        minimum: Inclusive lower bound.

    Returns:
        The validated integer.

    Raises:
        ContractValidationError: If the value is not an integer or is below
            ``minimum``.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        fail(path, "must be an integer")
    if value < minimum:
        fail(path, f"must be >= {minimum}")
    return value


def integer_tuple(value: Any, path: str, *, minimum: int) -> tuple[int, ...]:
    """Validate an array of lower-bounded integers.

    Args:
        value: Candidate integer-array value.
        path: JSON path used in validation errors.
        minimum: Inclusive lower bound for each integer.

    Returns:
        The validated integers as an immutable tuple.

    Raises:
        ContractValidationError: If the array or any integer is invalid.
    """
    return tuple(
        integer_value(item, f"{path}[{index}]", minimum=minimum) for index, item in enumerate(array_value(value, path))
    )


def number_value(value: Any, path: str, *, minimum: float) -> float:
    """Validate a finite lower-bounded number.

    Args:
        value: Candidate numeric value.
        path: JSON path used in validation errors.
        minimum: Inclusive lower bound.

    Returns:
        The validated value converted to ``float``.

    Raises:
        ContractValidationError: If the value is nonnumeric, nonfinite, or
            below ``minimum``.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        fail(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        fail(path, "must be finite")
    if result < minimum:
        fail(path, f"must be >= {minimum}")
    return result


def positive_number(value: Any, path: str) -> float:
    """Validate a positive finite number.

    Args:
        value: Candidate numeric value.
        path: JSON path used in validation errors.

    Returns:
        The validated value converted to ``float``.

    Raises:
        ContractValidationError: If the value is not finite and positive.
    """
    result = number_value(value, path, minimum=0.0)
    if result <= 0.0:
        fail(path, "must be positive")
    return result


def number_tuple(value: Any, path: str) -> tuple[float, ...]:
    """Validate an array of finite numbers.

    Args:
        value: Candidate numeric-array value.
        path: JSON path used in validation errors.

    Returns:
        The validated numbers as an immutable tuple of floats.

    Raises:
        ContractValidationError: If the array or any number is invalid.
    """
    return tuple(
        number_value(item, f"{path}[{index}]", minimum=-math.inf) for index, item in enumerate(array_value(value, path))
    )


def parameter_point_tuple(value: Any, path: str) -> tuple[ParameterPointValue, ...]:
    """Validate one finite JSON-scalar parameter binding.

    Exhaustive finite domains may contain nullable and discrete API arguments,
    not only real-valued mathematical parameters.

    Args:
        value: Candidate point represented as a JSON array.
        path: JSON path used in validation errors.

    Returns:
        Immutable, type-preserving scalar argument values.
    """

    result: list[ParameterPointValue] = []
    for index, item in enumerate(array_value(value, path)):
        item_path = f"{path}[{index}]"
        if isinstance(item, float) and not math.isfinite(item):
            fail(item_path, "must be finite")
        if item is not None and not isinstance(item, str | int | float | bool):
            fail(item_path, "must be a JSON scalar")
        result.append(item)
    return tuple(result)


def semantic_version(value: Any, path: str) -> str:
    """Validate a strict semantic version string.

    Args:
        value: Candidate semantic version.
        path: JSON path used in validation errors.

    Returns:
        The validated ``MAJOR.MINOR.PATCH`` string.

    Raises:
        ContractValidationError: If ``value`` is not a strict semantic
            version.
    """
    result = string_value(value, path)
    if not SEMVER_PATTERN.fullmatch(result):
        fail(path, "must be MAJOR.MINOR.PATCH")
    return result


def enum_value(enum_type: type[_EnumT], value: Any, path: str) -> _EnumT:
    """Validate and construct an enum member.

    Args:
        enum_type: Enum class to construct.
        value: Candidate string enum value.
        path: JSON path used in validation errors.

    Returns:
        The matching member of ``enum_type``.

    Raises:
        ContractValidationError: If ``value`` is not an allowed member.
    """
    raw = string_value(value, path)
    try:
        return enum_type(raw)  # type: ignore[call-arg]
    except ValueError:
        allowed = sorted(str(item.value) for item in enum_type)  # type: ignore[attr-defined]
        fail(path, f"must be one of {allowed}")


def freeze_json(value: Any, path: str) -> JsonValue:
    """Validate and recursively freeze a JSON-compatible value.

    Args:
        value: Candidate JSON-compatible value.
        path: JSON path used in validation errors.

    Returns:
        An immutable JSON scalar, array, or key-sorted object.

    Raises:
        ContractValidationError: If ``value`` contains unsupported or
            nonfinite data.
    """
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            fail(path, "must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            fail(path, "object keys must be strings")
        return FrozenObject(tuple((key, freeze_json(value[key], f"{path}.{key}")) for key in sorted(value)))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return FrozenArray(tuple(freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value)))
    return fail(path, "must be JSON-compatible")


def unique(values: list[Any], path: str, label: str) -> None:
    """Reject duplicate values.

    Args:
        values: Values that must be unique.
        path: JSON path used in validation errors.
        label: Human-readable name for the values.

    Raises:
        ContractValidationError: If ``values`` contains a duplicate.
    """
    if len(values) != len(set(values)):
        fail(path, f"duplicate {label}")


def fail(path: str, reason: str) -> NoReturn:
    """Raise a stable path-addressed validation error.

    Args:
        path: JSON path at which validation failed.
        reason: Stable human-readable failure reason.

    Returns:
        This function never returns.

    Raises:
        ContractValidationError: Always, with ``path`` and ``reason``.
    """
    raise ContractValidationError(path, reason)
