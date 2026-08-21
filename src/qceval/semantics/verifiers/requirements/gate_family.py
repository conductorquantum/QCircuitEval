"""Gate-family aliases, operation classification, and gate-basis checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from qceval.semantics.ir import OperationKind, Program

_GATE_FAMILY_ALIASES: dict[str, tuple[str, int]] = {
    "cnot": ("x", 1),
    "controlled_not": ("x", 1),
    "cx": ("x", 1),
    "ccx": ("x", 2),
    "toffoli": ("x", 2),
    "mcx": ("x", 2),
    "cz": ("z", 1),
    "ccz": ("z", 2),
    "ch": ("h", 1),
    "controlled_h": ("h", 1),
    "cy": ("y", 1),
    "cswap": ("swap", 1),
    "fredkin": ("swap", 1),
}


def _gate_family(name: str, extra_controls: int) -> tuple[str, int]:
    base, implied = _GATE_FAMILY_ALIASES.get(name.lower(), (name.lower(), 0))
    # Adapters differ on whether a controlled name keeps its controls inline
    # ("cz" on two wires) or externalizes them ("cz" naming plus an explicit
    # control entry); take the larger interpretation instead of summing.
    return base, max(implied, extra_controls)


_POW_FAMILY_BASES = {"xpow": "x", "ypow": "y", "zpow": "z"}

_POW_HALF_FAMILIES = {"xpow": ("sx", "sxdg")}


def _operation_family(operation: Any) -> tuple[str, int]:
    name = operation.name.lower()
    if name == "rx":
        return _rx_family(operation), len(operation.controls)
    if name in _POW_FAMILY_BASES:
        return _pow_family(name, operation), len(operation.controls)
    return _gate_family(name, len(operation.controls))


def _rx_family(operation: Any) -> str:
    angle = _numeric_gate_parameter(operation)
    if angle is None:
        angle = _rx_angle_from_matrix(operation)
    if angle is not None:
        normalized = angle % (2 * math.pi)
        if abs(normalized - math.pi / 2) < 1e-12:
            return "sx"
        if abs(normalized - 3 * math.pi / 2) < 1e-12:
            return "sxdg"
    return "rx"


def _pow_family(name: str, operation: Any) -> str:
    exponent = _numeric_gate_parameter(operation)
    half = _POW_HALF_FAMILIES.get(name)
    if half is not None and exponent is not None:
        if exponent % 2 == 0.5:
            return half[0]
        if exponent % 2 == 1.5:
            return half[1]
    if exponent is not None and exponent % 2 == 1.0:
        return _POW_FAMILY_BASES[name]
    return name


def _numeric_gate_parameter(operation: Any) -> float | None:
    if not operation.parameters:
        return None
    parameter = operation.parameters[0]
    try:
        return float(parameter.value)
    except (TypeError, ValueError):
        return None


def _rx_angle_from_matrix(operation: Any) -> float | None:
    """Recover an RX rotation angle from an attached exact matrix payload."""
    payload = dict(operation.semantic_data).get("matrix_complex128_hex")
    if not isinstance(payload, str):
        return None
    try:
        import numpy as np

        matrix = np.frombuffer(bytes.fromhex(payload), dtype=np.complex128).reshape(2, 2)
    except ValueError:
        return None
    # RX(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
    if abs(matrix[0, 0] - matrix[1, 1]) > 1e-12 or abs(matrix[0, 1] - matrix[1, 0]) > 1e-12:
        return None
    cosine = matrix[0, 0]
    sine = matrix[0, 1] * 1j
    if abs(cosine.imag) > 1e-12 or abs(sine.imag) > 1e-12:
        return None
    return 2.0 * math.atan2(sine.real, cosine.real)


_CLIFFORD_STATIC_FAMILIES = (
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "sx",
    "sxdg",
    "cy",
    "cz",
    "swap",
    "iswap",
    "ecr",
    "dcx",
)

_CLIFFORD_ROTATION_NAMES = frozenset({"rz", "rx", "ry", "p", "phase", "r1", "u1"})

_CLIFFORD_POW_NAMES = frozenset({"xpow", "ypow", "zpow"})

_CLIFFORD_ANGLE_TOLERANCE = 1e-12


def _clifford_gate_class_violation(program: Program) -> str | None:
    """Reject any gate outside the Clifford group, angle-aware for rotations.

    Named Clifford families are accepted as before. Uncontrolled single-qubit
    rotation gates (``rz``/``rx``/``ry``, phase gates, and cirq power gates)
    additionally count as Clifford exactly when their angle is a multiple of
    pi/2 (exponent multiple of one half), since such rotations equal S/X/Z
    powers up to global phase. Any other angle remains forbidden, and
    controlled rotations stay outside the allow-list because a controlled
    quarter-turn is not Clifford.
    """
    allowed = {_gate_family(name, 0) for name in _CLIFFORD_STATIC_FAMILIES}
    for operation in program.operations:
        if operation.kind is not OperationKind.GATE or operation.name.lower() in {"id", "i", "barrier"}:
            continue
        families = {_operation_family(operation), _gate_family(operation.name, len(operation.controls))}
        if families & allowed or _is_clifford_rotation(operation):
            continue
        return f"forbidden_gate_family:{sorted(families)[0][0]}"
    return None


def _is_clifford_rotation(operation: Any) -> bool:
    if operation.controls or len(operation.quantum_wires) != 1 or operation.power is not None:
        return False
    name = operation.name.lower()
    parameter = _numeric_gate_parameter(operation)
    if parameter is None or len(operation.parameters) != 1:
        return False
    if name in _CLIFFORD_ROTATION_NAMES:
        return _is_quarter_turn(parameter, math.pi / 2)
    if name in _CLIFFORD_POW_NAMES:
        return _is_quarter_turn(parameter, 0.5)
    return False


def _is_quarter_turn(value: float, unit: float) -> bool:
    remainder = math.fmod(value, unit)
    return min(abs(remainder), unit - abs(remainder)) <= _CLIFFORD_ANGLE_TOLERANCE


def _gate_basis_violation(program: Program, basis: Mapping[str, Any]) -> str | None:
    # Each operation matches both its specialized family (rx(pi/2) -> sx) and
    # its raw named family (rx), so "allowed rx" admits rx at any angle while
    # "forbidden sx" still catches sx smuggled in as a fixed rotation.
    members: list[set[tuple[str, int]]] = []
    for operation in program.operations:
        if operation.kind is not OperationKind.GATE or operation.name.lower() in {"id", "i", "barrier"}:
            continue
        members.append({_operation_family(operation), _gate_family(operation.name, len(operation.controls))})
    for label, family in _declared_families(basis.get("forbidden")):
        if any(family in families for families in members):
            return f"forbidden_gate_family:{label.lower()}"
    allowed = {family for _, family in _declared_families(basis.get("allowed"))}
    if allowed:
        for families in members:
            if not families & allowed:
                return f"forbidden_gate_family:{sorted(families)[0][0]}"
    present = {family for families in members for family in families}
    return _missing_required_family(basis.get("required"), present)


def _declared_families(value: Any) -> list[tuple[str, tuple[str, int]]]:
    if not isinstance(value, list | tuple):
        return []
    return [(str(item), _gate_family(str(item), 0)) for item in value if isinstance(item, str)]


def _missing_required_family(required: Any, present: set[tuple[str, int]]) -> str | None:
    if not isinstance(required, list | tuple):
        return None
    for item in required:
        if not isinstance(item, str):
            continue
        options = {_gate_family(option, 0) for option in item.split("_or_")}
        if not options & present:
            return f"missing_gate_family:{item.lower()}"
    return None


def _family_label(family: tuple[str, int]) -> str:
    base, controls = family
    return base if controls == 0 else f"{'c' * controls}{base}"


def _operation_wires(operation: Any) -> frozenset[int]:
    return frozenset((*operation.quantum_wires, *(control.wire for control in operation.controls)))


def _family_count_violation(policy: Mapping[str, Any], program: Program) -> str | None:
    counts = _combined_family_counts(program)
    checks = (
        _minimum_family_violation,
        _minimum_family_group_violation,
        _minimum_family_alternative_violation,
        _forbidden_family_violation,
    )
    for check in checks:
        reason = check(policy, counts)
        if reason is not None:
            return reason
    return None


def _minimum_family_violation(
    policy: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str | None:
    minimums = policy.get("min_gate_family_counts")
    if not isinstance(minimums, Mapping):
        return None
    for label, minimum in minimums.items():
        if counts.get(str(label).lower(), 0) < int(minimum):
            return f"requirement_failed:min_gate_family_counts.{str(label).lower()}"
    return None


def _minimum_family_group_violation(
    policy: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str | None:
    groups = policy.get("min_gate_family_group_counts")
    if not isinstance(groups, Mapping):
        return None
    for label, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        families = group.get("families", [])
        minimum = int(group.get("min", 1))
        observed = sum(counts.get(str(family).lower(), 0) for family in families)
        if observed < minimum:
            return f"requirement_failed:min_gate_family_group_counts.{str(label).lower()}"
    return None


def _minimum_family_alternative_violation(
    policy: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str | None:
    alternatives = policy.get("min_any_gate_family_counts")
    if not isinstance(alternatives, Mapping):
        return None
    for label, options in alternatives.items():
        if not _any_family_alternative_satisfied(options, counts):
            return f"requirement_failed:min_any_gate_family_counts.{str(label).lower()}"
    return None


def _forbidden_family_violation(
    policy: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str | None:
    forbidden = policy.get("forbidden_gate_family_counts")
    if not isinstance(forbidden, Mapping):
        return None
    for label, maximum in forbidden.items():
        if counts.get(str(label).lower(), 0) > int(maximum):
            return f"forbidden_gate_family:{str(label).lower()}"
    return None


def _combined_family_counts(program: Program) -> dict[str, int]:
    counts: dict[str, int] = {}
    for operation in program.operations:
        if operation.kind is not OperationKind.GATE:
            continue
        raw = operation.name.lower()
        family = _operation_family(operation)
        normalized = _family_label(family)
        for label in {raw, normalized}:
            counts[label] = counts.get(label, 0) + 1
        if len(_operation_wires(operation)) >= 2:
            counts["native_entangler"] = counts.get("native_entangler", 0) + 1
        if family == ("x", 1):
            counts["controlled_not"] = counts.get("controlled_not", 0) + 1
        if family[1] >= 2:
            counts["multi_control"] = counts.get("multi_control", 0) + 1
        if family[1] >= 1 and family[0] in {"p", "phase", "rz", "z"}:
            counts["controlled_phase"] = counts.get("controlled_phase", 0) + 1
    return counts


def _any_family_alternative_satisfied(
    alternatives: Any,
    counts: Mapping[str, int],
) -> bool:
    if not isinstance(alternatives, Sequence) or isinstance(alternatives, str | bytes):
        return False
    return any(
        isinstance(option, Mapping)
        and all(counts.get(str(family).lower(), 0) >= int(minimum) for family, minimum in option.items())
        for option in alternatives
    )
