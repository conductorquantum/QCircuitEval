"""Cirq qubit layout mapping onto contiguous Program IR wires."""

from __future__ import annotations

from typing import Any

from qceval.semantics.contracts import Contract, contract_to_dict


class _Unsupported(Exception):
    """Typed Cirq lowering limitation that becomes a non-verdict failure."""

    def __init__(self, reason: str, node_kind: str, location: str) -> None:
        self.reason = reason
        self.node_kind = node_kind
        self.location = location
        super().__init__(reason)


def _wire_map(qubits: list[Any], contract: Contract | None) -> dict[Any, int]:
    """Map Cirq qubits onto non-negative IR wire indices.

    Args:
        qubits: Sorted Cirq qubit objects from the circuit.
        contract: Optional contract whose qubit limit and semantic roles apply.

    Returns:
        Mapping from each qubit object to its IR wire index.

    Raises:
        _Unsupported: When the layout is ambiguous or exceeds the contract.
    """
    maximum = contract.limits.max_qubits if contract is not None else None
    role_values = _semantic_role_wire_map(qubits, contract)
    values = role_values if role_values is not None else _native_wire_values(qubits, maximum)
    if len(set(values)) != len(values) or any(value < 0 for value in values):
        raise _Unsupported("ambiguous_qubit_layout", "qubit", "circuit.all_qubits")
    if maximum is not None and any(value >= maximum for value in values):
        raise _Unsupported("qubit_layout_exceeds_contract", "qubit", "circuit.all_qubits")
    return dict(zip(qubits, values, strict=True))


def _native_wire_values(qubits: list[Any], maximum: int | None) -> list[int]:
    import cirq

    if all(isinstance(qubit, cirq.LineQubit) for qubit in qubits):
        return [int(qubit.x) for qubit in qubits]
    if all(isinstance(qubit, cirq.GridQubit) for qubit in qubits):
        if qubits and all(qubit.row == qubits[0].row for qubit in qubits):
            return [int(qubit.col) for qubit in qubits]
        if maximum is not None and len(qubits) == maximum:
            return list(range(len(qubits)))
        raise _Unsupported("explicit_qubit_layout_required", "GridQubit", "circuit.all_qubits")
    if all(isinstance(qubit, cirq.NamedQubit) for qubit in qubits):
        suffixes = ["".join(character for character in qubit.name if character.isdigit()) for qubit in qubits]
        if not all(suffixes):
            raise _Unsupported("explicit_qubit_layout_required", "NamedQubit", "circuit.all_qubits")
        return [int(value) for value in suffixes]
    raise _Unsupported("explicit_qubit_layout_required", "qubit", "circuit.all_qubits")


def _semantic_role_wire_map(qubits: list[Any], contract: Contract | None) -> list[int] | None:
    if contract is None:
        return None
    names = [str(getattr(qubit, "name", "")).lower() for qubit in qubits]
    requirements = {item["id"]: item["value"] for item in contract_to_dict(contract)["requirements"]}
    semantics = requirements.get("semantic_requirements", {})
    algorithm = semantics.get("algorithm") if isinstance(semantics, dict) else None
    if algorithm == "swap_test":
        return _swap_test_role_wires(names)
    if algorithm == "three_individual_swap_tests":
        return _individual_swap_role_wires(names)
    fixed = _fixed_role_wires(names)
    if fixed is not None or algorithm != "reversible_quantum_adder":
        return fixed
    return _adder_role_wires(names)


def _swap_test_role_wires(names: list[str]) -> list[int] | None:
    values = [0 if "anc" in name else 2 if any(token in name for token in ("zero", "ref")) else 1 for name in names]
    return values if len(set(values)) == len(values) else None


def _individual_swap_role_wires(names: list[str]) -> list[int] | None:
    offsets = {"q1_": 0, "q2_": 3, "anc_": 6}
    values = []
    for name in names:
        prefix = next((value for value in offsets if name.startswith(value)), None)
        suffix = _numeric_suffix(name)
        if prefix is None or suffix is None:
            return None
        values.append(offsets[prefix] + suffix)
    return values


def _fixed_role_wires(names: list[str]) -> list[int] | None:
    mappings = (
        {"c_i": 0, "b_i": 1, "a_i": 2},
        {"control": 0, "target": 1},
    )
    mapping = next((value for value in mappings if set(names) == set(value)), None)
    return None if mapping is None else [mapping[name] for name in names]


def _adder_role_wires(names: list[str]) -> list[int] | None:
    compact = [name.replace("_", "") for name in names]
    mapping = {"c0": 0, "b0": 1, "a0": 2, "b1": 3, "a1": 4, "b2": 5, "a2": 6, "c1": 8, "c2": 9}
    if set(names) == {*mapping, "s3"}:
        return [{**mapping, "s3": 7}[name] for name in names]
    if set(compact) == {*mapping, "c3"}:
        return [{**mapping, "c3": 7}[name] for name in compact]
    return None


def _numeric_suffix(value: str) -> int | None:
    digits = ""
    for character in reversed(value):
        if not character.isdigit():
            break
        digits = character + digits
    return int(digits) if digits else None
