"""Packaged finite classical target expansion for exact I/O verification."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from qceval.semantics.contracts.kinds import SystemRole
from qceval.semantics.targets import load_contract_target_document
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.classical_wires import contracted_classical_variables, measured_render_order
from qceval.semantics.verifiers.materialize import ArrayMaterialization, ClassicalTableMaterialization


class PackagedClassicalTargetProvider:
    """Expand hash-verified packaged finite classical target specs."""

    def classical_table(self, context: VerificationContext) -> ClassicalTableMaterialization:
        """Load and expand one target truth table.

        Args:
            context: Contract identifying the packaged target artifact.

        Returns:
            Complete target input/output table.
        """
        # Resolve through the package facade so monkeypatches on
        # ``verifiers.exact._packaged_target`` remain effective.
        from qceval.semantics.verifiers.exact import _packaged_target as load_target

        target = load_target(context)
        target_type = target.get("type")
        if target_type == "exhaustive_boolean_relation":
            return ClassicalTableMaterialization(_expand_boolean_relation(target))
        if target_type == "reversible_addition_relation":
            return ClassicalTableMaterialization(_expand_addition_relation(target))
        if target_type == "reversible_subtraction_relation":
            return ClassicalTableMaterialization(_expand_subtraction_relation(target))
        raise NotImplementedError(f"no packaged classical target for type {target_type!r}")

    def array(self, context: VerificationContext, representation: str) -> ArrayMaterialization:
        """Reject array requests at the classical-only target seam.

        Args:
            context: Verification context, unused by this route.
            representation: Requested dense representation.

        Returns:
            This method never returns a value.
        """
        del context
        raise NotImplementedError(f"classical target provider does not support {representation!r}")


def _packaged_target(context: VerificationContext) -> dict[str, object]:
    value = load_contract_target_document(context.contract)
    target = value.get("target")
    if not isinstance(target, dict):
        raise ValueError("classical target is missing")
    return target


def _classical_input_wires(context: VerificationContext) -> tuple[int, ...]:
    target = _packaged_target(context)
    wires = _input_wires_from_target(target)
    if wires is not None:
        return wires
    systems = tuple(
        system
        for system in context.contract.systems.items
        if system.role in {SystemRole.CLASSICAL_INPUT, SystemRole.CLASSICAL_IO, SystemRole.LOGICAL_INPUT}
    )
    if systems:
        return tuple(index for system in systems for index in system.indices)
    raise NotImplementedError("classical input domain is not declared on the packaged target or systems")


def _input_wires_from_target(target: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw = target.get("input_wires")
    if isinstance(raw, list | tuple) and all(isinstance(item, int) for item in raw):
        return tuple(raw)
    inputs = target.get("inputs")
    if isinstance(inputs, list | tuple) and inputs:
        qubit_wires = []
        for name in inputs:
            match = re.fullmatch(r"q(\d+)", str(name))
            if match is None:
                return tuple(range(len(inputs)))
            qubit_wires.append(int(match.group(1)))
        return tuple(qubit_wires)
    target_type = target.get("type")
    bits = target.get("operand_bits")
    if target_type == "reversible_addition_relation" and isinstance(bits, int) and not isinstance(bits, bool):
        # Carry bit plus interleaved (b_i, a_i) operand pairs.
        return tuple(range(1 + 2 * bits))
    if target_type == "reversible_subtraction_relation" and isinstance(bits, int) and not isinstance(bits, bool):
        # Operand ``a`` occupies the even wires after an unused low bit slot.
        return tuple(range(2, 2 * bits + 1, 2))
    return None


def _classical_output_bits(context: VerificationContext) -> tuple[int, ...]:
    systems = {item.name: item for item in context.contract.systems.items}
    classical_names = context.contract.observation.classical
    if classical_names:
        del systems
        bindings = contracted_classical_variables(
            context.contract,
            context.program,
            classical_names,
            require_all=True,
        )
        return tuple(bit for _, bit in bindings)
    indices = measured_render_order(context.program)
    if not indices or max(indices) >= context.program.num_clbits:
        raise NotImplementedError("declared classical outputs are absent from Program IR")
    return indices


def _strip_prefix_x_wires(target: Mapping[str, Any]) -> frozenset[int]:
    raw = target.get("strip_prefix_x_wires")
    if isinstance(raw, list | tuple) and all(isinstance(item, int) for item in raw):
        return frozenset(raw)
    witness = target.get("prompt_witness")
    bits = target.get("operand_bits")
    if not isinstance(witness, Mapping) or not isinstance(bits, int) or isinstance(bits, bool):
        return frozenset()
    target_type = target.get("type")
    if target_type == "reversible_addition_relation":
        wires: set[int] = set()
        if int(witness.get("carry_in", 0)):
            wires.add(0)
        b_value = int(witness.get("b", 0))
        a_value = int(witness.get("a", 0))
        for index in range(bits):
            if (b_value >> index) & 1:
                wires.add(2 * index + 1)
            if (a_value >> index) & 1:
                wires.add(2 * index + 2)
        return frozenset(wires)
    if target_type == "reversible_subtraction_relation":
        a_value = int(witness.get("a", 0))
        return frozenset(2 * index + 2 for index in range(bits) if (a_value >> index) & 1)
    return frozenset()


def _expand_boolean_relation(target: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    inputs = target.get("inputs")
    if not isinstance(inputs, list | tuple) or not inputs:
        raise NotImplementedError("boolean relation target is missing inputs")
    width = len(inputs)
    outputs = target.get("outputs")
    if isinstance(outputs, list | tuple) and outputs:
        rows = []
        for value in range(2**width):
            binding = {str(name): (value >> position) & 1 for position, name in enumerate(inputs)}
            bits = "".join(str(_eval_boolean_formula(str(formula), binding)) for formula in outputs)
            rows.append((f"{value:0{width}b}", bits))
        return tuple(rows)
    output = target.get("output")
    if not isinstance(output, str):
        raise NotImplementedError("boolean relation target is missing output formula")
    rows = []
    for value in range(2**width):
        binding = {str(name): (value >> position) & 1 for position, name in enumerate(inputs)}
        rows.append((f"{value:0{width}b}", str(_eval_boolean_formula(output, binding))))
    return tuple(rows)


def _eval_boolean_formula(formula: str, binding: Mapping[str, int]) -> int:
    text = formula.strip().lower()
    majority = re.fullmatch(r"majority\((.+)\)", text)
    if majority is not None:
        names = [part.strip() for part in majority.group(1).split(",")]
        return int(sum(binding[name] for name in names) >= (len(names) + 1) // 2)
    if " xor " in text:
        names = [part.strip() for part in text.split(" xor ")]
        value = 0
        for name in names:
            value ^= binding[name]
        return value
    if " or " in text:
        names = [part.strip() for part in text.split(" or ")]
        return int(any(binding[name] for name in names))
    if " and " in text:
        names = [part.strip() for part in text.split(" and ")]
        return int(all(binding[name] for name in names))
    if text in binding:
        return binding[text]
    raise NotImplementedError(f"unsupported boolean formula: {formula}")


def _expand_addition_relation(target: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    bits = target.get("operand_bits")
    if not isinstance(bits, int) or isinstance(bits, bool):
        raise NotImplementedError("addition relation target is missing operand_bits")
    width = 1 + 2 * bits
    modulus = 2 ** (bits + 1)
    rows = []
    for value in range(2**width):
        carry = (value >> 0) & 1
        b_value = sum(((value >> (2 * index + 1)) & 1) << index for index in range(bits))
        a_value = sum(((value >> (2 * index + 2)) & 1) << index for index in range(bits))
        rows.append((f"{value:0{width}b}", f"{(a_value + b_value + carry) % modulus:0{bits + 1}b}"))
    return tuple(rows)


def _expand_subtraction_relation(target: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    bits = target.get("operand_bits")
    witness = target.get("prompt_witness")
    if not isinstance(bits, int) or isinstance(bits, bool) or not isinstance(witness, Mapping):
        raise NotImplementedError("subtraction relation target is missing operand_bits/prompt_witness")
    b_value = int(witness["b"])
    modulus = 2**bits
    return tuple((f"{value:0{bits}b}", f"{(value - b_value) % modulus:0{bits}b}") for value in range(modulus))
