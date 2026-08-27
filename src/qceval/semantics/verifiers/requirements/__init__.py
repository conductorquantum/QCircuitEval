"""Executable hard requirements derived from audited semantic contracts."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from qceval.semantics.contracts import Contract, contract_hash
from qceval.semantics.contracts.kinds import FrozenArray, FrozenObject
from qceval.semantics.ir import Program, program_hash
from qceval.semantics.verifiers.requirements.cases import case_program_invariance_violation
from qceval.semantics.verifiers.requirements.gate_family import _gate_basis_violation
from qceval.semantics.verifiers.requirements.semantic import _semantic_violation
from qceval.semantics.verifiers.requirements.structural import (
    _framework_structural_policy,
    _measurement_exclusion_violation,
    _native_ir_semantic_violation,
    _selected_interface,
    _structural_violation,
    _terminal_violation,
)
from qceval.semantics.verifiers.result import (
    RESULT_SCHEMA_VERSION,
    EvidenceRecord,
    SemanticStatus,
    VerifierResult,
)

REQUIREMENTS_VERSION = "1.1.0"


def verify_program_requirements(
    contract: Contract,
    program: Program,
    *,
    framework: str,
    execution_metadata: Mapping[str, Any],
    source_code: str | None = None,
    candidate_unitary: Any | None = None,
    arguments: tuple[Any, ...] = (),
) -> VerifierResult | None:
    """Return a semantic failure for the first violated hard requirement.

    Args:
        contract: Audited behavior contract.
        program: Lowered framework-neutral program.
        framework: Candidate framework identifier.
        execution_metadata: Executor observations used only by declared rules.
        source_code: Optional candidate source for explicit anti-shortcut rules.
        candidate_unitary: Optional net unitary for nonlocality verification.
        arguments: Concrete positional arguments for this exhaustive-domain case.

    Returns:
        First hard-requirement failure, or ``None`` when all checks pass.
    """

    reason = _entry_point_signature_violation(contract, source_code)
    if reason is None:
        reason = _first_violation(
            contract,
            program,
            framework,
            execution_metadata,
            source_code,
            candidate_unitary,
            arguments,
        )
    if reason is not None:
        return _failure(contract, program, reason)
    return None


def verify_case_program_requirements(
    contract: Contract,
    cases: tuple[tuple[tuple[Any, ...], Program], ...],
) -> VerifierResult | None:
    """Return a failure when exhaustive cases synthesize argument answers.

    Args:
        contract: Audited behavior contract.
        cases: Concrete arguments paired with their lowered Program IR.

    Returns:
        Cross-case hard-requirement failure, or ``None`` when invariant.
    """
    reason = case_program_invariance_violation(contract, cases)
    if reason is None:
        return None
    if not cases:
        raise ValueError("case program verification requires at least one case")
    return _failure(contract, cases[0][1], reason)


def _first_violation(
    contract: Contract,
    program: Program,
    framework: str,
    execution_metadata: Mapping[str, Any],
    source_code: str | None,
    candidate_unitary: Any | None,
    arguments: tuple[Any, ...],
) -> str | None:
    if _audit_blocker_violation(contract) is not None:
        return "audit_blocker"
    values = {item.requirement_id: _plain(item.value) for item in contract.requirements}
    reason = _intrinsic_program_violation(
        contract,
        values,
        program,
        framework,
        execution_metadata,
        candidate_unitary,
    )
    if reason is not None:
        return reason
    structural = values.get("structural_constraints")
    if isinstance(structural, Mapping):
        reason = _structural_violation(
            contract,
            program,
            _framework_structural_policy(structural, framework),
            framework,
            execution_metadata,
            source_code,
            candidate_unitary,
        )
        if reason is not None:
            return reason
    basis = values.get("gate_basis")
    if isinstance(basis, Mapping):
        reason = _gate_basis_violation(program, basis)
        if reason is not None:
            return reason
    reason = _first_measurement_exclusion_violation(contract, program)
    if reason is not None:
        return reason
    semantics = values.get("semantic_requirements")
    if isinstance(semantics, Mapping):
        return _semantic_violation(
            contract,
            program,
            semantics,
            framework,
            execution_metadata,
            source_code,
            arguments,
        )
    return None


def _intrinsic_program_violation(
    contract: Contract,
    values: Mapping[str, Any],
    program: Program,
    framework: str,
    execution_metadata: Mapping[str, Any],
    candidate_unitary: Any | None,
) -> str | None:
    terminal = _terminal_observation_violation(values, program, framework, execution_metadata)
    if terminal is not None:
        return terminal
    return _native_ir_semantic_violation(contract, program, framework, candidate_unitary)


def _entry_point_signature_violation(contract: Contract, source_code: str | None) -> str | None:
    """Reject entry points whose declared parameters diverge from the contract.

    Every prompt instructs candidates to preserve the published entry-point
    signature. The check compares positional parameter names in declaration
    order; annotations and default values stay behavior-neutral because the
    grader always binds arguments positionally, so they are not rejected.
    Sources without a parseable matching definition are left to the executor
    and binding layers, which fail closed on arity mismatches.
    """
    if source_code is None:
        return None
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None
    definition = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == contract.signature.entry_point
        ),
        None,
    )
    if definition is None:
        return None
    expected = [argument.name for argument in contract.signature.arguments]
    declared = [argument.arg for argument in (*definition.args.posonlyargs, *definition.args.args)]
    if declared != expected or definition.args.vararg is not None or definition.args.kwarg is not None:
        return "requirement_failed:entry_point_signature"
    return None


def _audit_blocker_violation(contract: Contract) -> str | None:
    for requirement in contract.requirements:
        if requirement.requirement_id == "audit_blocker" or requirement.kind == "blocking_semantic_question":
            return "audit_blocker"
    return None


def _terminal_observation_violation(
    values: Mapping[str, Any],
    program: Program,
    framework: str,
    execution_metadata: Mapping[str, Any],
) -> str | None:
    terminal = values.get("terminal_observation")
    if not isinstance(terminal, Mapping):
        return None
    interface = terminal.get(framework)
    if not isinstance(interface, Mapping):
        return None
    return _terminal_violation(program, interface, execution_metadata)


def _first_measurement_exclusion_violation(contract: Contract, program: Program) -> str | None:
    for requirement in contract.requirements:
        if requirement.kind != "measurement_exclusion":
            continue
        exclusion = _plain(requirement.value)
        if isinstance(exclusion, Mapping):
            reason = _measurement_exclusion_violation(program, exclusion)
            if reason is not None:
                return reason
    return None


def conditional_quantum_wires(contract: Contract, program: Program) -> tuple[int, ...] | None:
    """Return framework-specific conditional output wires when declared.

    Args:
        contract: Audited instrument contract.
        program: Lowered framework-neutral program.

    Returns:
        Conditional quantum wires, or ``None`` when the contract omits them.
    """

    for requirement in contract.requirements:
        if requirement.requirement_id != "terminal_observation":
            continue
        value = _plain(requirement.value)
        if not isinstance(value, Mapping):
            return None
        interface = value.get(program.provenance.framework)
        if not isinstance(interface, Mapping):
            return None
        selected = _selected_interface(program, interface)
        if selected is None or "conditional_qubits" not in selected:
            return None
        wires = selected["conditional_qubits"]
        if not isinstance(wires, list | tuple) or not all(isinstance(item, int) for item in wires):
            return None
        return tuple(wires)
    return None


def _plain(value: Any) -> Any:
    if isinstance(value, FrozenArray):
        return [_plain(item) for item in value.items]
    if isinstance(value, FrozenObject):
        return {key: _plain(item) for key, item in value.items}
    return value


def _failure(contract: Contract, program: Program, reason: str) -> VerifierResult:
    evidence = EvidenceRecord(
        "contract_requirements",
        REQUIREMENTS_VERSION,
        reason,
        program_hash(program),
        contract.target.sha256,
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        SemanticStatus.SEMANTIC_FAIL,
        reason,
        contract_hash(contract),
        contract.target.sha256,
        REQUIREMENTS_VERSION,
        (evidence,),
    )


__all__ = [
    "conditional_quantum_wires",
    "verify_case_program_requirements",
    "verify_program_requirements",
]
