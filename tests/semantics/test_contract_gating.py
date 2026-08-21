"""Contract-driven call binding, admission, and routing tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import pytest

from qceval.evals.evaluator import Evaluator, _admission_failure
from qceval.evals.models import ExecutionResult
from qceval.semantics.contracts import AuditStatus, call_args_from_signature, parse_contract
from qceval.semantics.contracts.kinds import ArgumentSpec, RouteSpec, SignatureSpec
from qceval.semantics.portfolio import DefaultSemanticVerifier, _routes_engine
from qceval.semantics.verifiers.exact import (
    PackagedClassicalTargetProvider,
    _expand_addition_relation,
    _expand_boolean_relation,
    _expand_subtraction_relation,
    _strip_prefix_x_wires,
)
from qceval.semantics.verifiers.result import SemanticStatus
from qceval.semantics.verifiers.symbolic import SYMBOLIC_COMPLETENESS


def _signature(*names: str) -> SignatureSpec:
    return SignatureSpec(
        entry_point="answer",
        arguments=tuple(ArgumentSpec(name, "any", "any", True) for name in names),
        return_type="framework_quantum_program",
    )


def test_call_args_from_signature_zero_one_and_many() -> None:
    assert call_args_from_signature(_signature(), None) == ()
    assert call_args_from_signature(_signature("unknown_state"), True) == (True,)
    assert call_args_from_signature(_signature("G", "beta", "gamma"), [1, 2, 3]) == (1, 2, 3)


def test_call_args_from_code_matches_source_arity() -> None:
    from qceval.semantics.contracts import call_args_from_code

    assert call_args_from_code("def answer():\n    return 1\n", "answer", [1, 2]) == ()
    assert call_args_from_code("def answer(x):\n    return x\n", "answer", True) == (True,)
    assert call_args_from_code(
        "def answer(a, b, c):\n    return a\n",
        "answer",
        [1, 2, 3],
    ) == (1, 2, 3)


def test_call_args_from_signature_rejects_arity_mismatch() -> None:
    with pytest.raises(ValueError, match="length 2"):
        call_args_from_signature(_signature("a", "b", "c"), [1, 2])
    with pytest.raises(ValueError, match="not a sequence"):
        call_args_from_signature(_signature("a", "b"), 7)
    with pytest.raises(ValueError, match="missing required"):
        call_args_from_signature(_signature("state"), None)


def test_call_args_from_signature_optional_arguments_bind_defaults() -> None:
    optional = SignatureSpec(
        entry_point="answer",
        arguments=(ArgumentSpec("n_count", "integer", "fixed_3", False),),
        return_type="framework_quantum_program",
    )
    assert call_args_from_signature(optional, None) == ()
    assert call_args_from_signature(optional, 4) == (4,)


def _minimal_contract_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema_version": "1",
        "suite": "core",
        "task_id": "99",
        "contract_version": "1.0.0",
        "kind": "state",
        "shadow_only": False,
        "audit_status": "reviewed",
        "signature": {"entry_point": "answer", "arguments": [], "return_type": "quantum_program"},
        "systems": {
            "items": [
                {"name": "output", "kind": "quantum", "role": "logical_output", "indices": [0], "dimension": 2},
            ]
        },
        "observation": {
            "quantum": ["output"],
            "classical": [],
            "ignored": [],
            "marginalize": [],
            "bit_order": "not_applicable",
            "postselection": None,
        },
        "phase": {"global_phase_irrelevant": True, "relative_phase": "preserve"},
        "ancillas": {"items": []},
        "parameters": {"items": [], "quantifier": "none", "completeness": None, "diagnostic_points": []},
        "approximation": {
            "mode": "exact",
            "metric": "trace_distance",
            "tolerance": 1e-9,
            "uncertainty": 1e-12,
            "error_budget": 0.0,
        },
        "target": {
            "id": "task_99",
            "version": "1.0.0",
            "sha256": "a" * 64,
            "source": "test",
            "manifest": "targets/core/99/manifest.json",
            "independent_derivations": 1,
        },
        "routing": {
            "primary": [
                {
                    "engine": "state_exact",
                    "capabilities": ["state", "framework_neutral_ir"],
                    "cross_check": False,
                }
            ],
            "fallback": [],
        },
        "limits": {
            "wall_seconds": 10.0,
            "cpu_seconds": 10.0,
            "memory_mib": 2048,
            "max_qubits": 8,
            "max_dimension": 256,
            "max_cases": 256,
            "max_branches": 256,
            "max_expression_nodes": 10000,
        },
        "requirements": [],
        "diagnostics": [],
    }
    payload.update(overrides)
    return payload


def test_admission_rejects_shadow_only_and_blocked_contracts() -> None:
    shadow = parse_contract(_minimal_contract_payload(shadow_only=True, audit_status="provisional"))
    blocked = parse_contract(_minimal_contract_payload(audit_status="blocked"))
    reviewed = parse_contract(_minimal_contract_payload())

    shadow_details = _admission_failure(shadow, "core", "qiskit")
    blocked_details = _admission_failure(blocked, "core", "qiskit")

    assert shadow_details is not None
    assert shadow_details["passed"] is False
    assert shadow_details["reason"] == "contract_shadow_only"
    assert blocked_details is not None
    assert blocked_details["reason"] == "contract_audit_blocked"
    assert _admission_failure(reviewed, "core", "qiskit") is None


def test_evaluator_grade_execution_fails_closed_for_shadow_only() -> None:
    contract = parse_contract(_minimal_contract_payload(shadow_only=True, audit_status="provisional"))
    registry = MagicMock()
    registry.get.return_value = contract
    evaluator = Evaluator(
        "qiskit",
        "core",
        {"99": {"entry_point": "answer", "canonical_class": {}}},
        {},
        semantic_verifier=MagicMock(),
    )
    evaluator._contracts = registry
    details = evaluator.grade_execution(
        task_id="99",
        execution=ExecutionResult(probabilities=[1.0, 0.0], metadata={}),
        code="def answer():\n    return None\n",
    )
    assert details["passed"] is False
    assert details["semantic_status"] == SemanticStatus.EXECUTION_ERROR.value
    assert details["reason"] == "contract_shadow_only"


def test_routes_engine_detects_symbolic_primary() -> None:
    contract = parse_contract(_minimal_contract_payload())
    contract = replace(
        contract,
        parameters=replace(
            contract.parameters,
            completeness=SYMBOLIC_COMPLETENESS,
        ),
        routing=replace(
            contract.routing,
            primary=(
                RouteSpec(
                    engine="symbolic_family_bounded",
                    capabilities=("static", "parameter_family"),
                    cross_check=True,
                ),
            ),
        ),
    )
    assert _routes_engine(contract, "symbolic_family_bounded")
    assert not _routes_engine(contract, "distribution_exact")


def test_source_result_honors_structured_completeness_without_task_id() -> None:
    verifier = DefaultSemanticVerifier()
    contract = parse_contract(_minimal_contract_payload())
    contract = replace(
        contract,
        parameters=replace(contract.parameters, completeness="structured_qaoa_source_identity"),
    )
    request = MagicMock()
    request.contract = contract
    request.code = "def answer():\n    return None\n"
    request.framework = "qiskit"
    request.execution = None
    request.arguments = ()
    request.cases = ()
    result = verifier._source_result(request, contract.parameters.completeness)
    assert result is not None
    assert result.status is not SemanticStatus.VERIFIED_PASS


def test_classical_input_wires_from_target_inputs_and_relations() -> None:
    from qceval.semantics.verifiers import exact as exact_mod

    assert exact_mod._input_wires_from_target({"type": "exhaustive_boolean_relation", "inputs": ["q0", "q1"]}) == (0, 1)
    assert exact_mod._input_wires_from_target({"type": "reversible_addition_relation", "operand_bits": 3}) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    )
    assert exact_mod._input_wires_from_target({"type": "reversible_subtraction_relation", "operand_bits": 3}) == (
        2,
        4,
        6,
    )
    assert exact_mod._input_wires_from_target({"inputs": ["ci", "bi", "ai"]}) == (0, 1, 2)


def test_classical_truth_tables_expand_from_target_type() -> None:
    or_rows = _expand_boolean_relation(
        {"type": "exhaustive_boolean_relation", "inputs": ["q0", "q1"], "output": "q0 OR q1"}
    )
    assert or_rows == (("00", "0"), ("01", "1"), ("10", "1"), ("11", "1"))

    parity = _expand_boolean_relation(
        {
            "type": "exhaustive_boolean_relation",
            "inputs": ["q0", "q1", "q2"],
            "output": "q0 XOR q1 XOR q2",
        }
    )
    assert parity[7] == ("111", "1")
    assert parity[0] == ("000", "0")

    majority = _expand_boolean_relation(
        {
            "type": "exhaustive_boolean_relation",
            "inputs": ["ci", "bi", "ai"],
            "outputs": ["majority(ai,bi,ci)", "ai XOR bi", "ai XOR ci"],
        }
    )
    assert majority[0] == ("000", "000")
    assert majority[7] == ("111", "100")

    addition = _expand_addition_relation({"operand_bits": 3})
    assert len(addition) == 128
    # carry=1, b=1, a=4 encoded as wires 0,1,6 set -> value bits.
    value = (1 << 0) | (1 << 1) | (1 << 6)
    assert addition[value][1] == f"{(4 + 1 + 1) % 16:04b}"

    subtraction = _expand_subtraction_relation({"operand_bits": 3, "prompt_witness": {"b": 3}})
    assert subtraction[7] == ("111", "100")


def test_strip_prefix_x_wires_from_prompt_witness() -> None:
    assert _strip_prefix_x_wires(
        {
            "type": "reversible_addition_relation",
            "operand_bits": 3,
            "prompt_witness": {"a": 4, "b": 1, "carry_in": 1},
        }
    ) == frozenset({0, 1, 6})
    assert _strip_prefix_x_wires(
        {
            "type": "reversible_subtraction_relation",
            "operand_bits": 3,
            "prompt_witness": {"a": 7, "b": 3},
        }
    ) == frozenset({2, 4, 6})
    assert (
        _strip_prefix_x_wires(
            {
                "type": "reversible_subtraction_relation",
                "operand_bits": 3,
                "prompt_witness": {"a": 7, "b": 3},
                "strip_prefix_x_wires": [],
            }
        )
        == frozenset()
    )


def test_audit_status_enum_includes_blocked() -> None:
    assert AuditStatus.BLOCKED.value == "blocked"


def test_audit_blocker_requirement_fails_closed() -> None:
    from qceval.semantics.contracts.kinds import RequirementSpec
    from qceval.semantics.ir import Program, Provenance
    from qceval.semantics.verifiers.requirements import verify_program_requirements

    contract = parse_contract(_minimal_contract_payload())
    contract = replace(
        contract,
        requirements=(RequirementSpec("audit_blocker", "blocking_semantic_question", "audit", {"note": "blocked"}),),
    )
    program = Program(
        ir_version="1",
        num_qubits=1,
        num_clbits=0,
        operations=(),
        global_phase=None,
        classical_render_order=(),
        provenance=Provenance(framework="qiskit", framework_version="0", source_hash="abc", backend=None),
    )
    failure = verify_program_requirements(
        contract,
        program,
        framework="qiskit",
        execution_metadata={},
    )
    assert failure is not None
    assert failure.status is SemanticStatus.SEMANTIC_FAIL
    assert failure.reason == "audit_blocker"


def test_packaged_classical_provider_dispatches_on_type(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = PackagedClassicalTargetProvider()
    context = MagicMock()
    target = {
        "type": "exhaustive_boolean_relation",
        "inputs": ["q0", "q1"],
        "output": "q0 OR q1",
    }
    monkeypatch.setattr(
        "qceval.semantics.verifiers.exact._packaged_target",
        lambda _context: target,
    )
    table = provider.classical_table(context)
    assert table.rows[0] == ("00", "0")
    assert table.rows[-1] == ("11", "1")
