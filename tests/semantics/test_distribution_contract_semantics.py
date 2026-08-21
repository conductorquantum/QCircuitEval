"""Regressions for contract-driven distribution and instrument semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qceval.semantics.contracts import ContractRegistry
from qceval.semantics.contracts.kinds import (
    BitOrder,
    ObservationSpec,
    SystemKind,
    SystemRole,
    SystemSpec,
    SystemsSpec,
)
from qceval.semantics.ir import IR_VERSION, Operation, OperationKind, Program, Provenance
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.distribution_engine import DistributionEngine
from qceval.semantics.verifiers.distribution_materializers import (
    ExecutionDistributionMaterializer,
    ProbabilityTable,
    ProgramDistributionMaterializer,
)
from qceval.semantics.verifiers.instrument import InstrumentEngine
from qceval.semantics.verifiers.materialize import CandidateSemanticError
from qceval.semantics.verifiers.result import SemanticStatus


def _program(operations: tuple[Operation, ...], num_qubits: int, num_clbits: int, render: tuple[int, ...]) -> Program:
    return Program(
        IR_VERSION,
        num_qubits,
        num_clbits,
        operations,
        None,
        render,
        Provenance("test", "1", source_hash="a" * 64),
    )


def _context(contract, program: Program, probabilities: tuple[float, ...] | None = None) -> VerificationContext:
    return VerificationContext(contract, "contract", "target", "input", program, execution_probabilities=probabilities)


def test_declared_marginalized_variable_reaches_projection() -> None:
    base = ContractRegistry.from_package("core").get("core", "01")
    contract = replace(
        base,
        systems=SystemsSpec(
            (
                SystemSpec("output", SystemKind.CLASSICAL, SystemRole.CLASSICAL_OUTPUT, (0,), 2),
                SystemSpec("scratch", SystemKind.CLASSICAL, SystemRole.CLASSICAL_OUTPUT, (1,), 2),
            )
        ),
        observation=ObservationSpec((), ("output",), (), ("scratch",), BitOrder.BIG_ENDIAN, None),
    )
    program = _program(
        (
            Operation(OperationKind.GATE, "h", quantum_wires=(0,)),
            Operation(OperationKind.GATE, "x", quantum_wires=(1,)),
            Operation(OperationKind.MEASUREMENT, "measure", (0,), (0,)),
            Operation(OperationKind.MEASUREMENT, "measure", (1,), (1,)),
        ),
        2,
        2,
        (1, 0),
    )
    context = _context(contract, program)
    table = ProgramDistributionMaterializer().distribution(context)

    assert table.variables == ("output[0]", "scratch[1]")
    assert dict(table.rows) == {
        ("0", "1"): pytest.approx(0.5),
        ("1", "1"): pytest.approx(0.5),
    }

    class Target:
        def distribution(self, context):
            del context
            return ProbabilityTable(("output[0]",), ((("0",), 0.5), (("1",), 0.5)))

    result = DistributionEngine(ProgramDistributionMaterializer(), Target()).verify(context)
    assert result.status is SemanticStatus.VERIFIED_PASS


def test_contract_bit_order_overrides_candidate_render_and_storage_order() -> None:
    base = ContractRegistry.from_package("core").get("core", "01")
    contract = replace(
        base,
        systems=SystemsSpec((SystemSpec("output", SystemKind.CLASSICAL, SystemRole.CLASSICAL_OUTPUT, (0, 1), 4),)),
        observation=ObservationSpec((), ("output",), (), (), BitOrder.BIG_ENDIAN, None),
    )
    program = _program(
        (
            Operation(OperationKind.GATE, "x", quantum_wires=(0,)),
            Operation(OperationKind.MEASUREMENT, "measure", (0,), (1,)),
            Operation(OperationKind.MEASUREMENT, "measure", (1,), (0,)),
        ),
        2,
        2,
        (0, 1),
    )
    context = _context(contract, program, (0.0, 1.0, 0.0, 0.0))

    program_table = ProgramDistributionMaterializer().distribution(context)
    execution_table = ExecutionDistributionMaterializer(context.execution_probabilities).distribution(context)

    assert program_table.variables == ("output[0]", "output[1]")
    assert dict(program_table.rows) == {("1", "0"): 1.0}
    assert dict(execution_table.rows) == {("0", "0"): 0.0, ("1", "0"): 1.0, ("0", "1"): 0.0, ("1", "1"): 0.0}


def test_distribution_candidate_semantic_error_is_a_model_failure() -> None:
    contract = ContractRegistry.from_package("core").get("core", "01")
    program = _program((Operation(OperationKind.MEASUREMENT, "measure", (0,), (0,)),), 1, 1, (0,))

    class Candidate:
        def distribution(self, context):
            del context
            raise CandidateSemanticError("candidate_distribution_invalid")

    class Target:
        def distribution(self, context):
            raise AssertionError("target must not be loaded after candidate failure")

    result = DistributionEngine(Candidate(), Target()).verify(_context(contract, program))
    assert result.status is SemanticStatus.SEMANTIC_FAIL
    assert result.reason == "candidate_distribution_invalid"


def test_instrument_candidate_semantic_error_is_a_model_failure() -> None:
    contract = ContractRegistry.from_package("core").get("core", "01")
    program = _program((Operation(OperationKind.MEASUREMENT, "measure", (0,), (0,)),), 1, 1, (0,))

    class Candidate:
        def instrument(self, context):
            del context
            raise CandidateSemanticError("candidate_instrument_invalid")

    class Target:
        def instrument(self, context):
            raise AssertionError("target must not be loaded after candidate failure")

    result = InstrumentEngine(Candidate(), Target()).verify(_context(contract, program))
    assert result.status is SemanticStatus.SEMANTIC_FAIL
    assert result.reason == "candidate_instrument_invalid"


def test_distribution_execution_table_rejects_non_finite_candidate_values() -> None:
    contract = ContractRegistry.from_package("core").get("core", "01")
    program = _program(
        (
            Operation(OperationKind.MEASUREMENT, "measure", (0,), (0,)),
            Operation(OperationKind.MEASUREMENT, "measure", (1,), (1,)),
        ),
        2,
        2,
        (1, 0),
    )
    context = _context(contract, program, (np.nan, 1.0, 0.0, 0.0))

    table = ExecutionDistributionMaterializer(context.execution_probabilities).distribution(context)
    assert np.isnan(table.rows[0][1])
