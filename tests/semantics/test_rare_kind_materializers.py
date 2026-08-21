"""Focused engine coverage for rare-kind distinctions without portable sources."""

from __future__ import annotations

import numpy as np

from qceval.semantics.contracts import ContractRegistry
from qceval.semantics.ir import IR_VERSION, Control, Operation, OperationKind, Program, Provenance
from qceval.semantics.verifiers import ProgramIRMaterializer, isometry_engine
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.result import SemanticStatus
from qceval.semantics.verifiers.targets import CoreAnalyticTargetProvider


def _task28_program(*, restore_ancilla: bool) -> Program:
    compute = Operation(
        OperationKind.GATE,
        "x",
        quantum_wires=(3,),
        controls=(Control(0), Control(1)),
    )
    use = Operation(
        OperationKind.GATE,
        "x",
        quantum_wires=(4,),
        controls=(Control(2), Control(3)),
    )
    operations = (compute, use, compute) if restore_ancilla else (compute, use)
    return Program(
        IR_VERSION,
        5,
        0,
        operations,
        None,
        (),
        Provenance("test", "1", source_hash="a" * 64),
    )


def _task28_context(program: Program) -> VerificationContext:
    contract = ContractRegistry.from_package("core").get("core", "28")
    return VerificationContext(
        contract,
        "contract",
        contract.target.sha256,
        "input",
        program,
    )


def test_task28_materializer_rejects_uncomputed_ancilla_despite_same_default_output() -> None:
    """The complete logical isometry, not the default |00000> result, is checked.

    Compute/use without uncompute agrees on the benchmark's default input but
    leaks q3 for logical inputs with q0=q1=1.
    """
    materializer = ProgramIRMaterializer()
    targets = CoreAnalyticTargetProvider()
    valid_context = _task28_context(_task28_program(restore_ancilla=True))
    leaking_context = _task28_context(_task28_program(restore_ancilla=False))

    expected = targets.array(valid_context, "isometry").value
    valid = materializer.array(valid_context, "isometry").value
    leaking = materializer.array(leaking_context, "isometry").value
    leaked_rows = [basis for basis in range(32) if basis & (1 << 3)]

    assert np.allclose(valid, expected, atol=1e-12)
    assert not np.allclose(leaking, expected, atol=1e-12)
    assert np.linalg.norm(leaking[leaked_rows, :]) > 0.0

    engine = isometry_engine(materializer, targets)
    assert engine.verify(valid_context).status is SemanticStatus.VERIFIED_PASS
    assert engine.verify(leaking_context).status is SemanticStatus.SEMANTIC_FAIL
