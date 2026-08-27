"""Production verdict taxonomy and packaged route-closure tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qceval.semantics.contracts import ContractRegistry
from qceval.semantics.result_record import make_execution_error_result_record, read_result_record
from qceval.semantics.verifiers.exact.engines import ExactEngineSpec
from qceval.semantics.verifiers.exact.metrics import _numeric_result
from qceval.semantics.verifiers.result import (
    RESULT_SCHEMA_VERSION,
    SemanticStatus,
    VerifierResult,
)
from qceval.semantics.verifiers.router import reconcile_results
from qceval.semantics.verifiers.symbolic import SymbolicProof, _proof_status

PACKAGED_ENGINES = {
    "channel_exact",
    "classical_io_exhaustive",
    "distribution_exact",
    "instrument_exact",
    "isometry_exact",
    "state_exact",
    "symbolic_family_bounded",
    "unitary_exact",
}


def test_semantic_statuses_are_decisive_or_operational() -> None:
    assert {status.value for status in SemanticStatus} == {
        "verified_pass",
        "semantic_fail",
        "execution_error",
        "resource_limit",
    }


@pytest.mark.parametrize("historical_status", ["unsupported", "inconclusive"])
def test_historical_nondecisions_read_as_execution_errors(historical_status: str) -> None:
    payload = make_execution_error_result_record(
        suite="core",
        task_id="01",
        framework="qiskit",
        reason="historical_nondecision",
    )
    payload["result_schema_version"] = "2"
    payload["status"] = historical_status

    migrated = read_result_record(payload)

    assert migrated["result_schema_version"] == "3"
    assert migrated["status"] == "execution_error"
    assert migrated["passed"] is False
    assert {"name": "historical_status", "value": historical_status} in migrated["diagnostics"]


@pytest.mark.parametrize("suite", ["core", "qec"])
def test_packaged_contracts_have_one_closed_primary_route(suite: str) -> None:
    registry = ContractRegistry.from_package(suite)

    for contract in registry:
        assert len(contract.routing.primary) == 1
        assert contract.routing.primary[0].engine in PACKAGED_ENGINES
        assert contract.routing.primary[0].cross_check is False
        assert contract.routing.fallback == ()


def test_numeric_uncertainty_band_is_execution_error() -> None:
    context = SimpleNamespace(
        contract=SimpleNamespace(
            approximation=SimpleNamespace(tolerance=1.0, uncertainty=0.1),
        ),
        contract_hash="contract",
        input_hash="input",
        target_hash="target",
    )
    spec = ExactEngineSpec("state_exact", "state", "statevector", "trace_distance", ())

    result = _numeric_result(context, spec, 1.0, 0.0, 1, 0.0)

    assert result.status is SemanticStatus.EXECUTION_ERROR
    assert result.reason == "metric_in_uncertainty_band"


def test_cross_check_disagreement_is_execution_error() -> None:
    context = SimpleNamespace(contract_hash="contract", target_hash="target")
    results = tuple(
        VerifierResult(
            RESULT_SCHEMA_VERSION,
            status,
            "engine_result",
            "contract",
            "target",
            "engine",
        )
        for status in (SemanticStatus.VERIFIED_PASS, SemanticStatus.SEMANTIC_FAIL)
    )

    result = reconcile_results(results, context)

    assert result.status is SemanticStatus.EXECUTION_ERROR
    assert result.reason == "cross_check_disagreement"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("symbolic_identity_unresolved", SemanticStatus.EXECUTION_ERROR),
        ("symbolic_worker_protocol_error", SemanticStatus.EXECUTION_ERROR),
        ("symbolic_worker_timeout", SemanticStatus.RESOURCE_LIMIT),
        ("symbolic_expression_node_limit", SemanticStatus.RESOURCE_LIMIT),
    ],
)
def test_symbolic_nonproofs_are_operational_failures(reason: str, expected: SemanticStatus) -> None:
    proof = SymbolicProof("inconclusive", reason, None, (), 0, (), 0.0)

    assert _proof_status(proof) is expected


def test_choi_sanity_uses_target_trace_not_square_dimension() -> None:
    # Regression (audit L2): the Choi sanity residual compared the candidate
    # trace against sqrt(dim), which equals the input dimension only for
    # square channels. The trusted target's trace is the input dimension for
    # any channel geometry.
    import numpy as np

    from qceval.semantics.verifiers.exact.metrics import _array_error

    # Non-square channel: 1 qubit in, 2 qubits out (isometric embedding).
    # Choi matrix is 8x8 with trace d_in = 2, while sqrt(8) ~ 2.828.
    isometry = np.zeros((4, 2), dtype=complex)
    isometry[0, 0] = 1.0
    isometry[3, 1] = 1.0
    choi = np.einsum("ai,bj->aibj", isometry, isometry.conj()).reshape(8, 8)
    error, sanity = _array_error("choi", choi, choi.copy())
    assert error == 0.0
    assert sanity < 1e-12
