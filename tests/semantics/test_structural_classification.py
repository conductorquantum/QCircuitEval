"""Unit tests for semantic requirement classification helpers."""

from __future__ import annotations

from qceval.semantics.contracts import ContractRegistry
from qceval.semantics.verifiers.result import RESULT_SCHEMA_VERSION, SemanticStatus, VerifierResult
from qceval.semantics.verifiers.structural import (
    RequirementClass,
    attach_diagnostics,
    classify_requirements,
)


def test_classify_requirements_separates_hard_lifecycle_and_diagnostic() -> None:
    contract = ContractRegistry.from_package().get("core", "16")
    classified = classify_requirements(contract)
    authorities = {item.authority for item in classified}
    assert RequirementClass.HARD_API in authorities
    assert all(item.requirement.kind for item in classified)


def test_attach_diagnostics_preserves_verdict() -> None:
    result = VerifierResult(
        RESULT_SCHEMA_VERSION,
        SemanticStatus.VERIFIED_PASS,
        "ok",
        "contract",
        "target",
        "test-1.0.0",
    )
    updated = attach_diagnostics(result, {"legacy": "match", "alpha": "1"})
    assert updated.status is SemanticStatus.VERIFIED_PASS
    assert updated.reason == "ok"
    assert updated.diagnostics == (("alpha", "1"), ("legacy", "match"))
