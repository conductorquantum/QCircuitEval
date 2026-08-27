"""Narrow evaluator-to-semantic-verifier integration protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from qceval.semantics.contracts import Contract, contract_hash
from qceval.semantics.verifiers.result import RESULT_SCHEMA_VERSION, SemanticStatus, VerifierResult

FailureOrigin = Literal["candidate_execution", "grader_verification"]


@dataclass(frozen=True)
class SemanticExecutionCase:
    """One explicitly bound execution in a finite parameter domain."""

    arguments: tuple[Any, ...]
    execution: Any


@dataclass(frozen=True)
class SemanticVerificationRequest:
    """Inputs passed from evaluator execution to a semantic verifier."""

    contract: Contract
    framework: str
    execution: Any
    code: str | None
    arguments: tuple[Any, ...] = ()
    cases: tuple[SemanticExecutionCase, ...] = ()


class SemanticVerifier(Protocol):
    """Semantic verification seam owned by the evaluator."""

    def verify(self, request: SemanticVerificationRequest) -> VerifierResult:
        """Verify one already executed candidate.

        Args:
            request: Contract, framework, execution, and optional source.

        Returns:
            One decisive or operational verifier result.
        """
        ...


def exception_result(
    contract: Contract,
    exc: Exception,
    *,
    failure_origin: FailureOrigin = "grader_verification",
) -> VerifierResult:
    """Convert an integration exception to bounded result data.

    Args:
        contract: Resolved task contract.
        exc: Caught verifier exception.
        failure_origin: Stable boundary that raised the exception.

    Returns:
        Execution-error result without exception text or traceback.
    """
    prefix = (
        "candidate_execution_exception" if failure_origin == "candidate_execution" else "semantic_verifier_exception"
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        SemanticStatus.EXECUTION_ERROR,
        f"{prefix}:{type(exc).__name__}",
        contract_hash(contract),
        contract.target.sha256,
        "integration-1.0.0",
        diagnostics=(("failure_origin", failure_origin),),
    )


def validate_result_identity(contract: Contract, result: VerifierResult) -> VerifierResult:
    """Fail closed when a verifier returns evidence for another target.

    Args:
        contract: Contract requested by the evaluator.
        result: Result returned by the semantic verifier.

    Returns:
        Original result when identities agree, otherwise an execution error.
    """
    if result.contract_hash == contract_hash(contract) and result.target_hash == contract.target.sha256:
        return result
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        SemanticStatus.EXECUTION_ERROR,
        "semantic_result_identity_mismatch",
        contract_hash(contract),
        contract.target.sha256,
        "integration-1.0.0",
        diagnostics=(("failure_origin", "grader_verification"),),
    )
