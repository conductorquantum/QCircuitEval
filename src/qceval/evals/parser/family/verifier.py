"""Structured-family source verifier entry points and result helpers."""

from __future__ import annotations

import ast

from qceval.evals.parser.family.qaoa import _prove_qaoa
from qceval.evals.parser.family.rotation import _prove_rotation_family
from qceval.semantics.contracts import contract_hash
from qceval.semantics.integration import SemanticVerificationRequest
from qceval.semantics.ir import source_code_sha256
from qceval.semantics.verifiers.result import (
    SemanticStatus,
    VerifierResult,
    make_evidence,
    make_verifier_result,
)

FAMILY_ENGINE_VERSION = "1.1.0"
STRUCTURED_QAOA_COMPLETENESS = "structured_qaoa_source_identity"
STRUCTURED_ROTATION_COMPLETENESS = "structured_rotation_source_identity"


class StructuredFamilySourceVerifier:
    """Prove supported source families from exact gate and binding structure."""

    def verify(self, request: SemanticVerificationRequest) -> VerifierResult:
        """Prove or refute one contract-selected structured family.

        Args:
            request: Contract, framework, execution, and candidate source.

        Returns:
            Decisive proof/refutation or typed unsupported syntax.
        """
        completeness = request.contract.parameters.completeness
        if completeness not in {STRUCTURED_QAOA_COMPLETENESS, STRUCTURED_ROTATION_COMPLETENESS}:
            return _result(request, SemanticStatus.EXECUTION_ERROR, "family_completeness_unsupported")
        if request.code is None:
            return _result(request, SemanticStatus.EXECUTION_ERROR, "family_source_unavailable")
        try:
            tree = ast.parse(request.code)
        except SyntaxError:
            return _result(request, SemanticStatus.EXECUTION_ERROR, "family_source_syntax_error")
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == request.contract.signature.entry_point
            ),
            None,
        )
        if function is None:
            return _result(request, SemanticStatus.EXECUTION_ERROR, "family_entry_point_missing")
        if completeness == STRUCTURED_QAOA_COMPLETENESS:
            outcome, reason = _prove_qaoa(function)
        else:
            outcome, reason = _prove_rotation_family(function, request.contract.task_id)
        return _result(request, outcome, reason)


def _result(request: SemanticVerificationRequest, status: SemanticStatus, reason: str) -> VerifierResult:
    source_hash = source_code_sha256(request.code)
    evidence = make_evidence(
        "structured_family_source",
        FAMILY_ENGINE_VERSION,
        reason,
        input_hash=source_hash,
        target_hash=request.contract.target.sha256,
        cases_checked=1,
        preconditions=(f"framework={request.framework}", "universal_parameter_quantifier"),
    )
    return make_verifier_result(
        status,
        reason,
        contract_hash=contract_hash(request.contract),
        target_hash=request.contract.target.sha256,
        verifier_version=FAMILY_ENGINE_VERSION,
        evidence=(evidence,),
    )
