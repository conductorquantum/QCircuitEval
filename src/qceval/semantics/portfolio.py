"""Default lowering and verifier portfolio behind the evaluator seam."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.semantics.contracts import contract_hash
from qceval.semantics.integration import SemanticVerificationRequest, SemanticVerifier
from qceval.semantics.ir import source_code_sha256
from qceval.semantics.lowering import LoweringResult, LoweringStatus, SourceMetadata, default_lowering_registry
from qceval.semantics.verifiers import (
    BoundedSymbolicSourceVerifier,
    ClassicalIOEngine,
    DistributionEngine,
    InstrumentEngine,
    PackagedClassicalTargetProvider,
    PackagedDistributionTargetProvider,
    PackagedInstrumentTargetProvider,
    ProgramClassicalIOMaterializer,
    ProgramInstrumentMaterializer,
    ProgramIRMaterializer,
    StructuredFamilySourceVerifier,
    VerifierRegistry,
    VerifierRouter,
    channel_engine,
    isometry_engine,
    state_engine,
    unitary_engine,
)
from qceval.semantics.verifiers.family import (
    STRUCTURED_QAOA_COMPLETENESS,
    STRUCTURED_ROTATION_COMPLETENESS,
)
from qceval.semantics.verifiers.family_numeric import verify_analytic_family_unitary
from qceval.semantics.verifiers.observational import AdaptiveDistributionMaterializer
from qceval.semantics.verifiers.requirements import (
    verify_case_program_requirements,
    verify_program_requirements,
)
from qceval.semantics.verifiers.requirements.cases import case_program_invariance_required
from qceval.semantics.verifiers.result import RESULT_SCHEMA_VERSION, EvidenceRecord, SemanticStatus, VerifierResult
from qceval.semantics.verifiers.symbolic import SYMBOLIC_COMPLETENESS
from qceval.semantics.verifiers.targets import CoreAnalyticTargetProvider

PORTFOLIO_VERSION = "1.0.0"


class DefaultSemanticVerifier:
    """Lower executed candidates and route supported semantic contracts."""

    def __init__(self) -> None:
        materializer = ProgramIRMaterializer()
        targets = CoreAnalyticTargetProvider()
        engines = (
            state_engine(materializer, targets),
            unitary_engine(materializer, targets),
            isometry_engine(materializer, targets),
            channel_engine(materializer, targets),
            ClassicalIOEngine(ProgramClassicalIOMaterializer(), PackagedClassicalTargetProvider()),
            DistributionEngine(AdaptiveDistributionMaterializer(), PackagedDistributionTargetProvider()),
            InstrumentEngine(ProgramInstrumentMaterializer(), PackagedInstrumentTargetProvider()),
        )
        self._router = VerifierRouter(VerifierRegistry(engines))
        self._lowering = default_lowering_registry()
        self._symbolic: SemanticVerifier = BoundedSymbolicSourceVerifier()
        self._family: SemanticVerifier = StructuredFamilySourceVerifier()

    def verify(self, request: SemanticVerificationRequest) -> VerifierResult:
        """Lower and verify one evaluator request.

        Args:
            request: Contract, framework execution, and source.

        Returns:
            Routed semantic result or typed lowering failure.
        """
        if request.cases:
            return self._verify_cases(request)
        return self._verify_single(request)

    def _verify_single(
        self,
        request: SemanticVerificationRequest,
        shared_source_result: VerifierResult | None = None,
    ) -> VerifierResult:
        """Verify one bound execution, optionally reusing a source verdict."""
        completeness = request.contract.parameters.completeness
        # The bounded source engines are proofs over source syntax and do not
        # require framework execution.  Keeping this path available supports
        # static CUDA-Q auditing and standalone contract/corpus validation.
        if request.execution is None:
            source_result = self._source_result(request, completeness)
            if source_result is not None:
                return source_result
        source_hash = source_code_sha256(request.code)
        returned = _returned_value(request)
        adapter = self._lowering.get(request.framework)
        lowered = adapter.lower(
            returned,
            SourceMetadata(request.framework, source_hash, _backend(request.execution)),
            request.contract,
        )
        if lowered.status is not LoweringStatus.SUCCESS:
            assert lowered.error is not None
            return self._lowering_fallback(request, completeness, lowered, source_hash)
        assert lowered.program is not None
        metadata = getattr(request.execution, "metadata", {})
        requirement_failure = verify_program_requirements(
            request.contract,
            lowered.program,
            framework=request.framework,
            execution_metadata=metadata if isinstance(metadata, dict) else {},
            source_code=request.code,
            candidate_unitary=getattr(request.execution, "unitary", None),
            arguments=tuple(request.arguments or ()),
        )
        if requirement_failure is not None:
            return requirement_failure
        source_result = shared_source_result or self._source_result(request, completeness)
        if source_result is not None:
            if completeness == STRUCTURED_ROTATION_COMPLETENESS:
                # These contracts quantify over all real parameter values.
                # Their diagnostic and probe points are useful evidence, but
                # they cannot upgrade an unsupported source proof (including
                # parameter rebinding, mutation, or control flow) to a
                # universal pass.  Preserve the source proof's typed verdict.
                return source_result
            # The syntactic source proof is the fast path: a proof passes
            # outright and a policy refutation (forbidden gate family) is
            # terminal. Every other outcome - unsupported spellings, control
            # flow, rebinding, and syntactic mismatches - falls back to
            # numeric verification of the family at the executed diagnostic
            # and probe points, all of which must agree within the contract
            # tolerance for the reconciled family verdict to pass.
            if source_result.status is SemanticStatus.VERIFIED_PASS or _terminal_source_refutation(source_result):
                return source_result
            numeric = self._numeric_family_fallback(request, completeness, lowered.program)
            if numeric is not None:
                return numeric
            return source_result
        metadata_map = metadata if isinstance(metadata, dict) else {}
        probabilities = getattr(request.execution, "probabilities", None)
        return self._router.verify(
            request.contract,
            lowered.program,
            arguments=tuple(request.arguments or ()),
            execution_metadata=metadata_map,
            execution_probabilities=None if probabilities is None else tuple(float(value) for value in probabilities),
        )

    def _lowering_fallback(
        self,
        request: SemanticVerificationRequest,
        completeness: str | None,
        lowered: LoweringResult,
        source_hash: str,
    ) -> VerifierResult:
        assert lowered.error is not None
        if lowered.status is LoweringStatus.UNSUPPORTED and (
            completeness == SYMBOLIC_COMPLETENESS or _routes_engine(request.contract, "symbolic_family_bounded")
        ):
            # Source-proof contracts do not require a lowered program: the
            # bounded symbolic grammar enforces the gate basis and rejects
            # everything outside it, so an unsupported lowering (for example a
            # CUDA-Q kernel closure) can still be decided from source. Every
            # non-proof outcome remains fail-closed.
            source_result = self._source_result(request, completeness)
            if source_result is not None:
                return source_result
        return _lowering_failure(request, lowered.status, lowered.error.reason, source_hash)

    def _verify_cases(self, request: SemanticVerificationRequest) -> VerifierResult:
        case_requirement = self._case_requirement_result(request)
        if case_requirement is not None:
            return case_requirement
        # Source-level family proofs depend only on the candidate source, so
        # one shared verdict serves every bound case. This keeps the syntactic
        # fast path to a single proof and, when it is inconclusive, lets each
        # case fall back to numeric verification at its own bound point.
        shared_source = self._shared_source_result(request)
        results = tuple(
            (
                case.arguments,
                self._verify_single(
                    SemanticVerificationRequest(
                        request.contract,
                        request.framework,
                        case.execution,
                        request.code,
                        case.arguments,
                    ),
                    shared_source_result=shared_source,
                ),
            )
            for case in request.cases
        )
        return _reconcile_parameter_cases(results)

    def _shared_source_result(self, request: SemanticVerificationRequest) -> VerifierResult | None:
        completeness = request.contract.parameters.completeness
        if completeness in {
            SYMBOLIC_COMPLETENESS,
            STRUCTURED_QAOA_COMPLETENESS,
            STRUCTURED_ROTATION_COMPLETENESS,
        } or _routes_engine(request.contract, "symbolic_family_bounded"):
            return self._source_result(request, completeness)
        return None

    def _numeric_family_fallback(
        self,
        request: SemanticVerificationRequest,
        completeness: str | None,
        program: Any,
    ) -> VerifierResult | None:
        """Numerically verify one bound family point after an inconclusive proof."""
        arguments = tuple(request.arguments or ())
        if completeness == STRUCTURED_ROTATION_COMPLETENESS:
            metadata = getattr(request.execution, "metadata", {})
            probabilities = getattr(request.execution, "probabilities", None)
            routed = self._router.verify(
                request.contract,
                program,
                arguments=arguments,
                execution_metadata=metadata if isinstance(metadata, dict) else {},
                execution_probabilities=None
                if probabilities is None
                else tuple(float(value) for value in probabilities),
            )
            if routed.status is SemanticStatus.SEMANTIC_FAIL:
                marker = EvidenceRecord(
                    "family_numeric_fallback",
                    PORTFOLIO_VERSION,
                    "family_numeric_mismatch",
                    source_code_sha256(request.code),
                    request.contract.target.sha256,
                    preconditions=(f"arguments={arguments!r}",),
                )
                return replace(routed, reason="family_numeric_mismatch", evidence=(*routed.evidence, marker))
            return routed
        if completeness == SYMBOLIC_COMPLETENESS or _routes_engine(request.contract, "symbolic_family_bounded"):
            return verify_analytic_family_unitary(request.contract, program, arguments)
        return None

    def _case_requirement_result(self, request: SemanticVerificationRequest) -> VerifierResult | None:
        """Lower every finite case and enforce declared cross-case invariants."""
        if not case_program_invariance_required(request.contract):
            return None
        source_hash = source_code_sha256(request.code)
        adapter = self._lowering.get(request.framework)
        programs = []
        for case in request.cases:
            case_request = SemanticVerificationRequest(
                request.contract,
                request.framework,
                case.execution,
                request.code,
                case.arguments,
            )
            lowered = adapter.lower(
                _returned_value(case_request),
                SourceMetadata(request.framework, source_hash, _backend(case.execution)),
                request.contract,
            )
            # Individual-case verification below produces the precise typed
            # lowering failure. Do not mask it with a cross-case diagnostic.
            if lowered.status is not LoweringStatus.SUCCESS or lowered.program is None:
                return None
            programs.append((case.arguments, lowered.program))
        return verify_case_program_requirements(request.contract, tuple(programs))

    def _source_result(self, request: SemanticVerificationRequest, completeness: str | None) -> VerifierResult | None:
        if completeness == SYMBOLIC_COMPLETENESS or _routes_engine(request.contract, "symbolic_family_bounded"):
            return self._symbolic.verify(request)
        if completeness in {STRUCTURED_QAOA_COMPLETENESS, STRUCTURED_ROTATION_COMPLETENESS}:
            return self._family.verify(request)
        return None


def _routes_engine(contract: Any, engine: str) -> bool:
    return any(route.engine == engine for route in contract.routing.primary)


_TERMINAL_SOURCE_REFUTATIONS = ("symbolic_forbidden_gate_family", "symbolic_required_gate_family_missing")


def _terminal_source_refutation(result: VerifierResult) -> bool:
    """Return whether a source refutation is a terminal gate-policy verdict.

    Forbidden/required gate-family refutations restate the contract's declared
    gate basis, which numeric behavior can never satisfy, so they stay
    decisive. Model-level refutations (projective counterexamples, syntactic
    mismatches) instead defer to direct numeric verification of the executed
    candidate, which is sound even where the source model is not.
    """
    return result.status is SemanticStatus.SEMANTIC_FAIL and result.reason.startswith(_TERMINAL_SOURCE_REFUTATIONS)


def default_semantic_verifier() -> SemanticVerifier:
    """Build the default production semantic verifier portfolio.

    Returns:
        Lowering-aware semantic verifier.
    """
    return DefaultSemanticVerifier()


def _returned_value(request: SemanticVerificationRequest) -> Any:
    if request.framework == "cudaq":
        if request.code is None:
            return None
        return CudaqProgram(
            request.code,
            request.contract.signature.entry_point,
            tuple(request.arguments or ()),
            getattr(request.execution, "circuit", None),
        )
    return getattr(request.execution, "circuit", None)


def _backend(execution: Any) -> str | None:
    metadata = getattr(execution, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("backend") or metadata.get("probability_method")
    return None if value is None else str(value)


def _lowering_failure(
    request: SemanticVerificationRequest,
    status: LoweringStatus,
    reason: str,
    input_hash: str,
) -> VerifierResult:
    semantic_status = {
        LoweringStatus.UNSUPPORTED: SemanticStatus.EXECUTION_ERROR,
        LoweringStatus.EXECUTION_ERROR: SemanticStatus.EXECUTION_ERROR,
        LoweringStatus.RESOURCE_LIMIT: SemanticStatus.RESOURCE_LIMIT,
    }.get(status, SemanticStatus.EXECUTION_ERROR)
    evidence = EvidenceRecord(
        "framework_lowering",
        PORTFOLIO_VERSION,
        reason,
        input_hash,
        request.contract.target.sha256,
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        semantic_status,
        reason,
        contract_hash(request.contract),
        request.contract.target.sha256,
        PORTFOLIO_VERSION,
        (evidence,),
    )


def _reconcile_parameter_cases(
    results: tuple[tuple[tuple[Any, ...], VerifierResult], ...],
) -> VerifierResult:
    """Require every case in a finite exhaustive parameter domain to pass."""

    precedence = (
        SemanticStatus.SEMANTIC_FAIL,
        SemanticStatus.RESOURCE_LIMIT,
        SemanticStatus.EXECUTION_ERROR,
        SemanticStatus.VERIFIED_PASS,
    )
    status = next(value for value in precedence if any(result.status is value for _, result in results))
    first = results[0][1]
    evidence = tuple(
        replace(
            record,
            preconditions=(
                f"case_index={case_index}",
                f"case_status={result.status.value}",
                f"arguments={arguments!r}",
                *record.preconditions,
            ),
        )
        for case_index, (arguments, result) in enumerate(results)
        for record in result.evidence
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        "all_parameter_cases_passed" if status is SemanticStatus.VERIFIED_PASS else f"parameter_domain_{status.value}",
        first.contract_hash,
        first.target_hash,
        PORTFOLIO_VERSION,
        evidence,
        tuple(
            (
                f"parameter_case_count:{case_status.value}",
                str(sum(result.status is case_status for _, result in results)),
            )
            for case_status in SemanticStatus
        ),
    )
