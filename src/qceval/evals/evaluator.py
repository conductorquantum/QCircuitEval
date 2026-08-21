"""Fail-closed behavior evaluator orchestration for bundled tasks."""

from __future__ import annotations

from typing import Any

from qceval.evals.execution import ExecutionDispatcher
from qceval.evals.inputs import global_inputs, task6_witness_state
from qceval.evals.models import ExecutionResult
from qceval.evals.sandbox import ignore_candidate_library_warnings
from qceval.evals.tasks import load_tasks
from qceval.models import Framework, Suite
from qceval.semantics.contracts import AuditStatus, Contract, ContractRegistry, call_args_from_signature
from qceval.semantics.integration import (
    SemanticExecutionCase,
    SemanticVerificationRequest,
    SemanticVerifier,
    exception_result,
    validate_result_identity,
)
from qceval.semantics.portfolio import default_semantic_verifier
from qceval.semantics.result_record import make_execution_error_result_record, make_result_record
from qceval.semantics.verifiers.result import SemanticStatus, VerifierResult


def _candidate_error(exc: Exception) -> str:
    """Return bounded candidate-visible exception detail without a traceback."""
    detail = str(exc).strip().replace("\x00", "")
    value = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return value[:2000]


class Evaluator:
    """Execute candidates and grade only contracted behavior.

    Every semantic result is authoritative and fail-closed. Only
    :attr:`~qceval.semantics.verifiers.result.SemanticStatus.VERIFIED_PASS`
    passes; every other status fails. There is no fallback grader: a missing
    contract registry yields an authoritative ``execution_error`` result.

    Args:
        framework: Framework whose candidate code is executed.
        suite: Benchmark suite whose tasks and contracts are loaded.
        tasks: Raw task payloads keyed by normalized task identifier.
        qiskit_tasks: Retained for constructor compatibility and ignored.
        semantic_verifier: Optional verifier portfolio override.
    """

    def __init__(
        self,
        framework: Framework,
        suite: Suite,
        tasks: dict[str, dict[str, Any]],
        qiskit_tasks: dict[str, dict[str, Any]],
        *,
        semantic_verifier: SemanticVerifier | None = None,
    ) -> None:
        del qiskit_tasks
        self.framework = framework
        self.suite = suite
        self.tasks = tasks
        self.inputs = global_inputs(framework)
        self._dispatcher = ExecutionDispatcher(framework, tasks)
        self._verifier = semantic_verifier or default_semantic_verifier()
        self._contracts = _load_contracts(suite)

    def grade_code(
        self,
        *,
        task_id: str,
        code: str,
        entry_point: str,
    ) -> tuple[ExecutionResult, dict[str, Any]]:
        """Execute candidate source and grade its contracted behavior.

        Args:
            task_id: Task identifier normalized with ``zfill(2)``.
            code: Candidate Python source.
            entry_point: Candidate function to invoke.

        Returns:
            Framework execution and behavior-first grader details.
        """
        task_id = str(task_id).zfill(2)
        if self._contracts is None:
            execution = ExecutionResult(
                probabilities=[],
                metadata={"semantic_status": SemanticStatus.EXECUTION_ERROR.value},
            )
            return execution, _execution_error_details(self.suite, task_id, self.framework)
        contract = self._contract(task_id)
        admission = _admission_failure(contract, self.suite, self.framework)
        if admission is not None:
            execution = ExecutionResult(
                probabilities=[],
                metadata={"semantic_status": SemanticStatus.EXECUTION_ERROR.value},
            )
            return execution, admission
        try:
            call_args = self._call_args(contract, task_id)
        except Exception as exc:  # noqa: BLE001 - contract binding failures are typed grader results.
            result = validate_result_identity(
                contract,
                exception_result(contract, exc, failure_origin="grader_verification"),
            )
            execution = ExecutionResult(
                probabilities=[],
                metadata={"grader_error": type(exc).__name__},
            )
            return execution, _behavior_details(result, contract, self.framework)
        try:
            with ignore_candidate_library_warnings():
                execution = self._dispatcher.execute(
                    task_id=task_id,
                    code=code,
                    entry_point=entry_point,
                    call_args=call_args,
                )
        except Exception as exc:  # noqa: BLE001 - candidate failures become typed results.
            result = validate_result_identity(
                contract,
                exception_result(contract, exc, failure_origin="candidate_execution"),
            )
            execution = ExecutionResult(
                probabilities=[],
                metadata={
                    "candidate_failure_stage": "compile" if isinstance(exc, SyntaxError) else "run",
                    "candidate_error_type": type(exc).__name__,
                    "candidate_error": _candidate_error(exc),
                },
            )
            return execution, _behavior_details(result, contract, self.framework)
        return execution, self._grade_behavior(task_id=task_id, execution=execution, code=code)

    def execute_code(
        self,
        *,
        task_id: str,
        code: str,
        entry_point: str,
    ) -> ExecutionResult:
        """Execute candidate source without grading it.

        Args:
            task_id: Task identifier normalized with ``zfill(2)``.
            code: Candidate Python source.
            entry_point: Candidate function to invoke.

        Returns:
            Normalized framework execution result.
        """
        task_id = str(task_id).zfill(2)
        call_args = None
        if self._contracts is not None:
            call_args = self._call_args(self._contract(task_id), task_id)
        return self._dispatcher.execute(
            task_id=task_id,
            code=code,
            entry_point=entry_point,
            call_args=call_args,
        )

    def grade_execution(
        self,
        *,
        task_id: str,
        execution: ExecutionResult,
        code: str | None = None,
    ) -> dict[str, Any]:
        """Grade a precomputed execution using the semantic portfolio.

        Args:
            task_id: Task identifier normalized with ``zfill(2)``.
            execution: Candidate framework execution.
            code: Candidate source for source-level and symbolic checks.

        Returns:
            Behavior-first details with a fail-closed binary projection.
        """
        task_id = str(task_id).zfill(2)
        if self._contracts is None:
            return _execution_error_details(self.suite, task_id, self.framework)
        contract = self._contract(task_id)
        if code is None and contract.parameters.quantifier.value == "exhaustive":
            return _execution_error_details(
                self.suite,
                task_id,
                self.framework,
                reason="exhaustive_parameter_replay_requires_source",
            )
        return self._grade_behavior(task_id=task_id, execution=execution, code=code)

    def _grade_behavior(
        self,
        *,
        task_id: str,
        execution: ExecutionResult,
        code: str | None,
    ) -> dict[str, Any]:
        """Apply authoritative fail-closed behavior grading."""
        contract = self._contract(task_id)
        admission = _admission_failure(contract, self.suite, self.framework)
        if admission is not None:
            return admission
        try:
            arguments = self._call_args(contract, task_id)
        except Exception as exc:  # noqa: BLE001 - contract binding failures are typed grader results.
            result = exception_result(contract, exc, failure_origin="grader_verification")
            return _behavior_details(validate_result_identity(contract, result), contract, self.framework)
        try:
            cases = self._semantic_execution_cases(contract, task_id, code, execution, arguments)
        except Exception as exc:  # noqa: BLE001 - candidate replay failures become typed results.
            result = exception_result(contract, exc, failure_origin="candidate_execution")
            return _behavior_details(validate_result_identity(contract, result), contract, self.framework)
        request = SemanticVerificationRequest(
            contract,
            self.framework,
            execution,
            code,
            arguments,
            cases,
        )
        try:
            with ignore_candidate_library_warnings():
                result = self._verifier.verify(request)
        except Exception as exc:  # noqa: BLE001 - verifier failures become typed results.
            result = exception_result(contract, exc, failure_origin="grader_verification")
        result = validate_result_identity(contract, result)
        return _behavior_details(result, contract, self.framework)

    def _call_args(self, contract: Contract, task_id: str) -> tuple[Any, ...]:
        """Resolve positional entry-point arguments from the contract signature."""
        if contract.parameters.quantifier.value == "exhaustive" and contract.parameters.diagnostic_points:
            return self._bind_point(contract, contract.parameters.diagnostic_points[0])
        input_value = self.inputs.get(task_id)
        return call_args_from_signature(contract.signature, input_value)

    def _bind_point(self, contract: Contract, point: tuple[Any, ...]) -> tuple[Any, ...]:
        """Translate one diagnostic point into concrete entry-point arguments.

        Numeric points normally bind each signature argument positionally. A
        single ``single_qubit_program`` argument instead consumes the whole
        point as framework-native witness-state angles, so exhaustive replay
        can quantify over input quantum states with schema-1 numeric points.
        """
        arguments = _point_arguments(point)
        signature = contract.signature.arguments
        if len(signature) == 1 and signature[0].value_type == "single_qubit_program" and len(arguments) == 2:
            witness = task6_witness_state(self.framework, float(arguments[0]), float(arguments[1]))
            return (witness,)
        return arguments

    def _semantic_execution_cases(
        self,
        contract: Contract,
        task_id: str,
        code: str | None,
        execution: ExecutionResult,
        primary_arguments: tuple[Any, ...],
    ) -> tuple[SemanticExecutionCase, ...]:
        """Execute every declared point in a finite exhaustive parameter domain.

        Args:
            contract: Resolved behavior contract.
            task_id: Normalized task identifier.
            code: Candidate source code.
            execution: Already-computed execution for the first domain point.
            primary_arguments: Arguments bound to the primary execution.

        Returns:
            Explicitly bound diagnostic-point executions.
        """
        if code is None or not contract.parameters.diagnostic_points:
            return ()
        quantifier = contract.parameters.quantifier.value
        structured_rotation = contract.parameters.completeness == "structured_rotation_source_identity"
        symbolic_family = contract.parameters.completeness == (
            "bounded_symbolic_projective_identity_with_certified_literals"
        )
        if quantifier != "exhaustive" and not structured_rotation and not symbolic_family:
            return ()
        entry_point = self.tasks[task_id]["entry_point"]
        points = list(contract.parameters.diagnostic_points)
        if structured_rotation or symbolic_family:
            points.extend(_family_probe_points(self.suite, task_id, len(points[0])))
        cases = []
        for index, point in enumerate(points):
            # Structured rotation families take one vector argument; exhaustive
            # finite domains bind each parameter positionally.
            if structured_rotation:
                call_args: tuple[Any, ...] = (list(_point_arguments(point)),)
                reuse = call_args == primary_arguments
            elif symbolic_family:
                call_args = self._bind_point(contract, point)
                reuse = call_args == primary_arguments
            else:
                call_args = self._bind_point(contract, point)
                # The first diagnostic point defines the primary execution, so
                # its already-computed result is reused. Comparing bound
                # framework objects (circuits, QNodes, arrays) is not reliable.
                reuse = index == 0
            case_execution = (
                execution
                if reuse
                else self._dispatcher.execute(
                    task_id=task_id,
                    code=code,
                    entry_point=entry_point,
                    call_args=call_args,
                )
            )
            cases.append(
                SemanticExecutionCase(
                    call_args,
                    case_execution,
                )
            )
        return tuple(cases)

    def _contract(self, task_id: str) -> Contract:
        if self._contracts is None:
            raise ValueError(f"behavior contracts are unavailable for suite {self.suite!r}")
        return self._contracts.get(self.suite, task_id)


def build_evaluator(
    framework: Framework,
    suite: Suite = "core",
    *,
    semantic_verifier: SemanticVerifier | None = None,
) -> Evaluator:
    """Construct a behavior-first evaluator from bundled tasks and contracts.

    Args:
        framework: Framework whose candidate code is evaluated.
        suite: Benchmark suite whose assets are loaded.
        semantic_verifier: Optional verifier portfolio override.

    Returns:
        Behavior-authoritative evaluator.
    """
    return Evaluator(
        framework,
        suite,
        load_tasks(framework, suite),
        load_tasks("qiskit", suite),
        semantic_verifier=semantic_verifier,
    )


def _family_probe_points(suite: Suite, task_id: str, width: int) -> list[tuple[float, ...]]:
    """Derive deterministic pseudo-random probe points for family contracts.

    Structured-family contracts quantify over all real parameters. When the
    syntactic proof engines are inconclusive, the numeric fallback compares
    candidate behavior at the contract's diagnostic points plus these extra
    points, so a candidate cannot pass by matching only the published points.
    The seed is derived from suite and task id, keeping grading reproducible.

    Args:
        suite: Benchmark suite identifier.
        task_id: Normalized task identifier.
        width: Number of real parameters per point.

    Returns:
        Extra parameter points in ``[-pi, pi)``.
    """
    import hashlib
    import math
    import random

    seed = int.from_bytes(hashlib.sha256(f"{suite}:{task_id}:family_probe".encode()).digest()[:8], "big")
    generator = random.Random(seed)
    return [tuple(generator.uniform(-math.pi, math.pi) for _ in range(width)) for _ in range(3)]


def _point_arguments(point: tuple[Any, ...]) -> tuple[Any, ...]:
    """Preserve each diagnostic-point literal exactly as declared.

    Float-typed parameters stay floats even at integral values: coercing
    ``0.0`` to ``0`` breaks strictly typed candidate signatures (for example
    CUDA-Q builder kernels assert on integer rotation angles).
    """

    return tuple(point)


def _load_contracts(suite: Suite) -> ContractRegistry | None:
    try:
        return ContractRegistry.from_package(suite)
    except FileNotFoundError:
        return None


def _admission_failure(contract: Contract, suite: Suite, framework: Framework) -> dict[str, Any] | None:
    """Return execution-error details for non-authoritative contracts."""
    if contract.shadow_only:
        return _execution_error_details(suite, contract.task_id, framework, reason="contract_shadow_only")
    if contract.audit_status is AuditStatus.BLOCKED:
        return _execution_error_details(suite, contract.task_id, framework, reason="contract_audit_blocked")
    return None


def _behavior_details(
    result: VerifierResult,
    contract: Contract,
    framework: Framework,
) -> dict[str, Any]:
    record = make_result_record(
        result,
        contract,
        framework=framework,
        authoritative=True,
    )
    metric_evidence = next((item for item in result.evidence if item.metric is not None), None)
    passed = result.status is SemanticStatus.VERIFIED_PASS
    details = {
        "passed": passed,
        "verified_status": "verified_pass" if passed else "verified_fail",
        "semantic_status": result.status.value,
        "reason": result.reason,
        "metric": None if metric_evidence is None else metric_evidence.value,
        "metric_name": None if metric_evidence is None else metric_evidence.metric,
        "semantic_verification": record,
        "behavior_verdict": {
            "passed": passed,
            "source": "behavior",
            "semantic_status": result.status.value,
        },
        "score_authority": "behavior",
    }
    if contract.parameters.quantifier.value == "exhaustive":
        details["grader_type"] = "semantic_contract"
        details["num_cases"] = len(contract.parameters.diagnostic_points)
    return details


def _execution_error_details(
    suite: Suite,
    task_id: str,
    framework: Framework,
    *,
    reason: str = "behavior_contract_unavailable",
) -> dict[str, Any]:
    """Return an authoritative execution error for an invalid grader setup."""
    record = make_execution_error_result_record(
        suite=suite,
        task_id=task_id,
        framework=framework,
        reason=reason,
    )
    return {
        "passed": False,
        "verified_status": "verified_fail",
        "semantic_status": SemanticStatus.EXECUTION_ERROR.value,
        "reason": reason,
        "metric": None,
        "metric_name": None,
        "semantic_verification": record,
        "behavior_verdict": {
            "passed": False,
            "source": "behavior",
            "semantic_status": SemanticStatus.EXECUTION_ERROR.value,
        },
        "score_authority": "behavior",
    }
