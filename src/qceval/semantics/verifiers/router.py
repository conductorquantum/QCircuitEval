"""Data-driven verifier routing, preflight, and fail-closed reconciliation."""

from __future__ import annotations

import time

from qceval.semantics.contracts import Contract, RouteSpec, contract_hash
from qceval.semantics.ir import Program, program_hash
from qceval.semantics.telemetry import EventSink, event_now
from qceval.semantics.verifiers.base import CostEstimate, EngineDescriptor, VerificationContext, VerifierEngine
from qceval.semantics.verifiers.dynamic import DynamicSimulationError
from qceval.semantics.verifiers.registry import VerifierRegistry
from qceval.semantics.verifiers.result import (
    RESULT_SCHEMA_VERSION,
    EvidenceRecord,
    SemanticStatus,
    VerifierResult,
)

ROUTER_VERSION = "1.0.0"


class VerifierRouter:
    """Run contract-declared engines without task-specific branches."""

    def __init__(
        self,
        registry: VerifierRegistry,
        *,
        event_sink: EventSink | None = None,
        correlation_id: str = "local",
    ) -> None:
        """Initialize a router.

        Args:
            registry: Available semantic engines.
            event_sink: Optional bounded telemetry sink.
            correlation_id: Run-local event correlation identifier.
        """
        self._registry = registry
        self._event_sink = event_sink
        self._correlation_id = correlation_id

    def verify(
        self,
        contract: Contract,
        program: Program,
        *,
        arguments: tuple[object, ...] = (),
        execution_metadata: dict[str, object] | None = None,
        execution_probabilities: tuple[float, ...] | None = None,
    ) -> VerifierResult:
        """Route one lowered candidate according to its contract.

        Args:
            contract: Validated task contract.
            program: Framework-neutral Program IR.
            arguments: Positional entry-point arguments for argument-dependent targets.
            execution_metadata: Optional executor observations for materializer adaptation.
            execution_probabilities: Optional exact probabilities from framework execution.

        Returns:
            Reconciled fail-closed semantic result.
        """
        context = VerificationContext(
            contract=contract,
            contract_hash=contract_hash(contract),
            target_hash=contract.target.sha256,
            input_hash=program_hash(program),
            program=program,
            arguments=arguments,
            execution_metadata=execution_metadata,
            execution_probabilities=execution_probabilities,
        )
        started = time.perf_counter()
        self._emit("semantic_route_started", context, phase="routing")
        routes = contract.routing.primary
        if len(routes) != 1 or contract.routing.fallback:
            result = _router_result(
                SemanticStatus.EXECUTION_ERROR,
                "invalid_verifier_routing",
                context,
                (),
            )
            self._emit_result(context, result, started)
            return result
        result = reconcile_results(self._run_routes(routes, context), context)
        self._emit_result(context, result, started)
        return result

    def _run_routes(
        self,
        routes: tuple[RouteSpec, ...],
        context: VerificationContext,
    ) -> tuple[VerifierResult, ...]:
        results = []
        for index, route in enumerate(routes):
            engine = self._registry.get(route.engine)
            result = _run_engine(engine, route, context)
            results.append(result)
            if (
                index == 0
                and not route.cross_check
                and result.status
                in {
                    SemanticStatus.VERIFIED_PASS,
                    SemanticStatus.SEMANTIC_FAIL,
                }
            ):
                break
        return tuple(results)

    def _emit_result(self, context: VerificationContext, result: VerifierResult, started: float) -> None:
        self._emit(
            "semantic_route_completed",
            context,
            phase="routing",
            status=result.status.value,
            reason=result.reason,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _emit(self, event: str, context: VerificationContext, **fields: str | float | None) -> None:
        if self._event_sink is None:
            return
        self._event_sink.emit(
            event_now(
                event,
                self._correlation_id,
                context.contract_hash,
                context.target_hash,
                context.input_hash,
                **fields,
            )
        )


def reconcile_results(
    results: tuple[VerifierResult, ...],
    context: VerificationContext,
) -> VerifierResult:
    """Reconcile required engine evidence without choosing favorable results.

    Args:
        results: Ordered primary verifier results.
        context: Version and target context.

    Returns:
        Reconciled result.
    """
    if not results:
        return _router_result(SemanticStatus.EXECUTION_ERROR, "no_verifier_route", context, ())
    evidence = tuple(item for result in results for item in result.evidence)
    decisive = {result.status for result in results if result.status in _DECISIVE}
    if len(decisive) > 1:
        return _router_result(SemanticStatus.EXECUTION_ERROR, "cross_check_disagreement", context, evidence)
    if SemanticStatus.SEMANTIC_FAIL in decisive:
        return _router_result(SemanticStatus.SEMANTIC_FAIL, "semantic_failure", context, evidence)
    if SemanticStatus.VERIFIED_PASS in decisive:
        if any(result.status not in _DECISIVE for result in results):
            return _router_result(SemanticStatus.EXECUTION_ERROR, "required_cross_check_nondecisive", context, evidence)
        return _router_result(SemanticStatus.VERIFIED_PASS, "all_required_routes_passed", context, evidence)
    status = _nonsemantic_status(results)
    return _router_result(status, f"all_routes_{status.value}", context, evidence)


_DECISIVE = frozenset({SemanticStatus.VERIFIED_PASS, SemanticStatus.SEMANTIC_FAIL})
_NONSEMANTIC_PRECEDENCE = (
    SemanticStatus.EXECUTION_ERROR,
    SemanticStatus.RESOURCE_LIMIT,
)


def _run_engine(
    engine: VerifierEngine | None,
    route: RouteSpec,
    context: VerificationContext,
) -> VerifierResult:
    if engine is None:
        return _router_result(SemanticStatus.EXECUTION_ERROR, f"missing_engine:{route.engine}", context, ())
    descriptor = engine.descriptor()
    if context.contract.kind.value not in descriptor.kinds:
        return _router_result(SemanticStatus.EXECUTION_ERROR, "engine_kind_unsupported", context, ())
    if not set(route.capabilities).issubset(descriptor.capabilities):
        return _router_result(SemanticStatus.EXECUTION_ERROR, "engine_capability_missing", context, ())
    result = _guarded_engine_verify(engine, descriptor, context)
    if result.contract_hash != context.contract_hash or result.target_hash != context.target_hash:
        return _router_result(SemanticStatus.EXECUTION_ERROR, "engine_version_mismatch", context, result.evidence)
    return result


def _guarded_engine_verify(
    engine: VerifierEngine,
    descriptor: EngineDescriptor,
    context: VerificationContext,
) -> VerifierResult:
    try:
        estimate = engine.estimate(context)
        if _over_limit(estimate, context.contract):
            return _router_result(SemanticStatus.RESOURCE_LIMIT, "preflight_resource_limit", context, ())
        return engine.verify(context)
    except MemoryError:
        return _router_result(SemanticStatus.RESOURCE_LIMIT, "engine_memory_limit", context, ())
    except DynamicSimulationError as exc:
        # Typed simulation failures preserve execution-error/resource-limit
        # classification and their detailed reason code.
        return _router_result(exc.status, str(exc), context, ())
    except NotImplementedError:
        return _router_result(SemanticStatus.EXECUTION_ERROR, "engine_materialization_unsupported", context, ())
    except Exception as exc:  # noqa: BLE001 - engine failures become stable result data.
        evidence = (
            EvidenceRecord(
                engine=descriptor.name,
                engine_version=descriptor.version,
                reason=f"engine_exception:{type(exc).__name__}",
                input_hash=context.input_hash,
                target_hash=context.target_hash,
            ),
        )
        return _router_result(SemanticStatus.EXECUTION_ERROR, "engine_exception", context, evidence)


def _over_limit(estimate: CostEstimate, contract: Contract) -> bool:
    limits = contract.limits
    return (
        estimate.qubits > limits.max_qubits
        or estimate.dimension > limits.max_dimension
        or estimate.cases > limits.max_cases
        or estimate.branches > limits.max_branches
        or estimate.memory_mib > limits.memory_mib
        or estimate.wall_seconds > limits.wall_seconds
    )


def _nonsemantic_status(results: tuple[VerifierResult, ...]) -> SemanticStatus:
    statuses = {result.status for result in results}
    return next(status for status in _NONSEMANTIC_PRECEDENCE if status in statuses)


def _router_result(
    status: SemanticStatus,
    reason: str,
    context: VerificationContext,
    evidence: tuple[EvidenceRecord, ...],
) -> VerifierResult:
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        reason,
        context.contract_hash,
        context.target_hash,
        ROUTER_VERSION,
        evidence,
    )
