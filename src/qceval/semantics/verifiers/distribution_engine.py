"""Exact distribution engine with projection and Hellinger-infidelity comparison.

Compares candidate probability tables against independently packaged exact
targets after declared marginalization over unobserved classical variables.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Protocol

from qceval.evals.probabilities import hellinger_infidelity
from qceval.semantics.contracts import Contract
from qceval.semantics.targets import load_contract_target_document
from qceval.semantics.verifiers.base import CostEstimate, EngineDescriptor, VerificationContext
from qceval.semantics.verifiers.distribution_materializers import (
    DistributionMaterializer,
    ProbabilityTable,
    _observed_variables,
)
from qceval.semantics.verifiers.distribution_targets import analytic_distribution
from qceval.semantics.verifiers.dynamic import DynamicSimulationError
from qceval.semantics.verifiers.materialize import CandidateSemanticError
from qceval.semantics.verifiers.result import (
    RESULT_SCHEMA_VERSION,
    EvidenceRecord,
    SemanticStatus,
    VerifierResult,
)

DISTRIBUTION_ENGINE_VERSION = "1.1.0"


class DistributionTargetProvider(Protocol):
    """Load an independently generated exact target distribution."""

    def distribution(self, context: VerificationContext) -> ProbabilityTable:
        """Load one exact target distribution.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Independently generated target table.
        """
        ...


class DistributionEngine:
    """Compare the exact contracted classical observation and no stronger object."""

    def __init__(self, materializer: DistributionMaterializer, targets: DistributionTargetProvider) -> None:
        """Initialize an exact distribution engine.

        Args:
            materializer: Candidate exact-probability materializer.
            targets: Independent exact target provider.
        """
        self._materializer = materializer
        self._targets = targets

    def descriptor(self) -> EngineDescriptor:
        """Return immutable capability metadata.

        Returns:
            Distribution engine identity and capabilities.
        """
        return EngineDescriptor(
            "distribution_exact",
            DISTRIBUTION_ENGINE_VERSION,
            ("distribution",),
            ("distribution", "exact", "partial_observation", "framework_neutral_ir"),
        )

    def estimate(self, context: VerificationContext) -> CostEstimate:
        """Estimate complete outcome enumeration.

        Args:
            context: Verification context.

        Returns:
            Conservative finite-table cost.
        """
        variables = len(_observed_variables(context.contract))
        cases = min(2**variables, context.contract.limits.max_cases)
        return CostEstimate(context.program.num_qubits, cases, cases, max(1, math.ceil(cases * 24 / 2**20)), 0.1)

    def verify(self, context: VerificationContext) -> VerifierResult:
        """Compare exact normalized distributions after declared marginalization.

        Args:
            context: Verification context.

        Returns:
            Decisive or execution-error result.
        """
        started = time.perf_counter()
        contract = context.contract
        if contract.observation.postselection is not None:
            return _result(
                context,
                SemanticStatus.EXECUTION_ERROR,
                "postselection_route_required",
                None,
                0,
                time.perf_counter() - started,
            )
        metric = contract.approximation.metric
        if metric not in _METRICS:
            return _result(
                context,
                SemanticStatus.EXECUTION_ERROR,
                "distribution_metric_unsupported",
                None,
                0,
                time.perf_counter() - started,
            )
        observed = _observed_variables(contract)
        allowed_extra = set(_unobserved_variables(contract))
        try:
            materialized = self._materializer.distribution(context)
        except CandidateSemanticError as exc:
            return _result(
                context,
                SemanticStatus.SEMANTIC_FAIL,
                exc.reason,
                None,
                0,
                time.perf_counter() - started,
            )
        except DynamicSimulationError as exc:
            return _result(context, exc.status, exc.reason, None, 0, time.perf_counter() - started)
        actual, actual_sanity = _project(materialized, observed, allowed_extra)
        target_table = self._targets.distribution(context)
        target_candidates = (target_table.rows, *target_table.alternatives)
        projected_targets = [
            _project(ProbabilityTable(target_table.variables, rows), observed, set()) for rows in target_candidates
        ]
        if actual is None or any(target is None for target, _sanity in projected_targets):
            return _result(
                context,
                SemanticStatus.SEMANTIC_FAIL,
                "malformed_probability_table",
                None,
                0,
                time.perf_counter() - started,
                max(actual_sanity, *(sanity for _target, sanity in projected_targets)),
            )
        error = min(_METRICS[metric](actual, target) for target, _sanity in projected_targets if target is not None)
        target_sanity = max(sanity for _target, sanity in projected_targets)
        tolerance = contract.approximation.tolerance
        uncertainty = contract.approximation.uncertainty
        if error <= max(0.0, tolerance - uncertainty):
            status, reason = SemanticStatus.VERIFIED_PASS, f"{metric}_within_pass_bound"
        elif error > tolerance + uncertainty:
            status, reason = SemanticStatus.SEMANTIC_FAIL, f"{metric}_exceeds_fail_bound"
        else:
            status, reason = SemanticStatus.EXECUTION_ERROR, f"{metric}_in_uncertainty_band"
        return _result(
            context,
            status,
            reason,
            error,
            len(actual),
            time.perf_counter() - started,
            max(actual_sanity, target_sanity),
            metric=metric,
        )


class PackagedDistributionTargetProvider:
    """Read audited exact-distribution target artifacts from package data."""

    def __init__(self, arguments: tuple[Any, ...] = ()) -> None:
        self._arguments = arguments

    def distribution(self, context: VerificationContext) -> ProbabilityTable:
        """Load a hash-verified exact probability map.

        Args:
            context: Contract identifying the packaged target.

        Returns:
            Exact target probability table.

        Raises:
            ValueError: If the artifact is mismatched or not an exact table.
        """
        value = load_contract_target_document(context.contract)
        target = value.get("target")
        if not isinstance(target, dict):
            raise ValueError("target is not a distribution specification")
        arguments = context.arguments if context.arguments else self._arguments
        target = _argument_distribution_target(target, arguments)
        probabilities = target.get("probabilities")
        variables = _observed_variables(context.contract)
        if probabilities is None:
            probabilities = analytic_distribution(target, len(variables), arguments)
        rows = _probability_rows(probabilities, len(variables))
        raw_alternatives = target.get("accepted_probability_maps", [])
        if not isinstance(raw_alternatives, list):
            raise ValueError("accepted distribution alternatives are malformed")
        alternatives = tuple(_probability_rows(item, len(variables)) for item in raw_alternatives)
        return ProbabilityTable(variables, rows, alternatives)


def _probability_rows(probabilities: Any, width: int) -> tuple[tuple[tuple[str, ...], float], ...]:
    if not isinstance(probabilities, dict):
        raise ValueError("distribution probabilities are missing")
    rows = []
    for outcome, raw_probability in sorted(probabilities.items()):
        if not isinstance(outcome, str) or len(outcome) != width:
            raise ValueError("target outcome width mismatch")
        rows.append((tuple(outcome), _exact_probability(raw_probability)))
    return tuple(rows)


def _argument_distribution_target(target: dict[str, Any], arguments: tuple[Any, ...]) -> dict[str, Any]:
    if target.get("type") != "argument_distribution_cases":
        return target
    cases = target.get("cases")
    if not isinstance(cases, list):
        raise ValueError("argument-distribution target cases are missing")
    for case in cases:
        if not isinstance(case, dict) or case.get("arguments") != list(arguments):
            continue
        distribution = case.get("distribution")
        if not isinstance(distribution, dict):
            raise ValueError("argument-distribution target case is malformed")
        return distribution
    raise ValueError(f"no distribution target for arguments {arguments!r}")


def _unobserved_variables(contract: Contract) -> tuple[str, ...]:
    systems = {item.name: item for item in contract.systems.items}
    names = (*contract.observation.ignored, *contract.observation.marginalize)
    return tuple(f"{name}[{index}]" for name in names for index in systems[name].indices)


def _project(
    table: ProbabilityTable,
    observed: tuple[str, ...],
    allowed_extra: set[str],
) -> tuple[dict[tuple[str, ...], float] | None, float]:
    if len(table.variables) != len(set(table.variables)) or not set(observed).issubset(table.variables):
        return None, math.inf
    extras = set(table.variables) - set(observed)
    if not extras.issubset(allowed_extra):
        return None, math.inf
    indices = tuple(table.variables.index(name) for name in observed)
    projected = dict.fromkeys(itertools.product(("0", "1"), repeat=len(observed)), 0.0)
    seen = set()
    total = 0.0
    for outcome, probability in table.rows:
        if outcome in seen or len(outcome) != len(table.variables) or any(value not in {"0", "1"} for value in outcome):
            return None, math.inf
        if not math.isfinite(probability) or probability < 0:
            return None, math.inf
        seen.add(outcome)
        total += probability
        projected[tuple(outcome[index] for index in indices)] += probability
    sanity = abs(total - 1.0)
    if sanity > 1e-9:
        return None, sanity
    return projected, sanity


def _total_variation(
    actual: dict[tuple[str, ...], float],
    target: dict[tuple[str, ...], float],
) -> float:
    """Return the total-variation distance for certified approximate bounds."""
    return 0.5 * sum(abs(actual[key] - target[key]) for key in actual)


def _probability_vector(table: dict[tuple[str, ...], float]) -> tuple[float, ...]:
    """Return a stable basis-ordered vector from a projected probability map."""
    return tuple(table[key] for key in sorted(table))


_METRICS = {
    "hellinger_infidelity": lambda actual, target: hellinger_infidelity(
        _probability_vector(actual),
        _probability_vector(target),
    ),
    "total_variation": _total_variation,
}


def _exact_probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError("probability must be numeric or a rational string")
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        result = int(numerator) / int(denominator)
    else:
        result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("probability must be finite and non-negative")
    return result


def _result(
    context: VerificationContext,
    status: SemanticStatus,
    reason: str,
    value: float | None,
    cases: int,
    elapsed: float,
    sanity: float = 0.0,
    metric: str = "hellinger_infidelity",
) -> VerifierResult:
    evidence = EvidenceRecord(
        "distribution_exact",
        DISTRIBUTION_ENGINE_VERSION,
        reason,
        context.input_hash,
        context.target_hash,
        metric=metric if value is not None else None,
        value=value,
        tolerance=context.contract.approximation.tolerance if value is not None else None,
        uncertainty=context.contract.approximation.uncertainty if value is not None else None,
        cases_checked=cases,
        elapsed_seconds=elapsed,
        preconditions=(f"normalization_residual={sanity:.17g}",),
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        reason,
        context.contract_hash,
        context.target_hash,
        DISTRIBUTION_ENGINE_VERSION,
        (evidence,),
    )
