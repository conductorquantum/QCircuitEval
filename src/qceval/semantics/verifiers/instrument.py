"""Exact outcome-probability and conditional-state instrument verification."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qceval.semantics.targets import load_contract_target_document
from qceval.semantics.verifiers.base import CostEstimate, EngineDescriptor, VerificationContext
from qceval.semantics.verifiers.classical_wires import measured_render_order
from qceval.semantics.verifiers.dynamic import DynamicSimulationError, ExactBranchSimulator, reduced_density_matrix
from qceval.semantics.verifiers.materialize import CandidateSemanticError
from qceval.semantics.verifiers.requirements import conditional_quantum_wires
from qceval.semantics.verifiers.result import (
    RESULT_SCHEMA_VERSION,
    EvidenceRecord,
    SemanticStatus,
    VerifierResult,
)

INSTRUMENT_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class InstrumentBranch:
    """One outcome probability and normalized conditional density object."""

    outcome: str
    probability: float
    conditional: np.ndarray


@dataclass(frozen=True)
class InstrumentMaterialization:
    """Complete finite outcome-labeled instrument observation."""

    branches: tuple[InstrumentBranch, ...]
    omitted_mass: float = 0.0


class InstrumentMaterializer(Protocol):
    """Produce one exact candidate instrument observation."""

    def instrument(self, context: VerificationContext) -> InstrumentMaterialization:
        """Materialize candidate branches.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Complete exact instrument observation.
        """
        ...


class InstrumentTargetProvider(Protocol):
    """Load an independently derived instrument target."""

    def instrument(self, context: VerificationContext) -> InstrumentMaterialization:
        """Load target branches.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Complete target instrument observation.
        """
        ...


class ProgramInstrumentMaterializer:
    """Derive outcome-conditioned reduced states by exact IR branch simulation."""

    def __init__(self, simulator: ExactBranchSimulator | None = None) -> None:
        """Initialize a Program IR instrument materializer.

        Args:
            simulator: Optional exact branch simulator.
        """
        self._simulator = simulator or ExactBranchSimulator()

    def instrument(self, context: VerificationContext) -> InstrumentMaterialization:
        """Simulate and aggregate declared classical/quantum outputs.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Exact outcome-conditioned density matrices.
        """
        contract = context.contract
        systems = {item.name: item for item in contract.systems.items}
        classical_wires = measured_render_order(context.program)
        quantum_wires = conditional_quantum_wires(contract, context.program) or tuple(
            index for name in contract.observation.quantum for index in systems[name].indices
        )
        branches = self._simulator.run(
            context.program,
            max_branches=contract.limits.max_branches,
        )
        probabilities: dict[str, float] = {}
        weighted: dict[str, np.ndarray] = {}
        for branch in branches:
            outcome = "".join(str(branch.classical_bits[index]) for index in classical_wires)
            density = reduced_density_matrix(branch.statevector, quantum_wires, context.program.num_qubits)
            probabilities[outcome] = probabilities.get(outcome, 0.0) + branch.probability
            weighted[outcome] = weighted.get(outcome, np.zeros_like(density)) + branch.probability * density
        values = tuple(
            InstrumentBranch(outcome, probability, weighted[outcome] / probability)
            for outcome, probability in sorted(probabilities.items())
            if probability > np.finfo(float).tiny
        )
        return InstrumentMaterialization(values)


class PackagedInstrumentTargetProvider:
    """Read deterministic audited phase-estimation instrument targets."""

    def instrument(self, context: VerificationContext) -> InstrumentMaterialization:
        """Load a hash-verified outcome/conditional-state artifact.

        Args:
            context: Contract identifying the target artifact.

        Returns:
            Exact target instrument branches.
        """
        target = load_contract_target_document(context.contract).get("target")
        if not isinstance(target, dict) or target.get("type") != "phase_estimation_instrument":
            raise ValueError("target is not a phase-estimation instrument")
        outcomes = target.get("outcomes")
        if not isinstance(outcomes, dict):
            raise ValueError("instrument outcomes are missing")
        branches = []
        for outcome, spec in sorted(outcomes.items()):
            if not isinstance(spec, dict):
                raise ValueError("instrument branch must be an object")
            probability = _probability(spec.get("probability"))
            state = _basis_density(str(spec.get("conditional_target")))
            branches.append(InstrumentBranch(str(outcome), probability, state))
        return InstrumentMaterialization(tuple(branches))


class InstrumentEngine:
    """Compare all branch probabilities and conditional quantum states."""

    def __init__(self, materializer: InstrumentMaterializer, targets: InstrumentTargetProvider) -> None:
        """Initialize an exact instrument engine.

        Args:
            materializer: Candidate instrument materializer.
            targets: Independent target provider.
        """
        self._materializer = materializer
        self._targets = targets

    def descriptor(self) -> EngineDescriptor:
        """Return immutable capability metadata.

        Returns:
            Instrument engine identity and capabilities.
        """
        return EngineDescriptor(
            "instrument_exact",
            INSTRUMENT_ENGINE_VERSION,
            ("instrument",),
            ("instrument", "dynamic", "conditional_state", "framework_neutral_ir"),
        )

    def estimate(self, context: VerificationContext) -> CostEstimate:
        """Estimate exact branch materialization cost.

        Args:
            context: Verification context.

        Returns:
            Conservative branch/state cost.
        """
        dimension = 2**context.program.num_qubits
        branches = context.contract.limits.max_branches
        memory = max(1, math.ceil(branches * dimension * 16 / 2**20))
        cases = min(2**context.program.num_clbits, context.contract.limits.max_cases)
        return CostEstimate(
            context.program.num_qubits,
            dimension,
            cases,
            memory,
            min(10.0, branches / 1000),
            branches,
        )

    def verify(self, context: VerificationContext) -> VerifierResult:
        """Compare complete outcome and conditional-state behavior.

        Args:
            context: Verification context.

        Returns:
            Decisive or operational exact instrument result.
        """
        started = time.perf_counter()
        try:
            actual = self._materializer.instrument(context)
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
        target = self._targets.instrument(context)
        error, cases, residual = _instrument_error(actual, target, context.contract.approximation.uncertainty)
        if error is None:
            return _result(
                context,
                SemanticStatus.SEMANTIC_FAIL,
                "malformed_instrument",
                None,
                cases,
                time.perf_counter() - started,
                residual,
            )
        tolerance = context.contract.approximation.tolerance
        uncertainty = context.contract.approximation.uncertainty + actual.omitted_mass + target.omitted_mass
        if error <= max(0.0, tolerance - uncertainty):
            status, reason = SemanticStatus.VERIFIED_PASS, "instrument_within_pass_bound"
        elif error > tolerance + uncertainty:
            status, reason = SemanticStatus.SEMANTIC_FAIL, "instrument_exceeds_fail_bound"
        else:
            status, reason = SemanticStatus.EXECUTION_ERROR, "instrument_in_uncertainty_band"
        return _result(context, status, reason, error, cases, time.perf_counter() - started, residual)


def _instrument_error(
    actual: InstrumentMaterialization,
    target: InstrumentMaterialization,
    branch_floor: float,
) -> tuple[float | None, int, float]:
    actual_values = _validate(actual)
    target_values = _validate(target)
    if actual_values is None or target_values is None:
        return None, 0, math.inf
    outcomes = sorted(set(actual_values) | set(target_values))
    probability_error = 0.5 * sum(
        abs(actual_values.get(outcome, (0.0, None))[0] - target_values.get(outcome, (0.0, None))[0])
        for outcome in outcomes
    )
    conditional_error = 0.0
    for outcome in outcomes:
        actual_probability, actual_density = actual_values.get(outcome, (0.0, None))
        target_probability, target_density = target_values.get(outcome, (0.0, None))
        if max(actual_probability, target_probability) <= branch_floor:
            continue
        if actual_density is None or target_density is None or actual_density.shape != target_density.shape:
            return 1.0, len(outcomes), 0.0
        difference = actual_density - target_density
        scale = max(1.0, float(np.linalg.norm(target_density)))
        conditional_error = max(conditional_error, float(np.linalg.norm(difference) / scale))
    residual = max(abs(sum(value[0] for value in actual_values.values()) - 1.0), actual.omitted_mass)
    return max(probability_error, conditional_error), len(outcomes), residual


def _validate(value: InstrumentMaterialization) -> dict[str, tuple[float, np.ndarray | None]] | None:
    if not math.isfinite(value.omitted_mass) or value.omitted_mass < 0 or value.omitted_mass >= 1:
        return None
    result: dict[str, tuple[float, np.ndarray | None]] = {}
    for branch in value.branches:
        density = np.asarray(branch.conditional, dtype=np.complex128)
        if (
            not branch.outcome
            or branch.outcome in result
            or not math.isfinite(branch.probability)
            or branch.probability < 0
            or density.ndim != 2
            or density.shape[0] != density.shape[1]
            or not np.all(np.isfinite(density))
            or np.linalg.norm(density - density.conjugate().T) > 1e-9
            or abs(float(np.trace(density).real) - 1.0) > 1e-9
            or float(np.min(np.linalg.eigvalsh(density))) < -1e-9
        ):
            return None
        result[branch.outcome] = (branch.probability, density)
    if abs(sum(item[0] for item in result.values()) + value.omitted_mass - 1.0) > 1e-9:
        return None
    return result


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError("invalid branch probability")
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return int(numerator) / int(denominator)
    return float(value)


def _basis_density(bitstring: str) -> np.ndarray:
    if not bitstring or any(value not in {"0", "1"} for value in bitstring):
        raise ValueError("invalid conditional basis state")
    state = np.zeros(2 ** len(bitstring), dtype=np.complex128)
    state[int(bitstring, 2)] = 1.0
    return np.outer(state, state.conjugate())


def _result(
    context: VerificationContext,
    status: SemanticStatus,
    reason: str,
    value: float | None,
    cases: int,
    elapsed: float,
    residual: float = 0.0,
) -> VerifierResult:
    evidence = EvidenceRecord(
        "instrument_exact",
        INSTRUMENT_ENGINE_VERSION,
        reason,
        context.input_hash,
        context.target_hash,
        metric="max_branch_choi_distance" if value is not None else None,
        value=value,
        tolerance=context.contract.approximation.tolerance if value is not None else None,
        uncertainty=context.contract.approximation.uncertainty if value is not None else None,
        cases_checked=cases,
        elapsed_seconds=elapsed,
        preconditions=(f"normalization_residual={residual:.17g}",),
    )
    return VerifierResult(
        RESULT_SCHEMA_VERSION,
        status,
        reason,
        context.contract_hash,
        context.target_hash,
        INSTRUMENT_ENGINE_VERSION,
        (evidence,),
    )
