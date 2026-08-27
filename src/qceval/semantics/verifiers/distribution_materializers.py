"""Exact-distribution materializers for named classical observations.

Materializers turn framework-neutral Program IR (or evaluator-normalized
execution probabilities) into sparse exact probability tables over contracted
classical variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from qceval.semantics.contracts import Contract
from qceval.semantics.contracts.kinds import BitOrder
from qceval.semantics.verifiers.base import VerificationContext
from qceval.semantics.verifiers.classical_wires import contracted_classical_variables, measured_render_order


@dataclass(frozen=True)
class ProbabilityTable:
    """Sparse finite probability table over named binary variables."""

    variables: tuple[str, ...]
    rows: tuple[tuple[tuple[str, ...], float], ...]
    alternatives: tuple[tuple[tuple[tuple[str, ...], float], ...], ...] = ()


class DistributionMaterializer(Protocol):
    """Produce exact candidate probabilities from framework-neutral semantics."""

    def distribution(self, context: VerificationContext) -> ProbabilityTable:
        """Materialize one exact joint distribution.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Exact candidate probability table.
        """
        ...


class AdaptiveDistributionMaterializer:
    """Prefer exact execution probabilities, falling back to Program IR branches.

    Sampled executor fallbacks (CUDA-Q ``sample_fallback`` and Qiskit
    ``qasm_fallback``) carry shot noise far above exact tolerances, so those
    observations are ignored in favor of the lowered program distribution.
    """

    _SAMPLED_METHODS = frozenset({"sample_fallback", "qasm_fallback"})

    def __init__(self, simulator: Any | None = None) -> None:
        self._program = ProgramDistributionMaterializer(simulator)

    def distribution(self, context: VerificationContext) -> ProbabilityTable:
        """Materialize one exact joint distribution.

        Args:
            context: Contract, candidate Program IR, and optional execution data.

        Returns:
            Exact candidate probability table.
        """
        metadata = context.execution_metadata or {}
        expected_width = 2 ** len(measured_render_order(context.program))
        if (
            context.program.provenance.framework == "cudaq"
            or metadata.get("probability_method") in self._SAMPLED_METHODS
            or context.execution_probabilities is None
            or len(context.execution_probabilities) != expected_width
        ):
            return self._program.distribution(context)
        return ExecutionDistributionMaterializer(context.execution_probabilities).distribution(context)


class ProgramDistributionMaterializer:
    """Derive exact declared classical probabilities from Program IR branches."""

    def __init__(self, simulator: Any | None = None) -> None:
        """Initialize a Program IR distribution materializer.

        Args:
            simulator: Optional exact branch simulator.
        """
        if simulator is None:
            from qceval.semantics.verifiers.dynamic import ExactBranchSimulator

            simulator = ExactBranchSimulator()
        self._simulator = simulator

    def distribution(self, context: VerificationContext) -> ProbabilityTable:
        """Aggregate exact branch mass over declared classical variables.

        Args:
            context: Contract and candidate Program IR.

        Returns:
            Sparse exact probability table.
        """
        contract = context.contract
        bindings = _distribution_bindings(context)
        variables = tuple(variable for variable, _ in bindings)
        indices = tuple(bit for _, bit in bindings)
        probabilities: dict[tuple[str, ...], float] = {}
        for branch in self._simulator.run(context.program, max_branches=contract.limits.max_branches):
            outcome = tuple(str(branch.classical_bits[index]) for index in indices)
            probabilities[outcome] = probabilities.get(outcome, 0.0) + branch.probability
        return ProbabilityTable(variables, tuple(sorted(probabilities.items())))


class ExecutionDistributionMaterializer:
    """Expose evaluator-normalized exact probabilities as a semantic table."""

    def __init__(self, probabilities: Any) -> None:
        self._probabilities = tuple(float(value) for value in probabilities)

    def distribution(self, context: VerificationContext) -> ProbabilityTable:
        """Return probabilities in the validated public render order.

        Args:
            context: Contract and lowered-program context.

        Returns:
            Dense execution probabilities as a named exact table.
        """

        render = measured_render_order(context.program)
        bindings = _distribution_bindings(context)
        positions = {bit: position for position, bit in enumerate(render)}
        if any(bit not in positions for _, bit in bindings):
            raise ValueError("contracted classical output is absent from execution render order")
        variables = tuple(variable for variable, _ in bindings)
        expected = 2 ** len(render)
        if len(self._probabilities) != expected:
            raise ValueError("execution probability width differs from contracted observation")
        rows = []
        for index, probability in enumerate(self._probabilities):
            rendered = format(index, f"0{len(render)}b")
            outcome = tuple(rendered[positions[bit]] for _, bit in bindings)
            rows.append((outcome, probability))
        return ProbabilityTable(variables, tuple(rows))


def _distribution_bindings(context: VerificationContext) -> tuple[tuple[str, int], ...]:
    """Return observed and projectable extra variables in contract order."""
    observation = context.contract.observation
    observed = contracted_classical_variables(
        context.contract,
        context.program,
        observation.classical,
        require_all=True,
    )
    extras = contracted_classical_variables(
        context.contract,
        context.program,
        (*observation.ignored, *observation.marginalize),
        require_all=False,
    )
    return (*observed, *extras)


def _observed_variables(contract: Contract) -> tuple[str, ...]:
    systems = {item.name: item for item in contract.systems.items}
    values: list[str] = []
    for name in contract.observation.classical:
        system = systems[name]
        indices = system.indices
        if contract.observation.bit_order is BitOrder.LITTLE_ENDIAN:
            indices = tuple(reversed(indices))
        values.extend(f"{name}[{index}]" for index in indices)
    return tuple(values)
