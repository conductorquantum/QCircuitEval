"""Cirq candidate execution and circuit introspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cirq
import numpy as np

from qceval.evals.models import ExecutionResult
from qceval.evals.sandbox import execute_code_with_args, get_handler
from qceval.frameworks.cirq.metadata import (
    _deferred_probabilities,
    circuit_metadata,
    circuit_unitary,
    exact_probabilities,
    exact_statevector,
)


def execute_cirq_task(
    *,
    task_id: str,
    code: str,
    entry_point: str,
    inputs: dict[str, Any],
    call_args: tuple[Any, ...] | None = None,
    output_qubits: Sequence[int] | None = None,
) -> ExecutionResult:
    """Execute Cirq candidate code for one task.

    Args:
        task_id: Zero-padded task identifier.
        code: Candidate Python source.
        entry_point: Function name to call.
        inputs: Deterministic task inputs keyed by task id.
        call_args: Optional positional arguments for case-table execution.
        output_qubits: Declared output register from task assets. Accepted for
            dispatcher parity; Cirq derives measurement order from the circuit.

    Returns:
        Normalized execution result containing probabilities, metadata, unitary,
        and circuit object when available.

    Raises:
        TypeError: If candidate returns neither a Cirq circuit nor counts
            dictionary.
        Exception: Any candidate or framework exception raised during execution.
    """
    del output_qubits
    if call_args is not None:
        result = execute_code_with_args(code, entry_point, *call_args)
    else:
        result = get_handler(task_id, code, entry_point, inputs)
    if isinstance(result, dict):
        return ExecutionResult(
            probabilities=_bitstrings_to_array(result).tolist(),
            metadata={
                "returned_counts": True,
                "probability_method": "returned_counts",
                "gate_family_counts": {},
                "interaction_pairs": [],
            },
        )
    if not isinstance(result, cirq.Circuit):
        raise TypeError(f"Expected Cirq Circuit or dict, got {type(result)} instead.")
    metadata = circuit_metadata(result)
    try:
        statevector = exact_statevector(result)
        probabilities = exact_probabilities(result)
        metadata["probability_method"] = "statevector"
    except ValueError:
        # Mid-circuit measurement or classical control: the circuit has no
        # single final statevector.  Deferring measurements is an exact
        # rewrite, so the measured distribution stays exact.
        probabilities = _deferred_probabilities(result)
        statevector = None
        metadata["probability_method"] = "deferred_statevector"
    return ExecutionResult(
        probabilities=probabilities.tolist(),
        metadata=metadata,
        unitary=circuit_unitary(result),
        circuit=result,
        statevector=statevector,
    )


def _bitstrings_to_array(counts: dict[str, float]) -> np.ndarray:
    cleaned = {"".join(str(key).split()): float(value) for key, value in counts.items()}
    if not cleaned:
        raise ValueError("counts dictionary is empty")
    n_bits = len(next(iter(cleaned)))
    out = np.array([cleaned.get(format(i, f"0{n_bits}b"), 0.0) for i in range(2**n_bits)], dtype=float)
    total = float(out.sum())
    return out / total if total > 0 else out
