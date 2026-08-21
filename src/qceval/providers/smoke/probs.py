"""Probability fallback source generation for smoke provider."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

from qceval.evals.evaluator import load_tasks
from qceval.evals.inputs import global_inputs
from qceval.evals.probabilities import bitstring_index
from qceval.frameworks.qiskit import execute_qiskit_task
from qceval.models import ProviderRequest
from qceval.providers.smoke.utils import _normalize, _num_bits


def _probabilities_for(request: ProviderRequest, spec: Mapping[str, Any]) -> list[float] | None:
    grader_type = spec.get("type")
    if grader_type == "support_uniformity":
        return _support_probabilities(request, spec)
    if grader_type == "peak_match":
        return _peak_probabilities(request, spec)
    if grader_type == "exact_distribution":
        expected = spec.get("expected_distribution")
        if isinstance(expected, Mapping):
            return _distribution_from_mapping(expected)
        return _canonical_probabilities(request.task_id)
    return None


def _support_probabilities(request: ProviderRequest, spec: Mapping[str, Any]) -> list[float]:
    support = spec.get("support") or spec.get("canonical_support") or spec.get("expected_support")
    if support == "all":
        n_bits = _infer_qubit_count(request)
        support = [format(index, f"0{n_bits}b") for index in range(2**n_bits)]
    if not isinstance(support, Sequence) or isinstance(support, str):
        raise ValueError("support_uniformity requires support bitstrings")
    return _uniform_distribution([str(bitstring) for bitstring in support])


def _peak_probabilities(request: ProviderRequest, spec: Mapping[str, Any]) -> list[float]:
    if isinstance(spec.get("accepted_peak_sets"), Sequence):
        first_set = spec["accepted_peak_sets"][0]
        return _uniform_distribution([str(bitstring) for bitstring in first_set])
    if isinstance(spec.get("expected_peaks"), Sequence):
        return _uniform_distribution([str(bitstring) for bitstring in spec["expected_peaks"]])
    return _canonical_probabilities(request.task_id)


def _distribution_from_mapping(expected: Mapping[str, Any]) -> list[float]:
    bitstrings = [str(bitstring) for bitstring in expected]
    n_bits = len(bitstrings[0])
    probabilities = [0.0] * (2**n_bits)
    for bitstring, probability in expected.items():
        probabilities[bitstring_index(str(bitstring))] = float(probability)
    return _normalize(probabilities)


def _uniform_distribution(bitstrings: Sequence[str]) -> list[float]:
    n_bits = len(bitstrings[0])
    probabilities = [0.0] * (2**n_bits)
    mass = 1.0 / len(bitstrings)
    for bitstring in bitstrings:
        probabilities[bitstring_index(bitstring)] = mass
    return probabilities


@cache
def _canonical_probabilities(task_id: str) -> list[float]:
    from qceval.semantics.contracts.binding import call_args_from_code

    normalized_id = str(task_id).zfill(2)
    task = load_tasks("qiskit")[normalized_id]
    inputs = global_inputs("qiskit")
    input_value = inputs.get(normalized_id)
    call_args = call_args_from_code(task["canonical_solution"], task["entry_point"], input_value)
    execution = execute_qiskit_task(
        task_id=normalized_id,
        code=task["canonical_solution"],
        entry_point=task["entry_point"],
        inputs=inputs,
        call_args=call_args,
    )
    return [float(probability) for probability in execution.probabilities]


def _probability_code(entry_point: str, framework: str, probabilities: Sequence[float]) -> str | None:
    normalized = _normalize(probabilities)
    if framework == "cirq":
        return _cirq_probability_code(entry_point, normalized)
    if framework == "pennylane":
        return _pennylane_probability_code(entry_point, normalized)
    if framework == "cudaq":
        return _cudaq_probability_code(entry_point, normalized)
    return None


def _cirq_probability_code(entry_point: str, probabilities: Sequence[float]) -> str:
    n_bits = _num_bits(probabilities)
    counts = {format(index, f"0{n_bits}b"): float(prob) for index, prob in enumerate(probabilities) if prob > 0.0}
    return f"def {entry_point}(*args, **kwargs):\n    return {counts!r}\n"


def _pennylane_probability_code(entry_point: str, probabilities: Sequence[float]) -> str:
    return f"import numpy as np\n\ndef {entry_point}(*args, **kwargs):\n    return np.array({list(probabilities)!r})\n"


def _cudaq_probability_code(entry_point: str, probabilities: Sequence[float]) -> str:
    return f"import numpy as np\n\ndef {entry_point}(*args, **kwargs):\n    return np.array({list(probabilities)!r})\n"


def _infer_qubit_count(request: ProviderRequest) -> int:
    match = re.search(r"_(\d+)$", request.entry_point)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*(?:qubit|qubits|wire|wires)", request.prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1
