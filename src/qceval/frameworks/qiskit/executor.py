"""Qiskit candidate execution and circuit introspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qceval.evals.models import ExecutionResult
from qceval.evals.sandbox import execute_code_with_args, get_handler
from qceval.frameworks.qiskit.metadata import (
    circuit_metadata,
    circuit_unitary,
    circuit_without_measurements,
    measurement_pairs,
)


def execute_qiskit_task(
    *,
    task_id: str,
    code: str,
    entry_point: str,
    inputs: dict[str, Any],
    call_args: tuple[Any, ...] | None = None,
    output_qubits: Sequence[int] | None = None,
) -> ExecutionResult:
    """Execute Qiskit candidate code for one task.

    Args:
        task_id: Zero-padded task identifier.
        code: Candidate Python source.
        entry_point: Function name to call.
        inputs: Deterministic task inputs keyed by task id.
        call_args: Optional positional arguments for case-table execution.
        output_qubits: Declared output register from task assets. Accepted for
            dispatcher parity; Qiskit derives measurement order from the circuit.

    Returns:
        Normalized execution result containing probabilities, metadata, unitary,
        and circuit object when available.

    Raises:
        TypeError: If candidate returns neither a Qiskit circuit-like object nor
            counts dictionary.
        Exception: Any candidate or framework exception raised during execution.
    """
    del output_qubits
    if call_args is not None:
        result = execute_code_with_args(code, entry_point, *call_args)
    else:
        result = get_handler(task_id, code, entry_point, inputs)
    if isinstance(result, dict):
        return ExecutionResult(
            probabilities=counts_to_array(result).tolist(),
            metadata={
                "returned_counts": True,
                "probability_method": "returned_counts",
                "gate_family_counts": {},
                "interaction_pairs": [],
            },
        )
    if not isinstance(result, QuantumCircuit) and not hasattr(result, "data"):
        raise TypeError(f"Expected QuantumCircuit or dict, got {type(result)} instead.")
    return _execute_circuit(result)


def counts_to_array(counts: dict[str, int] | list[dict[str, int]]) -> np.ndarray:
    """Convert Qiskit counts to an integer-ordered probability vector.

    Args:
        counts: Counts dictionary, or the first counts dictionary from a list.

    Returns:
        Normalized probability vector.

    Raises:
        TypeError: If ``counts`` is not a dictionary or list of dictionaries.
        ValueError: If the counts dictionary is empty.
    """
    if isinstance(counts, list):
        counts = counts[0]
    if not isinstance(counts, dict):
        raise TypeError(f"Expected dict or list of dicts, got {type(counts)}")
    cleaned = _clean_counts(counts)
    n_bits = len(next(iter(cleaned)))
    out = np.array([cleaned.get(format(i, f"0{n_bits}b"), 0.0) for i in range(2**n_bits)], dtype=float)
    total = float(out.sum())
    return out / total if total > 0 else out


def exact_probabilities(circuit: QuantumCircuit) -> np.ndarray:
    """Compute exact probabilities for measured classical bits.

    Terminal measurements are removed before statevector simulation.  If no
    measurements are present, all qubits are treated as measured in index
    order.  Circuits whose measurements are non-terminal (a measured qubit is
    operated on again, a classical bit is overwritten from another qubit, or a
    later operation is classically conditioned) are routed through exact
    measurement deferral instead, because removing those measurements discards
    the collapse and yields the wrong recorded-bit distribution.

    Args:
        circuit: Qiskit circuit to simulate.

    Returns:
        Probability vector ordered by classical bit integer index.

    Raises:
        ValueError: If the circuit requires measurement deferral but contains
            classically conditioned operations that cannot be deferred exactly.
    """
    pairs = measurement_pairs(circuit)
    n_bits = max(clbit for _, clbit in pairs) + 1 if pairs else circuit.num_qubits
    if pairs and requires_measurement_deferral(circuit):
        return _deferred_exact_probabilities(circuit, n_bits)
    pairs = pairs or [(i, i) for i in range(circuit.num_qubits)]
    basis_probs = np.asarray(
        Statevector.from_instruction(circuit_without_measurements(circuit)).probabilities(), dtype=float
    )
    out = np.zeros(2**n_bits, dtype=float)
    for basis_index, probability in enumerate(basis_probs):
        # Qiskit statevector basis uses qubit indexes, but benchmark output is
        # ordered by the measured classical register integer value.
        out[_classical_index(basis_index, pairs)] += float(probability)
    total = float(out.sum())
    return out / total if total > 0 else out


def requires_measurement_deferral(circuit: QuantumCircuit) -> bool:
    """Return whether exact probabilities need measurement deferral.

    Removing measurements before statevector simulation is only valid when
    every measurement is terminal.  Deferral is required when a measured qubit
    is operated on again, when a classical bit is overwritten from a different
    qubit, or when any operation is classically conditioned.

    Args:
        circuit: Qiskit circuit whose measurement placement is inspected.

    Returns:
        Whether terminal-measurement removal would be semantically unsound.
    """
    measured_qubits: set[Any] = set()
    clbit_sources: dict[Any, Any] = {}
    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name
        if name == "barrier":
            continue
        if name == "measure":
            qubit = instruction.qubits[0]
            clbit = instruction.clbits[0]
            if clbit in clbit_sources and clbit_sources[clbit] is not qubit:
                return True
            clbit_sources[clbit] = qubit
            measured_qubits.add(qubit)
            continue
        if instruction.clbits or getattr(operation, "condition", None) is not None:
            return True
        if measured_qubits.intersection(instruction.qubits):
            return True
    return False


def _deferred_exact_probabilities(circuit: QuantumCircuit, n_bits: int) -> np.ndarray:
    """Compute exact recorded-bit probabilities via measurement deferral.

    Every measurement is replaced by a CNOT onto a fresh ancilla so collapse
    is captured unitarily; the final classical value of each bit is read from
    the ancilla of the last measurement writing it.  Resets are replaced by a
    SWAP with a fresh zero ancilla, which is exact and avoids the probabilistic
    collapse ``Statevector`` would otherwise perform.
    """
    from qiskit.circuit import QuantumRegister

    ancilla_count = sum(1 for instruction in circuit.data if instruction.operation.name in {"measure", "reset"})
    ancillas = QuantumRegister(ancilla_count, "qceval_deferred")
    deferred = circuit.copy_empty_like()
    deferred.add_register(ancillas)
    final_ancilla: dict[int, int] = {}
    next_ancilla = 0
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name == "measure":
            ancilla = ancillas[next_ancilla]
            next_ancilla += 1
            deferred.cx(instruction.qubits[0], ancilla)
            clbit_index = circuit.find_bit(instruction.clbits[0]).index
            final_ancilla[clbit_index] = deferred.find_bit(ancilla).index
            continue
        if operation.name == "reset":
            ancilla = ancillas[next_ancilla]
            next_ancilla += 1
            deferred.swap(instruction.qubits[0], ancilla)
            continue
        if instruction.clbits or getattr(operation, "condition", None) is not None:
            raise ValueError("classically conditioned operations cannot be measurement-deferred exactly")
        deferred.append(operation, instruction.qubits, [])
    pairs = [(qubit_index, clbit_index) for clbit_index, qubit_index in final_ancilla.items()]
    basis_probs = np.asarray(Statevector.from_instruction(deferred).probabilities(), dtype=float)
    out = np.zeros(2**n_bits, dtype=float)
    for basis_index, probability in enumerate(basis_probs):
        out[_classical_index(basis_index, pairs)] += float(probability)
    total = float(out.sum())
    return out / total if total > 0 else out


def _execute_circuit(circuit: QuantumCircuit) -> ExecutionResult:
    metadata = circuit_metadata(circuit)
    statevector = None
    try:
        deferred = requires_measurement_deferral(circuit)
        probabilities = exact_probabilities(circuit)
        if deferred:
            metadata["probability_method"] = "deferred_statevector"
        else:
            statevector = Statevector.from_instruction(circuit_without_measurements(circuit)).data
            metadata["probability_method"] = "statevector"
    except Exception as exc:
        probabilities = _qasm_probabilities(circuit)
        metadata["probability_method"] = "qasm_fallback"
        metadata["statevector_error"] = f"{type(exc).__name__}: {exc}"
    return ExecutionResult(
        probabilities=probabilities.tolist(),
        metadata=metadata,
        unitary=circuit_unitary(circuit),
        circuit=circuit,
        statevector=statevector,
    )


def _clean_counts(counts: dict[str, int]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in counts.items():
        clean_key = str(key).split()[0]
        cleaned[clean_key] = cleaned.get(clean_key, 0.0) + float(value)
    if not cleaned:
        raise ValueError("counts dictionary is empty")
    return cleaned


def _classical_index(basis_index: int, pairs: list[tuple[int, int]]) -> int:
    classical_index = 0
    for qubit_index, clbit_index in pairs:
        if (basis_index >> qubit_index) & 1:
            classical_index |= 1 << clbit_index
    return classical_index


def _qasm_probabilities(circuit: QuantumCircuit) -> np.ndarray:
    # Statevector extraction can fail for unsupported operations or control
    # flow; seeded QASM keeps the fallback reproducible.
    sim = AerSimulator(seed_simulator=42)
    compiled = transpile(circuit, sim)
    counts = sim.run(compiled, shots=2048, seed_simulator=42).result().get_counts()
    return counts_to_array(counts)
