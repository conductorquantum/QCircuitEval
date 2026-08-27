"""PennyLane candidate execution and tape introspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pennylane as qml

from qceval.evals.models import ExecutionResult
from qceval.evals.probabilities import num_bits
from qceval.evals.sandbox import execute_code_with_args, get_handler
from qceval.frameworks.pennylane.metadata import _measurement_wires, _metadata_from_tape, _unitary_from_tape


def execute_pennylane_task(
    *,
    task_id: str,
    code: str,
    entry_point: str,
    inputs: dict[str, Any],
    call_args: tuple[Any, ...] | None = None,
    output_qubits: Sequence[int] | None = None,
) -> ExecutionResult:
    """Execute PennyLane candidate code for one task.

    The executor temporarily wraps ``qml.QNode.__call__`` to capture the most
    recent tape.  When a tape is captured, probabilities and metadata are
    derived from the tape; otherwise array-like candidate returns are normalized
    directly.

    Args:
        task_id: Zero-padded task identifier.
        code: Candidate Python source.
        entry_point: Function name to call.
        inputs: Deterministic task inputs keyed by task id.
        call_args: Optional positional arguments for case-table execution.
        output_qubits: Declared output register from task assets, used when a
            measurement omits explicit wires.

    Returns:
        Normalized execution result containing probabilities, metadata, unitary,
        and captured tape when available.

    Raises:
        TypeError: If candidate output cannot be interpreted as probabilities or
            samples.
        Exception: Any candidate or framework exception raised during execution.
    """
    tapes: list[Any] = []
    original_call = qml.QNode.__call__
    # User functions return arbitrary arrays or samples, not tapes.  Recording
    # QNode calls gives structural graders access to the executed operations.
    qml.QNode.__call__ = _recording_call(original_call, tapes)
    try:
        if call_args is not None:
            result = execute_code_with_args(code, entry_point, *call_args)
        else:
            result = get_handler(task_id, code, entry_point, inputs)
    finally:
        qml.QNode.__call__ = original_call
    tape = tapes[-1] if tapes else None
    if tape is not None:
        if _has_mid_circuit_measurements(tape) or _can_use_qnode_probabilities(tape, output_wires=output_qubits):
            probabilities = _probabilities_from_qnode_result(result)
        else:
            probabilities = _probabilities_from_tape(tape, output_wires=output_qubits)
        if probabilities is None:
            probabilities = _probabilities_from_tape(tape, output_wires=output_qubits)
    else:
        probabilities = _probabilities_from_array(result)
    if tape is not None:
        metadata = _metadata_from_tape(tape, output_wires=output_qubits)
    else:
        metadata = {
            "probability_method": "returned_probabilities",
            "num_qubits": num_bits(probabilities),
            "measurement_count": 0,
            "return_measurement_count": 0,
            "non_measurement_operation_count": 0,
            "circuit_depth": 0,
            "repeated_block_count": 0,
            "measurement_pairs": [],
            "measurement_wires": [],
            "operation_counts": {},
            "gate_family_counts": {},
            "interaction_pairs": [],
            "entangling_gate_count": 0,
        }
    return ExecutionResult(
        probabilities=probabilities.tolist(),
        metadata=metadata,
        unitary=None if tape is None else _unitary_from_tape(tape),
        circuit=tape,
    )


def _recording_call(original_call: Any, tapes: list[Any]) -> Any:
    # Candidate entry points may call helper QNodes (for example a supplied
    # state-producing QNode) from inside the returned QNode.  Only top-level
    # tapes describe the returned result, so nested calls are not recorded.
    depth = 0

    def recording_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal depth
        top_level = depth == 0
        depth += 1
        try:
            if top_level:
                tapes.append(self.construct(args, kwargs))
            return original_call(self, *args, **kwargs)
        finally:
            depth -= 1

    return recording_call


def _probabilities_from_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
        total = float(np.sum(arr))
        if total > 0 and np.all(arr >= -1e-12):
            return np.asarray(arr, dtype=float) / total
    if arr.ndim == 1:
        return _samples_1d_to_probs(arr)
    if arr.ndim == 2:
        return _samples_2d_to_probs(arr)
    raise TypeError(f"Expected 1-D or 2-D PennyLane array, got shape {arr.shape}")


def _probabilities_from_qnode_result(values: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.floating):
        return None
    try:
        return _probabilities_from_array(arr)
    except TypeError:
        return None


def _can_use_qnode_probabilities(tape: Any, *, output_wires: Sequence[int] | None) -> bool:
    """Return whether a QNode probability result already has normalized order."""
    if len(tape.measurements) != 1 or not isinstance(tape.measurements[0], qml.measurements.ProbabilityMP):
        return False
    declared = list(tape.measurements[0].wires)
    return bool(declared) and declared == _measurement_wires(tape, output_wires=output_wires)


def _probabilities_from_tape(tape: Any, *, output_wires: Sequence[int] | None = None) -> np.ndarray:
    # Recompute analytic probabilities from recorded operations so sample-based
    # candidate code can be graded without shot noise.
    wire_order = list(tape.wires)
    measured = _measurement_wires(tape, output_wires=output_wires)
    basis_probs = np.abs(_statevector_from_tape(tape)) ** 2
    if not measured:
        # State read-out (qml.state()): the full-register distribution is the
        # result; there is no projective measurement to marginalize onto.
        total = float(basis_probs.sum())
        return basis_probs / total if total > 0 else basis_probs
    measured_positions = [wire_order.index(wire) for wire in measured]
    out = np.zeros(2 ** len(measured), dtype=float)
    for basis_index, probability in enumerate(basis_probs):
        out[_measured_index(basis_index, measured_positions, len(wire_order))] += float(probability)
    total = float(out.sum())
    return out / total if total > 0 else out


def _has_mid_circuit_measurements(tape: Any) -> bool:
    """Return whether a tape contains measurement values used during execution.

    Args:
        tape: Captured PennyLane tape.

    Returns:
        ``True`` when a mid-circuit measurement operation is present.
    """
    return any(type(operation).__name__ == "MidMeasureMP" for operation in tape.operations)


def _statevector_from_tape(tape: Any) -> np.ndarray:
    """Return final statevector for a captured PennyLane tape."""
    wire_order = list(tape.wires)
    state_tape = qml.tape.QuantumScript(tape.operations, [qml.state()])
    device = qml.device("default.qubit", wires=wire_order, shots=None)
    result = qml.execute((state_tape,), device, diff_method=None)[0]
    return np.asarray(result, dtype=complex)


def _samples_1d_to_probs(arr: np.ndarray) -> np.ndarray:
    counts = np.zeros(2, dtype=float)
    for sample in arr.tolist():
        counts[1 if sample > 0 else 0] += 1
    total = float(np.sum(counts))
    return counts / total if total > 0 else counts


def _samples_2d_to_probs(arr: np.ndarray) -> np.ndarray:
    counts = np.zeros(2 ** arr.shape[1], dtype=float)
    for row in arr.astype(int).tolist():
        idx = 0
        for bit in row:
            idx = (idx << 1) | int(bit)
        counts[idx] += 1
    total = float(np.sum(counts))
    return counts / total if total > 0 else counts


def _measured_index(basis_index: int, positions: list[int], n_wires: int) -> int:
    measured_index = 0
    for position in positions:
        bit = (basis_index >> (n_wires - position - 1)) & 1
        measured_index = (measured_index << 1) | bit
    return measured_index
