"""CUDA-Q candidate execution and probability extraction."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from qceval.evals.models import ExecutionResult
from qceval.evals.probabilities import as_prob_array
from qceval.evals.sandbox import execute_with_entry_point
from qceval.frameworks.cudaq.counts import (
    _array_metadata,
    _cudaq_counts_to_probabilities,
    _num_qubits,
    _probabilities_from_array,
    _project_measured_probabilities,
    _unitary_from_array,
)
from qceval.frameworks.cudaq.metadata import (
    _allocated_qubits_from_code,
    _base_metadata,
    _has_explicit_measurements,
    _measurement_indices_from_code,
    _operation_metadata_from_code,
)
from qceval.frameworks.cudaq.runtime import (
    _double_precision_target,
    _ignore_cudaq_deprecations,
    _import_cudaq,
    _is_argument_count_error,
    _is_cudaq_kernel,
    _kernel_arg_options,
    _require_kernel_arity,
)

SAMPLE_FALLBACK_SHOTS = 8192
SAMPLE_FALLBACK_SEED = 42


def execute_cudaq_task(
    *,
    task_id: str,
    code: str,
    entry_point: str,
    inputs: dict[str, Any],
    call_args: tuple[Any, ...] | None = None,
    output_qubits: Sequence[int] | None = None,
) -> ExecutionResult:
    """Execute CUDA-Q candidate code for one task.

    Candidate functions may return a CUDA-Q kernel, a counts mapping, or a
    probability vector.  Kernels are evaluated with statevector simulation when
    available and sampled as a deterministic fallback.

    Args:
        task_id: Zero-padded task identifier.
        code: Candidate Python source.
        entry_point: Function name to call.
        inputs: Deterministic task inputs keyed by task id.
        call_args: Positional arguments from the behavior contract signature.
            When omitted, the entry point is invoked with no arguments.
        output_qubits: Optional compact output register qubits from task assets.

    Returns:
        Normalized execution result containing probabilities and metadata.

    Raises:
        RuntimeError: If CUDA-Q is not importable.
        TypeError: If candidate output is not a supported CUDA-Q result shape.
        Exception: Any candidate or framework exception raised during execution.
    """
    _reject_builder_register_iteration(code)
    if call_args is None:
        from qceval.semantics.contracts.binding import call_args_from_code

        input_value = inputs.get(task_id)
        call_args = call_args_from_code(code, entry_point, input_value)
    cudaq = _import_cudaq()
    metadata = _base_metadata(cudaq)
    return execute_with_entry_point(
        code,
        entry_point,
        lambda function: _execute_case_entry_point(
            cudaq,
            function=function,
            metadata=metadata,
            code=code,
            task_id=task_id,
            entry_point=entry_point,
            call_args=call_args,
            output_qubits=output_qubits,
        ),
    )


def _reject_builder_register_iteration(code: str) -> None:
    """Reject non-terminating iteration over ``make_kernel`` registers.

    CUDA-Q's dynamic builder register supports indexing but its Python object
    does not terminate iteration. ``for qubit in register`` therefore spins
    while constructing the kernel instead of raising ``IndexError``. Detect
    that source shape before executing untrusted candidate code.
    """
    tree = ast.parse(code)
    builders: set[str] = set()
    registers: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            continue
        target = node.targets[0].id
        function = node.value.func
        if _ast_call_name(function) == "make_kernel":
            builders.add(target)
        elif (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in builders
            and function.attr in {"qalloc", "qvector", "qreg"}
        ):
            registers.add(target)
    if any(
        isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id in registers
        for node in ast.walk(tree)
    ):
        raise TypeError("CUDA-Q builder qubit registers must be indexed; direct iteration does not terminate")


def _ast_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _execute_case_entry_point(
    cudaq: Any,
    *,
    function: Any,
    metadata: dict[str, Any],
    code: str,
    task_id: str,
    entry_point: str,
    call_args: tuple[Any, ...],
    output_qubits: Sequence[int] | None,
) -> ExecutionResult:
    if _is_cudaq_kernel(function):
        return _execute_kernel(
            cudaq,
            kernel=function,
            metadata=metadata,
            code=code,
            task_id=task_id,
            entry_point=entry_point,
            kernel_args=call_args,
            output_qubits=output_qubits,
        )

    result = function(*call_args)
    return _execution_from_result(
        cudaq,
        result=result,
        metadata=metadata,
        code=code,
        task_id=task_id,
        entry_point=entry_point,
        kernel_args=call_args,
        output_qubits=output_qubits,
    )


def _execution_from_result(
    cudaq: Any,
    *,
    result: Any,
    metadata: dict[str, Any],
    code: str,
    task_id: str,
    entry_point: str,
    kernel_args: tuple[Any, ...] | None,
    output_qubits: Sequence[int] | None,
) -> ExecutionResult:
    if callable(result):
        return _execute_kernel(
            cudaq,
            kernel=result,
            metadata=metadata,
            code=code,
            task_id=task_id,
            entry_point=entry_point,
            # Replay needs these arguments to rebuild any factory closure.
            # The returned kernel may independently accept the same arguments;
            # the arity-safe retry loop below selects either ``kernel_args`` or
            # ``()`` without launching an incompatible CUDA-Q signature.
            kernel_args=kernel_args,
            replay_args=kernel_args,
            output_qubits=output_qubits,
        )
    if isinstance(result, Mapping):
        return _counts_result(result, metadata=metadata)
    unitary = _unitary_from_array(result)
    if unitary is not None:
        unitary_probabilities = np.zeros(unitary.shape[0], dtype=float)
        unitary_probabilities[0] = 1.0
        metadata.update(
            {
                "probability_method": "returned_unitary",
                "num_qubits": _num_qubits(unitary_probabilities),
                "measurement_count": 0,
                "non_measurement_operation_count": 0,
                "entangling_gate_count": 0,
                "circuit_depth": 0,
                "repeated_block_count": 0,
                "measurement_pairs": [],
                "has_measurements": False,
            }
        )
        return ExecutionResult(
            probabilities=unitary_probabilities.tolist(),
            metadata=metadata,
            unitary=unitary,
            circuit=None,
        )
    probabilities = _probabilities_from_array(result)
    if probabilities is not None:
        metadata.update(_array_metadata(probabilities, method="returned_probabilities"))
        return ExecutionResult(probabilities=probabilities.tolist(), metadata=metadata, unitary=None)
    raise TypeError(f"Expected CUDA-Q kernel, counts dict, or probabilities, got {type(result).__name__} instead.")


def _attempt_kernel_args(
    cudaq: Any,
    *,
    kernel: Any,
    args: tuple[Any, ...],
    metadata: dict[str, Any],
    code: str,
    entry_point: str,
    measured_qubits: list[int],
    explicit_measurements: bool,
    errors: list[Exception],
    replay_args: tuple[Any, ...] | None = None,
) -> tuple[np.ndarray, int, np.ndarray | None] | None:
    """Try one kernel argument combination.

    Returns ``(probabilities, n_qubits, statevector)`` on success, ``None`` to try
    the next arity option, or raises when the failure is not arity-related.
    """
    # Launching a kernel with the wrong runtime-argument arity can SIGSEGV
    # inside CUDA-Q instead of raising; fail typed up front.
    try:
        _require_kernel_arity(kernel, args)
    except TypeError as arity_exc:
        errors.append(arity_exc)
        return None
    try:
        if explicit_measurements:
            source_args = args if replay_args is None else replay_args
            statevector = _terminal_measurement_statevector(code, entry_point, source_args)
            metadata["probability_method"] = "statevector_replay"
        else:
            with _ignore_cudaq_deprecations(), _double_precision_target(cudaq):
                statevector = np.asarray(cudaq.get_state(kernel, *args), dtype=complex)
            metadata["probability_method"] = "statevector"
        n_qubits = _num_qubits(statevector)
        state_probabilities = as_prob_array(np.abs(statevector) ** 2)
        probabilities = _project_measured_probabilities(state_probabilities, measured_qubits)
        metadata["kernel_argument_count"] = len(args if replay_args is None else replay_args)
        return probabilities, n_qubits, statevector
    except Exception as state_exc:
        try:
            cudaq.set_random_seed(SAMPLE_FALLBACK_SEED)
            with _ignore_cudaq_deprecations():
                counts = cudaq.sample(kernel, *args, shots_count=SAMPLE_FALLBACK_SHOTS)
            probabilities = _cudaq_counts_to_probabilities(counts, from_sample=True)
            n_qubits = _allocated_qubits_from_code(code) or _num_qubits(probabilities)
            metadata["probability_method"] = "sample_fallback"
            metadata["shots_count"] = SAMPLE_FALLBACK_SHOTS
            metadata["seed"] = SAMPLE_FALLBACK_SEED
            metadata["statevector_error"] = f"{type(state_exc).__name__}: {state_exc}"
            metadata["kernel_argument_count"] = len(args if replay_args is None else replay_args)
            return probabilities, n_qubits, None
        except Exception as sample_exc:
            if _is_conditional_feedback_error(sample_exc):
                # Kernels that branch on measurement results cannot be
                # sampled by CUDA-Q; the exact branch simulation of the
                # lowered IR is the deterministic equivalent.
                try:
                    probabilities = _dynamic_branch_probabilities(code, entry_point, args)
                except NotImplementedError as unsupported:
                    raise RuntimeError(f"unsupported dynamic CUDA-Q kernel: {unsupported}") from sample_exc
                n_qubits = _allocated_qubits_from_code(code) or _num_qubits(probabilities)
                metadata["probability_method"] = "dynamic_branch_simulation"
                metadata["kernel_argument_count"] = len(args)
                return probabilities, n_qubits, None
            errors.append(sample_exc)
            if _is_argument_count_error(state_exc) or _is_argument_count_error(sample_exc):
                return None
            raise sample_exc from state_exc


def _execute_kernel(
    cudaq: Any,
    *,
    kernel: Any,
    metadata: dict[str, Any],
    code: str,
    task_id: str,
    entry_point: str,
    kernel_args: tuple[Any, ...] | None = None,
    replay_args: tuple[Any, ...] | None = None,
    output_qubits: Sequence[int] | None = None,
) -> ExecutionResult:
    # Prefer statevector probabilities for exact grading; sampled fallback keeps
    # kernels usable on targets that cannot expose state.
    measured_qubits = list(output_qubits) if output_qubits is not None else _measurement_indices_from_code(code)
    explicit_measurements = _has_explicit_measurements(code)
    source_args = tuple(kernel_args or ()) if replay_args is None else replay_args
    _validate_dynamic_source(code, entry_point, source_args, explicit_measurements)
    errors: list[Exception] = []
    for args in _kernel_arg_options(kernel_args):
        attempt = _attempt_kernel_args(
            cudaq,
            kernel=kernel,
            args=args,
            metadata=metadata,
            code=code,
            entry_point=entry_point,
            measured_qubits=measured_qubits,
            explicit_measurements=explicit_measurements,
            errors=errors,
            replay_args=replay_args,
        )
        if attempt is not None:
            probabilities, n_qubits, statevector = attempt
            break
    else:
        if errors:
            raise errors[-1]
        raise RuntimeError("CUDA-Q kernel execution failed")
    metadata["num_qubits"] = n_qubits
    metadata.update(_operation_metadata_from_code(code, default_measurement_qubits=measured_qubits))
    return ExecutionResult(
        probabilities=probabilities.tolist(),
        metadata=metadata,
        unitary=None,
        circuit=kernel,
        statevector=statevector,
    )


def _is_conditional_feedback_error(error: Exception) -> bool:
    message = str(error).lower()
    return "branch on measurement" in message or "conditional feedback" in message


def _terminal_measurement_statevector(code: str, entry_point: str, args: tuple[Any, ...]) -> np.ndarray:
    """Replay a measured kernel without terminal measurements for exact state."""
    from qceval.frameworks.cudaq.dynamic import has_conditional_feedback
    from qceval.frameworks.cudaq.replay import simulate_basis_cudaq

    if has_conditional_feedback(code, entry_point):
        raise ValueError("measurement-conditioned kernels require branch simulation")
    return simulate_basis_cudaq(
        code,
        entry_point,
        prep={},
        strip_leading_x_on=set(),
        call_args=args,
    )


def _validate_dynamic_source(
    code: str,
    entry_point: str,
    args: tuple[Any, ...],
    explicit_measurements: bool,
) -> None:
    """Reject conditional kernels outside the bounded dynamic grammar."""
    from qceval.frameworks.cudaq.dynamic import has_conditional_feedback, lower_dynamic_kernel
    from qceval.frameworks.cudaq.program import CudaqProgram
    from qceval.semantics.lowering.base import SourceMetadata

    if not explicit_measurements or not has_conditional_feedback(code, entry_point):
        return
    try:
        lower_dynamic_kernel(CudaqProgram(code, entry_point, args), SourceMetadata("cudaq", "", None))
    except NotImplementedError as exc:
        raise RuntimeError(f"unsupported dynamic CUDA-Q kernel: {exc}") from exc


def _dynamic_branch_probabilities(code: str, entry_point: str, args: tuple[Any, ...]) -> np.ndarray:
    """Exact measured distribution of a feed-forward kernel via branch simulation.

    Imported lazily: the semantic lowering is only needed on this path and the
    executor module must stay importable without it.
    """
    from qceval.frameworks.cudaq.dynamic import lower_dynamic_kernel
    from qceval.frameworks.cudaq.program import CudaqProgram
    from qceval.semantics.lowering.base import SourceMetadata
    from qceval.semantics.verifiers.dynamic import ExactBranchSimulator

    program = lower_dynamic_kernel(
        CudaqProgram(code, entry_point, tuple(args)),
        SourceMetadata("cudaq", "", None),
    )
    if program.num_clbits == 0:
        raise ValueError("feed-forward kernel declares no measured bits")
    out = np.zeros(1 << program.num_clbits, dtype=float)
    for branch in ExactBranchSimulator().run(program, max_branches=4096):
        index = sum(bit << position for position, bit in enumerate(branch.classical_bits))
        out[index] += branch.probability
    return as_prob_array(out)


def _counts_result(counts: Mapping[Any, Any], *, metadata: dict[str, Any]) -> ExecutionResult:
    probabilities = _cudaq_counts_to_probabilities(counts)
    metadata.update(_array_metadata(probabilities, method="returned_counts"))
    return ExecutionResult(probabilities=probabilities.tolist(), metadata=metadata, unitary=None)
