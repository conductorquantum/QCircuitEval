"""Dispatch framework-native programs into the dense circuit IR."""

from __future__ import annotations

from typing import Any

from qceval.evals.ir.core import Circuit
from qceval.frameworks.cirq.parser import from_cirq
from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.frameworks.pennylane.parser import from_pennylane
from qceval.frameworks.qiskit.parser import from_qiskit


def from_framework(
    obj: Any,
    *,
    framework: str,
    code: str | None = None,
    entry_point: str | None = None,
) -> Circuit:
    """Convert a framework-native circuit or program into a neutral circuit.

    Args:
        obj: Framework-native circuit, tape, or CUDA-Q program carrier.
        framework: Source framework name.
        code: CUDA-Q source code when ``obj`` is not a ``CudaqProgram``.
        entry_point: CUDA-Q entry-point name when ``obj`` is not a
            ``CudaqProgram``.

    Returns:
        The framework-neutral circuit representation.

    Raises:
        NotImplementedError: If the framework or a source operation is not
            supported by the unitary representation.
        ValueError: If a CUDA-Q entry point or gate definition is invalid.
    """
    if framework == "qiskit":
        return from_qiskit(obj)
    if framework == "cirq":
        return from_cirq(obj)
    if framework == "pennylane":
        return from_pennylane(obj)
    if framework == "cudaq":
        from qceval.frameworks.cudaq.parser import from_cudaq

        program = (
            obj
            if isinstance(obj, CudaqProgram)
            else CudaqProgram(
                code=str(code),
                entry_point=str(entry_point),
            )
        )
        return from_cudaq(program)
    raise NotImplementedError(f"unsupported framework: {framework!r}")
