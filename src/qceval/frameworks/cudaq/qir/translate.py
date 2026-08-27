"""CUDA-Q compiler-IR translation for semantic lowering.

CUDA-Q kernels are opaque JIT callables rather than inspectable circuit
objects.  The compiler's adaptive QIR is therefore the native post-execution
representation analogous to Qiskit's ``QuantumCircuit``, Cirq's ``Circuit``,
and a PennyLane tape.
"""

from __future__ import annotations

import dis
import importlib.metadata
import warnings
from collections.abc import Iterator
from types import FunctionType
from typing import Any

from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.frameworks.cudaq.qir.cfg import parse_adaptive_qir
from qceval.frameworks.cudaq.qir.models import QirParseLimits
from qceval.frameworks.cudaq.runtime import _import_cudaq, _is_cudaq_kernel, _kernel_arg_options
from qceval.semantics.ir import Program, Provenance
from qceval.semantics.lowering.base import SourceMetadata

QIR_FORMAT = "qir-adaptive"


def lower_cudaq_qir(
    source: CudaqProgram,
    metadata: SourceMetadata,
    *,
    limits: QirParseLimits | None = None,
) -> Program:
    """Compile one concrete CUDA-Q kernel invocation into Program IR.

    Args:
        source: Candidate source carrier with the native kernel produced by
            candidate execution.
        metadata: Candidate provenance supplied by the lowering registry.
        limits: Optional deterministic parser limits.

    Returns:
        Validated framework-neutral Program IR.

    Raises:
        NotImplementedError: If no native kernel is available or CUDA-Q cannot
            translate any of the concrete argument forms used by execution.
        ValueError: If adaptive QIR is malformed or outside the bounded parser.
    """
    if source.kernel is None:
        raise NotImplementedError("CUDA-Q adaptive QIR lowering requires the executed native kernel")
    cudaq = _import_cudaq()
    errors: list[Exception] = []
    kernel, bound_arguments = _unwrap_kernel_callable(source.kernel)
    for arguments in _translation_arguments(bound_arguments, source.call_args):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                text = cudaq.translate(kernel, *arguments, format=QIR_FORMAT)
        except Exception as exc:  # noqa: BLE001 - compiler failures become typed lowering results.
            errors.append(exc)
            continue
        if not isinstance(text, str) or not text.strip() or text.strip() == "{translation failed}":
            errors.append(RuntimeError("CUDA-Q returned no adaptive QIR"))
            continue
        provenance = Provenance(
            "cudaq",
            importlib.metadata.version("cudaq"),
            source_hash=metadata.source_hash,
            backend=metadata.backend,
        )
        return parse_adaptive_qir(text, provenance=provenance, limits=limits)
    detail = "unknown compiler failure" if not errors else f"{type(errors[-1]).__name__}: {errors[-1]}"
    raise NotImplementedError(f"CUDA-Q adaptive QIR translation failed: {detail}")


def _translation_arguments(bound: tuple[Any, ...], requested: tuple[Any, ...]) -> Iterator[tuple[Any, ...]]:
    seen: set[str] = set()
    for candidate in (bound, *_kernel_arg_options(requested)):
        fingerprint = repr(candidate)
        if fingerprint not in seen:
            seen.add(fingerprint)
            yield candidate


def _unwrap_kernel_callable(value: Any) -> tuple[Any, tuple[Any, ...]]:
    """Resolve a zero-argument closure that invokes one native CUDA-Q kernel."""
    if _is_cudaq_kernel(value):
        return value, ()
    closure = _closure_values(value)
    kernel_names = [name for name, item in closure.items() if _is_cudaq_kernel(item)]
    if len(kernel_names) != 1:
        raise NotImplementedError("CUDA-Q kernel wrapper must close over exactly one native kernel")
    kernel_name = kernel_names[0]
    return closure[kernel_name], _wrapper_arguments(value, closure, kernel_name)


def _closure_values(value: Any) -> dict[str, Any]:
    if not isinstance(value, FunctionType) or value.__closure__ is None:
        raise NotImplementedError(f"CUDA-Q QIR requires a native kernel, got {type(value).__name__}")
    return {name: cell.cell_contents for name, cell in zip(value.__code__.co_freevars, value.__closure__, strict=True)}


def _wrapper_arguments(value: FunctionType, closure: dict[str, Any], kernel_name: str) -> tuple[Any, ...]:
    instructions = tuple(dis.get_instructions(value))
    start = next(
        (
            index
            for index, instruction in enumerate(instructions)
            if instruction.opname == "LOAD_DEREF" and instruction.argval == kernel_name
        ),
        None,
    )
    if start is None:
        raise NotImplementedError("CUDA-Q kernel wrapper does not invoke its closed-over kernel")
    arguments: list[Any] = []
    for instruction in instructions[start + 1 :]:
        if instruction.opname == "LOAD_DEREF" and str(instruction.argval) in closure:
            arguments.append(closure[str(instruction.argval)])
            continue
        if instruction.opname == "LOAD_CONST":
            arguments.append(instruction.argval)
            continue
        if instruction.opname in {"PRECALL", "PUSH_NULL", "RESUME", "COPY_FREE_VARS"}:
            continue
        if instruction.opname in {"CALL", "CALL_FUNCTION", "RETURN_VALUE"}:
            break
        raise NotImplementedError(f"CUDA-Q kernel wrapper uses unsupported bytecode {instruction.opname!r}")
    return tuple(arguments)
