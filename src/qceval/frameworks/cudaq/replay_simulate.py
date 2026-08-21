"""CUDA-Q source-replay simulation and statevector extraction."""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from qceval.frameworks.cudaq.replay_transform import (
    _transform_source,
    parsed_measured_wires,
    parsed_num_qubits,
)

_counter = itertools.count()


def _import_transformed_module(source: str) -> Any:
    name = f"_qceval_cudaq_replay_{next(_counter)}_{uuid.uuid4().hex}"
    path = Path(tempfile.gettempdir()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load transformed CUDA-Q module from {path}")
    module = spec.loader.create_module(spec) if hasattr(spec.loader, "create_module") else None
    module = module or importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cleanup_module(module: Any) -> None:
    name = getattr(module, "__name__", None)
    file = getattr(module, "__file__", None)
    if name and name in sys.modules:
        del sys.modules[name]
    if file:
        Path(file).unlink(missing_ok=True)


def cudaq_num_qubits(code: str) -> int:
    """Count the qubits allocated by a CUDA-Q candidate.

    Args:
        code: CUDA-Q Python source code.

    Returns:
        Number of statically visible qubit allocations.

    Raises:
        SyntaxError: If ``code`` is not valid Python.
    """
    return parsed_num_qubits(code)


def cudaq_measured_wires(code: str) -> list[int]:
    """Collect measured wire indices in canonical LSB-first source order.

    Args:
        code: CUDA-Q Python source code.

    Returns:
        Statically resolved measured wire indices.

    Raises:
        SyntaxError: If ``code`` is not valid Python.
    """
    try:
        from qceval.frameworks.cudaq.metadata import _measurement_indices_from_code

        resolved = _measurement_indices_from_code(code)
        if resolved:
            return list(resolved)
    except Exception:  # noqa: BLE001 - fall back to the local subscript scan.
        pass
    return parsed_measured_wires(code)


def _is_cudaq_kernel(value: Any) -> bool:
    """Return whether ``value`` is a CUDA-Q kernel object."""
    value_type = type(value)
    return value_type.__name__ in {
        "PyKernel",
        "PyKernelDecorator",
    } or value_type.__module__.startswith("cudaq.")


def simulate_basis_cudaq(
    code: str,
    entry_point: str,
    *,
    prep: dict[int, int],
    strip_leading_x_on: set[int],
    call_args: tuple[Any, ...] = (),
) -> np.ndarray:
    """Replay a CUDA-Q kernel with injected basis-state preparation.

    Args:
        code: CUDA-Q Python source code.
        entry_point: Public kernel or kernel-factory name.
        prep: Mapping from wire indices to injected computational-basis bits.
        strip_leading_x_on: Wires whose leading X preparation is removed.
        call_args: Arguments passed to a kernel factory.

    Returns:
        The replayed kernel statevector.

    Raises:
        ImportError: If CUDA-Q is unavailable.
        AttributeError: If the transformed module lacks ``entry_point``.
        SyntaxError: If the source cannot be parsed or transformed.
    """
    import cudaq

    transformed = _transform_source(
        code,
        prep=prep,
        strip_leading_x_on=strip_leading_x_on,
        entry_point=entry_point,
    )
    module = _import_transformed_module(transformed)
    try:
        entry = getattr(module, entry_point)
        kernel = entry if _is_cudaq_kernel(entry) else entry(*call_args)
        with _double_precision():
            from qceval.frameworks.cudaq.runtime import _is_argument_count_error, _require_kernel_arity

            try:
                # Launching with the wrong runtime-argument arity can SIGSEGV
                # inside CUDA-Q instead of raising; fail typed up front.
                _require_kernel_arity(kernel, call_args)
                return np.asarray(cudaq.get_state(kernel, *call_args), dtype=complex)
            except Exception as exc:  # noqa: BLE001 - retry only argument-count mismatches.
                if not call_args or not _is_argument_count_error(exc):
                    raise
                _require_kernel_arity(kernel, ())
                return np.asarray(cudaq.get_state(kernel), dtype=complex)
    finally:
        _cleanup_module(module)


def cudaq_kernel_unitary(
    code: str,
    entry_point: str,
    num_qubits: int,
    call_args: tuple[Any, ...] = (),
) -> np.ndarray:
    """Construct a CUDA-Q kernel unitary in canonical little-endian order.

    Args:
        code: CUDA-Q Python source code.
        entry_point: Public kernel or kernel-factory name.
        num_qubits: Number of qubits in the kernel register.
        call_args: Arguments passed to a kernel factory.

    Returns:
        Dense unitary whose columns are replayed basis-state outputs.

    Raises:
        ImportError: If CUDA-Q is unavailable.
        AttributeError: If the transformed module lacks ``entry_point``.
        SyntaxError: If the source cannot be parsed or transformed.
    """
    dim = 1 << num_qubits
    unitary = np.zeros((dim, dim), dtype=complex)
    wires = range(num_qubits)
    for column in range(dim):
        prep = {wire: (column >> wire) & 1 for wire in wires}
        unitary[:, column] = simulate_basis_cudaq(
            code,
            entry_point,
            prep=prep,
            strip_leading_x_on=set(),
            call_args=call_args,
        )
    return unitary


@contextlib.contextmanager
def _double_precision() -> Any:
    """Run CUDA-Q state extraction on a double-precision backend."""
    import cudaq

    prior = cudaq.get_target().name
    try:
        cudaq.set_target("qpp-cpu")
        yield
    finally:
        try:
            cudaq.set_target(prior)
        except Exception:  # noqa: BLE001 - best-effort restore.
            cudaq.reset_target()
