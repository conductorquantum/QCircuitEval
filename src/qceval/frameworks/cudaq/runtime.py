"""CUDA-Q runtime utility helpers."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any


def _is_cudaq_kernel(value: Any) -> bool:
    value_type = type(value)
    return value_type.__name__ in {"PyKernel", "PyKernelDecorator"} or value_type.__module__.startswith("cudaq.")


def _import_cudaq() -> Any:
    try:
        import cudaq
    except ImportError as exc:
        raise RuntimeError(
            "CUDA-Q is not importable in this environment. Install project dependencies on a supported CUDA-Q "
            "runtime platform."
        ) from exc
    return cudaq


def _kernel_args(call_args: tuple[Any, ...] | None) -> tuple[Any, ...]:
    if call_args is None:
        return ()
    return tuple(-1 if value is None else value for value in call_args)


def _kernel_arg_options(call_args: tuple[Any, ...] | None) -> tuple[tuple[Any, ...], ...]:
    args = _kernel_args(call_args)
    if args:
        return (args, ())
    return (args,)


def _kernel_formal_arity(kernel: Any) -> int | None:
    """Return the declared argument count of a CUDA-Q kernel, when knowable.

    CUDA-Q crashes the whole process (SIGSEGV) when some kernels are launched
    with missing runtime arguments instead of raising, so callers must check
    arity before ``cudaq.get_state``/``cudaq.sample`` and fail typed.
    """
    try:
        signature = getattr(kernel, "signature", None)
        arg_types = getattr(signature, "arg_types", None)
        if arg_types is not None:
            return len(arg_types)
        arity = getattr(kernel, "formal_arity", None)
        if isinstance(arity, int) and not isinstance(arity, bool):
            return arity
    except Exception:  # noqa: BLE001 - introspection is best-effort only.
        return None
    return None


def _require_kernel_arity(kernel: Any, args: tuple[Any, ...]) -> None:
    """Raise a typed argument-count error before an unsafe kernel launch."""
    arity = _kernel_formal_arity(kernel)
    if arity is not None and arity != len(args):
        raise TypeError(f"invalid number of arguments: cudaq kernel requires {arity}, provided {len(args)}")


def _is_argument_count_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "invalid number of arguments" in message
        or "wrong number of arguments" in message
        or ("provided" in message and "required" in message)
    )


@contextmanager
def _ignore_cudaq_deprecations() -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"cudaq(\.|$)")
        yield


@contextmanager
def _double_precision_target(cudaq: Any) -> Any:
    """Extract exact statevectors on a double-precision simulator.

    GPU targets such as ``nvidia`` default to single precision, whose ~1e-9
    amplitude rounding exceeds the exact grading tolerances. Sampling still
    runs on the ambient target; only state extraction is pinned.
    """
    prior = cudaq.get_target().name
    if prior == "qpp-cpu":
        yield
        return
    try:
        cudaq.set_target("qpp-cpu")
    except Exception:  # noqa: BLE001 - fall back to the ambient target.
        yield
        return
    try:
        yield
    finally:
        try:
            cudaq.set_target(prior)
        except Exception:  # noqa: BLE001 - best-effort restore.
            cudaq.reset_target()
