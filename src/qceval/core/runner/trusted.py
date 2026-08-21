"""Fresh-process containment for trusted candidate regrading."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing import get_context
from multiprocessing.connection import Connection
from typing import Any

from qceval.core.bench import Adaptor
from qceval.core.runner.processes import terminate_worker_process
from qceval.core.runner.workers import _BoundedWriter
from qceval.models import QCEvalEvaluation, QCEvalTask

DEFAULT_TRUSTED_REGRADE_TIMEOUT_SECONDS = 300.0
_MAX_RESULT_BYTES = 32 * 1024 * 1024
_POLL_SECONDS = 0.05
_CLEANUP_GRACE_SECONDS = 0.2

_EvaluationHook = Callable[[QCEvalTask, str], QCEvalEvaluation | Mapping[str, Any]]


class TrustedRegradeError(RuntimeError):
    """A fresh trusted worker failed without an evaluation."""


class TrustedRegradeTimeout(TrustedRegradeError):
    """A fresh trusted worker exceeded its wall-clock budget."""


def evaluate_trusted_candidate(
    task: QCEvalTask,
    code: str,
    *,
    timeout_seconds: float = DEFAULT_TRUSTED_REGRADE_TIMEOUT_SECONDS,
    _evaluation_hook: _EvaluationHook | None = None,
) -> Mapping[str, Any]:
    """Evaluate one candidate in a new spawn worker and return JSON data only.

    The private hook exists for process-containment tests. Production callers
    omit it, causing the child to construct the bundled :class:`Adaptor`.

    Args:
        task: Bundled task identity to evaluate.
        code: Nonempty candidate Python source.
        timeout_seconds: Hard worker wall-clock timeout.
        _evaluation_hook: Test-only evaluation override executed in the child.

    Returns:
        Serialized evaluation produced by the fresh worker.
    """
    if timeout_seconds <= 0:
        raise ValueError("trusted regrade timeout must be greater than zero")

    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    release = context.Event()
    process = context.Process(
        target=_trusted_regrade_worker,
        args=(task, code, sender, release, _evaluation_hook),
    )
    verified_group_id: list[int | None] = [None]
    started_at = time.monotonic()
    try:
        process.start()
        sender.close()
        return _wait_for_trusted_result(
            process,
            receiver,
            timeout_seconds,
            started_at,
            verified_group_id,
        )
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            terminate_worker_process(
                process,
                grace_period=_CLEANUP_GRACE_SECONDS,
                verified_process_group_id=verified_group_id[0],
            )


def _wait_for_trusted_result(
    process: Any,
    receiver: Connection,
    timeout_seconds: float,
    started_at: float,
    verified_group_id: list[int | None],
) -> Mapping[str, Any]:
    while True:
        remaining = timeout_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            raise TrustedRegradeTimeout(f"trusted worker timed out after {timeout_seconds:.3f}s")
        if receiver.poll(min(_POLL_SECONDS, remaining)):
            result = _handle_worker_message(_receive_message(receiver), process.pid, verified_group_id)
            if result is not None:
                return result
        elif not process.is_alive():
            raise TrustedRegradeError(f"trusted worker exited without an evaluation, exitcode={process.exitcode}")


def _handle_worker_message(
    message: Mapping[str, Any],
    process_id: int | None,
    verified_group_id: list[int | None],
) -> Mapping[str, Any] | None:
    kind = message.get("kind")
    if kind == "ready":
        if message.get("isolated_process_group") is True:
            verified_group_id[0] = process_id
        return None
    if kind == "evaluation":
        evaluation = message.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise TrustedRegradeError("trusted worker returned a non-object evaluation")
        return dict(evaluation)
    if kind == "error":
        error_type = message.get("error_type", "WorkerError")
        error = message.get("error", "trusted worker failed")
        raise TrustedRegradeError(f"{error_type}: {error}")
    raise TrustedRegradeError("trusted worker returned an invalid message")


def _receive_message(connection: Connection) -> Mapping[str, Any]:
    try:
        encoded = connection.recv_bytes(maxlength=_MAX_RESULT_BYTES)
    except (EOFError, OSError) as exc:
        raise TrustedRegradeError(f"could not receive trusted worker result: {type(exc).__name__}: {exc}") from exc
    try:
        message = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrustedRegradeError("trusted worker returned invalid JSON") from exc
    if not isinstance(message, Mapping):
        raise TrustedRegradeError("trusted worker returned a non-object message")
    return message


def _trusted_regrade_worker(
    task: QCEvalTask,
    code: str,
    connection: Connection,
    release: Any,
    evaluation_hook: _EvaluationHook | None,
) -> None:
    isolated_process_group = _start_isolated_process_group()
    if not _send_message(
        connection,
        {"kind": "ready", "isolated_process_group": isolated_process_group},
    ):
        return
    try:
        with redirect_stdout(_BoundedWriter()), redirect_stderr(_BoundedWriter()):
            evaluation = _worker_evaluation(task, code, evaluation_hook)
        message: Mapping[str, Any] = {"kind": "evaluation", "evaluation": evaluation}
    except BaseException as exc:
        message = {
            "kind": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2048],
        }
    _send_message(connection, message)
    connection.close()
    # Keep the process-group leader alive until the parent has accepted the
    # serialized result and terminated the whole group, including descendants.
    release.wait()


def _worker_evaluation(
    task: QCEvalTask,
    code: str,
    evaluation_hook: _EvaluationHook | None,
) -> Mapping[str, Any]:
    if evaluation_hook is not None:
        evaluation = evaluation_hook(task, code)
    else:
        adapter = Adaptor()
        trusted_task = next(
            candidate
            for candidate in adapter.load_tasks(task.framework, suite=task.suite)
            if candidate.task_id == task.task_id
        )
        if trusted_task.entry_point != task.entry_point:
            raise ValueError("trusted worker task entry point mismatch")
        evaluation = adapter.evaluate(trusted_task, code)
    if isinstance(evaluation, QCEvalEvaluation):
        return evaluation.to_dict()
    if isinstance(evaluation, Mapping):
        return dict(evaluation)
    raise TypeError("trusted evaluation hook must return an evaluation mapping")


def _start_isolated_process_group() -> bool:
    if not hasattr(os, "setsid"):
        return False
    try:
        os.setsid()
    except OSError:
        return False
    return True


def _send_message(connection: Connection, message: Mapping[str, Any]) -> bool:
    try:
        encoded = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
        connection.send_bytes(encoded)
    except (OSError, TypeError, ValueError):
        return False
    return True
