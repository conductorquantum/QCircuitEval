"""Process cleanup helpers for candidate-execution workers."""

from __future__ import annotations

import os
import signal
from typing import Protocol


class _WorkerProcess(Protocol):
    @property
    def pid(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def kill(self) -> None: ...

    def terminate(self) -> None: ...


def terminate_worker_process(
    process: _WorkerProcess,
    *,
    grace_period: float = 1.0,
    verified_process_group_id: int | None = None,
) -> None:
    """Terminate a worker and descendants that share its isolated POSIX group.

    Workers call ``setsid()`` before executing candidate code, making the
    worker PID its process-group ID. If that invariant cannot be verified, use
    the multiprocessing process API so non-POSIX platforms and unexpectedly
    grouped workers never signal the caller's process group.

    Args:
        process: Worker process to terminate.
        grace_period: Seconds to wait between termination and kill signals.
        verified_process_group_id: Worker-owned process group acknowledged
            after ``setsid()`` and before candidate execution. This allows
            descendant cleanup even if candidate code exits the group leader.
    """
    group_id = _isolated_process_group(process, verified_process_group_id)
    if group_id is None:
        _terminate_process_only(process, grace_period)
        return

    if not _signal_process_group(group_id, signal.SIGTERM):
        _terminate_process_only(process, grace_period)
        return

    process.join(timeout=grace_period)
    # Signal the group even if its leader exited during the grace period:
    # descendants may still be alive or may have ignored SIGTERM.
    _signal_process_group(group_id, signal.SIGKILL)
    process.join(timeout=grace_period)
    if process.is_alive():
        process.kill()
        process.join(timeout=grace_period)


def _isolated_process_group(
    process: _WorkerProcess,
    verified_process_group_id: int | None = None,
) -> int | None:
    pid = process.pid
    if (
        pid is None
        or pid == os.getpid()
        or not hasattr(os, "getpgid")
        or not hasattr(os, "getpgrp")
        or not hasattr(os, "killpg")
    ):
        return None
    if verified_process_group_id is not None:
        if verified_process_group_id != pid or verified_process_group_id == os.getpgrp():
            return None
        return verified_process_group_id
    try:
        group_id = os.getpgid(pid)
        caller_group_id = os.getpgrp()
    except OSError:
        return None
    if group_id != pid or group_id == caller_group_id:
        return None
    return group_id


def _signal_process_group(group_id: int, signal_number: int) -> bool:
    try:
        os.killpg(group_id, signal_number)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


def _terminate_process_only(process: _WorkerProcess, grace_period: float) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=grace_period)
    if process.is_alive():
        process.kill()
        process.join(timeout=grace_period)
