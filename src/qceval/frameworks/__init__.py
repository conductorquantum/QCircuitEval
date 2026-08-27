"""Framework-specific pipeline adapters for QCircuitEval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from qceval.evals.models import ExecutionResult
from qceval.frameworks.cirq import execute_cirq_task
from qceval.frameworks.cudaq import execute_cudaq_task
from qceval.frameworks.pennylane import execute_pennylane_task
from qceval.frameworks.qiskit import execute_qiskit_task
from qceval.models import Framework


class TaskExecutor(Protocol):
    """Shared keyword-only signature for framework task executors."""

    def __call__(
        self,
        *,
        task_id: str,
        code: str,
        entry_point: str,
        inputs: dict[str, Any],
        call_args: tuple[Any, ...] | None = None,
        output_qubits: Sequence[int] | None = None,
    ) -> ExecutionResult:
        """Execute one candidate and return a normalized result."""


FRAMEWORK_EXECUTORS: Mapping[Framework, TaskExecutor] = {
    "qiskit": execute_qiskit_task,
    "cirq": execute_cirq_task,
    "pennylane": execute_pennylane_task,
    "cudaq": execute_cudaq_task,
}

__all__ = [
    "FRAMEWORK_EXECUTORS",
    "TaskExecutor",
    "execute_cirq_task",
    "execute_cudaq_task",
    "execute_pennylane_task",
    "execute_qiskit_task",
]
