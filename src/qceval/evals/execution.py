"""Framework execution dispatch shared by behavioral grading workers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qceval.evals.inputs import global_inputs
from qceval.evals.models import ExecutionResult
from qceval.frameworks import FRAMEWORK_EXECUTORS
from qceval.models import Framework


class ExecutionDispatcher:
    """Execute candidates through the shared framework executor registry.

    Args:
        framework: Framework whose candidate code is executed.
        tasks: Raw framework tasks keyed by normalized task identifier.
    """

    def __init__(self, framework: Framework, tasks: Mapping[str, Mapping[str, Any]]) -> None:
        self.framework = framework
        self.tasks = tasks
        self.inputs = global_inputs(framework)
        self._executor = FRAMEWORK_EXECUTORS[framework]

    def execute(
        self,
        *,
        task_id: str,
        code: str,
        entry_point: str,
        call_args: tuple[Any, ...] | None = None,
    ) -> ExecutionResult:
        """Execute one candidate with the registered framework adaptor.

        Args:
            task_id: Normalized task identifier.
            code: Candidate Python source.
            entry_point: Function to invoke from the candidate source.
            call_args: Optional explicit public arguments for replay.

        Returns:
            Normalized framework execution result.
        """
        return self._executor(
            task_id=task_id,
            code=code,
            entry_point=entry_point,
            inputs=self.inputs,
            call_args=call_args,
            output_qubits=_output_qubits(self.tasks[task_id]["canonical_class"]),
        )


def _output_qubits(spec: Mapping[str, Any]) -> tuple[int, ...] | None:
    qubits = spec.get("output_qubits")
    if qubits is None:
        check_spec = spec.get("structure_checks", spec.get("metadata_checks", {}))
        qubits = check_spec.get("required_measurement_qubits") if isinstance(check_spec, Mapping) else None
    if qubits is None:
        return None
    if not isinstance(qubits, Sequence) or isinstance(qubits, str | bytes):
        raise ValueError("output_qubits must be a sequence of integer qubit indices")
    return tuple(int(qubit) for qubit in qubits)
