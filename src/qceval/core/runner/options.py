"""Runner option validation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from qceval.models import BenchmarkRecord, RunOptions


def _normalized_options(options: RunOptions) -> RunOptions:
    if options.generation_concurrency <= 0:
        raise ValueError("generation_concurrency must be greater than zero")
    if options.evaluation_workers <= 0:
        raise ValueError("evaluation_workers must be greater than zero")
    if options.task_timeout is not None and options.task_timeout <= 0:
        raise ValueError("task_timeout must be greater than zero")
    if options.eval_timeout is not None and options.eval_timeout <= 0:
        raise ValueError("eval_timeout must be greater than zero")
    return options


def _completed_count(records: Sequence[BenchmarkRecord | None]) -> int:
    return sum(record is not None for record in records)
