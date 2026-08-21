"""Production candidate evaluation process-isolation tests."""

from qceval.core.bench import Adaptor
from qceval.core.runner.evaluation import EvaluationScheduler
from qceval.models import RunConfig, RunOptions


def test_production_adaptor_uses_a_worker_even_with_one_evaluation_slot() -> None:
    scheduler = EvaluationScheduler(
        adapter=Adaptor(),
        config=RunConfig(provider="smoke", frameworks=("qiskit",), source_hint=None, model=None),
        options=RunOptions(evaluation_workers=1),
    )
    try:
        assert scheduler._executor is not None
    finally:
        scheduler.close()
