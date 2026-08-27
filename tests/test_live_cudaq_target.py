from __future__ import annotations

import os

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks


@pytest.mark.live
def test_live_cudaq_target_smoke() -> None:
    target = os.environ.get("QCEVAL_LIVE_CUDAQ_TARGET")
    if not target:
        pytest.skip("set QCEVAL_LIVE_CUDAQ_TARGET to run CUDA-Q target smoke")
    cudaq = pytest.importorskip("cudaq")
    previous_target = cudaq.get_target().name

    try:
        cudaq.set_target(target)
        evaluator = build_evaluator("cudaq")
        task = load_tasks("cudaq")["58"]
        execution, details = evaluator.grade_code(
            task_id="58",
            code=task["canonical_solution"],
            entry_point=task["entry_point"],
        )
    finally:
        cudaq.set_target(previous_target)

    assert execution.metadata["cudaq_target"] == target
    assert execution.metadata["probability_method"] in {"statevector", "sample_fallback"}
    assert details["passed"] is True
