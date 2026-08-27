from __future__ import annotations

from pathlib import Path

import pytest
from scripts.plot_effort_price_performance import (
    SWEEP_MODELS,
    load_configurations,
    load_historical_configuration_ids,
    load_supplemental_configurations,
    pareto_frontier,
    write_plots,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "full-effort-pass1-five-model" / "analysis" / "effort-sweep.json"
IMPORTS = ROOT / "results" / "full-effort-pass1-five-model" / "imported-max" / "manifest.json"
SUPPLEMENTAL = (
    ROOT / "results" / "pass1-123a5e4-20260810T213903Z" / "offline-final-eight-20260811T181700Z" / "score-cost.json"
)


def test_pareto_frontier_rejects_equal_cost_and_equal_score_dominance() -> None:
    rows = [
        {"configuration_id": "a", "provider_cost_per_task_usd": 1.0, "score": 0.4},
        {"configuration_id": "b", "provider_cost_per_task_usd": 1.0, "score": 0.5},
        {"configuration_id": "c", "provider_cost_per_task_usd": 2.0, "score": 0.5},
        {"configuration_id": "d", "provider_cost_per_task_usd": 3.0, "score": 0.6},
    ]

    frontier = pareto_frontier(rows, "score")

    assert [row["configuration_id"] for row in frontier] == ["b", "d"]


def test_final_report_validation_and_plot_outputs(tmp_path: Path) -> None:
    if not REPORT.is_file() or not IMPORTS.is_file():
        pytest.skip("final production artifacts are not present")
    rows = load_configurations(REPORT)
    historical_ids = load_historical_configuration_ids(IMPORTS)

    outputs = write_plots(rows, tmp_path, historical_ids=historical_ids)

    assert {path.name for path in outputs} == {
        "price-vs-performance-overall.png",
        "price-vs-performance-core.png",
        "price-vs-performance-qec.png",
    }
    assert all(path.stat().st_size > 100_000 for path in outputs)


def test_supplemental_completed_run_adds_five_models_with_core_and_qec_scores() -> None:
    if not SUPPLEMENTAL.is_file():
        pytest.skip("prior completed-run artifacts are not present")

    rows = load_supplemental_configurations(SUPPLEMENTAL, excluded_models=SWEEP_MODELS)

    assert len(rows) == 5
    assert len({row["model_id"] for row in rows}) == 5
    assert all(row["records"] == 280 and row["cost_covered_records"] == 280 for row in rows)
    assert all(0 <= row["core_pass_at_1"] <= 1 and 0 <= row["qec_pass_at_1"] <= 1 for row in rows)
