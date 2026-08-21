from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from qceval.cli_plan import (
    ReasoningJob,
    apply_reasoning_job,
    jobs_from_registry,
    load_registry_efforts,
    sweep_manifest_path,
    sweep_output_path,
)


def test_load_registry_merges_efforts_in_first_seen_model_order(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "models": [
                    {"model_id": "model/a", "reasoning_efforts": ["max", "low"]},
                    {"model_id": "model/b", "reasoning_efforts": ["enabled"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "models": [
                    {"model_id": "model/a", "reasoning_efforts": ["medium", "low"]},
                    {"model_id": "model/c", "reasoning_efforts": ["none"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    models = load_registry_efforts([first, second])

    assert list(models) == ["model/a", "model/b", "model/c"]
    assert models["model/a"] == ("low", "medium", "max")
    assert models["model/b"] == ("enabled",)


def test_jobs_from_registry_keeps_all_levels_when_effort_omitted() -> None:
    jobs = jobs_from_registry(
        {"model/a": ("low", "max"), "model/b": ("enabled",)},
        requested_effort=None,
        model_filter=None,
    )

    assert jobs == (
        ReasoningJob("model/a", "low"),
        ReasoningJob("model/a", "max"),
        ReasoningJob("model/b", "enabled"),
    )


def test_jobs_from_registry_filters_specific_effort_and_model() -> None:
    jobs = jobs_from_registry(
        {"model/a": ("low", "max"), "model/b": ("max",)},
        requested_effort="max",
        model_filter="model/b",
    )

    assert jobs == (ReasoningJob("model/b", "max"),)


def test_jobs_from_registry_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="produced no jobs"):
        jobs_from_registry(
            {"model/a": ("low",)},
            requested_effort="max",
            model_filter=None,
        )


def test_sweep_file_paths_use_effort_and_efforts_manifest() -> None:
    job = ReasoningJob("model/a", "high")

    assert sweep_output_path(Path("results.json"), job, multi_model=False) == Path("results.effort-high.json")
    assert sweep_manifest_path(Path("results.json")) == Path("results.efforts.json")


def test_sweep_directory_paths_use_configuration_identity() -> None:
    job = ReasoningJob("model/a", "high")

    assert sweep_output_path(Path("results"), job, multi_model=True) == Path("results/model-a__effort-high.json")
    assert sweep_manifest_path(Path("results")) == Path("results/manifest.json")


def test_apply_reasoning_job_stamps_configuration_and_output() -> None:
    args = argparse.Namespace(
        model=None,
        reasoning_effort="all",
        reasoning_enabled=None,
        configuration_id=None,
    )
    job = ReasoningJob("model/a", "enabled")

    cloned = apply_reasoning_job(
        args,
        job,
        out=Path("result.json"),
        assign_configuration_id=True,
    )

    assert cloned.model == "model/a"
    assert cloned.reasoning_effort is None
    assert cloned.reasoning_enabled is True
    assert cloned.configuration_id == "model-a__effort-enabled"
    assert cloned.out == Path("result.json")


def test_published_registries_union_to_ten_models_and_33_jobs() -> None:
    models = load_registry_efforts(
        [
            Path("production/models.prompt-effort-sweep.json"),
            Path("production/models.max-reasoning.json"),
        ]
    )
    jobs = jobs_from_registry(models, requested_effort="all", model_filter=None)

    assert len(models) == 10
    assert len(jobs) == 33
    assert ("google/gemma-4-31b-it", "enabled") in {(job.model, job.effort) for job in jobs}
