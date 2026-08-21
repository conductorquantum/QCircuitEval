"""Opt-in live smoke test for the Coda provider."""

from __future__ import annotations

import os

import pytest

from qceval.core.bench import Adaptor
from qceval.core.runner import BenchmarkRunner
from qceval.models import RunConfig
from qceval.providers.registry import build_provider

pytestmark = pytest.mark.live


def test_live_coda_runs_one_qiskit_task() -> None:
    # Arrange
    if os.environ.get("QCEVAL_LIVE_CODA") != "1":
        pytest.skip("set QCEVAL_LIVE_CODA=1 to run live Coda tests")
    api_key = _coda_key()
    provider_config = {"coda_api_key": api_key}
    config = RunConfig(
        provider="coda",
        frameworks=("qiskit",),
        source_hint=None,
        model="coda/build",
        max_tasks=1,
        provider_config=provider_config,
    )

    # Act
    payload = BenchmarkRunner(
        config=config,
        provider=build_provider("coda", model="coda/build", config=provider_config),
        adapter=Adaptor(),
    ).run()

    # Assert
    assert len(payload["results"]) == 1
    assert payload["summary"]["total_tasks"] == 1


def _coda_key() -> str:
    value = os.environ.get("CODA_API_KEY")
    if value:
        return value
    pytest.skip("CODA_API_KEY is not configured")
