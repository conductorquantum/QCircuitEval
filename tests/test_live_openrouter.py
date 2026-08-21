from __future__ import annotations

import os
from pathlib import Path

import pytest

from qceval.core.bench import Adaptor
from qceval.core.runner import BenchmarkRunner
from qceval.models import RunConfig, RunOptions
from qceval.providers.registry import build_provider

pytestmark = pytest.mark.live


def test_live_openrouter_cache_replay_matches_concurrent_run(tmp_path: Path) -> None:
    # Arrange
    if os.environ.get("QCEVAL_LIVE") != "1":
        pytest.skip("set QCEVAL_LIVE=1 to run live OpenRouter tests")
    api_key = _openrouter_key()
    model = os.environ.get("QCEVAL_LIVE_MODEL", "openai/gpt-4o-mini")
    provider_config = {"openrouter_api_key": api_key, "temperature": 0.0, "timeout": 120.0}
    config = RunConfig(
        provider="openrouter",
        frameworks=("qiskit",),
        source_hint=None,
        model=model,
        max_tasks=2,
        provider_config=provider_config,
    )
    cache_dir = tmp_path / "cache"

    # Act
    serial = BenchmarkRunner(
        config=config,
        provider=build_provider("openrouter", model=model, config=provider_config),
        adapter=Adaptor(),
        options=RunOptions(cache_dir=cache_dir),
    ).run()
    concurrent = BenchmarkRunner(
        config=config,
        provider=build_provider("openrouter", model=model, config=provider_config),
        adapter=Adaptor(),
        options=RunOptions(cache_dir=cache_dir, generation_concurrency=2, evaluation_workers=2),
    ).run()

    # Assert
    assert _stable_results(serial["results"]) == _stable_results(concurrent["results"])
    assert concurrent["summary"]["total_tasks"] == 2


def _openrouter_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY")
    if value:
        return value
    env_path = Path(__file__).resolve().parents[1] / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, _, raw = line.partition("=")
        if key == "OPENROUTER_API_KEY" and raw.strip():
            return raw.strip()
    pytest.skip("OPENROUTER_API_KEY is not configured")


def _stable_results(results: list[dict]) -> list[dict]:
    stable = []
    for result in results:
        item = dict(result)
        provider_response = dict(item["provider_response"])
        provider_response["raw_response"] = None
        item["provider_response"] = provider_response
        stable.append(item)
    return stable
