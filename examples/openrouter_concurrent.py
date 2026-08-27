from __future__ import annotations

import os
from pathlib import Path

from qceval.core.bench import Adaptor
from qceval.core.io import write_output
from qceval.core.runner import BenchmarkRunner
from qceval.models import RunConfig, RunOptions
from qceval.providers.registry import build_provider


def main() -> None:
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get("QCEVAL_MODEL", "openai/gpt-4o-mini")
    provider_config = {"openrouter_api_key": api_key, "temperature": 0.0}
    config = RunConfig(
        provider="openrouter",
        frameworks=("qiskit",),
        source_hint=None,
        model=model,
        max_tasks=12,
        provider_config=provider_config,
    )
    options = RunOptions(
        generation_concurrency=8,
        evaluation_workers=4,
        cache_dir=Path(".qceval-cache"),
        stream_to=Path("results.openrouter.concurrent.jsonl"),
    )
    provider = build_provider(config.provider, model=config.model, config=config.provider_config)
    payload = BenchmarkRunner(config=config, provider=provider, adapter=Adaptor(), options=options).run()
    write_output(Path("results.openrouter.concurrent.json"), payload)


if __name__ == "__main__":
    main()
