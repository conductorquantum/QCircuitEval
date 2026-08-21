from __future__ import annotations

from pathlib import Path

from qceval.core.bench import Adaptor
from qceval.core.io import write_output
from qceval.core.runner import BenchmarkRunner
from qceval.models import RunConfig
from qceval.providers.smoke import SmokeProvider


def main() -> None:
    config = RunConfig(
        provider="smoke",
        frameworks=("qiskit",),
        source_hint=None,
        model="smoke-canonical",
        max_tasks=1,
    )
    payload = BenchmarkRunner(config=config, provider=SmokeProvider(), adapter=Adaptor(config.source_hint)).run()
    write_output(Path("results.smoke.qiskit.json"), payload)


if __name__ == "__main__":
    main()
