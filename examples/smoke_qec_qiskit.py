"""Run the bundled QEC suite against the smoke provider for Qiskit."""

from __future__ import annotations

from qceval.core.bench import Adaptor
from qceval.core.runner import BenchmarkRunner
from qceval.models import RunConfig, RunOptions
from qceval.providers.smoke import SmokeProvider


def main() -> None:
    """Run a local Qiskit QEC smoke benchmark and print the pass rate."""
    config = RunConfig(
        provider="smoke",
        frameworks=("qiskit",),
        source_hint=None,
        model="smoke-canonical",
        suites=("qec",),
    )
    payload = BenchmarkRunner(
        config=config,
        provider=SmokeProvider(),
        adapter=Adaptor(),
        options=RunOptions(eval_timeout=20, fail_fast=True),
    ).run()
    summary = payload["summary"]
    print(f"QEC Qiskit smoke pass_rate={summary['pass_rate']:.3f} passed={summary['passed']}/{summary['total_tasks']}")


if __name__ == "__main__":
    main()
