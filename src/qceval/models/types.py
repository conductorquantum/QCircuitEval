"""Common QCircuitEval type aliases."""

from __future__ import annotations

from typing import Literal

Framework = Literal["qiskit", "cirq", "pennylane", "cudaq"]
FrameworkChoice = Literal["qiskit", "cirq", "pennylane", "cudaq", "all"]
OutcomeStatus = Literal[
    "generated",
    "passed",
    "failed",
    "provider_failed",
    "compile_failed",
    "run_failed",
    "infrastructure_error",
]
Suite = Literal["core", "qec"]
SuiteChoice = Literal["core", "qec", "all"]
ProviderRole = Literal["system", "user", "assistant"]
