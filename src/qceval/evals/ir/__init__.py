"""Framework-neutral dense circuit IR and converters.

Qiskit, Cirq, and PennyLane convert via ``qceval.frameworks.*.parser``.
CUDA-Q converts from source via ``qceval.frameworks.cudaq.parser``.

Framework converters are loaded lazily so parsers can import
``qceval.evals.ir.core`` without a circular import.
"""

from __future__ import annotations

from typing import Any

from qceval.evals.ir.core import Circuit, Control, Gate, full_unitary

__all__ = [
    "Circuit",
    "Control",
    "Gate",
    "from_cirq",
    "from_cudaq",
    "from_framework",
    "from_pennylane",
    "from_qiskit",
    "full_unitary",
]


def __getattr__(name: str) -> Any:
    if name == "from_framework":
        from qceval.evals.ir.convert import from_framework

        return from_framework
    if name == "from_qiskit":
        from qceval.frameworks.qiskit.parser import from_qiskit

        return from_qiskit
    if name == "from_cirq":
        from qceval.frameworks.cirq.parser import from_cirq

        return from_cirq
    if name == "from_pennylane":
        from qceval.frameworks.pennylane.parser import from_pennylane

        return from_pennylane
    if name == "from_cudaq":
        from qceval.frameworks.cudaq.parser import from_cudaq

        return from_cudaq
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
