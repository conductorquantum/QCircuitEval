"""Framework lowering registry."""

from __future__ import annotations

from collections.abc import Mapping

from qceval.frameworks.cirq.lowering import CirqLoweringAdapter
from qceval.frameworks.cudaq.lowering import CudaqLoweringAdapter
from qceval.frameworks.pennylane.lowering import PennyLaneLoweringAdapter
from qceval.frameworks.qiskit.lowering import QiskitLoweringAdapter
from qceval.semantics.lowering.base import LoweringAdapter


class LoweringRegistry:
    """Immutable framework-name to adapter mapping."""

    def __init__(self, adapters: Mapping[str, LoweringAdapter]) -> None:
        """Initialize a registry.

        Args:
            adapters: Unique framework adapters.
        """
        self._adapters = dict(adapters)
        if not self._adapters or any(not key for key in self._adapters):
            raise ValueError("lowering registry requires named adapters")

    def get(self, framework: str) -> LoweringAdapter:
        """Return one adapter.

        Args:
            framework: Normalized framework name.

        Returns:
            Registered adapter.
        """
        return self._adapters[framework]


def default_lowering_registry() -> LoweringRegistry:
    """Return the currently enabled production lowering registry.

    Returns:
        Registry containing the completed framework slices.
    """
    return LoweringRegistry(
        {
            "qiskit": QiskitLoweringAdapter(),
            "cirq": CirqLoweringAdapter(),
            "pennylane": PennyLaneLoweringAdapter(),
            "cudaq": CudaqLoweringAdapter(),
        }
    )
