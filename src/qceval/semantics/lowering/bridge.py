"""Parity bridge from new Program IR to the dense-circuit extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qceval.evals.ir.core import Circuit
from qceval.frameworks.cirq.parser import from_cirq
from qceval.frameworks.cudaq.parser import from_cudaq
from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.frameworks.pennylane.parser import from_pennylane
from qceval.frameworks.qiskit.lowering import QiskitLoweringAdapter
from qceval.frameworks.qiskit.parser import from_qiskit
from qceval.semantics.lowering.base import LoweringResult, LoweringStatus, SourceMetadata


@dataclass(frozen=True)
class StaticBridgeResult:
    """New Program IR plus a legacy static-unitary circuit."""

    lowering: LoweringResult
    legacy_circuit: Circuit | None


def lower_qiskit_with_legacy_bridge(circuit: Any, metadata: SourceMetadata) -> StaticBridgeResult:
    """Lower with the new adapter and preserve existing unitary behavior.

    Args:
        circuit: Qiskit circuit.
        metadata: Candidate diagnostics.

    Returns:
        New lowering result and legacy circuit when both support the program.
    """
    lowering = QiskitLoweringAdapter().lower(circuit, metadata, None)
    if lowering.status is not LoweringStatus.SUCCESS:
        return StaticBridgeResult(lowering, None)
    legacy = from_qiskit(circuit)
    if lowering.program is None or lowering.program.num_qubits != legacy.num_qubits:
        raise RuntimeError("new and legacy Qiskit lowering widths disagree")
    return StaticBridgeResult(lowering, legacy)


def lower_cirq_with_legacy_bridge(circuit: Any, metadata: SourceMetadata) -> StaticBridgeResult:
    """Lower Cirq through new and legacy static paths.

    Args:
        circuit: Cirq circuit.
        metadata: Candidate diagnostics.

    Returns:
        New lowering and legacy static circuit when supported.
    """
    from qceval.frameworks.cirq.lowering import CirqLoweringAdapter

    return _static_bridge(CirqLoweringAdapter().lower(circuit, metadata, None), lambda: from_cirq(circuit))


def lower_pennylane_with_legacy_bridge(tape: Any, metadata: SourceMetadata) -> StaticBridgeResult:
    """Lower PennyLane through new and legacy static paths.

    Args:
        tape: Captured PennyLane tape.
        metadata: Candidate diagnostics.

    Returns:
        New lowering and legacy static circuit when supported.
    """
    from qceval.frameworks.pennylane.lowering import PennyLaneLoweringAdapter

    return _static_bridge(PennyLaneLoweringAdapter().lower(tape, metadata, None), lambda: from_pennylane(tape))


def lower_cudaq_with_legacy_bridge(program: CudaqProgram, metadata: SourceMetadata) -> StaticBridgeResult:
    """Lower CUDA-Q through new and legacy static paths.

    Args:
        program: CUDA-Q source carrier.
        metadata: Candidate diagnostics.

    Returns:
        New lowering and legacy static circuit when supported.
    """
    from qceval.frameworks.cudaq.lowering import CudaqLoweringAdapter

    return _static_bridge(CudaqLoweringAdapter().lower(program, metadata, None), lambda: from_cudaq(program))


def _static_bridge(lowering: LoweringResult, legacy_factory: Any) -> StaticBridgeResult:
    if lowering.status is not LoweringStatus.SUCCESS:
        return StaticBridgeResult(lowering, None)
    legacy = legacy_factory()
    if lowering.program is None or lowering.program.num_qubits != legacy.num_qubits:
        raise RuntimeError("new and legacy lowering widths disagree")
    return StaticBridgeResult(lowering, legacy)
