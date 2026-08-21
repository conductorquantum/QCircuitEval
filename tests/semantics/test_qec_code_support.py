"""Code-only support tests for contract-driven QEC grading."""

from __future__ import annotations

from typing import Any

import pytest

from qceval.frameworks.cudaq.lowering import _normalize_terminal_measurements
from qceval.semantics.ir import IR_VERSION, Operation, OperationKind, Program, Provenance


def test_cudaq_terminal_measurements_follow_contract_bit_order(monkeypatch: pytest.MonkeyPatch) -> None:
    program = Program(
        IR_VERSION,
        5,
        2,
        (
            Operation(OperationKind.MEASUREMENT, "mz", (4,), (0,)),
            Operation(OperationKind.MEASUREMENT, "mz", (3,), (1,)),
        ),
        None,
        (1, 0),
        Provenance("cudaq", "test"),
    )
    payload: dict[str, Any] = {
        "requirements": [
            {
                "id": "terminal_observation",
                "value": {"cudaq": {"kind": "measurement", "qubits": [3, 4]}},
            }
        ]
    }
    monkeypatch.setattr("qceval.frameworks.cudaq.lowering.contract_to_dict", lambda _contract: payload)

    normalized = _normalize_terminal_measurements(program, object())  # type: ignore[arg-type]

    assert [operation.classical_bits for operation in normalized.operations] == [(1,), (0,)]
    assert normalized.classical_render_order == (1, 0)


def test_cudaq_measurement_free_kernel_synthesizes_declared_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA-Q sampling measures implicitly, so unmeasured kernels adopt the contract observation."""
    program = Program(
        IR_VERSION,
        2,
        0,
        (
            Operation(OperationKind.GATE, "h", (0,)),
            Operation(OperationKind.GATE, "h", (1,)),
        ),
        None,
        (),
        Provenance("cudaq", "test"),
    )
    payload: dict[str, Any] = {
        "requirements": [
            {
                "id": "terminal_observation",
                "value": {"cudaq": {"mode": "terminal", "qubits": [0, 1]}},
            }
        ]
    }
    monkeypatch.setattr("qceval.frameworks.cudaq.lowering.contract_to_dict", lambda _contract: payload)

    normalized = _normalize_terminal_measurements(program, object())  # type: ignore[arg-type]

    measurements = [op for op in normalized.operations if op.kind is OperationKind.MEASUREMENT]
    assert [op.quantum_wires for op in measurements] == [(0,), (1,)]
    assert [op.classical_bits for op in measurements] == [(0,), (1,)]
    assert normalized.num_clbits == 2
    assert normalized.classical_render_order == (1, 0)
