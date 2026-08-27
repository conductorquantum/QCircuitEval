"""Dense matrix and statevector payloads attached to Program IR operations."""

from __future__ import annotations

import numpy as np

from qceval.semantics.ir import Operation
from qceval.semantics.lowering.utils import matrix_sha256_from_bytes, statevector_sha256_from_bytes
from qceval.semantics.verifiers.dynamic.simulator import DynamicSimulationError
from qceval.semantics.verifiers.result import SemanticStatus


def _has_dense_payload(operation: Operation) -> bool:
    data = dict(operation.semantic_data)
    return "matrix_complex128_hex" in data and "matrix_sha256" in data


def _semantic_matrix(operation: Operation) -> np.ndarray:
    data = dict(operation.semantic_data)
    payload = data.get("matrix_complex128_hex")
    digest = data.get("matrix_sha256")
    if payload is None or digest is None:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "dense_gate_payload_missing")
    try:
        raw = bytes.fromhex(payload)
        dimension = 2 ** len(operation.quantum_wires)
        matrix = np.frombuffer(raw, dtype=np.complex128).reshape((dimension, dimension))
    except (ValueError, TypeError) as exc:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "dense_gate_payload_invalid") from exc
    hashed = matrix_sha256_from_bytes(raw, dimension)
    if hashed != digest or not np.all(np.isfinite(matrix)):
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "dense_gate_payload_hash_mismatch")
    return matrix


def _semantic_statevector(operation: Operation) -> np.ndarray:
    data = dict(operation.semantic_data)
    payload = data.get("statevector_complex128_hex")
    digest = data.get("statevector_sha256")
    if payload is None or digest is None:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "statevector_payload_missing")
    try:
        raw = bytes.fromhex(payload)
        dimension = 2 ** len(operation.quantum_wires)
        state = np.frombuffer(raw, dtype=np.complex128).reshape((dimension,))
    except (ValueError, TypeError) as exc:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "statevector_payload_invalid") from exc
    hashed = statevector_sha256_from_bytes(raw, dimension)
    if hashed != digest or not np.all(np.isfinite(state)) or abs(float(np.linalg.norm(state)) - 1.0) > 1e-10:
        raise DynamicSimulationError(SemanticStatus.EXECUTION_ERROR, "statevector_payload_hash_mismatch")
    return state
