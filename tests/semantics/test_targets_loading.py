"""Tests for grouped-manifest target loading and hash verification."""

from __future__ import annotations

import pytest

from qceval.semantics.contracts import ContractRegistry
from qceval.semantics.lowering.base import LoweringStatus
from qceval.semantics.lowering.utils import (
    bounded_matrix_semantic_data,
    bounded_statevector_semantic_data,
    lowering_failure,
    matrix_sha256,
    normalize_parameter,
)
from qceval.semantics.targets import (
    PILOT_TASK_IDS,
    TargetValidationError,
    load_contract_target_document,
    load_packaged_target_manifest,
    verify_all_pilot_targets,
    verify_packaged_target,
)


def test_load_packaged_target_manifest_uses_grouped_suite_asset() -> None:
    manifest = load_packaged_target_manifest("16")
    assert manifest.suite == "core"
    assert manifest.task_id == "16"
    assert manifest.artifact == "target.json"


@pytest.mark.parametrize("task_id", PILOT_TASK_IDS)
def test_verify_packaged_pilot_targets(task_id: str) -> None:
    verification = verify_packaged_target(task_id)
    assert verification.task_id == task_id
    assert len(verification.artifact_sha256) == 64


def test_verify_all_pilot_targets() -> None:
    results = verify_all_pilot_targets()
    assert {item.task_id for item in results} == set(PILOT_TASK_IDS)


def test_load_contract_target_document_hashes_selected_task_only() -> None:
    contract = ContractRegistry.from_package().get("core", "01")
    document = load_contract_target_document(contract)
    assert document["task_id"] == "01"
    assert "target" in document


def test_matrix_sha256_is_shape_addressed() -> None:
    digest = matrix_sha256([[1, 0], [0, 1]])
    assert len(digest) == 64
    with pytest.raises(ValueError):
        matrix_sha256([[1, 0]])


def test_bounded_semantic_payloads_and_lowering_failure() -> None:
    matrix_data = dict(bounded_matrix_semantic_data([[1, 0], [0, -1]], wire_order="little_endian"))
    assert "matrix_sha256" in matrix_data
    assert matrix_data["matrix_wire_order"] == "little_endian"
    state = dict(bounded_statevector_semantic_data([1, 0], wire_order="big_endian"))
    assert "statevector_sha256" in state
    failure = lowering_failure(LoweringStatus.UNSUPPORTED, "nope", node_kind="x", detail="d")
    assert failure.status is LoweringStatus.UNSUPPORTED
    assert failure.error is not None
    assert failure.error.reason == "nope"
    with pytest.raises(ValueError):
        bounded_matrix_semantic_data([[1, 0], [0, 1]], wire_order="middle_endian")
    with pytest.raises(ValueError):
        bounded_statevector_semantic_data([1, 1])


def test_normalize_parameter_numbers_and_text() -> None:
    assert normalize_parameter(0).kind.value == "number"
    assert normalize_parameter("theta").kind.value == "text"
    with pytest.raises(ValueError):
        normalize_parameter(float("nan"))


def test_verify_packaged_target_rejects_unknown_task() -> None:
    with pytest.raises((TargetValidationError, ValueError, KeyError)):
        verify_packaged_target("99")
