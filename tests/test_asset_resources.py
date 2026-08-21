"""Tests for packaged asset resource helpers."""

from __future__ import annotations

import pytest

from qceval.assets._resources import (
    asset_path,
    asset_root,
    contract_resource,
    read_bytes,
    read_text,
    target_resource,
    task_resource,
)
from qceval.evals.tasks import load_tasks
from qceval.semantics.contracts import ContractRegistry


def test_asset_root_exposes_packaged_tree() -> None:
    root = asset_root()
    assert root.joinpath("contracts", "core.jsonl").is_file()
    assert root.joinpath("targets", "core", "manifest.json").is_file()


def test_task_contract_and_target_helpers_resolve_known_assets() -> None:
    assert task_resource("core", "qiskit").is_file()
    assert task_resource("qec", "cirq").is_file()
    assert contract_resource("core").is_file()
    assert contract_resource("qec").is_file()
    assert target_resource("core", "target.json").is_file()
    assert target_resource("qec", "manifest.json").is_file()


def test_read_helpers_round_trip_packaged_bytes() -> None:
    text = read_text("contracts", "core.jsonl")
    payload = read_bytes("contracts", "core.jsonl")
    assert text.startswith("{")
    assert payload == text.encode("utf-8")
    assert asset_path("contracts", "core.jsonl").read_bytes() == payload


def test_unknown_suite_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        task_resource("missing", "qiskit")
    with pytest.raises(ValueError, match="unknown suite"):
        contract_resource("missing")
    with pytest.raises(ValueError, match="unknown suite"):
        target_resource("missing", "manifest.json")
    with pytest.raises(ValueError, match="unknown suite"):
        load_tasks("qiskit", suite="missing")  # type: ignore[arg-type]


def test_loaders_agree_on_packaged_suite_sizes() -> None:
    core_tasks = load_tasks("qiskit", "core")
    qec_tasks = load_tasks("qiskit", "qec")
    assert len(core_tasks) == len(ContractRegistry.from_package("core"))
    assert len(qec_tasks) == len(ContractRegistry.from_package("qec"))
    assert len(core_tasks) > 0
    assert len(qec_tasks) > 0
