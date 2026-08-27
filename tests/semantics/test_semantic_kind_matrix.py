"""Positive and negative lowering coverage for every framework and contract kind."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from qceval.evals.evaluator import build_evaluator
from qceval.evals.tasks import load_tasks
from qceval.models import Framework
from qceval.semantics.contracts import ContractRegistry

FRAMEWORKS: tuple[Framework, ...] = ("qiskit", "cirq", "pennylane", "cudaq")
ROOT = Path(__file__).resolve().parents[2]
COMMON_KIND_REPRESENTATIVES = {
    "distribution": "01",
    "state": "16",
    "total_unitary": "27",
    "classical_io": "21",
}
RARE_KIND_REPRESENTATIVES = {
    "channel": "18",
    "isometry": "28",
    "instrument": "33",
}
_COMMON_PATH = Path(__file__).relative_to(ROOT)
_RARE_PATH = Path("tests/semantics/test_rare_kind_cross_framework.py")
FRAMEWORK_SEMANTIC_KIND_EVIDENCE: dict[tuple[Framework, str], dict[str, str]] = {}
for _framework in FRAMEWORKS:
    for _kind, _task_id in COMMON_KIND_REPRESENTATIVES.items():
        _case_id = f"{_kind}-{_task_id}-{_framework}"
        FRAMEWORK_SEMANTIC_KIND_EVIDENCE[(_framework, _kind)] = {
            "accepted": f"{_COMMON_PATH}::test_common_kind_representative_lowers_and_passes[{_case_id}]",
            "rejected": f"{_COMMON_PATH}::test_common_kind_behavior_mutation_is_rejected[{_case_id}]",
        }
    for _kind, _task_id in RARE_KIND_REPRESENTATIVES.items():
        _case_id = f"{_task_id}-{_framework}"
        FRAMEWORK_SEMANTIC_KIND_EVIDENCE[(_framework, _kind)] = {
            "accepted": f"{_RARE_PATH}::test_rare_kind_structural_alternate_is_accepted[{_case_id}]",
            "rejected": (f"{_RARE_PATH}::test_rare_kind_adversary_with_same_observed_output_is_rejected[{_case_id}]"),
        }

_REMOVALS = {
    ("distribution", "qiskit"): "    qc.cz(0, 1)\n",
    ("distribution", "cirq"): "    circuit.append(cirq.CZ(q[0], q[1]))\n",
    ("distribution", "pennylane"): "        qml.CZ(wires=[0, 1])\n",
    ("distribution", "cudaq"): "        cz(q[0], q[1])\n",
    ("state", "qiskit"): "    qc.cx(0,1)\n",
    ("state", "cirq"): "    circuit.append(cirq.CNOT(q[0], q[1]))\n",
    ("state", "pennylane"): "        qml.CNOT(wires=[0, 1])\n",
    ("state", "cudaq"): "        x.ctrl(q[0], q[1])\n",
    ("total_unitary", "qiskit"): "    qc.cz(0, 1)\n",
    ("total_unitary", "cirq"): "    circuit.append(cirq.CZ(q[0], q[1]))\n",
    ("total_unitary", "pennylane"): "        qml.CZ(wires=[0, 1])\n",
    ("total_unitary", "cudaq"): "        z.ctrl(q[0], q[1])\n",
    ("classical_io", "qiskit"): "    qc.ccx(0, 1, 2)\n",
    ("classical_io", "cirq"): "    circuit.append(cirq.CCX(q[0], q[1], q[2]))\n",
    ("classical_io", "pennylane"): "        qml.Toffoli(wires=[0, 1, 2])\n",
    ("classical_io", "cudaq"): "        x.ctrl([q[0], q[1]], q[2])\n",
}


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize(("kind", "task_id"), COMMON_KIND_REPRESENTATIVES.items())
def test_common_kind_representative_lowers_and_passes(framework: Framework, kind: str, task_id: str) -> None:
    task = load_tasks(framework, "core")[task_id]

    _, details = build_evaluator(framework, "core").grade_code(
        task_id=task_id,
        code=task["canonical_solution"],
        entry_point=task["entry_point"],
    )

    assert details["passed"] is True, (framework, kind, details.get("reason"))
    assert details["semantic_status"] == "verified_pass"


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize(("kind", "task_id"), COMMON_KIND_REPRESENTATIVES.items())
def test_common_kind_behavior_mutation_is_rejected(framework: Framework, kind: str, task_id: str) -> None:
    task = load_tasks(framework, "core")[task_id]
    removal = _REMOVALS[(kind, framework)]
    canonical = task["canonical_solution"]
    assert removal in canonical, (kind, framework, removal)
    candidate = canonical.replace(removal, "", 1)

    _, details = build_evaluator(framework, "core").grade_code(
        task_id=task_id,
        code=candidate,
        entry_point=task["entry_point"],
    )

    assert details["passed"] is False, (framework, kind, details.get("reason"))
    assert details["semantic_status"] == "semantic_fail", (framework, kind, details.get("reason"))


def test_framework_semantic_kind_matrix_has_positive_and_negative_evidence() -> None:
    """Fail when a newly packaged kind lacks accept/reject evidence per framework."""
    packaged_kinds = {contract.kind.value for contract in ContractRegistry.from_package("core")}
    expected_pairs = {(framework, kind) for framework in FRAMEWORKS for kind in packaged_kinds}

    assert set(FRAMEWORK_SEMANTIC_KIND_EVIDENCE) == expected_pairs
    for pair, verdicts in FRAMEWORK_SEMANTIC_KIND_EVIDENCE.items():
        assert set(verdicts) == {"accepted", "rejected"}, pair
        for verdict, reference in verdicts.items():
            path_text, separator, test_node_id = reference.partition("::")
            function_name, parameter_separator, parameter_id = test_node_id.partition("[")
            assert separator and function_name, (pair, verdict, reference)
            assert parameter_separator and parameter_id.endswith("]"), (pair, verdict, reference)
            path = ROOT / path_text
            assert path.is_file(), (pair, verdict, path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            test_functions = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            }
            assert function_name in test_functions, (pair, verdict, reference)
