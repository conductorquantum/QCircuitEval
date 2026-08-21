"""Executable per-contract scientific validation inventory.

Canonical-solution sweeps are intentionally absent from this inventory.  A
contract is complete only when it has independently reviewed target evidence,
a nonpassing plausible implementation, and a noncanonical passing
implementation.
"""

from __future__ import annotations

import ast
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from tests.semantics.scientific_validation_inventory import (
    CORE_TASK_IDS,
    QEC_TASK_IDS,
    RARE_KIND_TASK_IDS,
    REQUIRED_VALIDATION_CATEGORIES,
    SCIENTIFIC_VALIDATION_INVENTORY,
)

from qceval.evals.evaluator import build_evaluator
from qceval.evals.tasks import load_tasks
from qceval.semantics._target_generators import (
    pilot_target_generator,
    pilot_target_provenance,
)
from qceval.semantics.contracts import ContractRegistry
from qceval.semantics.contracts.kinds import FrozenArray, FrozenObject
from qceval.semantics.targets import load_contract_target_document

ROOT = Path(__file__).resolve().parents[2]


def _test_functions(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def test_scientific_validation_inventory_exactly_covers_packaged_contracts() -> None:
    packaged = {contract.key for suite in ("core", "qec") for contract in ContractRegistry.from_package(suite)}

    assert set(SCIENTIFIC_VALIDATION_INVENTORY) == packaged
    assert len(packaged) == 58 + 12
    for key, evidence in SCIENTIFIC_VALIDATION_INVENTORY.items():
        assert set(evidence) == REQUIRED_VALIDATION_CATEGORIES, key
        for category, references in evidence.items():
            assert references, (key, category)
            for reference in references:
                path_text, separator, function_name = reference.partition("::")
                assert separator and function_name, (key, category, reference)
                path = ROOT / path_text
                assert path.is_file(), (key, category, path)
                assert function_name in _test_functions(path), (key, category, reference)


def test_core_target_derivation_inventory_is_complete() -> None:
    """Bind every Core target to reviewed derivation and cross-check evidence."""
    audit = json.loads((ROOT / "src/qceval/assets/contracts/core-audit-source.json").read_text(encoding="utf-8"))[
        "tasks"
    ]
    pilot_ids = {task_id for task_id in CORE_TASK_IDS if pilot_target_generator(task_id) is not None}
    assert pilot_ids == {"02", "27", "28", "42"}
    assert set(audit) == set(CORE_TASK_IDS) - pilot_ids
    provenance_ids = {task_id for task_id in CORE_TASK_IDS if pilot_target_provenance(task_id) is not None}
    assert provenance_ids == pilot_ids

    registry = ContractRegistry.from_package("core")
    for contract in registry:
        document = load_contract_target_document(contract)
        assert contract.target.independent_derivations >= 2
        generator = pilot_target_generator(contract.task_id)
        if generator is not None:
            assert document == generator()
            record = pilot_target_provenance(contract.task_id)
            assert record is not None
            assert record["audit_status"] == "reviewed"
            assert record["derivation"].strip()
            assert record["crosscheck"].strip()
            assert record["reviewer"].strip()
            assert record["review_evidence"].strip()
            continue
        record = audit[contract.task_id]
        assert record["derivation"].strip()
        assert record["crosscheck"].strip()
        assert record["reviewer"].strip()
        assert document["target"] == record["target"]


def _syndrome(error: int | None, supports: Sequence[Sequence[int]]) -> str:
    return "".join(reversed(["0" if error is None or error not in support else "1" for support in supports]))


def _distribution_cases(points: Sequence[tuple[Any, ...]], outcome: Callable[[tuple[Any, ...]], str]) -> dict[str, Any]:
    return {
        "type": "argument_distribution_cases",
        "cases": [
            {
                "arguments": list(point),
                "distribution": {
                    "type": "exact_distribution",
                    "probabilities": {outcome(point): "1"},
                },
            }
            for point in points
        ],
    }


def _independent_qec_targets() -> dict[str, dict[str, Any]]:
    binary = ((0,), (1,))
    errors3 = tuple((error,) for error in (None, 0, 1, 2))
    errors7 = tuple((error,) for error in (None, *range(7)))
    errors9 = tuple((error,) for error in (None, *range(9)))
    logical_errors3 = tuple(itertools.product((0, 1), (None, 0, 1, 2)))
    logical_errors5 = tuple(itertools.product((0, 1), (None, *range(5))))
    logical_errors7 = tuple(itertools.product((0, 1), (None, *range(7))))
    logical_pairs = tuple(itertools.product((0, 1), repeat=2))
    shor_support = tuple("".join(blocks) for blocks in itertools.product(("000", "111"), repeat=3))
    steane_zero = (
        "0000000",
        "0011110",
        "0101101",
        "0110011",
        "1001011",
        "1010101",
        "1100110",
        "1111000",
    )
    steane_one = tuple("".join("1" if bit == "0" else "0" for bit in word) for word in steane_zero)

    def exact_state(amplitudes: Mapping[str, str]) -> dict[str, Any]:
        return {"type": "exact_state", "amplitudes": amplitudes}

    return {
        "qec01": _distribution_cases(binary, lambda point: str(point[0])),
        "qec02": _distribution_cases(errors3, lambda point: _syndrome(point[0], ((0, 1), (1, 2)))),
        "qec03": _distribution_cases(logical_errors3, lambda point: str(point[0])),
        "qec04": _distribution_cases(logical_errors3, lambda point: str(point[0])),
        "qec05": _distribution_cases(logical_pairs, lambda point: f"{point[0] ^ point[1]}{point[0]}"),
        "qec06": exact_state(dict.fromkeys(shor_support, "1/sqrt(8)")),
        "qec07": {
            "type": "argument_state_cases",
            "cases": [
                {"arguments": [0], "state": exact_state(dict.fromkeys(steane_zero, "1/sqrt(8)"))},
                {"arguments": [1], "state": exact_state(dict.fromkeys(steane_one, "1/sqrt(8)"))},
            ],
        },
        "qec08": _distribution_cases(
            errors7,
            lambda point: _syndrome(point[0], ((0, 2, 4, 6), (1, 2, 5, 6), (3, 4, 5, 6))),
        ),
        "qec09": _distribution_cases(logical_errors7, lambda point: "000000" + str(point[0])),
        "qec10": _distribution_cases(
            errors9,
            lambda point: _syndrome(point[0], ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8))),
        ),
        "qec11": _distribution_cases(
            errors9,
            lambda point: _syndrome(point[0], (tuple(range(6)), tuple(range(3, 9)))),
        ),
        "qec12": _distribution_cases(logical_errors5, lambda point: str(point[0])),
    }


@pytest.mark.parametrize("task_id", QEC_TASK_IDS)
def test_qec_target_matches_independent_formula(task_id: str) -> None:
    """Check QEC truth tables and codewords without calling the target generator."""
    contract = ContractRegistry.from_package("qec").get("qec", task_id)
    assert contract.target.independent_derivations >= 2
    assert load_contract_target_document(contract)["target"] == _independent_qec_targets()[task_id]


def _plain(value: Any) -> Any:
    if isinstance(value, FrozenArray):
        return [_plain(item) for item in value.items]
    if isinstance(value, FrozenObject):
        return {key: _plain(item) for key, item in value.items}
    return value


def _terminal_qiskit_interface(contract: Any) -> Mapping[str, Any]:
    requirement = next(item for item in contract.requirements if item.requirement_id == "terminal_observation")
    return _plain(requirement.value)["qiskit"]


def _returned_circuit_name(function: ast.FunctionDef) -> tuple[str, ast.Return]:
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    returned = max(returns, key=lambda node: node.lineno)
    returned_names = {node.id for node in ast.walk(returned) if isinstance(node, ast.Name)}
    candidates: list[str] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        callee = node.value.func
        is_circuit = (isinstance(callee, ast.Name) and callee.id == "QuantumCircuit") or (
            isinstance(callee, ast.Attribute) and callee.attr == "QuantumCircuit"
        )
        if is_circuit:
            candidates.extend(
                target.id for target in node.targets if isinstance(target, ast.Name) and target.id in returned_names
            )
    assert candidates
    return candidates[-1], returned


def _insert_terminal_x(source: str, entry_point: str, wire: int) -> str:
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_point)
    circuit_name, returned = _returned_circuit_name(function)
    measurements = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "measure"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == circuit_name
    ]
    anchor = min(measurements, key=lambda node: node.lineno) if measurements else returned
    lines = source.splitlines()
    lines.insert(anchor.lineno - 1, f"{' ' * anchor.col_offset}{circuit_name}.x({wire})")
    return "\n".join(lines) + "\n"


def _known_wrong_core_candidate(task_id: str, canonical: str, entry_point: str, wire: int) -> str:
    substitutions = {
        "04": ("qc.rz(2 * gamma[i], qr[v])", "qc.rz(-2 * gamma[i], qr[v])"),
        "14": (
            "    for i in range(3):\n        qc.h(i)\n        qc.measure(i,i)",
            "    for i in range(3):\n        qc.h(i)\n    qc.x(1)\n    for i in range(3):\n        qc.measure(i,i)",
        ),
        "47": ("    qc.cx(0, 2)\n", ""),
        "54": ("zz(0, 1, 0.5 * half)", "zz(0, 1, 0.6 * half)"),
    }
    if task_id not in substitutions:
        return _insert_terminal_x(canonical, entry_point, wire)
    old, new = substitutions[task_id]
    assert canonical.count(old) >= 1, (task_id, old)
    return canonical.replace(old, new, 1)


@pytest.mark.parametrize("task_id", sorted(set(CORE_TASK_IDS) - RARE_KIND_TASK_IDS))
def test_core_known_wrong_implementation_is_rejected(task_id: str) -> None:
    task = load_tasks("qiskit", "core")[task_id]
    contract = ContractRegistry.from_package("core").get("core", task_id)
    interface = _terminal_qiskit_interface(contract)
    wire = int((interface.get("qubits") or [0])[0])
    candidate = _known_wrong_core_candidate(
        task_id,
        task["canonical_solution"],
        task["entry_point"],
        wire,
    )

    _, details = build_evaluator("qiskit", "core").grade_code(
        task_id=task_id,
        code=candidate,
        entry_point=task["entry_point"],
    )

    assert candidate != task["canonical_solution"]
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail", (task_id, details.get("reason"))


def _decompose_first_gate(source: str) -> str:
    """Replace one native gate with an exact, nontrivial gate decomposition."""
    priority = {gate: index for index, gate in enumerate(("cx", "cz", "x", "z", "h", "rx"))}
    tree = ast.parse(source)
    candidates: list[tuple[int, int, int, ast.Expr]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr in priority
        ):
            candidates.append((priority[node.value.func.attr], node.lineno, node.col_offset, node))
    assert candidates
    _, _, _, expression = min(candidates)
    call = expression.value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    owner = ast.get_source_segment(source, call.func.value)
    arguments = [ast.get_source_segment(source, argument) for argument in call.args]
    gate = call.func.attr
    if gate in {"cx", "cz"}:
        replacement_gate = "cz" if gate == "cx" else "cx"
        control, target = arguments
        calls = [
            f"{owner}.h({target})",
            f"{owner}.{replacement_gate}({control}, {target})",
            f"{owner}.h({target})",
        ]
    elif gate in {"x", "z"}:
        replacement_gate = "z" if gate == "x" else "x"
        target = arguments[0]
        calls = [f"{owner}.h({target})", f"{owner}.{replacement_gate}({target})", f"{owner}.h({target})"]
    elif gate == "h":
        target = arguments[0]
        calls = [f"{owner}.ry(1.5707963267948966, {target})", f"{owner}.x({target})"]
    else:
        angle, target = arguments
        calls = [f"{owner}.h({target})", f"{owner}.rz({angle}, {target})", f"{owner}.h({target})"]
    lines = source.splitlines()
    lines[expression.lineno - 1 : expression.end_lineno] = [f"{' ' * expression.col_offset}{item}" for item in calls]
    return "\n".join(lines) + "\n"


_SPECIAL_CORE_ALTERNATES = {
    "39": """\
from qiskit import QuantumCircuit

def quantum_state_preparation(parameters):
    theta, phi = parameters
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.ry(phi, 0)
    return qc
""",
    "40": """\
from qiskit import QuantumCircuit

def VQE_2(parameters):
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.rz(parameters[0], 0); qc.rz(parameters[1], 0); qc.ry(parameters[2], 0)
    qc.rz(parameters[3], 1); qc.cx(0, 1); qc.rz(parameters[4], 0)
    qc.rz(parameters[5], 0); qc.ry(parameters[6], 0); qc.rz(parameters[7], 1)
    return qc
""",
    "41": """\
from qiskit import QuantumCircuit

def VQE_Z2(param):
    qc = QuantumCircuit(2)
    qc.rz(param[3], 1); qc.rz(param[0], 0); qc.ry(param[4], 1)
    qc.ry(param[1], 0); qc.rz(param[5], 1); qc.rz(param[2], 0)
    return qc
""",
    "42": """\
from qiskit import QuantumCircuit
import numpy as np

def U_gate_decompose(theta, phi, lam):
    qc = QuantumCircuit(1)
    qc.rz(lam, 0); qc.sx(0); qc.rz(theta + np.pi, 0)
    qc.sx(0); qc.rz(phi + np.pi, 0)
    return qc
""",
    "27": """\
from qiskit import QuantumCircuit

def decompose_CNOT():
    qc = QuantumCircuit(2)
    for _ in range(2):
        qc.h(1); qc.cz(0, 1); qc.h(1)
        qc.h(0); qc.cz(1, 0); qc.h(0)
        qc.h(1); qc.cz(0, 1); qc.h(1)
        if _ == 0:
            qc.h(0); qc.cz(1, 0); qc.h(0)
    return qc
""",
}


def _alternate_core_candidate(task_id: str, canonical: str) -> str:
    if task_id in _SPECIAL_CORE_ALTERNATES:
        return _SPECIAL_CORE_ALTERNATES[task_id]
    substitutions = {
        "04": ("for u, v in G.edges:", "for u, v in reversed(list(G.edges)):", 1),
        "11": (
            "    qc.x(4)\n    qc.h(4)\n    qc.h([0, 1, 2, 3])",
            "    qc.h([0, 1, 2, 3])\n    qc.x(4)\n    qc.h(4)",
            1,
        ),
        "14": (
            "    for i in range(3):\n        qc.h(i)",
            "    qc.h(2)\n    qc.h(1)\n    qc.h(0)",
            1,
        ),
        "20": (
            "    qc.x(3)\n    qc.h([0,1,2,3])",
            "    qc.h([0,1,2])\n    qc.x(3)\n    qc.h(3)",
            1,
        ),
        "36": (
            "    qc.measure(0, 0)\n    qc.measure(1, 1)",
            "    qc.measure([1, 0], [1, 0])",
            1,
        ),
        "43": ("optimization_level=1", "optimization_level=0", 1),
        "46": (
            "    for i in range(3):\n        qc.h(i)",
            "    for i in (2, 1, 0):\n        qc.h(i)",
            1,
        ),
        "47": (
            "    qc.x(2)\n    qc.h([0, 1, 2])",
            "    qc.h([0, 1])\n    qc.x(2)\n    qc.h(2)",
            1,
        ),
    }
    if task_id not in substitutions:
        return _decompose_first_gate(canonical)
    old, new, count = substitutions[task_id]
    assert canonical.count(old) >= count, (task_id, old)
    return canonical.replace(old, new, count)


@pytest.mark.parametrize("task_id", sorted(set(CORE_TASK_IDS) - RARE_KIND_TASK_IDS))
def test_core_alternate_valid_implementation_passes(task_id: str) -> None:
    task = load_tasks("qiskit", "core")[task_id]
    candidate = _alternate_core_candidate(task_id, task["canonical_solution"])

    _, details = build_evaluator("qiskit", "core").grade_code(
        task_id=task_id,
        code=candidate,
        entry_point=task["entry_point"],
    )

    assert candidate != task["canonical_solution"]
    assert details["passed"] is True, (task_id, details.get("semantic_status"), details.get("reason"))
    assert details["semantic_status"] == "verified_pass"


class _RemoveMeasurements(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:  # noqa: N802
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "measure"
        ):
            return None
        return self.generic_visit(node)

    def visit_For(self, node: ast.For) -> ast.AST | None:  # noqa: N802
        updated = self.generic_visit(node)
        assert isinstance(updated, ast.For)
        return updated if updated.body else None


def _reorder_terminal_measurements(
    source: str,
    entry_point: str,
    qubits: Sequence[int],
    classical_bits: Sequence[int],
) -> str:
    tree = _RemoveMeasurements().visit(ast.parse(source))
    assert isinstance(tree, ast.Module)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_point)
    return_index = next(index for index, node in enumerate(function.body) if isinstance(node, ast.Return))
    measurement = ast.parse(f"qc.measure({list(reversed(qubits))}, {list(reversed(classical_bits))})").body[0]
    function.body.insert(return_index, measurement)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


@pytest.mark.parametrize(
    "task_id",
    tuple(task_id for task_id in QEC_TASK_IDS if task_id not in {"qec01", "qec03", "qec04", "qec12"}),
)
def test_qec_alternate_terminal_measurement_order_passes(task_id: str) -> None:
    """Exercise equivalent grouped/reordered terminal observation mappings."""
    task = load_tasks("qiskit", "qec")[task_id]
    contract = ContractRegistry.from_package("qec").get("qec", task_id)
    interface = _terminal_qiskit_interface(contract)
    candidate = _reorder_terminal_measurements(
        task["canonical_solution"],
        task["entry_point"],
        interface["qubits"],
        interface["classical_bits"],
    )

    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id=task_id,
        code=candidate,
        entry_point=task["entry_point"],
    )

    assert candidate != task["canonical_solution"]
    assert details["passed"] is True, (task_id, details.get("semantic_status"), details.get("reason"))
    assert details["semantic_status"] == "verified_pass"


def test_qec12_alternate_encoder_and_decoder_order_passes() -> None:
    """Reverse commuting fanout orders around the real syndrome correction."""
    task = load_tasks("qiskit", "qec")["qec12"]
    canonical = task["canonical_solution"]
    forward = "qc.cx(0, 1); qc.cx(0, 2); qc.cx(0, 3); qc.cx(0, 4)"
    reverse = "qc.cx(0, 4); qc.cx(0, 3); qc.cx(0, 2); qc.cx(0, 1)"
    assert canonical.count(forward) == 1
    assert canonical.count(reverse) == 1
    marker = "__QEC12_ENCODER_FANOUT__"
    candidate = canonical.replace(forward, marker, 1).replace(reverse, forward, 1).replace(marker, reverse, 1)

    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec12",
        code=candidate,
        entry_point=task["entry_point"],
    )

    assert candidate != canonical
    assert details["passed"] is True, (details.get("semantic_status"), details.get("reason"))
    assert details["semantic_status"] == "verified_pass"
