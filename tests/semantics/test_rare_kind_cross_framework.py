"""Cross-framework soundness regressions for rare Core semantic kinds."""

from __future__ import annotations

import pytest

from qceval.evals.evaluator import build_evaluator, load_tasks
from qceval.semantics.contracts import ContractRegistry

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")
RARE_KIND_SOUNDNESS_INVENTORY = {
    ("channel", "18"): {
        "accepted": frozenset(FRAMEWORKS),
        "rejected": frozenset(FRAMEWORKS),
    },
    ("isometry", "28"): {
        "accepted": frozenset(FRAMEWORKS),
        "rejected": frozenset(FRAMEWORKS),
    },
    ("instrument", "33"): {
        "accepted": frozenset(FRAMEWORKS),
        "rejected": frozenset(FRAMEWORKS),
    },
    ("instrument", "34"): {
        "accepted": frozenset(FRAMEWORKS),
        "rejected": frozenset(FRAMEWORKS),
    },
}
RARE_KIND_TASKS = tuple(task_id for _, task_id in RARE_KIND_SOUNDNESS_INVENTORY)

_TASK18_WRONG_ANGLES = {
    "qiskit": ("qc.rx(pi/2,0)", "qc.rx(-pi/2,0)"),
    "cirq": ("cirq.rx(np.pi / 2).on(q[0])", "cirq.rx(-np.pi / 2).on(q[0])"),
    "pennylane": ("qml.RX(np.pi / 2, wires=0)", "qml.RX(-np.pi / 2, wires=0)"),
    "cudaq": (
        "rx(3.141592653589793 / 2.0, q[0])",
        "rx(-3.141592653589793 / 2.0, q[0])",
    ),
}
_TASK18_REVERSED_BELL_PAIRS = {
    "qiskit": (
        "qc.h(1)\n    qc.cx(1,2)",
        "qc.h(2)\n    qc.cx(2,1)",
    ),
    "cirq": (
        "cirq.H(q[1]))\n    circuit.append(cirq.CNOT(q[1], q[2]))",
        "cirq.H(q[2]))\n    circuit.append(cirq.CNOT(q[2], q[1]))",
    ),
    "pennylane": (
        "qml.Hadamard(wires=1)\n        qml.CNOT(wires=[1, 2])",
        "qml.Hadamard(wires=2)\n        qml.CNOT(wires=[2, 1])",
    ),
    "cudaq": (
        "h(q[1])\n        x.ctrl(q[1], q[2])",
        "h(q[2])\n        x.ctrl(q[2], q[1])",
    ),
}

_TASK28_UNCOMPUTE_LINES = {
    "qiskit": "    qc.ccx(0, 1, 3)\n",
    "cirq": "    circuit.append(cirq.CCX(q[0], q[1], q[3]))\n",
    "pennylane": "        qml.Toffoli(wires=[0, 1, 3])\n",
    "cudaq": "        x.ctrl([q[0], q[1]], q[3])\n",
}
_TASK28_ALTERNATE_NETWORK = {
    "qiskit": (
        ("qc.ccx(0, 1, 3)", "qc.ccx(0, 2, 3)", 2),
        ("qc.ccx(2, 3, 4)", "qc.ccx(1, 3, 4)", 1),
    ),
    "cirq": (
        ("cirq.CCX(q[0], q[1], q[3])", "cirq.CCX(q[0], q[2], q[3])", 2),
        ("cirq.CCX(q[2], q[3], q[4])", "cirq.CCX(q[1], q[3], q[4])", 1),
    ),
    "pennylane": (
        ("qml.Toffoli(wires=[0, 1, 3])", "qml.Toffoli(wires=[0, 2, 3])", 2),
        ("qml.Toffoli(wires=[2, 3, 4])", "qml.Toffoli(wires=[1, 3, 4])", 1),
    ),
    "cudaq": (
        ("x.ctrl([q[0], q[1]], q[3])", "x.ctrl([q[0], q[2]], q[3])", 2),
        ("x.ctrl([q[2], q[3]], q[4])", "x.ctrl([q[1], q[3]], q[4])", 1),
    ),
}

_INSTRUMENT_FINAL_SYSTEM_FLIPS = {
    ("33", "qiskit"): ("    return qc", "    qc.x(1)\n    return qc"),
    ("34", "qiskit"): ("    return qc", "    qc.x(1)\n    return qc"),
    ("33", "cirq"): (
        '    circuit.append(cirq.measure(q[2], q[1], key="result"))',
        '    circuit.append(cirq.X(q[0]))\n    circuit.append(cirq.measure(q[2], q[1], key="result"))',
    ),
    ("34", "cirq"): (
        '    circuit.append(cirq.measure(q[3], q[2], q[1], key="result"))',
        '    circuit.append(cirq.X(q[0]))\n    circuit.append(cirq.measure(q[3], q[2], q[1], key="result"))',
    ),
    ("33", "pennylane"): (
        "        return qml.probs(wires=[1, 0])",
        "        qml.PauliX(wires=2)\n        return qml.probs(wires=[1, 0])",
    ),
    ("34", "pennylane"): (
        "        return qml.probs(wires=[2, 1, 0])",
        "        qml.PauliX(wires=3)\n        return qml.probs(wires=[2, 1, 0])",
    ),
    ("33", "cudaq"): ("        mz(q[0])", "        x(q[2])\n        mz(q[0])"),
    ("34", "cudaq"): ("        mz(q[0])", "        x(q[3])\n        mz(q[0])"),
}

_QISKIT_DEFERRED_INSTRUMENTS = {
    "33": """\
from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

def ipe_s_gate():
    qc = QuantumCircuit(3, 2)
    qc.x(2)
    qc.h(0)
    qc.h(1)
    qc.cp(pi / 2, 0, 2)
    qc.cp(pi / 2, 1, 2)
    qc.cp(pi / 2, 1, 2)
    qc.append(QFT(2, inverse=True, do_swaps=True).to_gate(), [0, 1])
    qc.measure([0, 1], [0, 1])
    return qc
""",
    "34": """\
from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

def ipe_t_gate():
    qc = QuantumCircuit(4, 3)
    qc.x(3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.cp(pi / 4, 0, 3)
    for _ in range(2):
        qc.cp(pi / 4, 1, 3)
    for _ in range(4):
        qc.cp(pi / 4, 2, 3)
    qc.append(QFT(3, inverse=True, do_swaps=True).to_gate(), [0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc
""",
}

_CIRQ_FIRST_COUNTING_BLOCK = """\
    circuit.append(cirq.H(q[1]))
    circuit.append(cirq.CZ(q[1], q[0]))
    circuit.append(cirq.H(q[1]))
"""
_CIRQ_LAST_COUNTING_BLOCK = {
    "33": """\
    circuit.append(cirq.H(q[2]))
    circuit.append(cirq.CZ(q[2], q[0]) ** 0.5)
    circuit.append(cirq.Z(q[2]) ** -0.5)
    circuit.append(cirq.H(q[2]))
""",
    "34": """\
    circuit.append(cirq.H(q[3]))
    circuit.append(cirq.CZ(q[3], q[0]) ** 0.25)
    circuit.append(cirq.Z(q[3]) ** -0.25)
    circuit.append(cirq.H(q[3]))
""",
}
_FIRST_PHASE_LINE = {
    ("33", "pennylane"): "        qml.ctrl(qml.PhaseShift(np.pi / 2, wires=2), control=0)\n",
    ("34", "pennylane"): "        qml.ctrl(qml.PhaseShift(np.pi / 4, wires=3), control=0)\n",
    ("33", "cudaq"): "        r1.ctrl(np.pi / 2.0, q[0], q[2])\n",
    ("34", "cudaq"): "        r1.ctrl(np.pi / 4.0, q[0], q[3])\n",
}
_LAST_PHASE_LINE = {
    ("33", "pennylane"): "        qml.ctrl(qml.PhaseShift(np.pi / 2, wires=2), control=1)\n",
    ("34", "pennylane"): "        qml.ctrl(qml.PhaseShift(np.pi / 4, wires=3), control=2)\n",
    ("33", "cudaq"): "        r1.ctrl(np.pi / 2.0, q[1], q[2])\n",
    ("34", "cudaq"): "        r1.ctrl(np.pi / 4.0, q[2], q[3])\n",
}


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, f"mutation anchor changed: {old!r}"
    return source.replace(old, new, 1)


def _remove_last(source: str, line: str) -> str:
    before, found, after = source.rpartition(line)
    assert found, f"mutation anchor changed: {line!r}"
    return before + after


def _move_after_last(source: str, item: str, anchor: str) -> str:
    source = _replace_once(source, item, "")
    before, found, after = source.rpartition(anchor)
    assert found, f"mutation anchor changed: {anchor!r}"
    return before + anchor + item + after


def _wrong_candidate(task_id: str, framework: str, canonical: str) -> str:
    if task_id == "18":
        return _replace_once(canonical, *_TASK18_WRONG_ANGLES[framework])
    if task_id == "28":
        return _remove_last(canonical, _TASK28_UNCOMPUTE_LINES[framework])
    return _replace_once(canonical, *_INSTRUMENT_FINAL_SYSTEM_FLIPS[(task_id, framework)])


def _valid_alternate(task_id: str, framework: str, canonical: str) -> str:
    if task_id == "18":
        return _replace_once(canonical, *_TASK18_REVERSED_BELL_PAIRS[framework])
    if task_id == "28":
        result = canonical
        for old, new, expected_count in _TASK28_ALTERNATE_NETWORK[framework]:
            assert result.count(old) == expected_count, f"mutation anchor changed: {old!r}"
            result = result.replace(old, new)
        return result
    if framework == "qiskit":
        return _QISKIT_DEFERRED_INSTRUMENTS[task_id]
    if framework == "cirq":
        return _move_after_last(canonical, _CIRQ_FIRST_COUNTING_BLOCK, _CIRQ_LAST_COUNTING_BLOCK[task_id])
    return _move_after_last(
        canonical,
        _FIRST_PHASE_LINE[(task_id, framework)],
        _LAST_PHASE_LINE[(task_id, framework)],
    )


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("task_id", RARE_KIND_TASKS)
def test_rare_kind_adversary_with_same_observed_output_is_rejected(
    task_id: str,
    framework: str,
) -> None:
    """Reject behavior hidden by the task's old all-zero/fixed-output observation."""
    task = load_tasks(framework, suite="core")[task_id]
    code = _wrong_candidate(task_id, framework, task["canonical_solution"])

    execution, details = build_evaluator(framework, suite="core").grade_code(
        task_id=task_id,
        code=code,
        entry_point=task["entry_point"],
    )

    assert code != task["canonical_solution"]
    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"
    if task_id == "18":
        assert execution.probabilities == pytest.approx([0.5, 0.5], abs=1e-6)
    else:
        # Task 28 still renders 0000 on the default input despite leaking its
        # work qubit on other inputs. Instrument mutants retain the exact 01 /
        # 001 classical phase outcome while corrupting only the conditional state.
        assert execution.probabilities[0 if task_id == "28" else 1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("task_id", RARE_KIND_TASKS)
def test_rare_kind_structural_alternate_is_accepted(
    task_id: str,
    framework: str,
) -> None:
    """Accept equivalent Bell, clean-ancilla, and phase-estimation constructions."""
    task = load_tasks(framework, suite="core")[task_id]
    code = _valid_alternate(task_id, framework, task["canonical_solution"])

    _, details = build_evaluator(framework, suite="core").grade_code(
        task_id=task_id,
        code=code,
        entry_point=task["entry_point"],
    )

    assert code != task["canonical_solution"]
    assert details["passed"] is True, (task_id, framework, details.get("reason"))
    assert details["semantic_status"] == "verified_pass"


def test_every_packaged_rare_kind_has_explicit_cross_framework_soundness_inventory() -> None:
    """Fail when a future rare-kind task lacks explicit accept/reject coverage."""
    rare_kinds = frozenset({"channel", "instrument", "isometry"})
    packaged = {
        (contract.kind.value, contract.task_id)
        for contract in ContractRegistry.from_package("core")
        if contract.kind.value in rare_kinds
    }

    assert set(RARE_KIND_SOUNDNESS_INVENTORY) == packaged
    for coverage in RARE_KIND_SOUNDNESS_INVENTORY.values():
        assert coverage["accepted"] == frozenset(FRAMEWORKS)
        assert coverage["rejected"] == frozenset(FRAMEWORKS)


# Residual construction gaps are intentionally explicit:
# - Qiskit exercises dynamic-IPE versus deferred-QPE equivalence. Cirq,
#   PennyLane, and CUDA-Q use reordered commuting phase blocks because their
#   benchmark return interfaces do not expose a portable dynamic alternate.
