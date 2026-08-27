"""QEC contract, target, and adversarial semantic-grading tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from qceval.evals.evaluator import build_evaluator
from qceval.evals.tasks import load_tasks
from qceval.semantics.contracts import ContractRegistry, ContractValidationError, contract_hash, parse_contract
from qceval.semantics.targets import load_contract_target_document

QEC_IDS = tuple(f"qec{index:02d}" for index in range(1, 13))
FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")


def test_qec_registry_is_shared_complete_and_hash_addressed() -> None:
    registry = ContractRegistry.from_package("qec")

    assert len(registry) == 12
    assert tuple(contract.task_id for contract in registry) == QEC_IDS
    for contract in registry:
        assert contract.schema_version == "2"
        assert contract.suite == "qec"
        assert contract.shadow_only is False
        assert len(contract_hash(contract)) == 64
        document = load_contract_target_document(contract)
        assert document["suite"] == "qec"
        assert document["task_id"] == contract.task_id
        assert document["kind"] == contract.kind.value


def test_qec_contract_signature_and_domain_match_every_framework_prompt() -> None:
    registry = ContractRegistry.from_package("qec")
    framework_tasks = {framework: load_tasks(framework, "qec") for framework in FRAMEWORKS}

    for task_id in QEC_IDS:
        contract = registry.get("qec", task_id)
        for framework in FRAMEWORKS:
            task = framework_tasks[framework][task_id]
            assert task["entry_point"] == contract.signature.entry_point
            assert task["canonical_class"]["case_arg_names"] == [
                argument.name for argument in contract.signature.arguments
            ]
            if contract.parameters.diagnostic_points:
                assert (
                    task["canonical_class"]["cases"] == framework_tasks["qiskit"][task_id]["canonical_class"]["cases"]
                )
                assert [tuple(case["args"]) for case in task["canonical_class"]["cases"]] == list(
                    contract.parameters.diagnostic_points
                )


def test_qec12_prompts_share_the_fixed_repetition_encoding_convention() -> None:
    for framework in FRAMEWORKS:
        prompt = load_tasks(framework, "qec")["qec12"]["prompt"]
        assert "Encoding convention:" in prompt
        assert "Do not initialize all five data" in prompt
        assert "four encoding CNOTs" not in prompt
        assert "must remain unconditional fixed circuit structure" not in prompt


def test_distribution_targets_match_legacy_cases_during_migration() -> None:
    registry = ContractRegistry.from_package("qec")
    tasks = load_tasks("qiskit", "qec")

    for task_id in QEC_IDS:
        contract = registry.get("qec", task_id)
        if contract.kind.value != "distribution":
            continue
        target = load_contract_target_document(contract)["target"]
        cases = target["cases"]
        expected = tasks[task_id]["canonical_class"]["cases"]
        assert [case["arguments"] for case in cases] == [case["args"] for case in expected]
        assert [next(iter(case["distribution"]["probabilities"])) for case in cases] == [
            case["grader"]["expected_dominants"][0] for case in expected
        ]


def test_state_targets_preserve_codeword_relative_phase() -> None:
    registry = ContractRegistry.from_package("qec")
    shor = load_contract_target_document(registry.get("qec", "qec06"))["target"]
    steane = load_contract_target_document(registry.get("qec", "qec07"))["target"]

    assert len(shor["amplitudes"]) == 8
    assert set(shor["amplitudes"].values()) == {"1/sqrt(8)"}
    assert len(steane["cases"]) == 2
    assert all(len(case["state"]["amplitudes"]) == 8 for case in steane["cases"])
    assert all(set(case["state"]["amplitudes"].values()) == {"1/sqrt(8)"} for case in steane["cases"])


def test_schema_two_accepts_nullable_exhaustive_points_and_schema_one_rejects_them() -> None:
    contract = ContractRegistry.from_package("qec").get("qec", "qec02")
    payload = json.loads(json.dumps(_contract_dict(contract)))

    assert parse_contract(payload).parameters.diagnostic_points[0] == (None,)
    payload["schema_version"] = "1"
    with pytest.raises(ContractValidationError, match="schema version 1 supports only numeric points"):
        parse_contract(payload)


def test_generated_qec_assets_are_reproducible() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, str(root / "ci/generate_qec_semantic_assets.py"), "--check"],
        cwd=root,
        check=True,
    )


@pytest.mark.parametrize(
    ("task_id", "mutate", "expected_reason"),
    [
        (
            "qec06",
            lambda code: code.replace(
                "    for index in range(9):",
                "    qc.z(0)\n    for index in range(9):",
            ),
            "semantic_failure",
        ),
        (
            "qec03",
            lambda code: code.replace(
                "    if error_qubit is not None:\n        qc.x(error_qubit)\n",
                "",
            ),
            "requirement_failed:case_program_invariance",
        ),
        (
            "qec01",
            lambda code: code.replace("    qc.measure", "    qc.h(0)\n    qc.measure", 1),
            "parameter_domain_semantic_fail",
        ),
        (
            "qec01",
            lambda code: code.replace(
                "from qiskit import QuantumCircuit\n",
                "from qiskit import QuantumCircuit\nif False:\n    import stim\n",
            ),
            "parameter_domain_semantic_fail",
        ),
    ],
)
def test_qec_release_blocking_mutants_fail(
    task_id: str,
    mutate: Any,
    expected_reason: str,
) -> None:
    task = load_tasks("qiskit", "qec")[task_id]
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id=task_id,
        code=mutate(task["canonical_solution"]),
        entry_point=task["entry_point"],
    )

    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"
    assert details["reason"] == expected_reason


_TRIVIAL_SHORTCUTS = {
    "qec01": (
        "bit_flip_encode_decode",
        """
from qiskit import QuantumCircuit
def bit_flip_encode_decode(logical_bit: int):
    qc = QuantumCircuit(3, 1)
    if logical_bit == 1:
        qc.x(0)
    qc.measure(0, 0)
    return qc
""",
    ),
    "qec02": (
        "bit_flip_syndrome",
        """
from qiskit import QuantumCircuit
def bit_flip_syndrome(error_qubit):
    qc = QuantumCircuit(5, 2)
    if error_qubit == 0:
        qc.x(0); qc.x(3)
    if error_qubit == 1:
        qc.x(1); qc.x(3); qc.x(4)
    if error_qubit == 2:
        qc.x(2); qc.x(4)
    qc.measure(3, 0); qc.measure(4, 1)
    return qc
""",
    ),
    "qec03": (
        "bit_flip_correct",
        """
from qiskit import QuantumCircuit
def bit_flip_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(3, 1)
    if logical_bit == 1:
        qc.x(0)
    if error_qubit is not None:
        qc.x(error_qubit)
        qc.x(error_qubit)
    qc.measure(0, 0)
    return qc
""",
    ),
    "qec04": (
        "phase_flip_correct",
        """
from qiskit import QuantumCircuit
def phase_flip_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(3, 1)
    if logical_bit == 1:
        qc.x(0)
    if error_qubit is not None:
        qc.z(error_qubit)
    qc.measure(0, 0)
    return qc
""",
    ),
    "qec05": (
        "repetition_logical_cnot",
        """
from qiskit import QuantumCircuit
def repetition_logical_cnot(control_bit, target_bit):
    qc = QuantumCircuit(6, 2)
    if control_bit == 1:
        qc.x(0)
    if control_bit ^ target_bit == 1:
        qc.x(3)
    qc.measure(0, 0); qc.measure(3, 1)
    return qc
""",
    ),
    "qec06": (
        "shor_encode_zero",
        """
from qiskit import QuantumCircuit
def shor_encode_zero():
    qc = QuantumCircuit(9, 9)
    for index in range(9):
        qc.h(index)
        qc.measure(index, index)
    return qc
""",
    ),
    "qec07": (
        "steane_encode",
        """
from qiskit import QuantumCircuit
def steane_encode(logical_bit):
    qc = QuantumCircuit(7, 7)
    if logical_bit == 1:
        for index in range(7):
            qc.x(index)
    for index in range(7):
        qc.measure(index, index)
    return qc
""",
    ),
    "qec08": (
        "steane_z_syndrome",
        """
from qiskit import QuantumCircuit
def steane_z_syndrome(error_qubit):
    qc = QuantumCircuit(10, 3)
    lut = {0: [7], 1: [8], 2: [7, 8], 3: [9], 4: [7, 9], 5: [8, 9], 6: [7, 8, 9]}
    if error_qubit is not None:
        qc.x(error_qubit)
        for ancilla in lut[error_qubit]:
            qc.x(ancilla)
    qc.measure(7, 0); qc.measure(8, 1); qc.measure(9, 2)
    return qc
""",
    ),
    "qec09": (
        "steane_x_correct",
        """
from qiskit import QuantumCircuit
def steane_x_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(10, 1)
    if logical_bit == 1:
        qc.x(0)
    qc.measure(0, 0)
    return qc
""",
    ),
    "qec10": (
        "shor_z_syndrome",
        """
from qiskit import QuantumCircuit
def shor_z_syndrome(error_qubit):
    qc = QuantumCircuit(15, 6)
    lut = {0: [9], 1: [9, 10], 2: [10], 3: [11], 4: [11, 12], 5: [12], 6: [13], 7: [13, 14], 8: [14]}
    if error_qubit is not None:
        qc.x(error_qubit)
        for ancilla in lut[error_qubit]:
            qc.x(ancilla)
    for offset in range(6):
        qc.measure(9 + offset, offset)
    return qc
""",
    ),
    "qec11": (
        "shor_x_syndrome",
        """
from qiskit import QuantumCircuit
def shor_x_syndrome(error_qubit):
    qc = QuantumCircuit(11, 2)
    if error_qubit is not None:
        qc.z(error_qubit)
        if error_qubit <= 2:
            qc.x(9)
        elif error_qubit <= 5:
            qc.x(9); qc.x(10)
        else:
            qc.x(10)
    qc.measure(9, 0); qc.measure(10, 1)
    return qc
""",
    ),
    "qec12": (
        "rep5_correct",
        """
from qiskit import QuantumCircuit
def rep5_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(9, 1)
    if logical_bit == 1:
        qc.x(0)
    if error_qubit is not None:
        qc.x(error_qubit)
        qc.x(error_qubit)
    qc.measure(0, 0)
    return qc
""",
    ),
}


@pytest.mark.parametrize("task_id", sorted(_TRIVIAL_SHORTCUTS))
def test_trivial_shortcut_circuits_are_rejected(task_id: str) -> None:
    """No QEC contract may be passable by a circuit that skips the code entirely.

    These are the trivial 1-to-3-gate and hardcoded-lookup shortcut families the
    core-suite rigor audit found passable; every one must fail closed here.
    """

    entry_point, code = _TRIVIAL_SHORTCUTS[task_id]
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id=task_id,
        code=code,
        entry_point=entry_point,
    )

    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_qec_contracts_enable_ir_anti_shortcut_policies() -> None:
    registry = ContractRegistry.from_package("qec")
    for contract in registry:
        semantic = next(
            requirement
            for requirement in contract.requirements
            if requirement.requirement_id == "semantic_requirements"
        )
        value = dict(semantic.value.items)
        assert value["forbid_state_preparation"] is True
        assert "unitary" in value["forbidden_gate_families"].items
        assert "dense_unitary" in value["forbidden_gate_families"].items
        if "required_interactions" in value:
            if contract.task_id == "qec11":
                # The H-conjugated extraction (H on the data support, CX
                # data->ancilla, H back) is equally correct, so qec11 keeps only
                # the undirected interaction pairs.
                assert "required_controlled_x_interactions" not in value
            else:
                assert "required_controlled_x_interactions" in value or "required_parity_interactions" in value
            if contract.task_id == "qec01":
                assert "required_any_interaction_sequences" in value
            else:
                assert value["reject_canceling_interaction_padding"] is True
        if "argument_conditioned_gate" in value:
            gate = dict(value["argument_conditioned_gate"].items)
            assert gate["verification"] == "program_ir_concrete_case"
        if contract.task_id in {"qec03", "qec04", "qec09", "qec12"}:
            assert "required_controlled_correction" in value
        if contract.task_id == "qec06":
            assert "required_connected_interaction_groups" in value


def test_qec_ancilla_headroom_admits_textbook_bit_flip_correction() -> None:
    code = """
from qiskit import QuantumCircuit

def bit_flip_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(5, 1)
    if logical_bit:
        qc.x(0)
    qc.cx(0, 1); qc.cx(0, 2)
    if error_qubit is not None:
        qc.x(error_qubit)
    qc.cx(0, 3); qc.cx(1, 3)
    qc.cx(1, 4); qc.cx(2, 4)
    qc.x(4); qc.ccx(3, 4, 0); qc.x(4)
    qc.ccx(3, 4, 1)
    qc.x(3); qc.ccx(3, 4, 2); qc.x(3)
    qc.cx(0, 1); qc.cx(0, 2); qc.ccx(2, 1, 0)
    qc.measure(0, 0)
    return qc
"""
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec03",
        code=code,
        entry_point="bit_flip_correct",
    )

    assert details["passed"] is True, details


@pytest.mark.parametrize(
    "injection",
    [
        """    for wire in range(3):
        if error_qubit == wire:
            qc.x(wire)
""",
        """    match error_qubit:
        case 0:
            qc.x(0)
        case 1:
            qc.x(1)
        case 2:
            qc.x(2)
""",
    ],
)
def test_qec_ir_error_check_accepts_valid_python_control_flow(injection: str) -> None:
    task = load_tasks("qiskit", "qec")["qec03"]
    code = task["canonical_solution"].replace(
        "    if error_qubit is not None:\n        qc.x(error_qubit)\n",
        injection,
    )
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec03",
        code=code,
        entry_point=task["entry_point"],
    )

    assert details["passed"] is True, details


@pytest.mark.parametrize(
    "dead_code",
    [
        """    def never_called(_logical_bit, error_qubit):
        if error_qubit is not None:
            qc.x(error_qubit)
""",
        """    if error_qubit is not None:
        qc.z(error_qubit)
        if error_qubit > 99:
            qc.x(error_qubit)
""",
    ],
)
def test_qec_ir_error_check_rejects_dead_code_and_wrong_pauli(dead_code: str) -> None:
    code = f"""
from qiskit import QuantumCircuit

def bit_flip_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(3, 1)
    if logical_bit:
        qc.x(0)
    qc.cx(0, 1); qc.cx(0, 2)
{dead_code}    qc.measure(0, 0)
    return qc
"""
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec03",
        code=code,
        entry_point="bit_flip_correct",
    )

    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_qec_correction_cannot_be_replaced_by_direct_error_cancellation() -> None:
    code = """
from qiskit import QuantumCircuit

def bit_flip_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(3, 1)
    if logical_bit:
        qc.x(0)
    qc.cx(0, 1); qc.cx(0, 2)
    if error_qubit is not None:
        qc.x(error_qubit)
        qc.rz(0.0, 0)
        qc.x(error_qubit)
    qc.measure(0, 0)
    return qc
"""
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec03",
        code=code,
        entry_point="bit_flip_correct",
    )

    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_qec_canceling_interaction_padding_is_rejected() -> None:
    code = """
from qiskit import QuantumCircuit

def bit_flip_encode_decode(logical_bit):
    qc = QuantumCircuit(3, 1)
    if logical_bit:
        qc.x(0)
    qc.cx(0, 1); qc.cx(0, 1)
    qc.cx(0, 2); qc.cx(0, 2)
    qc.measure(0, 0)
    return qc
"""
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec01",
        code=code,
        entry_point="bit_flip_encode_decode",
    )

    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_qec_encode_decode_accepts_the_other_valid_interaction_order() -> None:
    code = """
from qiskit import QuantumCircuit

def bit_flip_encode_decode(logical_bit):
    qc = QuantumCircuit(3, 1)
    if logical_bit:
        qc.x(0)
    qc.cx(0, 2); qc.cx(0, 1)
    qc.cx(0, 1); qc.cx(0, 2)
    qc.measure(0, 0)
    return qc
"""
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec01",
        code=code,
        entry_point="bit_flip_encode_decode",
    )

    assert details["passed"] is True, details


def test_qec03_accepts_reverse_order_uncomputation_decoder() -> None:
    template = """
from qiskit import QuantumCircuit

def bit_flip_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(3, 1)
    if logical_bit == 1:
        qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    if error_qubit is not None:
        qc.x(error_qubit)
    {decoder}
    qc.ccx(2, 1, 0)
    qc.measure(0, 0)
    return qc
"""
    reverse_order = template.format(decoder="qc.cx(0, 2); qc.cx(0, 1)")
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec03",
        code=reverse_order,
        entry_point="bit_flip_correct",
    )
    assert details["passed"] is True, details

    same_order = template.format(decoder="qc.cx(0, 1); qc.cx(0, 2)")
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec03",
        code=same_order,
        entry_point="bit_flip_correct",
    )
    assert details["passed"] is True, details


def test_qec04_accepts_palindromic_hadamard_reverse_decoder() -> None:
    code = """
from qiskit import QuantumCircuit

def phase_flip_correct(logical_bit, error_qubit):
    qc = QuantumCircuit(3, 1)
    if logical_bit == 1:
        qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(0); qc.h(1); qc.h(2)
    if error_qubit is not None:
        qc.z(error_qubit)
    qc.h(2); qc.h(1); qc.h(0)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.ccx(2, 1, 0)
    qc.measure(0, 0)
    return qc
"""
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec04",
        code=code,
        entry_point="phase_flip_correct",
    )
    assert details["passed"] is True, details


def test_qec_state_injection_is_rejected() -> None:
    code = """
import numpy as np
from qiskit import QuantumCircuit

def shor_encode_zero():
    state = np.zeros(512)
    for first in (0, 7):
        for second in (0, 7):
            for third in (0, 7):
                state[first | (second << 3) | (third << 6)] = 1 / np.sqrt(8)
    qc = QuantumCircuit(9, 9)
    qc.initialize(state, range(9))
    qc.measure(range(9), range(9))
    return qc
"""
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec06",
        code=code,
        entry_point="shor_encode_zero",
    )

    assert details["passed"] is False
    assert details["semantic_status"] == "semantic_fail"


def test_qec_forbidden_call_policy_allows_innocent_local_names() -> None:
    task = load_tasks("qiskit", "qec")["qec02"]
    code = task["canonical_solution"].replace(
        "    qc = QuantumCircuit(5, 2)\n",
        '    sample = 2\n    decoder = "majority vote"\n    optimizer = None\n    qc = QuantumCircuit(5, 2)\n',
    )
    _, details = build_evaluator("qiskit", "qec").grade_code(
        task_id="qec02",
        code=code,
        entry_point=task["entry_point"],
    )

    assert details["passed"] is True, details


def _contract_dict(contract: Any) -> dict[str, Any]:
    from qceval.semantics.contracts import contract_to_dict

    return contract_to_dict(contract)
