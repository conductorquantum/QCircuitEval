"""Regression assertions for the 2026-07 prompt/contract asset audit fixes."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import pytest

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")


def _rows(suite: str, framework: str) -> dict[str, dict]:
    text = resources.files(f"qceval.assets.{suite}").joinpath(f"{framework}.jsonl").read_text(encoding="utf-8")
    return {row["task_id"]: row for row in map(json.loads, text.splitlines())}


def _contracts(suite: str) -> dict[str, dict]:
    text = resources.files("qceval.assets.contracts").joinpath(f"{suite}.jsonl").read_text(encoding="utf-8")
    return {row["task_id"]: row for row in map(json.loads, text.splitlines())}


def _requirement(contract: dict, kind: str) -> dict:
    return next(req["value"] for req in contract["requirements"] if req["kind"] == kind)


def _prompts(suite: str, task_id: str) -> dict[str, str]:
    return {framework: _rows(suite, framework)[task_id]["prompt"] for framework in FRAMEWORKS}


def _task_requirements(prompt: str) -> str:
    return prompt.split("Task requirements:\n", 1)[1].split("\n\nSubmission contract:", 1)[0]


def test_core_task_requirements_share_framework_neutral_graded_output_wording() -> None:
    leftovers = (
        "Return the QuantumCircuit",
        "Return the cirq.Circuit",
        "Return the QNode result",
        "Return an unmeasured kernel",
        "Return an unmeasured QuantumCircuit",
        "Return qml.probs",
        "Output both qubits.",
        "Output all",
        "Output only",
    )
    for task_id in _rows("core", "qiskit"):
        requirements = {
            framework: _task_requirements(prompt) for framework, prompt in _prompts("core", task_id).items()
        }
        endings = []
        for framework, text in requirements.items():
            if "The graded object is the unmeasured program." in text:
                endings.append("The graded object is the unmeasured program.")
            else:
                index = text.rfind("The graded output is ")
                assert index >= 0, (task_id, framework)
                endings.append(text[index:])
            for leftover in leftovers:
                assert leftover not in text, (task_id, framework, leftover)
        assert len(set(endings)) == 1, (task_id, endings)


def test_all_280_prompt_hashes_match_their_contracts() -> None:
    count = 0
    for suite in ("core", "qec"):
        contracts = _contracts(suite)
        for framework in FRAMEWORKS:
            for task_id, row in _rows(suite, framework).items():
                expected = _requirement(contracts[task_id], "prompt_hashes")[framework]
                assert hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() == expected, (
                    suite,
                    framework,
                    task_id,
                )
                count += 1
    assert count == 280


def test_task40_prompts_contain_the_ordered_ansatz() -> None:
    for framework, prompt in _prompts("core", "40").items():
        assert "Apply X to qubit 0, preparing |01> in q1q0 order." in prompt, framework
        assert "convention below" not in prompt, framework
        # The ordered ansatz is framework-neutral: CUDA-Q used to restate it in
        # its own longer wording, which is drift rather than an interface need.
        assert "then CX control q0 target q1" in prompt, framework


def test_task03_prompts_pin_variable_mapping_and_ancilla_restoration() -> None:
    for framework, prompt in _prompts("core", "03").items():
        requirements = _task_requirements(prompt)
        assert requirements.startswith("Implement Grover search for this 3-CNF formula"), framework
        assert "Map x1, x2, and x3 to qubits q0, q1, and q2, respectively." in prompt, framework
        assert "Restore every work qubit to |0> before returning." in prompt, framework
        assert "for example a phase-kickback ancilla" not in prompt, framework
        assert "oracle" not in requirements.lower(), framework
        assert "diffuser" not in requirements.lower(), framework


def test_no_prompt_contains_a_construction_requirements_section() -> None:
    for suite in ("core", "qec"):
        for framework in FRAMEWORKS:
            for task_id, row in _rows(suite, framework).items():
                assert "Construction requirements:" not in row["prompt"], (suite, framework, task_id)


# Core tasks whose graded object is a named algorithm that at least one of the
# four frameworks ships a prebuilt implementation of. Everything else grades a
# state, an arithmetic relation, a decomposition, or an exact circuit, where
# there is no library shortcut to take.
#
# Do not "complete" this set by adding the remaining named algorithms. Simon
# (13, 14, 46), Deutsch-Jozsa (11, 12), Bernstein-Vazirani (20), and the
# phase-kickback period finder (47) are absent on purpose: Qiskit, Cirq,
# PennyLane, and CUDA-Q ship no implementation of any of them, so a ban would
# forbid nothing while advertising a shortcut that does not exist. The rule is
# "ban where a library shortcut exists", not "ban every famous algorithm".
# If a framework later ships one of these, add the task here with the symbols
# it introduces. See the anti-shortcut policy table in
# docs/prompt-simplicity-review.md.
PREBUILT_BAN_FAMILIES = {
    "grover search / amplitude amplification": ("01", "03", "05", "45"),
    "qaoa": ("04", "48", "55"),
    "quantum fourier transform": ("08",),
    "phase estimation": ("09", "25", "51", "56"),
    "order finding": ("31", "32", "52"),
    "linear systems": ("10", "53"),
    "hadamard test": ("15", "58"),
    "time evolution / trotter": ("49", "50", "54", "57"),
}
PREBUILT_BAN_TASKS = frozenset(task_id for family in PREBUILT_BAN_FAMILIES.values() for task_id in family)


def test_core_prebuilt_algorithm_policy_follows_the_algorithm_rule_not_task_ids() -> None:
    """Near-duplicate tasks must be graded under the same rule.

    The ban used to follow the 51-58 task-id range, so order finding was banned
    from ``PhaseEstimation`` on 52 and allowed it on 31/32; the same split hit
    HHL (10 vs 53), the Hadamard test (15 vs 58), phase estimation of a Grover
    iterate (25 vs 51), and QAOA (04/48 vs 55).
    """
    policy = (
        "Build the task's graded algorithm or required decomposition from gates. Do not use a "
        "library routine or constructor that directly supplies it. Constructors for subordinate "
        "components are allowed unless the task requires you to decompose that component."
    )
    for framework in FRAMEWORKS:
        rows = _rows("core", framework)
        banned = {task_id for task_id, row in rows.items() if row["submission"].get("prebuilt_algorithm_ban")}
        assert banned == set(PREBUILT_BAN_TASKS), (framework, sorted(banned ^ set(PREBUILT_BAN_TASKS)))
        for task_id, row in rows.items():
            expected = task_id in PREBUILT_BAN_TASKS
            assert (policy in row["prompt"]) is expected, (framework, task_id)
            declared = bool(row["canonical_class"].get("forbidden_imports"))
            assert declared is expected, (framework, task_id)
            assert "external algorithm library" not in row["prompt"], (framework, task_id)


def test_core_08_qft_prebuilt_shortcuts_fail_structural_check() -> None:
    """Core 08 must reject prebuilt QFT constructors even through aliases or newer names."""
    from qceval.evals.evaluator import build_evaluator

    evaluator = build_evaluator("qiskit", suite="core")
    bypasses = {
        "module_alias": """\
from qiskit import QuantumCircuit
import qiskit.circuit.library as library

def qft_6():
    qc = QuantumCircuit(6)
    qc.append(library.QFT(num_qubits=6), range(6))
    return qc
""",
        "renamed_import": """\
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT as myqft

def qft_6():
    qc = QuantumCircuit(6)
    qc.append(myqft(num_qubits=6), range(6))
    return qc
""",
        "newer_constructor": """\
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate

def qft_6():
    qc = QuantumCircuit(6)
    qc.append(QFTGate(num_qubits=6), range(6))
    return qc
""",
    }
    for name, code in bypasses.items():
        _, details = evaluator.grade_code(task_id="08", code=code, entry_point="qft_6")
        assert details["passed"] is False, (name, details.get("reason"))
        assert details.get("reason") == "requirement_failed:forbidden_imports", (name, details)


def test_every_prompt_ban_is_backed_by_a_contract_restriction() -> None:
    """A prompt must not state a rule the grader does not enforce.

    Core 45-50 carried the deterministic bullet with no ``forbidden_calls`` in
    their contracts, so the identical sentence was enforced on 51-58 and
    decorative on 45-50.
    """
    contracts = _contracts("core")
    rows = _rows("core", "qiskit")
    for task_id, row in rows.items():
        constraints = _requirement(contracts[task_id], "structural_constraints")["frameworks"]["qiskit"]
        if row["submission"].get("strict"):
            assert "Do not run a sampler, optimizer, or backend." in row["prompt"], task_id
            assert "run(" in constraints.get("forbidden_calls", []), task_id
            assert "execute(" in constraints.get("forbidden_calls", []), task_id
        if row["submission"].get("prebuilt_algorithm_ban"):
            assert constraints.get("forbidden_imports"), task_id


def test_all_pennylane_core_prompts_require_one_analytic_probability_interface() -> None:
    for task_id, row in _rows("core", "pennylane").items():
        contract = row["prompt"].split("\n\nSubmission contract:\n", 1)[1]
        assert 'qml.device("default.qubit", wires=n, shots=None)' in contract, task_id
        assert "qml.sample" not in contract, task_id
        assert "shots are optional" not in contract, task_id
        assert "Prefer analytic probabilities" not in contract, task_id


def test_leftover_how_to_recipes_are_absent() -> None:
    cudaq_06 = _rows("core", "cudaq")["06"]["prompt"]
    assert "ry/rz rotations computed from the amplitudes" not in cudaq_06
    assert "using q0 as the ancilla" in cudaq_06
    for task_id in ("qec08", "qec09"):
        assert "InsertStrategy" not in _rows("qec", "cirq")[task_id]["prompt"]
        assert "cirq.measure(" not in _rows("qec", "cirq")[task_id]["prompt"]
    assert "Measure ancillas 7, 8, and 9 into classical bits 0, 1, and 2." in _rows("qec", "cirq")["qec08"]["prompt"]
    assert "Measure data qubits 0 through 6 into classical bits 0 through 6." in _rows("qec", "cirq")["qec09"]["prompt"]
    for framework, prompt in _prompts("qec", "qec10").items():
        assert "as defined in the Shor code convention" in prompt, framework
        assert "convention below" not in prompt, framework
    for framework in FRAMEWORKS:
        for task_id in ("01", "02", "19"):
            prompt = _rows("core", framework)[task_id]["prompt"]
            assert " using Qiskit." not in prompt, (framework, task_id)
            assert " using Cirq." not in prompt, (framework, task_id)
            assert " using PennyLane." not in prompt, (framework, task_id)


def test_task07_prompts_pin_registers_without_a_gate_recipe() -> None:
    for framework, prompt in _prompts("core", "07").items():
        assert "Prepare q0, q1, q2 in |0>, |1>, |1> respectively" in prompt, framework
        assert "q3-q5 in |000>" in prompt, framework
        assert "Use q6-q8 as the SWAP-test ancillas" in prompt, framework
        assert "ancilla q(6+i) compares q_i with q_(3+i)" in prompt, framework


def test_task26_prompts_pin_registers_without_a_gate_recipe() -> None:
    for framework, prompt in _prompts("core", "26").items():
        assert "GHZ state on q3-q5 and |+++> on q6-q8" in prompt, framework
        assert "ancillas q0-q2" in prompt, framework
        assert "ancilla q_i compares q_(3+i) with q_(6+i)" in prompt, framework


def test_task04_prompts_pin_qaoa_init_and_angle_convention() -> None:
    for framework, prompt in _prompts("core", "04").items():
        assert "Initialize all five problem qubits in |+>" in prompt, framework
        assert "RX(2*beta_k)" in prompt, framework
        assert "exp(-i*gamma_k*Z_i Z_j)" in prompt, framework
        assert "Construction requirements:" not in prompt, framework


def test_task05_prompts_leave_phase_matching_to_the_candidate() -> None:
    for framework, prompt in _prompts("core", "05").items():
        assert "phase-matched Grover amplitude amplification" in prompt, framework
        assert "J+1 phase-matched iterations" not in prompt, framework
        assert "arcsin" not in prompt, framework


def test_task08_prompts_define_the_qft_unitary_without_a_swap_recipe() -> None:
    for framework, prompt in _prompts("core", "08").items():
        assert "U[j,k] = exp(2*pi*i*j*k/64)/8" in prompt, framework
        assert "SWAP" not in _task_requirements(prompt), framework


def test_task37_prompts_pin_control_and_target_qubits() -> None:
    for framework, prompt in _prompts("core", "37").items():
        assert "Qubit 0 is the control and qubit 1 is the target." in prompt, framework


def test_task41_prompts_pin_the_product_ansatz() -> None:
    for framework, prompt in _prompts("core", "41").items():
        assert "Apply RZ(param[0]), RY(param[1]), RZ(param[2]) on q0" in prompt, framework
        assert "RZ(param[3]), RY(param[4]), RZ(param[5]) on q1" in prompt, framework


def test_task42_prompts_pin_the_u_gate_convention() -> None:
    for framework, prompt in _prompts("core", "42").items():
        assert "U(theta, phi, lam) = RZ(phi) . RY(theta) . RZ(lam)" in prompt, framework
    assert "cirq.X**0.5" in _rows("core", "cirq")["42"]["prompt"]
    assert "Use RX(pi/2)" in _rows("core", "cudaq")["42"]["prompt"]


def test_task11_and_task12_prompts_pin_endianness_without_oracle_recipes() -> None:
    for framework, prompt in _prompts("core", "11").items():
        assert "s=1100, written in q3q2q1q0 order" in prompt, framework
        assert "q0 the least-significant bit" in prompt, framework
        assert "f(x) = (s dot x) mod 2" in prompt, framework
        assert "q4 as the oracle target" in prompt, framework
        assert "phase-kickback" not in _task_requirements(prompt), framework
        assert "controls act on qubits 2 and 3" not in prompt, framework
    for framework, prompt in _prompts("core", "12").items():
        assert "constant-zero oracle on four query qubits q0-q3" in prompt, framework
        assert "qubit 4" not in _task_requirements(prompt), framework
        assert _rows("core", framework)["12"]["canonical_class"]["metadata_checks"]["min_num_qubits"] == 4


def test_hidden_string_endianness_is_pinned_for_tasks_14_and_20() -> None:
    for framework, prompt in _prompts("core", "13").items():
        assert "\n -" not in _task_requirements(prompt), framework
        assert "Use q0-q1 as the query register and higher-indexed qubits as the oracle output register." in prompt, (
            framework
        )
    for framework, prompt in _prompts("core", "14").items():
        assert "s2=1, s1=1, s0=0" in prompt, framework
        assert "Use q0-q2 as the query register and higher-indexed qubits as the oracle output register." in prompt, (
            framework
        )
        assert "\n -" not in _task_requirements(prompt), framework
    for framework, prompt in _prompts("core", "20").items():
        assert "a2 = 0, a1 = 1, a0 = 1" in prompt, framework


def test_task25_prompts_pin_plus_g_convention_and_legacy_peaks_match() -> None:
    for framework in FRAMEWORKS:
        row = _rows("core", framework)["25"]
        assert "G = (2|s><s| - I)(I - 2|00><00|)" in row["prompt"], framework
        assert "eigenphases are 1/6 and 5/6" in row["prompt"], framework
        assert "for example with a Z on the control" not in row["prompt"], framework
        assert "compensate the sign" not in row["prompt"], framework
        assert row["canonical_class"]["accepted_peak_sets"] == [["001", "111"], ["010", "110"]], framework


def test_task18_prompts_pin_the_channel_without_the_gate_recipe() -> None:
    for framework, prompt in _prompts("core", "18").items():
        assert "from Alice's qubit 0 to Bob's qubit 2" in prompt, framework
        assert "rho -> RX(pi/2) rho RX(pi/2)^dagger" in prompt, framework
        assert "apply RX(pi/2), then transfer" not in prompt, framework
        assert "standard three-qubit teleportation construction" not in prompt, framework


def test_task29_prompts_pin_chsh_observables_not_the_ry_recipe() -> None:
    for framework, prompt in _prompts("core", "29").items():
        assert "|Phi+> = (|00> + |11>)/sqrt(2) with Alice on qubit 0 and Bob on qubit 1" in prompt, framework
        assert "Alice measures Z when alice = 0 and -X when alice = 1" in prompt, framework
        assert "(Z - X)/sqrt(2)" in prompt and "(Z + X)/sqrt(2)" in prompt, framework
        assert "RY(-pi/2)" not in prompt and "RY(-pi/4)" not in prompt and "RY(+pi/4)" not in prompt, framework


def test_task32_and_task39_use_shared_graded_output_wording() -> None:
    expectations = {
        "32": "The graded output is the phase register.",
        "39": "The graded object is the unmeasured program.",
    }
    for task_id, sentence in expectations.items():
        for framework in FRAMEWORKS:
            prompt = _rows("core", framework)[task_id]["prompt"]
            requirements = prompt.split("Task requirements:\n", 1)[1].split("\n\nSubmission contract:", 1)[0]
            assert sentence in requirements, (task_id, framework)
            assert "Return the QuantumCircuit" not in requirements, (task_id, framework)
            assert "Return the cirq.Circuit" not in requirements, (task_id, framework)
            assert "Return the QNode result" not in requirements, (task_id, framework)
            assert "Return an unmeasured kernel" not in requirements, (task_id, framework)


def test_task46_and_task47_use_shared_graded_output_wording() -> None:
    expectations = {
        "46": "The graded output is the three query qubits q0, q1, and q2.",
        "47": "The graded output is input qubits q0 and q1.",
    }
    for task_id, sentence in expectations.items():
        for framework in FRAMEWORKS:
            prompt = _rows("core", framework)[task_id]["prompt"]
            requirements = prompt.split("Task requirements:\n", 1)[1].split("\n\nSubmission contract:", 1)[0]
            assert sentence in requirements, (task_id, framework)
            assert "Return an unmeasured kernel; the grader reads the marginal" not in requirements, (
                task_id,
                framework,
            )
            assert 'key="result"' not in requirements, (task_id, framework)
            assert "qml.probs(wires=" not in requirements, (task_id, framework)


def test_task56_contract_no_longer_requires_the_clock2_q4_interaction() -> None:
    frameworks = _requirement(_contracts("core")["56"], "structural_constraints")["frameworks"]
    for framework, block in frameworks.items():
        pairs = block["required_interactions"]
        assert [2, 4] not in pairs, framework
        assert [2, 3] in pairs and [0, 4] in pairs and [1, 4] in pairs, framework


def test_task57_prompts_use_the_consistent_splitting_qualifier() -> None:
    for framework, prompt in _prompts("core", "57").items():
        assert "Any consistent symmetric second-order splitting of H is accepted" in prompt, framework
        assert "same splitting used in every Trotter step" in prompt, framework
        assert "-gammaX0" not in prompt, framework
        assert "-(gamma/2)*X0X1" not in prompt, framework


def test_task38_prompts_use_the_compact_exact_circuit_format() -> None:
    for framework, prompt in _prompts("core", "38").items():
        requirements = _task_requirements(prompt)
        assert requirements.startswith("Build this two-layer unmeasured circuit on q0-q2."), framework
        assert "CNOT(q0, q1), CNOT(q1, q2), and CNOT(q2, q0), in that order" in requirements, framework
        assert "theta[d,i,k] = (9d + 3i + k + 1)*pi/19" in requirements, framework
        assert "##" not in requirements, framework
        assert "\\[" not in requirements, framework
        assert "\n" not in requirements, framework


def test_iterative_phase_estimation_prompts_allow_deferred_measurement_equivalents() -> None:
    for task_id in ("33", "34"):
        for framework, prompt in _prompts("core", task_id).items():
            requirements = _task_requirements(prompt)
            assert "equivalent coherent circuit" in requirements, (task_id, framework)
            assert "deferring the intermediate measurements and feed-forward" in requirements, (task_id, framework)
            assert "one-bit-per-round" not in requirements, (task_id, framework)


def test_task58_prompts_pin_the_control_on_one_convention() -> None:
    for framework, prompt in _prompts("core", "58").items():
        assert "control U on the ancilla state |1>" in prompt, framework
        assert "rightmost factor acting first" in prompt, framework


def test_task24_cirq_and_pennylane_structural_blocks_have_the_state_task_flags() -> None:
    frameworks = _requirement(_contracts("core")["24"], "structural_constraints")["frameworks"]
    for framework in ("cirq", "pennylane"):
        block = frameworks[framework]
        assert block["forbid_returned_counts"] is True, framework
        assert block["forbid_returned_probabilities"] is True, framework
        assert block["min_measurement_count"] == 4, framework
        assert block["min_num_qubits"] == 4, framework
        assert block["min_non_measurement_operation_count"] == 1, framework


@pytest.mark.parametrize(
    ("task_id", "framework", "expected"),
    [("32", "cirq", 12), ("34", "cudaq", 2)],
)
def test_structural_min_num_qubits_corrections(task_id: str, framework: str, expected: int) -> None:
    frameworks = _requirement(_contracts("core")[task_id], "structural_constraints")["frameworks"]
    assert frameworks[framework]["min_num_qubits"] == expected


def test_qec01_prompts_no_longer_include_the_fanout_construction_recipe() -> None:
    recipe = "Encode by fanning data qubit 0 out onto qubits 1 and 2 with CNOTs controlled by qubit 0"
    commuting = "The commuting fan-out CNOTs may appear in either order within each half."
    for framework, prompt in _prompts("qec", "qec01").items():
        assert "Construction requirements:" not in prompt, framework
        assert recipe not in prompt, framework
        assert commuting not in prompt, framework


def test_qec02_prompts_state_the_framework_native_output_order() -> None:
    for framework in ("qiskit", "cirq", "cudaq"):
        prompt = _rows("qec", framework)["qec02"]["prompt"]
        assert "the Z1Z2 outcome in ancilla q4, measured into classical bit 1." in prompt, framework
    pennylane = _rows("qec", "pennylane")["qec02"]["prompt"]
    assert "Return probabilities for wires [4, 3]" in pennylane
    assert "wire 3 is output bit 0 and wire 4 is output bit 1" in pennylane


def test_pennylane_qec_prompts_use_wire_probability_outputs_not_classical_measurements() -> None:
    expected_outputs = {
        "qec01": "Return probabilities for wire 0 only.",
        "qec02": "Return probabilities for wires [4, 3]",
        "qec03": "return probabilities for wire 0 only.",
        "qec04": "return probabilities for wire 0 only.",
        "qec05": "return probabilities for wires [3, 0]",
        "qec06": "Return probabilities for data wires [8, 7, 6, 5, 4, 3, 2, 1, 0].",
        "qec07": "Return probabilities for data wires [6, 5, 4, 3, 2, 1, 0].",
        "qec08": "Return probabilities for ancilla wires [9, 8, 7]",
        "qec09": "Return probabilities for data wires [6, 5, 4, 3, 2, 1, 0].",
        "qec10": "Return probabilities for ancilla wires [14, 13, 12, 11, 10, 9]",
        "qec11": "Return probabilities for ancilla wires [10, 9]",
        "qec12": "return probabilities for data wire 0 only.",
    }
    for task_id, row in _rows("qec", "pennylane").items():
        requirements = _task_requirements(row["prompt"])
        assert "into classical bit" not in requirements, task_id
        assert "into classical bits" not in requirements, task_id
        assert expected_outputs[task_id] in requirements, task_id


def test_qec_encoder_completion_rule_is_framework_neutral_with_a_cirq_moment_detail() -> None:
    neutral = "Complete the entire encoder before the first data-ancilla interaction."
    cirq_detail = "In Cirq, no syndrome-extraction gate may share a moment with an encoder gate."
    for task_id in ("qec08", "qec09", "qec10", "qec11"):
        for framework, prompt in _prompts("qec", task_id).items():
            requirements = _task_requirements(prompt)
            assert neutral in requirements, (task_id, framework)
            assert (cirq_detail in requirements) is (framework == "cirq"), (task_id, framework)
            assert "in moment order" not in requirements, (task_id, framework)


def test_every_enforced_encoder_state_requirement_is_disclosed_in_the_prompt() -> None:
    """No task may be failed by an undisclosed structural requirement.

    ``required_encoder_state_before_ancilla_use`` is a hard grader failure in
    ``semantics/verifiers/requirements/interactions.py``. qec10 and qec11 once
    carried it silently while only qec08 and qec09 stated it.
    """
    neutral = "Complete the entire encoder before the first data-ancilla interaction."
    enforced = {
        task_id
        for task_id, contract in _contracts("qec").items()
        if "required_encoder_state_before_ancilla_use" in _requirement(contract, "prompt_semantics")
    }
    assert enforced == {"qec08", "qec09", "qec10", "qec11"}
    for task_id in sorted(enforced):
        for framework, prompt in _prompts("qec", task_id).items():
            assert neutral in _task_requirements(prompt), (task_id, framework)


def test_qec03_qec04_qec05_prompts_state_the_encoding_invariance() -> None:
    for task_id in ("qec03", "qec04"):
        for framework, prompt in _prompts("qec", task_id).items():
            assert "Encoding convention:" in prompt, (task_id, framework)
            assert (
                "Apply X to data qubit 0 if and only if logical_bit is 1; apart from this initial X, the circuit "
                "must not depend on logical_bit." in prompt
            ), (task_id, framework)
    for framework, prompt in _prompts("qec", "qec05").items():
        assert "Apply X to data qubit 0 if and only if control_bit is 1" in prompt, framework
        assert "apply X to data qubit 3 if and only if target_bit is 1" in prompt, framework


def test_qec07_prompts_restore_the_logical_one_prescription() -> None:
    for framework, prompt in _prompts("qec", "qec07").items():
        assert "Logical |1_L> is obtained by applying X on all 7 data qubits after encoding |0_L>." in prompt, framework


def test_qec10_contract_dropped_the_inter_group_ordering_requirement() -> None:
    value = _requirement(_contracts("qec")["qec10"], "prompt_semantics")
    assert "required_inter_group_before_intra_group" not in value
    assert "required_encoder_state_before_ancilla_use" in value


def test_qec11_contract_uses_only_undirected_interaction_pairs() -> None:
    value = _requirement(_contracts("qec")["qec11"], "prompt_semantics")
    assert "required_controlled_x_interactions" not in value
    pairs = {tuple(sorted(pair)) for pair in value["required_interactions"]}
    expected = {(data, 9) for data in range(6)} | {(data, 10) for data in range(3, 9)}
    assert pairs == expected


def test_task09_prompts_pin_three_counting_qubits() -> None:
    for framework, prompt in _prompts("core", "09").items():
        assert "counting qubits q0-q2" in prompt, framework
        assert "samples the eigenphase fractions" not in prompt, framework


def test_prompt_simplicity_review_removes_grader_leaks_and_recipes() -> None:
    forbidden_by_task = {
        "10": ("no-ancilla-measurement rule",),
        "21": ("Do not prepare a fixed input state",),
        "23": ("do not begin the circuit with X gates", "stripped before grading"),
        "35": ("checked exhaustively",),
        "45": ("Directly preparing the output state",),
        "46": ("do not hardcode measurement outcomes", "standard single-query interference pattern"),
        "47": ("do not hardcode measurement outcomes", "Readout convention:", "do not append an inverse QFT"),
        "51": ("library routines are not", "Counting qubit j controls"),
        "52": ("counting qubit j controls", "circuit ends with the inverse QFT"),
        "53": ("Then apply the eigenvalue-inversion rotation", "Uncompute the phase estimation"),
        "56": ("Clock qubit j controls", "standard inverse QFT"),
    }
    for task_id, leftovers in forbidden_by_task.items():
        for framework, prompt in _prompts("core", task_id).items():
            requirements = _task_requirements(prompt)
            for leftover in leftovers:
                assert leftover not in requirements, (task_id, framework, leftover)


def test_task49_prompts_accept_every_consistent_first_order_factor_ordering() -> None:
    for framework, prompt in _prompts("core", "49").items():
        assert "Any fixed ordering of the three exponential factors is accepted" in prompt, framework
        assert "apply the ZZ factor first" not in prompt, framework


def test_task47_prompts_do_not_pin_a_terminal_transform() -> None:
    for framework, prompt in _prompts("core", "47").items():
        requirements = _task_requirements(prompt)
        assert "via phase kickback" in requirements, framework
        assert "Hadamard" not in requirements, framework
        assert "inverse QFT" not in requirements, framework


def test_task31_prompts_pin_the_work_register() -> None:
    for framework, prompt in _prompts("core", "31").items():
        assert "Work register: q4-q7" in prompt, framework
        assert "q4 the least-significant work bit, initialized to integer 1" in prompt, framework


def test_cudaq_prompts_do_not_prescribe_allocation_style() -> None:
    cudaq_10 = _rows("core", "cudaq")["10"]["prompt"]
    assert "Use q0 as the success ancilla, q1-q3 as the clock register" in cudaq_10
    assert "single six-qubit qvector" not in cudaq_10
    cudaq_33 = _rows("core", "cudaq")["33"]["prompt"]
    assert "single cudaq.qvector" not in cudaq_33
    assert "do not allocate separate cudaq.qubit()" not in cudaq_33


def test_qec12_prompts_restore_the_stabilizer_mapping_and_exact_ancilla_count() -> None:
    for framework, prompt in _prompts("qec", "qec12").items():
        assert "Stabilizer-to-ancilla mapping:" in prompt, framework
        assert "Ancilla qubits 5, 6, 7, and 8 extract Z0Z1, Z1Z2, Z2Z3, and Z3Z4, respectively." in prompt, framework
        assert "using syndrome ancillas q5-q8" in prompt, framework
        assert "Up to two additional clean work qubits q9-q10 may be used" in prompt, framework
        assert "11 qubits maximum" in prompt, framework
        assert "9 qubits total" not in prompt, framework
        assert "at most four ancilla qubits" not in prompt, framework
        assert "four encoding CNOTs" not in prompt, framework


def test_qec_requirement_blocks_have_clean_openings_and_no_duplicate_mappings() -> None:
    for framework in FRAMEWORKS:
        for task_id, row in _rows("qec", framework).items():
            requirements = _task_requirements(row["prompt"])
            assert requirements == requirements.lstrip(), (task_id, framework)
        assert "Input domain:\n- logical_bit is the integer 0 or 1." in _rows("qec", framework)["qec01"]["prompt"]
        for task_id in ("qec08", "qec10"):
            assert "Stabilizer-to-ancilla mapping:" not in _rows("qec", framework)[task_id]["prompt"], (
                task_id,
                framework,
            )


def test_interface_parity_labels_match_the_actual_cross_framework_text() -> None:
    """``interface_parity`` must be derived from the text, not asserted by hand.

    The field was previously unread by any code and claimed ``shared`` for 21
    tasks whose requirement blocks differed, which is how CUDA-Q prose drifted
    away from the other three frameworks unnoticed.
    """
    for suite in ("core", "qec"):
        for task_id in _rows(suite, "qiskit"):
            requirements = {
                framework: _task_requirements(prompt) for framework, prompt in _prompts(suite, task_id).items()
            }
            expected = "shared" if len(set(requirements.values())) == 1 else "framework_specific"
            for framework in FRAMEWORKS:
                declared = _rows(suite, framework)[task_id]["interface_parity"]
                assert declared == expected, (suite, task_id, framework, declared)


def test_only_genuine_interface_differences_remain_framework_specific() -> None:
    """Core framework wording is confined to real API differences.

    Core 06 takes a different input type per framework, and 27/42/43 name gate
    spellings that do not exist in every framework. Every other Core task must
    be byte-identical across the four prompts.
    """
    divergent = {task_id for task_id, row in _rows("core", "qiskit").items() if row["interface_parity"] != "shared"}
    assert divergent == {"06", "42", "43"}


def test_sx_alias_note_covers_every_task_that_requires_sx() -> None:
    """Cirq and CUDA-Q cannot spell SX, so both SX tasks must say so.

    ``frameworks/cudaq/lowering.py::_normalize_decomposition_gates`` accepts the
    RX(pi/2) substitute for Core 42 and 43 alike; the note was previously only
    on 42, leaving 43 to fail on a framework-incidental AttributeError.
    """
    cirq_note = "cirq has no cirq.SX attribute"
    cudaq_note = "CUDA-Q has no native SX instruction."
    for task_id in ("42", "43"):
        prompts = _prompts("core", task_id)
        assert "SX" in _task_requirements(prompts["qiskit"]), task_id
        assert cirq_note in _task_requirements(prompts["cirq"]), task_id
        assert cudaq_note in _task_requirements(prompts["cudaq"]), task_id
        for framework in ("qiskit", "pennylane"):
            assert cirq_note not in _task_requirements(prompts[framework]), (task_id, framework)
            assert cudaq_note not in _task_requirements(prompts[framework]), (task_id, framework)


def test_cudaq_submission_capability_follows_the_contract_observation() -> None:
    """The measure/do-not-measure bullet must match what the grader observes.

    ``frameworks/cudaq/lowering.py::_normalize_terminal_measurements`` samples
    the contract-declared register whether or not the kernel calls ``mz``, so a
    row claiming ``measured_distribution`` without a declared register (or the
    reverse) renders an instruction the grader does not back.
    """
    contracts = _contracts("core")
    for task_id, row in _rows("core", "cudaq").items():
        observation = next(
            item["value"] for item in contracts[task_id]["requirements"] if item["id"] == "terminal_observation"
        )
        interface = observation.get("cudaq", {})
        declared = isinstance(interface, dict) and isinstance(interface.get("qubits"), list)
        measured = row["submission"]["capability"] == "measured_distribution"
        assert measured is declared, (task_id, row["submission"]["capability"], interface)


def test_cudaq_canonical_solutions_agree_with_their_own_submission_contract() -> None:
    """No reference solution may contradict the prompt it ships with."""
    for task_id, row in _rows("core", "cudaq").items():
        contract = row["prompt"].split("\n\nSubmission contract:\n", 1)[1]
        uses_mz = "mz(" in row["canonical_solution"]
        if "Return an unmeasured kernel; do not call mz" in contract:
            assert not uses_mz, task_id
        else:
            assert "or return the kernel unmeasured" in contract, task_id


def test_prompts_use_one_ascii_ket_notation() -> None:
    """Mixed Unicode/ASCII notation described the same states two ways.

    Core 16 and Core 29 both name the Bell state; before this pass one used
    ``|Phi+> = (|00> + |11>)/sqrt(2)`` and the other the Unicode spelling.
    """
    banned = "⟩⟨√−–ψΦ"
    for suite in ("core", "qec"):
        for framework in FRAMEWORKS:
            for task_id, row in _rows(suite, framework).items():
                requirements = _task_requirements(row["prompt"])
                found = sorted({character for character in banned if character in requirements})
                assert not found, (suite, framework, task_id, found)


def test_task07_states_register_contents_instead_of_a_conflicting_ket_reading() -> None:
    """Core 07 must not restate a bit order that contradicts the LSB bullet.

    The prompt used to read its kets left-to-right onto ascending indices,
    the opposite of the ``Qubit 0 is the least-significant bit`` bullet in the
    same prompt, and the reading changes which ancillas are deterministic.
    """
    for framework, prompt in _prompts("core", "07").items():
        requirements = _task_requirements(prompt)
        assert "left-to-right onto ascending qubit indices" not in requirements, framework
        assert "Prepare q0, q1, q2 in |0>, |1>, |1> respectively" in requirements, framework
        assert "q3-q5 in |000>" in requirements, framework


def test_task25_dropped_the_default_counting_register_restatement() -> None:
    for framework, prompt in _prompts("core", "25").items():
        requirements = _task_requirements(prompt)
        assert "For the default n_count=3" not in requirements, framework
        assert "Use q0 through q(n_count-1) as the counting register" in requirements, framework


def test_pennylane_core_readout_bullet_covers_tasks_with_no_output_register() -> None:
    """PennyLane grades the tape, but the prompt must still say what to return.

    Ten Core tasks state ``The graded object is the unmeasured program`` and
    name no output register, yet the contract still requires ``qml.probs``.
    """
    for task_id, row in _rows("core", "pennylane").items():
        contract = row["prompt"].split("\n\nSubmission contract:\n", 1)[1]
        assert "Wire 0 is the least-significant bit." in contract, task_id
        assert "When the task requests no output register" in contract, task_id
