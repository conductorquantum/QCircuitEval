"""Uniform prompt-scaffold consistency and mutation tests."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
from ci import check_asset_consistency as consistency


def _prompt(framework: str, signature: str, *, body: str = "    ...") -> str:
    return (
        f"{consistency.PROMPT_HEADER}\n\n"
        f"```python\n{consistency.CANONICAL_IMPORTS[framework]}\n\n"
        f"def {signature}:\n{body}"
        f"{consistency.TASK_REQUIREMENTS_MARKER}"
        "Build the requested quantum program.\n\n"
        "Submission contract:\n"
        "- Preserve the exact function signature."
    )


def _row(prompt: str, *, task_id: str = "example", entry_point: str = "answer") -> dict[str, str]:
    return {"task_id": task_id, "entry_point": entry_point, "prompt": prompt}


def _contract(*arguments: tuple[str, bool], entry_point: str = "answer") -> dict[str, object]:
    return {
        "entry_point": entry_point,
        "arguments": [{"name": name, "required": required} for name, required in arguments],
    }


def _problems(
    prompt: str,
    contract: Mapping[str, object] | None = None,
    *,
    suite: str = "core",
    framework: str = "qiskit",
    task_id: str = "example",
    entry_point: str = "answer",
) -> list[str]:
    problems, _ = consistency._prompt_scaffold_problems(
        f"{suite}/{framework}/{task_id}",
        suite,
        framework,
        _row(prompt, task_id=task_id, entry_point=entry_point),
        contract or _contract(entry_point=entry_point),
    )
    return problems


def test_all_280_asset_prompts_use_uniform_scaffolds_and_contract_signatures() -> None:
    count = 0
    for suite in consistency.SUITES:
        contract_signatures = consistency._contract_signatures(suite)
        prompt_signatures: dict[str, dict[str, consistency.PromptSignature]] = {
            task_id: {} for task_id in contract_signatures
        }
        for framework in consistency.FRAMEWORKS:
            rows = consistency._task_rows(suite, framework)
            assert set(rows) == set(contract_signatures)
            for task_id, row in rows.items():
                where = f"{suite}/{framework}/{task_id}"
                problems, signature = consistency._prompt_scaffold_problems(
                    where,
                    suite,
                    framework,
                    row,
                    contract_signatures[task_id],
                )
                assert problems == []
                assert signature is not None
                prompt_signatures[task_id][framework] = signature
                scaffold = consistency._inspect_prompt_scaffold(row["prompt"], framework)
                assert "Submission contract:" in scaffold.requirements
                assert (
                    consistency._submission_contract_problems(
                        where,
                        suite,
                        framework,
                        row,
                    )
                    == []
                )
                count += 1
        assert consistency._cross_framework_signature_problems(suite, prompt_signatures) == []
    assert count == 280


@pytest.mark.parametrize(("suite", "expected"), [("core", "1.5.0"), ("qec", "1.7.0")])
def test_prompt_contract_versions_mark_the_new_protocol_generation(suite: str, expected: str) -> None:
    versions = {
        json.loads(line)["contract_version"]
        for line in consistency.contract_resource(suite).read_text(encoding="utf-8").splitlines()
        if line
    }

    assert versions == {expected}


@pytest.mark.parametrize(
    ("framework", "replacement"),
    [
        ("qiskit", "import qiskit"),
        ("cirq", "from cirq import Circuit"),
        ("pennylane", "import pennylane"),
        ("cudaq", "from cudaq import kernel"),
    ],
)
def test_wrong_or_noncanonical_import_is_rejected(framework: str, replacement: str) -> None:
    prompt = _prompt(framework, "answer()")
    prompt = prompt.replace(consistency.CANONICAL_IMPORTS[framework], replacement, 1)

    assert "exact canonical four-line format" in "\n".join(_problems(prompt, framework=framework))


def test_missing_import_is_rejected() -> None:
    prompt = _prompt("qiskit", "answer()").replace(
        f"{consistency.CANONICAL_IMPORTS['qiskit']}\n\n",
        "",
        1,
    )

    assert "exactly one import and one function" in "\n".join(_problems(prompt))


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("other(theta)", "scaffold declares 'other'"),
        ("answer(phi)", "arguments/default presence"),
        ("answer(theta=0)", "arguments/default presence"),
    ],
)
def test_entry_point_argument_name_and_default_presence_mutants_are_rejected(
    signature: str,
    expected: str,
) -> None:
    problems = _problems(
        _prompt("qiskit", signature),
        _contract(("theta", True)),
    )

    assert expected in "\n".join(problems)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            _prompt("qiskit", "answer()").replace(
                "def answer():",
                "@staticmethod\ndef answer():",
            ),
            "undecorated",
        ),
        (
            _prompt("qiskit", "answer()", body="    value = 1\n    ..."),
            "body must contain only an ellipsis",
        ),
        (
            _prompt("qiskit", "answer()").replace(
                "\ndef answer():",
                "\nVALUE = 1\n\ndef answer():",
            ),
            "exactly one import and one function",
        ),
    ],
)
def test_decorator_and_extra_code_mutants_are_rejected(prompt: str, expected: str) -> None:
    assert expected in "\n".join(_problems(prompt))


def test_prose_only_cudaq_prompt_is_rejected() -> None:
    prompt = "Write CUDA-Q code defining answer().\nReturn an unmeasured kernel."

    assert "fixed scaffold header" in "\n".join(_problems(prompt, framework="cudaq"))


def test_cross_framework_signature_drift_is_rejected() -> None:
    shared = consistency.PromptSignature("answer", (("theta", None),))
    drifted = consistency.PromptSignature("answer", (("phi", None),))
    signatures = {
        "01": {
            "qiskit": shared,
            "cirq": shared,
            "pennylane": shared,
            "cudaq": drifted,
        }
    }

    problems = consistency._cross_framework_signature_problems("core", signatures)

    assert len(problems) == 1
    assert "cross-framework scaffold signatures differ" in problems[0]


def test_submission_contract_wording_drift_is_rejected() -> None:
    row = consistency._task_rows("core", "qiskit")["51"]
    mutated = dict(row)
    mutated["prompt"] = row["prompt"].replace(
        "Return a deterministic quantum program.",
        "Return the deterministic quantum program.",
        1,
    )

    problems = consistency._submission_contract_problems(
        "core/qiskit/51",
        "core",
        "qiskit",
        mutated,
    )

    assert problems == ["core/qiskit/51: submission contract does not match the centralized renderer"]


def test_core_task_25_preserves_literal_n_count_default_of_three() -> None:
    contract = consistency._contract_signatures("core")["25"]
    for framework in consistency.FRAMEWORKS:
        row = consistency._task_rows("core", framework)["25"]
        scaffold = consistency._inspect_prompt_scaffold(row["prompt"], framework)
        assert scaffold.signature.arguments == (("n_count", "3"),)

        mutated = dict(row)
        mutated["prompt"] = row["prompt"].replace("n_count=3", "n_count=4", 1)
        problems, _ = consistency._prompt_scaffold_problems(
            f"core/{framework}/25",
            "core",
            framework,
            mutated,
            contract,
        )
        assert any("core task 25 must declare n_count=3" in problem for problem in problems)
