"""Verify task-file and contract agreement for grading-relevant fields.

The executor collects probabilities for the qubits named by each task's
``canonical_class`` (``output_qubits``, falling back to the structural
``required_measurement_qubits``), while the contract declares the same
register in its ``framework_interface`` terminal-observation requirement.
Nothing else forces the two copies to agree, so this check fails CI when a
future edit changes one file and not the other.

Register order is validated as strongly as each framework's grading path
consumes it:

* Every framework: the task-file register and the contract register must
  contain the same qubits. This is deliberately a set comparison because the
  semantic materializer derives public bit order from the contract and the
  physical measurement mapping. Framework lowering still preserves an
  explicitly requested measurement order for faithful diagnostics.
* Every interface that declares a ``render_order``: it must be a permutation
  of the declared ``qubits`` and must list them in descending index order,
  the QCircuitEval ``q[n-1]...q[0]`` rendering convention.
* Every interface that declares ``wires`` (PennyLane): the declared sequence
  must already be in the same descending order, because that is the order
  the executor actually renders probabilities in.

It also rejects reintroduction of the removed duplicate ``complete_prompt``
field unless it exactly matches ``prompt``.

Every prompt must also use the same minimal scaffold shape: a canonical
framework import, one undecorated top-level function with the semantic
contract's signature, and an ellipsis-only body. Keeping imports and callable
signatures in the prompt removes framework-specific incidental failures while
leaving all quantum-program construction to the model.

Two further checks keep the four framework prompts from drifting apart:

* ``interface_parity`` is derived, not asserted: ``shared`` must mean the four
  task-requirement blocks are byte-identical, so a framework-specific sentence
  cannot be added to one prompt without relabeling the task.
* A CUDA-Q Core row is ``measured_distribution`` exactly when its contract
  declares a CUDA-Q measurement register, because that is the condition under
  which the lowering samples that register regardless of ``mz``. Without this,
  a row can render a measure/do-not-measure instruction the grader ignores.
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qceval.assets._resources import contract_resource, task_resource  # noqa: E402
from qceval.assets.submission_contracts import HEADING, expected_submission_contract  # noqa: E402
from qceval.evals.execution import _output_qubits  # noqa: E402

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")
SUITES = ("core", "qec")

PROMPT_HEADER = "Complete the function in the Python scaffold below. Return complete Python source only."
TASK_REQUIREMENTS_MARKER = "\n```\n\nTask requirements:\n"
CANONICAL_IMPORTS = {
    "qiskit": "from qiskit import QuantumCircuit",
    "cirq": "import cirq",
    "pennylane": "import pennylane as qml",
    "cudaq": "import cudaq",
}


@dataclass(frozen=True)
class PromptSignature:
    """Normalized prompt signature used for cross-framework parity checks."""

    entry_point: str
    arguments: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class PromptScaffold:
    """Validated scaffold details needed by consistency checks and tests."""

    signature: PromptSignature
    requirements: str


def _split_prompt(prompt: str) -> tuple[str, str]:
    prefix = f"{PROMPT_HEADER}\n\n```python\n"
    if not prompt.startswith(prefix):
        raise ValueError("prompt does not start with the fixed scaffold header and opening python fence")
    scaffold, marker, requirements = prompt[len(prefix) :].partition(TASK_REQUIREMENTS_MARKER)
    if not marker:
        raise ValueError("prompt is missing the closing fence or Task requirements marker")
    if not requirements.strip():
        raise ValueError("Task requirements must be nonempty")
    return scaffold, requirements


def _scaffold_function(scaffold: str) -> ast.FunctionDef:
    """Parse the scaffold and validate its import-plus-ellipsis-function structure."""
    try:
        module = ast.parse(scaffold)
    except SyntaxError as exc:
        raise ValueError(f"python scaffold is not valid syntax: {exc.msg}") from exc

    if len(module.body) != 2:
        raise ValueError("python scaffold must contain exactly one import and one function definition")
    import_node, function_node = module.body
    if not isinstance(import_node, ast.Import | ast.ImportFrom):
        raise ValueError("python scaffold must begin with the canonical framework import")
    if not isinstance(function_node, ast.FunctionDef):
        raise ValueError("python scaffold must contain one synchronous top-level function definition")
    if function_node.decorator_list:
        raise ValueError("scaffold function must be undecorated")
    if len(function_node.body) != 1 or not (
        isinstance(function_node.body[0], ast.Expr)
        and isinstance(function_node.body[0].value, ast.Constant)
        and function_node.body[0].value.value is Ellipsis
    ):
        raise ValueError("scaffold function body must contain only an ellipsis")
    return function_node


def _inspect_prompt_scaffold(prompt: str, framework: str) -> PromptScaffold:
    """Parse and strictly validate one framework prompt scaffold."""
    scaffold, requirements = _split_prompt(prompt)
    function_node = _scaffold_function(scaffold)

    arguments = function_node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    if arguments.posonlyargs or arguments.vararg or arguments.kwonlyargs or arguments.kwarg:
        raise ValueError("scaffold signature must use ordinary positional parameters only")
    if function_node.returns is not None or any(argument.annotation is not None for argument in positional):
        raise ValueError("scaffold signature must omit type annotations")

    default_offset = len(positional) - len(arguments.defaults)
    normalized_arguments = tuple(
        (
            argument.arg,
            None if index < default_offset else ast.unparse(arguments.defaults[index - default_offset]),
        )
        for index, argument in enumerate(positional)
    )
    canonical_arguments = ", ".join(
        name if default is None else f"{name}={default}" for name, default in normalized_arguments
    )
    expected_lines = [
        CANONICAL_IMPORTS[framework],
        "",
        f"def {function_node.name}({canonical_arguments}):",
        "    ...",
    ]
    if scaffold.splitlines() != expected_lines:
        raise ValueError("python scaffold must use the exact canonical four-line format")

    return PromptScaffold(
        signature=PromptSignature(function_node.name, normalized_arguments),
        requirements=requirements,
    )


def _prompt_scaffold_problems(
    where: str,
    suite: str,
    framework: str,
    row: Mapping[str, Any],
    contract_signature: Mapping[str, Any],
) -> tuple[list[str], PromptSignature | None]:
    """Return scaffold/contract disagreements and the normalized signature."""
    try:
        scaffold = _inspect_prompt_scaffold(str(row.get("prompt", "")), framework)
    except (KeyError, ValueError) as exc:
        return [f"{where}: {exc}"], None

    problems: list[str] = []
    signature = scaffold.signature
    contract_entry_point = str(contract_signature.get("entry_point", ""))
    row_entry_point = str(row.get("entry_point", ""))
    if row_entry_point != contract_entry_point:
        problems.append(
            f"{where}: task entry_point {row_entry_point!r} != contract entry_point {contract_entry_point!r}"
        )
    if signature.entry_point != contract_entry_point:
        problems.append(
            f"{where}: scaffold declares {signature.entry_point!r} but contract entry_point is {contract_entry_point!r}"
        )

    contract_arguments = contract_signature.get("arguments", ())
    expected_arguments = tuple(
        (str(argument.get("name", "")), not bool(argument.get("required", False)))
        for argument in contract_arguments
        if isinstance(argument, Mapping)
    )
    actual_arguments = tuple((name, default is not None) for name, default in signature.arguments)
    if actual_arguments != expected_arguments:
        problems.append(
            f"{where}: scaffold arguments/default presence {actual_arguments!r} != contract {expected_arguments!r}"
        )

    if suite == "core" and str(row.get("task_id")) == "25":
        defaults = dict(signature.arguments)
        if defaults.get("n_count") != "3":
            problems.append(f"{where}: core task 25 must declare n_count=3")
    return problems, signature


def _cross_framework_signature_problems(
    suite: str,
    signatures: Mapping[str, Mapping[str, PromptSignature]],
) -> list[str]:
    """Require every task's normalized scaffold signature to match in all frameworks."""
    problems: list[str] = []
    for task_id, task_signatures in sorted(signatures.items()):
        if set(task_signatures) != set(FRAMEWORKS):
            continue
        if len(set(task_signatures.values())) != 1:
            rendered = ", ".join(f"{framework}={task_signatures[framework]!r}" for framework in FRAMEWORKS)
            problems.append(f"{suite}/{task_id}: cross-framework scaffold signatures differ: {rendered}")
    return problems


def _submission_contract_problems(
    where: str,
    suite: str,
    framework: str,
    row: Mapping[str, Any],
) -> list[str]:
    """Require the packaged submission section to match its typed renderer."""
    prompt = str(row.get("prompt", ""))
    marker = f"\n\n{HEADING}\n"
    if prompt.count(marker) != 1:
        return [f"{where}: prompt must contain exactly one {HEADING!r} section"]
    _, tail = prompt.split(marker, 1)
    actual = HEADING + "\n" + tail.split("\n\n", 1)[0]
    try:
        expected = expected_submission_contract(suite, framework, row)
    except (KeyError, ValueError) as exc:
        return [f"{where}: {exc}"]
    if actual != expected:
        return [f"{where}: submission contract does not match the centralized renderer"]
    return []


def _task_rows(suite: str, framework: str) -> dict[str, dict[str, Any]]:
    text = task_resource(suite, framework).read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines() if line]
    return {str(record["task_id"]): record for record in records}


def _contract_interfaces(suite: str) -> dict[str, dict[str, Any]]:
    """Return each task's per-framework observation interface.

    A contract can carry several ``framework_interface`` requirements (for
    example a return-type map next to the terminal observation); only values
    that name observed qubits or wires define the observation register.
    """
    interfaces: dict[str, dict[str, Any]] = {}
    for line in contract_resource(suite).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        contract = json.loads(line)
        merged: dict[str, Any] = {}
        for item in contract["requirements"]:
            if item["kind"] != "framework_interface":
                continue
            for framework, interface in item["value"].items():
                if isinstance(interface, Mapping) and ("qubits" in interface or "wires" in interface):
                    merged[framework] = interface
        interfaces[str(contract["task_id"])] = merged
    return interfaces


def _contract_signatures(suite: str) -> dict[str, dict[str, Any]]:
    """Return each task's semantic-contract callable signature."""
    signatures: dict[str, dict[str, Any]] = {}
    for line in contract_resource(suite).read_text(encoding="utf-8").splitlines():
        if line:
            contract = json.loads(line)
            signatures[str(contract["task_id"])] = contract["signature"]
    return signatures


def _interface_register(interface: Mapping[str, Any]) -> tuple[int, ...] | None:
    qubits = interface.get("qubits", interface.get("wires"))
    if qubits is None:
        return None
    return tuple(int(qubit) for qubit in qubits)


def _render_order_problems(where: str, interface: Mapping[str, Any]) -> list[str]:
    """Validate a declared render order against the declared register.

    The render order (``render_order``, or the ``wires`` sequence itself for
    PennyLane interfaces) must name exactly the observed qubits and must list
    them in descending index order: rendered bitstrings follow the
    QCircuitEval ``q[n-1]...q[0]`` convention, so any other sequence would
    silently permute grading-time bit positions.
    """
    render_order = interface.get("render_order", interface.get("wires"))
    if render_order is None:
        return []
    register = _interface_register(interface)
    rendered = tuple(int(qubit) for qubit in render_order)
    if register is None or sorted(rendered) != sorted(register):
        return [
            f"{where}: contract render_order {list(rendered)} is not a "
            f"permutation of the observed qubits {sorted(register or ())}"
        ]
    if list(rendered) != sorted(rendered, reverse=True):
        return [
            f"{where}: contract render_order {list(rendered)} is not in descending QCircuitEval q[n-1]...q[0] order"
        ]
    return []


def _row_problems(
    where: str,
    row: Mapping[str, Any],
    interface: Mapping[str, Any] | None,
) -> list[str]:
    problems: list[str] = []
    if "complete_prompt" in row and row["complete_prompt"] != row["prompt"]:
        problems.append(f"{where}: complete_prompt differs from prompt")
    canonical = row.get("canonical_class")
    if not isinstance(canonical, Mapping):
        problems.append(f"{where}: missing canonical_class")
        return problems
    if interface is not None:
        problems.extend(_render_order_problems(where, interface))
    executor_qubits = _output_qubits(canonical)
    contract_qubits = _interface_register(interface) if interface is not None else None
    if executor_qubits is None or contract_qubits is None:
        return problems
    if set(executor_qubits) != set(contract_qubits):
        # Set comparison is sound here: no grading path consumes the
        # task-file sequence order (see the module docstring).
        problems.append(
            f"{where}: executor output qubits {sorted(executor_qubits)} "
            f"!= contract observation {sorted(contract_qubits)}"
        )
    return problems


def _cudaq_measurement_registers(suite: str) -> dict[str, bool]:
    """Return, per task, whether the contract declares a CUDA-Q measurement register.

    This mirrors ``frameworks.cudaq.lowering._contract_measurement_wires``, which
    keys off the ``qubits`` list alone: when it is present the lowering samples
    that register whether or not the kernel calls ``mz``, and when it is absent
    no terminal observation is synthesized at all.
    """
    declared: dict[str, bool] = {}
    for line in contract_resource(suite).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        contract = json.loads(line)
        requirement = next(
            (item for item in contract["requirements"] if item["id"] == "terminal_observation"),
            None,
        )
        interface = None if requirement is None else requirement["value"].get("cudaq")
        declared[str(contract["task_id"])] = isinstance(interface, Mapping) and isinstance(
            interface.get("qubits"), list
        )
    return declared


def _cudaq_capability_problems(suite: str, rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Require each CUDA-Q Core capability to match its contract observation.

    A row that claims ``measured_distribution`` renders a bullet telling the
    model to measure the requested register; any other capability renders one
    forbidding ``mz``. Deriving the flag from the contract keeps that instruction
    from contradicting what the grader actually observes.
    """
    if suite != "core":
        return []
    problems: list[str] = []
    declared = _cudaq_measurement_registers(suite)
    for task_id, row in sorted(rows.items()):
        capability = str(row.get("submission", {}).get("capability", ""))
        measured = capability == "measured_distribution"
        if measured != declared.get(task_id, False):
            problems.append(
                f"{suite}/cudaq/{task_id}: submission capability {capability!r} disagrees with the "
                f"contract, which {'declares' if declared.get(task_id) else 'declares no'} "
                "CUDA-Q measurement register"
            )
    return problems


def _interface_parity_problems(
    suite: str,
    requirements: Mapping[str, Mapping[str, str]],
    parities: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """Require ``interface_parity`` to describe the real cross-framework text.

    ``shared`` must mean the four task-requirement blocks are byte-identical, so
    a framework-specific sentence cannot be added without relabeling the task.
    Framework wording is expected only where input types, unavailable gate
    aliases, moment ordering, or return interfaces genuinely differ; labeling
    keeps those cases deliberate instead of letting one framework's prose drift.
    """
    problems: list[str] = []
    for task_id, per_framework in sorted(requirements.items()):
        if set(per_framework) != set(FRAMEWORKS):
            continue
        expected = "shared" if len(set(per_framework.values())) == 1 else "framework_specific"
        declared = parities.get(task_id, {})
        if len(set(declared.values())) != 1:
            problems.append(f"{suite}/{task_id}: interface_parity differs across frameworks: {declared}")
            continue
        actual = next(iter(declared.values()), "")
        if actual != expected:
            differing = sorted(
                framework for framework, text in per_framework.items() if text != per_framework["qiskit"]
            )
            detail = f" (differs in: {', '.join(differing)})" if differing else ""
            problems.append(
                f"{suite}/{task_id}: interface_parity is {actual!r} but the task requirements are "
                f"{expected.replace('_', '-')}{detail}"
            )
    return problems


def main() -> int:
    problems: list[str] = []
    for suite in SUITES:
        interfaces = _contract_interfaces(suite)
        contract_signatures = _contract_signatures(suite)
        prompt_signatures: dict[str, dict[str, PromptSignature]] = {task_id: {} for task_id in contract_signatures}
        requirements: dict[str, dict[str, str]] = {task_id: {} for task_id in contract_signatures}
        parities: dict[str, dict[str, str]] = {task_id: {} for task_id in contract_signatures}
        for framework in FRAMEWORKS:
            rows = _task_rows(suite, framework)
            if set(rows) != set(interfaces) or set(rows) != set(contract_signatures):
                problems.append(f"{suite}/{framework}: task ids differ from contract ids")
                continue
            if framework == "cudaq":
                problems.extend(_cudaq_capability_problems(suite, rows))
            for task_id, row in sorted(rows.items()):
                where = f"{suite}/{framework}/{task_id}"
                try:
                    # Compare the task-requirements block alone: the submission
                    # contract is framework-specific by construction.
                    body = _split_prompt(str(row.get("prompt", "")))[1]
                    requirements[task_id][framework] = body.split(f"\n{HEADING}\n", 1)[0]
                except ValueError:
                    pass
                parities[task_id][framework] = str(row.get("interface_parity", ""))
                problems.extend(_row_problems(where, row, interfaces[task_id].get(framework)))
                problems.extend(_submission_contract_problems(where, suite, framework, row))
                scaffold_problems, prompt_signature = _prompt_scaffold_problems(
                    where,
                    suite,
                    framework,
                    row,
                    contract_signatures[task_id],
                )
                problems.extend(scaffold_problems)
                if prompt_signature is not None:
                    prompt_signatures[task_id][framework] = prompt_signature
        problems.extend(_cross_framework_signature_problems(suite, prompt_signatures))
        problems.extend(_interface_parity_problems(suite, requirements, parities))
    if problems:
        print("asset consistency failures:\n" + "\n".join(problems))
        return 1
    print("verified task-file, contract, and prompt-scaffold consistency for all suites and frameworks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
