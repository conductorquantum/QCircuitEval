"""Centralized submission-contract rendering for bundled prompts.

Every prompt's ``Submission contract:`` section is generated from one template
per framework plus a small typed capability record stored on the task row:

.. code-block:: json

    "submission": {
        "capability": "measured_distribution",
        "strict": false,
        "prebuilt_algorithm_ban": false
    }

A Core row's capability must agree with the framework interface its contract
declares in ``terminal_observation``: a CUDA-Q row is ``measured_distribution``
exactly when the contract names a CUDA-Q measurement register, because
``frameworks.cudaq.lowering`` then samples that register whether or not the
kernel calls ``mz``. ``ci/check_asset_consistency.py`` enforces the agreement.

Capabilities describe the return interface of the graded object:

- ``measured_distribution`` — terminal measurement of the requested outputs.
- ``unmeasured_unitary`` — unmeasured program; the total unitary is graded.
- ``unmeasured_state`` — unmeasured program; the prepared state is graded.
- ``analytic_probabilities`` — analytic QNode readout (tape capture).

``strict`` selects the tighter interface wording used by the deterministic
fixed-recipe tasks (Core 45-58): a fixed measurement key/order and a ban on
executing samplers, optimizers, or backends. ``prebuilt_algorithm_ban`` is a
separate Core-only policy flag for tasks whose structural grader rejects
prebuilt high-level implementations (Core 51-58). QEC rows use
``error_injection`` instead of those flags to include the Pauli-error bullet
only when the task signature takes an error argument.

Prompt/asset consistency tests assert that every packaged prompt's section
byte-matches this renderer, so wording cannot drift per task.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

HEADING = "Submission contract:"

CAPABILITIES = frozenset(
    {
        "measured_distribution",
        "unmeasured_unitary",
        "unmeasured_state",
        "analytic_probabilities",
    }
)

_DENSE_BAN = (
    "- Build the circuit from individual gates. A solution that wraps the complete answer "
    "in one matrix gate spanning every qubit is rejected."
)
_NO_RESULT_RETURNS = "- Do not return sampled counts, probability arrays, statevectors, or hardcoded dictionaries."
_LSB = "- Qubit 0 is the least-significant bit."
_LSB_WIRE = "- Wire 0 is the least-significant bit."
_DETERMINISTIC = "- Return a deterministic quantum program. Do not run a sampler, optimizer, or backend."
_NO_PREBUILT_ALGORITHMS = (
    "- Build the task's graded algorithm or required decomposition from gates. Do not use a "
    "library routine or constructor that directly supplies it. Constructors for subordinate "
    "components are allowed unless the task requires you to decompose that component."
)


def _core_qiskit(capability: str, strict: bool, prebuilt_algorithm_ban: bool) -> list[str]:
    bullets = [
        _DENSE_BAN,
        "- Preserve the exact function signature and return a Qiskit QuantumCircuit.",
        _NO_RESULT_RETURNS,
        _LSB,
    ]
    if capability == "measured_distribution":
        if strict:
            bullets.append(
                "- Measure exactly the requested output qubits into same-index classical "
                "bits unless the prompt says otherwise."
            )
        else:
            bullets.append(
                "- Map the requested logical output qubits to classical bits in the "
                "task-specific output order. For sparse outputs, either same-index "
                "classical bits or a compact classical register preserving that order is "
                "accepted."
            )
            bullets.append(
                "- Measure exactly the register requested by the prompt. Do not measure "
                "ancillas unless explicitly requested."
            )
    else:
        bullets.append("- Do not add any measurements. Return the unmeasured circuit.")
    if strict:
        bullets.append(_DETERMINISTIC)
    if prebuilt_algorithm_ban:
        bullets.append(_NO_PREBUILT_ALGORITHMS)
    return bullets


def _core_cirq(capability: str, strict: bool, prebuilt_algorithm_ban: bool) -> list[str]:
    bullets = [
        _DENSE_BAN,
        "- Preserve the exact function signature and return a cirq.Circuit.",
        _NO_RESULT_RETURNS,
        _LSB,
    ]
    if capability == "measured_distribution":
        if strict:
            bullets.append('- Measure requested output qubits with key "result" in descending qubit order.')
        else:
            bullets.append(
                "- Measure logical output qubits in descending logical-qubit order; the "
                "measurement-key name is arbitrary."
            )
            bullets.append(
                "- Measure exactly the register requested by the prompt. Do not measure "
                "ancillas unless explicitly requested."
            )
    else:
        bullets.append("- Do not add any measurements to the circuit. Return the unmeasured cirq.Circuit.")
    if strict:
        bullets.append(_DETERMINISTIC)
    if prebuilt_algorithm_ban:
        bullets.append(_NO_PREBUILT_ALGORITHMS)
    return bullets


def _core_pennylane(capability: str, strict: bool, prebuilt_algorithm_ban: bool) -> list[str]:
    del capability
    bullets = [
        _DENSE_BAN,
        "- Preserve the exact function signature.",
        '- Return qml.probs(...) from a QNode on qml.device("default.qubit", wires=n, '
        "shots=None); return the evaluated probability array, not the QNode function "
        "itself. The circuit operations recorded before qml.probs are the graded quantum "
        "program.",
        _LSB_WIRE,
        "- Return probabilities for exactly the requested output wires in descending wire "
        "order; include no ancillas unless explicitly requested. When the task requests no "
        "output register and grades the program itself, use every wire the program acts on, "
        "in descending wire order.",
    ]
    if strict:
        bullets.append(_DETERMINISTIC)
    if prebuilt_algorithm_ban:
        bullets.append(_NO_PREBUILT_ALGORITHMS)
    return bullets


def _core_cudaq(capability: str, strict: bool, prebuilt_algorithm_ban: bool) -> list[str]:
    bullets = [
        _DENSE_BAN,
        "- Preserve the exact function signature.",
        "- Return a @cudaq.kernel-decorated callable (or PyKernel) without executing it; "
        "do not call cudaq.sample yourself.",
        _LSB,
    ]
    if capability == "measured_distribution":
        bullets.append(
            "- Measure exactly the register requested by the prompt, or return the kernel "
            "unmeasured; an unmeasured kernel is sampled on that same register. Do not "
            "measure ancillas unless explicitly requested."
        )
    else:
        bullets.append("- Return an unmeasured kernel; do not call mz, measure, or cudaq.sample.")
    if strict:
        bullets.append(_DETERMINISTIC)
    if prebuilt_algorithm_ban:
        bullets.append(_NO_PREBUILT_ALGORITHMS)
    return bullets


_CORE_TEMPLATES = {
    "qiskit": _core_qiskit,
    "cirq": _core_cirq,
    "pennylane": _core_pennylane,
    "cudaq": _core_cudaq,
}

_QEC_ERROR_BULLET = {
    "qiskit": "- Inject the requested Pauli error as an actual circuit gate.",
    "cirq": "- Inject the requested Pauli error as an actual circuit gate.",
    "pennylane": "- Inject the requested Pauli error as an actual QNode gate.",
    "cudaq": "- Inject the requested Pauli error as an actual kernel gate.",
}


def _qec_bullets(framework: str, error_injection: bool) -> list[str]:
    if framework == "pennylane":
        bullets = [
            "- Preserve the exact function signature and return a QNode result from qml.probs(...).",
            "- Do not return hardcoded arrays, counts, statevectors, unitaries, or dictionaries.",
        ]
        if error_injection:
            bullets.append(_QEC_ERROR_BULLET[framework])
        bullets += [
            "- Use elementary gates only. Do not use state-preparation, state-injection, "
            "dense-matrix, or arbitrary-unitary constructors.",
            "- Do not use QEC, decoder, noisy-device, sampler, or optimizer libraries or APIs.",
            "- Wire 0 is the least-significant bit.",
            "- Return probabilities for exactly the requested output wires, listed from "
            "highest output-bit significance to lowest; include no unused work wires.",
        ]
        return bullets
    if framework == "cudaq":
        bullets = [
            "- Preserve the exact function signature and return a @cudaq.kernel-decorated callable or PyKernel.",
            "- Declare wrapper runtime parameters as kernel parameters too; the returned "
            "kernel is invoked with those values.",
            "- Return the kernel without executing it; do not return counts, probability "
            "arrays, statevectors, unitaries, or dictionaries.",
        ]
        if error_injection:
            bullets.append(_QEC_ERROR_BULLET[framework])
        bullets += [
            "- Use elementary gates only. Do not use state-preparation, state-injection, "
            "dense-matrix, or arbitrary-unitary constructors.",
            "- Do not use QEC, decoder, noise-model, simulator, sampler, or optimizer libraries or APIs.",
            "- Qubit 0 is the least-significant bit.",
            "- Measure exactly the requested outputs and no unused work qubits.",
        ]
        return bullets
    return_form = (
        "- Preserve the exact function signature and return a Qiskit QuantumCircuit."
        if framework == "qiskit"
        else "- Preserve the exact function signature and return a cirq.Circuit."
    )
    lsb = (
        "- Qubit 0 and classical bit 0 are the least-significant bits."
        if framework == "qiskit"
        else "- Qubit 0 is the least-significant bit."
    )
    bullets = [
        return_form,
        "- Return the circuit without executing it; do not return counts, probability "
        "arrays, statevectors, unitaries, or dictionaries.",
    ]
    if error_injection:
        bullets.append(_QEC_ERROR_BULLET[framework])
    bullets += [
        "- Use elementary gates only. Do not use state-preparation, state-injection, "
        "dense-matrix, or arbitrary-unitary constructors.",
        "- Do not use QEC, decoder, noise-model, simulator, sampler, or optimizer libraries or APIs.",
        lsb,
        "- Measure exactly the requested outputs and no unused work qubits.",
    ]
    return bullets


def render_submission_contract(
    suite: str,
    framework: str,
    *,
    capability: str,
    strict: bool = False,
    prebuilt_algorithm_ban: bool = False,
    error_injection: bool = False,
) -> str:
    """Render one prompt's submission-contract section (heading plus bullets).

    Args:
        suite: ``core`` or ``qec``.
        framework: Target framework name.
        capability: Typed return-interface capability for the row.
        strict: Core-only tighter interface wording (fixed key/order, no execution).
        prebuilt_algorithm_ban: Core-only ban on prebuilt high-level implementations.
        error_injection: QEC-only flag adding the Pauli-error bullet.

    Returns:
        The complete section text starting with ``Submission contract:``.

    Raises:
        ValueError: If the capability or suite is unknown.
    """
    if capability not in CAPABILITIES:
        raise ValueError(f"unknown submission capability: {capability}")
    if suite == "core":
        bullets = _CORE_TEMPLATES[framework](capability, strict, prebuilt_algorithm_ban)
    elif suite == "qec":
        if strict or prebuilt_algorithm_ban:
            raise ValueError("strict and prebuilt_algorithm_ban are Core-only submission flags")
        bullets = _qec_bullets(framework, error_injection)
    else:
        raise ValueError(f"unknown suite: {suite}")
    return HEADING + "\n" + "\n".join(bullets)


def expected_submission_contract(suite: str, framework: str, row: Mapping[str, Any]) -> str:
    """Render the submission contract declared by a bundled task row.

    Args:
        suite: ``core`` or ``qec``.
        framework: Target framework name.
        row: Bundled task payload carrying a ``submission`` capability record.

    Returns:
        The section text the packaged prompt must contain.

    Raises:
        ValueError: If the row lacks a typed submission capability.
    """
    submission = row.get("submission")
    if not isinstance(submission, Mapping) or "capability" not in submission:
        raise ValueError(f"task row {suite}/{framework}/{row.get('task_id')} lacks submission capability")
    return render_submission_contract(
        suite,
        framework,
        capability=str(submission["capability"]),
        strict=bool(submission.get("strict", False)),
        prebuilt_algorithm_ban=bool(submission.get("prebuilt_algorithm_ban", False)),
        error_injection=bool(submission.get("error_injection", False)),
    )
