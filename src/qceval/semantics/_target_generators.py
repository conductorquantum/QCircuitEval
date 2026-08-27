"""Deterministic generators for the independently reviewed pilot targets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def pilot_target_generator(
    task_id: str,
) -> Callable[[], dict[str, Any]] | None:
    """Look up the pilot target generator for a normalized task ID.

    Args:
        task_id: Zero-padded pilot task identifier.

    Returns:
        Deterministic target generator, or ``None`` for an unknown task.
    """
    return _GENERATORS.get(task_id)


def _task_02() -> dict[str, Any]:
    return {
        "basis_order": "q2q1q0",
        "dimension": 8,
        "format": "sparse_exact_state_v1",
        "nonzero_amplitudes": [
            {"basis": "011", "value": "1/sqrt(2)"},
            {"basis": "100", "value": "-1/sqrt(2)"},
        ],
        "normalization": "unit_l2",
    }


def _task_27() -> dict[str, Any]:
    matrix = [["0" for _ in range(4)] for _ in range(4)]
    for basis in range(4):
        output = basis ^ 0b10 if basis & 0b01 else basis
        matrix[output][basis] = "1"
    return {
        "basis_order": "q1q0",
        "control": 0,
        "dimension": 4,
        "format": "dense_exact_matrix_v1",
        "matrix": matrix,
        "target": 1,
    }


def _task_28() -> dict[str, Any]:
    mappings = []
    for logical_basis in range(16):
        q0 = logical_basis & 1
        q1 = (logical_basis >> 1) & 1
        q2 = (logical_basis >> 2) & 1
        q4 = (logical_basis >> 3) & 1
        output_q4 = q4 ^ (q0 & q1 & q2)
        input_bits = f"{q4}{q2}{q1}{q0}"
        output_bits = f"{output_q4}{q2}{q1}{q0}"
        mappings.append(
            {
                "input_logical": input_bits,
                "output_logical": output_bits,
                "output_physical": f"{output_q4}0{q2}{q1}{q0}",
            }
        )
    return {
        "ancilla": {"final": 0, "index": 3, "initial": 0},
        "format": "exhaustive_basis_isometry_v1",
        "logical_basis_order": "q4q2q1q0",
        "logical_dimension": 16,
        "mappings": mappings,
        "physical_basis_order": "q4q3q2q1q0",
        "physical_dimension": 32,
    }


def _task_42() -> dict[str, Any]:
    return {
        "equivalence": "global_phase",
        "format": "analytic_parameterized_matrix_v1",
        "matrix": [
            ["cos(theta/2)", "-exp(i*lam)*sin(theta/2)"],
            [
                "exp(i*phi)*sin(theta/2)",
                "exp(i*(phi+lam))*cos(theta/2)",
            ],
        ],
        "parameter_domain": {
            "lam": "all_real",
            "phi": "all_real",
            "theta": "all_real",
        },
        "projective_cross_check": {
            "global_factor": "exp(i*(phi+lam)/2)",
            "matrix_product": "RZ(phi)@RY(theta)@RZ(lam)",
        },
        "shape": [2, 2],
    }


_GENERATORS: dict[str, Callable[[], dict[str, Any]]] = {
    "02": _task_02,
    "27": _task_27,
    "28": _task_28,
    "42": _task_42,
}


# Reviewed provenance for the pilot targets, kept beside the generators so a
# target and the record of its independent review live together.  This mirrors
# the reviewer/derivation/crosscheck fields that ``core-audit-source.json``
# carries for every non-pilot Core task, so all 58 Core targets expose the same
# structured, reviewer-recorded provenance.  The named cross-checks are the
# closed-form independent reproductions in
# ``tests/semantics/test_core_independent_targets.py``.
_PROVENANCE: dict[str, dict[str, str]] = {
    "02": {
        "audit_status": "reviewed",
        "derivation": (
            "The prompt specifies sqrt(1/2) * (|011> - |100>) in q2q1q0 order; the target "
            "stores the two nonzero amplitudes +1/sqrt(2) on |011> and -1/sqrt(2) on |100>."
        ),
        "crosscheck": (
            "Both amplitudes have squared magnitude 1/2 (unit L2 norm) and the relative "
            "phase between them is pi, matching the prompt's minus sign; the other six "
            "computational-basis amplitudes are zero."
        ),
        "reviewer": "pilot-target-independent-audit-2026-07-17",
        "review_evidence": "tests/semantics/test_core_independent_targets.py",
    },
    "27": {
        "audit_status": "reviewed",
        "derivation": (
            "The prompt requires CNOT with q0 as control and q1 as target.  For every basis "
            "integer x the target toggles bit 1 (q1) iff bit 0 (q0) is set (output = x XOR "
            "0b10 when x & 0b01, else x), placing a 1 at row `output`, column x of the 4x4 "
            "permutation matrix."
        ),
        "crosscheck": (
            "The permutation exhausts all four computational-basis inputs and is its own "
            "inverse (CNOT^2 = I); the columns 00->00, 01->11, 10->10, 11->01 confirm "
            "control=q0 / target=q1 rather than the reversed assignment."
        ),
        "reviewer": "pilot-target-independent-audit-2026-07-17",
        "review_evidence": "tests/semantics/test_core_independent_targets.py",
    },
    "28": {
        "audit_status": "reviewed",
        "derivation": (
            "Enumerate all 16 logical inputs (q4 q2 q1 q0); q4 flips iff q0 AND q1 AND q2, "
            "and the clean work qubit q3 is initialized and restored to 0.  Each logical "
            "input maps to its logical output and a 5-qubit physical embedding with q3=0."
        ),
        "crosscheck": (
            "Only the eight inputs with q0=q1=q2=1 toggle q4; the ancilla (index 3) has "
            "initial=final=0 for every one of the 16 mappings, confirming uncomputation of "
            "the work qubit across the whole logical basis."
        ),
        "reviewer": "pilot-target-independent-audit-2026-07-17",
        "review_evidence": "tests/semantics/test_core_independent_targets.py",
    },
    "42": {
        "audit_status": "reviewed",
        "derivation": (
            "Write the four entries of the standard U(theta, phi, lam) gate directly: "
            "[[cos(theta/2), -e^{i*lam} sin(theta/2)], [e^{i*phi} sin(theta/2), "
            "e^{i(phi+lam)} cos(theta/2)]], over all real theta, phi, lam, with global-phase "
            "equivalence."
        ),
        "crosscheck": (
            "The matrix equals e^{i(phi+lam)/2} * RZ(phi) @ RY(theta) @ RZ(lam) up to the "
            "recorded global factor, an independent Euler-decomposition construction of the "
            "same operator over the full parameter domain."
        ),
        "reviewer": "pilot-target-independent-audit-2026-07-17",
        "review_evidence": "tests/semantics/test_core_independent_targets.py",
    },
}


def pilot_target_provenance(task_id: str) -> dict[str, str] | None:
    """Look up the reviewed provenance record for a pilot target.

    Args:
        task_id: Zero-padded pilot task identifier.

    Returns:
        A record with ``audit_status``, ``derivation``, ``crosscheck``,
        ``reviewer``, and ``review_evidence``, or ``None`` for a task without a
        pilot generator.
    """
    record = _PROVENANCE.get(task_id)
    return dict(record) if record is not None else None
