"""Versioned multi-label error taxonomy for benchmark records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

ERROR_TAXONOMY_VERSION: Final = "1"

ErrorAxis = Literal[
    "generation_execution_reliability",
    "interface_observation_validity",
    "construction_resource_fidelity",
    "interaction_lifecycle_fidelity",
    "shortcut_provenance_violation",
    "behavioral_target_mismatch",
    "parameter_domain_robustness",
]

ERROR_AXES: Final[tuple[ErrorAxis, ...]] = (
    "generation_execution_reliability",
    "interface_observation_validity",
    "construction_resource_fidelity",
    "interaction_lifecycle_fidelity",
    "shortcut_provenance_violation",
    "behavioral_target_mismatch",
    "parameter_domain_robustness",
)

AXIS_LABELS: Final[dict[ErrorAxis, str]] = {
    "generation_execution_reliability": "Generation / execution",
    "interface_observation_validity": "Interface / observation",
    "construction_resource_fidelity": "Construction / resources",
    "interaction_lifecycle_fidelity": "Interaction / lifecycle",
    "shortcut_provenance_violation": "Shortcut / provenance",
    "behavioral_target_mismatch": "Behavioral target",
    "parameter_domain_robustness": "Parameter robustness",
}

AXIS_GROUPS: Final[dict[str, tuple[ErrorAxis, ...]]] = {
    "execution": (
        "generation_execution_reliability",
        "interface_observation_validity",
    ),
    "algorithmic": (
        "construction_resource_fidelity",
        "interaction_lifecycle_fidelity",
        "shortcut_provenance_violation",
    ),
    "semantic": (
        "behavioral_target_mismatch",
        "parameter_domain_robustness",
    ),
}

AXIS_GROUP_LABELS: Final[dict[str, str]] = {
    "execution": "Execution",
    "algorithmic": "Algorithmic",
    "semantic": "Semantic",
}

TaxonomyOutcome = Literal[
    "verified_pass",
    "observed_error",
    "grader_nondecision",
    "resource_limit",
    "ungraded",
]

TAXONOMY_OUTCOMES: Final[tuple[TaxonomyOutcome, ...]] = (
    "verified_pass",
    "observed_error",
    "grader_nondecision",
    "resource_limit",
    "ungraded",
)

_EXECUTION_FAILURE_STATUSES = frozenset({"provider_failed", "compile_failed", "run_failed"})
_CASE_STATUSES = ("verified_pass", "semantic_fail", "execution_error", "resource_limit")

_INTERFACE_REASONS = frozenset(
    {
        "malformed_instrument",
        "malformed_probability_table",
        "malformed_semantic_object",
        "objective_api_invalid",
        "requirement_failed:entry_point_signature",
        "requirement_failed:forbidden_measurement_qubits",
        "requirement_failed:max_measurement_count",
        "requirement_failed:min_measurement_count",
        "requirement_failed:required_measurement_qubits",
        "terminal_observation_mismatch",
    }
)
_CONSTRUCTION_REASONS = frozenset(
    {
        "objective_family_invalid",
        "objective_optimization_budget_exceeded",
        "requirement_failed:min_entangling_gate_count",
        "requirement_failed:min_non_measurement_operation_count",
        "requirement_failed:min_num_qubits",
        "requirement_failed:trotter_step_count",
        "structured_qaoa_beta_binding_mismatch",
        "structured_qaoa_gamma_binding_mismatch",
        "structured_qaoa_layer_domain_mismatch",
        "structured_qaoa_wire_domain_mismatch",
        "structured_rotation_family_mismatch",
    }
)
_INTERACTION_REASONS = frozenset(
    {
        "requirement_failed:argument_conditioned_gate",
        "requirement_failed:disconnected_interaction_group",
        "requirement_failed:must_include_entangling_uncompute",
        "requirement_failed:net_unitary_nonlocal",
        "requirement_failed:required_any_interaction_groups",
        "requirement_failed:required_any_interaction_sequences",
        "requirement_failed:required_controlled_correction",
        "requirement_failed:required_interaction_groups",
        "requirement_failed:required_interactions",
        "structured_qaoa_edge_domain_mismatch",
        "structured_qaoa_gate_topology_mismatch",
    }
)
_SHORTCUT_REASONS = frozenset(
    {
        "requirement_failed:forbid_dense_evolution_shortcuts",
        "requirement_failed:forbid_eigensolver_shortcuts",
        "requirement_failed:forbid_library_shortcuts",
        "requirement_failed:forbid_optimizer",
        "requirement_failed:forbid_returned_counts",
        "requirement_failed:forbid_returned_probabilities",
        "requirement_failed:forbid_returned_unitary",
        "requirement_failed:forbid_state_preparation",
        "requirement_failed:forbid_unitary_shortcuts",
        "requirement_failed:forbidden_calls",
        "requirement_failed:forbidden_imports",
        "requirement_failed:forbidden_probability_method",
    }
)
_BEHAVIORAL_REASONS = frozenset(
    {
        "approximation_lower_bound_fails",
        "objective_gap_exceeds_bound",
        "semantic_sanity_check_failed",
        "symbolic_projective_counterexample",
    }
)
_DENSE_GATE_FAMILIES = frozenset({"matrixgate", "qubitunitary", "unitary", "unitarygate"})

_AGGREGATE_REASONS = frozenset(
    {
        "parameter_domain_semantic_fail",
        "semantic_failure",
    }
)
_GRADER_REASON_EXACT = frozenset(
    {
        "audit_blocker",
        "requirement_failed:invalid_argument_gate_contract",
        "requirement_failed:invalid_argument_gate_value",
        "requirement_failed:invalid_connected_interaction_group",
        "requirement_failed:invalid_controlled_correction_contract",
        "requirement_failed:invalid_interaction_contract",
        "terminal_observation_contract_invalid",
    }
)


def classify_error_taxonomy(
    record_status: str,
    evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Classify one benchmark record using only persisted, auditable evidence.

    Args:
        record_status: Benchmark-level status stored on the record.
        evaluation: Serialized evaluation object, when one exists.

    Returns:
        Versioned JSON-compatible taxonomy record.
    """
    semantic = _semantic_result(evaluation)
    preliminary = _preliminary_taxonomy(record_status, evaluation, semantic)
    if preliminary is not None:
        return preliminary

    if semantic is None:
        return _taxonomy_record(outcome="ungraded")

    semantic_status = str(semantic.get("status", ""))
    reason_codes = _reason_codes(semantic)
    case_counts = _parameter_case_status_counts(semantic)
    if semantic_status == "verified_pass":
        return _taxonomy_record(
            outcome="verified_pass",
            reason_codes=reason_codes,
            parameter_case_status_counts=case_counts,
        )
    if semantic_status == "resource_limit":
        return _taxonomy_record(
            outcome="resource_limit",
            axes={"generation_execution_reliability"},
            reason_codes=reason_codes,
            parameter_case_status_counts=case_counts,
        )
    if semantic_status == "execution_error":
        return _taxonomy_record(
            outcome="observed_error",
            axes={"generation_execution_reliability"},
            reason_codes=reason_codes,
            parameter_case_status_counts=case_counts,
        )
    if semantic_status != "semantic_fail":
        return _taxonomy_record(
            outcome="ungraded",
            reason_codes=reason_codes,
            parameter_case_status_counts=case_counts,
        )

    actionable = _actionable_semantic_fail_reasons(semantic)
    grader_reasons = {reason for reason in actionable if _is_grader_reason(reason)}
    observed_error_reasons = actionable - grader_reasons
    axes = {axis for reason in observed_error_reasons for axis in _axes_for_reason(reason)}
    if case_counts["verified_pass"] and case_counts["semantic_fail"]:
        axes.add("parameter_domain_robustness")
    unclassified = {reason for reason in observed_error_reasons if not _axes_for_reason(reason)}
    outcome: TaxonomyOutcome = "observed_error" if observed_error_reasons or axes else "grader_nondecision"
    return _taxonomy_record(
        outcome=outcome,
        axes=axes,
        reason_codes=reason_codes,
        grader_reason_codes=grader_reasons,
        unclassified_reason_codes=unclassified,
        parameter_case_status_counts=case_counts,
    )


def _preliminary_taxonomy(
    record_status: str,
    evaluation: Mapping[str, Any] | None,
    semantic: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if record_status == "infrastructure_error":
        return _taxonomy_record(
            outcome="grader_nondecision",
            reason_codes={"benchmark_status:infrastructure_error"},
        )
    if _evaluation_error_type(evaluation) == "InfrastructureError":
        return _taxonomy_record(
            outcome="grader_nondecision",
            reason_codes={"evaluation_error_type:InfrastructureError"},
        )
    if semantic is not None and _diagnostic_value(semantic, "failure_origin") == "grader_verification":
        return _taxonomy_record(
            outcome="grader_nondecision",
            reason_codes=_reason_codes(semantic),
            parameter_case_status_counts=_parameter_case_status_counts(semantic),
        )
    if record_status in _EXECUTION_FAILURE_STATUSES:
        return _taxonomy_record(
            outcome="observed_error",
            axes={"generation_execution_reliability"},
            reason_codes={f"benchmark_status:{record_status}"},
        )
    return None


def _taxonomy_record(
    *,
    outcome: TaxonomyOutcome,
    axes: set[ErrorAxis] | None = None,
    reason_codes: set[str] | None = None,
    grader_reason_codes: set[str] | None = None,
    unclassified_reason_codes: set[str] | None = None,
    parameter_case_status_counts: Counter[str] | None = None,
) -> dict[str, Any]:
    selected = axes or set()
    case_counts = parameter_case_status_counts or Counter()
    return {
        "taxonomy_version": ERROR_TAXONOMY_VERSION,
        "multi_label": True,
        "outcome": outcome,
        "axes": [axis for axis in ERROR_AXES if axis in selected],
        "reason_codes": sorted(reason_codes or set()),
        "grader_reason_codes": sorted(grader_reason_codes or set()),
        "unclassified_reason_codes": sorted(unclassified_reason_codes or set()),
        "parameter_case_status_counts": {status: case_counts[status] for status in _CASE_STATUSES},
    }


def _semantic_result(evaluation: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(evaluation, Mapping):
        return None
    semantic = evaluation.get("semantic_result")
    return semantic if isinstance(semantic, Mapping) else None


def _evaluation_error_type(evaluation: Mapping[str, Any] | None) -> str | None:
    if not isinstance(evaluation, Mapping):
        return None
    return _text(evaluation.get("error_type")) or None


def _reason_codes(semantic: Mapping[str, Any]) -> set[str]:
    reasons = {_text(semantic.get("reason_code"))}
    for evidence in _mapping_items(semantic.get("evidence")):
        reasons.add(_text(evidence.get("reason_code")))
    return {reason for reason in reasons if reason}


def _actionable_semantic_fail_reasons(semantic: Mapping[str, Any]) -> set[str]:
    evidence = _mapping_items(semantic.get("evidence"))
    reasons = set()
    for item in evidence:
        case_status = _case_status(item)
        reason = _text(item.get("reason_code"))
        if reason and case_status not in {"verified_pass", "execution_error", "resource_limit"}:
            reasons.add(reason)
    top_level = _text(semantic.get("reason_code"))
    if not reasons and top_level:
        reasons.add(top_level)
    specific = {reason for reason in reasons if reason not in _AGGREGATE_REASONS and not _is_pass_reason(reason)}
    if specific:
        return specific
    return {top_level} if top_level and not _is_pass_reason(top_level) else set()


def _parameter_case_status_counts(semantic: Mapping[str, Any]) -> Counter[str]:
    counts = _case_count_diagnostics(semantic)
    if counts is not None:
        return counts
    counts = _case_status_diagnostics(semantic)
    if counts is not None:
        return counts
    evidence_cases: set[tuple[str, str]] = set()
    for item in _mapping_items(semantic.get("evidence")):
        case_status = _case_status(item)
        index = _precondition_value(item, "case_index")
        if case_status in _CASE_STATUSES and index is not None:
            evidence_cases.add((index, case_status))
    return Counter(status for _, status in evidence_cases)


def _case_count_diagnostics(semantic: Mapping[str, Any]) -> Counter[str] | None:
    count_diagnostics: dict[str, int] = {}
    for diagnostic in _mapping_items(semantic.get("diagnostics")):
        name = _text(diagnostic.get("name"))
        raw_count = _text(diagnostic.get("value"))
        if not name.startswith("parameter_case_count:"):
            continue
        status = name.partition(":")[2]
        try:
            count = int(raw_count)
        except ValueError:
            continue
        if status in _CASE_STATUSES and count >= 0:
            count_diagnostics[status] = count
    return Counter(count_diagnostics) if set(count_diagnostics) == set(_CASE_STATUSES) else None


def _case_status_diagnostics(semantic: Mapping[str, Any]) -> Counter[str] | None:
    diagnostic_cases: dict[str, str] = {}
    for diagnostic in _mapping_items(semantic.get("diagnostics")):
        name = _text(diagnostic.get("name"))
        status = _text(diagnostic.get("value"))
        if name.startswith("parameter_case_status:") and status in _CASE_STATUSES:
            diagnostic_cases[name.partition(":")[2]] = status
    return Counter(diagnostic_cases.values()) if diagnostic_cases else None


def _case_status(evidence: Mapping[str, Any]) -> str | None:
    return _precondition_value(evidence, "case_status")


def _precondition_value(evidence: Mapping[str, Any], name: str) -> str | None:
    prefix = f"{name}="
    for value in _string_items(evidence.get("preconditions")):
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return None


def _diagnostic_value(semantic: Mapping[str, Any], name: str) -> str | None:
    for diagnostic in _mapping_items(semantic.get("diagnostics")):
        if diagnostic.get("name") == name:
            return _text(diagnostic.get("value")) or None
    return None


def _axes_for_reason(reason: str) -> set[ErrorAxis]:
    axes: set[ErrorAxis] = set()
    if reason in _INTERFACE_REASONS or reason.startswith("forbidden_measured_qubit:"):
        axes.add("interface_observation_validity")
    if (
        reason in _CONSTRUCTION_REASONS
        or reason.startswith("missing_gate_family:")
        or reason.startswith("forbidden_gate_family:")
        or reason.startswith("symbolic_forbidden_gate_family:")
        or reason.startswith("requirement_failed:min_gate_family_counts.")
        or reason.startswith("requirement_failed:min_gate_family_group_counts.")
        or reason.startswith("requirement_failed:min_any_gate_family_counts.")
    ):
        axes.add("construction_resource_fidelity")
    if reason in _INTERACTION_REASONS or reason.startswith(
        (
            "requirement_failed:missing_controlled_x_interaction:",
            "requirement_failed:missing_interaction:",
            "requirement_failed:missing_parity_interaction:",
        )
    ):
        axes.add("interaction_lifecycle_fidelity")
    if reason in _SHORTCUT_REASONS or _is_dense_gate_shortcut(reason):
        axes.add("shortcut_provenance_violation")
    if reason in _BEHAVIORAL_REASONS or reason.endswith("_exceeds_fail_bound"):
        axes.add("behavioral_target_mismatch")
    return axes


def _is_dense_gate_shortcut(reason: str) -> bool:
    for prefix in ("forbidden_gate_family:", "symbolic_forbidden_gate_family:"):
        if reason.startswith(prefix):
            return reason.removeprefix(prefix) in _DENSE_GATE_FAMILIES
    return False


def _is_grader_reason(reason: str) -> bool:
    return reason in _GRADER_REASON_EXACT or reason.startswith("requirement_failed:invalid_")


def _is_pass_reason(reason: str) -> bool:
    return reason.endswith(("_within_pass_bound", "_upper_bound_passes", "_family_identity")) or reason in {
        "all_parameter_cases_passed",
        "metric_within_pass_bound",
        "symbolic_projective_identity",
    }


def _mapping_items(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_items(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


__all__ = [
    "AXIS_GROUP_LABELS",
    "AXIS_GROUPS",
    "AXIS_LABELS",
    "ERROR_AXES",
    "ERROR_TAXONOMY_VERSION",
    "ErrorAxis",
    "TaxonomyOutcome",
    "TAXONOMY_OUTCOMES",
    "classify_error_taxonomy",
]
