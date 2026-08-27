"""Fallback canonical-source generation for smoke provider tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qceval.models import ProviderRequest
from qceval.providers.smoke.deterministic import _deterministic_code
from qceval.providers.smoke.probs import _probabilities_for, _probability_code
from qceval.providers.smoke.structural import _structural_code
from qceval.providers.smoke.unitary import _unitary_code

PENNYLANE_GENERATED_FALLBACK_TASKS = {"10", "33", "34", "39", "40", "41", "42", "43", "50"}
_CUDAQ_FALLBACK_TASK_IDS = frozenset({"10", "27", "32", "33", "34", "37", "41", "42", "43", "44", "50"})


def generated_canonical_code(request: ProviderRequest) -> str | None:
    """Return generated canonical code for tasks without bundled source.

    Args:
        request: Provider request containing framework, task id, prompt, entry
            point, and canonical class metadata.

    Returns:
        Python source for supported smoke tasks, or ``None`` when no generated
        fallback exists.
    """
    spec = request.metadata.get("canonical_class")
    if not isinstance(spec, Mapping):
        return None
    if (
        request.framework == "pennylane"
        and "canonical_solution" in request.metadata
        and str(request.task_id).zfill(2) not in PENNYLANE_GENERATED_FALLBACK_TASKS
    ):
        return None
    if request.framework == "cudaq" and request.task_id not in _CUDAQ_FALLBACK_TASK_IDS:
        return None
    if _is_unitary_spec(spec):
        return _unitary_code(request.entry_point, request.framework, str(spec["target_unitary"]))
    if _is_structural_spec(spec):
        return _structural_code(request.entry_point, request.framework)
    if spec.get("type") == "deterministic_dominant":
        return _deterministic_code(request.entry_point, request.framework, spec)
    probabilities = _probabilities_for(request, spec)
    if probabilities is None:
        return None
    return _probability_code(request.entry_point, request.framework, probabilities)


def _is_unitary_spec(spec: Mapping[str, Any]) -> bool:
    return spec.get("type") == "exact_distribution" and spec.get("comparison") == "unitary"


def _is_structural_spec(spec: Mapping[str, Any]) -> bool:
    return spec.get("type") == "structural" and spec.get("structural_name") == "vqe_z2_ansatz"
