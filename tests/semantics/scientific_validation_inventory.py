"""Release-gate inventory for per-contract scientific validation evidence."""

from __future__ import annotations

from typing import Final, TypeAlias

from tests.semantics.test_core_independent_targets import CORE_INDEPENDENT_TARGET_EVIDENCE

ValidationEvidence: TypeAlias = dict[str, tuple[str, ...]]


REQUIRED_VALIDATION_CATEGORIES: Final = frozenset({"independent_target", "known_wrong", "alternate_valid"})
CORE_TASK_IDS: Final = tuple(f"{index:02d}" for index in range(1, 59))
QEC_TASK_IDS: Final = tuple(f"qec{index:02d}" for index in range(1, 13))
RARE_KIND_TASK_IDS: Final = frozenset({"18", "28", "33", "34"})

_THIS_MODULE = "tests/semantics/test_scientific_validation_inventory.py"
_RARE_MODULE = "tests/semantics/test_rare_kind_cross_framework.py"

SCIENTIFIC_VALIDATION_INVENTORY: Final[dict[tuple[str, str], ValidationEvidence]] = {}

for task_id in CORE_TASK_IDS:
    known_wrong = (
        f"{_RARE_MODULE}::test_rare_kind_adversary_with_same_observed_output_is_rejected"
        if task_id in RARE_KIND_TASK_IDS
        else f"{_THIS_MODULE}::test_core_known_wrong_implementation_is_rejected"
    )
    alternate_valid = (
        f"{_RARE_MODULE}::test_rare_kind_structural_alternate_is_accepted"
        if task_id in RARE_KIND_TASK_IDS
        else f"{_THIS_MODULE}::test_core_alternate_valid_implementation_passes"
    )
    SCIENTIFIC_VALIDATION_INVENTORY[("core", task_id)] = {
        "independent_target": CORE_INDEPENDENT_TARGET_EVIDENCE[task_id],
        "known_wrong": (known_wrong,),
        "alternate_valid": (alternate_valid,),
    }

for task_id in QEC_TASK_IDS:
    alternate_valid = {
        "qec01": (
            "tests/semantics/test_qec_semantic_contracts.py::"
            "test_qec_encode_decode_accepts_the_other_valid_interaction_order"
        ),
        "qec03": (
            "tests/semantics/test_qec_semantic_contracts.py::test_qec03_accepts_reverse_order_uncomputation_decoder"
        ),
        "qec04": (
            "tests/semantics/test_qec_semantic_contracts.py::test_qec04_accepts_palindromic_hadamard_reverse_decoder"
        ),
        "qec12": f"{_THIS_MODULE}::test_qec12_alternate_encoder_and_decoder_order_passes",
    }.get(task_id, f"{_THIS_MODULE}::test_qec_alternate_terminal_measurement_order_passes")
    SCIENTIFIC_VALIDATION_INVENTORY[("qec", task_id)] = {
        "independent_target": (f"{_THIS_MODULE}::test_qec_target_matches_independent_formula",),
        "known_wrong": ("tests/semantics/test_qec_semantic_contracts.py::test_trivial_shortcut_circuits_are_rejected",),
        "alternate_valid": (alternate_valid,),
    }
