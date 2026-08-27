"""CUDA-Q source metadata extraction from candidate code."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from qceval.evals.structure import (
    OperationSignature,
    detect_repeated_blocks,
)
from qceval.frameworks.cudaq.metadata_patterns import (
    _CTRL_LIST_RE,
    _CTRL_PAIR_FAMILIES,
    _CTRL_PAIR_RE,
    _CTRL_ROT_DIRECT_RE,
    _CTRL_ROT_PAIR_RE,
    _CTRL_ROTATION_FAMILIES,
    _DIRECT_ENTANGLING_NAMES,
    _GATE_CALL_RE,
    _GATE_NAMES,
    _MEASURE_QUBIT_RE,
    _MEASUREMENT_CALL_RE,
    _QALLOC_RE,
    _QVECTOR_RE,
    _ROT_Q_RE,
    _SINGLE_Q_FAMILIES,
    _SINGLE_Q_RE,
    _THREE_Q_RE,
    _TWO_Q_RE,
    _add_family,
    _all_pairs,
    _pair,
    _SpannedOperationSignature,
)
from qceval.frameworks.cudaq.metadata_source import (
    _current_target_name,
    _merge_gate_family_counts,
    _source_facts,
)


def _base_metadata(cudaq: Any) -> dict[str, Any]:
    return {
        "framework": "cudaq",
        "num_qubits": None,
        "measurement_count": None,
        "non_measurement_operation_count": None,
        "operation_counts": {},
        "gate_family_counts": {},
        "interaction_pairs": [],
        "entangling_gate_count": None,
        "circuit_depth": None,
        "repeated_block_count": None,
        "measurement_pairs": [],
        "has_measurements": None,
        "cudaq_target": _current_target_name(cudaq),
    }


def _operation_metadata_from_code(
    code: str,
    *,
    default_measurement_qubits: Sequence[int] | None = None,
) -> dict[str, Any]:
    facts = _source_facts(code)
    op_counts: dict[str, int] = {}
    gate_family_counts: dict[str, int] = {}
    interaction_pairs: list[list[int]] = []
    entangling = 0
    measurement_qubits = list(facts.measurement_indices)
    block_ops = _operation_signatures_from_code(code)
    for line in code.splitlines():
        clean = line.split("#", 1)[0]
        for match in _GATE_CALL_RE.finditer(clean):
            name = match.group(1)
            if name not in _GATE_NAMES:
                continue
            op_counts[name] = op_counts.get(name, 0) + 1
            entangling += int(".ctrl(" in clean or name in _DIRECT_ENTANGLING_NAMES)
        entangling_spans = _entangling_family_metadata(
            clean,
            gate_family_counts,
            interaction_pairs,
        )
        _single_gate_family_metadata(
            clean,
            entangling_spans,
            gate_family_counts,
        )
    _merge_gate_family_counts(
        gate_family_counts,
        facts.gate_family_counts,
    )
    interaction_pairs.extend(pair for pair in facts.interaction_pairs if pair not in interaction_pairs)
    measurement_count = max(
        op_counts.get("mz", 0) + op_counts.get("measure", 0),
        len(measurement_qubits),
    )
    if measurement_count == 0 and default_measurement_qubits:
        measurement_qubits = list(default_measurement_qubits)
        measurement_count = len(measurement_qubits)
    return {
        "measurement_count": measurement_count,
        "non_measurement_operation_count": sum(
            count for name, count in op_counts.items() if name not in {"mz", "measure"}
        ),
        "circuit_depth": len(block_ops),
        "repeated_block_count": detect_repeated_blocks(block_ops),
        "measurement_pairs": [[qubit, index] for index, qubit in enumerate(measurement_qubits)],
        "operation_counts": op_counts,
        "gate_family_counts": gate_family_counts,
        "interaction_pairs": interaction_pairs,
        "entangling_gate_count": entangling,
        "has_measurements": measurement_count > 0,
        "measurement_qubits": measurement_qubits,
    }


def _entangling_family_metadata(
    line: str,
    gate_family_counts: dict[str, int],
    interaction_pairs: list[list[int]],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _CTRL_LIST_RE.finditer(line):
        if match.group(1) == "x":
            _add_family(gate_family_counts, "ccx")
            interaction_pairs.extend(
                _all_pairs(
                    [
                        int(match.group(2)),
                        int(match.group(3)),
                        int(match.group(4)),
                    ]
                )
            )
        spans.append(match.span())
    for match in _CTRL_PAIR_RE.finditer(line):
        family = _CTRL_PAIR_FAMILIES.get(match.group(1))
        if family is not None:
            _add_family(gate_family_counts, family)
            interaction_pairs.append(_pair(int(match.group(2)), int(match.group(3))))
        spans.append(match.span())
    for match in _THREE_Q_RE.finditer(line):
        _add_family(gate_family_counts, match.group(1))
        interaction_pairs.extend(
            _all_pairs(
                [
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                ]
            )
        )
        spans.append(match.span())
    for match in _TWO_Q_RE.finditer(line):
        _add_family(gate_family_counts, match.group(1))
        interaction_pairs.append(_pair(int(match.group(2)), int(match.group(3))))
        spans.append(match.span())
    for match in _CTRL_ROT_PAIR_RE.finditer(line):
        _add_family(
            gate_family_counts,
            _CTRL_ROTATION_FAMILIES.get(match.group(1), "cr1"),
        )
        interaction_pairs.append(_pair(int(match.group(2)), int(match.group(3))))
        spans.append(match.span())
    for match in _CTRL_ROT_DIRECT_RE.finditer(line):
        _add_family(gate_family_counts, match.group(1))
        interaction_pairs.append(_pair(int(match.group(2)), int(match.group(3))))
        spans.append(match.span())
    return spans


def _operation_signatures_from_code(
    code: str,
) -> list[OperationSignature]:
    signatures: list[OperationSignature] = []
    for line in code.splitlines():
        signatures.extend(_operation_signatures_from_line(line.split("#", 1)[0]))
    return signatures


def _operation_signatures_from_line(
    line: str,
) -> list[OperationSignature]:
    line_matches = _entangling_signatures_from_line(line)
    entangling_spans = [span for span, _ in line_matches]
    line_matches.extend(_single_qubit_signatures_from_line(line, entangling_spans))
    return [
        signature
        for _, signature in sorted(
            line_matches,
            key=lambda item: item[0][0],
        )
    ]


def _entangling_signatures_from_line(
    line: str,
) -> list[_SpannedOperationSignature]:
    matches: list[_SpannedOperationSignature] = []
    for match in _CTRL_LIST_RE.finditer(line):
        if match.group(1) == "x":
            matches.append((match.span(), ("ccx", _three_qubit_indices(match))))
    for match in _CTRL_PAIR_RE.finditer(line):
        family = _CTRL_PAIR_FAMILIES.get(match.group(1))
        if family is not None:
            matches.append((match.span(), (family, _two_qubit_indices(match))))
    for match in _THREE_Q_RE.finditer(line):
        matches.append(
            (
                match.span(),
                (match.group(1), _three_qubit_indices(match)),
            )
        )
    for match in _TWO_Q_RE.finditer(line):
        matches.append(
            (
                match.span(),
                (match.group(1), _two_qubit_indices(match)),
            )
        )
    return matches


def _single_qubit_signatures_from_line(
    line: str,
    entangling_spans: Sequence[tuple[int, int]],
) -> list[_SpannedOperationSignature]:
    matches: list[_SpannedOperationSignature] = []
    for pattern in (_ROT_Q_RE, _SINGLE_Q_RE):
        for match in pattern.finditer(line):
            if not _span_overlaps(match.span(), entangling_spans):
                matches.append(
                    (
                        match.span(),
                        (match.group(1), (int(match.group(2)),)),
                    )
                )
    return matches


def _two_qubit_indices(match: re.Match[str]) -> tuple[int, int]:
    return (int(match.group(2)), int(match.group(3)))


def _three_qubit_indices(
    match: re.Match[str],
) -> tuple[int, int, int]:
    return (
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def _single_gate_family_metadata(
    line: str,
    entangling_spans: Sequence[tuple[int, int]],
    gate_family_counts: dict[str, int],
) -> None:
    for match in _GATE_CALL_RE.finditer(line):
        if _span_overlaps(match.span(), entangling_spans):
            continue
        name = match.group(1)
        if name in _SINGLE_Q_FAMILIES:
            _add_family(gate_family_counts, name)


def _span_overlaps(
    span: tuple[int, int],
    spans: Sequence[tuple[int, int]],
) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def _measurement_indices_from_code(code: str) -> list[int]:
    facts = _source_facts(code)
    if facts.measurement_indices:
        return facts.measurement_indices
    return [int(match.group(1)) for match in _MEASURE_QUBIT_RE.finditer(code)]


def _has_explicit_measurements(code: str) -> bool:
    return _MEASUREMENT_CALL_RE.search(code) is not None


def _allocated_qubits_from_code(code: str) -> int | None:
    facts = _source_facts(code)
    if facts.total_allocated is not None:
        return facts.total_allocated
    total = sum(int(match.group(1)) for match in _QVECTOR_RE.finditer(code))
    total += sum(int(match.group(1)) for match in _QALLOC_RE.finditer(code))
    return total or None
