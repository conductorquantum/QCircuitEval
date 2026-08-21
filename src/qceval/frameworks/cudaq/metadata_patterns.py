"""CUDA-Q source-parser patterns and gate-family constants."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Sequence
from itertools import combinations

from qceval.evals.structure import OperationSignature

_GATE_NAMES = frozenset(
    {
        "x",
        "y",
        "z",
        "h",
        "s",
        "t",
        "rx",
        "ry",
        "rz",
        "r1",
        "u1",
        "u2",
        "u3",
        "p",
        "phase",
        "cx",
        "cz",
        "swap",
        "iswap",
        "ccx",
        "cswap",
        "cr1",
        "crz",
        "crx",
        "cry",
        "cu1",
        "cp",
        "cphase",
        "mz",
        "measure",
    }
)
_DIRECT_ENTANGLING_NAMES = {
    "cx",
    "cz",
    "cy",
    "swap",
    "iswap",
    "ccx",
    "cswap",
    "cr1",
    "crz",
    "crx",
    "cry",
    "cu1",
    "cp",
    "cphase",
}
_GATE_CALL_RE = re.compile(r"\b([a-z][a-z0-9_]*)\s*(?:\.ctrl)?\(")
_MEASURE_QUBIT_RE = re.compile(r"\bmz\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\[\s*(\d+)\s*]\s*\)")
_MEASUREMENT_CALL_RE = re.compile(r"\b(?:mz|measure)\s*\(")
_QVECTOR_RE = re.compile(r"(?:cudaq\.)?qvector\(\s*(\d+)\s*\)")
_QALLOC_RE = re.compile(r"\.qalloc\(\s*(\d+)\s*\)")
_QREF = r"[A-Za-z_][A-Za-z0-9_]*\s*\[\s*(\d+)\s*\]"
_CTRL_PAIR_RE = re.compile(rf"\b([a-z])\s*\.ctrl\s*\(\s*{_QREF}\s*,\s*{_QREF}\s*\)")
_CTRL_LIST_RE = re.compile(rf"\b([a-z])\s*\.ctrl\s*\(\s*\[\s*{_QREF}\s*,\s*{_QREF}\s*\]\s*,\s*{_QREF}\s*\)")
_TWO_Q_RE = re.compile(rf"\b(cx|cz|cy|swap|iswap)\s*\(\s*{_QREF}\s*,\s*{_QREF}\s*\)")
_THREE_Q_RE = re.compile(rf"\b(ccx|cswap)\s*\(\s*{_QREF}\s*,\s*{_QREF}\s*,\s*{_QREF}\s*\)")
_CTRL_ROT_PAIR_RE = re.compile(rf"\b(rz|rx|ry|r1|u1|p|phase)\s*\.ctrl\s*\(\s*[^,()]+,\s*{_QREF}\s*,\s*{_QREF}\s*\)")
_CTRL_ROT_DIRECT_RE = re.compile(rf"\b(cr1|crz|crx|cry|cu1|cp|cphase)\s*\(\s*[^,()]+,\s*{_QREF}\s*,\s*{_QREF}\s*\)")
_SINGLE_Q_RE = re.compile(rf"\b([hxyzst])\s*\(\s*{_QREF}\s*\)")
_ROT_Q_RE = re.compile(rf"\b(r[xyz])\s*\(\s*[^,]+,\s*{_QREF}\s*\)")
_CTRL_PAIR_FAMILIES = {"x": "cx", "z": "cz", "y": "cy", "h": "ch"}
_CTRL_ROTATION_FAMILIES = {
    "rz": "crz",
    "rx": "crx",
    "ry": "cry",
    "r1": "cr1",
    "u1": "cu1",
    "p": "cp",
    "phase": "cp",
}
_PARAMETRIC_GATES = {
    "rx",
    "ry",
    "rz",
    "r1",
    "u1",
    "u2",
    "u3",
    "p",
    "phase",
}
_DIRECT_CTRL_ROTATIONS = {
    "cr1",
    "crz",
    "crx",
    "cry",
    "cu1",
    "cp",
    "cphase",
}
_SINGLE_Q_FAMILIES = {
    "h",
    "x",
    "y",
    "z",
    "s",
    "t",
    "rx",
    "ry",
    "rz",
    "r1",
    "u1",
    "u2",
    "u3",
    "p",
    "phase",
}
_INT_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}
_SpannedOperationSignature = tuple[tuple[int, int], OperationSignature]


def _add_family(counts: dict[str, int], family: str) -> None:
    counts[family] = counts.get(family, 0) + 1


def _pair(a: int, b: int) -> list[int]:
    return [min(a, b), max(a, b)]


def _all_pairs(indices: Sequence[int]) -> list[list[int]]:
    return [_pair(a, b) for a, b in combinations(sorted(indices), 2)]
