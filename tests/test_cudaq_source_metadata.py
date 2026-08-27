"""Source-level CUDA-Q metadata extraction (no kernel execution).

Covers the register resolution added for slice aliases, parameter-default
register sizes, and whole-register measurement, plus the measured-wire
resolution used by the operator oracle.
"""

from __future__ import annotations

from qceval.frameworks.cudaq.metadata import (
    _allocated_qubits_from_code,
    _operation_metadata_from_code,
)
from qceval.frameworks.cudaq.replay import cudaq_measured_wires

# Whole-register measurement: ``mz(q)`` must resolve to every wire of ``q``.
WHOLE_REGISTER = """
import cudaq

def kernel():
    q = cudaq.qvector(4)
    h(q[0])
    cx(q[0], q[1])
    mz(q)
"""

# Sliced sub-registers: ``x = q[0:3]`` / ``y = q[3:6]`` then ``mz(x)``.
SLICE_ALIAS = """
import cudaq

def Simon():
    q = cudaq.qvector(6)
    x = q[0:3]
    y = q[3:6]
    h(x)
    cx(x[0], y[0])
    h(x)
    mz(x)
"""

# Register sized by a function-parameter default (``n_count=3``).
PARAM_DEFAULT = """
import cudaq

def qpe(n_count=3):
    counting = cudaq.qvector(n_count)
    target = cudaq.qvector(2)
    h(counting)
    mz(counting)
"""


def test_whole_register_measurement_resolves_all_wires() -> None:
    meta = _operation_metadata_from_code(WHOLE_REGISTER)
    assert meta["measurement_count"] == 4
    assert sorted(meta["measurement_qubits"]) == [0, 1, 2, 3]
    assert cudaq_measured_wires(WHOLE_REGISTER) == [0, 1, 2, 3]


def test_slice_alias_resolves_subregister() -> None:
    meta = _operation_metadata_from_code(SLICE_ALIAS)
    assert _allocated_qubits_from_code(SLICE_ALIAS) == 6
    assert meta["measurement_count"] == 3
    assert sorted(meta["measurement_qubits"]) == [0, 1, 2]


def test_parameter_default_register_size_resolves() -> None:
    assert _allocated_qubits_from_code(PARAM_DEFAULT) == 5
    meta = _operation_metadata_from_code(PARAM_DEFAULT)
    assert meta["measurement_count"] == 3
    assert sorted(meta["measurement_qubits"]) == [0, 1, 2]
