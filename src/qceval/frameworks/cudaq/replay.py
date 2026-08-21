"""CUDA-Q source replay: AST transforms and statevector extraction.

This module is a compatibility facade. Prefer importing from
``qceval.frameworks.cudaq.replay_transform`` or
``qceval.frameworks.cudaq.replay_simulate``.
"""

from __future__ import annotations

from qceval.frameworks.cudaq.replay_simulate import (
    _cleanup_module,
    _double_precision,
    _import_transformed_module,
    _is_cudaq_kernel,
    cudaq_kernel_unitary,
    cudaq_measured_wires,
    cudaq_num_qubits,
    simulate_basis_cudaq,
)
from qceval.frameworks.cudaq.replay_transform import (
    _KERNEL_DECORATORS,
    _KERNEL_FACTORY_NAMES,
    _MEASURE_NAMES,
    _QUBIT_ALLOC_NAMES,
    _SINGLE_ALLOC_NAMES,
    _VECTOR_ALLOC_NAMES,
    _attr_or_name,
    _bare_x_wire,
    _const_int,
    _filter_body,
    _find_builder_entry,
    _find_kernel,
    _find_kernel_factory_var,
    _has_allocation,
    _is_measurement,
    _is_method_on,
    _KernelInfo,
    _qubit_allocation,
    _resolve_kernel,
    _subscript_indices,
    _target_wire,
    _transform_source,
    _x_gate,
    parsed_measured_wires,
    parsed_num_qubits,
)

__all__ = [
    "_KERNEL_DECORATORS",
    "_KERNEL_FACTORY_NAMES",
    "_MEASURE_NAMES",
    "_QUBIT_ALLOC_NAMES",
    "_SINGLE_ALLOC_NAMES",
    "_VECTOR_ALLOC_NAMES",
    "_KernelInfo",
    "_attr_or_name",
    "_bare_x_wire",
    "_cleanup_module",
    "_const_int",
    "_double_precision",
    "_filter_body",
    "_find_builder_entry",
    "_find_kernel",
    "_find_kernel_factory_var",
    "_has_allocation",
    "_import_transformed_module",
    "_is_cudaq_kernel",
    "_is_measurement",
    "_is_method_on",
    "_qubit_allocation",
    "_resolve_kernel",
    "_subscript_indices",
    "_target_wire",
    "_transform_source",
    "_x_gate",
    "cudaq_kernel_unitary",
    "cudaq_measured_wires",
    "cudaq_num_qubits",
    "parsed_measured_wires",
    "parsed_num_qubits",
    "simulate_basis_cudaq",
]
