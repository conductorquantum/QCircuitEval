"""CUDA-Q AST constant evaluation and intrinsic gate matrices.

This module is a compatibility facade. Prefer importing from
``qceval.frameworks.cudaq.constfold`` or ``qceval.frameworks.cudaq.matrices``.
"""

from __future__ import annotations

from qceval.frameworks.cudaq.constfold import (
    _CONSTANT_MATH_CALLS,
    _FLOAT_COMPARISONS,
    _apply_float_operator,
    _apply_int_operator,
    _attr_or_name,
    _bindable_argument,
    _bound_none_comparison,
    _collect_constant_assignments,
    _const_bool,
    _const_call_float,
    _const_constant,
    _const_float,
    _const_ifexp_float,
    _const_int,
    _const_int_expr,
    _const_sequence,
    _constant_subscript_float,
    _cudaq_constant_bindings,
    _pi_constant,
)
from qceval.frameworks.cudaq.matrices import (
    _CUDAQ_MATRICES,
    _MAX_REGISTERED_DIMENSION,
    _SQRT2_INV,
    _cudaq_registered_matrices,
    _rotation_matrix,
    _static_numeric_binop,
    _static_numeric_call,
    _static_numeric_value,
)

__all__ = [
    "_CONSTANT_MATH_CALLS",
    "_CUDAQ_MATRICES",
    "_FLOAT_COMPARISONS",
    "_MAX_REGISTERED_DIMENSION",
    "_SQRT2_INV",
    "_apply_float_operator",
    "_apply_int_operator",
    "_attr_or_name",
    "_bindable_argument",
    "_bound_none_comparison",
    "_collect_constant_assignments",
    "_const_bool",
    "_const_call_float",
    "_const_constant",
    "_const_float",
    "_const_ifexp_float",
    "_const_int",
    "_const_int_expr",
    "_const_sequence",
    "_constant_subscript_float",
    "_cudaq_constant_bindings",
    "_cudaq_registered_matrices",
    "_pi_constant",
    "_rotation_matrix",
    "_static_numeric_binop",
    "_static_numeric_call",
    "_static_numeric_value",
]
