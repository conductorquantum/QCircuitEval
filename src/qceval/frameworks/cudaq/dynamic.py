"""Parse CUDA-Q kernels with measurement-conditioned branches.

The legacy source replay reconstructs a fixed gate list and therefore cannot
express feed-forward.  This module parses the bounded dynamic subset —
``name = mz(q[i])`` assignments and single-level ``if``/``else`` branches over
those names — directly into Program IR measurement operations and
``ClassicalCondition``-guarded gates, exactly the representation the Qiskit
and PennyLane adapters produce for their native conditionals.
"""

from __future__ import annotations

import ast
import importlib.metadata
from typing import Any

from qceval.frameworks.cudaq.gates import (
    _CUDAQ_IGNORED_CALLS,
    _CUDAQ_MEASUREMENT_CALLS,
    _cudaq_gate_from_call,
    _subscript_indices,
)
from qceval.frameworks.cudaq.parser import _cudaq_register, _find_cudaq_kernel
from qceval.frameworks.cudaq.program import CudaqProgram
from qceval.frameworks.cudaq.values import (
    _attr_or_name,
    _cudaq_constant_bindings,
    _cudaq_registered_matrices,
)
from qceval.semantics.ir import (
    IR_VERSION,
    ClassicalCondition,
    Control,
    Operation,
    OperationKind,
    Program,
    Provenance,
)
from qceval.semantics.lowering.base import SourceMetadata
from qceval.semantics.lowering.utils import bounded_matrix_semantic_data, matrix_sha256


def has_conditional_feedback(code: str, entry_point: str) -> bool:
    """Return whether the kernel branches on a measurement result.

    Args:
        code: CUDA-Q candidate Python source.
        entry_point: Kernel-factory function name.

    Returns:
        True when an ``if`` inside the kernel tests a measurement-bound name.
    """
    try:
        tree = ast.parse(code)
        kernel = _find_cudaq_kernel(tree, entry_point)
    except (SyntaxError, ValueError, NotImplementedError):
        return False
    measured_names = {
        statement.targets[0].id
        for statement in ast.walk(kernel)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
        and _attr_or_name(statement.value.func) in _CUDAQ_MEASUREMENT_CALLS
    }
    if not measured_names:
        return False
    for node in ast.walk(kernel):
        if isinstance(node, ast.If) and measured_names & {
            name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)
        }:
            return True
    return False


def lower_dynamic_kernel(source: CudaqProgram, metadata: SourceMetadata) -> Program:
    """Lower a bounded feed-forward kernel to Program IR.

    Args:
        source: CUDA-Q source carrier with entry point and call arguments.
        metadata: Candidate source/backend diagnostics.

    Returns:
        Program IR with measurement operations and conditioned gates.

    Raises:
        NotImplementedError: For any construct outside the bounded subset.
    """
    tree = ast.parse(source.code)
    kernel = _find_cudaq_kernel(tree, source.entry_point)
    constants = _cudaq_constant_bindings(
        source.code,
        source.entry_point,
        kernel,
        call_args=source.call_args,
    )
    registered = _cudaq_registered_matrices(tree)
    register_name, num_qubits = _cudaq_register(kernel)
    state = _LoweringState(constants, registered, register_name, num_qubits)
    for statement in kernel.body:
        state.lower_statement(statement, condition=None)
    return Program(
        IR_VERSION,
        num_qubits,
        state.next_bit,
        tuple(state.operations),
        None,
        tuple(reversed(range(state.next_bit))),
        Provenance(
            "cudaq",
            importlib.metadata.version("cudaq"),
            source_hash=metadata.source_hash,
            backend=metadata.backend,
        ),
    )


class _LoweringState:
    """Sequential statement lowering with measurement-bit bookkeeping."""

    def __init__(
        self,
        constants: dict[str, Any],
        registered: dict[str, Any],
        register_name: str,
        num_qubits: int,
    ) -> None:
        self.constants = constants
        self.registered = registered
        self.register_name = register_name
        self.num_qubits = num_qubits
        self.operations: list[Operation] = []
        self.bit_names: dict[str, int] = {}
        self.next_bit = 0

    def lower_statement(self, statement: ast.stmt, condition: ClassicalCondition | None) -> None:
        if isinstance(statement, ast.Assign):
            self._lower_assign(statement, condition)
            return
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            self._lower_call(statement.value, condition)
            return
        if isinstance(statement, ast.If):
            if condition is not None:
                raise NotImplementedError("nested measurement-conditioned branches are unsupported")
            self._lower_branch(statement)
            return
        if isinstance(statement, ast.Pass):
            return
        raise NotImplementedError(f"unsupported dynamic CUDA-Q statement: {type(statement).__name__}")

    def _lower_assign(self, statement: ast.Assign, condition: ClassicalCondition | None) -> None:
        if (
            len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and _attr_or_name(statement.value.func) in _CUDAQ_MEASUREMENT_CALLS
        ):
            if condition is not None:
                raise NotImplementedError("conditioned measurements are unsupported")
            self.bit_names[statement.targets[0].id] = self._measure(statement.value)
            return
        if isinstance(statement.value, ast.Call) and _attr_or_name(statement.value.func) in _CUDAQ_IGNORED_CALLS:
            return
        # Classical parameter plumbing (already captured in constants).
        if not any(
            isinstance(node, ast.Call) and _attr_or_name(node.func) not in {"float", "float32", "float64"}
            for node in ast.walk(statement.value)
        ):
            return
        raise NotImplementedError("unsupported dynamic CUDA-Q assignment")

    def _lower_call(self, call: ast.Call, condition: ClassicalCondition | None) -> None:
        name = _attr_or_name(call.func)
        if name in _CUDAQ_MEASUREMENT_CALLS:
            if condition is not None:
                raise NotImplementedError("conditioned measurements are unsupported")
            self._measure(call)
            return
        if name == "reset":
            self._lower_reset(call, condition)
            return
        gate = _cudaq_gate_from_call(
            call,
            constants=self.constants,
            registered=self.registered,
            register_name=self.register_name,
            num_qubits=self.num_qubits,
        )
        if gate is None:
            return
        gate_name, matrix, targets, controls = gate
        try:
            semantic_data = bounded_matrix_semantic_data(matrix, wire_order="little_endian")
        except ValueError:
            semantic_data = (("matrix_sha256", matrix_sha256(matrix)),)
        expanded_targets = [(target,) for target in targets] if not controls and matrix.shape == (2, 2) else [targets]
        for target_group in expanded_targets:
            self.operations.append(
                Operation(
                    OperationKind.GATE,
                    gate_name.lower(),
                    quantum_wires=tuple(target_group),
                    controls=tuple(Control(wire, 1) for wire in controls),
                    semantic_data=semantic_data,
                    condition=condition,
                    source_location=f"dynamic[{len(self.operations)}]",
                )
            )

    def _lower_reset(self, call: ast.Call, condition: ClassicalCondition | None) -> None:
        if condition is not None:
            raise NotImplementedError("conditioned resets are unsupported")
        wires = [wire for argument in call.args for wire in _subscript_indices(argument)]
        if len(wires) != 1:
            raise NotImplementedError("dynamic resets must target a single resolved qubit")
        self.operations.append(
            Operation(
                OperationKind.RESET,
                "reset",
                quantum_wires=(wires[0],),
                source_location=f"dynamic[{len(self.operations)}]",
            )
        )

    def _lower_branch(self, statement: ast.If) -> None:
        bit, value = self._branch_condition(statement.test)
        for inner in statement.body:
            self.lower_statement(inner, ClassicalCondition((bit,), value))
        for inner in statement.orelse:
            self.lower_statement(inner, ClassicalCondition((bit,), 1 - value))

    def _branch_condition(self, test: ast.expr) -> tuple[int, int]:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            bit, value = self._branch_condition(test.operand)
            return bit, 1 - value
        if isinstance(test, ast.Name) and test.id in self.bit_names:
            return self.bit_names[test.id], 1
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq | ast.NotEq)
            and isinstance(test.left, ast.Name)
            and test.left.id in self.bit_names
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value in {0, 1}
        ):
            expected = int(bool(test.comparators[0].value))
            if isinstance(test.ops[0], ast.NotEq):
                expected = 1 - expected
            return self.bit_names[test.left.id], expected
        raise NotImplementedError("unsupported measurement branch condition")

    def _measure(self, call: ast.Call) -> int:
        wires: list[int] = []
        for argument in call.args:
            wires.extend(_subscript_indices(argument))
        if len(wires) != 1:
            raise NotImplementedError("dynamic measurements must target a single resolved qubit")
        bit = self.next_bit
        self.next_bit += 1
        self.operations.append(
            Operation(
                OperationKind.MEASUREMENT,
                "mz",
                quantum_wires=(wires[0],),
                classical_bits=(bit,),
                semantic_data=(("basis", "z"),),
                source_location=f"measurement[{bit}]",
            )
        )
        return bit
