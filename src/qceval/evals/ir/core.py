"""Framework-neutral dense circuit IR and unitary construction.

This is the shared static-unitary representation used by framework converters
and legacy parity bridges. It is not an equivalence checker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

Matrix = np.ndarray
GateRepresentation = Literal["full", "controlled_base"]


@dataclass(frozen=True)
class Control:
    """A positive or open control on one wire."""

    wire: int
    value: Literal[0, 1] = 1

    def __post_init__(self) -> None:
        if self.wire < 0:
            raise ValueError("control wire must be non-negative")
        if self.value not in (0, 1):
            raise ValueError("control value must be 0 or 1")


@dataclass(frozen=True)
class Gate:
    """One unitary operation on a subset of wires."""

    matrix: Matrix
    wires: tuple[int, ...] = ()
    controls: tuple[Control, ...] = ()
    targets: tuple[int, ...] = ()
    representation: GateRepresentation = "full"
    name: str = "gate"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=complex)
        object.__setattr__(self, "matrix", matrix)
        controls = tuple(
            control if isinstance(control, Control) else Control(int(control)) for control in self.controls
        )
        object.__setattr__(self, "controls", controls)
        active_wires = _active_gate_wires(self)
        _validate_gate_wires(self, active_wires, controls)
        _validate_gate_matrix(self.name, matrix, len(active_wires))

    @classmethod
    def full(
        cls,
        matrix: Matrix,
        wires: Sequence[int],
        *,
        name: str = "gate",
    ) -> Gate:
        """Build a full-local matrix gate.

        Args:
            matrix: Unitary matrix over all active wires.
            wires: Active wire indices in matrix order.
            name: Diagnostic gate name.

        Returns:
            A validated full-local gate.

        Raises:
            ValueError: If the wires or unitary matrix are invalid.
        """
        return cls(
            matrix=np.asarray(matrix, dtype=complex),
            wires=tuple(wires),
            name=name,
        )

    @classmethod
    def controlled(
        cls,
        matrix: Matrix,
        *,
        targets: Sequence[int],
        controls: Sequence[int | Control],
        name: str = "gate",
    ) -> Gate:
        """Build a compact controlled-base gate.

        Args:
            matrix: Unitary matrix acting on the target wires.
            targets: Target wire indices in matrix order.
            controls: Control wire indices or explicit control conditions.
            name: Diagnostic gate name.

        Returns:
            A validated controlled-base gate.

        Raises:
            ValueError: If the controls, targets, or unitary matrix are
                invalid.
        """
        return cls(
            matrix=np.asarray(matrix, dtype=complex),
            targets=tuple(targets),
            controls=tuple(control if isinstance(control, Control) else Control(int(control)) for control in controls),
            representation="controlled_base",
            name=name,
        )


def _active_gate_wires(gate: Gate) -> tuple[int, ...]:
    if gate.representation == "full":
        if gate.controls or gate.targets:
            raise ValueError(f"{gate.name}: full-local gates cannot also set controls/targets")
        return gate.wires
    if gate.representation == "controlled_base":
        if gate.wires:
            raise ValueError(f"{gate.name}: controlled-base gates use targets, not wires")
        if not gate.targets:
            raise ValueError(f"{gate.name}: controlled-base gate requires at least one target")
        return gate.targets
    raise ValueError(f"{gate.name}: unknown representation {gate.representation!r}")


def _validate_gate_wires(
    gate: Gate,
    active_wires: tuple[int, ...],
    controls: tuple[Control, ...],
) -> None:
    if len(set(active_wires)) != len(active_wires):
        raise ValueError(f"{gate.name}: active wires must be unique")
    if any(wire < 0 for wire in active_wires):
        raise ValueError(f"{gate.name}: active wires must be non-negative")
    control_wires = {control.wire for control in controls}
    if len(control_wires) != len(controls):
        raise ValueError(f"{gate.name}: control wires must be unique")
    if set(gate.targets) & control_wires:
        raise ValueError(f"{gate.name}: target/control overlap")


def _validate_gate_matrix(
    name: str,
    matrix: np.ndarray,
    num_active_wires: int,
) -> None:
    dim = 1 << num_active_wires
    if matrix.shape != (dim, dim):
        raise ValueError(f"{name}: matrix shape {matrix.shape} does not match {num_active_wires} active wires")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name}: matrix must contain only finite values")
    identity = np.eye(dim, dtype=complex)
    if not np.allclose(
        matrix.conjugate().T @ matrix,
        identity,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError(f"{name}: matrix must be unitary")


@dataclass(frozen=True)
class Circuit:
    """Framework-neutral unitary circuit."""

    num_qubits: int
    gates: tuple[Gate, ...] = ()

    def __post_init__(self) -> None:
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        for gate in self.gates:
            active = gate.wires if gate.representation == "full" else gate.targets
            used = (*active, *(control.wire for control in gate.controls))
            if any(wire >= self.num_qubits for wire in used):
                raise ValueError(f"{gate.name}: wire outside {self.num_qubits}-qubit register")


def full_unitary(circuit: Circuit) -> np.ndarray:
    """Construct a dense unitary using batched local-gate application.

    Args:
        circuit: Validated framework-neutral circuit.

    Returns:
        The circuit's dense unitary in little-endian basis order.
    """
    dim = 1 << circuit.num_qubits
    unitary = np.eye(dim, dtype=np.complex128)
    for gate in circuit.gates:
        active_wires = gate.wires if gate.representation == "full" else gate.targets
        controls = tuple((control.wire, control.value) for control in gate.controls)
        indices = _gate_application_indices(
            circuit.num_qubits,
            tuple(active_wires),
            controls,
        )
        if indices.size == 0:
            continue
        blocks = unitary[indices]
        unitary[indices] = np.matmul(gate.matrix, blocks)
    return unitary


@lru_cache(maxsize=1024)
def _gate_application_indices(
    num_qubits: int,
    active_wires: tuple[int, ...],
    controls: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Return cached row groups affected by one local gate shape."""
    dim = 1 << num_qubits
    active_mask = 0
    for wire in active_wires:
        active_mask |= 1 << wire
    indices = np.arange(dim, dtype=np.intp)
    bases = indices[(indices & active_mask) == 0]
    for wire, value in controls:
        bases = bases[((bases >> wire) & 1) == value]
    offsets = np.zeros(1 << len(active_wires), dtype=np.intp)
    for sub in range(1 << len(active_wires)):
        offset = 0
        for position, wire in enumerate(active_wires):
            offset |= ((sub >> position) & 1) << wire
        offsets[sub] = offset
    grouped = bases[:, None] | offsets[None, :]
    grouped.setflags(write=False)
    return grouped


def _to_little_endian_matrix(
    matrix: np.ndarray,
    num_targets: int,
) -> np.ndarray:
    """Convert a framework's big-endian local matrix to little-endian."""
    if num_targets <= 1:
        return matrix
    dim = 1 << num_targets
    permutation = [int(format(index, f"0{num_targets}b")[::-1], 2) for index in range(dim)]
    return matrix[np.ix_(permutation, permutation)]
