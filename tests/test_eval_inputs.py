from __future__ import annotations

import numpy as np

from qceval.evals.inputs import TASK6_DEFAULT_WITNESS, global_inputs, task6_statevector, task6_witness_state


def test_global_inputs_build_framework_specific_task6() -> None:
    # Arrange
    frameworks = ("qiskit", "cirq", "pennylane", "cudaq")

    # Act
    inputs = [global_inputs(framework) for framework in frameworks]

    # Assert
    assert all("06" in item for item in inputs)
    assert all(np.isclose(item["42"][0], (25 * np.pi) / 54) for item in inputs)
    # CUDA-Q kernels cannot receive a foreign circuit, so the unknown state is
    # provided as an explicit statevector matching the default witness angles.
    cudaq_state = global_inputs("cudaq")["06"]
    theta, _phi = TASK6_DEFAULT_WITNESS
    assert np.isclose(np.abs(cudaq_state[0]) ** 2, np.cos(theta / 2.0) ** 2)
    assert np.isclose(np.vdot(cudaq_state, cudaq_state).real, 1.0)


def test_task6_witness_states_are_normalized_and_distinct() -> None:
    # Arrange: the three declared diagnostic witness angles from the contract.
    points = (
        (2.0943951023931953, 0.6283185307179586),
        (1.0471975511965976, 2.443460952792061),
        (2.6, 4.2),
    )

    # Act
    states = [task6_statevector(theta, phi) for theta, phi in points]

    # Assert: unit norm and pairwise-distinct |<0|psi>|^2, so a hardcoded
    # distribution cannot satisfy every replayed case.
    overlaps = [float(np.abs(state[0]) ** 2) for state in states]
    assert all(np.isclose(np.vdot(state, state).real, 1.0) for state in states)
    assert len({round(overlap, 6) for overlap in overlaps}) == len(points)


def test_task6_witness_state_builds_each_framework_representation() -> None:
    theta, phi = TASK6_DEFAULT_WITNESS
    expected = task6_statevector(theta, phi)

    qiskit_input = task6_witness_state("qiskit", theta, phi)
    cirq_input = task6_witness_state("cirq", theta, phi)
    pennylane_input = task6_witness_state("pennylane", theta, phi)
    cudaq_input = task6_witness_state("cudaq", theta, phi)

    from qiskit.quantum_info import Statevector

    assert np.allclose(np.asarray(Statevector.from_instruction(qiskit_input).data), expected)
    import cirq

    assert np.allclose(cirq.final_state_vector(cirq_input, dtype=np.complex128), expected)
    assert np.allclose(np.asarray(pennylane_input()).reshape(-1), expected)
    assert np.allclose(cudaq_input, expected)
