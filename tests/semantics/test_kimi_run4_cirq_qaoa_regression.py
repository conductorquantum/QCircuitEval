"""Regression for a Cirq QAOA candidate rejected during the kimi-run4 audit."""

from __future__ import annotations

from qceval.evals.evaluator import build_evaluator

_CODE = """\
import cirq

def qaoa_maxcut_ansatz(G, beta, gamma):
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit()

    circuit.append(cirq.H.on_each(*qubits))

    for k in range(5):
        for i, j in G.edges():
            circuit.append([
                cirq.CNOT(qubits[i], qubits[j]),
                cirq.Rz(rads=2 * gamma[k]).on(qubits[j]),
                cirq.CNOT(qubits[i], qubits[j]),
            ])
        circuit.append(cirq.Rx(rads=2 * beta[k]).on_each(*qubits))

    circuit.append(cirq.measure(qubits[4], qubits[3], qubits[2], qubits[1], qubits[0], key='m'))

    return circuit
"""


def test_cirq_qaoa_rotation_constructor_spelling_is_recognized() -> None:
    _, details = build_evaluator("cirq", suite="core").grade_code(
        task_id="04",
        code=_CODE,
        entry_point="qaoa_maxcut_ansatz",
    )
    assert details["passed"] is True, details.get("reason")
