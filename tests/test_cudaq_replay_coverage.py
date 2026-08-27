"""CUDA-Q source-transform and statevector replay coverage."""

from __future__ import annotations

import pytest

pytest.importorskip("cudaq")


def test_strip_survives_kept_x_before_operand_prep() -> None:
    # Regression: the leading-prep layer is the whole prefix of bare
    # single-qubit X gates. A kept X on a non-operand wire (e.g. a baked
    # carry-in) must not end stripping, or the operand-prep X gates behind it
    # survive and every swept input is XORed with the baked demonstration
    # operands (the Qadd task-22 failure mode: 64/64 wrong_function).
    import numpy as np

    from qceval.frameworks.cudaq.replay import simulate_basis_cudaq

    code = (
        "import cudaq\n\n\n"
        "def prep():\n"
        "    @cudaq.kernel\n"
        "    def kernel():\n"
        "        q = cudaq.qvector(3)\n"
        "        x(q[0])\n"  # kept: wire 0 is not an operand wire
        "        x(q[1])\n"  # operand prep: must be stripped
        "        x(q[2])\n"  # operand prep: must be stripped
        "    return kernel\n"
    )
    statevector = simulate_basis_cudaq(code, "prep", prep={}, strip_leading_x_on={1, 2})
    probs = np.abs(np.asarray(statevector, dtype=complex)) ** 2
    # Only the kept x(q[0]) survives: state |001> (little-endian index 1).
    assert probs[1] == pytest.approx(1.0, abs=1e-6)


def test_simulate_basis_cudaq_bare_kernel_entry_point() -> None:
    # Regression: a bare ``@cudaq.kernel`` exposed directly as the entry point
    # (instead of a factory that returns a kernel) must be replayable by the
    # state/operator oracle. Previously ``simulate_basis_cudaq`` *called* the
    # entry point, got ``None`` for a bare kernel, and ``cudaq.get_state(None)``
    # raised ``'NoneType' object has no attribute 'compile'`` -- making the
    # functional oracle inconclusive and failing a correct circuit as
    # ``wrong_state``.
    import numpy as np

    from qceval.frameworks.cudaq.replay import simulate_basis_cudaq

    code = (
        "import cudaq\n\n\n@cudaq.kernel\ndef bell():\n    q = cudaq.qvector(2)\n    h(q[0])\n    x.ctrl(q[0], q[1])\n"
    )
    statevector = simulate_basis_cudaq(code, "bell", prep={}, strip_leading_x_on=set())
    probs = np.abs(np.asarray(statevector, dtype=complex)) ** 2
    # Bell state |Phi+>: equal support on |00> and |11> (indices 0 and 3).
    assert probs[0] == pytest.approx(0.5, abs=1e-6)
    assert probs[3] == pytest.approx(0.5, abs=1e-6)
    assert probs[1] + probs[2] == pytest.approx(0.0, abs=1e-6)


def test_replay_strips_measurements_nested_in_loops() -> None:
    # Regression (audit M13): the transform only stripped top-level measurement
    # statements. A Bell kernel measured inside a loop collapsed the state, so
    # ``cudaq.get_state`` reported one deterministic branch as the exact
    # statevector instead of the 50/50 superposition.
    import numpy as np

    from qceval.frameworks.cudaq.replay import simulate_basis_cudaq

    code = (
        "import cudaq\n\n\n"
        "@cudaq.kernel\n"
        "def bell():\n"
        "    q = cudaq.qvector(2)\n"
        "    h(q[0])\n"
        "    x.ctrl(q[0], q[1])\n"
        "    for i in range(2):\n"
        "        mz(q[i])\n"
    )
    statevector = simulate_basis_cudaq(code, "bell", prep={}, strip_leading_x_on=set())
    probs = np.abs(np.asarray(statevector, dtype=complex)) ** 2
    assert probs[0] == pytest.approx(0.5, abs=1e-6)
    assert probs[3] == pytest.approx(0.5, abs=1e-6)


def test_replay_rejects_measurements_whose_result_is_consumed() -> None:
    # A measurement feeding classical logic cannot be stripped without changing
    # semantics; replay must fail closed instead of returning a collapsed branch.
    from qceval.frameworks.cudaq.replay import _transform_source

    code = (
        "import cudaq\n\n\n"
        "@cudaq.kernel\n"
        "def kernel():\n"
        "    q = cudaq.qvector(2)\n"
        "    h(q[0])\n"
        "    b = mz(q[0])\n"
        "    if b:\n"
        "        x(q[1])\n"
    )
    with pytest.raises(ValueError, match="dynamic simulation"):
        _transform_source(code, prep={}, strip_leading_x_on=set())
