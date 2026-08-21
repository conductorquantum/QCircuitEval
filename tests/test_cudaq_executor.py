from __future__ import annotations

import pytest

from qceval.frameworks.cudaq import counts as cudaq_counts
from qceval.frameworks.cudaq import execute_cudaq_task


def test_cudaq_rejects_nonterminating_builder_register_iteration() -> None:
    code = """
import cudaq

def answer():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)
    for qubit in qubits:
        kernel.h(qubit)
    return kernel
"""

    with pytest.raises(TypeError, match="direct iteration does not terminate"):
        execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})


def test_cudaq_executes_kernel_x_returns_one_state() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq

def answer():
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        x(q[0])

    return kernel
"""

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.probabilities == [0.0, 1.0]
    assert result.metadata["probability_method"] == "statevector"


def test_cudaq_bare_kernel_entry_point_executes() -> None:
    # Regression: candidates that expose a bare ``@cudaq.kernel`` directly as the
    # entry point (instead of a factory returning a kernel) used to be *called*
    # by the sandbox, which returns ``None`` and raised
    # ``TypeError: ... got NoneType``. The normal execution path must detect the
    # kernel and run it like the case-table path already does.
    pytest.importorskip("cudaq")
    code = """
import cudaq


@cudaq.kernel
def answer():
    q = cudaq.qvector(1)
    x(q[0])


kernel = answer
"""

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.probabilities == [0.0, 1.0]


def test_cudaq_executes_bell_kernel_returns_support_on_00_and_11() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq

def answer():
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])

    return kernel
"""

    # Act
    result = execute_cudaq_task(task_id="16", code=code, entry_point="answer", inputs={})

    # Assert
    support = {format(index, "02b") for index, probability in enumerate(result.probabilities) if probability > 0.01}
    assert support == {"00", "11"}
    assert result.probabilities[0] == pytest.approx(0.5)
    assert result.probabilities[3] == pytest.approx(0.5)


def test_cudaq_state_order_uses_asymmetric_fixture() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = """
import cudaq

def answer():
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        x(q[1])

    return kernel
"""

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.probabilities == [0.0, 0.0, 1.0, 0.0]


def test_cudaq_accepts_counts_dict_return() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = "def answer():\n    return {'00': 1, '11': 3}\n"

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.probabilities == [0.25, 0.0, 0.0, 0.75]
    assert result.metadata["probability_method"] == "returned_counts"


def test_cudaq_accepts_probability_array_return() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = "import numpy as np\ndef answer():\n    return np.array([0.25, 0.75])\n"

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.probabilities == [0.25, 0.75]
    assert result.metadata["probability_method"] == "returned_probabilities"
    assert result.metadata["measurement_count"] == 0


def test_cudaq_accepts_unitary_array_return() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = "import numpy as np\ndef answer():\n    return np.eye(2, dtype=complex)\n"

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.unitary is not None
    assert result.metadata["probability_method"] == "returned_unitary"
    assert result.probabilities == [1.0, 0.0]


def test_cudaq_rejects_bad_return_type_with_typeerror() -> None:
    # Arrange
    pytest.importorskip("cudaq")
    code = "def answer():\n    return 1\n"

    # Act
    with pytest.raises(TypeError) as exc:
        execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert "Expected CUDA-Q kernel" in str(exc.value)


def test_cudaq_counts_normalization_rejects_invalid() -> None:
    # Arrange
    pytest.importorskip("cudaq")

    # Act
    with pytest.raises(ValueError) as empty:
        cudaq_counts._cudaq_counts_to_probabilities({})
    with pytest.raises(ValueError) as unequal:
        cudaq_counts._cudaq_counts_to_probabilities({"0": 1, "11": 1})

    # Assert
    assert "counts dictionary is empty" in str(empty.value)
    assert "counts bitstrings must have equal length" in str(unequal.value)


def test_cudaq_sample_fallback_path_records_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    cudaq = pytest.importorskip("cudaq")
    code = """
import cudaq

def answer():
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        x(q[0])

    return kernel
"""
    monkeypatch.setattr(cudaq, "get_state", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")))

    # Act
    result = execute_cudaq_task(task_id="01", code=code, entry_point="answer", inputs={})

    # Assert
    assert result.metadata["probability_method"] == "sample_fallback"
    assert result.metadata["shots_count"] == 8192
    assert result.metadata["seed"] == 42
    assert result.metadata["statevector_error"]
    assert result.probabilities[1] == pytest.approx(1.0, abs=0.05)


def test_cudaq_terminal_measurement_replay_handles_closed_over_factory_inputs() -> None:
    # Regression: task inputs are consumed by the Python factory, then closed over
    # in the returned zero-argument measured kernel. Exact replay must bind the
    # factory input without forwarding it to the zero-argument kernel.
    pytest.importorskip("cudaq")
    code = """
import cudaq

def answer(flag=None):
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        if flag:
            x(q[0])
        mz(q[0])

    return kernel
"""

    # Act
    result = execute_cudaq_task(
        task_id="06",
        code=code,
        entry_point="answer",
        inputs={"06": True},
        call_args=(True,),
    )

    # Assert
    assert result.metadata["probability_method"] == "statevector_replay"
    assert result.metadata["kernel_argument_count"] == 1
    assert result.probabilities[1] == pytest.approx(1.0, abs=0.05)


def test_double_precision_target_pins_state_extraction_and_restores() -> None:
    # GPU targets such as ``nvidia`` are single precision; exact statevector
    # grading must run on ``qpp-cpu`` and restore the ambient target after.
    from qceval.frameworks.cudaq.runtime import _double_precision_target

    class _FakeCudaq:
        def __init__(self, name: str, fail_set: bool = False) -> None:
            self.name = name
            self.fail_set = fail_set
            self.calls: list[str] = []

        def get_target(self):
            return type("Target", (), {"name": self.name})()

        def set_target(self, name: str) -> None:
            self.calls.append(name)
            if self.fail_set:
                raise RuntimeError("no such target")
            self.name = name

        def reset_target(self) -> None:
            self.calls.append("<reset>")

    gpu = _FakeCudaq("nvidia")
    with _double_precision_target(gpu):
        assert gpu.name == "qpp-cpu"
    assert gpu.calls == ["qpp-cpu", "nvidia"]
    assert gpu.name == "nvidia"

    already_double = _FakeCudaq("qpp-cpu")
    with _double_precision_target(already_double):
        pass
    assert already_double.calls == []

    broken = _FakeCudaq("nvidia", fail_set=True)
    with _double_precision_target(broken):
        assert broken.name == "nvidia"
    assert broken.calls == ["qpp-cpu"]
