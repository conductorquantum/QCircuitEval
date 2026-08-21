from __future__ import annotations

from typing import Any

from qceval.models import ProviderRequest, ProviderResponse, QCEvalEvaluation, QCEvalTask


class SuiteStubProvider:
    name = "stub"
    trusted_metadata = True

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(code=str(request.metadata["canonical_solution"]), model=request.model)


class SuiteFailingProvider:
    name = "stub"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(code=None, model=request.model, error="failed")


class SuiteStubAdapter:
    def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
        return [
            QCEvalTask(
                task_id="01",
                framework="qiskit",
                prompt=f"{suite} prompt",
                entry_point="answer",
                category="cat",
                canonical_class={"type": "deterministic_dominant"},
                suite=suite,  # type: ignore[arg-type]
                raw={"canonical_solution": "def answer():\n    return None\n"},
            )
        ]

    def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
        return QCEvalEvaluation(compiled=True, ran=True, passed=True)

    def metadata(self) -> dict[str, Any]:
        return {"path": None}
