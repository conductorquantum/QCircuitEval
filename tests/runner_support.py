from __future__ import annotations

import time
from typing import Any

from qceval.models import ProviderRequest, ProviderResponse, QCEvalEvaluation, QCEvalTask


class StubProvider:
    name = "stub"
    trusted_metadata = True

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(code=str(request.metadata["canonical_solution"]), model=request.model)


class StubFailingProvider:
    name = "stub"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(code=None, model=request.model, error="failed")


class SlowProvider:
    name = "slow"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        time.sleep(5)
        return ProviderResponse(code="def answer():\n    return None\n", model=request.model)


class RepairProvider:
    name = "repair"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if request.attempt_index == 0:
            return ProviderResponse(code="bad", model=request.model)
        semantic_feedback = "Previous code ran but did not satisfy the task checks."
        if any(semantic_feedback in message.content for message in request.messages):
            return ProviderResponse(code="good", model=request.model)
        return ProviderResponse(code="still bad", model=request.model)


class StubAdapter:
    evaluation = QCEvalEvaluation(compiled=True, ran=True, passed=True, metric=0.0)

    def load_tasks(self, framework: str, suite: str = "core") -> list[QCEvalTask]:
        return [
            QCEvalTask(
                task_id="01",
                framework="qiskit",
                prompt="p",
                entry_point="answer",
                category="cat",
                canonical_class={"type": "exact_distribution"},
                suite=suite,  # type: ignore[arg-type]
                raw={"canonical_solution": "code"},
            )
        ]

    def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
        return self.evaluation

    def metadata(self) -> dict[str, Any]:
        return {"path": None, "branch": "main", "commit": "abc"}
