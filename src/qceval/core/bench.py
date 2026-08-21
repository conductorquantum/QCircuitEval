"""Load bundled QCircuitEval tasks and grade generated candidate code.

This module provides the default task adapter used by the runner.  It keeps the
benchmark self-contained by loading JSONL assets shipped in the package and by
delegating all grading to framework-specific evaluators in :mod:`qceval.evals`.
External providers only need to return candidate source code; the adapter turns
that source into a stable :class:`qceval.models.QCEvalEvaluation`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qceval.evals.evaluator import Evaluator, build_evaluator, load_tasks
from qceval.models import Framework, QCEvalEvaluation, QCEvalTask, Suite
from qceval.semantics.contracts import ContractRegistry, contract_to_dict
from qceval.serialization import to_jsonable

SUPPORTED_FRAMEWORKS: tuple[Framework, ...] = ("qiskit", "cirq", "pennylane", "cudaq")
DEFAULT_FRAMEWORKS: tuple[Framework, ...] = SUPPORTED_FRAMEWORKS


class Adaptor:
    """Default task adapter backed by bundled tasks and local graders.

    The adapter is intentionally small: it exposes benchmark tasks to providers,
    evaluates generated source code, and returns serializable metadata for run
    output.  Task and evaluator objects are cached per framework so repeated
    evaluations do not reread package assets.

    Args:
        source_hint: Optional path-like label recorded in run metadata.  The
            value is not read at runtime; it exists for compatibility with
            callers that want output to preserve the source dataset they meant
            to evaluate.
    Attributes:
        source_hint: Normalized string form of ``source_hint`` or ``None``.
    """

    def __init__(self, source_hint: str | Path | None = None) -> None:
        self.source_hint = None if source_hint is None else str(Path(source_hint).expanduser())
        self._tasks: dict[tuple[Suite, Framework], list[QCEvalTask]] = {}
        self._evaluators: dict[tuple[Suite, Framework], Evaluator] = {}
        self._metadata: dict[str, Any] | None = None

    def load_tasks(self, framework: Framework, suite: Suite = "core") -> list[QCEvalTask]:
        """Return packaged tasks for one framework.

        Args:
            framework: Framework literal to load.  Supported values are listed
                in :data:`SUPPORTED_FRAMEWORKS`.
            suite: Benchmark suite to load.

        Returns:
            Ordered list of :class:`qceval.models.QCEvalTask` objects converted
            from the bundled JSONL asset for ``framework``.

        Raises:
            ValueError: If ``framework`` is not supported.
        """
        self._check_framework(framework)
        key = (suite, framework)
        if key not in self._tasks:
            self._tasks[key] = [self._task_from_raw(framework, task) for task in load_tasks(framework, suite).values()]
            contracts = ContractRegistry.from_package(suite)
            for task in self._tasks[key]:
                _validate_runtime_prompt_hash(task, contracts)
        return self._tasks[key]

    def evaluate(self, task: QCEvalTask, code: str) -> QCEvalEvaluation:
        """Evaluate candidate code for a task.

        Unlike :class:`qceval.evals.evaluator.Evaluator`, this method catches
        syntax and runtime failures and encodes them as failed evaluations.  The
        runner can therefore treat provider output, compile failures, and grader
        failures through one result model.

        Args:
            task: Task metadata previously returned by :meth:`load_tasks`.
            code: Candidate Python source returned by a provider.

        Returns:
            Evaluation object describing compile status, execution status,
            grader result, probabilities, metadata, and any captured error.
        """
        evaluator = self._evaluator(task.suite, task.framework)
        try:
            execution, details = evaluator.grade_code(
                task_id=task.task_id,
                code=code,
                entry_point=task.entry_point,
            )
        except SyntaxError:
            return _failed_evaluation(compiled=False, error_type="InfrastructureError")
        except Exception:
            return _failed_evaluation(compiled=False, error_type="InfrastructureError")
        passed = bool(details["passed"])
        candidate_stage = execution.metadata.get("candidate_failure_stage")
        if candidate_stage in {"compile", "run"}:
            return QCEvalEvaluation(
                compiled=candidate_stage != "compile",
                ran=False,
                passed=False,
                probabilities=to_jsonable(execution.probabilities),
                execution_metadata=to_jsonable(execution.metadata),
                grader_details=to_jsonable(details),
                verified_status=str(details.get("verified_status", "execution_error")),
                semantic_result=to_jsonable(details.get("semantic_verification")),
                error=str(execution.metadata.get("candidate_error") or "candidate execution failed"),
                error_type=str(execution.metadata.get("candidate_error_type") or "CandidateError"),
            )
        return QCEvalEvaluation(
            compiled=True,
            ran=True,
            passed=passed,
            metric=_metric(details),
            metric_name=_metric_name(details),
            probabilities=to_jsonable(execution.probabilities),
            execution_metadata=to_jsonable(execution.metadata),
            grader_details=to_jsonable(details),
            verified_status=str(details.get("verified_status", "verified_pass" if passed else "verified_fail")),
            semantic_result=to_jsonable(details.get("semantic_verification")),
            error=None if passed else _public_failure_message(details),
        )

    def metadata(self) -> dict[str, Any]:
        """Return adapter provenance for run output.

        Returns:
            Mapping stored in the top-level ``qceval`` output field.  Bundled
            assets do not depend on an external path.  The package version is
            always recorded, and source checkouts also record their Git commit.
        """
        if self._metadata is None:
            commit = _checkout_commit()
            self._metadata = {
                "source": "bundled-qceval",
                "package_version": _package_version(),
                "source_hint": self.source_hint,
                "path": None,
                "branch": None,
                "commit": commit,
                "commit_status": "available" if commit is not None else "unavailable",
                "dirty": _checkout_dirty(),
            }
        return dict(self._metadata)

    def _evaluator(self, suite: Suite, framework: Framework) -> Evaluator:
        self._check_framework(framework)
        key = (suite, framework)
        if key not in self._evaluators:
            self._evaluators[key] = build_evaluator(framework, suite)
        return self._evaluators[key]

    @staticmethod
    def _check_framework(framework: Framework) -> None:
        if framework not in SUPPORTED_FRAMEWORKS:
            raise ValueError(f"framework must be one of: {', '.join(SUPPORTED_FRAMEWORKS)}")

    @staticmethod
    def _task_from_raw(framework: Framework, task: dict[str, Any]) -> QCEvalTask:
        return QCEvalTask(
            task_id=str(task["task_id"]).zfill(2),
            framework=framework,
            prompt=str(task.get("prompt") or task.get("complete_prompt") or ""),
            entry_point=str(task["entry_point"]),
            category=None if task.get("category") is None else str(task["category"]),
            canonical_class=task.get("canonical_class"),
            suite=task.get("suite", "core"),
            raw=task,
        )


def _failed_evaluation(*, compiled: bool, error_type: str) -> QCEvalEvaluation:
    return QCEvalEvaluation(
        compiled=compiled,
        ran=False,
        passed=False,
        verified_status="execution_error",
        error_type=error_type,
        error="grader adapter failed before producing a typed evaluation",
    )


def _package_version() -> str:
    """Return the installed QCircuitEval distribution version."""
    try:
        return importlib.metadata.version("qceval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _checkout_commit() -> str | None:
    """Return HEAD when this module is imported from a source checkout."""
    checkout = _source_checkout()
    if checkout is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _checkout_dirty() -> bool | None:
    """Return source-checkout dirtiness, or ``None`` outside Git checkouts."""
    checkout = _source_checkout()
    if checkout is None:
        return None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _source_checkout() -> Path | None:
    """Return the source-checkout root for this module, when available."""
    package_dir = Path(__file__).resolve().parents[1]
    checkout = package_dir.parent.parent
    if package_dir != (checkout / "src" / "qceval").resolve() or not (checkout / ".git").exists():
        return None
    return checkout


def _metric(details: dict[str, Any]) -> Any:
    if "metric" in details:
        return details["metric"]
    return None


def _metric_name(details: dict[str, Any]) -> str | None:
    if "metric_name" in details:
        return details["metric_name"]
    return None


def _validate_runtime_prompt_hash(task: QCEvalTask, contracts: ContractRegistry) -> None:
    """Fail startup when the served prompt differs from its pinned contract."""
    contract = contract_to_dict(contracts.get(task.suite, task.task_id))
    expected: str | None = None
    for requirement in contract["requirements"]:
        if requirement["kind"] != "prompt_hashes":
            continue
        value = requirement.get("value")
        if isinstance(value, Mapping) and isinstance(value.get(task.framework), str):
            expected = value[task.framework]
            break
    actual = hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
    if expected is None or actual != expected:
        raise ValueError(f"runtime prompt identity mismatch for {task.suite}/{task.framework}/{task.task_id}")


def _public_failure_message(details: Mapping[str, Any]) -> str:
    """Return a bounded failure string that excludes contract/target oracles."""
    semantic = details.get("semantic_verification")
    if isinstance(semantic, Mapping):
        status = semantic.get("status") or details.get("semantic_status") or "failed"
        reason = semantic.get("reason_code") or semantic.get("reason") or "grader_failed"
        return f"{status}: {reason}"
    status = details.get("semantic_status") or details.get("verified_status") or "failed"
    reason = details.get("reason") or details.get("error_type") or "grader_failed"
    return f"{status}: {reason}"
