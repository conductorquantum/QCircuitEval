Python API
==========

The Python API uses the same provider, runner, framework executor, and
behavior-authoritative evaluator as the CLI. Library callers cannot opt into a
retired grader: a runtime pass still requires semantic ``verified_pass``.


Running a Benchmark
-------------------

.. code-block:: python

   from pathlib import Path

   from qceval.core.bench import Adaptor
   from qceval.core.io import write_output
   from qceval.core.runner import BenchmarkRunner
   from qceval.models import RunConfig, RunOptions
   from qceval.providers.registry import build_provider

   config = RunConfig(
       provider="smoke",
       frameworks=("qiskit",),
       source_hint=None,
       model="smoke-canonical",
       max_tasks=5,
       suites=("core",),
   )
   provider = build_provider(
       config.provider,
       model=config.model,
       config={"smoke_mode": "canonical"},
   )
   runner = BenchmarkRunner(
       config=config,
       provider=provider,
       adapter=Adaptor(),
       options=RunOptions(),
   )

   payload = runner.run()
   write_output(Path("results.json"), payload)

The payload matches :doc:`output`. ``summary["passed"]`` and
``summary["pass_rate"]`` count only behavior-verified passes.


Run Configuration
-----------------

:class:`~qceval.models.RunConfig` holds benchmark identity:

.. code-block:: python

   @dataclass(frozen=True)
   class RunConfig:
       provider: str
       frameworks: tuple[Framework, ...]
       source_hint: Path | None
       model: str | None
       max_tasks: int | None = None
       task_numbers: tuple[int, ...] | None = None
       provider_config: Mapping[str, Any] = field(default_factory=dict)
       suites: tuple[Suite, ...] = ("core",)
       samples_per_task: int = 1
       pass_k: int = 1
       max_attempts: int = 1
       feedback_max_chars: int = 2000
       feedback_policy: FeedbackPolicy = field(default_factory=FeedbackPolicy)

The supported frameworks are Qiskit, Cirq, PennyLane, and CUDA-Q. ``core`` and
``qec`` are valid suite selectors. Both suites evaluate packaged behavior
contracts; QEC contracts enumerate their finite input domains exhaustively.

Pass@K and feedback repair retain their existing validation:
``pass_k <= samples_per_task`` and multi-sample runs cannot be combined with
multi-attempt feedback runs.

:class:`~qceval.models.RunOptions` holds operational policy:

.. code-block:: python

   @dataclass(frozen=True)
   class RunOptions:
       generation_concurrency: int = 1
       evaluation_workers: int = 1
       cache_dir: Path | None = None
       resume_from: Path | None = None
       stream_to: Path | None = None
       task_timeout: float | None = None
       eval_timeout: float | None = None
       fail_fast: bool = False
       progress: bool = False
       prompt_frameworks: tuple[Framework, ...] | None = None
       regrade_frameworks: tuple[Framework, ...] | None = None
       input_from: Path | None = None
       stop_on_infrastructure_error: bool = False

Worker counts, isolation, caching, streaming, resume, selective prompt/regrade
phases, and infrastructure stop policy change execution mechanics only. They do
not change contracts, verifier routes, targets, or score authority.


Loading Tasks
-------------

Use :func:`~qceval.evals.evaluator.load_tasks` for raw bundled assets:

.. code-block:: python

   from qceval.evals.evaluator import load_tasks

   tasks = load_tasks("qiskit", suite="core")
   for task_id, task in sorted(tasks.items()):
       print(task_id, task["entry_point"])

Use :class:`~qceval.core.bench.Adaptor` for provider-facing task objects:

.. code-block:: python

   from qceval.core.bench import Adaptor

   adapter = Adaptor()
   tasks = adapter.load_tasks("qiskit", suite="core")
   print(tasks[0].prompt)

:class:`~qceval.models.QCEvalTask` exposes prompt, entry point, category, suite,
raw asset data, and the compatibility ``canonical_class`` field. That field can
support executor metadata and smoke-provider generation, but callers must not
use it as score authority.


Inspecting Behavior Contracts
-----------------------------

Load and validate a packaged registry independently of task assets:

.. code-block:: python

   from qceval.semantics.contracts import ContractRegistry, contract_hash

   registry = ContractRegistry.from_package("core")
   contract = registry.get("core", "01")

   print(contract.kind.value)
   print(contract.contract_version)
   print(contract_hash(contract))
   print([route.engine for route in contract.routing.primary])

Registries can also be loaded from a JSONL path and compared with
:meth:`qceval.semantics.contracts.ContractRegistry.diff`. This API applies to
both packaged suites; ``ContractRegistry.from_package("qec")`` loads the QEC
registry the same way.


Evaluating Code Directly
------------------------

The adapter is the stable exception-catching boundary:

.. code-block:: python

   from qceval.core.bench import Adaptor

   adapter = Adaptor()
   task = adapter.load_tasks("qiskit", suite="core")[0]
   evaluation = adapter.evaluate(task, generated_code)

   print(evaluation.compiled)
   print(evaluation.ran)
   print(evaluation.passed)
   print(evaluation.grader_details["semantic_status"])
   print(evaluation.grader_details["score_authority"])
   print(evaluation.semantic_result["contract"])

``evaluation.passed`` is true only for ``verified_pass``.
``evaluation.semantic_result`` contains the complete bounded, versioned record.

For lower-level access, construct the evaluator:

.. code-block:: python

   from qceval.evals.evaluator import build_evaluator, load_tasks

   tasks = load_tasks("qiskit", suite="core")
   evaluator = build_evaluator("qiskit", suite="core")
   task = tasks["01"]

   execution, details = evaluator.grade_code(
       task_id="01",
       code=generated_code,
       entry_point=task["entry_point"],
   )

   print(details["semantic_status"])
   print(details["behavior_verdict"])

:meth:`qceval.evals.evaluator.Evaluator.grade_code` runs the framework executor,
lowers the native return value, checks requirements, routes the contract, and
creates the semantic result. Evaluator exceptions are not caught at this lower
level.

Use :meth:`qceval.evals.evaluator.Evaluator.execute_code` only when you need raw
executor output for diagnostics:

.. code-block:: python

   execution = evaluator.execute_code(
       task_id="01",
       code=generated_code,
       entry_point=task["entry_point"],
   )

This method does not grade and its result must not be interpreted as a score.


Using a Custom Semantic Verifier
--------------------------------

``build_evaluator`` accepts an object implementing the
:class:`~qceval.semantics.integration.SemanticVerifier` protocol:

.. code-block:: python

   evaluator = build_evaluator(
       "qiskit",
       suite="core",
       semantic_verifier=my_semantic_verifier,
   )

The verifier receives a
:class:`~qceval.semantics.integration.SemanticVerificationRequest` and returns
a :class:`~qceval.semantics.verifiers.VerifierResult`. The evaluator validates
contract and target identity before projecting the status. A verifier exception
or identity mismatch becomes ``execution_error`` and fails closed.


Providers and Reports
---------------------

Provider APIs are unchanged. A provider receives
:class:`~qceval.models.ProviderRequest` and returns
:class:`~qceval.models.ProviderResponse`; it does not execute or grade code. See
:doc:`providers` for custom and built-in providers.

Use :func:`~qceval.reports.summarize` to build the same compatibility and
semantic summaries as the CLI:

.. code-block:: python

   from qceval.reports import summarize

   summary = summarize(records, run_config=config)
   print(summary["pass_rate"])
   print(summary["semantic"]["status_counts"])
   print(summary["semantic"]["coverage"])

See :doc:`output` before comparing runs with different contract, target, IR, or
verifier identities.


Writing Output
--------------

.. code-block:: python

   from pathlib import Path

   from qceval.core.io import write_output

   write_output(Path("results.json"), payload)
   write_output(Path("results.jsonl"), payload)

The writer preserves semantic result records in both formats. JSONL remains the
recommended format for streaming and resume.
