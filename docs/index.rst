QCircuitEval
============

QCircuitEval (``qceval``) is a beta, self-contained benchmark harness for
quantum circuit code generation. Scores and APIs may still change between
published revisions. It ships prompts and task assets, framework-specific
executors, versioned behavior contracts and targets, a verifier portfolio, and
machine-readable reporting.

One ``qceval run`` entry point handles a single framework/model configuration,
all seven named reasoning efforts, or a registry-defined Pass@1 matrix. Matrix
jobs remain independent scores and are indexed by a versioned sweep manifest.

Runtime grading is behavior-authoritative and fail closed. Only a
``verified_pass`` semantic result passes. Behavioral mismatches are
``semantic_fail``; inability to execute or decide verification is
``execution_error``; deterministic limits are ``resource_limit``. No retired
grader or fallback route is consulted.

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   cli
   providers
   evaluation
   grader
   output
   error_taxonomy
   semantic_grading
   production_inventory
   leaderboard
   contributing
   api


What Defines a Score
--------------------

The prompt and framework task asset still define what code a provider sees and
which entry point the unchanged framework executor calls. A separate behavior
contract defines what the returned program must do. Contracts are keyed by
``(suite, task_id)`` and identify:

* the public signature and named input, output, ancilla, and work systems;
* the semantic object to verify, such as a state, total unitary, isometry,
  channel, distribution, classical mapping, instrument, or objective;
* observation, phase, parameter, approximation, and resource policies;
* an independently versioned target artifact; and
* one validated primary verifier route.

Contracts and targets are content-hashed. Changing a contract, target,
intermediate-representation version, verifier release, or relevant environment
creates a distinct result identity rather than silently changing an old score.

The bundled task field ``canonical_class`` remains available in task and
provider metadata for executor compatibility and deterministic smoke-provider
code generation. It is not score authority.


Evaluation Architecture
-----------------------

A benchmark record moves through these stages:

1. **Task loading.** :class:`~qceval.core.bench.Adaptor` reads the selected
   framework's bundled prompt and task asset.
2. **Generation.** The runner sends the prompt, framework, task ID, and entry
   point to a provider. Providers return code or a provider error.
3. **Framework execution.** The existing Qiskit, Cirq, PennyLane, or CUDA-Q
   executor runs the generated entry point and captures its native circuit,
   probabilities, and execution metadata.
4. **Semantic lowering.** A framework adapter converts the returned program to
   the versioned, framework-neutral Program IR. Constructs that cannot be
   represented faithfully and inspection failures become ``execution_error``;
   deterministic limits become ``resource_limit``.
5. **Verification.** The contract routes the IR and observed execution through
   its single validated primary engine.
6. **Reporting.** The evaluator records the semantic status, bounded evidence,
   version identities, resources, environment, and ``score_authority:
   "behavior"``.

The provider protocol, task routing, multiprocessing, response cache, JSONL
resume, feedback protocol, and framework executors retain their existing roles.
They can change throughput or diagnostics, but not score authority.


Fail-Closed Statuses
--------------------

Semantic verification has four statuses:

``verified_pass``
    The routed verifier supplied sufficient evidence that the candidate
    satisfies the contract. This is the only passing status.

``semantic_fail``
    The verifier decisively found a contracted behavioral mismatch.

``execution_error``
    Candidate execution, faithful lowering, verifier processing, uncertainty
    resolution, route validation, or result-identity validation failed.

``resource_limit``
    Contract limits or runtime limits prevented verification.

The last three statuses all project to ``passed=False``. Detailed reason codes
preserve whether an execution error originated in lowering, materialization,
symbolic proof, route configuration, or a numerical uncertainty band.

Benchmark runner statuses additionally include ``generated`` (prompt-only
selective runs) and ``infrastructure_error`` (provider or worker infrastructure
boundaries). Those are runner outcomes, not semantic verifier statuses.


Coverage and Support
--------------------

The package includes behavior contracts for both the ``core`` and ``qec``
suites. Framework lowering and verifier support is capability-dependent: a
packaged contract does not guarantee that every candidate construct or
requested engine can be processed. Capability gaps are reported as
``execution_error`` and over-limit cases as ``resource_limit``.

QEC contracts declare finite exhaustive input domains. The evaluator executes
every declared input, verifies each exact distribution or state against an
independently derived stabilizer target, and enforces Program IR construction
requirements such as required interactions, argument-conditioned error gates,
and controlled corrections. Core and QEC share fail-closed statuses and
behavior score authority, but reports should keep their suite rates separate.


Framework Executors
-------------------

The framework executors continue to provide the runtime observations needed by
semantic verification:

* **Qiskit** executes ``QuantumCircuit`` results, extracts statevector or seeded
  fallback probabilities, and records circuit metadata.
* **Cirq** executes ``cirq.Circuit`` results, ignores terminal measurements for
  statevector extraction where appropriate, and records native metadata.
* **PennyLane** captures a QNode tape and its operations, measurements, and
  observed probabilities.
* **CUDA-Q** executes kernels through ``cudaq.get_state`` or seeded sampling
  fallback and records source-derived and runtime metadata.

Executor probability and unitary helpers may support execution, diagnostics, or
semantic engine materialization. They are not independent legacy score paths.
See :doc:`evaluation` for the detailed data flow.


Where to Go Next
----------------

* :doc:`quickstart` covers installation, a bounded run, and local regrade.
* :doc:`cli` documents run, model-registry matrices, and contract-registry commands.
* :doc:`providers` describes provider requests and responses.
* :doc:`evaluation` explains executors, lowering, routing, and support limits.
* :doc:`grader` documents contracts, engines, and invoking the grader locally.
* :doc:`output` documents semantic result records and aggregate rates.
* :doc:`error_taxonomy` defines the seven evidence-based error axes.
* :doc:`semantic_grading` summarizes status and version policy.
* :doc:`contributing` covers pull requests, the code of conduct, and grader contributions.
* :doc:`api` shows the supported Python interfaces.
