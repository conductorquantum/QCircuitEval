Evaluation Pipeline
===================

QCircuitEval executes generated code with its existing framework executors,
then grades only against a versioned behavior contract. The behavior result is
authoritative. Only ``verified_pass`` passes; no older predicate grader,
reference distribution, or unitary comparison is a fallback score path.


Task Assets and Contracts
-------------------------

The :class:`~qceval.core.bench.Adaptor` loads framework task assets through
:func:`~qceval.evals.evaluator.load_tasks` and creates a
:class:`~qceval.evals.evaluator.Evaluator`. The assets remain the source for:

* the prompt sent to the provider;
* ``task_id``, ``entry_point``, category, and suite;
* canonical source used by the smoke provider where available; and
* executor compatibility metadata.

The ``canonical_class`` asset field may still be exposed through
:class:`~qceval.models.QCEvalTask` and provider metadata. Executors also use a
small amount of this metadata, such as compact output-qubit selection. For
both suites, the field does not determine runtime correctness. QEC case tables
remain compatibility fixtures used to audit migration parity.

Behavior contracts are loaded independently from
``qceval.assets.contracts.<suite>.jsonl`` by
:class:`~qceval.semantics.contracts.ContractRegistry`. Each validated contract
contains a versioned signature, semantic kind, systems, observations, phase and
ancilla policies, parameter completeness, approximation policy, target
identity, verifier routes, resource limits, hard requirements, and
non-authoritative diagnostics.

The package provides independent ``core`` and ``qec`` contract registries.
There is one QEC contract per prompt, shared by Qiskit, Cirq, PennyLane, and
CUDA-Q. Framework-specific return and observation conventions are fields
inside that shared contract rather than separate scoring specifications.


QEC Semantic Contracts
----------------------

Parameterized QEC contracts enumerate the complete prompt domain, including the
no-error case and every permitted single-error location. The evaluator invokes
the same candidate entry point for every binding and returns ``verified_pass``
only when every binding passes. Syndrome, correction, and logical-gate tasks
use exact Hellinger-infidelity comparison against independently generated
GF(2)/stabilizer targets. Shor and Steane encoders use phase-sensitive exact
state comparison, so a wrong-sign codeword cannot pass merely because it has
the expected measurement support.

Prompt-derived hard requirements additionally verify terminal observation,
minimum register size, required stabilizer interactions, prohibited simulator
or decoder shortcuts, and source evidence that the requested Pauli error is
inserted as an argument-conditioned physical gate. Exact Program IR simulation
is used for CUDA-Q distribution evidence to avoid backend ordering and sampling
effects. QEC details retain ``num_cases`` for compatibility; versioned semantic
evidence is authoritative and ``score_authority`` remains ``behavior``.


Adapter Failure Boundary
------------------------

:meth:`qceval.core.bench.Adaptor.evaluate` converts evaluator exceptions into a
stable :class:`~qceval.models.QCEvalEvaluation`:

* ``SyntaxError`` becomes ``compiled=False``;
* another execution exception becomes ``compiled=True, ran=False``; and
* a completed semantic verification sets ``passed`` only when its status is
  ``verified_pass``.

The outer ``--eval-timeout`` can expire during framework execution or semantic
verification. Because it is not evidence of invalid syntax, its
``EvaluationTimeout`` record is runtime-shaped (``compiled=True, ran=False``)
and carries the semantic status ``resource_limit``.

Provider failures remain runner outcomes and do not enter the evaluator.
Compile and runtime failures are reported as runner outcomes and as
``execution_error`` in semantic aggregate reporting. Provider transport and
worker-isolation failures that cannot produce a normal evaluation outcome are
recorded as benchmark ``infrastructure_error``.


Sandbox and Framework Execution
-------------------------------

Generated source is executed by ``qceval.evals.sandbox`` in a temporary working
directory and isolated namespace. The requested entry point is called with
task-specific arguments, and the framework executor normalizes the result into
an :class:`~qceval.evals.models.ExecutionResult`.

.. warning::

   The sandbox uses Python ``exec()`` and is not a security boundary. Do not
   execute untrusted generated code without additional operating-system or
   container isolation.

The execution result can contain probabilities, framework metadata, a native
circuit or tape, and optional unitary data. These values feed lowering,
materialization, requirements, and diagnostics. A probability vector or unitary
does not pass merely by matching a retired threshold.


Qiskit
^^^^^^

The Qiskit executor accepts a ``QuantumCircuit`` or counts-like mapping. For a
circuit it prefers ``Statevector.from_instruction`` after handling
measurements, and uses seeded Aer sampling when exact state extraction is not
available. It records qubit and classical-bit counts, operations,
measurements, interactions, and the probability method.


Cirq
^^^^

The Cirq executor accepts a ``cirq.Circuit`` or counts-like mapping. It uses
Cirq statevector simulation with terminal measurements removed where supported
and records operations, measurements, interactions, and native circuit data.


PennyLane
^^^^^^^^^

The PennyLane executor captures QNode tape construction. It records operations,
wires, measurements, interactions, and observed probabilities. Returning raw
probabilities without a tape remains distinguishable in execution metadata so a
contract requirement can reject the shortcut.


CUDA-Q
^^^^^^

The CUDA-Q executor accepts a kernel and supported compatibility return shapes.
It prefers ``cudaq.get_state`` and falls back to seeded ``cudaq.sample`` when
needed. It records probability method, target, sampling information, and
source-derived gate metadata. Compact output-register selection remains an
executor compatibility feature.


Semantic IR and Lowering
------------------------

The default semantic verifier obtains the returned native program and sends it
to the lowering adapter registered for Qiskit, Cirq, PennyLane, or CUDA-Q. A
successful adapter emits the versioned Program IR, a framework-neutral
representation with a stable semantic hash.

Lowering is typed and fail closed:

``success``
    A Program IR is available for requirements and verifier routing.

``unsupported``
    The adapter cannot faithfully represent a construct or capability.

``execution_error``
    Inspection or conversion failed.

``resource_limit``
    Lowering exceeded a declared bound.

``unsupported`` is an internal lowering status: it becomes the final
``execution_error`` verdict while preserving the adapter reason. Inspection
errors also become ``execution_error``; limits remain ``resource_limit``. The
runtime does not guess, simplify an unknown construct, or fall back to another
grader.

After lowering, prompt-derived hard requirements are checked against the IR,
framework metadata, and source where required. A requirement mismatch is a
semantic failure. Diagnostics do not independently control the score.


Verifier Portfolio
------------------

Contracts route one of the supported semantic kinds:

* state, total-unitary, isometry, and channel engines;
* classical-input/output, distribution, and instrument engines;
* bounded symbolic verification for contracts that declare a supported
  completeness strategy.

Materialization depends on the contract, target, framework lowering, candidate
constructs, and resource limits. Packaged routes are checked for registry
closure during release. A materialization or capability gap is
``execution_error`` rather than a fallback trigger.

Distribution contracts use the executor's observed probabilities and a
packaged, versioned target provider, compared under the contract-declared
Hellinger-infidelity metric with an exact tolerance. This is a semantic engine
route, not the retired peak-match or ``canonical_class`` authority.


Routing and Reconciliation
--------------------------

:class:`~qceval.semantics.verifiers.VerifierRouter` reads the primary route
from the contract. It does not contain task-ID branches.

1. The router requires exactly one primary route and no fallback.
2. It resolves the engine and checks kind and capability
   support.
3. Engine cost estimates are compared with contract limits before execution.
4. It runs that engine and validates result identities.

Invalid routing, engine exceptions, identity mismatches, missing capabilities,
and unresolved uncertainty become ``execution_error``. Deterministic limit
violations remain ``resource_limit``.


Semantic Statuses
-----------------

Every completed behavior verification has one of four statuses:

* ``verified_pass``: sufficient evidence that the contract is satisfied;
* ``semantic_fail``: decisive contracted behavior mismatch;
* ``execution_error``: execution or verification could not produce a verdict;
  or
* ``resource_limit``: a deterministic limit prevented verification.

Only ``verified_pass`` sets ``passed=True``. The binary compatibility fields
``verified_status`` and benchmark ``status`` may project these outcomes to
``verified_fail`` and ``failed``, but the semantic record preserves
the reason.


Semantic Result Records
-----------------------

The evaluator validates that verifier output names the requested contract and
target hashes. It then emits a bounded semantic result record containing:

* result schema and semantic status;
* contract schema/version/hash and target version/hash;
* IR version and semantic input hash;
* verifier release and engine versions;
* metric evidence, tolerances, preconditions, and cases checked;
* bounded diagnostics and resource use; and
* Python, framework, and platform identity.

``grader_details.score_authority`` and
``grader_details.behavior_verdict.source`` are always ``"behavior"`` for
runtime grading. See :doc:`output` for the serialized shape and
:doc:`semantic_grading` for reporting rates.


Operational Controls
--------------------

Generation concurrency, evaluation worker processes, per-task isolation,
timeouts, caching, JSONL streaming, and resume continue to operate around the
same adapter boundary. They do not change contracts, routes, targets, or
status projection. A timeout is nonpassing; it is never converted into a
behavioral pass.
