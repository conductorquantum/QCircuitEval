Behavior-first grader
=====================

QCircuitEval grades generated quantum-circuit code against versioned behavior
contracts and independently derived targets. Canonical source syntax and the
legacy ``canonical_class`` task field are not score authority.

Only ``verified_pass`` passes. The production grader has no fallback graders,
no fallback contract routes, and no ``unsupported`` or ``inconclusive`` final
verdicts.

A normal ``qceval run`` grades as it generates. ``--suite`` defaults to
``core``; pass ``--suite all`` when the artifact includes QEC. ``--input``
accepts JSONL streams and JSON run envelopes, including
``results/published/<configuration_id>.json``. ``--provider`` defaults to
``smoke``; no API key is required:

.. code-block:: bash

    uv run qceval run \
      --regrade all \
      --suite all \
      --input previous.jsonl \
      --out regraded.jsonl

Mixed prompt/regrade phases are documented in :doc:`cli`. Official
leaderboard scoring is a separate trusted-regrade path in :doc:`leaderboard`.
Grader contributions (contracts, engines, lowering, requirements) are welcome;
see :doc:`contributing`.


Score inputs
------------

===============================  ==========================================
Artifact                         Role in scoring
===============================  ==========================================
Framework task asset             Prompt, entry point, executor metadata
Behavior contract                Authoritative specification
Target document                  Independently derived expected object
Program IR + verifier engine     Evidence and verdict
===============================  ==========================================

Contracts are keyed by ``(suite, task_id)`` and shared across Qiskit, Cirq,
PennyLane, and CUDA-Q. Framework-specific observation and structural
conventions are fields in that shared contract.

Every packaged contract has exactly one primary route. Release checks require
that the route names a registered production engine, sets ``cross_check`` to
false, and has an empty ``fallback`` list.


Evaluation pipeline
-------------------

::

    provider code
      -> framework executor
      -> framework lowering -> Program IR
      -> hard structural / anti-shortcut requirements
      -> optional source-family proof
      -> one contract-selected behavioral engine
      -> identity validation and result record

Execution and lowering failures never invoke another grader. The stable reason
code records the specific failure while the final status remains one of the
four values below.


Verdict taxonomy
----------------

===================  ======================================================
Status               Meaning
===================  ======================================================
``verified_pass``    Sufficient evidence proves the behavior contract
``semantic_fail``    Decisive behavior or hard-requirement mismatch
``execution_error``  Execution or verification could not produce a verdict
``resource_limit``   A deterministic resource limit stopped verification
===================  ======================================================

``execution_error`` includes:

* candidate execution and framework-inspection errors;
* constructs that cannot be faithfully lowered or simulated;
* missing/invalid contracts, routes, engines, targets, or capabilities;
* unavailable source, unsupported source grammar, and unresolved symbolic
  proofs;
* verifier disagreement and numerical uncertainty bands; and
* result-identity or worker-protocol failures.

The original cause remains in ``reason_code`` and evidence records. Historical
schema-v1/v2 records carrying ``unsupported`` or ``inconclusive`` are migrated
to ``execution_error`` when read, with the old value retained as a
``historical_status`` diagnostic.

Timeouts, memory ceilings, expression-size limits, and deterministic verifier
preflight limits are ``resource_limit`` rather than ``execution_error``.


Suites and engines
------------------

The Core suite contains 58 tasks per framework:

================  ======  ================================================
Kind              Count   Primary engine
================  ======  ================================================
Distribution      26      ``distribution_exact``
State             17      ``state_exact``
Total unitary     6       5× ``unitary_exact``, 1× ``symbolic_family_bounded`` (task ``42``)
Classical I/O     5       ``classical_io_exhaustive``
Instrument        2       ``instrument_exact``
Channel           1       ``channel_exact``
Isometry          1       ``isometry_exact``
================  ======  ================================================

The QEC suite contains 12 tasks per framework: ten exact distribution
contracts and two exact state contracts. Parameterized QEC tasks execute every
declared no-error and physical-error case; every case must pass.

Distribution contracts use bounded, symmetric Hellinger infidelity,

.. math::

   1 - \left(\sum_i \sqrt{p_i q_i}\right)^2.

Other engine metrics include trace distance, operator norm, normalized Choi
Frobenius distance, maximum branch Choi distance, and exhaustive classical
table equality.


Hard requirements
-----------------

Program IR requirements run before behavioral engines. Depending on the
contract they enforce:

* terminal observation and measured-wire policy;
* minimum register, operation, measurement, and entangling-gate counts;
* required interactions and net-unitary entanglement;
* gate-basis and decomposition requirements;
* returned-count/probability/unitary bans;
* state-preparation and dense-matrix shortcut bans; and
* QEC parity extraction, argument-conditioned errors, connected code blocks,
  and controlled corrections.

A hard-requirement violation is ``semantic_fail`` because the contract was
decisively violated.


Source-family proofs
--------------------

Core tasks 04, 39, 40, 41, and 42 have universal parameter-family
requirements.

* Structured QAOA and rotation verifiers check gate order, wires, and parameter
  bindings against audited templates.
* Rotation families can use exact state comparison at declared diagnostic
  points for nonstandard but behaviorally valid spellings.
* Task 42 runs a bounded symbolic projective-identity proof in an isolated
  worker.

A symbolic proof or refutation is decisive. Timeout/memory/expression limits
are ``resource_limit``. Worker failures, unsupported proof grammar, or an
unresolved identity are ``execution_error``. They are never silently converted
to pass or behavioral fail.


Numerical boundary policy
-------------------------

The exact engines use contract tolerance and uncertainty metadata:

* below the pass bound -> ``verified_pass``;
* above the fail bound -> ``semantic_fail``;
* inside the uncertainty band -> ``execution_error``.

This preserves epistemic accuracy without exposing a third behavioral verdict.
The metric value, tolerance, uncertainty, and reason remain in evidence.


Release invariants
------------------

CI and release tests require:

1. 58 Core and 12 QEC contracts, all reviewed and non-shadow.
2. Exactly one registered primary route per contract.
3. ``cross_check == false`` and no fallback routes.
4. No final status outside the four-value taxonomy.
5. All 280 canonical framework/task instances return ``verified_pass``.
6. Generated contract and target assets reproduce byte-for-byte.

See :doc:`evaluation` for executor/lowering details, :doc:`output` for the
serialized schema, and :doc:`contributing` to change the grader.
