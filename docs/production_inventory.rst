Production Inventory
====================

Module-size and responsibility inventory for first-party Python under
``src/qceval``. Production modules should stay under 400 lines unless listed as
a documented exception. Splits must follow cohesive responsibility, not
arbitrary line cuts. Do not introduce generic ``utils.py`` / ``helpers.py``
modules or circular wrapper packages.


Status
------

Production modules were split into cohesive packages with stable import
facades. Several runtime modules remain over 400 lines where further splitting
would obscure a single responsibility (CLI entry, provider transport, dense
requirement tables, analytic catalogs, or production campaign helpers).


Modules currently over 400 lines
--------------------------------

Measured from the current checkout (line counts drift; re-measure after large
edits):

======= ===========================================================
Lines   Path
======= ===========================================================
403     ``frameworks/qiskit/symbolic/portable.py``
408     ``frameworks/cudaq/qir/ssa.py``
423     ``frameworks/qiskit/lowering.py``
466     ``error_taxonomy.py``
466     ``evals/evaluator.py``
483     ``production/deferred.py``
485     ``semantics/verifiers/targets.py`` (analytic catalog exception)
489     ``production/endpoints.py``
501     ``evals/parser/family/ast_utils.py``
517     ``semantics/verifiers/requirements/interactions.py``
538     ``semantics/verifiers/requirements/structural.py``
599     ``core/runner/base.py``
643     ``cli.py``
673     ``reporting/feedback_lineage.py``
999     ``providers/openrouter.py``
1130    ``semantics/verifiers/requirements/semantic.py``
======= ===========================================================

Treat new growth beyond this set as a signal to split by responsibility, not as
permission to add more oversized modules casually.


Completed splits (facades preserve public imports)
--------------------------------------------------

* ``semantics/targets/`` — schema / load / verify
* ``semantics/verifiers/requirements/`` — structural / gate_family / interactions / semantic
* ``semantics/verifiers/dynamic/`` — simulator / apply / payload
* ``semantics/verifiers/exact/`` — engines / classical / materializers / metrics
* ``semantics/verifiers/observational.py`` — facade over distribution materializers + engine
* ``evals/parser/family/`` — ast_utils / qaoa / rotation / verifier
* ``frameworks/qiskit/symbolic/`` — validation / portable / proof / worker
* ``frameworks/cudaq/qir/`` — models / cfg / ssa / gates / tokens / translate
* ``frameworks/cudaq/values.py`` / ``replay.py`` — facades over constfold/matrices and transform/simulate
* ``frameworks/cirq`` / ``pennylane`` lowering — wire_map / operations / adapter


Documented size / coverage exceptions
-------------------------------------

Declarative schemas, validation tables, analytic catalogs, offline generators,
subprocess workers, and branch-dense adapters may be omitted from import
coverage or left over 400 lines when splitting would reduce clarity.
See ``[tool.coverage.run] omit`` in ``pyproject.toml`` for the authoritative
coverage omission list.


Architectural invariants
------------------------

* Runtime target loading goes only through
  ``semantics.targets.load_contract_target_document`` (grouped manifests hash
  per-task documents).
* Framework-neutral semantics live under ``qceval.semantics``; framework-specific
  lowering stays under ``qceval.frameworks.<name>``.
* Fail-closed statuses: ``verified_pass``, ``semantic_fail``,
  ``execution_error``, ``resource_limit``.
* Contracts, canonical solutions, and oracle data must never enter provider
  prompts or feedback.
* Authoritative source hashing: ``semantics.ir.source_code_sha256``.
* Dense payload hashing: ``semantics.lowering.utils`` matrix/statevector helpers.
