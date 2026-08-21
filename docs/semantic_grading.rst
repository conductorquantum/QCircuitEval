Behavior-first semantic grading
===============================

QCircuitEval grades a candidate against a versioned behavioral contract, not
against the syntax of a canonical solution. The behavior result is authoritative
and fail closed: only ``verified_pass`` counts as a pass. No retired predicate,
reference-distribution, or unitary grader is consulted as a fallback.

Both suites have packaged contracts and independently derived,
hash-pinned targets. Both paths emit authoritative behavior results. Only
``verified_pass`` passes.


Suite verification models
-------------------------

Core and QEC tasks lower candidate programs to Program IR and route the
contracted semantic object through the verifier portfolio. QEC contracts
declare finite exhaustive parameter domains: the evaluator executes every
declared diagnostic point (including the no-error case and every permitted
single-error location), and a task passes only when every point passes.

QEC distribution contracts compare exact candidate probabilities against
independently generated GF(2)/stabilizer targets using Hellinger infidelity.
QEC state contracts (the Shor and Steane encoders) use phase-sensitive exact
state comparison, so a wrong-sign codeword cannot pass merely because it has
the expected measurement support.

QEC hard requirements additionally reject decoder-library imports, simulator
and sampler shortcuts, state-preparation amplitude injection, and dense-matrix
gates, and verify Program IR construction evidence: required data-ancilla
interactions, connected encoder topology, argument-conditioned physical error
gates on the concrete executed case, and multi-controlled correction gates.
Adjacent canceling gate padding is removed before topology checks are
evaluated. The legacy ``canonical_class`` case tables remain compatibility
fixtures used to audit migration parity; they are not a scoring path when the
packaged QEC contract registry is installed.


Contract and target model
-------------------------

Contracts are keyed by ``(suite, task_id)`` and validated independently of the
framework-specific task assets. A contract fixes:

* the public entry-point signature and named quantum systems;
* the semantic object to verify, such as a state, unitary, isometry, channel,
  distribution, classical mapping, instrument, or objective;
* observation, phase, ancilla, parameter, and approximation policies;
* hard construction requirements and deterministic resource limits;
* a content-addressed target with independent derivation provenance; and
* one validated primary verifier route.

Family proofs (for example structured QAOA/rotation completeness and the
``symbolic_family_bounded`` engine on core task ``42``) are completeness and
routing strategies, not a separate ``BehaviorKind``. The ``objective`` kind and
certified-approximation engines exist in the portfolio interface; packaged core
and QEC contracts currently route exact engines only.

The target artifact is not generated from the bundled canonical program. Its
identity and SHA-256 digest are recorded by both the contract and target
manifest. A result is accepted only when those identities match the routed
verifier output.

The field-reference schema checked in at
``docs/semantic_contract_schema.json`` describes core ``schema_version: "1"``
contracts. Packaged QEC contracts use ``schema_version: "2"``. The Python
contract validator remains authoritative for packaged release checks. Use
``qceval contracts validate|list|hash|diff`` to inspect packaged or local JSONL
registries.


Evaluation flow
---------------

For every assigned task, QCircuitEval performs the following steps:

1. The existing framework executor runs the requested entry point and captures
   the returned circuit, tape, kernel, probabilities, and bounded metadata.
2. A framework adapter lowers the native result into versioned Program IR.
   Unsupported constructs, inspection errors, and resource limits remain typed
   nonpassing outcomes.
3. Prompt-derived hard requirements are checked against Program IR, executor
   metadata, and source names only where the contract explicitly requires a
   construction constraint.
4. The contract router runs its declared verifier portfolio. Required routes
   are reconciled without selecting the most favorable result.
5. The evaluator validates contract, target, IR, engine, and environment
   identities before recording the semantic status and bounded evidence.

Framework probabilities and unitaries can materialize semantic evidence, but
they have no independent scoring authority. For both suites,
``canonical_class`` is compatibility metadata only; the packaged behavior
contract and its hash-pinned target are the scoring specification.


Source parser architecture
--------------------------

Most framework lowering inspects runtime objects: Qiskit ``QuantumCircuit``,
Cirq ``Circuit``, and PennyLane tapes. Source parsing is reserved for semantics
that those objects cannot expose exactly. All bounded AST implementations live
under ``qceval.evals.parser`` and ``qceval.frameworks``. Framework execution and
the public metadata entry points remain under ``qceval.frameworks`` and call the
parsers as needed. This keeps parsing policy separate from runtime integration.

``qceval.frameworks.cudaq``
    Parses CUDA-Q allocation, gate calls, controls, adjoints, measurements,
    bounded loops, runtime argument bindings, registered operations, and
    measurement-conditioned branches. CUDA-Q kernels are JIT callables without
    an introspectable circuit object, so exact source replay is the supported
    path to Program IR. Admitted constructs have one enumerated meaning;
    unresolved or data-dependent constructs return ``execution_error``.

``qceval.evals.parser.family``
    Parses the supported Qiskit, Cirq, PennyLane, and CUDA-Q spellings of the
    structured QAOA and rotation-family grammars. It resolves gate order, wires,
    and parameter indices and compares them with a manually verified template.
    Structural congruence proves equality of the parameterized operator for all
    declared inputs; an unresolved binding cannot produce a pass.

``qceval.frameworks.qiskit.symbolic``
    Runs the restricted symbolic-family grammar in a resource-capped subprocess.
    It accepts whitelisted imports and calls, builds exact SymPy matrices,
    certifies eligible numeric literals, and proves projective operator identity
    for all real parameters. A numerical counterexample is a decisive failure;
    timeout/memory limits return ``resource_limit``; unmatched literals,
    worker failures, or unsimplified residuals return ``execution_error``.

``qceval.evals.parser.source``
    Performs the small framework-neutral name scan used by explicit
    anti-shortcut requirements. This scan enforces construction constraints; it
    does not establish behavioral equivalence by itself.

Source parsing never approximates its way to a pass. It either reconstructs the
admitted semantics exactly, proves the contracted identity, finds a decisive
counterexample, or returns ``execution_error``/``resource_limit``. CUDA-Q static
replay is followed by the same hash-pinned target comparison as native-object
lowering. Dynamic replay emits measurement and ``ClassicalCondition``
operations, after which the exact branch engine enumerates Born probabilities
and conditional post-measurement states.


Verifier routing
----------------

Contracts can route exact state, total-unitary, isometry, channel,
classical-input/output, distribution, instrument, bounded symbolic,
structured-family, certified-approximation, and objective engines. This is a
portfolio interface, not a claim of universal support. Materialization depends
on the contract, framework, candidate constructs, and declared limits.

The router evaluates cost and capability claims before execution. Packaged
contracts have exactly one registered primary route, no cross-check, and no
fallback. Invalid routing, unavailable capabilities, unresolved verification,
or numerical uncertainty produce ``execution_error`` with a stable reason
code; deterministic limits produce ``resource_limit``.


Statuses and rates
------------------

Every completed semantic evaluation has one of four statuses:

``verified_pass``
    Sufficient evidence proves the contract. This is the only passing status.

``semantic_fail``
    A decisive contracted behavior or hard-requirement mismatch was found.

``execution_error``
    Candidate execution, faithful lowering, verifier processing, uncertainty
    resolution, route validation, or identity validation failed.

``resource_limit``
    A deterministic contract or worker limit prevented verification.

Reports publish all status counts and three rates with explicit denominators:

* strict pass rate: ``verified_pass / assigned``;
* coverage: ``(verified_pass + semantic_fail) / assigned``; and
* adjudicated pass rate: ``verified_pass / decisive``.

Adjudicated pass rate must not be ranked without coverage. Binary compatibility
fields project the four states to pass/fail, but the semantic record remains
the source of truth.


Reproducibility and release checks
----------------------------------

Each result records the contract, target, IR, verifier release, engine,
framework, Python, platform, resource, and score-authority identities. Readers
must stratify incompatible identities instead of combining them silently.
``grader_details.score_authority`` and
``grader_details.behavior_verdict.source`` are always ``"behavior"``.

Release maintainers use the checked-in commands documented in
``ci/README.md`` to verify generated semantic assets, target hashes,
independent derivations, and the reviewed regression corpus. The optional
category-sample export includes matching contracts and targets. The hidden
behavior corpus used for independent target derivation is not redistributed as
a review workbook.

Production evaluations run in spawned worker processes, bound stdout and
stderr, and terminate timed-out workers. This contains failures but is not a
hostile-code security boundary. Execute untrusted code only inside a separately
secured container or virtual machine.
