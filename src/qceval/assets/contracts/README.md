# Behavior contracts

Behavior contracts are the authoritative grading specifications for
QCircuitEval tasks. They define what a candidate must do without requiring it
to match a canonical implementation.

## Registry files

- [`core.jsonl`](core.jsonl) contains one contract for each Core task.
- [`qec.jsonl`](qec.jsonl) contains one contract for each QEC task.

Each registry uses JSON Lines with one complete contract per line. Contracts
are keyed by `(suite, task_id)` and sorted by that stable key.

A contract is shared across Qiskit, Cirq, PennyLane, and CUDA-Q.
Framework-specific return, observation, and structure rules live inside the
shared contract.

## Contract field guide

### Identity and admission

- `schema_version` selects the strict parser schema.
- `suite` and `task_id` form the registry key.
- `contract_version` identifies the task specification revision.
- `audit_status` records whether the contract is provisional, reviewed, or
  blocked.
- `shadow_only` prevents a contract from producing an authoritative pass.

Blocked or shadow-only contracts return `execution_error`.

### Candidate interface

`signature` defines the required entry point, arguments, argument domains, and
return interface. The evaluator uses it to bind candidate calls.

### Semantic object

`kind` selects the object that the engine must verify:

- `state`
- `total_unitary`
- `isometry`
- `channel`
- `instrument`
- `distribution`
- `classical_io`
- `objective`

`systems` names the quantum and classical inputs, outputs, ancillas, work
registers, and environments. Each system has explicit indices and a dimension.

### Observation policy

`observation` defines which systems affect the verdict. It records:

- observed quantum and classical systems;
- ignored or marginalized systems;
- classical bit order; and
- any postselection event and minimum probability.

The framework adapters normalize native conventions into this policy before
comparison.

### Phase and ancilla policy

`phase` states whether global phase is irrelevant and whether relative phase
must be preserved.

`ancillas` states how each ancilla starts and whether it must be restored,
discarded, or left unconstrained.

These fields prevent a distribution-only match from passing a phase-sensitive
state task.

### Parameters and completeness

`parameters` defines parameter names, domains, units, periodicity, binding, and
quantification.

The quantifier can be:

- `none` for a fixed task;
- `all` for a full parameter family;
- `exhaustive` for every listed finite case; or
- `bounded` for a declared bounded domain.

`completeness` selects a supported proof strategy when source or symbolic
reasoning is required. `diagnostic_points` bind finite executions and
cross-checks.

QEC uses exhaustive domains. Every listed case must pass.

### Approximation policy

`approximation` defines:

- exact or approximate mode;
- metric name;
- pass tolerance;
- uncertainty allowance; and
- algorithmic error budget.

The engine returns `verified_pass` below the pass bound and `semantic_fail`
above the fail bound. An unresolved uncertainty band returns
`execution_error`.

### Target binding

`target` binds the contract to one mathematical target with:

- a stable target identifier and version;
- a SHA-256 digest;
- a provenance label;
- a manifest path; and
- the required number of separate derivations.

The referenced manifest selects `targets/<suite>/target.json`. The loader
hashes only the selected task document, not the full grouped file. The digest
must match the contract and manifest.

### Routing

`routing.primary` names the verifier engine and required capabilities.

Packaged contracts use:

- one primary route;
- no fallback route; and
- `cross_check: false`.

The router contains no task-specific branch. It resolves the engine named by
the contract, checks kind and capability support, evaluates cost, and runs the
engine.

Missing engines, invalid routes, capability gaps, and identity mismatches
return `execution_error`. A deterministic preflight limit returns
`resource_limit`.

### Resource limits

`limits` bounds:

- wall and CPU time;
- memory;
- qubit and Hilbert-space dimension;
- parameter cases;
- dynamic branches; and
- symbolic expression nodes.

The router checks engine estimates before execution. Verifiers enforce tighter
runtime bounds where required.

### Hard requirements

`requirements` contains prompt-derived constraints that behavior alone cannot
prove. Each item has a stable identifier, kind, source, and typed JSON value.

Requirements can check public interfaces, terminal observation, register
shape, operation counts, interactions, gate families, source names, and
anti-shortcut policies.

A requirement mismatch is a decisive `semantic_fail`. An unavailable
inspection path returns `execution_error`; it does not skip the requirement.

### Diagnostics

`diagnostics` enables bounded observations for debugging and reporting.
Diagnostics never override the behavior verdict.

## Grading flow

The evaluator applies a contract in this order:

1. Load and validate the contract registry.
2. Reject blocked or shadow-only contracts.
3. Execute the candidate with contract-bound arguments.
4. Replay every case for an exhaustive domain.
5. Lower the native circuit, tape, or kernel into Program IR.
6. Check hard requirements.
7. Run the declared source proof when required.
8. Route the lowered program to the contract-selected engine.
9. Load and hash-check the selected target document.
10. Validate result identities and emit bounded evidence.

There is no favorable fallback. Unsupported lowering, target errors, verifier
exceptions, and unresolved proofs remain nonpassing outcomes.

## Status model

The semantic result has four possible statuses:

- `verified_pass`: sufficient evidence proves the contract;
- `semantic_fail`: a decisive mismatch exists;
- `execution_error`: the grader could not establish a verdict; or
- `resource_limit`: a declared limit stopped the check.

Only `verified_pass` passes the benchmark.

The output record binds the verdict to the contract hash, target hash, Program
IR hash, engine version, framework, and environment.

## Editing contracts

Treat the checked-in registries as generated release artifacts.

For Core changes, update the curated pilot sources in
`qceval.semantics._core_contracts` and the prompt assets when the public task
changed, then refresh the packaged contract, manifest, and target files.

For QEC changes, update the shared specification in
`ci/generate_qec_semantic_assets.py`, then regenerate:

```bash
uv run python ci/generate_qec_semantic_assets.py --write
```

Validate and inspect the result:

```bash
uv run qceval contracts validate --suite core
uv run qceval contracts validate --suite qec
uv run qceval contracts hash --suite core
uv run qceval contracts hash --suite qec
uv run python ci/generate_qec_semantic_assets.py --check
```

Run canonical grading for every affected task on all four frameworks. A
contract change can alter accepted behavior even when its target hash stays
the same.

## Provider isolation

Contracts are grading oracles. Do not include contract fields, target data,
hashes, expected outputs, canonical solutions, or verifier evidence in provider
prompts or repair feedback.

`qceval.core.prompt_safety` enforces this boundary for provider-facing text.
