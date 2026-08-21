# QCircuitEval assets

This package contains the prompts and grading data that define a QCircuitEval
release. The grader treats each asset type as a separate input with a narrow
role.

## Directory layout

- [`core/`](core/) contains the 58 Core tasks for Qiskit, Cirq, PennyLane, and
  CUDA-Q.
- [`qec/`](qec/) contains the 12 quantum error correction tasks for the same
  four frameworks.
- [`contracts/`](contracts/) contains one behavior contract per task.
- [`targets/`](targets/) contains the expected mathematical objects selected by
  those contracts.
- [`_resources.py`](_resources.py) provides package-safe access through
  `importlib.resources`.

Each framework task file uses JSON Lines. Each row contains the prompt, task
identifier, entry point, executor metadata, and compatibility fields.

The `canonical_solution` field supports the smoke provider and regression
tests. The `canonical_class` field supports compatibility checks and selected
executor metadata. Neither field decides whether model-generated code passes,
but the executor does consume `canonical_class.output_qubits` (falling back to
the structural `required_measurement_qubits`) to select which qubits'
probabilities are collected, which matters for PennyLane and CUDA-Q. The
contract declares the same register in its terminal-observation requirement;
`ci/check_asset_consistency.py` fails CI when the two copies disagree.

## What decides a score

The behavior contract and its hash-pinned target define correctness.

The grader follows this path:

1. It loads a framework task and the contract with the same `(suite, task_id)`.
2. It executes the candidate entry point with the contract-declared arguments.
3. A framework adapter lowers the returned program to versioned Program IR.
4. It checks prompt-derived API, structure, and anti-shortcut requirements.
5. It runs a bounded source proof when the contract requires one.
6. It routes the lowered program to the single verifier engine named by the
   contract.
7. It validates the contract, target, Program IR, and engine identities.
8. It writes a semantic result with bounded evidence.

Program IR is a framework-neutral representation of circuit behavior. Qiskit,
Cirq, PennyLane, and CUDA-Q use separate executors and lowering adapters, then
share the same contract router and verifier engines.

There is no legacy fallback grader. A probability vector, unitary, canonical
source match, or `canonical_class` match has no score authority on its own.

## Contracts and targets

Contracts specify what the grader must observe and how it must compare the
candidate with the target. They define:

- the entry-point signature and named systems;
- the semantic object, such as a state, unitary, distribution, or channel;
- observation, bit-order, phase, ancilla, and parameter policies;
- the metric, tolerance, uncertainty, and error budget;
- hard construction requirements;
- deterministic resource limits;
- the target identity and SHA-256 digest; and
- one primary verifier route.

Targets contain the expected mathematical objects. Each suite has:

- `manifest.json`, which records target identity, provenance, derivations,
  dimensions, normalization, and the artifact digest; and
- `target.json`, which stores the grouped target documents.

The loader selects one task document from `target.json`, serializes that
document in canonical form, and checks its SHA-256 digest against both the
contract and manifest. A missing target, malformed manifest, or hash mismatch
returns a nonpassing `execution_error`.

Targets do not come from the bundled canonical solution. They encode separate
prompt-derived specifications and checks.

## Hard requirements

Behavioral equality is necessary but not always sufficient. A task can require
evidence that the candidate built the requested quantum program.

Requirements can enforce:

- minimum qubit, operation, measurement, and entangling-gate counts;
- required interactions and connected circuit structure;
- terminal observation and measured-wire policy;
- required gate families or decompositions;
- bans on returned counts, probabilities, or unitaries;
- bans on simulator, decoder, state-injection, and dense-matrix shortcuts; and
- QEC syndrome extraction, physical error insertion, and controlled
  correction structure.

A decisive requirement violation returns `semantic_fail`.

## Verifier portfolio

The contract chooses one registered production engine. Current engines cover:

- exact state comparison with the declared phase and observation policy;
- total-unitary and isometry comparison;
- channel comparison through normalized Choi representations;
- instrument comparison across measurement branches;
- exact distribution comparison with Hellinger infidelity;
- exhaustive classical input/output tables; and
- bounded symbolic and structured-family proofs.

Each engine checks its required capabilities and estimates cost before work
starts. An engine cannot claim support outside its registered kinds and
capabilities.

## Parameterized tasks

QEC contracts declare finite exhaustive input domains. The evaluator executes
the no-error case and every declared error case. Every case must pass.

Some Core tasks declare structured or symbolic parameter families. Their
bounded source verifiers prove supported identities for the full declared
domain. A timeout or unsupported proof step cannot become a pass.

## Result statuses

Every completed semantic check returns one of four statuses:

- `verified_pass`: sufficient evidence proves the contract;
- `semantic_fail`: a decisive behavior or hard-requirement mismatch exists;
- `execution_error`: execution, lowering, routing, target loading, or
  verification could not produce a verdict; or
- `resource_limit`: a deterministic limit stopped verification.

Only `verified_pass` sets `passed=True`.

For metrics with an uncertainty band, a value below the pass bound passes. A
value above the fail bound fails. A value inside the unresolved band returns
`execution_error`.

## Public prompt scaffold

Every Core and QEC prompt supplies the same amount of framework plumbing: one
canonical framework import and the exact public outer function signature with
an empty ellipsis body. The task requirements follow that scaffold. Models are
responsible for all quantum-program implementation details and any additional,
task-specific imports.

The scaffold never supplies decorators, registers or qubit allocation, gates,
inner kernels, or algorithm steps. This keeps missing framework imports from
becoming a benchmark confound without hiding incorrect framework API usage.

## Oracle isolation

External model providers receive the task prompt and public entry-point
information. They must not receive contracts, targets, canonical solutions,
expected probabilities, hashes, derivations, or semantic evidence.

The local smoke provider can read `canonical_solution` to run deterministic
regression checks. It does not send that source to an external model.

`qceval.core.prompt_safety` rejects provider prompts and repair feedback that
contain grading-oracle fields. Keep this boundary intact when adding provider
features or feedback modes.

## Execution isolation

The evaluator runs candidate source in a temporary working directory and
isolated Python namespace. Worker processes enforce output, time, and resource
bounds around production evaluations.

The Python sandbox uses `exec()` and is not a security boundary. Run untrusted
candidate code inside a secured container or virtual machine.

## Loading packaged assets

Runtime code must use the helpers in `_resources.py`. These helpers work for
editable installs, wheels, and other `importlib.resources` backends.

```python
from qceval.assets._resources import contract_resource, target_resource, task_resource

task_rows = task_resource("core", "qiskit").read_text(encoding="utf-8")
contract_rows = contract_resource("core").read_text(encoding="utf-8")
target_bytes = target_resource("core", "target.json").read_bytes()
```

Do not build runtime paths from the repository checkout.

## Maintainer checks

Validate both contract registries:

```bash
uv run qceval contracts validate --suite core
uv run qceval contracts validate --suite qec
```

Check generated QEC contracts, manifests, and targets:

```bash
uv run python ci/generate_qec_semantic_assets.py --check
```

Confirm that a clean wheel contains and loads every asset:

```bash
uv run python ci/check_packaged_assets_wheel.py
```

Run the focused asset tests:

```bash
uv run pytest tests/test_asset_resources.py \
  tests/semantics/test_targets_loading.py \
  tests/semantics/test_qec_semantic_contracts.py
```

Changing a prompt, contract, manifest, or target can change recorded hashes and
benchmark identity. Regenerate the affected suite, inspect the semantic diff,
run canonical grading across all four frameworks, and review the change as a
benchmark revision.
