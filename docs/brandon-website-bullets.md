# QCircuitEval website and blog bullets

Draft copy for Brandon. These bullets are deliberately modular: use the shorter
positioning bullets in a blog post and pull the implementation detail into the
website.

## One-line description

- QCircuitEval is an open-source, deterministic behavior-contract benchmark for LLM-generated quantum circuit code across Qiskit, Cirq, PennyLane, and CUDA-Q.
- Shorter option: QCircuitEval tests whether AI-generated quantum programs are correct, not whether they look like one reference answer.
- Grader-first option: The benchmark is public; the durable technical value is the grader that proves behavior and rejects shortcuts.

## Why release it

- QCircuitEval's published Core results are already crowded near the top: 7 of 33 published model-and-reasoning configurations score at least 90%.
- The highest published Core result is 92.67%: 215 verified passes out of 232 assigned framework/task records for `openai/gpt-5.6-sol` at high reasoning effort.
- The next two published Core results are 92.24% (214/232) and 91.81% (213/232), both from other reasoning settings of the same model.
- Gemini 3.1 Pro Preview and Grok 4.6 each reach 90.95% (211/232) on the published Core suite.
- Those are strict Pass@1 rates: one generated answer, one attempt, and every assigned failure stays in the denominator.
- The repository publishes 9,240 individual records from 33 configurations across 10 models, with 280 records in each configuration.
- This top-end clustering is direct evidence that the current Core task set is becoming saturated as a discriminator among leading systems.
- The repository does **not** establish why scores are high. Training-data contamination is a plausible interpretation, but it is not measured or demonstrated by the checked-in results.
- Releasing the prompts and benchmark under the MIT License lets researchers train and evaluate smaller models without pretending that secrecy restores a saturated benchmark's value.
- The more reusable contribution is the grading system: versioned contracts, independently derived targets, framework-neutral Program IR, exact verifier engines, anti-shortcut requirements, and reproducible evidence.

## What is included

- QCircuitEval supports four quantum software frameworks: Qiskit, Cirq, PennyLane, and CUDA-Q.
- The Core suite has 58 tasks per framework: 232 framework/task records in an all-framework Core run.
- The QEC suite adds 12 quantum error-correction tasks per framework: 48 framework/task records in an all-framework QEC run.
- A complete Core-plus-QEC run therefore contains 70 tasks per framework and 280 framework/task records.
- The 58 Core tasks are tagged as 42 quantum-algorithm tasks, 10 state-preparation tasks, and 6 decomposition tasks.
- Core spans Grover search and amplitude amplification, QAOA, QFT and phase estimation, Simon and Bernstein–Vazirani, HHL, Shor/order finding, quantum walks, Hadamard and swap tests, teleportation, arithmetic, state preparation, gate decomposition, and Trotterized dynamics.
- The later Core task IDs 45–58 cover amplitude amplification, Simon's algorithm, period finding, QAOA, Ising and Heisenberg evolution, quantum walks, quantum counting, Shor order finding, HHL, spectral-sum phase estimation, and a Hadamard-test matrix element.
- The QEC suite covers bit-flip encoding, syndrome extraction and correction; phase-flip correction; a repetition-code logical CNOT; Shor and Steane encoding; Steane and Shor syndrome tasks; and five-qubit repetition correction.
- The repository does not identify a QuanBench+ task manifest or map QCircuitEval tasks to QuanBench+ tasks, so a precise “N tasks added over QuanBench+” claim is not verified here.

## The grader is the product

- QCircuitEval does not use an LLM as a judge.
- It does not ask whether generated code resembles a canonical solution.
- It does not give canonical source syntax, the legacy `canonical_class` field, or a same-framework reference program authority over the score.
- Instead, every task is paired with a versioned behavior contract and a hash-pinned target that was derived independently of the bundled canonical implementation.
- The contract says what must be observed: the public function signature, named systems, parameter domain, output policy, semantic object, metric, tolerance, resource limits, and any required construction details.
- Each contract selects exactly one production verifier route. Packaged contracts have no fallback route and do not choose whichever grader gives the most favorable result.
- Only `verified_pass` counts as correct.
- A decisive behavior or construction mismatch is `semantic_fail`.
- If execution, lowering, routing, target loading, or verification cannot produce a justified decision, the result is `execution_error`, not a guessed pass.
- If a deterministic time, memory, branch, expression, qubit, or dimension limit stops verification, the result is `resource_limit`, not a guessed pass.
- Numerical values inside a contract's uncertainty band become `execution_error`; QCircuitEval does not turn numerical ambiguity into a pass or a behavioral failure.
- This fail-closed design is aimed at preventing unsupported or partially checked code from being graded as correct.

## Two complementary checks: behavior and construction

- QCircuitEval combines behavioral grading with hard semantic and structural requirements.
- The behavioral layer asks whether the returned program implements the contracted state, operator, channel, classical relation, measurement distribution, or measurement-conditioned process.
- The construction layer asks whether the candidate actually built the requested quantum program instead of synthesizing the expected answer through a shortcut.
- Behavioral equality is necessary, but it is not always sufficient when the prompt explicitly requires a circuit construction, gate family, decomposition, interaction topology, syndrome extraction, or controlled correction.
- Hard requirements run before the behavioral engine and can decisively reject a candidate even when its observed output matches.
- This is more than a conventional unit test over a few known outputs: QCircuitEval lowers native programs into a framework-neutral intermediate representation and checks their operations, wires, controls, measurements, interactions, and classical conditions.
- Qiskit circuits, Cirq circuits, and PennyLane tapes are inspected as runtime objects.
- CUDA-Q kernels are JIT callables without an introspectable circuit object, so QCircuitEval performs bounded source replay to reconstruct Program IR.
- CUDA-Q replay supports admitted allocation, gates, controls, adjoints, bounded loops, argument bindings, registered operations, measurements, and measurement-conditioned branches.
- Unsupported or data-dependent CUDA-Q constructs do not get approximated into a pass; they produce a typed nonpassing result.
- The same shared contracts and verifier engines sit behind all four framework-specific executors and lowering adapters.

## Correctness without a canonical reference program

- QCircuitEval grades against a mathematical behavior contract, not one preferred implementation.
- The target artifact is not generated from the bundled canonical solution.
- Contract and target identities are content-hashed, and the grader checks those hashes before accepting a result.
- Equivalent implementations can pass even when their source, gate spelling, decomposition, or framework idiom differs from the bundled example.
- Where a prompt leaves several mathematically valid constructions open, the target can represent the valid set rather than a single syntax tree.
- For example, a Trotter task enumerates valid symmetric factor orderings and checks set membership, admitting the allowed orderings while rejecting a wrong Hamiltonian that merely lands near the ideal state.
- Structured parameter-family checks validate operation order, wires, and parameter bindings against audited templates across the declared family.
- Core task 42 uses a bounded symbolic projective-identity proof for all real parameter values in its admitted grammar.
- A symbolic timeout, unsupported grammar, worker failure, or unresolved identity cannot silently become a pass.

## Exact semantic coverage

- The 58 Core contracts contain 26 distribution tasks, 17 state tasks, 6 total-unitary tasks, 5 exhaustive classical-I/O tasks, 2 instruments, 1 channel, and 1 isometry.
- Five of the six total-unitary tasks use exact unitary comparison; Core task 42 uses the bounded symbolic-family engine.
- The 12 QEC contracts contain 10 exact distribution tasks and 2 exact state tasks.
- State contracts compare pure states with a trace-distance metric and the contract's phase and observation policy.
- Total-unitary and isometry contracts use operator norm, including logical-subspace and ancilla-restoration policy where declared.
- Channel contracts compare normalized Choi representations with Frobenius distance.
- Instrument contracts enumerate measurement branches and jointly check branch probabilities and normalized conditional quantum states.
- Classical-I/O contracts exhaustively compare the complete finite deterministic input/output relation, rather than checking a handful of examples.
- Distribution contracts compare the full observed and target probability distributions.
- Parameterized QEC tasks execute every declared case, including the no-error case and each permitted physical-error location; every case must pass.
- The Shor and Steane encoder tasks use phase-sensitive exact state checks, so matching measurement support with the wrong codeword signs is not enough.
- Dynamic measurement branches are simulated exactly in Program IR, including Born probabilities and conditional post-measurement states.

## Why Hellinger, not KL

- Distribution grading uses Hellinger infidelity: `1 - (sum_i sqrt(p_i q_i))^2`.
- Hellinger infidelity is symmetric and bounded between 0 and 1, which makes the score easier to interpret across tasks.
- Perfectly overlapping distributions have Hellinger infidelity 0; disjoint distributions have Hellinger infidelity 1.
- The production Core and QEC contracts route distribution tasks through Hellinger infidelity, not KL divergence.
- A smoothed KL-divergence helper remains in the codebase, but it has no production contract or verifier call site and is not a score-authority path.
- QCircuitEval's distribution check uses the entire probability vector, not only the most likely bit string or a few hand-picked outputs.
- Hellinger is only one part of the verifier portfolio: other exact routes use trace distance, operator norm, normalized Choi Frobenius distance, maximum branch distance, and exhaustive table equality.

## How shortcuts are caught

- QCircuitEval can reject code that returns precomputed counts, probabilities, or a unitary instead of constructing the requested program.
- It can reject direct state-amplitude injection through operations such as `initialize`, `StatePrep`, `set_statevector`, and related framework spellings.
- It can reject a nontrivial opaque dense unitary spanning the full register.
- It can enforce minimum qubit, operation, measurement, and entangling-gate counts.
- It can require specific measured wires and reject measurements on forbidden wires.
- It can require concrete interactions between named data, ancilla, control, and target qubits.
- It can enforce allowed gate families and decomposition requirements.
- It can ban task-level library, optimizer, eigensolver, simulator, sampler, decoder, and dense-evolution shortcuts when the contract calls for an explicit construction.
- A net-unitary nonlocality check catches fake compliance made from canceling entangling-gate pairs.
- Restricting that nonlocality check to the measured register also catches attempts to park an irrelevant entangler on unobserved ancillas.
- QEC topology checks remove adjacent canceling gate padding before checking required interactions.
- QEC contracts verify that a requested physical error appears as an argument-conditioned gate in the concrete Program IR case.
- QEC correction tasks can require actual multi-controlled correction gates.
- Cross-case Program IR invariance catches functions that branch on public arguments and hardcode the expected answer for each row of a finite truth table.
- The adversarial regression tests include trivial one-to-three-gate circuits, hardcoded QEC answer synthesis, canceling interaction padding, missing entangling structure, state preparation, and a QFT matrix hidden inside a custom gate.
- These checks are contract-specific: QCircuitEval does not pretend that one syntactic rule proves every quantum program correct.

## Results and what they do not prove

- The published result set contains 33 independent one-sample, one-attempt Pass@1 configurations with no unresolved infrastructure failures.
- Core and QEC scores are reported separately because they use different verification models.
- The strongest published QEC result is 87.5% (42/48), below the strongest Core result of 92.67% (215/232).
- Framework implementations of the same task are correlated observations, so the reporting protocol says confidence intervals should cluster on `(suite, task_id)`.
- The repository supports the statement that leading-model Core scores are top-heavy or saturated.
- The repository does not contain a contamination study, training-corpus audit, or memorization experiment. Do not present contamination as a measured cause.
- The repository does not contain a head-to-head QuanBench+ regrade, confusion matrix, or measured false-positive/false-negative rate.
- We can say the grader is designed to reduce false positives from output-only shortcuts and canonical-answer matching; we cannot attach a comparative false-positive percentage from this repository.
- We can say ambiguous and unsupported cases are kept separate as `execution_error` or `resource_limit`; we cannot claim a measured false-negative rate from this repository.

## Pass@K and feedback repair

- QCircuitEval implements Pass@K-style sampling: generate multiple independent candidates for a task and estimate the probability that at least one of the first `k` candidates is correct.
- Pass@K matters because hosted LLM inference can vary even in settings expected to be deterministic; “Non-Determinism of ‘Deterministic’ LLM Settings” reports run-to-run output and accuracy variation at temperature zero ([arXiv:2408.04667](https://arxiv.org/abs/2408.04667)).
- QCircuitEval keeps Pass@K separate from the official deterministic Pass@1 leaderboard protocol.
- It also implements iterative feedback repair: one initial generation followed by bounded repair attempts, stopping at the first verified pass.
- Feedback repair sends the model the original prompt, its earlier code, and bounded candidate-observable diagnostics.
- Feedback never reveals contract targets, thresholds, grader metrics, verifier evidence, hashes, reason codes, or error-taxonomy labels.
- Pass@K and feedback repair cannot be combined in one run because they answer different questions: independent-sample reliability versus improvement after diagnostics.
- Feedback reporting tracks the terminal pass rate over assigned repair chains, cumulative compile and pass rates by attempt, costs, and error-family transitions.
- The official leaderboard remains strict Pass@1: one sample, one attempt, temperature 0.0 when exposed, and no manual answer edits.

## Reproducibility and auditability

- Every semantic result records contract, target, Program IR, engine, verifier release, framework, Python, platform, resource, and score-authority identities.
- Reports expose the metric value, tolerance, uncertainty, cases checked, bounded evidence, and stable reason codes.
- The grader reports strict pass rate, decision coverage, and adjudicated pass rate with explicit denominators.
- Adjudicated pass rate is not meant to be ranked without coverage.
- Official submissions are regraded locally; submitted pass/fail labels and summaries are not trusted as score authority.
- The trusted scorer checks task IDs, entry points, framework and suite coverage, protocol settings, source identity, package and commit provenance, and every available candidate.
- Candidate regrades run in newly spawned processes with hard wall-clock deadlines and no worker reuse.
- Contracts, targets, and generated semantic assets are release-tested for reproducibility and hash consistency.
- Release checks require all 280 bundled canonical framework/task instances to receive `verified_pass`.
- The repository ships CPU and GPU Dockerfiles and documents read-only, network-disabled trusted scoring.
- The Python execution sandbox is explicitly not a security boundary; untrusted candidate code should run in a separately secured container or virtual machine.

## Availability

- QCircuitEval is beta software; scores and APIs may change between published revisions.
- The code is available at [github.com/conductorquantum/QCircuitEval](https://github.com/conductorquantum/QCircuitEval).
- The project homepage is [conductorquantum.com/qcircuiteval](https://conductorquantum.com/qcircuiteval).
- The project is released under the permissive MIT License.
- QCircuitEval requires Python 3.11 or later and ships a `qceval` command-line interface.
- The package includes the prompts, framework executors, behavior contracts, target artifacts, verifier engines, reporting code, published result records, and trusted leaderboard scorer.

## Claims to keep out of published copy unless separately sourced

- **UNVERIFIED IN THIS REPOSITORY:** QCircuitEval tasks appear in model training data.
- **UNVERIFIED IN THIS REPOSITORY:** contamination caused the published 90%+ Core scores.
- **UNVERIFIED IN THIS REPOSITORY:** an exact number of “Algorithm Zoo” or QEC tasks added relative to QuanBench+.
- **UNVERIFIED IN THIS REPOSITORY:** QCircuitEval has a measured false-positive or false-negative advantage over QuanBench+.
- **UNVERIFIED IN THIS REPOSITORY:** QuanBench+ uses a particular production implementation or threshold beyond what an external QuanBench+ source establishes.
- **UNVERIFIED IN THIS REPOSITORY:** any Ray-paper result or comparison.
