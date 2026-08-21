Leaderboard Protocol
====================

This page defines the public QCircuitEval submission protocol. The goal is to
make reported scores comparable across models, providers, and machines.


Official Tracks
---------------

Official tracks use one sample, one attempt, and the complete selected suite.
They do not use Pass@K, feedback repair, manual retries, or task subsets. Core
and QEC scores remain separate because their verification models differ. When
the provider exposes temperature, it must be ``0.0``. Providers that do not
expose temperature must report it as ``null`` with source ``not_exposed``.

The primary scientific endpoint is strict assigned-denominator Pass@1:
``verified_pass`` records divided by every assigned record. Provider, compile,
run, semantic, and infrastructure failures therefore cannot improve the
headline rate. Infrastructure failures are additionally marked as requiring a
rerun before an official final row is accepted.

.. list-table:: Tracks
   :header-rows: 1
   :widths: 25 25 25 25

   * - Track
     - Suites
     - Frameworks
     - Protocol
   * - ``core-all-single``
     - ``core``
     - Qiskit, Cirq, PennyLane, CUDA-Q
     - One sample, one attempt
   * - ``qec-all-single``
     - ``qec``
     - Qiskit, Cirq, PennyLane, CUDA-Q
     - One sample, one attempt
   * - ``full-all-single``
     - ``core`` and ``qec``
     - Qiskit, Cirq, PennyLane, CUDA-Q
     - One sample, one attempt; report suite rates separately

Use the ``custom`` track only for private experiments. Custom scores should not
be compared with official leaderboard rows. The scorer's
``--unsafe-structural-only`` mode is restricted to this track, skips trusted
grading, labels its output as unsafe, and can never produce an official row.


Required Command Shape
----------------------

Official runs must use the bundled task order, no ``--max-tasks`` limit, and
the default one-shot protocol:

.. code-block:: bash

   uv run qceval run \
     --provider <provider> \
     --model <model-label> \
     --framework all \
     --suite core \
     --temperature 0.0 \
     --max-retries 3 \
     --retry-base-delay 1.0 \
     --retry-max-delay 60.0 \
     --out results.core-all-single.jsonl

The omitted sampling flags use their one-shot defaults:
``samples_per_task=1``, ``pass_k=1``, and ``max_attempts=1``. Provider transport
retries repeat a failed request under the disclosed retry policy; they do not
create another benchmark sample or a feedback attempt.

The published ten-model, 33-job matrix can be expanded with one command:

.. code-block:: bash

   uv run qceval run \
     --provider openrouter \
     --registry production/models.prompt-effort-sweep.json \
                production/models.max-reasoning.json \
     --reasoning-effort all \
     --framework all \
     --suite all \
     --temperature 0.0 \
     --out results/

Official matrix runs do not use ``--max-tasks``. Mixed-model sweeps omit
OpenRouter route pins because endpoint tags differ by model. Every manifest job
is a separate Pass@1 configuration and score; never pool the files into one
rate.

Runs with ``--suite all`` must report core and QEC rates separately. Do not
publish one aggregate rate across the two suites.

Operational settings such as ``--generation-concurrency``,
``--evaluation-workers``, ``--task-timeout``, and ``--eval-timeout`` may be used
if disclosed. They must not change prompts, contracts, targets, verifier
routes, task order, or selected tasks. ``--timeout`` is a provider HTTP timeout
only; whole-task isolation requires an explicit ``--task-timeout``. Official
OpenRouter rows that pin a singular endpoint must disclose the full pin set
documented in :doc:`cli`.


Scientific Reporting And Analysis
---------------------------------

Report each suite's four framework results before any suite-wide aggregate.
An aggregate may combine frameworks within one suite only after those
per-framework results are shown; Core and QEC must never be pooled into one
rate. Framework implementations with the same ``(suite, task_id)`` are
correlated observations of one task, not independent tasks. Confidence
intervals or other resampling analyses must therefore cluster on
``(suite, task_id)`` so all framework and sample observations for that task
remain together.

Pass@5 and Feedback@5 are separate experimental protocols, not extensions of
the official Pass@1 row. They may be retained as documented future
experiments, but their samples or attempts must not be combined with Pass@1 or
presented as an official one-shot score.


Low-Cost Validation Diagnostic
------------------------------

The repository's low-cost scientific validation plan uses a single cheap
OpenRouter model with ``--reasoning-enabled`` when that is the model's only
exposed reasoning control, temperature ``0.0``, one sample,
``pass_k=1``, and one attempt. Its first live-provider request uses
``--max-retries 0`` so
configuration and transport failures remain visible. This diagnostic is not a
benchmark result or a model comparison; the fixed three-retry policy above
remains the official submission setting. No live multi-sample or feedback run
is part of that validation plan.


Allowed And Disallowed Assistance
---------------------------------

Allowed:

- The selected provider/model may generate candidate Python code from the
  prompt it receives.
- The official QCircuitEval runner may execute, simulate, and grade that code.
- Provider-level transient retries may use the fixed retry policy above.

Disallowed:

- Reading bundled canonical solutions, behavior contracts, target artifacts, or
  hidden expected values before generation.
- Manual editing of generated answers.
- Selecting the best result from multiple independent runs.
- Task-ID-specific answer maps or post-processing rules.
- External search, retrieval, tools, or quantum solvers unless the submission is
  labeled as a separate tool-augmented track.
- Changing task assets, contracts, targets, verifier routes, executor behavior,
  or source requirements.


Submission Artifacts
--------------------

Each public submission must include:

1. The QCircuitEval output file, in JSON or JSONL format, with
   ``schema_version == "qceval.run.v2"``.
2. A metadata JSON file with these fields:

   .. code-block:: json

      {
        "submitter": "Example Lab",
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.5",
        "model_version_or_date": "2026-05-26",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "date_utc": "2026-05-26T00:00:00Z",
        "qceval_commit": "<full 40- or 64-character lowercase git commit>",
        "image": "qceval:0.1.0@sha256:<digest>",
        "command": "qceval run ...",
        "allowed_tools": "none",
        "retry_policy": "max_retries=3, base=1.0, max=60.0",
        "cache_policy": "no prefilled cache",
        "disclosure": "No external tools or manual answer edits."
      }

3. Any non-default operational flags, such as concurrency, timeouts, endpoint
   overrides, or provider mode flags.

Every required metadata value must be a non-empty string. ``provider`` and
``model`` must exactly match the run artifact. ``qceval_commit`` must be a full
lowercase hexadecimal commit ID, and ``image`` must contain an OCI
``@sha256:`` digest with 64 lowercase hexadecimal characters.
Official tracks always require the metadata artifact; ``--strict`` applies the
same disclosure requirement to custom validation.

Do not submit API keys, raw secrets, local cache directories, or private
provider logs.


Scoring Script
--------------

This is the official-track trusted scorer. To regrade a local JSON or JSONL
run without leaderboard metadata, use ``qceval run --regrade all --suite all``
as in :doc:`cli` and :doc:`grader`.

Validate and score a submission with:

.. code-block:: bash

   uv run python scripts/score_submission.py \
     results.core-all-single.jsonl \
     --track core-all-single \
     --strict \
     --metadata submission-metadata.json \
     --trusted-regrade-timeout 300 \
     --out leaderboard-row.json

The scorer checks:

- output schema version,
- result count against ``summary.total_tasks``,
- valid terminal statuses,
- official one-shot protocol fields and exposed temperature ``0.0``,
- exact bundled task IDs, entry points, source identity, and framework/suite
  coverage for the selected track,
- top-level, per-result, and strict-metadata provider/model consistency,
- submitted ``qceval.package_version`` and ``qceval.commit`` consistency with
  the trusted local adapter whenever those local values are available,
- non-empty and plausibly formatted disclosure metadata for every official
  track,
- a clean trusted source checkout when scoring from Git,
- absence of ``infrastructure_error`` outcomes in an official final artifact,
- and, most importantly, every available candidate is regraded locally with
  the bundled :class:`qceval.core.bench.Adaptor`.

The submitted statuses, evaluations, and score summary are not authoritative.
The tabulated summary and compact leaderboard row are computed from the trusted
local regrade. A candidate-less provider failure remains a non-passing provider
failure. A local grader or infrastructure error fails closed and requires a
rerun; it is never converted into a model failure or accepted in an official
final row.

Each candidate regrade runs in a newly spawned process with a hard wall-clock
deadline. The worker creates its own POSIX process group before candidate
execution, and timeout and completion cleanup terminate that group, including
descendants. Workers are never reused across records, and the parent scorer
accepts only JSON evaluation data before deriving the trusted status itself.
Empty and whitespace-only code is handled as ``provider_failed`` without
starting candidate evaluation.

This process boundary is integrity containment, not a Python security sandbox.
Candidate code still has the worker's operating-system permissions. Run trusted
scoring in the documented unprivileged, read-only, network-disabled container
when evaluating untrusted submissions.

Trusted leaderboard rows include ``trusted_local_adapter`` with the local
``package_version``, commit, commit availability, and dirty-checkout state.
Installed wheels and sealed images may not contain ``.git``; those rows retain
the installed package version and state ``commit_status == "unavailable"``
with a null commit instead of treating the submitter's commit claim as trusted.

For custom diagnostics only, structural validation can be requested explicitly:

.. code-block:: bash

   uv run python scripts/score_submission.py \
     custom-results.json \
     --track custom \
     --unsafe-structural-only \
     --out unsafe-custom-row.json

Such output has ``validation_mode == "unsafe_custom_structural_only"`` and
``trusted_regrade == false``. It is not eligible for official acceptance.


Sealed Evaluation Image
-----------------------

The repository includes two Dockerfiles:

- ``Dockerfile`` builds a CPU-oriented image for Qiskit, Cirq, PennyLane, and
  lightweight CUDA-Q checks.
- ``Dockerfile.gpu`` builds on NVIDIA's CUDA-Q NGC image and is the recommended
  image for the official core track on a GPU host.

Build the CPU image from a clean checkout:

.. code-block:: bash

   docker build -t qceval:0.1.0 .

Build the GPU image:

.. code-block:: bash

   docker build \
     -f Dockerfile.gpu \
     -t qceval:0.1.0-gpu \
     --build-arg CUDAQ_IMAGE=nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.14.2 \
     .

The ``cu13`` image matches the ``cuda-quantum-cu13`` runtime pinned by
``uv.lock``. If the host driver only supports CUDA 12, use a CUDA-Q ``cu12``
NGC tag and regenerate the lockfile with the matching CUDA-Q runtime package.

For local smoke verification without network access:

.. code-block:: bash

   mkdir -p results
   docker run --rm \
     --network none \
     --read-only \
     --tmpfs /tmp:rw,nosuid,nodev,size=1g \
     --user "$(id -u):$(id -g)" \
     -v "$PWD/results:/results" \
     qceval:0.1.0 \
     qceval run \
       --provider smoke \
       --framework all \
       --suite core \
       --max-tasks 1 \
       --out /results/smoke.jsonl

For CUDA-Q GPU verification, the host must have a compatible NVIDIA driver and
NVIDIA Container Toolkit configured. Expose the GPU at runtime with
``--gpus all``:

.. code-block:: bash

   docker run --rm \
     --gpus all \
     --network none \
     --read-only \
     --tmpfs /tmp:rw,nosuid,nodev,size=1g \
     --user "$(id -u):$(id -g)" \
     -v "$PWD/results:/results" \
     qceval:0.1.0-gpu \
     qceval run \
       --provider smoke \
       --framework cudaq \
       --suite core \
       --max-tasks 1 \
       --out /results/smoke.cudaq.jsonl

To confirm Docker can see the GPU before running QCircuitEval:

.. code-block:: bash

   docker run --rm --gpus all --entrypoint nvidia-smi qceval:0.1.0-gpu

For provider-backed official runs, enable network only for the generation run
that contacts the provider endpoint. Run the scorer with ``--network none``:

.. code-block:: bash

   docker run --rm \
     --network none \
     --read-only \
     --tmpfs /tmp:rw,nosuid,nodev,size=1g \
     --user "$(id -u):$(id -g)" \
     -v "$PWD/results:/results" \
     qceval:0.1.0 \
     python scripts/score_submission.py \
       /results/results.core-all-single.jsonl \
       --track core-all-single \
       --strict \
       --metadata /results/submission-metadata.json

The image installs dependencies from ``uv.lock`` and runs as an unprivileged
user. In the recommended ``--read-only`` mode, bundled source, task assets,
contracts, targets, and verifier code are not writable at runtime; only mounted
result directories and ``/tmp`` are writable.


Leaderboard Acceptance Checklist
--------------------------------

- Official track name selected.
- No task subset and no task, contract, target, verifier, or executor edits.
- ``samples_per_task=1``, ``pass_k=1``, ``max_attempts=1``.
- Exposed temperature is exactly ``0.0``.
- Fixed retry policy disclosed.
- Provider, model, endpoint, date, QCircuitEval commit, and image digest
  disclosed.
- Artifact provider/model and bundled task identities are consistent.
- Submitted package/commit provenance matches the trusted scorer when locally
  verifiable, and source-checkout scoring is clean.
- No ``infrastructure_error`` remains; affected tasks were rerun.
- Trusted local regrade completed for every candidate.
- Scoring script passes with ``--strict``.
- Result artifact and metadata artifact contain no secrets.
