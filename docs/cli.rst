Command-Line Interface
======================

``qceval run`` drives the benchmark pipeline: loading tasks, calling the
provider, executing generated code, behavior verification, and result writing.
``qceval contracts`` inspects versioned behavior-contract registries. Almost
all configuration is passed through explicit flags. Credential flags may fall
back to ``OPENROUTER_API_KEY`` or ``CODA_API_KEY`` in the process environment,
then to a ``.env`` file in the current working directory; no other environment
variables are read for run configuration.

This page documents every flag, explains how they interact, and covers the
validation rules that the CLI enforces before a run begins.


Invocation
----------

.. code-block:: bash

    qceval run [flags]

The ``run`` subcommand is required for benchmark execution. Calling ``qceval``
without a subcommand prints a usage error.

At completion, ``qceval run`` prints a compact table with the overall pass rate,
per-suite/framework rows when multiple scopes were evaluated, failure counts,
and the output path. In feedback mode, the table is explicitly labeled as an
attempt-record summary and a separate lineage line reports the terminal pass
rate over assigned chains; use the latter as the Feedback@N headline outcome.


Required Flags
--------------

``--out PATH``
    The file path where results will be written. Parent directories are created
    automatically if they do not exist. This is the only universally required
    flag.

    The file extension determines the default output format: ``.json`` produces
    a single JSON document, ``.jsonl`` produces newline-delimited JSON. You can
    override this with ``--output-format``.


Provider Selection
------------------

``--provider {smoke,openrouter,coda}``
    Selects which code-generation provider to use. Defaults to ``smoke``.

    The ``smoke`` provider returns deterministic responses without calling any
    external API. It is useful for verifying the installation and testing the
    evaluation pipeline. The ``openrouter`` provider calls an OpenRouter
    chat-completions endpoint. The ``coda`` provider calls the Coda Agents
    endpoint. See :doc:`providers` for details on built-in providers.

``--model MODEL``
    The model identifier or output label recorded for the provider. For the
    ``openrouter`` provider, this is the OpenRouter model slug (e.g.,
    ``anthropic/claude-sonnet-4.5``). Required when ``--provider openrouter`` is
    set unless ``--registry`` supplies the model roster. Optional for the
    ``smoke`` provider, which defaults to
    ``smoke-canonical``. Optional for the ``coda`` provider; when omitted, the
    CLI derives ``coda/build``, ``coda/build-fast``, ``coda/learn``, or
    ``coda/learn-fast`` from the selected Coda flags.

``--registry PATH [PATH ...]``
    Expand one or more model-registry JSON files into a Pass@1 matrix. Each
    registry entry supplies ``model_id`` and ``reasoning_efforts``. Multiple
    files are unioned in first-seen model order; duplicate models merge efforts
    from weakest to strongest. Omit ``--reasoning-effort`` or use
    ``--reasoning-effort all`` to keep every listed level, including the
    unnamed ``enabled`` setting.


Framework Selection
-------------------

``--framework {qiskit,cirq,pennylane,cudaq,all} [...]``
    Selects one or more quantum frameworks to evaluate. Defaults to ``all``.
    Pass multiple names after one flag or repeat the flag, for example
    ``--framework qiskit cirq``. ``all`` cannot be combined with a named
    framework.

    When set to ``all``, the runner loads and evaluates the Qiskit, Cirq,
    PennyLane, and CUDA-Q task sets sequentially (58 core tasks each, 232 core
    tasks total). When set to a specific framework, only that framework's
    bundled tasks are evaluated.

    CUDA-Q is a required project dependency. The lockfile pins matching CUDA-Q
    metadata and runtime packages; GPU acceleration still depends on NVIDIA's
    supported platform and driver stack.

    The framework determines which executor is used to compile and simulate the
    generated code. See :doc:`evaluation` for details on how each framework's
    executor works.

``--suite {core,qec,all}``
    Selects the benchmark suite. Defaults to ``core``.

    ``core`` is the historical core suite, currently 58 tasks per framework.
    ``qec`` selects the 12 quantum-error-correction prompt/task assets per
    framework, with IDs ``qec01`` through ``qec12``. ``all`` selects core first
    and QEC second for each framework.

    Core uses packaged semantic contracts and independent targets. QEC executes
    every finite case declared in its task assets and enforces distribution,
    metadata, and source requirements. Both paths are behavior-authoritative.

``--max-tasks N``
    Limits the number of tasks evaluated per framework. Must be a positive
    integer. When omitted, all tasks in the selected framework(s) are evaluated.

    This is useful for quick smoke tests or debugging a specific task range.
    Tasks are evaluated in their bundled order. Core task IDs are zero-padded
    suite-local IDs ``01`` through ``58`` (contiguous), and QEC task IDs range
    from ``qec01`` through ``qec12``. If multiple suites are selected, the
    limit applies independently to each suite/framework pair.

``--tasks N [N ...]``
    Selects an exact subset of suite-local task numbers, such as
    ``--tasks 1 7 42``. Numbers are normalized to bundled IDs, so ``1``
    selects core task ``01`` and QEC task ``qec01`` when their respective
    suites are selected. Requested tasks preserve bundled order. This flag is
    mutually exclusive with ``--max-tasks``.


Selective Prompting and Regrading
---------------------------------

By default, every selected framework is both prompted and graded. To rerun
only selected parts of an experiment, choose the two phases independently.
When ``--framework`` is also supplied, the phase selections must be contained
within it; explicit phase selections determine which frameworks are processed.

``--rerun {qiskit,cirq,pennylane,cudaq,all} [...]``
    Requests fresh candidate code for the listed frameworks. A framework named
    only here is written with status ``generated`` and no evaluation, so the
    exact response can be graded later. ``--prompt`` is an alias.

``--regrade {qiskit,cirq,pennylane,cudaq,all} [...]``
    Runs the current grader for the listed frameworks. When a framework also
    appears in ``--prompt``, the freshly generated response is graded. When it
    appears only in ``--regrade``, QCircuitEval loads its stored response from
    ``--input`` instead of calling the provider.

``--input PATH``
    Existing JSON or JSONL run output supplying candidate responses for
    regrade-only frameworks. It is required whenever ``--regrade`` includes a
    framework absent from ``--prompt``. JSON envelopes (``qceval.run.v2``)
    and JSONL streams are both accepted; ``--resume-from`` remains JSONL-only.
    The selected task, framework, suite, and sample must each be present in
    the input artifact; QCircuitEval uses the latest stored feedback attempt
    for each sample.

To regrade every stored candidate without calling a model, pass ``--suite all``
(the default suite is ``core``):

.. code-block:: bash

    uv run qceval run \
      --regrade all \
      --suite all \
      --input previous.jsonl \
      --out regraded.jsonl

The same command accepts a published envelope:

.. code-block:: bash

    uv run qceval run \
      --regrade all \
      --suite all \
      --input results/published/<configuration_id>.json \
      --out regraded.jsonl

``--provider`` defaults to ``smoke``, so regrade-only runs do not need an API
key. ``--framework`` still restricts which stored records to grade when set.

This mixed-phase example re-prompts Qiskit task 7 without grading it, prompts
and grades Cirq task 7, and regrades the stored CUDA-Q task 7 response:

.. code-block:: bash

    qceval run \
      --tasks 7 \
      --rerun qiskit cirq \
      --regrade cirq cudaq \
      --input previous.jsonl \
      --out refreshed.jsonl

Selective phase runs cannot use ``--resume-from``, feedback repair,
``--fail-fast``, or ``--task-timeout``. These modes have a
single generate-and-grade lifecycle and therefore do not preserve the explicit
phase boundary.


Output Format
-------------

``--output-format {auto,json,jsonl}``
    Overrides the output format inferred from the file extension. Defaults to
    ``auto``, which uses the extension of the ``--out`` path.

    - ``json``: A single JSON document with ``results``, ``summary``, and
      metadata at the top level. Human-readable with indentation.
    - ``jsonl``: One JSON object per line. Each result record is a separate
      line, followed by a final summary envelope. Better for streaming
      processing and large result sets.

    See :doc:`output` for the full schema of both formats.


OpenRouter Configuration
------------------------

These flags configure the ``openrouter`` provider. They are ignored when
another provider is selected.

``--openrouter-api-key KEY``
    Your OpenRouter API key. The key is passed directly to the OpenRouter API
    as a Bearer token; it is never written to disk or logged.

``--openrouter-api-key-file PATH``
    Read the API key from a file instead of placing it in process arguments.
    This is recommended for production workers. The file and direct-key flags
    are mutually exclusive.

    If neither credential flag is supplied, the CLI checks
    ``OPENROUTER_API_KEY`` in the process environment and then a ``.env`` file
    in the current working directory. Explicit flags take precedence over the
    environment, which takes precedence over ``.env``.

``--openrouter-base-url URL``
    The chat-completions endpoint URL. Defaults to
    ``https://openrouter.ai/api/v1/chat/completions``. Override this to use an
    OpenRouter-compatible proxy or a local model server that exposes the same
    API shape.

``--openrouter-endpoint-tag TAG``
    Exact OpenRouter endpoint tag. When set, the provider pins
    ``provider.only`` to this singular route and disables fallbacks.

``--openrouter-max-output-tokens N``
    Frozen model output ceiling sent to the pinned endpoint.

``--openrouter-output-limit-source {author_native,benchmark_floor}``
    Evidence source for the frozen output ceiling.

``--openrouter-endpoint-cap-status {catalog_numeric,undisclosed_first_party_exception}``
    Catalog evidence status for the selected endpoint completion cap.

``--openrouter-output-token-parameter {max_tokens,max_completion_tokens}``
    Exact output-ceiling parameter exposed by the pinned endpoint.

``--openrouter-route-revision REV``
    Frozen route revision recorded in request identity and per-result
    provenance.

``--configuration-id ID``
    Frozen campaign configuration identity recorded in cache keys, results,
    and route provenance.

All six OpenRouter pin fields
(``--openrouter-endpoint-tag``, ``--openrouter-max-output-tokens``,
``--openrouter-output-limit-source``, ``--openrouter-endpoint-cap-status``,
``--openrouter-output-token-parameter``, ``--openrouter-route-revision``)
plus ``--configuration-id`` must be supplied together. Partial pin sets are
rejected before the run starts. When an endpoint tag is pinned, omitted
``--temperature`` stays unset rather than defaulting to ``0.2``.

``--max-retries N``
    Maximum number of additional transport-request retries for transient HTTP
    errors (408, 429, 5xx, connection errors). Defaults to ``3``. These retries do
    not increment ``sample_index`` or ``attempt_index``. Set to ``0`` to
    disable retries.

``--retry-base-delay SECONDS``
    Base delay in seconds for exponential backoff between retries. Defaults to
    ``1.0``. The delay doubles with each retry attempt up to a cap of 60
    seconds.

``--retry-max-delay SECONDS``
    Maximum delay cap in seconds for retry backoff and numeric ``Retry-After``
    headers. Defaults to ``60.0``.


Coda Configuration
------------------

These flags configure the ``coda`` provider. They are ignored when another
provider is selected.

``--coda-api-key KEY``
    Your Coda API key. The key is passed directly to the Coda API as a Bearer
    token; it is never written to disk or logged. When omitted, the CLI checks
    ``CODA_API_KEY`` in the process environment and then ``.env`` in the
    current working directory.

``--coda-agents-url URL``
    The full Coda agents endpoint URL. Defaults to
    ``https://api.conductorquantum.com/v0/coda/agents``. Override this to use a
    gateway or local endpoint.

``--coda-mode {build,learn}``
    Coda agent mode. Defaults to ``build``. This affects the Coda request body
    and the default QCircuitEval model label.

``--coda-fast``
    Sends ``fast: true`` in the Coda request body and changes the default model
    label suffix to ``-fast``.

``--coda-prefer-structured-response``
    Prefer Coda structured response code when it defines the requested entry
    point. Without this flag, token-stream code is used when both emitted
    sources define the entry point.

Coda does not expose model selection through this API. ``--model`` is an output
label only when ``--provider coda`` is selected; it is not sent to Coda. Coda
also does not expose temperature. ``--temperature`` is ignored for Coda and the
CLI prints one warning when the flag is present.


Generation Parameters
---------------------

``--temperature FLOAT``
    The sampling temperature passed to the provider. Lower values produce more
    deterministic output. When omitted, the provider uses its built-in default
    (0.2 for the OpenRouter provider). Must be zero or greater.

``--reasoning-effort {max,xhigh,high,medium,low,minimal,none,all}``
    OpenRouter reasoning effort. The selected model must support the requested
    level. ``all`` expands a single model into seven named-effort jobs
    (``none`` through ``max``), or keeps all efforts listed by ``--registry``.
    The explicit value is recorded in the run protocol and provider response
    metadata.

``--reasoning-enabled``
    Enable reasoning for an OpenRouter model that does not expose named effort
    levels. Mutually exclusive with ``--reasoning-effort``.


Pass@1 Matrix Expansion
-----------------------

A matrix invocation is still one ``qceval run`` command, but each expanded
``(model, reasoning effort)`` pair is an independent Pass@1 job with its own
result file and score. Never pool records from separate jobs into one rate.

For a file-shaped ``--out``, a single-model effort sweep writes files such as
``results.effort-none.json`` through ``results.effort-max.json`` and records
them in ``results.efforts.json``. For a directory-shaped ``--out``, every job
writes ``<configuration_id>.json`` and the directory receives
``manifest.json``.

The manifest uses schema ``qceval.effort_sweep.v1``. Each job entry records
``model``, ``reasoning_effort``, ``configuration_id``, ``out``, and
``exit_code``. A failing job stops later jobs; the partial manifest still
records every job that started.

Multi-job sweeps reject Coda, ``--resume-from``, selective
``--rerun``/``--regrade``, and a user-supplied ``--configuration-id``.
``--reasoning-enabled`` cannot be combined with ``--registry`` because
registries represent unnamed reasoning as the ``enabled`` effort. Sweeps stamp
their own configuration identity. Unpinned mixed-model OpenRouter sweeps do not
send that identity as route provenance; OpenRouter receives
``configuration_id`` only when a complete endpoint pin is supplied.

``--timeout SECONDS``
    The HTTP request timeout in seconds for the provider API call. When omitted,
    the provider uses its built-in default (120 seconds for OpenRouter, 900
    seconds for Coda). Must be a positive number.

    This is a per-request provider timeout only. It does not become a whole-task
    kill timeout. Use ``--task-timeout`` when each generate-and-evaluate task
    must run in an isolated worker with a hard wall-clock deadline. Selective
    ``--rerun``/``--regrade`` runs also keep ``--timeout`` as a provider-request
    timeout and reject ``--task-timeout``.


Evaluation Protocol
-------------------

``--samples-per-task N``
    Number of independent candidate generations per task. Defaults to ``1``.
    Values greater than ``1`` enable Pass@K-style sampling. QCircuitEval does
    not change temperature automatically; use ``--temperature`` explicitly for
    stochastic sampling.

``--pass-k K``
    Pass@K cutoff used in task-level reporting. Defaults to ``1``. Must satisfy
    ``1 <= K <= --samples-per-task``.

``--max-attempts N``
    Total attempts per task in feedback-repair mode, including the initial
    attempt. Defaults to ``1``. Values greater than ``1`` enable iterative
    repair from bounded execution diagnostics. Thus ``--max-attempts 5`` means
    one initial generation followed by at most four repairs, with termination
    at the first verified pass.

``--feedback-max-chars N``
    Maximum characters of diagnostic text included in each repair feedback
    message. Defaults to ``2000``.

Feedback uses the versioned ``feedback.execution_trace.v1`` policy. Every
repair request contains the original prompt followed by the full sequence of
prior assistant code and user feedback turns. Compile and runtime failures
return their bounded execution trace. A semantic failure returns only
candidate-observable output (the output vector and allowlisted execution
metadata). Contract targets, grader metrics and thresholds, verifier evidence
and reason codes, and error-taxonomy labels are never sent to the model.

Provider transport failures are retried by the provider's normal retry policy;
an exhausted provider failure terminates and censors that chain instead of
asking the model to repair it. Verifier-side ``execution_error`` and
``resource_limit`` outcomes also terminate as non-decisions. An
``execution_error`` explicitly attributed to ``candidate_execution`` remains
repairable.

Pass@K and feedback repair are separate protocols. Pass@K estimates the chance
that at least one independent sample passes. Feedback repair measures cumulative
improvement after diagnostic feedback. The CLI rejects combining
``--samples-per-task > 1`` with ``--max-attempts > 1``.

Strict assigned-denominator Pass@1 is the primary official one-shot endpoint.
The Pass@5 and Feedback@5 commands below document separate future experiment
shapes; they are not additional samples or attempts for an official Pass@1
run. See :doc:`leaderboard` for the authoritative reporting order, suite
separation, and task-cluster analysis rules.

When JSONL output is selected, QCircuitEval automatically streams result records
for Pass@K and feedback runs so interrupted jobs can be resumed.


Throughput and Resume
---------------------

``--generation-concurrency N``
    Runs up to ``N`` provider requests concurrently. Defaults to ``1``, which
    preserves the historical serial behavior.

``--evaluation-workers N``
    Evaluates generated code in up to ``N`` worker processes. Every candidate
    is evaluated in an isolated spawned worker, including when ``N=1``; the
    option controls concurrency rather than whether isolation is enabled.
    Processes are used because the evaluator changes the working directory and
    framework executors may temporarily patch global state. Defaults to ``1``.

``--cache-dir PATH``
    Enables a portable JSON response cache. Cache keys include the provider,
    model, prompt, framework, task ID, and generation settings. API keys are not
    written to cache files.

``--resume-from PATH``
    Reuses completed ``kind: result`` records from a prior JSONL output file.
    This is useful after interrupted runs. The resumed output is self-contained:
    completed records are re-emitted before new records are appended.

``--task-timeout SECONDS``
    Runs each task in an isolated worker process and terminates it after the
    deadline. The deadline covers generation plus evaluation. This prevents one
    stuck provider call from blocking the full benchmark.

``--eval-timeout SECONDS``
    Records a task as a non-passing candidate ``resource_limit`` with
    ``error_type: EvaluationTimeout`` if framework execution or semantic
    verification exceeds the deadline. It is not reported as a syntax or
    compile failure. This is a guardrail for pathological generated code, not
    a security boundary. Defaults to ``60`` seconds.

``--fail-fast``
    Stops the run after the first non-passing task. This is useful for smoke
    tests and debugging because later tasks are not generated or evaluated.

``--stop-on-infrastructure-error``
    After an ``infrastructure_error`` record, drains the active generation
    chunk and leaves later prompts pending instead of continuing the full
    schedule. Useful for production campaigns that must not keep spending
    after provider or worker infrastructure fails.

``--progress``
    Shows a ``tqdm`` progress bar and writes one concise status line per
    completed task to stderr.

When the output format is JSONL and concurrency, resume, ``task_timeout``, or
``timeout`` is enabled, QCircuitEval streams one result line as each task
completes, followed by the final summary line. The final in-memory payload and
JSON output remain ordered by framework and task ID, regardless of completion
order.


Smoke Provider Configuration
----------------------------

``--smoke-mode {canonical,empty,error}``
    Controls the behavior of the ``smoke`` provider. Ignored when
    another provider is selected.

    - ``canonical``: Returns the bundled canonical solution or a deterministic
      compatibility response. It is the default, but it does not override
      lowering or verifier support.
    - ``empty``: Returns empty code for every task. Empty and whitespace-only
      responses are not ``ok``, so the runner records ``provider_failed``
      without evaluating.
    - ``error``: Returns a provider error for every task. Useful for testing the
      provider-failure path.


Contract Registry Commands
--------------------------

The ``contracts`` command validates and inspects behavior contracts without
running providers or candidate code:

.. code-block:: bash

    qceval contracts validate --suite core
    qceval contracts list --suite core
    qceval contracts hash --suite core
    qceval contracts diff old.jsonl new.jsonl

``validate``, ``list``, and ``hash`` accept ``--path`` to inspect a local JSONL
registry instead of the packaged suite. A missing packaged suite registry is an
error for these inspection commands and becomes ``execution_error`` during
evaluation.


Metadata
--------

``--source-hint HINT``
    An optional string recorded in the output metadata under
    ``qceval.source_hint``. This is purely informational---it is not read or
    used at runtime. It can be used to record which version of a task set was
    used to build the bundle, or any other provenance note.


Validation Rules
----------------

The CLI enforces several validation rules before starting a run:

- ``--out`` is always required.
- When ``--provider openrouter`` is set, an OpenRouter credential must resolve
  from a flag, ``OPENROUTER_API_KEY``, or ``.env``. ``--model`` is required
  unless ``--registry`` supplies the roster.
- When any OpenRouter pin flag or ``--configuration-id`` is set, the full pin
  set (endpoint tag, max output tokens, output-limit source, endpoint-cap
  status, output-token parameter, route revision, and configuration ID) must
  be supplied together.
- When ``--provider coda`` is set, a Coda credential must resolve from
  ``--coda-api-key``, ``CODA_API_KEY``, or ``.env``. ``--model`` is optional
  for Coda.
- When ``--temperature`` is provided with ``--provider coda``, the CLI warns
  that Coda does not expose temperature and the flag is ignored.
- ``--max-tasks`` must be a positive integer (greater than zero).
- ``--timeout`` must be a positive number (greater than zero).
- ``--temperature`` must be zero or greater.
- ``--max-retries`` must be zero or greater.
- ``--retry-max-delay`` must be greater than zero.
- Every ``--framework``, ``--rerun`` (or ``--prompt``), and ``--regrade`` value must be one of
  ``qiskit``, ``cirq``, ``pennylane``, ``cudaq``, or ``all``. ``all`` cannot
  be combined with named frameworks in the same selection.
- ``--tasks`` values must be positive integers, must not be combined with
  ``--max-tasks``, and must occur in at least one selected suite.
- A regrade-only framework requires ``--input``. ``--input`` must be a
  ``.json`` envelope or ``.jsonl`` stream. Selective phase runs cannot be
  combined with ``--resume-from``, feedback repair, ``--fail-fast``, or
  ``--task-timeout``. ``--resume-from`` remains JSONL-only.
- ``--suite`` must be one of ``core``, ``qec``, or ``all``.
- ``--smoke-mode`` must be one of ``canonical``, ``empty``, or ``error``.
- ``--coda-mode`` must be either ``build`` or ``learn``.
- ``--generation-concurrency`` and ``--evaluation-workers`` must be positive
  integers.
- ``--samples-per-task``, ``--pass-k``, ``--max-attempts``, and
  ``--feedback-max-chars`` must be positive integers.
- ``--pass-k`` must be no greater than ``--samples-per-task``.
- ``--samples-per-task > 1`` cannot be combined with ``--max-attempts > 1``.
  The CLI prints the ``RunConfig`` validation error and exits with code 2.
- ``--fail-fast`` cannot be used with multi-sample or feedback modes. The CLI
  prints: ``fail-fast is incompatible with multi-sample or feedback modes.``
- When ``--samples-per-task > 1`` and ``--temperature 0.0`` are both set, the
  CLI warns that all samples will be identical and suggests
  ``--temperature 0.8`` for Pass@K.
- ``--task-timeout`` must be a positive number when provided.
- ``--eval-timeout`` must be a positive number when provided.
- ``--resume-from`` must point to an existing ``.jsonl`` output file.
- A multi-job matrix cannot use Coda, ``--resume-from``, selective
  ``--rerun``/``--regrade``, or a user ``--configuration-id``.
- ``--reasoning-enabled`` cannot be combined with ``--registry``.

If any validation fails, the CLI prints an error message to standard error and
exits with code 2 without starting the run.


Exit Codes
----------

- **0**: The run completed successfully. Results were written to ``--out``.
- **2**: A CLI argument was missing or invalid. No run was attempted.

A non-zero exit code does *not* mean that all tasks failed. A run that completes
with a 0% pass rate still exits with code 0 because the run itself succeeded.
Check the ``summary.pass_rate`` field in the output to determine benchmark
results.


Examples
--------

Bounded check with the smoke provider:

.. code-block:: bash

    qceval run --provider smoke --framework qiskit --out results.json

CUDA-Q smoke test:

.. code-block:: bash

    qceval run --provider smoke --framework cudaq --out results.smoke.cudaq.json

Default framework evaluation:

.. code-block:: bash

    qceval run --provider smoke --framework all --out results.json

QEC behavior run:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --framework qiskit \
      --suite qec \
      --eval-timeout 20 \
      --fail-fast \
      --out results.qec.json

This command executes every QEC case and writes behavior-authoritative results.

Core plus QEC integration run:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --framework all \
      --suite all \
      --out results.full.json

Core and QEC records both use packaged behavior contracts; QEC contracts
execute every declared input case.

Real model evaluation with OpenRouter:

.. code-block:: bash

    qceval run \
      --provider openrouter \
      --openrouter-api-key <your-key> \
      --model anthropic/claude-sonnet-4.5 \
      --framework all \
      --temperature 0.0 \
      --timeout 180 \
      --out results.jsonl

Published Pass@1 matrix (ten models, 33 jobs):

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

This official shape omits ``--max-tasks``. Mixed-model sweeps omit OpenRouter
endpoint pins because endpoint tags differ by model.

High-throughput OpenRouter run with cache and JSONL streaming:

.. code-block:: bash

    uv run qceval run \
      --provider openrouter \
      --openrouter-api-key <your-api-key> \
      --model anthropic/claude-sonnet-4.5 \
      --framework all \
      --temperature 0.0 \
      --task-timeout 90 \
      --generation-concurrency 8 \
      --evaluation-workers 4 \
      --cache-dir .qceval-cache \
      --out results.jsonl

Coda Build run:

.. code-block:: bash

    uv run qceval run \
      --provider coda \
      --coda-api-key <your-coda-api-key> \
      --framework qiskit \
      --max-tasks 1 \
      --out results.coda.json

Coda Learn fast run with structured response preference:

.. code-block:: bash

    uv run qceval run \
      --provider coda \
      --coda-api-key <your-coda-api-key> \
      --coda-mode learn \
      --coda-fast \
      --coda-prefer-structured-response \
      --framework qiskit \
      --max-tasks 1 \
      --out results.coda.learn-fast.json

Pass@1, deterministic one-shot:

.. code-block:: bash

    uv run qceval run \
      --provider openrouter \
      --openrouter-api-key <your-api-key> \
      --model anthropic/claude-sonnet-4.5 \
      --framework all \
      --temperature 0.0 \
      --samples-per-task 1 \
      --pass-k 1 \
      --out results.pass1.jsonl

Pass@5 with explicit stochastic sampling:

.. code-block:: bash

    uv run qceval run \
      --provider openrouter \
      --openrouter-api-key <your-api-key> \
      --model anthropic/claude-sonnet-4.5 \
      --framework all \
      --temperature 0.8 \
      --samples-per-task 5 \
      --pass-k 5 \
      --generation-concurrency 8 \
      --evaluation-workers 4 \
      --cache-dir .qceval-cache \
      --out results.pass5.jsonl

Feedback repair with five total attempts:

.. code-block:: bash

    uv run qceval run \
      --provider openrouter \
      --openrouter-api-key <your-api-key> \
      --model anthropic/claude-sonnet-4.5 \
      --framework all \
      --temperature 0.2 \
      --max-attempts 5 \
      --generation-concurrency 8 \
      --evaluation-workers 4 \
      --cache-dir .qceval-cache \
      --out results.feedback5.jsonl

Quick three-task smoke test:

.. code-block:: bash

    qceval run \
      --provider smoke \
      --framework qiskit \
      --max-tasks 3 \
      --out results.json

Regrade stored JSON or JSONL locally (no model call):

.. code-block:: bash

    uv run qceval run \
      --regrade all \
      --suite all \
      --input previous.jsonl \
      --out regraded.jsonl
