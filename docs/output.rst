Output Schema
=============

QCircuitEval writes a JSON document or newline-delimited JSONL. Both formats
carry the same benchmark records and semantic grading data. The run envelope
schema is ``qceval.run.v2``; each embedded behavior result has its own versioned
result schema.


Choosing a Format
-----------------

``--output-format`` selects ``json`` or ``jsonl``. With ``auto`` (the default),
the ``.json`` or ``.jsonl`` extension selects the format.

JSON stores one top-level object:

.. code-block:: json

   {
     "schema_version": "qceval.run.v2",
     "run_id": "3cae...",
     "provider": "openrouter",
     "model": "example/model",
     "suites": ["core"],
     "qceval": {
       "source": "bundled-qceval",
       "package_version": "0.1.0",
       "commit": "<checkout commit or null>",
       "commit_status": "available",
       "dirty": false,
       "source_hint": null,
       "path": null,
       "branch": null
     },
     "results": [],
     "summary": {}
   }

The ``qceval`` object identifies the installed package version used to load the
bundled tasks. Runs imported directly from a source checkout also include its
Git commit and whether tracked or untracked source changes are present.
Installed wheels leave ``commit`` as ``null``, set ``commit_status`` to
``"unavailable"``, report ``dirty`` as ``null``, and do not depend on or embed
an external asset path. Optional ``source_hint``, ``path``, and ``branch``
fields record provenance when supplied. Pin runs may also surface
``configuration_id`` on the summary envelope.

JSONL writes each result with ``"kind": "result"`` and finishes with one
``"kind": "summary"`` envelope. Concurrent runs may stream completion order,
but the final in-memory payload and JSON output retain deterministic task order.


Benchmark Result Records
------------------------

Each task attempt produces a benchmark record:

.. code-block:: json

   {
     "framework": "qiskit",
     "suite": "core",
     "task_id": "01",
     "sample_index": 0,
     "attempt_index": 0,
     "entry_point": "example_entry_point",
     "category": "example",
     "provider": "smoke",
     "model": "smoke-canonical",
     "status": "passed",
     "feedback": {},
     "request_trace": {},
     "lineage": {},
     "provider_response": {},
     "evaluation": {},
     "error_taxonomy": {}
   }

The benchmark-level ``status`` is retained for runner compatibility:

``generated``
    Fresh candidate code was written without evaluation. Selective
    ``--rerun``/``--prompt``-only frameworks use this status so the response can
    be graded later.

``passed``
    Evaluation completed and its semantic status was ``verified_pass``.

``failed``
    Evaluation completed but did not produce ``verified_pass``. Inspect the
    semantic status to distinguish a behavior mismatch from an execution or
    resource failure.

``provider_failed``
    The provider did not return nonempty candidate code. Empty and
    whitespace-only responses are provider failures and are not evaluated.

``compile_failed``
    Candidate source could not be compiled.

``run_failed``
    Candidate execution or the evaluation boundary raised an error.

``infrastructure_error``
    Provider transport, worker isolation, or another infrastructure boundary
    failed before a normal evaluation outcome could be recorded. Official
    leaderboard rows treat these as requiring a rerun.

``sample_index`` and ``attempt_index`` identify Pass@K samples and feedback
repair attempts. Provider response, usage, raw-response, and feedback fields are
unchanged by behavior grading. Published Pass@1 envelopes in
``results/published`` omit ``raw_response`` and generation ids so the public
tree does not carry vendor telemetry; token usage and reported cost remain.

``request_trace`` has schema ``qceval.request_trace.v1``. It records the exact
ordered ``role`` and ``content`` messages presented to the provider, plus stable
SHA-256 hashes of the original prompt and serialized transcript. This makes the
initial prompt and every full-transcript repair request directly auditable.

``lineage`` has schema ``qceval.feedback_lineage.v1``. It records the top-level
``run_id``, deterministic ``chain_id``, current and parent attempt indices,
current and parent code hashes, the feedback source attempt, feedback policy
version, terminal flag, and stop reason. A resumed JSONL run preserves its
existing run ID. Legacy resume files receive a deterministic run ID derived
from their contents.

``error_taxonomy`` is a versioned, multi-label classification derived from the
stored benchmark status and semantic evidence. It is present even when
``evaluation`` is absent. See :doc:`error_taxonomy` for the seven axes,
exclusions, and denominator rules.


Evaluation Object
-----------------

An evaluated record contains a :class:`~qceval.models.QCEvalEvaluation`:

.. code-block:: json

   {
     "compiled": true,
     "ran": true,
     "passed": true,
     "metric": 0.0,
     "metric_name": "hellinger_infidelity",
     "probabilities": [1.0, 0.0, 0.0, 0.0],
     "execution_metadata": {},
     "grader_details": {
       "passed": true,
       "verified_status": "verified_pass",
       "semantic_status": "verified_pass",
       "reason": "all_required_routes_passed",
       "score_authority": "behavior",
       "behavior_verdict": {
         "passed": true,
         "source": "behavior",
         "semantic_status": "verified_pass"
       },
       "semantic_verification": {}
     },
     "verified_status": "verified_pass",
     "semantic_result": {},
     "error": null,
     "error_type": null
   }

``compiled`` and ``ran``
    Describe candidate execution, not semantic correctness.

``passed``
    The fail-closed binary projection. It is true only for semantic
    ``verified_pass``.

``verified_status``
    A compatibility projection: ``verified_pass`` or ``verified_fail`` for new
    runtime records. Use ``semantic_result.status`` for the precise outcome.

``metric`` and ``metric_name``
    The first metric-bearing verifier evidence item, if one exists. Metrics are
    engine-specific and may be absent. Distribution contracts report Hellinger
    infidelity; state, unitary, and channel contracts report their own metrics.
    No single metric is globally authoritative across engines.

``probabilities`` and ``execution_metadata``
    Observations produced by the unchanged framework executor. They support
    semantic engines and diagnostics but do not independently determine a pass.

``grader_details``
    The behavior integration projection. ``score_authority`` and
    ``behavior_verdict.source`` are ``"behavior"``. The nested
    ``semantic_verification`` is the serialized semantic result record.

``semantic_result``
    The same semantic result promoted to a stable first-class evaluation field.

``error`` and ``error_type``
    Bounded failure detail and the runner's compatibility error category.


Semantic Result
---------------

``semantic_result.status`` is one of:

* ``verified_pass``;
* ``semantic_fail``;
* ``execution_error``; or
* ``resource_limit``.

Only ``verified_pass`` has ``semantic_result.passed == true``. Readers must not
treat another status as a pass or discard it from the strict denominator.

A current semantic result has this shape:

.. code-block:: json

   {
     "result_schema_version": "3",
     "status": "verified_pass",
     "reason_code": "all_required_routes_passed",
     "summary": "all required routes passed",
     "passed": true,
     "authoritative": true,
     "contract": {
       "suite": "core",
       "task_id": "01",
       "schema_version": "1",
       "contract_version": "1.0.0",
       "hash": "..."
     },
     "target": {"version": "1.0.0", "hash": "..."},
     "ir": {"version": "...", "semantic_hash": "..."},
     "verifier": {
       "release_version": "1.0.0",
       "engines": [{"name": "distribution", "version": "1.1.0"}],
       "metric": "hellinger_infidelity",
       "tolerance": 1e-9
     },
     "evidence": [],
     "requirements": [],
     "diagnostics": [],
     "resources": {
       "wall_seconds": 0.0,
       "peak_rss_mib": null,
       "evidence_truncated": false
     },
     "environment": {
       "python": "3.11.0",
       "framework": "qiskit",
       "platform": "darwin"
     }
   }

Evidence entries identify the engine and version, reason code, input and target
hashes, optional metric/value/tolerance/uncertainty, cases checked, elapsed
time, peak memory, and preconditions. Text and evidence counts are bounded.


QEC Semantic Results
--------------------

QEC uses the same top-level semantic result shape. Its contract identity has
``schema_version: "2"``, the shared QEC contract hash, and an independently
verified target hash. Exhaustive contracts add ``num_cases`` to compatibility
details. Per-binding metrics and requirement outcomes appear in the normal
bounded evidence array, with the concrete arguments recorded as evidence
preconditions. A QEC task passes only when every declared binding produces a
decisive ``verified_pass``.


Summary
-------

The compatibility summary retains:

* ``total_tasks``, ``assigned_tasks``, ``passed``, ``failed``, and
  ``pass_rate``;
* provider, compile, and runtime failure counts;
* infrastructure-failure, rerun-required, and compatibility
  ``scoreable_tasks`` counts;
* ``verified_status_counts``;
* framework, suite, and suite/framework breakdowns;
* compact task rows;
* run protocol, task totals, Pass@K, feedback, and feedback-lineage summaries
  where applicable;
* provider-reported USD cost totals, coverage, and mean cost per logical task;
  and
* seven-axis error-taxonomy counts, rates, and mapping coverage.

``pass_rate`` is strict because ``passed`` counts only ``verified_pass`` and
the denominator is ``assigned_tasks``. Infrastructure failures remain in this
headline denominator so a non-decision can never improve a score; they are also
reported separately as requiring a rerun. ``scoreable_tasks`` is retained for
operational compatibility and is not the headline score denominator.

``summary.cost`` reports the provider-record and logical-task coverage of
``provider_response.usage.cost_usd``. The top-level
``mean_reported_cost_per_task_usd`` is emitted only when every provider record
for every logical task has a reported cost. Partial totals remain visible, but
missing cost is never treated as zero. For Pass@K and feedback protocols, one
logical task includes all samples or repair attempts for the same
``(suite, framework, task_id)``.

For scientific reporting, this strict Pass@1 rate is the primary one-shot
endpoint. Show Core and QEC separately and show each framework's rate before a
within-suite cross-framework aggregate. Records that share
``(suite, task_id)`` across frameworks are correlated observations. The
authoritative reporting and task-cluster analysis rules are in
:doc:`leaderboard`.

``summary.error_taxonomy`` reports ``axis_counts`` and ``axis_rates`` for all
seven axes. Every rate uses ``assigned`` as its denominator. The classification
is multi-label, so axis counts can overlap. ``classification_coverage`` is the
fraction of observed-error records with at least one mapped axis. Outcome,
unclassified-reason, taxonomy-version, and per-framework counts remain
available for audit and stratification. Per-suite and suite/framework strata
use their own assigned denominators. ``axis_labels`` supplies compact radar
labels without changing the stable axis identifiers.

Feedback Lineage Summary
------------------------

When ``max_attempts > 1``, ``summary.feedback_lineage`` reports the repair
protocol at the chain level. ``assigned_chains`` is the common denominator for
all unconditional rates. ``terminal_pass_rate`` is therefore verified terminal
passes divided by all assigned chains, including provider failures, resource
limits, grader non-decisions, invalid chains, and interrupted chains.

``levels`` reports, for the initial generation and each repair level:

* chains attempted at that level;
* first verified passes at that level;
* first-pass hazard, defined as first passes divided by chains attempted at the
  level; and
* cumulative verified passes and cumulative pass rate over all assigned chains.

``terminal_stop_reason_counts`` separates ``verified_pass``,
``max_attempts_exhausted``, ``provider_failure``, ``resource_limit``,
``grader_nondecision``, invalid chains, and incomplete chains. Token totals are
provider-reported; the mean is emitted only for chains with a reported total on
every attempt. Prompt, completion, reasoning, and cached-token aggregates carry
their own reporting coverage so absent provider fields are not confused with
measured zeros.

``taxonomy_transitions`` compares the initial and terminal seven-axis
classifications. Each axis reports cleared, persistent, surfaced, absent, and
unknown counts. Unconditional transition rates use all assigned chains.
Conditional clearance and persistence use only paired, classifiable chains
where the axis was initially present. Censored, invalid, incomplete,
unclassified, and incompatible-taxonomy endpoints remain ``unknown`` rather
than being counted as error absence. Coverage fields and taxonomy-version
counts must accompany any plotted transition rates.

For paper plots, ``taxonomy_transitions.groups`` also reports execution,
algorithmic, and semantic error families. Each family is the union of its
constituent axes at each chain endpoint, so a chain with two algorithmic axes
counts once in the algorithmic bar. A chain can still contribute to more than
one family.

``taxonomy_transitions.diverging_plot`` is a renderer-ready view of these
families. Cleared incidence is positive and surfaced incidence is negative.
The unit is error-family transitions per assigned feedback chain, expressed as
percentage points for plotting. The diamond statistic is the sum of family
clearance rates minus the sum of family surfaced rates. Persistent and unknown
incidence remain separate and do not enter the signed delta. Because the
families are multi-label, the stacked incidence can exceed 100 percent.

The diamond includes a deterministic 95 percent task-cluster percentile
bootstrap interval using 10,000 resamples and seed 0. Clusters are identified
by ``(suite, task_id)`` so framework or sample observations of the same task
remain together. The interval is unavailable, with an explicit reason, when a
report contains fewer than two task clusters. Classification coverage must be
shown with the plotted estimate because unknown endpoints are not treated as
error-free.

When semantic metadata is present, ``summary.semantic`` adds:

``assigned``
    All assigned result records.

``status_counts``
    Counts for each observed four-state status.

``strict_pass_rate``
    ``verified_pass / assigned``.

``coverage``
    ``(verified_pass + semantic_fail) / assigned``.

``adjudicated_pass_rate``
    ``verified_pass / (verified_pass + semantic_fail)``. This rate must be read
    with coverage and must not replace the strict score.

``nonsemantic_counts``
    Execution-error, resource-limit, and ungraded counts.

The semantic summary also includes status transitions, version groups,
compatibility warnings, per-framework status counts, and performance
percentiles. If the same task appears with incompatible contract, target, or
verifier identities, readers should stratify those groups rather than combine
them into one score.


Effort-Sweep Manifest
---------------------

``--reasoning-effort all`` and ``--registry`` matrix runs write a separate run
artifact for every expanded configuration plus a manifest:

.. code-block:: json

   {
     "schema_version": "qceval.effort_sweep.v1",
     "jobs": [
       {
         "model": "openai/gpt-5.6-sol",
         "reasoning_effort": "max",
         "configuration_id": "openai-gpt-5-6-sol__effort-max",
         "out": "results/openai-gpt-5-6-sol__effort-max.json",
         "exit_code": 0
       }
     ]
   }

File-shaped output such as ``results.json`` produces per-effort files and
``results.efforts.json``. Directory-shaped output produces
``<configuration_id>.json`` files and ``manifest.json``. The jobs are
independent Pass@1 runs. Consumers must not concatenate them into a single
score.


Compatibility and Stability
---------------------------

``qceval.run.v2`` readers can resume compatible v1 JSONL rows. Missing sample,
attempt, and feedback fields receive historical defaults. The semantic result
reader accepts supported historical semantic schemas and migrates them to the
current record shape for reporting. Historical ``unsupported`` and
``inconclusive`` statuses become ``execution_error`` with the old status
retained as a diagnostic, without making historical results
authoritative.

Within a run schema version, consumers should tolerate additive object fields,
ignore object key order, and use explicit task/sample/attempt identities. They
should use ``semantic_result.status`` and version identity for new scoring
analysis, not infer authority from ``canonical_class``, metric names, or
probability arrays.
