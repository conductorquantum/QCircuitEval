Quickstart
==========

This page walks through installing QCircuitEval, running your first benchmark
with the built-in smoke provider, and then switching to a real model through
OpenRouter. For the design philosophy behind the grading system, see
:doc:`index`.


Requirements
------------

QCircuitEval requires Python 3.11 or later. It uses `uv
<https://docs.astral.sh/uv/>`_ for dependency management, but any
PEP 517-compatible installer will work.


Installation
------------

Clone the repository and install with development extras:

.. code-block:: bash

    git clone <repo-url>
    cd QCircuitEval
    uv sync --extra dev --extra docs

This installs the ``qceval`` package in editable mode along with Qiskit,
Cirq, PennyLane, CUDA-Q, and the development tools (pytest, ruff, mypy,
coverage, and Sphinx).
The lockfile pins matching CUDA-Q metadata and runtime packages so
``import cudaq`` is available after sync. GPU acceleration still depends on
NVIDIA's supported platform and driver stack; the default CPU target is enough
for bundled smoke tests.


Your First Run
--------------

The fastest way to verify the installation is to run the ``smoke`` provider
against one task per framework:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --framework all \
      --max-tasks 1 \
      --eval-timeout 10 \
      --fail-fast \
      --out results.json

The smoke provider does not call any external API. In ``canonical`` mode (the
default), it returns the bundled Qiskit canonical solution where present and
generates deterministic compatibility responses from task metadata otherwise.
The ``--max-tasks``, ``--eval-timeout``, and ``--fail-fast`` flags keep this
smoke path bounded. The output is written to ``results.json``.

The command prints a compatibility summary and writes semantic status counts to
``summary.semantic``. Canonical smoke output is an integration diagnostic, not
an assertion that every candidate can be verified. Inspect the four-state
status counts and detailed reason code whenever a task does not pass.


Running All Frameworks
----------------------

To evaluate across the default ``all`` framework set, use ``--framework all``:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --framework all \
      --out results.json

This loads the Qiskit, Cirq, PennyLane, and CUDA-Q core task sets (58 tasks
each, 232 total) and evaluates them sequentially. The output includes
per-framework breakdowns in the ``summary.by_framework`` section.

CUDA-Q can still be selected alone:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --framework cudaq \
      --out results.smoke.cudaq.json

CUDA-Q smoke uses the active CUDA-Q target. The default CPU target is enough for
normal development. To prove a GPU-capable target on a configured NVIDIA host,
run the opt-in live target test:

.. code-block:: bash

    QCEVAL_LIVE_CUDAQ_TARGET=nvidia \
      uv run pytest tests/test_live_cudaq_target.py -q


Running QEC
-----------

The default suite is ``core``. Select ``--suite qec`` to run the 12 QEC tasks
for each selected framework. The evaluator executes every declared input case
and verifies each one against the packaged QEC behavior contract:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --framework qiskit \
      --suite qec \
      --eval-timeout 20 \
      --fail-fast \
      --out results.qec.json

With ``--suite all``, core tasks run first and QEC tasks run second:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --framework all \
      --suite all \
      --out results.full.json

Keep core and QEC rates separate when publishing results. Both suites use
semantic contracts and independent targets; QEC contracts additionally
enumerate their finite input domains exhaustively.


Limiting the Number of Tasks
----------------------------

During development or debugging, you may want to run only a subset of tasks.
The ``--max-tasks`` flag limits how many tasks are evaluated per framework:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --framework all \
      --max-tasks 3 \
      --out results.json

This evaluates three tasks per framework (12 total) instead of the full 232.


Using a Real Model via OpenRouter
---------------------------------

To benchmark a real code-generation model, use the ``openrouter`` provider. You
need an `OpenRouter <https://openrouter.ai/>`_ API key and a model identifier:

.. code-block:: bash

    uv run qceval run \
      --provider openrouter \
      --openrouter-api-key <your-api-key> \
      --model anthropic/claude-sonnet-4.5 \
      --framework qiskit \
      --out results.json

An OpenRouter credential and ``--model`` are required. When the credential flag
is omitted, the CLI checks ``OPENROUTER_API_KEY`` in the environment and then
``.env`` in the current working directory. Explicit credential flags take
precedence.

For example, a local ``.env`` file may contain:

.. code-block:: text

    OPENROUTER_API_KEY=

The repository ignores ``.env``. Do not commit API keys.

You can also tune the generation parameters:

.. code-block:: bash

    uv run qceval run \
      --provider openrouter \
      --openrouter-api-key <your-api-key> \
      --model anthropic/claude-sonnet-4.5 \
      --framework qiskit \
      --temperature 0.0 \
      --timeout 180 \
      --out results.json

See :doc:`cli` for the full list of flags and their defaults.

For larger runs, enable bounded provider concurrency and a response cache:

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

JSONL output is streamed as tasks complete, and the final payload remains sorted
by framework and task ID. ``--task-timeout`` isolates each task so one stuck
provider request does not block the full run.


Running a Pass@1 Matrix
-----------------------

The same ``qceval run`` entry point expands model registries and reasoning
efforts. This bounded smoke matrix runs one task per framework for all ten
published models and 33 distinct model/effort jobs:

.. code-block:: bash

    uv run qceval run \
      --provider smoke \
      --registry production/models.prompt-effort-sweep.json \
                 production/models.max-reasoning.json \
      --reasoning-effort all \
      --framework all \
      --max-tasks 1 \
      --eval-timeout 10 \
      --fail-fast \
      --out results/

For an official OpenRouter run, change the provider, supply credentials, add
``--suite all --temperature 0.0``, and omit ``--max-tasks``. Mixed-model sweeps
omit endpoint pins because provider endpoint tags differ by model. Each job has
its own result file and score; do not pool jobs into one rate. The directory's
``manifest.json`` indexes every expanded configuration.


Choosing an Output Format
--------------------------

By default, the output format is inferred from the file extension. A ``.json``
extension produces a single JSON document; a ``.jsonl`` extension produces one
JSON object per line:

.. code-block:: bash

    # JSON output (single document)
    uv run qceval run --provider smoke --framework qiskit --out results.json

    # JSONL output (one line per result, summary envelope at the end)
    uv run qceval run --provider smoke --framework qiskit --out results.jsonl

You can override the inference with ``--output-format json`` or
``--output-format jsonl``. See :doc:`output` for the full schema.


Local grading
-------------

Every ``qceval run`` above grades as it generates. ``--suite`` defaults to
``core``; pass ``--suite all`` when the original run included QEC. ``--input``
accepts JSONL streams and JSON run envelopes (including published
``results/published/<configuration_id>.json`` files):

.. code-block:: bash

    uv run qceval run \
      --regrade all \
      --suite all \
      --input results.jsonl \
      --out regraded.jsonl

``--provider`` defaults to ``smoke``, so this path needs no API key. See
:doc:`cli` for mixed prompt/regrade phases and :doc:`grader` for verdicts.
Leaderboard submissions use ``scripts/score_submission.py`` instead; see
:doc:`leaderboard`.


Verifying the Installation
--------------------------

The project ships a comprehensive test suite. To run it:

.. code-block:: bash

    uv run pytest --cov=src/qceval --cov-report=term-missing

All tests should pass with at least 85% code coverage. You can also run the
linter and formatter checks:

.. code-block:: bash

    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run sphinx-build -W -b html docs docs/_build/html


Where to Go Next
-----------------

- :doc:`cli` documents every flag in detail, including ``--regrade``.
- :doc:`providers` explains how providers work and how to write your own.
- :doc:`grader` covers contracts, engines, and local regrade.
- :doc:`evaluation` covers the grading pipeline.
- :doc:`output` describes the output schema.
- :doc:`contributing` explains how to change the grader.
