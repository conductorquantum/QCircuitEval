# Contributing

QCircuitEval accepts pull requests. Grader work is in scope: behavior
contracts, independently derived targets, verifier engines, framework
lowering to Program IR, and hard structural requirements.

This file is the contributor checklist. GitHub shows it on new pull
requests. The Sphinx page `docs/contributing.rst` points here. Participation
is covered by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security reports go
through [SECURITY.md](SECURITY.md).

## Local setup

```bash
uv sync --extra dev --extra docs
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run sphinx-build -W -b html docs docs/_build/html
```

Quality CI also regenerates and checks packaged semantic assets. After
contract or target edits:

```bash
uv run python ci/generate_semantic_assets.py --check
uv run python ci/generate_qec_semantic_assets.py --check
uv run python ci/check_asset_consistency.py
uv run python ci/check_packaged_assets_wheel.py
uv run qceval contracts validate --suite core
uv run qceval contracts validate --suite qec
```

Regenerate with the same scripts without `--check` when hashes must update,
then inspect the semantic diff. Changing a prompt, contract, or target is a
benchmark revision: canonical solutions across all four frameworks must still
`verified_pass`.

## Local grading

A normal `qceval run` generates and grades. `--suite` defaults to `core`, so
pass `--suite all` when the artifact includes QEC. `--input` accepts JSONL
streams and JSON run envelopes (including `results/published/*.json`).
`--provider` defaults to `smoke`; omit OpenRouter credentials on regrade-only
runs:

```bash
uv run qceval run \
  --regrade all \
  --suite all \
  --input previous.jsonl \
  --out regraded.jsonl
```

Leaderboard acceptance uses `scripts/score_submission.py`, which regrades
every candidate in a fresh worker. That is the protocol scorer, not the
everyday CLI regrade.

## Grader contributions

Authoritative scoring lives under `src/qceval/semantics/` and the packaged
assets in `src/qceval/assets/`. Framework executors and lowering adapters
live under `src/qceval/frameworks/<name>/`.

Useful entry points:

- Contracts and targets: `src/qceval/assets/contracts/`,
  `src/qceval/assets/targets/`, and the generators in `ci/`.
- Engines and routing: `src/qceval/semantics/verifiers/` and the default
  portfolio in `src/qceval/semantics/portfolio.py`.
- Lowering: `src/qceval/semantics/lowering/` plus each framework adapter.
- Hard requirements: `src/qceval/semantics/verifiers/requirements/`.
- Design notes: `docs/grader.rst`, `src/qceval/assets/README.md`.

To add an engine, implement `VerifierEngine` under
`src/qceval/semantics/verifiers/`, register it on `DefaultSemanticVerifier`
in `src/qceval/semantics/portfolio.py`, point a contract route at the
engine's descriptor name, and add tests under `tests/semantics/`. Canonical
solutions for all four frameworks must still `verified_pass`.

Keep these invariants:

- Only `verified_pass` passes. Do not add a fallback grader, a fallback
  contract route, or a third behavioral verdict.
- Contracts are keyed by `(suite, task_id)` and shared across frameworks.
  One registered primary route, `cross_check` false, empty `fallback`.
- Providers must never see contracts, targets, canonical solutions, or
  other oracle fields.
- Fail closed: inability to lower, decide, or stay inside resource limits
  is `execution_error` or `resource_limit`, not a silent pass or fail.

Lowering changes should preserve faithful Program IR or report
`execution_error`.

## Pull requests

Open a PR against `main`. Tests and Quality workflows must pass. Keep
production modules cohesive; see `docs/production_inventory.rst` before
growing already-large files. Do not commit API keys, `.env`, or local
result caches.
