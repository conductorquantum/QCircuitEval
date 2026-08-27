# QCircuitEval

[![Tests](https://img.shields.io/github/actions/workflow/status/conductorquantum/QCircuitEval/tests.yml?branch=main&label=Tests&logo=github)](https://github.com/conductorquantum/QCircuitEval/actions/workflows/tests.yml)
[![Quality](https://img.shields.io/github/actions/workflow/status/conductorquantum/QCircuitEval/quality.yml?branch=main&label=Quality&logo=github)](https://github.com/conductorquantum/QCircuitEval/actions/workflows/quality.yml)
![Python >=3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

QCircuitEval is a **beta** deterministic behavior-contract benchmark for
LLM-generated quantum circuit code across Qiskit, Cirq, PennyLane, and CUDA-Q.
It includes 58 core and 12 QEC tasks per framework. Scores and APIs may still
change between published revisions.

## Install

```bash
uv sync --extra dev --extra docs
```

## Run

Bounded local check across all frameworks:

```bash
uv run qceval run --provider smoke --framework all \
  --max-tasks 1 --eval-timeout 10 --fail-fast --out results.json
```

One OpenRouter model:

```bash
uv run qceval run --provider openrouter \
  --model openai/gpt-5.6-sol --reasoning-effort max \
  --framework all --suite all --temperature 0.0 --out results.jsonl
```

The published Pass@1 model/effort matrix through the same entry point:

```bash
uv run qceval run --provider openrouter \
  --registry production/models.prompt-effort-sweep.json \
             production/models.max-reasoning.json \
  --reasoning-effort all --framework all --suite all \
  --temperature 0.0 --out results/
```

Set `OPENROUTER_API_KEY` or pass `--openrouter-api-key`. Each expanded job
writes its own result and score; the manifest indexes those jobs. Do not pool
their records into one rate.

## Local grading

`qceval run` grades as it generates. `--suite` defaults to `core`; pass
`--suite all` when the artifact includes QEC. `--input` accepts JSONL
streams and JSON run envelopes (including `results/published/*.json`).
`--provider` defaults to `smoke`; no API key is required:

```bash
uv run qceval run \
  --regrade all \
  --suite all \
  --input previous.jsonl \
  --out regraded.jsonl
```

Official leaderboard scoring is a separate trusted-regrade path:
`uv run python scripts/score_submission.py …`. See [CLI](docs/cli.rst) and
[Grader](docs/grader.rst).

## Contributing

Pull requests are welcome, including changes to the grader: contracts,
targets, verifier engines, framework lowering, and hard requirements. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
Report security issues through [SECURITY.md](SECURITY.md), not public issues.

## Documentation

- [Quickstart](docs/quickstart.rst)
- [CLI and Pass@1 matrices](docs/cli.rst)
- [Grader](docs/grader.rst)
- [Output schemas](docs/output.rst)
- [Leaderboard protocol](docs/leaderboard.rst)
- [Python API](docs/api.rst)
- [Contributing](docs/contributing.rst)
