# Published Pass@1 results

This directory is the canonical public result set: 9,240 records in 33
independent configurations across 10 models. It has exactly the shape
`qceval run` writes for a directory destination, so the same tooling reads
published and freshly generated results:

- `<configuration_id>.json` — one `qceval.run.v2` run envelope per
  configuration, holding 280 results (Core and QEC across all four frameworks)
  and the run summary.
- `manifest.json` — the `qceval.effort_sweep.v1` matrix manifest listing every
  job's model, reasoning effort, configuration ID, output path, and exit code.
- `provenance.json` — the campaign each artifact came from, plus per-artifact
  record counts, sizes, and SHA-256 digests.

The set combines all 28 configurations from the latest complete prompt-effort
campaign with the five non-overlapping configurations from the latest complete
maximum-reasoning campaign. Where the campaigns overlap, the newer prompt-effort
artifact is authoritative. All artifacts are one-sample, one-attempt Pass@1
results with no unresolved infrastructure failures.

Because these envelopes were rebuilt from offline-regraded campaign shards, they
carry every field of a live run except `run_identity`, which the merge step does
not reconstruct. Public artifacts also omit provider `raw_response` payloads and
generation ids; token usage and reported cost remain.

GPT-5.6-sol `low` and `max` generation costs are reported at the later OpenRouter
list price used by the rest of the sol ladder ($15/M output tokens, $2.50/M
prompt tokens). Those two jobs were originally billed at twice that rate; token
counts are unchanged.

The same 33-configuration matrix can be expanded with:

```bash
uv run qceval run --provider openrouter \
  --registry production/models.prompt-effort-sweep.json \
             production/models.max-reasoning.json \
  --reasoning-effort all --framework all --suite all \
  --temperature 0.0 --out results/
```

Official runs omit `--max-tasks`. Mixed-model runs omit OpenRouter endpoint
pins because tags differ by model.

Every manifest job is a separate Pass@1 configuration. Do not concatenate jobs
or pool them into one score. Report Core and QEC separately, and show each
framework before any within-suite aggregate.

To regrade one published envelope locally without calling a model:

```bash
uv run qceval run \
  --regrade all \
  --suite all \
  --input results/published/<configuration_id>.json \
  --out regraded.jsonl
```
