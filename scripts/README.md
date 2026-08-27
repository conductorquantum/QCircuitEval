# Scripts

User-facing command-line tools. Run them through the project environment:

```bash
uv run python scripts/<name>.py ...
```

- `score_submission.py` — validate leaderboard protocol and metadata, then
  locally regrade each candidate in a fresh, timeout-bounded worker with the
  bundled adapter before scoring. Everyday regrade of a run artifact uses
  `qceval run --regrade all --suite all --input previous.jsonl --out regraded.jsonl`,
  not this script. `--input` also accepts JSON run envelopes.
- `merge_run_records.py` — merge compatible framework-sharded JSONL runs,
  reject incomplete/conflicting inputs, and recompute a single valid summary.
- `analyze_effort_sweep.py` — emit effort/cost/token/failure tables and plots
  from a complete prompt-effort analysis directory. The input artifacts can be
  produced by one registry-expanded ``qceval run``; each manifest job remains a
  separate score.
- `plot_effort_price_performance.py` — render price/performance figures from
  scored sweep summaries.
- `summarize_run_costs.py` — report each model/protocol score alongside its
  provider-reported average USD cost per logical task.
- `hash_benchmark_content.py` — hash packaged prompts, contracts, and targets
  for reproducibility identities.

Pinned OpenRouter model registries live in [`production/`](../production/).
Maintainer utilities live under [`ci/`](../ci/).
