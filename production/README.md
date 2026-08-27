# Model registries

OpenRouter model registries used for published Pass@1 runs:

- `models.prompt-effort-sweep.json` — five-model effort ladder
- `models.max-reasoning.json` — eight-model maximum-reasoning roster
- `models.effort-sweep.json` — archived effort-sweep roster

The published union is ten models and 33 distinct jobs:

```bash
uv run qceval run --provider openrouter \
  --registry production/models.prompt-effort-sweep.json \
             production/models.max-reasoning.json \
  --reasoning-effort all --framework all --suite all \
  --temperature 0.0 --out results/
```

Official runs omit `--max-tasks`. Mixed-model invocations omit endpoint pins
because tags differ by model. Each output file is its own score; do not pool
the jobs.

## EC2 runners

These scripts are the production worker pool. They are not part of `qceval run`.

`production/run_aws.sh` generates and grades on a temporary EC2 pool. It copies
`OPENROUTER_API_KEY` onto the workers, keeps network, and runs framework-sharded
`qceval run` jobs from a models file or a frozen TSV queue. Default region is
`us-east-1`. Use `--plan-only` to write `queue.tsv` without launching instances.

`production/run_offline_aws.sh` is the isolated grader. It regrades stored
candidates with `--regrade` / `--input` and never places an API key on the
workers. After provisioning it attaches an evaluation security group with no
egress, then refuses to continue if OpenRouter is still reachable. The pool is
fixed at six `c7i.2xlarge` workers (default region `us-west-1`). One worker
calibrates `--evaluation-workers` at 2/4/8; `scripts/select_offline_workers.py`
picks the fastest setting whose verdicts match the two-worker baseline.
CUDA-Q shards instead default to one evaluation worker and a 900-second
per-candidate safety ceiling because concurrent JIT compilation can contend
for CPU and memory. The scheduler does not start that clock until a worker
slot is available. A shard containing `EvaluationTimeout` is rejected rather
than published; rerun it after diagnosing the grader capacity. Valid shards
are merged into `<configuration_id>__pass1.regraded.jsonl` files, which is the
input `scripts/publish_latest_results.py` consumes.

Both runners require a clean tracked checkout, an Ubuntu AMI that assigns
public IPs, and SSH from the controller. Offline grading additionally requires
the controller IPv4 `/32` as the only SSH ingress on both security groups.
