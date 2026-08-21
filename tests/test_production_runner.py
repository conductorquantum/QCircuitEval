from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
RUNNER = REPO_ROOT / "production" / "run_aws.sh"
MODEL_MANIFEST = REPO_ROOT / "production" / "models.max-reasoning.txt"
EFFORTS = ("max", "xhigh", "high", "medium", "low", "minimal", "none")


def _run_plan(tmp_path: Path, input_flag: str, input_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(RUNNER),
            input_flag,
            str(input_path),
            "--out-dir",
            str(tmp_path / "output"),
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _render_reasoning_args(setting: str) -> list[str]:
    source = RUNNER.read_text(encoding="utf-8")
    helpers = source.split('REPO_ROOT="', 1)[0]
    command = f'{helpers}\nargs=()\nappend_reasoning_args "{setting}" args\nprintf "%s\\n" "${{args[@]}}"\n'
    completed = subprocess.run(
        ["bash"],
        cwd=REPO_ROOT,
        input=command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.splitlines()


def test_max_reasoning_manifest_expands_to_explicit_jobs(tmp_path: Path) -> None:
    manifest_models = [
        line.split()[0]
        for line in MODEL_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected_jobs = len(manifest_models) * 3 * 4

    completed = _run_plan(tmp_path, "--models", MODEL_MANIFEST)

    assert completed.returncode == 0, completed.stderr
    assert f"planned {expected_jobs} jobs" in completed.stdout
    rows = [line.split("\t") for line in (tmp_path / "output" / "queue.tsv").read_text().splitlines()]
    assert len(rows) == expected_jobs
    assert {row[1] for row in rows} == set(manifest_models)
    assert all(row[2] != "default" for row in rows)
    for row in rows:
        flags = _render_reasoning_args(row[2])
        if row[2] == "enabled":
            assert flags == ["--reasoning-enabled"]
        else:
            assert flags == ["--reasoning-effort", row[2]]


@pytest.mark.parametrize("setting", ["default", "garbage"])
def test_models_reject_implicit_or_malformed_reasoning_setting(tmp_path: Path, setting: str) -> None:
    manifest = tmp_path / "models.txt"
    manifest.write_text(f"example/model {setting}\n", encoding="utf-8")

    completed = _run_plan(tmp_path, "--models", manifest)

    assert completed.returncode == 2
    assert f"invalid reasoning setting for example/model: {setting}" in completed.stderr


@pytest.mark.parametrize("setting", ["default", "garbage"])
def test_prebuilt_queue_rejects_implicit_or_malformed_reasoning_setting(tmp_path: Path, setting: str) -> None:
    queue = tmp_path / "input.tsv"
    queue.write_text(f"job-1\texample/model\t{setting}\tpass1\tqiskit\tcore\t0\n", encoding="utf-8")

    completed = _run_plan(tmp_path, "--queue", queue)

    assert completed.returncode == 2
    assert f"invalid reasoning setting on queue line 1: {setting}" in completed.stderr


def test_endpoint_pinned_fifteen_column_queue_is_accepted(tmp_path: Path) -> None:
    queue = tmp_path / "input.tsv"
    queue.write_text(
        "job-1\tx-ai/grok-4.6\txhigh\tpass1\tqiskit\tall\t0\txai\t128000\t"
        "benchmark_floor\tundisclosed_first_party_exception\tmax_tokens\troute-01\tnot_exposed\t70\n",
        encoding="utf-8",
    )

    completed = _run_plan(tmp_path, "--queue", queue)

    assert completed.returncode == 0, completed.stderr
    assert "planned 1 jobs" in completed.stdout


def test_endpoint_pinned_queue_rejects_decimal_128k_for_glm_5_2(tmp_path: Path) -> None:
    queue = tmp_path / "input.tsv"
    queue.write_text(
        "job-1\tz-ai/glm-5.2\tmax\tpass1\tqiskit\tall\t0\tz-ai/fp8\t128000\t"
        "author_native\tcatalog_numeric\tmax_tokens\troute-01\texplicit_zero\t70\n",
        encoding="utf-8",
    )

    completed = _run_plan(tmp_path, "--queue", queue)

    assert completed.returncode == 2
    assert "GLM-5.2 queue row violates the frozen max_tokens=131072 contract" in completed.stderr


def test_endpoint_pinned_queue_rejects_grok_exception_outside_frozen_scope(tmp_path: Path) -> None:
    queue = tmp_path / "input.tsv"
    queue.write_text(
        "job-1\texample/model\thigh\tpass1\tqiskit\tall\t0\txai\t128000\t"
        "benchmark_floor\tundisclosed_first_party_exception\tmax_tokens\troute-01\tnot_exposed\t70\n",
        encoding="utf-8",
    )

    completed = _run_plan(tmp_path, "--queue", queue)

    assert completed.returncode == 2
    assert "Grok endpoint cap exception is outside its frozen scope" in completed.stderr


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (8, "", "invalid output ceiling"),
        (9, "inferred", "invalid output limit source"),
        (10, "unknown", "invalid endpoint cap status"),
        (11, "unknown", "invalid output parameter"),
        (13, "default", "invalid temperature behavior"),
    ],
)
def test_endpoint_pinned_queue_rejects_invalid_route_fields(
    tmp_path: Path, column: int, value: str, message: str
) -> None:
    fields = [
        "job-1",
        "example/model",
        "high",
        "pass1",
        "qiskit",
        "all",
        "0",
        "author/region",
        "128000",
        "benchmark_floor",
        "catalog_numeric",
        "max_tokens",
        "route-01",
        "not_exposed",
        "70",
    ]
    fields[column] = value
    queue = tmp_path / "input.tsv"
    queue.write_text("\t".join(fields) + "\n", encoding="utf-8")

    completed = _run_plan(tmp_path, "--queue", queue)

    assert completed.returncode == 2
    assert message in completed.stderr


@pytest.mark.parametrize("setting", [*EFFORTS, "enabled"])
def test_each_accepted_reasoning_setting_emits_one_explicit_flag(setting: str) -> None:
    flags = _render_reasoning_args(setting)

    if setting == "enabled":
        assert flags == ["--reasoning-enabled"]
    else:
        assert flags == ["--reasoning-effort", setting]


JOB_ID = "example-model__max__pass1__qiskit"
MODEL = "example/model"
FRAMEWORK = "qiskit"

WORKER_HARNESS = """
OUT_DIR="$1"
job_file="$2"
STATE="$OUT_DIR/state"
PROVIDER_TIMEOUT=600
MAX_RETRIES=3
RETRY_BASE_DELAY=1
RETRY_MAX_DELAY=60
GENERATION_CONCURRENCY=8
EVALUATION_WORKERS=2
TASK_TIMEOUT=2700
EVAL_TIMEOUT=180
mkdir -p "$STATE/done" "$STATE/failed" "$OUT_DIR/shards"
ssh() { return "${FAKE_SSH_EXIT:-0}"; }
scp() {
  (( ${FAKE_SCP_EXIT:-0} == 0 )) || return "${FAKE_SCP_EXIT}"
  cp "$FAKE_SHARD" "${!#}"
}
SSH=(ssh)
SCP=(scp)
base="$(basename "$job_file")"
if run_remote_job 192.0.2.1 "$job_file"; then
  mv "$job_file" "$STATE/done/${base#*__}"
else
  mv "$job_file" "$STATE/failed/${base#*__}"
fi
"""


def _shard_lines(model: str = MODEL, framework: str = FRAMEWORK, results: int = 2) -> list[str]:
    lines = [json.dumps({"kind": "result", "model": model, "framework": framework}) for _ in range(results)]
    summary = {"kind": "summary", "model": model, "summary": {"total_tasks": results, "by_framework": {framework: {}}}}
    return [*lines, json.dumps(summary)]


def _run_worker(tmp_path: Path, shard_lines: list[str] | str, *, ssh_exit: int = 0, scp_exit: int = 0) -> Path:
    out_dir = tmp_path / "output"
    running = out_dir / "state" / "running"
    running.mkdir(parents=True)
    job_file = running / f"worker-0__{JOB_ID}.job"
    job_file.write_text(f"{JOB_ID}\t{MODEL}\tmax\tpass1\t{FRAMEWORK}\tcore\t0\n", encoding="utf-8")
    shard = tmp_path / "remote-shard.jsonl"
    content = shard_lines if isinstance(shard_lines, str) else "".join(f"{line}\n" for line in shard_lines)
    shard.write_text(content, encoding="utf-8")
    helpers = RUNNER.read_text(encoding="utf-8").split('REPO_ROOT="', 1)[0]
    completed = subprocess.run(
        ["bash", "-s", str(out_dir), str(job_file)],
        cwd=REPO_ROOT,
        input=helpers + WORKER_HARNESS,
        env={
            "PATH": "/usr/bin:/bin",
            "FAKE_SSH_EXIT": str(ssh_exit),
            "FAKE_SCP_EXIT": str(scp_exit),
            "FAKE_SHARD": str(shard),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return out_dir


def _job_state(out_dir: Path) -> str:
    done = out_dir / "state" / "done" / f"{JOB_ID}.job"
    failed = out_dir / "state" / "failed" / f"{JOB_ID}.job"
    assert done.exists() != failed.exists()
    return "done" if done.exists() else "failed"


def test_complete_shard_is_validated_published_and_marked_done(tmp_path: Path) -> None:
    out_dir = _run_worker(tmp_path, _shard_lines())

    assert _job_state(out_dir) == "done"
    published = out_dir / "shards" / f"{JOB_ID}.jsonl"
    assert published.read_text(encoding="utf-8").splitlines() == _shard_lines()
    assert not (out_dir / "shards" / ".staging" / f"{JOB_ID}.jsonl").exists()


def test_remote_command_failure_marks_job_failed(tmp_path: Path) -> None:
    out_dir = _run_worker(tmp_path, _shard_lines(), ssh_exit=1)

    assert _job_state(out_dir) == "failed"
    assert not (out_dir / "shards" / f"{JOB_ID}.jsonl").exists()


def test_failed_scp_marks_job_failed(tmp_path: Path) -> None:
    out_dir = _run_worker(tmp_path, _shard_lines(), scp_exit=1)

    assert _job_state(out_dir) == "failed"
    assert not (out_dir / "shards" / f"{JOB_ID}.jsonl").exists()


def test_shard_without_summary_is_not_published(tmp_path: Path) -> None:
    out_dir = _run_worker(tmp_path, _shard_lines()[:-1])

    assert _job_state(out_dir) == "failed"
    assert not (out_dir / "shards" / f"{JOB_ID}.jsonl").exists()


def test_malformed_jsonl_is_not_published(tmp_path: Path) -> None:
    out_dir = _run_worker(tmp_path, "".join(f"{line}\n" for line in _shard_lines()[:-1]) + "{not json\n")

    assert _job_state(out_dir) == "failed"
    assert not (out_dir / "shards" / f"{JOB_ID}.jsonl").exists()


def test_summary_with_foreign_model_is_not_published(tmp_path: Path) -> None:
    lines = _shard_lines()
    summary = json.loads(lines[-1])
    summary["model"] = "other/model"
    lines[-1] = json.dumps(summary)

    out_dir = _run_worker(tmp_path, lines)

    assert _job_state(out_dir) == "failed"
    assert not (out_dir / "shards" / f"{JOB_ID}.jsonl").exists()


def test_summary_count_mismatch_is_not_published(tmp_path: Path) -> None:
    lines = _shard_lines()
    summary = json.loads(lines[-1])
    summary["summary"]["total_tasks"] = 5
    lines[-1] = json.dumps(summary)

    out_dir = _run_worker(tmp_path, lines)

    assert _job_state(out_dir) == "failed"
    assert not (out_dir / "shards" / f"{JOB_ID}.jsonl").exists()


def test_flock_queue_claims_are_unique_under_concurrency(tmp_path: Path) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    queue_helpers = source[source.index("lock_queue() {") : source.index("worker_loop() {")]
    state = tmp_path / "state"
    (state / "pending").mkdir(parents=True)
    (state / "running").mkdir()
    for index in range(80):
        (state / "pending" / f"job-{index:03d}.job").write_text(f"job-{index:03d}\n", encoding="utf-8")
    harness = f"""
set -Eeuo pipefail
STATE={state!s}
die() {{ printf '%s\n' "$*" >&2; exit 2; }}
{queue_helpers}
claim_all() {{
  local worker=$1 claimed
  exec {{QUEUE_LOCK_FD}}>"$STATE/queue.lock"
  while claimed="$(claim_job "$worker")"; do
    basename "$claimed" >>"$STATE/$worker.claims"
  done
}}
for worker in $(seq 0 7); do claim_all "worker-$worker" & done
wait
cat "$STATE"/*.claims | sort >"$STATE/all.claims"
"""

    completed = subprocess.run(
        ["bash"],
        cwd=REPO_ROOT,
        input=harness,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    claims = (state / "all.claims").read_text(encoding="utf-8").splitlines()
    assert len(claims) == 80
    assert len(set(claims)) == 80
    assert not list((state / "pending").glob("*.job"))
    assert len(list((state / "running").glob("*.job"))) == 80
