#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run framework-sharded QCircuitEval jobs on a temporary EC2 worker pool.

Usage:
  production/run_aws.sh --instances N --models FILE --out-dir DIR [options]
  production/run_aws.sh --instances N --queue FILE  --out-dir DIR [options]

Required for a live run:
  --ami-id ID                 Ubuntu x86_64 AMI ID
  --subnet-id ID              subnet that assigns reachable public IPs
  --security-group-id ID      security group allowing SSH from this controller
  --key-name NAME             EC2 key-pair name
  --ssh-key PATH              matching local private key

Inputs:
  --models FILE               lines are: OPENROUTER_MODEL  SETTING
                              SETTING is an effort or enabled
  --queue FILE                prebuilt legacy seven-column or pinned fourteen/fifteen-column TSV queue
  --env-file FILE             dotenv file containing OPENROUTER_API_KEY (default: .env)
  --max-tasks N               per-framework prompt limit; omit for the full suites
  --suite {core,qec,all}       default: core
  --plan-only                 generate and validate queue.tsv without AWS mutations

Worker/run controls:
  --instance-type TYPE        default: c7i.2xlarge
  --root-gb N                 default: 100
  --generation-concurrency N  default: 8
  --evaluation-workers N      default: 2
  --provider-timeout SECONDS  timeout for one provider request (default: 600)
  --max-retries N             provider retries after the first request (default: 3)
  --retry-base-delay SECONDS  exponential-backoff base delay (default: 1)
  --retry-max-delay SECONDS   exponential-backoff delay cap (default: 60)
  --task-timeout SECONDS      hard limit for one logical task (default: 2700)
  --eval-timeout SECONDS      hard limit for one grader invocation (default: 180)
  --keep-instances            do not terminate the worker pool on exit
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

reasoning_setting_is_valid() {
  case "${1:-}" in
    max|xhigh|high|medium|low|minimal|none|enabled) return 0 ;;
    *) return 1 ;;
  esac
}

append_reasoning_args() {
  local setting=$1
  local destination=$2
  reasoning_setting_is_valid "$setting" || die "invalid reasoning setting: $setting"
  [[ "$destination" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || die "invalid reasoning argument destination: $destination"
  case "$setting" in
    enabled) eval "$destination+=(--reasoning-enabled)" ;;
    *) eval "$destination+=(--reasoning-effort \"$setting\")" ;;
  esac
}

validate_local_shard() {
  local shard=$1 job_id=$2 model=$3 framework=$4
  python3 - "$shard" "$job_id" "$model" "$framework" <<'PY'
import json
import sys

path, job_id, model, framework = sys.argv[1:]


def fail(message):
    raise SystemExit(f"shard validation failed for {job_id}: {message}")


records = []
try:
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                fail(f"line {number} is blank")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                fail(f"line {number} is not valid JSON: {error}")
except OSError as error:
    fail(str(error))
if not records:
    fail("shard contains no records")
summaries = [record for record in records if record.get("kind") == "summary"]
if len(summaries) != 1:
    fail(f"expected exactly one summary record, found {len(summaries)}")
if records[-1].get("kind") != "summary":
    fail("the summary record is not the final line")
results = [record for record in records if record.get("kind") == "result"]
if len(results) != len(records) - 1:
    fail("shard contains records that are neither results nor the summary")
summary = summaries[0]
if summary.get("model") != model:
    fail(f"summary model {summary.get('model')!r} does not match job model {model!r}")
foreign_models = sorted({str(record.get("model")) for record in results if record.get("model") != model})
if foreign_models:
    fail(f"result records report foreign models: {', '.join(foreign_models)}")
result_frameworks = sorted({str(record.get("framework")) for record in results})
if result_frameworks != [framework]:
    fail(f"result records report frameworks {result_frameworks}, expected ['{framework}']")
counts = summary.get("summary") or {}
if counts.get("total_tasks") != len(results):
    fail(f"summary total_tasks {counts.get('total_tasks')!r} does not match {len(results)} result records")
summary_frameworks = sorted(counts.get("by_framework") or {})
if summary_frameworks != [framework]:
    fail(f"summary covers frameworks {summary_frameworks}, expected ['{framework}']")
PY
}

run_remote_job() {
  local ip=$1 job_file=$2
  local job_id model setting protocol framework suite max_tasks
  local endpoint_tag max_output output_limit_source endpoint_cap_status output_token_parameter route_revision temperature_behavior assigned_tasks
  local -a job_fields
  IFS=$'\t' read -r -a job_fields <"$job_file" || return 1
  job_id=${job_fields[0]} model=${job_fields[1]} setting=${job_fields[2]} protocol=${job_fields[3]}
  framework=${job_fields[4]} suite=${job_fields[5]} max_tasks=${job_fields[6]}
  if (( ${#job_fields[@]} == 15 )); then
    endpoint_tag=${job_fields[7]} max_output=${job_fields[8]} output_limit_source=${job_fields[9]}
    endpoint_cap_status=${job_fields[10]} output_token_parameter=${job_fields[11]}
    route_revision=${job_fields[12]} temperature_behavior=${job_fields[13]} assigned_tasks=${job_fields[14]}
  elif (( ${#job_fields[@]} == 14 )); then
    endpoint_tag=${job_fields[7]} max_output=${job_fields[8]} output_limit_source=${job_fields[9]}
    endpoint_cap_status=catalog_numeric output_token_parameter=${job_fields[10]}
    route_revision=${job_fields[11]} temperature_behavior=${job_fields[12]} assigned_tasks=${job_fields[13]}
  fi
  local temperature samples pass_k attempts
  case "$protocol" in
    pass1) temperature=0.0; samples=1; pass_k=1; attempts=1 ;;
    pass5) temperature=0.8; samples=5; pass_k=5; attempts=1 ;;
    feedback5) temperature=0.2; samples=1; pass_k=1; attempts=5 ;;
    *) return 1 ;;
  esac
  local remote_out="/home/ubuntu/qceval-results/shards/$job_id.jsonl"
  local args=(
    /home/ubuntu/.local/bin/uv run qceval run
    --provider openrouter
    --openrouter-api-key-file /home/ubuntu/.openrouter-key
    --model "$model"
    --framework "$framework"
    --suite "$suite"
    --source-hint "$(git rev-parse HEAD)"
    --out "$remote_out"
    --output-format jsonl
    --timeout "$PROVIDER_TIMEOUT"
    --max-retries "$MAX_RETRIES"
    --retry-base-delay "$RETRY_BASE_DELAY"
    --retry-max-delay "$RETRY_MAX_DELAY"
    --generation-concurrency "$GENERATION_CONCURRENCY"
    --evaluation-workers "$EVALUATION_WORKERS"
    --samples-per-task "$samples"
    --pass-k "$pass_k"
    --max-attempts "$attempts"
    --cache-dir /home/ubuntu/qceval-results/cache
    --task-timeout "$TASK_TIMEOUT"
    --eval-timeout "$EVAL_TIMEOUT"
    --progress
  )
  if [[ -n "${endpoint_tag:-}" ]]; then
    args+=(
      --openrouter-endpoint-tag "$endpoint_tag"
      --openrouter-max-output-tokens "$max_output"
      --openrouter-output-limit-source "$output_limit_source"
      --openrouter-endpoint-cap-status "$endpoint_cap_status"
      --openrouter-output-token-parameter "$output_token_parameter"
      --openrouter-route-revision "$route_revision"
      --stop-on-infrastructure-error
    )
    case "$temperature_behavior" in
      explicit_zero) args+=(--temperature 0.0) ;;
      not_exposed) ;;
      *) return 1 ;;
    esac
  else
    args+=(--temperature "$temperature")
  fi
  append_reasoning_args "$setting" args
  ((max_tasks == 0)) || args+=(--max-tasks "$max_tasks")
  local quoted
  printf -v quoted '%q ' "${args[@]}"
  local staged="$OUT_DIR/shards/.staging/$job_id.jsonl"
  mkdir -p "$OUT_DIR/shards/.staging" || return 1
  rm -f "$staged" || return 1
  "${SSH[@]}" "ubuntu@$ip" "cd /home/ubuntu/qceval && $quoted" || return 1
  "${SCP[@]}" "ubuntu@$ip:$remote_out" "$staged" || return 1
  validate_local_shard "$staged" "$job_id" "$model" "$framework" || return 1
  mv "$staged" "$OUT_DIR/shards/$job_id.jsonl" || return 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

slug() {
  printf '%s' "$1" | tr '/: .' '----' | tr -cd '[:alnum:]_-'
}

INSTANCES=""
MODELS_FILE=""
INPUT_QUEUE=""
OUT_DIR=""
ENV_FILE=".env"
AMI_ID=""
SUBNET_ID=""
SECURITY_GROUP_ID=""
KEY_NAME=""
SSH_KEY=""
INSTANCE_TYPE="c7i.2xlarge"
ROOT_GB=100
MAX_TASKS=0
SUITE=core
GENERATION_CONCURRENCY=8
EVALUATION_WORKERS=2
PROVIDER_TIMEOUT=600
MAX_RETRIES=3
RETRY_BASE_DELAY=1
RETRY_MAX_DELAY=60
TASK_TIMEOUT=2700
EVAL_TIMEOUT=180
PLAN_ONLY=0
KEEP_INSTANCES=0
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

while (($#)); do
  case "$1" in
    --instances) INSTANCES="$2"; shift 2 ;;
    --models) MODELS_FILE="$2"; shift 2 ;;
    --queue) INPUT_QUEUE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --ami-id) AMI_ID="$2"; shift 2 ;;
    --subnet-id) SUBNET_ID="$2"; shift 2 ;;
    --security-group-id) SECURITY_GROUP_ID="$2"; shift 2 ;;
    --key-name) KEY_NAME="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --root-gb) ROOT_GB="$2"; shift 2 ;;
    --max-tasks) MAX_TASKS="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --generation-concurrency) GENERATION_CONCURRENCY="$2"; shift 2 ;;
    --evaluation-workers) EVALUATION_WORKERS="$2"; shift 2 ;;
    --provider-timeout) PROVIDER_TIMEOUT="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --retry-base-delay) RETRY_BASE_DELAY="$2"; shift 2 ;;
    --retry-max-delay) RETRY_MAX_DELAY="$2"; shift 2 ;;
    --task-timeout) TASK_TIMEOUT="$2"; shift 2 ;;
    --eval-timeout) EVAL_TIMEOUT="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    --keep-instances) KEEP_INSTANCES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$OUT_DIR" ]] || die "--out-dir is required"
[[ -n "$MODELS_FILE" || -n "$INPUT_QUEUE" ]] || die "provide --models or --queue"
[[ -z "$MODELS_FILE" || -z "$INPUT_QUEUE" ]] || die "--models and --queue are mutually exclusive"
[[ "$MAX_TASKS" =~ ^[0-9]+$ ]] || die "--max-tasks must be a non-negative integer"
[[ "$SUITE" =~ ^(core|qec|all)$ ]] || die "--suite must be core, qec, or all"
for positive in ROOT_GB GENERATION_CONCURRENCY EVALUATION_WORKERS PROVIDER_TIMEOUT RETRY_MAX_DELAY TASK_TIMEOUT EVAL_TIMEOUT; do
  value=${!positive}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$positive must be a positive integer"
done
[[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || die "--max-retries must be a non-negative integer"
[[ "$RETRY_BASE_DELAY" =~ ^[0-9]+$ ]] || die "--retry-base-delay must be a non-negative integer"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
QUEUE="$OUT_DIR/queue.tsv"

if [[ -n "$MODELS_FILE" ]]; then
  [[ -f "$MODELS_FILE" ]] || die "models file not found: $MODELS_FILE"
  : >"$QUEUE"
  while read -r model setting extra; do
    [[ -z "${model:-}" || "$model" == \#* ]] && continue
    [[ -z "${extra:-}" ]] || die "model lines must have exactly two fields: $model $setting $extra"
    reasoning_setting_is_valid "$setting" || die "invalid reasoning setting for $model: $setting"
    model_slug="$(slug "$model")"
    for protocol in pass1 pass5 feedback5; do
      for framework in qiskit cirq pennylane cudaq; do
        job_id="${model_slug}__${setting}__${protocol}__${framework}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$job_id" "$model" "$setting" "$protocol" "$framework" "$SUITE" "$MAX_TASKS" >>"$QUEUE"
      done
    done
  done <"$MODELS_FILE"
else
  [[ -f "$INPUT_QUEUE" ]] || die "queue file not found: $INPUT_QUEUE"
  cp "$INPUT_QUEUE" "$QUEUE"
fi

awk -F '\t' '
  NF != 7 && NF != 14 && NF != 15 { print "queue line " NR " does not have seven, fourteen, or fifteen columns" > "/dev/stderr"; exit 1 }
  $4 !~ /^(pass1|pass5|feedback5)$/ { print "invalid protocol on line " NR > "/dev/stderr"; exit 1 }
  $5 !~ /^(qiskit|cirq|pennylane|cudaq)$/ { print "invalid framework on line " NR > "/dev/stderr"; exit 1 }
  $6 !~ /^(core|qec|all)$/ { print "invalid suite on line " NR > "/dev/stderr"; exit 1 }
  $7 !~ /^[0-9]+$/ { print "invalid max_tasks on line " NR > "/dev/stderr"; exit 1 }
  NF >= 14 && $8 == "" { print "missing endpoint tag on line " NR > "/dev/stderr"; exit 1 }
  NF >= 14 && $9 !~ /^[1-9][0-9]*$/ { print "invalid output ceiling on line " NR > "/dev/stderr"; exit 1 }
  NF >= 14 && $10 !~ /^(author_native|benchmark_floor)$/ { print "invalid output limit source on line " NR > "/dev/stderr"; exit 1 }
  NF == 15 && $11 !~ /^(catalog_numeric|undisclosed_first_party_exception)$/ { print "invalid endpoint cap status on line " NR > "/dev/stderr"; exit 1 }
  NF == 15 && $11 == "undisclosed_first_party_exception" && ($2 != "x-ai/grok-4.6" || $8 != "xai" || $9 != "128000" || $10 != "benchmark_floor") { print "Grok endpoint cap exception is outside its frozen scope on line " NR > "/dev/stderr"; exit 1 }
  NF == 15 && $2 == "z-ai/glm-5.2" && ($3 != "max" || $9 != "131072" || $10 != "author_native" || $11 != "catalog_numeric" || $12 != "max_tokens") { print "GLM-5.2 queue row violates the frozen max_tokens=131072 contract on line " NR > "/dev/stderr"; exit 1 }
  NF == 14 && $11 !~ /^(max_tokens|max_completion_tokens)$/ { print "invalid output parameter on line " NR > "/dev/stderr"; exit 1 }
  NF == 15 && $12 !~ /^(max_tokens|max_completion_tokens)$/ { print "invalid output parameter on line " NR > "/dev/stderr"; exit 1 }
  NF == 14 && $12 == "" { print "missing route revision on line " NR > "/dev/stderr"; exit 1 }
  NF == 15 && $13 == "" { print "missing route revision on line " NR > "/dev/stderr"; exit 1 }
  NF == 14 && $13 !~ /^(explicit_zero|not_exposed)$/ { print "invalid temperature behavior on line " NR > "/dev/stderr"; exit 1 }
  NF == 15 && $14 !~ /^(explicit_zero|not_exposed)$/ { print "invalid temperature behavior on line " NR > "/dev/stderr"; exit 1 }
  NF == 14 && $14 !~ /^[1-9][0-9]*$/ { print "invalid assigned task count on line " NR > "/dev/stderr"; exit 1 }
  NF == 15 && $15 !~ /^[1-9][0-9]*$/ { print "invalid assigned task count on line " NR > "/dev/stderr"; exit 1 }
' "$QUEUE" || die "invalid queue"

queue_line=0
while IFS=$'\t' read -r _ _ queue_setting _; do
  ((queue_line += 1))
  reasoning_setting_is_valid "$queue_setting" || \
    die "invalid reasoning setting on queue line $queue_line: $queue_setting"
done <"$QUEUE"

JOB_COUNT="$(wc -l <"$QUEUE" | tr -d ' ')"
((JOB_COUNT > 0)) || die "queue is empty"
if ((PLAN_ONLY)); then
  echo "planned $JOB_COUNT jobs in $QUEUE"
  exit 0
fi

[[ "$INSTANCES" =~ ^[1-9][0-9]*$ ]] || die "--instances must be a positive integer"
for value in AMI_ID SUBNET_ID SECURITY_GROUP_ID KEY_NAME SSH_KEY; do
  [[ -n "${!value}" ]] || die "missing required AWS/SSH option for $value"
done
[[ -f "$SSH_KEY" ]] || die "SSH key not found: $SSH_KEY"
[[ -f "$ENV_FILE" ]] || die "dotenv file not found: $ENV_FILE"
for command in aws flock git python3 rsync scp ssh uv; do
  command -v "$command" >/dev/null || die "required command not found: $command"
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "production runs require clean tracked files"

RUN_ID="qceval-$(date -u +%Y%m%dT%H%M%SZ)-$$"
TMP_DIR="$(mktemp -d)"
KEY_FILE="$TMP_DIR/openrouter.key"
BUNDLE="$TMP_DIR/qceval.bundle"
INSTANCE_IDS=()
SUCCESS=0

python3 - "$ENV_FILE" "$KEY_FILE" <<'PY'
import ast
import os
import sys

source, destination = sys.argv[1:]
value = None
for raw in open(source, encoding="utf-8"):
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    name, candidate = line.split("=", 1)
    if name.strip() != "OPENROUTER_API_KEY":
        continue
    candidate = candidate.strip()
    if candidate[:1] in {'"', "'"}:
        candidate = ast.literal_eval(candidate)
    value = candidate
    break
if not value:
    raise SystemExit("OPENROUTER_API_KEY is missing or empty")
fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write(value)
PY

terminate_instances() {
  if ((${#INSTANCE_IDS[@]})) && ((KEEP_INSTANCES == 0)); then
    aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}" >/dev/null || true
    echo "termination requested for: ${INSTANCE_IDS[*]}" >&2
  elif ((${#INSTANCE_IDS[@]})); then
    echo "instances retained: ${INSTANCE_IDS[*]}" >&2
  fi
}

cleanup() {
  status=$?
  terminate_instances
  rm -rf "$TMP_DIR"
  if ((status != 0 || SUCCESS == 0)); then
    echo "run did not complete successfully; collected artifacts remain in $OUT_DIR" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

git bundle create "$BUNDLE" HEAD
while IFS= read -r instance_id; do
  INSTANCE_IDS+=("$instance_id")
done < <(
  aws ec2 run-instances \
    --region "$AWS_REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --count "$INSTANCES" \
    --key-name "$KEY_NAME" \
    --subnet-id "$SUBNET_ID" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_ID},{Key=Project,Value=QCircuitEval},{Key=RunId,Value=$RUN_ID},{Key=ManagedBy,Value=production-runner}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=$RUN_ID-root},{Key=Project,Value=QCircuitEval},{Key=RunId,Value=$RUN_ID}]" \
    --query 'Instances[].InstanceId' \
    --output text | tr '\t' '\n'
)
((${#INSTANCE_IDS[@]} == INSTANCES)) || die "AWS returned ${#INSTANCE_IDS[@]} instances, expected $INSTANCES"
printf '%s\n' "${INSTANCE_IDS[@]}" >"$OUT_DIR/instance_ids.txt"
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}"

IPS=()
while IFS= read -r ip; do
  IPS+=("$ip")
done < <(
  aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --query 'Reservations[].Instances[].PublicIpAddress' \
    --output text | tr '\t' '\n'
)
((${#IPS[@]} == INSTANCES)) || die "not every instance has a public IP"

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i "$SSH_KEY")
SCP=(scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i "$SSH_KEY")

provision() {
  local ip=$1
  local host="ubuntu@$ip"
  for _ in {1..30}; do
    "${SSH[@]}" "$host" true >/dev/null 2>&1 && break
    sleep 5
  done
  "${SCP[@]}" "$BUNDLE" "$host:/home/ubuntu/qceval.bundle"
  "${SCP[@]}" "$KEY_FILE" "$host:/home/ubuntu/.openrouter-key"
  "${SSH[@]}" "$host" bash -s <<'REMOTE'
set -Eeuo pipefail
cloud-init status --wait >/dev/null
sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl build-essential git-lfs jq
git lfs install --skip-smudge >/dev/null
rm -rf /home/ubuntu/qceval
GIT_LFS_SKIP_SMUDGE=1 git clone -q /home/ubuntu/qceval.bundle /home/ubuntu/qceval
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
cd /home/ubuntu/qceval
/home/ubuntu/.local/bin/uv sync --frozen -q
chmod 600 /home/ubuntu/.openrouter-key
mkdir -p /home/ubuntu/qceval-results/shards /home/ubuntu/qceval-results/cache
REMOTE
}

pids=()
for ip in "${IPS[@]}"; do
  provision "$ip" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

STATE="$OUT_DIR/state"
mkdir -p "$STATE/pending" "$STATE/running" "$STATE/done" "$STATE/failed" "$OUT_DIR/shards" "$OUT_DIR/merged"
while IFS= read -r line; do
  job_id="${line%%$'\t'*}"
  printf '%s\n' "$line" >"$STATE/pending/$job_id.job"
done <"$QUEUE"

lock_queue() {
  [[ -n "${QUEUE_LOCK_FD:-}" ]] || die "queue lock file descriptor is unavailable"
  flock -x "$QUEUE_LOCK_FD"
}

unlock_queue() {
  flock -u "$QUEUE_LOCK_FD"
}

claim_job() {
  local worker=$1
  local files=() file claimed
  lock_queue
  files=("$STATE/pending"/*.job)
  if [[ ! -e "${files[0]}" ]]; then
    unlock_queue
    return 1
  fi
  file="${files[0]}"
  claimed="$STATE/running/${worker}__$(basename "$file")"
  mv "$file" "$claimed"
  unlock_queue
  printf '%s' "$claimed"
}

worker_loop() {
  local ip=$1 worker=$2 job_file base
  # Open independently inside each background worker. Linux flock locks are
  # associated with open file descriptions, so opening here avoids every
  # worker inheriting one shared description from the controller.
  exec {QUEUE_LOCK_FD}>"$STATE/queue.lock"
  while job_file="$(claim_job "$worker")"; do
    base="$(basename "$job_file")"
    if run_remote_job "$ip" "$job_file"; then
      mv "$job_file" "$STATE/done/${base#*__}"
    else
      mv "$job_file" "$STATE/failed/${base#*__}"
    fi
  done
}

worker_pids=()
for index in "${!IPS[@]}"; do
  worker_loop "${IPS[$index]}" "worker-$index" &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

failed=("$STATE/failed"/*.job)
if [[ -e "${failed[0]}" ]]; then
  die "one or more jobs failed; inspect $STATE/failed and retained shard outputs"
fi

while IFS=$'\t' read -r model setting protocol; do
  inputs=()
  while IFS=$'\t' read -r job_id queue_model queue_setting queue_protocol _; do
    if [[ "$queue_model" == "$model" && "$queue_setting" == "$setting" && "$queue_protocol" == "$protocol" ]]; then
      inputs+=("$OUT_DIR/shards/$job_id.jsonl")
    fi
  done <"$QUEUE"
  merged_name="$(slug "$model")__${setting}__${protocol}.jsonl"
  uv run python scripts/merge_run_records.py --out "$OUT_DIR/merged/$merged_name" "${inputs[@]}"
done < <(cut -f2-4 "$QUEUE" | sort -u)

SUCCESS=1
echo "completed $JOB_COUNT jobs; merged outputs are in $OUT_DIR/merged"
