#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run credential-free, network-disabled Pass@1 regrading on an EC2 worker pool.

Usage:
  production/run_offline_aws.sh --instances 6 --queue FILE --candidates-dir DIR \
    --out-dir DIR --ami-id ID --subnet-id ID \
    --provisioning-security-group-id ID --evaluation-security-group-id ID \
    --key-name NAME --ssh-key PATH [options]

Options:
  --instance-type TYPE        default: c7i.2xlarge
  --root-gb N                 default: 100
  --eval-timeout SECONDS      default: 180
  --cudaq-eval-timeout SEC    default: 900
  --cudaq-evaluation-workers N
                              default: 1
  --campaign-manifest FILE    required for a canonical non-default campaign queue
  --scope-manifest FILE       required when grading a derived subset queue
  --plan-only                 validate local inputs without AWS mutations
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

slug() {
  printf '%s' "$1" | tr '/: .' '----' | tr -cd '[:alnum:]_-'
}

sha256_file() {
  sha256sum "$1" | cut -d' ' -f1
}

validate_shard() {
  local path=$1 model=$2 configuration=$3 framework=$4 expected=${5:-70}
  python3 - "$path" "$model" "$configuration" "$framework" "$expected" <<'PY'
import json
import sys

path, model, configuration, framework, expected = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
payloads = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
results = [payload for payload in payloads if payload.get("kind") == "result"]
summaries = [payload for payload in payloads if payload.get("kind") == "summary"]
if len(results) != expected or len(summaries) != 1 or payloads[-1].get("kind") != "summary":
    raise SystemExit(f"invalid shard shape: results={len(results)} summaries={len(summaries)}")
if {payload.get("model") for payload in results} != {model}:
    raise SystemExit("shard model mismatch")
if {payload.get("framework") for payload in results} != {framework}:
    raise SystemExit("shard framework mismatch")
if any(payload.get("status") in {"generated", "infrastructure_error"} for payload in results):
    raise SystemExit("offline shard contains an ungraded or infrastructure result")
if any(
    (payload.get("evaluation") or {}).get("error_type") == "EvaluationTimeout"
    for payload in results
):
    raise SystemExit("offline shard contains a grader evaluation timeout")
for payload in results:
    response = payload.get("provider_response") or {}
    route = (response.get("metadata") or {}).get("route") or {}
    usage = response.get("usage") or {}
    if route.get("route_verified") is not True or usage.get("cost_usd") is None:
        raise SystemExit("offline shard lost route or cost provenance")
    if route.get("configuration_id") != configuration:
        raise SystemExit("offline shard has incompatible configuration provenance")
summary = summaries[0].get("summary") or {}
if summary.get("total_tasks") != expected:
    raise SystemExit("offline shard summary count mismatch")
PY
}

verify_security_group_snapshot() {
  local snapshot=$1 expected_cidr=$2 require_no_egress=$3
  python3 - "$snapshot" "$expected_cidr" "$require_no_egress" <<'PY'
import json
import sys

path, expected, no_egress = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
groups = json.load(open(path, encoding="utf-8")).get("SecurityGroups") or []
if len(groups) != 1:
    raise SystemExit("security-group readback did not contain exactly one group")
group = groups[0]
ssh_cidrs = []
ssh_ipv6 = []
for rule in group.get("IpPermissions") or []:
    if rule.get("IpProtocol") == "tcp" and rule.get("FromPort") == 22 and rule.get("ToPort") == 22:
        ssh_cidrs.extend(item.get("CidrIp") for item in rule.get("IpRanges") or [])
        ssh_ipv6.extend(item.get("CidrIpv6") for item in rule.get("Ipv6Ranges") or [])
if ssh_cidrs != [expected]:
    raise SystemExit(f"SSH ingress is {ssh_cidrs!r}, expected only {expected!r}")
if ssh_ipv6:
    raise SystemExit(f"unexpected IPv6 SSH ingress is present: {ssh_ipv6!r}")
if no_egress and group.get("IpPermissionsEgress"):
    raise SystemExit("evaluation security group has outbound rules")
PY
}

INSTANCES=""
QUEUE=""
CANDIDATES_DIR=""
OUT_DIR=""
AMI_ID=""
SUBNET_ID=""
PROVISIONING_SECURITY_GROUP_ID=""
EVALUATION_SECURITY_GROUP_ID=""
KEY_NAME=""
SSH_KEY=""
INSTANCE_TYPE="c7i.2xlarge"
ROOT_GB=100
EVAL_TIMEOUT=180
CUDAQ_EVAL_TIMEOUT=900
CUDAQ_EVALUATION_WORKERS=1
CAMPAIGN_MANIFEST=""
SCOPE_MANIFEST=""
PLAN_ONLY=0
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-1}}"

while (($#)); do
  case "$1" in
    --instances) INSTANCES=$2; shift 2 ;;
    --queue) QUEUE=$2; shift 2 ;;
    --candidates-dir) CANDIDATES_DIR=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    --ami-id) AMI_ID=$2; shift 2 ;;
    --subnet-id) SUBNET_ID=$2; shift 2 ;;
    --provisioning-security-group-id) PROVISIONING_SECURITY_GROUP_ID=$2; shift 2 ;;
    --evaluation-security-group-id) EVALUATION_SECURITY_GROUP_ID=$2; shift 2 ;;
    --key-name) KEY_NAME=$2; shift 2 ;;
    --ssh-key) SSH_KEY=$2; shift 2 ;;
    --instance-type) INSTANCE_TYPE=$2; shift 2 ;;
    --root-gb) ROOT_GB=$2; shift 2 ;;
    --eval-timeout) EVAL_TIMEOUT=$2; shift 2 ;;
    --cudaq-eval-timeout) CUDAQ_EVAL_TIMEOUT=$2; shift 2 ;;
    --cudaq-evaluation-workers) CUDAQ_EVALUATION_WORKERS=$2; shift 2 ;;
    --campaign-manifest) CAMPAIGN_MANIFEST=$2; shift 2 ;;
    --scope-manifest) SCOPE_MANIFEST=$2; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -n "$QUEUE" && -f "$QUEUE" ]] || die "--queue must identify the frozen Pass@1 queue"
[[ -n "$CANDIDATES_DIR" && -f "$CANDIDATES_DIR/manifest.json" ]] || die "candidate manifest is missing"
[[ -n "$OUT_DIR" ]] || die "--out-dir is required"
QUEUE_SHARDS="$(wc -l <"$QUEUE" | tr -d ' ')"
[[ "$QUEUE_SHARDS" =~ ^[1-9][0-9]*$ ]] && ((QUEUE_SHARDS % 4 == 0)) || \
  die "offline queue must contain a positive multiple of four framework shards"
awk -F '\t' '
  NF != 16 || $4 != "pass1" || $6 != "all" || $15 != 70 || $16 == "" { exit 1 }
  { ids[$1]++; configurations[$16]++; scopes[$16 SUBSEP $5]++ }
  END {
    for (id in ids) if (ids[id] != 1) exit 1
    for (scope in scopes) if (scopes[scope] != 1) exit 1
    for (configuration in configurations) {
      if (configurations[configuration] != 4) exit 1
      if (scopes[configuration SUBSEP "qiskit"] != 1) exit 1
      if (scopes[configuration SUBSEP "cirq"] != 1) exit 1
      if (scopes[configuration SUBSEP "pennylane"] != 1) exit 1
      if (scopes[configuration SUBSEP "cudaq"] != 1) exit 1
    }
  }
' "$QUEUE" || die "offline queue is not a unique schema-v2 Pass@1 queue"
CONFIGURATIONS="$(awk -F '\t' '{print $16}' "$QUEUE" | sort -u | wc -l | tr -d ' ')"
BASE_MODELS="$(awk -F '\t' '{print $2}' "$QUEUE" | sort -u | wc -l | tr -d ' ')"
((QUEUE_SHARDS == CONFIGURATIONS * 4)) || \
  die "offline queue must contain four framework shards per configuration"
EXPECTED_RECORDS=$((CONFIGURATIONS * 280))
if ((QUEUE_SHARDS != 36)); then
  if [[ -n "$CAMPAIGN_MANIFEST" && -n "$SCOPE_MANIFEST" ]]; then
    die "use only one of --campaign-manifest or --scope-manifest"
  fi
  if [[ -z "$CAMPAIGN_MANIFEST" && -z "$SCOPE_MANIFEST" ]]; then
    die "a non-default queue requires --campaign-manifest or --scope-manifest"
  fi
fi
if [[ -n "$CAMPAIGN_MANIFEST" ]]; then
  [[ -f "$CAMPAIGN_MANIFEST" ]] || die "campaign manifest is missing"
  QUEUE_SHA256="$(sha256_file "$QUEUE")"
  jq -e \
    --arg queue_sha256 "$QUEUE_SHA256" \
    --argjson models "$BASE_MODELS" \
    --argjson configurations "$CONFIGURATIONS" \
    --argjson shards "$QUEUE_SHARDS" \
    --argjson records "$EXPECTED_RECORDS" \
    '.artifacts.queue.sha256 == $queue_sha256 and
     .base_models == $models and
     .configurations == $configurations and
     .shards == $shards and
     .logical_requests == $records' \
    "$CAMPAIGN_MANIFEST" >/dev/null || die "campaign manifest does not match the canonical queue"
fi
if [[ -n "$SCOPE_MANIFEST" ]]; then
  QUEUE_SHA256="$(sha256_file "$QUEUE")"
  jq -e \
    --arg queue_sha256 "$QUEUE_SHA256" \
    --argjson models "$BASE_MODELS" \
    --argjson configurations "$CONFIGURATIONS" \
    --argjson shards "$QUEUE_SHARDS" \
    --argjson records "$EXPECTED_RECORDS" \
    '.schema_version == "qceval.pass1_scope.v1" and
     .queue_sha256 == $queue_sha256 and
     .base_models == $models and
     .configurations == $configurations and
     .shards == $shards and
     .logical_requests == $records and
     (.excluded_models | length) > 0' \
    "$SCOPE_MANIFEST" >/dev/null || die "scope manifest does not match the derived queue"
  PARENT_QUEUE="$(jq -r '.parent_queue' "$SCOPE_MANIFEST")"
  PARENT_QUEUE_SHA256="$(jq -r '.parent_queue_sha256' "$SCOPE_MANIFEST")"
  [[ -f "$PARENT_QUEUE" && "$(sha256_file "$PARENT_QUEUE")" == "$PARENT_QUEUE_SHA256" ]] || \
    die "scope manifest parent queue is missing or hash-invalid"
fi
jq -e \
  --argjson models "$BASE_MODELS" \
  --argjson configurations "$CONFIGURATIONS" \
  --argjson records "$EXPECTED_RECORDS" \
  '.base_models == $models and
   .configurations == $configurations and
   .records == $records and
   (.artifacts | length) == $configurations and
   all(.artifacts[]; .records == 280)' \
  "$CANDIDATES_DIR/manifest.json" >/dev/null || die "candidate manifest is incomplete"
while IFS=$'\t' read -r path expected; do
  [[ -f "$path" ]] || die "candidate artifact is missing: $path"
  [[ "$(sha256_file "$path")" == "$expected" ]] || die "candidate artifact hash mismatch: $path"
done < <(jq -r '.artifacts[] | [.path,.sha256] | @tsv' "$CANDIDATES_DIR/manifest.json")
mkdir -p "$OUT_DIR"
cp "$QUEUE" "$OUT_DIR/offline-queue.tsv"
cp "$CANDIDATES_DIR/manifest.json" "$OUT_DIR/candidate-manifest.json"
if ((PLAN_ONLY)); then
  echo "validated $QUEUE_SHARDS offline shards and $CONFIGURATIONS candidate artifacts"
  exit 0
fi

[[ "$INSTANCES" =~ ^[1-9][0-9]*$ ]] || die "--instances must be a positive integer"
((INSTANCES == 6)) || die "the production offline pool requires exactly six workers"
for variable in AMI_ID SUBNET_ID PROVISIONING_SECURITY_GROUP_ID EVALUATION_SECURITY_GROUP_ID KEY_NAME SSH_KEY; do
  [[ -n "${!variable}" ]] || die "missing required AWS option: $variable"
done
[[ -f "$SSH_KEY" ]] || die "SSH key not found: $SSH_KEY"
[[ "$(stat -c '%a' "$SSH_KEY")" == "600" ]] || die "SSH private key must have mode 0600"
for command in aws curl flock git jq python3 scp ssh uv; do
  command -v "$command" >/dev/null || die "required command not found: $command"
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "offline grading requires a clean tracked checkout"

CONTROLLER_IP="$(curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')"
[[ "$CONTROLLER_IP" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || die "could not determine controller IPv4 address"
SSH_CIDR="$CONTROLLER_IP/32"
aws ec2 describe-security-groups --region "$AWS_REGION" --group-ids "$PROVISIONING_SECURITY_GROUP_ID" \
  >"$OUT_DIR/provisioning-security-group.json"
aws ec2 describe-security-groups --region "$AWS_REGION" --group-ids "$EVALUATION_SECURITY_GROUP_ID" \
  >"$OUT_DIR/evaluation-security-group.json"
verify_security_group_snapshot "$OUT_DIR/provisioning-security-group.json" "$SSH_CIDR" 0
verify_security_group_snapshot "$OUT_DIR/evaluation-security-group.json" "$SSH_CIDR" 1

RUN_ID="qceval-offline-$(date -u +%Y%m%dT%H%M%SZ)-$$"
TMP_DIR="$(mktemp -d)"
BUNDLE="$TMP_DIR/qceval.bundle"
INSTANCE_IDS=()
SUCCESS=0

terminate_instances() {
  if ((${#INSTANCE_IDS[@]} == 0)); then
    return
  fi
  aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}" >/dev/null
  aws ec2 wait instance-terminated --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}"
  aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}" \
    --query 'Reservations[].Instances[].{InstanceId:InstanceId,State:State.Name}' --output json \
    >"$OUT_DIR/termination-readback.json"
  jq -e --argjson expected "${#INSTANCE_IDS[@]}" \
    'length == $expected and all(.[]; .State == "terminated")' \
    "$OUT_DIR/termination-readback.json" >/dev/null
}

cleanup() {
  local status=$?
  if ((status != 0 || SUCCESS == 0)); then
    terminate_instances || true
  fi
  rm -rf "$TMP_DIR"
  if ((status != 0 || SUCCESS == 0)); then
    echo "offline grading did not complete; retained artifacts are in $OUT_DIR" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

git bundle create "$BUNDLE" HEAD
while IFS= read -r instance_id; do INSTANCE_IDS+=("$instance_id"); done < <(
  aws ec2 run-instances \
    --region "$AWS_REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --count "$INSTANCES" \
    --key-name "$KEY_NAME" \
    --subnet-id "$SUBNET_ID" \
    --security-group-ids "$PROVISIONING_SECURITY_GROUP_ID" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_ID},{Key=Project,Value=QCircuitEval},{Key=RunId,Value=$RUN_ID},{Key=ManagedBy,Value=offline-grader}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=$RUN_ID-root},{Key=Project,Value=QCircuitEval},{Key=RunId,Value=$RUN_ID}]" \
    --query 'Instances[].InstanceId' --output text | tr '\t' '\n'
)
((${#INSTANCE_IDS[@]} == 6)) || die "AWS did not return six instance IDs"
printf '%s\n' "${INSTANCE_IDS[@]}" >"$OUT_DIR/instance-ids.txt"
aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}" \
  --query 'Reservations[].Instances[].{InstanceId:InstanceId,ImageId:ImageId,InstanceType:InstanceType,SubnetId:SubnetId,SecurityGroupIds:SecurityGroups[].GroupId,State:State.Name}' \
  --output json >"$OUT_DIR/aws-launch-readback.json"
jq -e \
  --argjson expected "$INSTANCES" \
  --arg image "$AMI_ID" \
  --arg instance_type "$INSTANCE_TYPE" \
  --arg subnet "$SUBNET_ID" \
  --arg provisioning_group "$PROVISIONING_SECURITY_GROUP_ID" \
  'length == $expected and all(.[];
    .ImageId == $image and
    .InstanceType == $instance_type and
    .SubnetId == $subnet and
    .SecurityGroupIds == [$provisioning_group] and
    (.State == "pending" or .State == "running"))' \
  "$OUT_DIR/aws-launch-readback.json" >/dev/null
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}"
mapfile -t IPS < <(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}" \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text | tr '\t' '\n')
((${#IPS[@]} == 6)) || die "not every worker has a public IP"
printf '%s\n' "${IPS[@]}" >"$OUT_DIR/worker-ips.txt"

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i "$SSH_KEY")
SCP=(scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i "$SSH_KEY")

provision() {
  local ip=$1 host="ubuntu@$1"
  for _ in {1..60}; do
    "${SSH[@]}" "$host" true >/dev/null 2>&1 && break
    sleep 5
  done
  "${SCP[@]}" "$BUNDLE" "$host:/home/ubuntu/qceval.bundle"
  "${SSH[@]}" "$host" bash -s <<'REMOTE'
set -Eeuo pipefail
cloud-init status --wait >/dev/null
sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential curl git-lfs jq
git lfs install --skip-smudge >/dev/null
rm -rf /home/ubuntu/qceval
GIT_LFS_SKIP_SMUDGE=1 git clone -q /home/ubuntu/qceval.bundle /home/ubuntu/qceval
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
cd /home/ubuntu/qceval
/home/ubuntu/.local/bin/uv sync --frozen -q
mkdir -p /home/ubuntu/qceval-results/shards /home/ubuntu/qceval-candidates
test ! -e /home/ubuntu/.openrouter-key
test ! -e /home/ubuntu/qceval/.env
REMOTE
}

pids=()
for ip in "${IPS[@]}"; do provision "$ip" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done

for instance_id in "${INSTANCE_IDS[@]}"; do
  aws ec2 modify-instance-attribute --region "$AWS_REGION" --instance-id "$instance_id" \
    --groups "$EVALUATION_SECURITY_GROUP_ID"
done
aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[@]}" \
  --query 'Reservations[].Instances[].{InstanceId:InstanceId,SecurityGroupIds:SecurityGroups[].GroupId,State:State.Name}' \
  --output json >"$OUT_DIR/evaluation-attachment-readback.json"
jq -e \
  --argjson expected "$INSTANCES" \
  --arg evaluation_group "$EVALUATION_SECURITY_GROUP_ID" \
  'length == $expected and all(.[];
    .SecurityGroupIds == [$evaluation_group] and .State == "running")' \
  "$OUT_DIR/evaluation-attachment-readback.json" >/dev/null
for index in "${!INSTANCE_IDS[@]}"; do
  attached="$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "${INSTANCE_IDS[$index]}" \
    --query 'Reservations[0].Instances[0].SecurityGroups[].GroupId' --output text)"
  [[ "$attached" == "$EVALUATION_SECURITY_GROUP_ID" ]] || die "worker ${INSTANCE_IDS[$index]} has unexpected security groups"
  "${SSH[@]}" "ubuntu@${IPS[$index]}" bash -s <<'REMOTE'
set -Eeuo pipefail
test ! -e /home/ubuntu/.openrouter-key
test ! -e /home/ubuntu/qceval/.env
if curl -fsS --max-time 10 https://openrouter.ai >/dev/null 2>&1; then
  echo "OpenRouter remained reachable after evaluation isolation" >&2
  exit 1
fi
REMOTE
done

copy_candidate() {
  local ip=$1 configuration=$2 local_path remote_path
  local_path="$(jq -r --arg configuration "$configuration" '.artifacts[] | select(.configuration_id == $configuration) | .path' "$CANDIDATES_DIR/manifest.json")"
  [[ -f "$local_path" ]] || return 1
  remote_path="/home/ubuntu/qceval-candidates/$(basename "$local_path")"
  if ! "${SSH[@]}" "ubuntu@$ip" test -f "$remote_path"; then
    "${SCP[@]}" "$local_path" "ubuntu@$ip:$remote_path"
  fi
  printf '%s' "$remote_path"
}

CALIBRATION_DIR="$OUT_DIR/calibration"
mkdir -p "$CALIBRATION_DIR"
CALIBRATION_MODEL="$(jq -r '.artifacts[0].model_id' "$CANDIDATES_DIR/manifest.json")"
CALIBRATION_CONFIGURATION="$(jq -r '.artifacts[0].configuration_id' "$CANDIDATES_DIR/manifest.json")"
CALIBRATION_INPUT="$(copy_candidate "${IPS[0]}" "$CALIBRATION_CONFIGURATION")"
for workers in 2 4 8; do
  remote_out="/home/ubuntu/qceval-results/calibration-$workers.jsonl"
  start="$(date +%s%N)"
  calibration_status=failed
  if "${SSH[@]}" "ubuntu@${IPS[0]}" bash -s -- "$CALIBRATION_MODEL" "$CALIBRATION_INPUT" "$remote_out" "$workers" "$EVAL_TIMEOUT" <<'REMOTE'
set -Eeuo pipefail
model=$1 input=$2 output=$3 workers=$4 timeout=$5
cd /home/ubuntu/qceval
env -u OPENROUTER_API_KEY .venv/bin/qceval run \
  --provider openrouter --model "$model" --framework qiskit --suite all \
  --source-hint 37bffc7ae6b98ecd2c78bdfba1d249c3c15ded70 \
  --regrade qiskit --input "$input" --out "$output" --output-format jsonl \
  --evaluation-workers "$workers" --eval-timeout "$timeout" --samples-per-task 1 --pass-k 1 --max-attempts 1
REMOTE
  then
    if "${SCP[@]}" "ubuntu@${IPS[0]}:$remote_out" "$CALIBRATION_DIR/workers-$workers.jsonl" && \
      validate_shard "$CALIBRATION_DIR/workers-$workers.jsonl" "$CALIBRATION_MODEL" \
        "$CALIBRATION_CONFIGURATION" qiskit 70; then
      calibration_status=passed
    fi
  fi
  elapsed_ms=$((($(date +%s%N) - start) / 1000000))
  printf '%s\t%s\t%s\n' "$workers" "$elapsed_ms" "$calibration_status" >>"$CALIBRATION_DIR/attempts.tsv"
done

EVALUATION_WORKERS="$(python3 "$REPO_ROOT/scripts/select_offline_workers.py" "$CALIBRATION_DIR")"

STATE="$OUT_DIR/state"
mkdir -p "$STATE/pending" "$STATE/running" "$STATE/done" "$STATE/failed" "$OUT_DIR/shards" "$OUT_DIR/logs" "$OUT_DIR/merged"
while IFS= read -r line; do
  job_id="${line%%$'\t'*}"
  printf '%s\n' "$line" >"$STATE/pending/$job_id.job"
done <"$QUEUE"

lock_queue() { flock -x "$QUEUE_LOCK_FD"; }
unlock_queue() { flock -u "$QUEUE_LOCK_FD"; }
claim_job() {
  local worker=$1 files=() file claimed
  lock_queue
  files=("$STATE/pending"/*.job)
  if [[ ! -e "${files[0]}" ]]; then unlock_queue; return 1; fi
  file="${files[0]}"
  claimed="$STATE/running/${worker}__$(basename "$file")"
  mv "$file" "$claimed"
  unlock_queue
  printf '%s' "$claimed"
}

run_remote_job() {
  local ip=$1 job_file=$2
  local job_id model setting protocol framework suite max_tasks endpoint output source cap parameter revision temperature assigned configuration remote_input remote_out staged workers timeout
  IFS=$'\t' read -r job_id model setting protocol framework suite max_tasks endpoint output source cap parameter revision temperature assigned configuration <"$job_file"
  remote_input="$(copy_candidate "$ip" "$configuration")"
  remote_out="/home/ubuntu/qceval-results/shards/$job_id.jsonl"
  staged="$OUT_DIR/shards/.${job_id}.staging"
  workers="$EVALUATION_WORKERS"
  timeout="$EVAL_TIMEOUT"
  if [[ "$framework" == "cudaq" ]]; then
    workers="$CUDAQ_EVALUATION_WORKERS"
    timeout="$CUDAQ_EVAL_TIMEOUT"
  fi
  if ! "${SSH[@]}" "ubuntu@$ip" bash -s -- "$model" "$framework" "$remote_input" "$remote_out" "$workers" "$timeout" >"$OUT_DIR/logs/$job_id.log" 2>&1 <<'REMOTE'
set -Eeuo pipefail
model=$1 framework=$2 input=$3 output=$4 workers=$5 timeout=$6
cd /home/ubuntu/qceval
env -u OPENROUTER_API_KEY .venv/bin/qceval run \
  --provider openrouter --model "$model" --framework "$framework" --suite all \
  --source-hint 37bffc7ae6b98ecd2c78bdfba1d249c3c15ded70 \
  --regrade "$framework" --input "$input" --out "$output" --output-format jsonl \
  --evaluation-workers "$workers" --eval-timeout "$timeout" --samples-per-task 1 --pass-k 1 --max-attempts 1 --progress
REMOTE
  then
    return 1
  fi
  if ! "${SCP[@]}" "ubuntu@$ip:$remote_out" "$staged"; then
    return 1
  fi
  if ! validate_shard "$staged" "$model" "$configuration" "$framework" 70; then
    rm -f "$staged"
    return 1
  fi
  mv "$staged" "$OUT_DIR/shards/$job_id.jsonl"
}

worker_loop() {
  local ip=$1 worker=$2 job_file base
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
for index in "${!IPS[@]}"; do worker_loop "${IPS[$index]}" "worker-$index" & worker_pids+=("$!"); done
for pid in "${worker_pids[@]}"; do wait "$pid"; done
failed=("$STATE/failed"/*.job)
[[ ! -e "${failed[0]}" ]] || die "one or more offline shards failed"
[[ "$(find "$OUT_DIR/shards" -maxdepth 1 -name '*.jsonl' | wc -l | tr -d ' ')" == "$QUEUE_SHARDS" ]] || \
  die "offline pool did not produce $QUEUE_SHARDS shards"

while IFS=$'\t' read -r model setting protocol configuration; do
  inputs=()
  while IFS=$'\t' read -r job_id queue_model queue_setting queue_protocol framework suite max_tasks endpoint output source cap parameter revision temperature assigned queue_configuration; do
    if [[ "$queue_configuration" == "$configuration" ]]; then
      inputs+=("$OUT_DIR/shards/$job_id.jsonl")
    fi
  done <"$QUEUE"
  merged="$OUT_DIR/merged/${configuration}__pass1.regraded.jsonl"
  uv run python scripts/merge_run_records.py --out "$merged" "${inputs[@]}"
  python3 - "$merged" "$model" "$configuration" <<'PY'
import json
import sys
path, model, configuration = sys.argv[1:]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
results = [row for row in rows if row.get("kind") == "result"]
if len(results) != 280 or {row.get("model") for row in results} != {model}:
    raise SystemExit("merged configuration artifact is not exactly 280 records")
if {
    (((row.get("provider_response") or {}).get("metadata") or {}).get("route") or {}).get("configuration_id")
    for row in results
} != {configuration}:
    raise SystemExit("merged artifact contains incompatible configuration provenance")
summary = rows[-1].get("summary") or {}
if set(summary.get("by_suite") or {}) != {"core", "qec"}:
    raise SystemExit("merged model artifact does not report Core and QEC separately")
PY
done < <(awk -F '\t' '{print $2 "\t" $3 "\t" $4 "\t" $16}' "$QUEUE" | sort -u)

uv run python scripts/summarize_run_costs.py "$OUT_DIR"/merged/*.jsonl \
  --lite-prompts 70 --target-prompts 70 \
  --json-out "$OUT_DIR/score-cost.json" --tsv-out "$OUT_DIR/score-cost.tsv" \
  --markdown-out "$OUT_DIR/score-cost.md" >"$OUT_DIR/score-cost.stdout.tsv"

terminate_instances
INSTANCE_IDS=()
SUCCESS=1
echo "completed $QUEUE_SHARDS offline shards and $CONFIGURATIONS 280-record configuration artifacts"
