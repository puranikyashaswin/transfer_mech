#!/usr/bin/env bash
# run_gate0.sh — Local orchestrator. Starts the EC2 instance, provisions it,
# runs the resumable Gate 0 runner over SSH, pulls results back, stops the
# instance. Safe to re-run: the remote runner resumes from checkpoints, and
# this script is idempotent (it won't double-start or re-provision).
#
# Usage:
#   ./run_gate0.sh            # full run (163 bugs)
#   LIMIT=20 ./run_gate0.sh   # smoke run on 20 bugs
#   ./run_gate0.sh --stop     # just stop the instance
set -euo pipefail

# ---- Config ----------------------------------------------------------------
REGION="${REGION:-us-east-2}"
INSTANCE_ID="${INSTANCE_ID:?set INSTANCE_ID to your stopped GPU instance (g6e.xlarge or g5.xlarge)}"
KEY="${KEY:?set KEY to your EC2 SSH key path}"
SSH_USER="${SSH_USER:-ubuntu}"
S3_BUCKET="${S3_BUCKET:?set S3_BUCKET for result backups}"
LIMIT="${LIMIT:-}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the folder holding humaneval_java.jsonl and temp_hej/}"
REMOTE_HOME="/home/$SSH_USER"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -i "$KEY")

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---- Stop-only mode ---------------------------------------------------------
if [ "${1:-}" = "--stop" ]; then
  log "Stopping instance $INSTANCE_ID ..."
  aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
  log "Stop requested."
  exit 0
fi

# ---- 1. Start instance (idempotent) ----------------------------------------
STATE=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
log "Instance $INSTANCE_ID state: $STATE"
if [ "$STATE" != "running" ]; then
  log "Starting instance..."
  aws ec2 start-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
  log "Instance running."
fi

IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
if [ -z "$IP" ] || [ "$IP" = "None" ]; then
  log "No public IP yet; waiting 20s and re-reading..."
  sleep 20
  IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
fi
log "Public IP: $IP"

# ---- 2. Wait for SSH --------------------------------------------------------
log "Waiting for SSH..."
for i in $(seq 1 40); do
  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" 'echo SSH_OK' 2>/dev/null | grep -q SSH_OK; then
    log "SSH is up."
    break
  fi
  [ "$i" = "40" ] && { log "FATAL: SSH never came up"; exit 7; }
  sleep 5
done

# ---- 3. Provision: rsync harness + dataset ---------------------------------
log "Provisioning: rsync eval_harness + runner + dataset..."
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" 'mkdir -p ~/transfer_mech ~/data'
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  "$LOCAL_REPO/transfer_mech/eval_harness/" "$SSH_USER@$IP:~/transfer_mech/eval_harness/"
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  "$LOCAL_REPO/transfer_mech/runner/" "$SSH_USER@$IP:~/transfer_mech/runner/"
# Dataset: prefer pulling from S3 if present (durable), else rsync from local.
if ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "aws s3 ls s3://$S3_BUCKET/datasets/temp_hej/ >/dev/null 2>&1"; then
  log "Dataset found in S3; syncing down (durable source)."
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
    "aws s3 sync s3://$S3_BUCKET/datasets/temp_hej/ ~/data/temp_hej/ --no-progress"
else
  log "Dataset not in S3; rsyncing from local and pushing to S3 for durability."
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "$DATA_DIR/temp_hej/" "$SSH_USER@$IP:~/data/temp_hej/"
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
    "aws s3 sync ~/data/temp_hej/ s3://$S3_BUCKET/datasets/temp_hej/ --no-progress" || \
    log "WARN: could not push dataset to S3 (continuing)"
fi

# ---- 4. Run Gate 0 (resumable; survives SSH drop via nohup + checkpoint) ----
log "Launching Gate 0 runner on remote (nohup, resumable)..."
LIMIT_ENV=""; [ -n "$LIMIT" ] && LIMIT_ENV="LIMIT=$LIMIT"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "
  chmod +x ~/transfer_mech/runner/gate0_remote.sh
  cd ~/transfer_mech
  nohup env $LIMIT_ENV bash ~/transfer_mech/runner/gate0_remote.sh > ~/transfer_mech/gate0_nohup.out 2>&1 &
  echo \$! > ~/transfer_mech/gate0.pid
  echo LAUNCHED
"

# ---- 5. Poll until done (resilient to SSH drops) ---------------------------
log "Polling for completion (Ctrl-C safe: remote keeps running; re-run to resume polling)..."
while true; do
  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
      'test -f ~/transfer_mech/results/gate0/gate0_report.json && echo DONE' 2>/dev/null | grep -q DONE; then
    log "Gate 0 report present."
    break
  fi
  if ! ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
      'kill -0 $(cat ~/transfer_mech/gate0.pid) 2>/dev/null && echo ALIVE' 2>/dev/null | grep -q ALIVE; then
    # process not alive and no report -> check nohup output for fatal
    log "WARN: runner process not alive and no report yet; tail of nohup:"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" 'tail -20 ~/transfer_mech/gate0_nohup.out' 2>/dev/null || true
    # give it one more chance in case it just finished writing
    sleep 10
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" \
      'test -f ~/transfer_mech/results/gate0/gate0_report.json && echo DONE' 2>/dev/null | grep -q DONE && break
    log "FATAL: runner died without producing a report. See nohup output above."
    exit 8
  fi
  sleep 20
done

# ---- 6. Pull results back + show report ------------------------------------
log "Pulling results back to local..."
mkdir -p "$LOCAL_REPO/transfer_mech/results/gate0"
rsync -az -e "ssh ${SSH_OPTS[*]}" \
  "$SSH_USER@$IP:~/transfer_mech/results/gate0/" "$LOCAL_REPO/transfer_mech/results/gate0/"

log "================ GATE 0 REPORT ================"
cat "$LOCAL_REPO/transfer_mech/results/gate0/gate0_report.json"
log "==============================================="

PASSED=$(python3 -c "import json;print(json.load(open('$LOCAL_REPO/transfer_mech/results/gate0/gate0_report.json'))['gate0_passed'])")
log "GATE0_PASSED=$PASSED"
log "Results also synced to s3://$S3_BUCKET/transfer_mech/gate0/"
log "Instance left RUNNING (stop with: ./run_gate0.sh --stop) so you can inspect. Remember to stop it."
