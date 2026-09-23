#!/usr/bin/env bash
# run_gate1.sh — Local orchestrator for the Gate 1 diversity pilot.
# Provisions a running GPU instance, rsyncs harness + data, runs the resumable
# remote runner, pulls results back. Idempotent; safe to re-run to resume.
# Usage:
#   INSTANCE_IP=<ip> ./run_gate1.sh            # full pilot (50 bugs)
#   INSTANCE_IP=<ip> N_PILOT=20 ./run_gate1.sh # smaller smoke
#   INSTANCE_IP=<ip> ./run_gate1.sh --pull     # just pull results
set -euo pipefail

REGION="${REGION:-us-east-2}"
KEY="${KEY:?set KEY to your EC2 SSH key path}"
SSH_USER="${SSH_USER:-ubuntu}"
S3_BUCKET="${S3_BUCKET:?set S3_BUCKET for result backups}"
INSTANCE_IP="${INSTANCE_IP:?set INSTANCE_IP to the running GPU instance public IP}"
N_PILOT="${N_PILOT:-50}"
K_SAMPLES="${K_SAMPLES:-5}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the folder holding humaneval_java.jsonl and temp_hej/}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -i "$KEY")
RHOST="$SSH_USER@$INSTANCE_IP"
log(){ printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

if [ "${1:-}" = "--pull" ]; then
  mkdir -p "$LOCAL_REPO/transfer_mech/results/gate1"
  rsync -az -e "ssh ${SSH_OPTS[*]}" "$RHOST:~/transfer_mech/results/gate1/" "$LOCAL_REPO/transfer_mech/results/gate1/"
  log "Pulled results. Summary:"
  cat "$LOCAL_REPO/transfer_mech/results/gate1/gate1_summary.json" 2>/dev/null || log "no summary yet"
  exit 0
fi

log "Target: $RHOST  N_PILOT=$N_PILOT K=$K_SAMPLES"
log "Waiting for SSH..."
for i in $(seq 1 40); do
  ssh "${SSH_OPTS[@]}" "$RHOST" 'echo SSH_OK' 2>/dev/null | grep -q SSH_OK && { log "SSH up."; break; }
  [ "$i" = "40" ] && { log "FATAL: SSH never came up"; exit 7; }
  sleep 5
done

log "Provisioning: rsync harness + runner + bugs file..."
ssh "${SSH_OPTS[@]}" "$RHOST" 'mkdir -p ~/transfer_mech ~/data'
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "$LOCAL_REPO/transfer_mech/eval_harness/" "$RHOST:~/transfer_mech/eval_harness/"
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "$LOCAL_REPO/transfer_mech/runner/" "$RHOST:~/transfer_mech/runner/"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$DATA_DIR/humaneval_java.jsonl" "$RHOST:~/data/humaneval_java.jsonl"

# dataset: pull from S3 if present (durable), else rsync local
if ssh "${SSH_OPTS[@]}" "$RHOST" "aws s3 ls s3://$S3_BUCKET/datasets/temp_hej/ >/dev/null 2>&1"; then
  log "Dataset from S3 (durable)."
  ssh "${SSH_OPTS[@]}" "$RHOST" "aws s3 sync s3://$S3_BUCKET/datasets/temp_hej/ ~/data/temp_hej/ --no-progress"
else
  log "Dataset rsync from local + push to S3."
  rsync -az -e "ssh ${SSH_OPTS[*]}" "$DATA_DIR/temp_hej/" "$RHOST:~/data/temp_hej/"
  ssh "${SSH_OPTS[@]}" "$RHOST" "aws s3 sync ~/data/temp_hej/ s3://$S3_BUCKET/datasets/temp_hej/ --no-progress" || true
fi

log "Launching Gate 1 runner (nohup, resumable)..."
ssh "${SSH_OPTS[@]}" "$RHOST" "
  chmod +x ~/transfer_mech/runner/gate1_remote.sh
  cd ~/transfer_mech
  nohup env N_PILOT=$N_PILOT K_SAMPLES=$K_SAMPLES bash ~/transfer_mech/runner/gate1_remote.sh > ~/transfer_mech/gate1_nohup.out 2>&1 &
  echo \$! > ~/transfer_mech/gate1.pid
  echo LAUNCHED
"

log "Polling (Ctrl-C safe; re-run to resume)..."
while true; do
  if ssh "${SSH_OPTS[@]}" "$RHOST" 'test -f ~/transfer_mech/results/gate1/gate1_summary.json && echo DONE' 2>/dev/null | grep -q DONE; then
    log "Summary present."; break
  fi
  if ! ssh "${SSH_OPTS[@]}" "$RHOST" 'kill -0 $(cat ~/transfer_mech/gate1.pid) 2>/dev/null && echo ALIVE' 2>/dev/null | grep -q ALIVE; then
    log "WARN: runner not alive, no summary yet. nohup tail:"
    ssh "${SSH_OPTS[@]}" "$RHOST" 'tail -25 ~/transfer_mech/gate1_nohup.out' 2>/dev/null || true
    sleep 10
    ssh "${SSH_OPTS[@]}" "$RHOST" 'test -f ~/transfer_mech/results/gate1/gate1_summary.json && echo DONE' 2>/dev/null | grep -q DONE && break
    log "FATAL: runner died without summary."; exit 8
  fi
  # progress line
  ssh "${SSH_OPTS[@]}" "$RHOST" 'wc -l < ~/transfer_mech/results/gate1/gate1_checkpoint.jsonl 2>/dev/null' 2>/dev/null | sed 's/^/  checkpoint bugs done: /' || true
  sleep 30
done

log "Pulling results..."
mkdir -p "$LOCAL_REPO/transfer_mech/results/gate1"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$RHOST:~/transfer_mech/results/gate1/" "$LOCAL_REPO/transfer_mech/results/gate1/"
log "================ GATE 1 SUMMARY ================"
cat "$LOCAL_REPO/transfer_mech/results/gate1/gate1_summary.json"
log "==============================================="
log "Instance left RUNNING ($INSTANCE_IP). Stop it when done."
