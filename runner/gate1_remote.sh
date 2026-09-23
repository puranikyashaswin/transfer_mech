#!/usr/bin/env bash
# gate1_remote.sh — Resumable, locked, S3-backed Gate 1 (diversity pilot) runner.
# Runs ON the GPU instance. Idempotent: re-run resumes from per-bug checkpoint.
set -euo pipefail

RUN_ID="${RUN_ID:-gate1-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_ROOT="${WORK_ROOT:-$HOME/transfer_mech}"
STATE_DIR="${STATE_DIR:-$WORK_ROOT/.state/gate1}"
RESULTS_DIR="${RESULTS_DIR:-$WORK_ROOT/results/gate1}"
DATASET_DIR="${DATASET_DIR:-$HOME/data/temp_hej}"
BUGS_FILE="${BUGS_FILE:-$HOME/data/humaneval_java.jsonl}"
S3_BUCKET="${S3_BUCKET:?set S3_BUCKET for result backups}"
S3_PREFIX="${S3_PREFIX:-transfer_mech/gate1}"
LOCKFILE="${LOCKFILE:-$STATE_DIR/gate1.lock}"
LOGFILE="${LOGFILE:-$RESULTS_DIR/gate1.log}"
N_PILOT="${N_PILOT:-50}"
K_SAMPLES="${K_SAMPLES:-5}"

mkdir -p "$STATE_DIR" "$RESULTS_DIR"
exec > >(tee -a "$LOGFILE") 2>&1
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

exec 9>"$LOCKFILE"
if ! flock -n 9; then log "FATAL: lock held ($LOCKFILE)"; exit 3; fi
log "Lock acquired. RUN_ID=$RUN_ID  N_PILOT=$N_PILOT  K=$K_SAMPLES"

# manifest
{
  echo "run_id=$RUN_ID"; echo "host=$(hostname)"
  echo "instance_id=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id || echo unknown)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo none)"
  python3 -c "import torch,transformers;print('torch='+torch.__version__,'cuda='+str(torch.cuda.is_available()),'transformers='+transformers.__version__)" 2>&1 || true
  echo "java=$(java -version 2>&1 | head -1 || echo MISSING)"
  echo "n_pilot=$N_PILOT k=$K_SAMPLES"
} > "$RESULTS_DIR/manifest.txt"

s3_sync(){ log "S3 sync..."; aws s3 sync "$RESULTS_DIR/" "s3://$S3_BUCKET/$S3_PREFIX/" --no-progress || log "WARN: S3 sync failed"; }
step_done(){ [ -f "$STATE_DIR/$1.done" ]; }
mark_done(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$STATE_DIR/$1.done"; log "STEP DONE: $1"; }

# JDK (for JUnit eval)
if ! command -v javac >/dev/null 2>&1; then
  log "Installing JDK..."
  mkdir -p "$HOME/jdk"
  curl -sL -o /tmp/t17.tgz "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
  tar -xzf /tmp/t17.tgz -C "$HOME/jdk"; rm -f /tmp/t17.tgz
fi
export JAVA_HOME="$(echo "$HOME"/jdk/jdk-17*/ 2>/dev/null || true)"
[ -n "$JAVA_HOME" ] && export PATH="$JAVA_HOME/bin:$PATH"
# fallback to system java (AMI ships Java 11)
command -v javac >/dev/null 2>&1 || { log "FATAL: no javac"; exit 4; }

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../eval_harness" && pwd)"

if ! step_done gate1_run; then
  log "STEP gate1_run: running pilot (resumable per-bug)..."
  python3 "$HARNESS_DIR/gate1_pilot.py" \
    --bugs "$BUGS_FILE" --dataset "$DATASET_DIR" \
    --out "$RESULTS_DIR/gate1_checkpoint.jsonl" \
    --summary "$RESULTS_DIR/gate1_summary.json" \
    --n-pilot "$N_PILOT" --k "$K_SAMPLES" \
    --java "$(command -v java)" --javac "$(command -v javac)"
  mark_done gate1_run
  s3_sync
else
  log "STEP gate1_run: already done, skipping."
fi

log "GATE 1 PILOT COMPLETE. Results: $RESULTS_DIR | S3: s3://$S3_BUCKET/$S3_PREFIX/"
cat "$RESULTS_DIR/gate1_summary.json"
