#!/usr/bin/env bash
# gate0_remote.sh — Resumable, locked, S3-backed Gate 0 runner.
# Runs ON the EC2 host. Idempotent: safe to re-run after any interruption;
# it resumes from the last completed step and never redoes finished work.
#
# Design (senior-engineer practices):
#   - set -euo pipefail         : fail fast, no silent errors
#   - flock on a lockfile       : no two runs at once (no leakage/dupes)
#   - step markers in STATE_DIR : each completed step drops a .done marker
#   - per-bug JSONL checkpoint  : Gate 0 appends per bug; resume skips done bugs
#   - S3 sync after each phase  : durable record even if the instance dies
#   - structured log + manifest : everything recorded, timestamped, hashed
set -euo pipefail

# ---- Config (overridable via env) ------------------------------------------
RUN_ID="${RUN_ID:-gate0-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_ROOT="${WORK_ROOT:-$HOME/transfer_mech}"
STATE_DIR="${STATE_DIR:-$WORK_ROOT/.state/gate0}"
RESULTS_DIR="${RESULTS_DIR:-$WORK_ROOT/results/gate0}"
DATASET_DIR="${DATASET_DIR:-$HOME/data/temp_hej}"
S3_BUCKET="${S3_BUCKET:?set S3_BUCKET for result backups}"
S3_PREFIX="${S3_PREFIX:-transfer_mech/gate0}"
LOCKFILE="${LOCKFILE:-$STATE_DIR/gate0.lock}"
LOGFILE="${LOGFILE:-$RESULTS_DIR/gate0.log}"
LIMIT="${LIMIT:-}"   # e.g. LIMIT=20 for a smoke run

mkdir -p "$STATE_DIR" "$RESULTS_DIR"
exec > >(tee -a "$LOGFILE") 2>&1   # record everything to log + stdout

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# ---- Lock: prevent concurrent runs -----------------------------------------
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  log "FATAL: another gate0 run holds the lock ($LOCKFILE). Exiting."
  exit 3
fi
log "Lock acquired. RUN_ID=$RUN_ID"

# ---- Manifest: record environment for reproducibility ----------------------
manifest() {
  {
    echo "run_id=$RUN_ID"
    echo "host=$(hostname)"
    echo "instance_id=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id || echo unknown)"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "java=$(command -v java || echo MISSING)"
    echo "javac=$(command -v javac || echo MISSING)"
    java -version 2>&1 | head -1 || true
    echo "python3=$(python3 --version 2>&1 || echo MISSING)"
    echo "dataset_dir=$DATASET_DIR"
    echo "dataset_buggy_count=$(ls "$DATASET_DIR"/src/main/java/humaneval/buggy/ 2>/dev/null | wc -l)"
    echo "dataset_test_count=$(ls "$DATASET_DIR"/src/test/java/humaneval/ 2>/dev/null | wc -l)"
    echo "script_sha256=$(sha256sum "$0" | awk '{print $1}')"
  } > "$RESULTS_DIR/manifest.txt"
  log "Manifest written."
}

s3_sync() {
  log "Syncing results to s3://$S3_BUCKET/$S3_PREFIX/ ..."
  aws s3 sync "$RESULTS_DIR/" "s3://$S3_BUCKET/$S3_PREFIX/" --no-progress || \
    log "WARN: S3 sync failed (will retry next phase)"
}

step_done() { [ -f "$STATE_DIR/$1.done" ]; }
mark_done() { date -u +%Y-%m-%dT%H:%M:%SZ > "$STATE_DIR/$1.done"; log "STEP DONE: $1"; }

# ---- Step 1: toolchain ------------------------------------------------------
if ! step_done toolchain; then
  log "STEP toolchain: verifying JDK..."
  if ! command -v javac >/dev/null 2>&1; then
    log "javac not found; installing Temurin 17 to \$HOME/jdk ..."
    mkdir -p "$HOME/jdk"
    curl -sL -o /tmp/temurin17.tar.gz \
      "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
    tar -xzf /tmp/temurin17.tar.gz -C "$HOME/jdk"
    rm -f /tmp/temurin17.tar.gz
  fi
  export JAVA_HOME="$(echo "$HOME"/jdk/jdk-17*/ 2>/dev/null || true)"
  [ -n "$JAVA_HOME" ] && export PATH="$JAVA_HOME/bin:$PATH"
  command -v javac >/dev/null 2>&1 || { log "FATAL: no javac after install"; exit 4; }
  mark_done toolchain
else
  export JAVA_HOME="$(echo "$HOME"/jdk/jdk-17*/ 2>/dev/null || true)"
  [ -n "$JAVA_HOME" ] && export PATH="$JAVA_HOME/bin:$PATH"
  log "STEP toolchain: already done, skipping."
fi

# ---- Step 2: dataset present ------------------------------------------------
if ! step_done dataset; then
  log "STEP dataset: verifying HumanEval-Java dataset..."
  [ -d "$DATASET_DIR/src/main/java/humaneval/buggy" ] || { log "FATAL: dataset missing at $DATASET_DIR"; exit 5; }
  [ -d "$DATASET_DIR/src/test/java/humaneval" ] || { log "FATAL: tests missing"; exit 5; }
  mark_done dataset
else
  log "STEP dataset: already done, skipping."
fi

manifest

# ---- Step 3: Gate 0 checks (resumable per-bug) ------------------------------
# gate0_checks.py writes per-bug JSONL; on resume it skips bugs already in the
# checkpoint file. We pass --resume so it picks up where it left off.
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../eval_harness" && pwd)"
LIMIT_ARG=(); [ -n "$LIMIT" ] && LIMIT_ARG=(--limit "$LIMIT")

if ! step_done gate0_run; then
  log "STEP gate0_run: running Gate 0 checks (resumable)..."
  python3 "$HARNESS_DIR/gate0_checks.py" \
    --dataset "$DATASET_DIR" \
    --workdir "$RESULTS_DIR" \
    --java "$(command -v java)" \
    --javac "$(command -v javac)" \
    --resume \
    "${LIMIT_ARG[@]}"
  mark_done gate0_run
  s3_sync
else
  log "STEP gate0_run: already done, skipping."
fi

# ---- Step 4: verdict + final sync -------------------------------------------
if ! step_done verdict; then
  log "STEP verdict: reading gate0_report.json ..."
  if [ -f "$RESULTS_DIR/gate0_report.json" ]; then
    PASSED=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/gate0_report.json'))['gate0_passed'])")
    log "GATE0_PASSED=$PASSED"
  else
    log "FATAL: gate0_report.json missing"
    exit 6
  fi
  mark_done verdict
  s3_sync
else
  log "STEP verdict: already done."
fi

log "GATE 0 RUN COMPLETE. Results: $RESULTS_DIR  |  S3: s3://$S3_BUCKET/$S3_PREFIX/"
cat "$RESULTS_DIR/gate0_report.json"
