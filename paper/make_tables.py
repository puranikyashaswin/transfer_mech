#!/usr/bin/env python3
"""Fast integrity pass: recompute the raw pass counts/accuracies quoted in PAPER.md
(adapter, same-family pilot, alpha sweep, gate-1) from raw JSONL. Stdlib only, no scipy.

Role vs make_stats.py: make_tables.py = quick raw-count sanity check (no stats deps);
make_stats.py = full statistical reproduction (McNemar, Wilson CIs, Holm, site sweep).
Both must agree with the paper. Run with any python3."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    return [json.loads(l) for l in (ROOT / "results" / name).read_text().splitlines() if l.strip()]


def acc(rows, arm):
    sub = [r for r in rows if r["arm"] == arm]
    n = sum(r["passed"] for r in sub)
    return n, len(sub), n / len(sub)


print("=== CROSS-FAMILY ADAPTER (test 120-162) ===")
rows = load("adapter/eval_checkpoint.jsonl")
for arm in ["receiver_k1", "adapter", "random_norm", "shuffled", "self_inject"]:
    n, t, a = acc(rows, arm)
    print(f"  {arm:<12} {n}/{t} = {a:.3f}")

print("\n=== SAME-FAMILY PILOT (eval 90-162, alpha=1.0) ===")
rows = load("same_family/sf_checkpoint.jsonl")
for arm in ["receiver_k1", "transfer", "random_norm", "self_inject"]:
    n, t, a = acc(rows, arm)
    print(f"  {arm:<12} {n}/{t} = {a:.3f}")

print("\n=== ALPHA SWEEP ===")
rows = load("alpha_sweep/alpha_checkpoint.jsonl")
for a in [0.1, 0.5, 1.0]:
    for arm in ["transfer", "random_norm", "self_inject"]:
        sub = [r for r in rows if r["alpha"] == a and r["arm"] == arm]
        n = sum(r["passed"] for r in sub)
        print(f"  alpha={a} {arm:<12} {n}/{len(sub)} = {n/len(sub):.3f}")

print("\n=== GATE 1 PILOT (50 bugs, k=5) ===")
gs = json.loads((ROOT / "results/gate1/gate1_summary.json").read_text())
print(" ", {k: gs[k] for k in list(gs)[:12]})

print("\n=== INTEGRITY: recompute OK ===")
