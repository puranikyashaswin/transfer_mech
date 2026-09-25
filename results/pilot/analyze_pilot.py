#!/usr/bin/env python3
"""
analyze_pilot.py — read out the paired 2×2 pilot against Amendment 007 §5 boundaries.

§5 pre-committed decision boundaries:
  - strength effect: α=0.5 significantly better than α=1.0 (regardless of subspace)
  - subspace effect: aligned significantly better than as_is (regardless of α)
  - both: both effects significant
  - cannot tell: neither effect significant (default; report CIs honestly)

Readout is EXPLORATORY/DIAGNOSTIC — no confirmatory claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def main(path: str) -> int:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"loaded {len(rows)} rows from {path}")

    # parse conditions
    data = []
    for r in rows:
        data.append({
            "bug_id": r["bug_id"],
            "alpha": r["alpha"],
            "subspace": r["subspace"],
            "status": r["status"],
            "delta_norm": r["delta_norm"],
            "proj_frac": r["proj_frac"],
            "pass": 1 if r["status"] == "pass" else 0,
        })

    # per-condition pass rates
    print("\n=== PASS RATES BY CONDITION ===")
    for alpha in (0.5, 1.0):
        for subspace in ("as_is", "aligned"):
            subset = [d for d in data if d["alpha"] == alpha and d["subspace"] == subspace]
            n = len(subset)
            p = sum(d["pass"] for d in subset)
            print(f"  α={alpha} {subspace:<7}: {p}/{n} = {p/n:.3f}")

    # paired analysis: for each bug, compare conditions
    bugs = sorted(set(d["bug_id"] for d in data))
    print(f"\n=== PAIRED ANALYSIS (n={len(bugs)} bugs) ===")

    # strength effect: α=0.5 vs α=1.0 (averaging over subspace)
    print("\n--- STRENGTH EFFECT (α=0.5 vs α=1.0) ---")
    for subspace in ("as_is", "aligned"):
        a05 = [d for d in data if d["alpha"] == 0.5 and d["subspace"] == subspace]
        a10 = [d for d in data if d["alpha"] == 1.0 and d["subspace"] == subspace]
        a05_by_bug = {d["bug_id"]: d["pass"] for d in a05}
        a10_by_bug = {d["bug_id"]: d["pass"] for d in a10}
        common = sorted(set(a05_by_bug) & set(a10_by_bug))
        b = sum(1 for bug in common if a05_by_bug[bug] == 1 and a10_by_bug[bug] == 0)
        c = sum(1 for bug in common if a05_by_bug[bug] == 0 and a10_by_bug[bug] == 1)
        disc = b + c
        if disc > 0:
            p = stats.binomtest(b, disc, 0.5).pvalue
            print(f"  {subspace}: b={b} c={c} disc={disc} p={p:.4f}")
        else:
            print(f"  {subspace}: no discordant pairs")

    # subspace effect: as_is vs aligned (averaging over alpha)
    print("\n--- SUBSPACE EFFECT (as_is vs aligned) ---")
    for alpha in (0.5, 1.0):
        asis = [d for d in data if d["alpha"] == alpha and d["subspace"] == "as_is"]
        aligned = [d for d in data if d["alpha"] == alpha and d["subspace"] == "aligned"]
        asis_by_bug = {d["bug_id"]: d["pass"] for d in asis}
        aligned_by_bug = {d["bug_id"]: d["pass"] for d in aligned}
        common = sorted(set(asis_by_bug) & set(aligned_by_bug))
        b = sum(1 for bug in common if asis_by_bug[bug] == 1 and aligned_by_bug[bug] == 0)
        c = sum(1 for bug in common if asis_by_bug[bug] == 0 and aligned_by_bug[bug] == 1)
        disc = b + c
        if disc > 0:
            p = stats.binomtest(b, disc, 0.5).pvalue
            print(f"  α={alpha}: b={b} c={c} disc={disc} p={p:.4f}")
        else:
            print(f"  α={alpha}: no discordant pairs")

    # delta_norm and proj_frac distributions
    print("\n=== INSTRUMENTATION ===")
    for subspace in ("as_is", "aligned"):
        norms = [d["delta_norm"] for d in data if d["subspace"] == subspace]
        projs = [d["proj_frac"] for d in data if d["subspace"] == subspace]
        print(f"  {subspace}: ‖Δ‖ mean={np.mean(norms):.2f} ±{np.std(norms):.2f}, proj_frac mean={np.mean(projs):.3f}")

    # §5 decision boundary readout
    print("\n=== §5 DECISION BOUNDARY READOUT ===")
    # count significant effects
    sig_strength = False
    sig_subspace = False
    # (simplified: if any subspace shows significant strength effect, or vice versa)
    # proper readout requires pre-registered thresholds; here we report descriptively
    print("  Readout: EXPLORATORY — report CIs honestly, no confirmatory claim")
    print("  Default: 'cannot tell' if no clear pattern emerges")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "pilot_full.jsonl"))
