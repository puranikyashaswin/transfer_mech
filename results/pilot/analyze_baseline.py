#!/usr/bin/env python3
"""
analyze_baseline.py — compare baseline ridge-map arm vs adapter vs receiver on dev.

Answers: adapter-architecture-specific failure, or general injection failure?
Readout cells (all dev, n=40 each):
  receiver_k1 (a=0)        from pilot_baseline.jsonl
  adapter a=1.0 as_is      from pilot_full.jsonl
  adapter a=0.5 as_is      from pilot_full.jsonl
  adapter a=1.0 aligned    from pilot_full.jsonl
  adapter a=0.5 aligned    from pilot_full.jsonl
  baseline_ridge a=1.0     from baseline_ridge_dev.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy import stats


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def passrate(rows):
    n = len(rows)
    p = sum(1 for r in rows if r["status"] == "pass")
    ce = sum(1 for r in rows if r["status"] == "compile_error")
    return p, n, ce


def mcnemar(rows_a, rows_b, key="bug_id"):
    a = {r[key]: r["status"] == "pass" for r in rows_a}
    b = {r[key]: r["status"] == "pass" for r in rows_b}
    common = sorted(set(a) & set(b))
    bw = sum(1 for k in common if a[k] and not b[k])
    cw = sum(1 for k in common if not a[k] and b[k])
    disc = bw + cw
    p = stats.binomtest(bw, disc, 0.5).pvalue if disc > 0 else 1.0
    return bw, cw, disc, p


def main():
    here = Path(__file__).resolve().parent
    pilot = load(here / "pilot_full.jsonl")
    base = load(here / "pilot_baseline.jsonl")
    ridge = load(here / "baseline_ridge_dev.jsonl")

    recv = [r for r in base if r["subspace"] == "as_is"]

    cells = {
        "receiver_k1 (a=0)": recv,
        "adapter a=1.0 as_is": [r for r in pilot if r["alpha"] == 1.0 and r["subspace"] == "as_is"],
        "adapter a=0.5 as_is": [r for r in pilot if r["alpha"] == 0.5 and r["subspace"] == "as_is"],
        "adapter a=1.0 aligned": [r for r in pilot if r["alpha"] == 1.0 and r["subspace"] == "aligned"],
        "adapter a=0.5 aligned": [r for r in pilot if r["alpha"] == 0.5 and r["subspace"] == "aligned"],
        "baseline_ridge a=1.0": [r for r in ridge if r["alpha"] == 1.0],
    }

    print("=== PASS RATES (dev, n=40/cell) ===")
    for name, rows in cells.items():
        p, n, ce = passrate(rows)
        print(f"  {name:<22}: pass={p}/{n}={p/n:.3f}  compile_err={ce}/{n}={ce/n:.3f}")

    print("\n=== McNEMAR vs receiver_k1 (dev) ===")
    for name, rows in cells.items():
        if name.startswith("receiver"):
            continue
        b, c, disc, p = mcnemar(rows, recv)
        ws = b / disc if disc else float("nan")
        print(f"  {name:<22}: wins={b} loses={c} disc={disc} winshare={ws:.3f} p={p:.4f}")

    print("\n=== KEY QUESTION: ridge vs adapter (a=1.0, as_is) ===")
    b, c, disc, p = mcnemar(cells["baseline_ridge a=1.0"], cells["adapter a=1.0 as_is"])
    ws = b / disc if disc else float("nan")
    print(f"  ridge wins={b} adapter wins={c} disc={disc} ridge winshare={ws:.3f} p={p:.4f}")
    print("  -> if ridge >> adapter: failure is adapter-architecture-specific")
    print("  -> if ridge ~= adapter (both <= baseline): failure is general to injection")

    print("\n=== delta norms ===")
    import statistics
    for name, rows in cells.items():
        norms = [r["delta_norm"] for r in rows if r.get("delta_norm") is not None and r["delta_norm"] > 0]
        if norms:
            print(f"  {name:<22}: mean |d| = {statistics.mean(norms):.2f} (n={len(norms)})")


if __name__ == "__main__":
    main()
