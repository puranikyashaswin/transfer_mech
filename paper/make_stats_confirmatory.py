#!/usr/bin/env python3
"""Confirmatory one-shot evaluation statistics for the transfer_mech paper.

Amendment 006 §4/§5: analyzes the 8-arm ONE-SHOT confirmatory eval output
(results/confirmatory/eval_out.jsonl). Arms (§4): receiver_k1, adapter,
random_norm (5 draws per bug), shuffled, self_inject, unrelated_task,
text_hint, donor_k1. §5 pre-registers: paired exact McNemar of adapter vs
receiver_k1 and adapter vs random_norm (majority of 5 draws), win-share
b/(b+c) with exact 95% binomial CI on the discordant pairs (headline metric,
pre-registered claim threshold 60%), and an HONEST-n / equivalence-bound
pre-commitment: because achievable n < the powered n (859-1411 per the §1
power calc), NO headline effect is claimed — the data are reported as a
boundary result with the CI on the paired effect.

Only COMPLETE bugs are analyzed: a bug is complete iff each of the 7
single-shot arms has >=1 row AND random_norm has all 5 draws (0..4).

Run with ~/.venvs/doctools/bin/python (needs scipy). Defensive: if the
eval_out.jsonl is missing or empty this prints NOT RUN YET and exits 0 so
the script can be committed before the confirmatory run lands.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest, beta

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "confirmatory" / "eval_out.jsonl"

SINGLE_ARMS = [
    "receiver_k1",
    "adapter",
    "shuffled",
    "self_inject",
    "unrelated_task",
    "text_hint",
    "donor_k1",
]
RANDOM_ARM = "random_norm"
N_DRAWS = 5
MAJORITY = 3  # >=3 of 5 draws passing collapses random_norm to a per-bug pass
CLAIM_THRESHOLD = 0.60  # pre-registered win-share claim threshold (§5)
POWERED_N = 859  # lower end of the §1 power-calc required-n range (859-1411)


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (p, c - h, c + h)


def winshare_ci(b, c):
    """Exact two-sided 95% binomial CI on the win-share p = b/(b+c)
    (Clopper-Pearson via the beta distribution)."""
    n = b + c
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = b / n
    lo = beta.ppf(0.025, b, c + 1) if b > 0 else 0.0
    hi = beta.ppf(0.975, b + 1, c) if c > 0 else 1.0
    return (p, lo, hi)


def paired_mcnemar(passes, arm_a, arm_b):
    """Exact McNemar on per-bug pass/fail. b: A-pass/B-fail; c: A-fail/B-pass."""
    b = c = 0
    for bug, pa in passes.items():
        a, bb = pa[arm_a], pa[arm_b]
        if a and not bb:
            b += 1
        elif not a and bb:
            c += 1
    n = b + c
    p = binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
    return b, c, p


def main():
    rows = load(EVAL)
    if not rows:
        print("=" * 70)
        print("CONFIRMATORY EVAL (Amendment 006): NOT RUN YET")
        print("=" * 70)
        print(f"Expected input: {EVAL}")
        print("File missing or empty — commit this script now, run after the eval lands.")
        return 0

    # ---- assemble per-bug per-arm outcomes --------------------------------
    # passes[bug][arm] -> bool for single-shot arms; draws[bug][draw] -> bool
    # for random_norm. Only "pass" status counts as success; every other
    # status (test_fail, compile_error, runtime_error, timeout, error) fails.
    raw = defaultdict(dict)          # bug -> arm -> list of pass bools (single arms)
    draws = defaultdict(dict)        # bug -> draw -> pass bool (random_norm)
    for r in rows:
        bug, arm = r["bug_id"], r["arm"]
        ok = r.get("status") == "pass"
        if arm == RANDOM_ARM:
            draws[bug][int(r["draw"])] = ok
        else:
            raw[bug].setdefault(arm, []).append(ok)

    all_bugs = sorted(set(raw) | set(draws))
    complete, dropped = [], []
    for bug in all_bugs:
        singles_ok = all(len(raw[bug].get(a, [])) >= 1 for a in SINGLE_ARMS)
        draws_ok = len(draws.get(bug, {})) == N_DRAWS
        (complete if (singles_ok and draws_ok) else dropped).append(bug)

    # Per-bug pass/fail for complete bugs. Single-shot arms: pass if ANY row
    # passes (spec: >=1 row required; duplicates should not exist in one-shot,
    # but collapse defensively). random_norm: collapse by MAJORITY of 5 draws.
    passes = {}
    for bug in complete:
        pa = {a: any(raw[bug][a]) for a in SINGLE_ARMS}
        pa[RANDOM_ARM] = sum(draws[bug].values()) >= MAJORITY
        passes[bug] = pa

    n = len(complete)

    # ======================================================================
    print("=" * 70)
    print("CONFIRMATORY ONE-SHOT EVAL (Amendment 006 §4/§5)")
    print("=" * 70)
    print(f"input: {EVAL}  ({len(rows)} rows)")

    # ---- (1) completeness --------------------------------------------------
    print()
    print("-" * 70)
    print("(1) COMPLETE BUGS (all 7 single-shot arms >=1 row AND random_norm 5 draws)")
    print("-" * 70)
    print(f"n_complete = {n}   n_dropped = {len(dropped)}")
    if dropped:
        print(f"dropped bug_ids: {dropped}")
    if n == 0:
        print("No complete bugs — nothing to analyze.")
        return 0

    # ---- (2) per-arm pass rates -------------------------------------------
    print()
    print("-" * 70)
    print("(2) PER-ARM PASS RATES (95% Wilson CI)")
    print("-" * 70)
    for arm in SINGLE_ARMS:
        k = sum(passes[b][arm] for b in complete)
        p, lo, hi = wilson(k, n)
        print(f"  {arm:15s} {k:3d}/{n} = {p:.3f}  CI[{lo:.3f},{hi:.3f}]")
    # random_norm: mean over bugs of per-bug fraction of 5 draws passing
    per_bug_frac = [sum(draws[b].values()) / N_DRAWS for b in complete]
    mean_frac = sum(per_bug_frac) / n
    per_draw = [sum(draws[b].get(d, False) for b in complete) for d in range(N_DRAWS)]
    spread = (max(per_draw) - min(per_draw)) / n
    print(f"  {RANDOM_ARM:15s} mean pass fraction {mean_frac:.3f}  "
          f"per-draw pass counts {per_draw} (of {n})  spread {spread:.3f}")

    # ---- (3) PRIMARY PRE-REGISTERED TEST: paired McNemar ------------------
    print()
    print("-" * 70)
    print("(3) PRIMARY PRE-REGISTERED TEST (§5): paired exact McNemar")
    print("    adapter vs receiver_k1  and  adapter vs random_norm(majority>=3/5)")
    print("-" * 70)
    mcnemar = {}
    for ctrl in ["receiver_k1", RANDOM_ARM]:
        b, c, p = paired_mcnemar(passes, "adapter", ctrl)
        mcnemar[ctrl] = (b, c, p)
        label = f"adapter vs {ctrl}" + (" (majority)" if ctrl == RANDOM_ARM else "")
        print(f"  {label}:")
        print(f"    b (adapter-pass / control-fail) = {b}")
        print(f"    c (adapter-fail / control-pass) = {c}")
        print(f"    discordant = {b + c}   exact McNemar p = {p:.4f}"
              f"{'  <-- SIGNIFICANT (p<0.05)' if p < 0.05 else ''}")

    # ---- (4) WIN-SHARE on discordant pairs (headline metric) --------------
    print()
    print("-" * 70)
    print("(4) WIN-SHARE on discordant pairs (pre-registered headline metric)")
    print(f"    win-share = b/(b+c), exact 95% binomial CI; claim threshold = {CLAIM_THRESHOLD:.2f}")
    print("-" * 70)
    for ctrl in ["receiver_k1", RANDOM_ARM]:
        b, c, _ = mcnemar[ctrl]
        p, lo, hi = winshare_ci(b, c)
        label = f"adapter vs {ctrl}" + (" (majority)" if ctrl == RANDOM_ARM else "")
        if b + c == 0:
            print(f"  {label}: no discordant pairs — win-share undefined")
            continue
        relation = ("CLEARS" if lo > CLAIM_THRESHOLD else
                    "EXCLUDES-below" if hi < CLAIM_THRESHOLD else "INCLUDES")
        verdict = ("CI clears 0.60 -> claim threshold met" if lo > CLAIM_THRESHOLD else
                   "CI entirely below 0.60 -> adapter UNDERPERFORMS control" if hi < CLAIM_THRESHOLD else
                   "CI includes 0.60 -> claim threshold NOT established")
        print(f"  {label}: win-share = {b}/{b + c} = {p:.3f}  "
              f"exact 95% CI [{lo:.3f},{hi:.3f}]  -> {relation} 0.60; {verdict}")

    # ---- (5) HONEST-n / EQUIVALENCE BOUND (§5 pre-commitment, verbatim) ---
    print()
    print("-" * 70)
    print("(5) HONEST-n / EQUIVALENCE BOUND (§5 PRE-COMMITMENT — no headline claim)")
    print("-" * 70)
    print(f"  achieved n = {n} complete bugs; pre-registered powered n = {POWERED_N}-1411.")
    for ctrl in ["receiver_k1", RANDOM_ARM]:
        b, c, _ = mcnemar[ctrl]
        k = b + c
        # Paired effect on the full-n scale: (pass diff) = (b - c)/n, with the
        # CI obtained by scaling the exact CI on the conditional discordance
        # probability p_plus = b/(b+c) to the full-n scale via k/n
        # (same construction as make_stats.py Claim 1).
        diff = 100 * (b - c) / n
        if k:
            p_plus, lo, hi = winshare_ci(b, c)
            scale = k / n
            d_lo = (2 * lo - 1) * scale * 100
            d_hi = (2 * hi - 1) * scale * 100
        else:
            d_lo = d_hi = 0.0
        label = f"adapter - {ctrl}" + (" (majority)" if ctrl == RANDOM_ARM else "")
        print(f"  {label}: paired effect = {diff:+.1f}pp  95% CI [{d_lo:+.1f}pp, {d_hi:+.1f}pp]")
        print(f"    -> the data are consistent with an effect in [{d_lo:+.1f}pp, {d_hi:+.1f}pp];")
        print(f"       the pre-registered powered claim (n>={POWERED_N}) could NOT be tested")
        print(f"       at n={n}; this is a boundary result.")
        adapter_rate = sum(passes[x]["adapter"] for x in complete) / n
        ctrl_rate = sum(passes[x][ctrl] for x in complete) / n
        if adapter_rate < ctrl_rate:
            print(f"    -> adapter pass rate ({adapter_rate:.3f}) is BELOW {ctrl} "
                  f"({ctrl_rate:.3f}): clean negative/boundary result.")
    # Explicit underperformance check against every control arm.
    adapter_rate = sum(passes[x]["adapter"] for x in complete) / n
    for ctrl in SINGLE_ARMS + [RANDOM_ARM]:
        if ctrl == "adapter":
            continue
        ctrl_rate = sum(passes[x][ctrl] for x in complete) / n
        if adapter_rate < ctrl_rate:
            print(f"  NOTE: adapter ({adapter_rate:.3f}) underperforms {ctrl} "
                  f"({ctrl_rate:.3f}) — clean negative/boundary result vs this control.")

    # ---- (6) discordant-restricted subset ---------------------------------
    print()
    print("-" * 70)
    print("(6) DISCORDANT-RESTRICTED SUBSET (donor_k1 passes AND receiver_k1 fails)")
    print("-" * 70)
    disc = [x for x in complete
            if passes[x]["donor_k1"] and not passes[x]["receiver_k1"]]
    nd = len(disc)
    print(f"n = {nd} bugs (of {n} complete)")
    if nd:
        for arm in ["adapter", "text_hint", "self_inject", "unrelated_task"]:
            k = sum(passes[x][arm] for x in disc)
            print(f"  {arm:22s} {k}/{nd} = {k / nd:.3f}")
        kr = sum(1 for x in disc if sum(draws[x].values()) >= MAJORITY)
        print(f"  {'random_norm (majority)':22s} {kr}/{nd} = {kr / nd:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
