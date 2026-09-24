#!/usr/bin/env python3
"""Statistical rigor for the paper's key claims. Paired tests on per-bug outcomes.
Run with ~/.venvs/doctools/bin/python (needs scipy)."""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    return [json.loads(l) for l in (ROOT / "results" / name).read_text().splitlines() if l.strip()]


def paired_mcnemar(bybug, armA, armB):
    """McNemar exact test: are armA and armB per-bug pass/fail different?"""
    b = c = 0  # b: A pass & B fail; c: A fail & B pass
    for bug, arms in bybug.items():
        if armA in arms and armB in arms:
            a, bb = arms[armA], arms[armB]
            if a == 1 and bb == 0:
                b += 1
            elif a == 0 and bb == 1:
                c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5).pvalue
    return b, c, p


def toby(rows):
    bybug = defaultdict(dict)
    for r in rows:
        bybug[r["bug_id"]][r["arm"]] = r["passed"]
    return bybug


print("=" * 70)
print("CLAIM 1: same-family transfer == random (the central tie)")
print("=" * 70)
bybug = toby(load("same_family/sf_checkpoint.jsonl"))
b, c, p = paired_mcnemar(bybug, "transfer", "random_norm")
print(f"transfer-vs-random: discordant pairs b={b} (transfer-only), c={c} (random-only)")
print(f"McNemar exact p = {p:.4f}  -> {'NOT significantly different' if p>0.05 else 'SIGNIFICANT'}")

b, c, p = paired_mcnemar(bybug, "transfer", "receiver_k1")
print(f"transfer-vs-noop:   b={b}, c={c}, McNemar p = {p:.4f}")
b, c, p = paired_mcnemar(bybug, "random_norm", "receiver_k1")
print(f"random-vs-noop:     b={b}, c={c}, McNemar p = {p:.4f}")

# Paired 95% CI on the content-vs-random discordance difference (transfer - random),
# matching the "+/-9pp" claim in the Discussion/STATS.md. Method: exact two-sided
# binomial CI on p_plus = b/(b+c) (the conditional discordance probability), then
# scale the difference to the full-n scale via the discordance rate k/n:
#   diff_on_n_scale = (2*p_plus - 1) * (k/n).   (b transfer-only, c random-only)
from scipy.stats import beta as _beta
n_sf = len(bybug)
_bc, _cc, _ = paired_mcnemar(bybug, "transfer", "random_norm")
_k = _bc + _cc
if _k:
    lo = _beta.ppf(0.025, _bc, _cc + 1)
    hi = _beta.ppf(0.975, _bc + 1, _cc)
    scale = _k / n_sf
    d_lo = (2 * lo - 1) * scale * 100
    d_hi = (2 * hi - 1) * scale * 100
    print(f"paired 95% CI on (transfer - random) pass-rate diff: [{d_lo:+.1f}pp, {d_hi:+.1f}pp]  (b={_bc}, c={_cc}, k={_k} discordant, n={n_sf})")

print()
print("=" * 70)
print("CLAIM 2: alpha sweep — does transfer beat random at ANY alpha?")
print("=" * 70)
rows = load("alpha_sweep/alpha_checkpoint.jsonl")
for a in [0.1, 0.5, 1.0]:
    bb = defaultdict(dict)
    for r in rows:
        if r["alpha"] == a:
            bb[r["bug_id"]][r["arm"]] = r["passed"]
    b_, c_, p_ = paired_mcnemar(bb, "transfer", "random_norm")
    print(f"alpha={a}: transfer-vs-random b={b_} c={c_}  McNemar p={p_:.4f}")

print()
print("=" * 70)
print("CLAIM 3: cross-family adapter — is adapter < random significant?")
print("=" * 70)
bybug = toby(load("adapter/eval_checkpoint.jsonl"))
for pair in [("adapter", "random_norm"), ("adapter", "receiver_k1"), ("adapter", "self_inject")]:
    b, c, p = paired_mcnemar(bybug, *pair)
    print(f"{pair[0]}-vs-{pair[1]}: b={b} c={c}  McNemar p={p:.4f}")

print()
print("=" * 70)
print("POWER: how big a difference could we have detected?")
print("=" * 70)
n_sf = len(bybug := toby(load("same_family/sf_checkpoint.jsonl")))
print(f"same-family n={n_sf} bugs; adapter n=43. With n~70, a paired difference of")
print("~10-12pp would be needed for p<0.05; observed differences are ~0pp (ties).")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (p, c - h, c + h)


print()
print("=" * 70)
print("CLAIM 4 (rigorous run, Amendment-003 controls): full control set + text-hint dissociation")
print("=" * 70)
rrows = load("rigorous/rigorous_checkpoint.jsonl")
rbybug = defaultdict(dict)
rrand = defaultdict(dict)
for r in rrows:
    if r["arm"] == "random_norm":
        rrand[r["bug_id"]][r["draw"]] = bool(r["passed"])
    else:
        rbybug[r["bug_id"]][r["arm"]] = bool(r["passed"])
rbugs = sorted(rbybug)
n = len(rbugs)
print(f"n={n} bugs. Pass rates (95% Wilson CI):")
for arm in ["receiver_k1", "donor_k1", "unrelated_task", "self_inject", "transfer", "text_hint"]:
    k = sum(rbybug[b].get(arm, False) for b in rbugs)
    p, lo, hi = wilson(k, n)
    print(f"  {arm:15s} {k:2d}/{n} = {p:.3f}  CI[{lo:.3f},{hi:.3f}]")
rk = [sum(rrand[b].get(d, False) for b in rbugs) for d in range(5)]
print(f"  random_norm     mean acc {sum(rk)/(5*n):.3f}  per-draw pass {rk}  spread {max(rk)/n - min(rk)/n:.3f}")


def rmcn(a1, a2):
    return paired_mcnemar(rbybug, a1, a2)


print("Paired McNemar (exact):")
for a1, a2 in [("transfer", "self_inject"), ("transfer", "unrelated_task"), ("transfer", "receiver_k1"), ("text_hint", "transfer")]:
    b, c, p = rmcn(a1, a2)
    sig = "  <-- SIGNIFICANT" if p < 0.05 else ""
    print(f"  {a1} vs {a2}: b={b} c={c}  p={p:.4f}{sig}")
# transfer vs random (majority of 5 draws)
bmaj = sum(1 for x in rbugs if rbybug[x].get("transfer") and sum(rrand[x].values()) < 3)
cmaj = sum(1 for x in rbugs if not rbybug[x].get("transfer") and sum(rrand[x].values()) >= 3)
nn = bmaj + cmaj
pmaj = binomtest(min(bmaj, cmaj), nn, 0.5).pvalue if nn else 1.0
print(f"  transfer vs random(majority): b={bmaj} c={cmaj}  p={pmaj:.4f}")
# discordant-restricted
disc = [x for x in rbugs if rbybug[x].get("donor_k1") and not rbybug[x].get("receiver_k1")]
print(f"Discordant-restricted (donor passes & receiver fails, n={len(disc)}):")
for arm in ["transfer", "text_hint", "self_inject", "unrelated_task"]:
    k = sum(rbybug[x].get(arm, False) for x in disc)
    print(f"  {arm:15s} {k}/{len(disc)}")
kr = sum(1 for x in disc if sum(rrand[x].values()) >= 3)
print(f"  random(majority) {kr}/{len(disc)}")

print()
print("=" * 70)
print("CLAIM 5 (site sweep, Amendment 004): 2x2 boundary, Holm-corrected")
print("=" * 70)
srows = load("site_sweep/site_checkpoint.jsonl")
sbybug = defaultdict(dict)
for r in srows:
    sbybug[r["bug_id"]][r["cell"] + "/" + r["arm"]] = bool(r["passed"])
sbugs = sorted(sbybug)
ns = len(sbugs)


def sacc(k):
    return sum(sbybug[x].get(k, False) for x in sbugs) / ns


def smcn(a, b):
    B = sum(1 for x in sbugs if sbybug[x].get(a) and not sbybug[x].get(b))
    C = sum(1 for x in sbugs if sbybug[x].get(b) and not sbybug[x].get(a))
    nn = B + C
    p = binomtest(min(B, C), nn, 0.5).pvalue if nn else 1.0
    return B, C, p


cells = ["s1_t1", "s1_tall", "sall_t1", "sall_tall"]
print(f"n={ns} bugs. Per-cell transfer-vs-random:")
pvals = []
for c in cells:
    ta, ra = sacc(c + "/transfer"), sacc(c + "/random")
    B, C, p = smcn(c + "/transfer", c + "/random")
    pvals.append((c, p, B, C, 100 * (ta - ra)))
    print(f"  {c:9s} transfer={ta:.3f} random={ra:.3f} effect={100*(ta-ra):+.1f}pp  b={B} c={C} p={p:.4f}")
print(f"  baseline receiver_k1 = {sacc('baseline/receiver_k1'):.3f}")
print("Holm correction (alpha=0.05, success = p<thr AND effect>=+3pp in transfer>random direction):")
m = len(pvals)
for rank, (c, p, B, C, eff) in enumerate(sorted(pvals, key=lambda t: t[1])):
    thr = 0.05 / (m - rank)
    pos = (p < thr) and (eff >= 3) and (B > C)
    nom = (p < 0.05) and (eff >= 3) and (B > C)
    verdict = "POSITIVE (survives Holm)" if pos else ("SUGGESTIVE (nominal p<0.05, fails Holm)" if nom else "not positive")
    print(f"  rank{rank+1} {c:9s} p={p:.4f} Holm thr={thr:.4f} effect={eff:+.1f}pp -> {verdict}")
