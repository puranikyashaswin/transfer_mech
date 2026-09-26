#!/usr/bin/env python3
"""Paired statistics for Amendment 008 confirmatory run (layer 17, n=215 test).

Headline family (Amendment 008 §7): ridge_map vs adapter at layer 17.
All contrasts within this family use Holm correction.

random_norm is paired per-bug via per_bug_pass_frac (majority vote, matching
Amendment 006's reported rule: pass if >=3/5 draws pass).

Layer-independent arms (receiver_k1, text_hint, donor_k1) were NOT rerun at
layer 17 — they are bit-identical across layers (no hook_layer arg) and are
reused from Amendment 006 only for descriptive context, not for the L17 family.
"""
import json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval_out.jsonl"
RIDGE = HERE / "ridge_out.jsonl"

# ---------- load ----------
def load_eval():
    by_arm = {}
    with open(EVAL) as f:
        for line in f:
            r = json.loads(line)
            by_arm.setdefault(r["arm"], {})[r["bug_id"]] = r
    return by_arm

def load_ridge():
    rows = {}
    with open(RIDGE) as f:
        for line in f:
            r = json.loads(line)
            rows[r["bug_id"]] = r
    return rows

def passed(r):
    return 1 if r.get("status") == "pass" else 0

def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)

def mcnemar_exact(b, c):
    """Exact two-sided binomial p for McNemar discordants b, c."""
    n = b + c
    if n == 0:
        return 1.0, 0.0
    k = min(b, c)
    # two-sided exact: 2 * P(X <= k), X ~ Bin(n, 0.5), capped at 1
    p = 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(p, 1.0), n

def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

def fmt_p(p):
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"

# ---------- main ----------
ev = load_eval()
rg = load_ridge()

bugs = sorted(next(iter(ev.values())).keys())
assert len(bugs) == 215, f"expected 215 bugs, got {len(bugs)}"

# Build per-bug binary vectors
vec = {}
for arm, rows in ev.items():
    if arm == "random_norm":
        # majority over the 5 draws
        # rows here are individual draw rows; group by bug
        per_bug = {}
        with open(EVAL) as f:
            for line in f:
                r = json.loads(line)
                if r["arm"] == "random_norm":
                    per_bug.setdefault(r["bug_id"], []).append(passed(r))
        vec["random_norm"] = {b: (1 if sum(v) >= 3 else 0) for b, v in per_bug.items()}
    else:
        vec[arm] = {b: passed(r) for b, r in rows.items()}

# ridge: alpha key check
r0 = next(iter(rg.values()))
print("ridge row keys:", sorted(r0.keys()))
rvec = {b: passed(r) for b, r in rg.items()}
vec["ridge_map"] = rvec

# ---------- per-arm table ----------
print("\n=== Per-arm accuracy (layer 17, n=215 test) ===")
print(f"{'arm':<16} {'k/n':<10} {'acc':<7} {'Wilson 95% CI':<20}")
arms_order = ["adapter", "ridge_map", "random_norm", "receiver_k1",
              "text_hint", "donor_k1", "shuffled", "self_inject", "unrelated_task"]
accs = {}
for arm in arms_order:
    v = vec[arm]
    k = sum(v.values())
    n = len(v)
    accs[arm] = k / n
    lo, hi = wilson(k, n)
    print(f"{arm:<16} {k:>3}/{n:<6} {k/n:.3f}   [{lo:.3f}, {hi:.3f}]")

# ---------- paired contrasts ----------
def paired_diff_ci(a, b, bugs_list, z=1.959964):
    """CI for mean of paired differences (a - b) per bug."""
    va, vb = vec[a], vec[b]
    n = len(bugs_list)
    d = [va[bug] - vb[bug] for bug in bugs_list]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return mean - z * se, mean + z * se

def compare(a, b):
    """McNemar between arms a and b. Returns dict."""
    va, vb = vec[a], vec[b]
    b10 = sum(1 for bug in bugs if va[bug] == 1 and vb[bug] == 0)  # a pass, b fail
    b01 = sum(1 for bug in bugs if va[bug] == 0 and vb[bug] == 1)  # a fail, b pass
    p, ndisc = mcnemar_exact(b10, b01)
    h = cohens_h(accs[a], accs[b])
    ci_lo, ci_hi = paired_diff_ci(a, b, bugs)
    return {"a": a, "b": b, "diff": accs[a] - accs[b],
            "b10": b10, "b01": b01, "n_disc": ndisc,
            "p_exact": p, "cohens_h": h,
            "ci_lo": ci_lo, "ci_hi": ci_hi}

# Amendment 008 §7 headline family: ridge vs adapter is the primary contrast.
# Secondary contrasts within this family (Holm-corrected together):
contrasts = [
    ("ridge_map", "adapter"),       # §7 headline
    ("adapter", "receiver_k1"),     # does adapter beat k1 baseline?
    ("adapter", "random_norm"),     # does adapter beat norm-matched noise?
    ("ridge_map", "receiver_k1"),
    ("ridge_map", "random_norm"),
    ("adapter", "shuffled"),
    ("adapter", "self_inject"),
    ("adapter", "unrelated_task"),
]

results = [compare(a, b) for a, b in contrasts]

# Holm correction across the family
results.sort(key=lambda r: r["p_exact"])
m = len(results)
for i, r in enumerate(results):
    r["p_holm"] = min(1.0, r["p_exact"] * (m - i))

print("\n=== Paired contrasts (McNemar exact, Holm within Amendment 008 family) ===")
print(f"{'contrast':<28} {'Δacc':<7} {'95% CI':<18} {'a+/b-':<6} {'a-/b+':<6} {'p_exact':<10} {'p_holm':<10} {'h':<6}")
for r in results:
    lbl = f"{r['a']} vs {r['b']}"
    ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
    print(f"{lbl:<28} {r['diff']:+.3f}  {ci:<18} {r['b10']:<6} {r['b01']:<6} {fmt_p(r['p_exact']):<10} {fmt_p(r['p_holm']):<10} {r['cohens_h']:+.3f}")

# ---------- failure-mode profile for adapter ----------
print("\n=== Adapter arm status distribution (L17) ===")
from collections import Counter
c = Counter(r["status"] for r in ev["adapter"].values())
for s, n in c.most_common():
    print(f"  {s:<15} {n:>3} ({n/215*100:.1f}%)")

print("\n=== Amendment 006 L12 adapter reference: 111/215=0.516, compile-fail 22.3% ===")
print(f"=== Amendment 008 L17 adapter:            {sum(vec['adapter'].values())}/215={accs['adapter']:.3f} ===")
