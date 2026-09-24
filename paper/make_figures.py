#!/usr/bin/env python3
"""Generate paper figures from raw logs. Run with ~/.venvs/doctools/bin/python."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    p = ROOT / "results" / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------- Figure 1: arm comparison across the three studies ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

# (a) cross-family adapter (43 bugs)
ax = axes[0]
rows = load("adapter/eval_checkpoint.jsonl")
arms = ["receiver_k1", "adapter", "random_norm", "shuffled", "self_inject"]
labels = ["no-op", "adapter", "random", "shuffled", "self"]
acc = []
for arm in arms:
    sub = [r for r in rows if r["arm"] == arm]
    acc.append(sum(r["passed"] for r in sub) / len(sub))
colors = ["#888", "#c0392b", "#2980b9", "#95a5a6", "#7f8c8d"]
ax.bar(labels, acc, color=colors)
ax.axhline(acc[0], ls="--", c="#888", lw=1)
ax.set_title("(a) Cross-family (Qwen→DeepSeek)\ntrained adapter, 43 test bugs")
ax.set_ylabel("pass rate (execution-verified)")
ax.tick_params(axis="x", rotation=30)
for i, v in enumerate(acc):
    ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)

# (b) same-family pilot (73 bugs)
ax = axes[1]
rows = load("same_family/sf_checkpoint.jsonl")
arms = ["receiver_k1", "transfer", "random_norm", "self_inject"]
labels = ["no-op", "transfer", "random", "self"]
acc = []
for arm in arms:
    sub = [r for r in rows if r["arm"] == arm]
    acc.append(sum(r["passed"] for r in sub) / len(sub))
colors = ["#888", "#27ae60", "#2980b9", "#7f8c8d"]
ax.bar(labels, acc, color=colors)
ax.axhline(acc[0], ls="--", c="#888", lw=1)
ax.set_title("(b) Same-family (Qwen→Qwen-3B)\nnorm-matched ridge α=1.0, 73 bugs")
ax.tick_params(axis="x", rotation=30)
for i, v in enumerate(acc):
    ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)

# (c) alpha sweep
ax = axes[2]
rows = load("alpha_sweep/alpha_checkpoint.jsonl")
alphas = [0.1, 0.5, 1.0]
baseline = 19 / 73
for arm, color, marker in [("transfer", "#27ae60", "o"), ("random_norm", "#2980b9", "s"),
                            ("self_inject", "#7f8c8d", "^")]:
    ys = []
    for a in alphas:
        sub = [r for r in rows if r["alpha"] == a and r["arm"] == arm]
        ys.append(sum(r["passed"] for r in sub) / len(sub))
    ax.plot(alphas, ys, marker=marker, color=color, label=arm.replace("_", " "))
ax.axhline(baseline, ls="--", c="#888", lw=1, label=f"no-op ({baseline:.2f})")
ax.set_xscale("log")
ax.set_xticks(alphas); ax.set_xticklabels([str(a) for a in alphas])
ax.set_xlabel("injection strength α")
ax.set_title("(c) Same-family α-sweep\n68–69 bugs per cell")
ax.legend(fontsize=8, frameon=False)

fig.suptitle("Single-vector residual transfer never beats a matched random perturbation", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig1_transfer_vs_controls.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig1_transfer_vs_controls.png", dpi=200, bbox_inches="tight")
print("wrote fig1")

# ---------- Figure 2: full control set (rigorous run, 73 bugs, α=1.0) ----------
# Final numbers recomputed from rigorous_checkpoint.jsonl (Amendment-003 controls).
rrows = load("rigorous/rigorous_checkpoint.jsonl")
rbybug = defaultdict(dict)
rrand = defaultdict(dict)
for r in rrows:
    if r["arm"] == "random_norm":
        rrand[r["bug_id"]][r["draw"]] = bool(r["passed"])
    else:
        rbybug[r["bug_id"]][r["arm"]] = bool(r["passed"])
rbugs = sorted(rbybug)


def raccs(arm):
    return sum(rbybug[b].get(arm, False) for b in rbugs) / len(rbugs)


fig, ax = plt.subplots(figsize=(8, 4.4))
order = ["receiver_k1", "donor_k1", "unrelated_task", "self_inject", "random_norm", "transfer", "text_hint"]
labels = ["receiver\nno-op", "donor\nself", "unrelated\ntask", "self\ninject", "random\n(R=5)", "transfer", "text\nhint"]
vals = []
for a in order:
    if a == "random_norm":
        vals.append(sum(rrand[b].get(d, False) for b in rbugs for d in range(5)) / (len(rbugs) * 5))
    else:
        vals.append(raccs(a))
colors = ["#888", "#95a5a6", "#b2bec3", "#7f8c8d", "#2980b9", "#27ae60", "#c0392b"]
bars = ax.bar(labels, vals, color=colors)
ax.axhline(vals[0], ls="--", c="#888", lw=1)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("pass rate (execution-verified)")
ax.set_title("Full control set (73 bugs): text-hint rescues the receiver; transfer ties random\n"
             "text-hint≫transfer McNemar p=0.023 · transfer vs self p=1.000 · transfer vs random p=0.774")
fig.tight_layout()
fig.savefig(FIG / "fig2_controls.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig2_controls.png", dpi=200, bbox_inches="tight")
print("wrote fig2 (controls);", {a: round(v, 3) for a, v in zip(order, vals)})

# ---------- Figure 5: 2x2 site sweep (the boundary) ----------
srows = load("site_sweep/site_checkpoint.jsonl")
sbybug = defaultdict(dict)
for r in srows:
    sbybug[r["bug_id"]][r["cell"] + "/" + r["arm"]] = bool(r["passed"])
sbugs = sorted(sbybug)


def sacc(k):
    return sum(sbybug[b].get(k, False) for b in sbugs) / len(sbugs)


cells = ["s1_t1", "s1_tall", "sall_t1", "sall_tall"]
clabels = ["s1_t1\n(1 layer,\n1 token)", "s1_tall\n(1 layer,\nall tok)", "sall_t1\n(all layers,\n1 token)", "sall_tall\n(all layers,\nall tok)"]
tr = [sacc(c + "/transfer") for c in cells]
ra = [sacc(c + "/random") for c in cells]
baseline = sacc("baseline/receiver_k1")

x = np.arange(len(cells))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.4))
b1 = ax.bar(x - w / 2, tr, w, label="transfer", color="#27ae60")
b2 = ax.bar(x + w / 2, ra, w, label="random (matched)", color="#2980b9")
ax.axhline(baseline, ls="--", c="#888", lw=1, label=f"receiver no-op ({baseline:.2f})")
for xi, v in zip(x - w / 2, tr):
    ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, ra):
    ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(clabels, fontsize=8)
ax.set_ylabel("pass rate (execution-verified)")
ax.set_title("2×2 site sweep (73 bugs): transfer trends above noise ONLY at sall_t1\n"
             "(all donor layers → 1 token, +15.1pp, nominal p=0.043, fails Holm); all-token injection collapses to 0")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fig5_site_sweep.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig5_site_sweep.png", dpi=200, bbox_inches="tight")
print("wrote fig5 (sweep);", {c: {"transfer": round(t, 3), "random": round(r, 3)} for c, t, r in zip(cells, tr, ra)})

# ---------- Figure 4: mechanistic probe ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
mech = json.loads((ROOT / "results/mech/mech_results.json").read_text())
arms = ["transfer", "self_inject", "random_norm"]
labels = ["transfer", "self", "random"]
colors = ["#27ae60", "#7f8c8d", "#2980b9"]

def agg(key):
    return [np.mean([r[key] for r in mech if r["arm"] == a]) for a in arms]

ax = axes[0]
ax.bar(labels, agg("cos"), color=colors)
ax.set_ylim(0.90, 0.98)
ax.set_title("(a) last-layer residual\ncosine to baseline")
for i, v in enumerate(agg("cos")):
    ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

ax = axes[1]
ax.bar(labels, agg("kl_base_to_inject"), color=colors)
ax.set_title("(b) KL(baseline ‖ injected)\nnext-token distribution")
for i, v in enumerate(agg("kl_base_to_inject")):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)

ax = axes[2]
ax.bar(labels, agg("argmax_changed"), color=colors)
ax.set_ylim(0, 0.7)
ax.set_title("(c) fraction argmax next-token\nchanged by injection")
for i, v in enumerate(agg("argmax_changed")):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

fig.suptitle("Mechanistic probe: transfer ≡ self-inject; random perturbs MORE yet ties downstream", y=1.04, fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig4_mech_probe.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig4_mech_probe.png", dpi=200, bbox_inches="tight")
print("wrote fig4")
print("ALL FIGURES DONE ->", FIG)
