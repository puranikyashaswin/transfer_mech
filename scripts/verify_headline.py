#!/usr/bin/env python3
"""Recompute the paper's headline numbers from the raw execution logs.

Standard library only, CPU only, runs in about a second. Every number in the
README's results table is checked here against the per-bug JSONL logs that
the GPU runs wrote; nothing is read from a summary file.

  python3 scripts/verify_headline.py          # print the table, exit 1 on any mismatch

Confirmatory run (Amendment 006, receiver layer 12, results/confirmatory/):
  adapter vs receiver_k1, exact McNemar, paired effect with the scaled
  Clopper-Pearson CI used in paper/make_stats_confirmatory.py.
Corrected-site re-run (Amendment 008, receiver layer 17, results/confirmatory_L17/):
  ridge map vs trained adapter, exact McNemar, Wald CI on paired differences
  as in results/confirmatory_L17/stats_L17.py.
Dev split (exploratory, results/pilot/): ridge map vs trained adapter, as in
  results/pilot/analyze_baseline.py.
Exploratory HumanEval-Java control set (Amendment 003, results/rigorous/):
  receiver baseline vs text hint vs injected transfer, as in paper/make_stats.py.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIRMATORY = ROOT / "results" / "confirmatory" / "eval_out.jsonl"
L17_EVAL = ROOT / "results" / "confirmatory_L17" / "eval_out.jsonl"
L17_RIDGE = ROOT / "results" / "confirmatory_L17" / "ridge_out.jsonl"
DEV_RIDGE = ROOT / "results" / "pilot" / "baseline_ridge_dev.jsonl"
DEV_ADAPTER = ROOT / "results" / "pilot" / "pilot_full.jsonl"
RIGOROUS = ROOT / "results" / "rigorous" / "rigorous_checkpoint.jsonl"

SINGLE_ARMS = ["receiver_k1", "adapter", "shuffled", "self_inject",
               "unrelated_task", "text_hint", "donor_k1"]
N_DRAWS = 5


# --- statistics (stdlib) -------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: 2 * P(X <= min(b, c)), X ~ Bin(b + c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(f, lo=0.0, hi=1.0, iters=100) -> float:
    """Root of a function that is increasing on [lo, hi]."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial CI, the same interval as scipy beta.ppf."""
    lo = 0.0 if k == 0 else _bisect(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    hi = 1.0 if k == n else _bisect(lambda p: alpha / 2 - _binom_cdf(k, n, p))
    return lo, hi


def paired_effect_cp(b: int, c: int, n: int) -> tuple[float, float, float]:
    """Paired effect (b - c) / n with the CI from make_stats_confirmatory.py:
    the exact CI on the win-share b / (b + c), rescaled to the full-n scale."""
    k = b + c
    lo, hi = clopper_pearson(b, k)
    scale = k / n
    return (b - c) / n, (2 * lo - 1) * scale, (2 * hi - 1) * scale


def paired_effect_wald(a: list[int], b: list[int], z: float = 1.959964) -> tuple[float, float, float]:
    """Mean paired difference with a normal-approximation CI, as in stats_L17.py."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n)
    return mean, mean - z * se, mean + z * se


def discordant(a: dict[str, bool], b: dict[str, bool], bugs: list[str]) -> tuple[int, int]:
    """(a pass & b fail, a fail & b pass) over the shared bug list."""
    return (sum(a[x] and not b[x] for x in bugs),
            sum(b[x] and not a[x] for x in bugs))


# --- loading -------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def confirmatory_outcomes(rows: list[dict]):
    """Per-bug pass/fail for complete bugs, following make_stats_confirmatory.py:
    a bug is complete iff every single-shot arm has a row and random_norm has all
    5 draws. Only status == "pass" counts as success."""
    singles: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    draws: dict[str, set] = defaultdict(set)
    status: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        if r["arm"] == "random_norm":
            draws[r["bug_id"]].add(r.get("draw", r.get("seed")))
        else:
            singles[r["bug_id"]][r["arm"]].append(r["status"] == "pass")
            status[r["arm"]][r["bug_id"]] = r["status"]
    complete = sorted(b for b in singles
                      if all(singles[b][a] for a in SINGLE_ARMS) and len(draws[b]) == N_DRAWS)
    passes = {a: {b: any(singles[b][a]) for b in complete} for a in SINGLE_ARMS}
    return complete, passes, status


def l17_outcomes(eval_rows: list[dict], ridge_rows: list[dict]):
    passes: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in eval_rows:
        if r["arm"] != "random_norm":
            passes[r["arm"]][r["bug_id"]] = r["status"] == "pass"
    for r in ridge_rows:
        passes["ridge_map"][r["bug_id"]] = r["status"] == "pass"
    bugs = sorted(passes["ridge_map"])
    return bugs, passes


# --- headline ------------------------------------------------------------------

def headline() -> dict[str, float]:
    """Every number the README reports, recomputed from the raw logs."""
    out: dict[str, float] = {}

    bugs, p, status = confirmatory_outcomes(load_jsonl(CONFIRMATORY))
    n = len(bugs)
    b, c = discordant(p["adapter"], p["receiver_k1"], bugs)
    eff, lo, hi = paired_effect_cp(b, c, n)
    out.update({
        "n_test": n,
        "receiver_pass": sum(p["receiver_k1"].values()),
        "adapter_pass": sum(p["adapter"].values()),
        "adapter_vs_receiver_b": b,
        "adapter_vs_receiver_c": c,
        "adapter_vs_receiver_p": mcnemar_exact(b, c),
        "adapter_vs_receiver_effect_pp": 100 * eff,
        "adapter_vs_receiver_ci_lo_pp": 100 * lo,
        "adapter_vs_receiver_ci_hi_pp": 100 * hi,
        "adapter_compile_error_pct": 100 * sum(s == "compile_error" for s in status["adapter"].values()) / n,
        "receiver_compile_error_pct": 100 * sum(s == "compile_error" for s in status["receiver_k1"].values()) / n,
        "text_hint_pass": sum(p["text_hint"].values()),
    })

    bugs17, p17 = l17_outcomes(load_jsonl(L17_EVAL), load_jsonl(L17_RIDGE))
    b, c = discordant(p17["ridge_map"], p17["adapter"], bugs17)
    eff, lo, hi = paired_effect_wald([int(p17["ridge_map"][x]) for x in bugs17],
                                     [int(p17["adapter"][x]) for x in bugs17])
    out.update({
        "l17_n": len(bugs17),
        "l17_ridge_acc": sum(p17["ridge_map"].values()) / len(bugs17),
        "l17_adapter_acc": sum(p17["adapter"][x] for x in bugs17) / len(bugs17),
        "l17_receiver_acc": sum(p17["receiver_k1"][x] for x in bugs17) / len(bugs17),
        "l17_ridge_vs_adapter_p": mcnemar_exact(b, c),
        "l17_ridge_vs_adapter_ci_lo_pp": 100 * lo,
        "l17_ridge_vs_adapter_ci_hi_pp": 100 * hi,
    })

    ridge = {r["bug_id"]: r["status"] == "pass" for r in load_jsonl(DEV_RIDGE) if r["alpha"] == 1.0}
    adapter = {r["bug_id"]: r["status"] == "pass" for r in load_jsonl(DEV_ADAPTER)
               if r["alpha"] == 1.0 and r["subspace"] == "as_is"}
    dev = sorted(ridge)
    b, c = discordant(ridge, adapter, dev)
    out.update({
        "dev_n": len(dev),
        "dev_ridge_pass": sum(ridge.values()),
        "dev_ridge_wins": b,
        "dev_adapter_wins": c,
        "dev_ridge_vs_adapter_p": mcnemar_exact(b, c),
    })

    rig: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in load_jsonl(RIGOROUS):
        if r["arm"] != "random_norm":
            rig[r["arm"]][r["bug_id"]] = bool(r["passed"])
    bugs_r = sorted(rig["receiver_k1"])
    b, c = discordant(rig["text_hint"], rig["transfer"], bugs_r)
    out.update({
        "rig_n": len(bugs_r),
        "rig_receiver_pct": 100 * sum(rig["receiver_k1"].values()) / len(bugs_r),
        "rig_text_hint_pct": 100 * sum(rig["text_hint"].values()) / len(bugs_r),
        "rig_text_hint_vs_transfer_p": mcnemar_exact(b, c),
    })
    return out


# Values as reported in paper/PAPER.md (abstract, §7, §12.4, §12.7, §12.8), with
# the rounding the paper uses.
EXPECTED: dict[str, tuple[float, int]] = {
    "n_test": (215, 0),
    "receiver_pass": (137, 0),
    "adapter_pass": (111, 0),
    "adapter_vs_receiver_b": (18, 0),
    "adapter_vs_receiver_c": (44, 0),
    "adapter_vs_receiver_p": (0.0013, 4),
    "adapter_vs_receiver_effect_pp": (-12.1, 1),
    "adapter_vs_receiver_ci_lo_pp": (-18.3, 1),
    "adapter_vs_receiver_ci_hi_pp": (-4.6, 1),
    "adapter_compile_error_pct": (22.3, 1),
    "receiver_compile_error_pct": (2.3, 1),
    "text_hint_pass": (135, 0),
    "l17_n": (215, 0),
    "l17_ridge_acc": (0.605, 3),
    "l17_adapter_acc": (0.595, 3),
    "l17_receiver_acc": (0.633, 3),
    "l17_ridge_vs_adapter_p": (0.890, 3),
    "l17_ridge_vs_adapter_ci_lo_pp": (-5.7, 1),
    "l17_ridge_vs_adapter_ci_hi_pp": (7.5, 1),
    "dev_n": (40, 0),
    "dev_ridge_pass": (26, 0),
    "dev_ridge_wins": (13, 0),
    "dev_adapter_wins": (1, 0),
    "dev_ridge_vs_adapter_p": (0.0018, 4),
    "rig_n": (73, 0),
    "rig_receiver_pct": (26.0, 1),
    "rig_text_hint_pct": (47.9, 1),
    "rig_text_hint_vs_transfer_p": (0.023, 3),
}


def check(got: dict[str, float]) -> list[str]:
    """Names of every headline number that does not match the paper at its precision."""
    return [k for k, (want, nd) in EXPECTED.items() if round(got[k], nd) != round(want, nd)]


def main() -> int:
    got = headline()
    bad = check(got)
    width = max(map(len, EXPECTED))
    for k, (want, nd) in EXPECTED.items():
        mark = "FAIL" if k in bad else "ok"
        print(f"{k:<{width}}  paper {want:>9}  logs {round(got[k], nd):>9}  {mark}")
    print(f"\n{len(EXPECTED) - len(bad)}/{len(EXPECTED)} headline numbers reproduced from raw logs.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
