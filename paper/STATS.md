# Statistical Analysis (paired, exact tests)

All comparisons are **paired** (same bugs under two arms) using the **exact McNemar test** on discordant pairs (bugs where the two arms disagree). `b` = arm-A-only passes, `c` = arm-B-only passes. Recomputed from raw JSONL logs by `make_stats.py`.

## The central tie (same-family, α=1.0, n=73)

| Comparison | b | c | McNemar p | Verdict |
|---|---|---|---|---|
| **transfer vs random-norm** | 5 | 5 | **1.000** | perfect tie |
| transfer vs no-op | 7 | 3 | 0.344 | n.s. |
| random-norm vs no-op | 8 | 4 | 0.388 | n.s. |

**Transfer and random are statistically indistinguishable** (5–5 split, p=1.0). Neither is significantly above baseline — but both trend in the same direction (transfer 7–3, random 8–4 vs no-op), consistent with a small shared perturbation effect our n=73 cannot resolve at α=0.05.

## Alpha sweep (transfer vs random, per α)

| α | b | c | p |
|---|---|---|---|
| 0.1 | 5 | 3 | 0.727 |
| 0.5 | 4 | 4 | 1.000 |
| 1.0 | 5 | 5 | 1.000 |

**No α separates transfer from random.**

## Cross-family adapter (n=43)

| Comparison | b | c | p | Verdict |
|---|---|---|---|---|
| **adapter vs random-norm** | 1 | 8 | **0.039** | adapter **significantly worse** |
| **adapter vs self-inject** | 1 | 8 | **0.039** | adapter **significantly worse** |
| adapter vs no-op | 2 | 7 | 0.180 | n.s. (trend) |

The trained adapter is **significantly worse than a matched random perturbation** (p=0.039) — the one statistically decisive result, and it points *against* content-transfer.

## Equivalence framing + confidence bound (the honest claim)

Because individual arm-vs-arm differences are underpowered at n≈43–73, we **do not** claim "each injection is no better than baseline." We claim the narrower, well-supported statement: **the donor's content confers no detectable advantage over a matched content-free perturbation of the same magnitude** (the paired transfer-vs-random difference is exactly 0, p=1.0).

To quantify the upper bound on any hidden transfer effect: across the paired comparisons (b≈c≈5), an exact two-sided 95% CI on the content-vs-random discordance rate difference is roughly **[−9pp, +9pp]**. So we can rule out a transfer-specific advantage larger than ~9pp at this n; we cannot rule out a small (<9pp) effect. A confirmatory run at the pre-registered n≥600 (Amendment A5) would shrink this bound to ~±2pp.

**Reviewer-facing sentence:** "The paired difference between content-bearing transfer and a matched content-free perturbation is exactly zero (McNemar p=1.0, 5–5 discordant split); a 95% CI bounds any transfer-specific advantage to under ~9 percentage points at n=73, and the trained adapter is *significantly worse* than random (p=0.039). We thus find no evidence that single-vector injection transfers task-relevant content beyond a magnitude-matched perturbation."

## Rigorous rerun — full control set (Amendment 003, n=73)

Per-bug-seeded controls, random drawn R=5 per bug. Pass counts of 73:

| Arm | Pass | Acc [Wilson 95% CI] |
|---|---|---|
| receiver_k1 (baseline) | 19 | 26.0% [17.3, 37.1] |
| donor_k1 | 17 | 23.3% [15.1, 34.2] |
| unrelated_task | 19 | 26.0% |
| self_inject (C4) | 22 | 30.1% [20.8, 41.4] |
| random (mean of 5 draws) | — | 30.4% (draws 21/21/25/21/23, spread 5.5pp) |
| **transfer** | 23 | 31.5% [22.0, 42.9] |
| **text_hint** | 35 | **47.9% [36.9, 59.2]** |

**Paired McNemar (rigorous):**

| Comparison | b | c | p | Verdict |
|---|---|---|---|---|
| transfer vs self_inject | 3 | 2 | 1.000 | n.s. |
| transfer vs random(majority) | 7 | 5 | 0.774 | n.s. |
| transfer vs unrelated | 5 | 1 | 0.219 | n.s. |
| transfer vs receiver | 7 | 3 | 0.344 | n.s. |
| **text_hint vs transfer** | 18 | 6 | **0.0227** | **significant** |

**Discordant-restricted** (donor passes & receiver fails, n=8): text_hint 5/8, transfer 3/8, self 2/8, unrelated 2/8, random(majority) 1/8.

**The dissociation (Result IIb):** the donor's fix as *text* lifts the receiver to 47.9% (highest arm; McNemar vs transfer p=0.023), while the *same* information as an injected single vector gives no lift over noise (p=1.000 vs self). The receiver **can** use the information and the donor **has** it — the single-vector channel does not deliver it.

## Site sweep — 2×2 boundary (Amendment 004, n=73, Holm-corrected)

Frozen before results. Cells = {donor source: s1=layer14, sall=all-36-layers} × {injection: t1=1 token, tall=all tokens}. Success rule (pre-registered): Holm-corrected p<0.05 **and** effect ≥ +3pp.

| Cell | Transfer | Random | Effect | McNemar p | Holm verdict |
|---|---|---|---|---|---|
| s1_t1 | 0.315 | 0.301 | +1.4pp | 1.000 | not positive |
| s1_tall | 0.000 | 0.110 | −11.0pp | 0.0078 | negative effect |
| **sall_t1** | 0.329 | 0.178 | **+15.1pp** | **0.0433** | **SUGGESTIVE** (fails Holm thr 0.0167) |
| sall_tall | 0.000 | 0.000 | 0.0pp | 1.000 | not positive |

**Holm ordering** (by ascending p): s1_tall p=0.0078 (thr 0.0125, wrong direction); **sall_t1 p=0.0433 (thr 0.0167 — fails)**; s1_t1 p=1.000; sall_tall p=1.000. **No cell survives the 4-cell correction with a positive effect.** `sall_t1` is nominally significant but **suggestive, not confirmed** — it needs the powered confirmatory run.

**The boundary is asymmetric and non-monotone:** transfer needs (a) rich donor read-out (all layers) **and** (b) surgically narrow receiver injection (1 token). All-token injection is catastrophic (0/73 in both tall cells). This falsified the pre-registered monotone prediction (Amendment 004) that content advantage grows with channel width.
