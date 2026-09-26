# Baseline Linear Map Readout — critic-roadmap P1.3 (exploratory, dev split)

**Status:** COMPLETE, exploratory only. Test split untouched. 40 dev bugs.

## Question
Is the confirmatory negative (adapter 111/215 < receiver_k1 0.637) specific to the
**trained adapter's architecture**, or is single-vector cross-model injection broken
in general at this site (donor L14 → receiver L12, final prompt token, additive)?

## Method
- Map: centered ridge (λ=1.0), no nonlinearity, no bottleneck, no task training.
  Fit target = receiver deviation from its own mean (h_r − r_mean), so
  W·(h_d − d_mean) IS the additive delta (adapter-parallel semantics).
- Fit on TRAIN split only (bugs 0–79), anti-leakage. Frozen, applied to dev (80–119, n=40).
- Decoding: exact confirmatory config (α=1.0, temp 0.2, top-p 0.95, seed 42, 1024 new tokens).
- Offline train fit: **R²=0.865, mean cos=0.940** (high offline alignment — the trap).

## Dev pass rates (n=40/cell, with 95% Wilson CI)
| Condition | Pass | CI | compile_err |
|---|---|---|---|
| receiver_k1 (α=0) | 23/40 = 57.5% | [0.42, 0.72] | 10% |
| **baseline_ridge α=1.0** | **26/40 = 65.0%** | [0.50, 0.78] | 5% |
| adapter α=0.5 aligned | 24/40 = 60.0% | [0.45, 0.74] | 7.5% |
| adapter α=1.0 aligned | 21/40 = 52.5% | — | 7.5% |
| adapter α=0.5 as_is | 19/40 = 47.5% | — | 5% |
| adapter α=1.0 as_is | 14/40 = 35.0% | [0.22, 0.51] | 20% |

## Paired McNemar (dev, exact binomial)
| Comparison | wins | loses | disc | winshare | p |
|---|---|---|---|---|---|
| **ridge vs adapter α=1.0 as_is** | **13** | **1** | 14 | **0.929** | **0.0018** |
| ridge vs receiver_k1 | 3 | 0 | 3 | 1.000 | 0.25 (underpowered) |
| adapter α=1.0 as_is vs receiver_k1 | 2 | 11 | 13 | 0.154 | 0.0225 |

## Delta norms (median)
- adapter α=1.0 as_is: **81.7** (range 5–172)
- adapter α=0.5 aligned: 6.2
- baseline_ridge α=1.0: **3.0** (range 1.6–7.9)
- **adapter/ridge ratio: 27.5×**

## Three conclusions
1. **Failure is adapter-architecture-specific, not injection-general.** The channel is
   intact: naive ridge through it matches (nominally beats) baseline; trained adapter
   through it collapses (significantly below baseline).
2. **Task-trained optimization is the harm.** Only the task-trained map fails
   catastrophically; it inflated norm 27× and pushed 85% of energy off-manifold.
3. **Reframes the central claim.** Not "single-vector transfer doesn't work" but
   "the field's recommended fix (trained bottleneck connector) actively destroys a
   channel a naive linear map leaves intact."

## Scope
n=40 dev exploratory. ridge-vs-receiver (3–0) suggestive, not significant;
ridge-vs-adapter (13–1, p=0.0018) significant on dev. Confirmatory ridge-vs-baseline
would need its own amendment + powered n (~860+ bugs, beyond corpus ceiling).

## Artifacts
- `baseline_ridge_map.pt` (W [2048,1536], fit train-only)
- `baseline_ridge_dev.jsonl` (40 rows, dev)
- `fit_linear_map.py`, `eval_baseline_map.py`, `analyze_baseline.py`
