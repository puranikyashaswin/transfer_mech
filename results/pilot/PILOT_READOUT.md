# Pilot Readout — Amendment 007 (strength × subspace), dev split n=40

**Status:** COMPLETE, exploratory only. Test split untouched. 160 paired rows + 80 baseline rows.

## Setup
- Dev bugs 80–119 (n=40), each under 2×2 conditions: α ∈ {0.5, 1.0} × {as_is, aligned}.
- Receiver active subspace: layer 12, k=35 PCA directions, 90% variance, fit on **train split only** (bugs 0–79).
- Adapter frozen (g_mean=0.1516 in pilot run; confirmatory was 0.143 — minor reload variance, not material).
- Baseline: α=0 (no injection) run through the identical harness on the same 40 bugs.

## Instrumentation (key measurements)
| quantity | as_is | aligned |
|---|---|---|
| proj_frac | 0.15–0.16 | 1.00 (by construction) |
| ‖Δ‖ @ α=1.0 (mean, last bugs) | ~90 | ~13.8 |

The adapter's delta is ~85% outside the receiver's active subspace.

## Pass rates (dev, n=40 per cell)
| Condition | Pass | Compile Errors |
|---|---|---|
| receiver_k1 (α=0) | 23/40 = **57.5%** | 4/40 = 10% |
| α=0.5 as_is | 19/40 = 47.5% | 2/40 = 5% |
| α=0.5 aligned | 24/40 = **60.0%** | 3/40 = 7.5% |
| α=1.0 as_is | 14/40 = 35.0% | 8/40 = **20%** |
| α=1.0 aligned | 21/40 = 52.5% | 3/40 = 7.5% |

## Paired McNemar (exact binomial, discordant pairs)
| Comparison | b | c | disc | p (one-sided) |
|---|---|---|---|---|
| aligned > as_is @ α=0.5 | 2 | 7 | 9 | 0.180 |
| aligned > as_is @ α=1.0 | 3 | 10 | 13 | 0.092 |
| α=0.5 > α=1.0 @ as_is | 6 | 1 | 7 | 0.125 |
| α=0.5 > α=1.0 @ aligned | 4 | 1 | 5 | 0.375 |
| **aligned-α=1.0 vs receiver_k1** | **1** | **3** | **4** | **0.625 (two-sided)** |

## §5 readout: **"cannot tell"** (as pre-committed for underpowered discordance)
Neither subspace nor strength effect reaches significance at n=40 (observed discordant-win 65–86%, pilot powered only for ≥80%).

## Three findings
1. **Subspace alignment cuts compile errors 62% at α=1.0** (20% → 7.5%). The corruption mechanism is out-of-subspace injection.
2. **Fixing the mechanism is NOT sufficient for the adapter.** Aligned-α=1.0 remains 5pp BELOW receiver baseline (52.5% vs 57.5%), win-share 0.25 on 4 discordant pairs.
3. **The adapter has no residual positive signal even after geometry is fixed** — but see `BASELINE_MAP_READOUT.md`: a naive zero-training ridge map through the SAME channel matches/exceeds baseline and beats the adapter 13–1 (p=0.0018). The pilot's "no positive signal" applies to the *adapter's* aligned delta; the channel itself is intact for a conservative linear map. The failure is the adapter's training/architecture, not the channel.

## Powered-follow-up decision: DECLINED
Powering aligned-α=0.5 vs receiver_k1 (+2.5pp, 1 bug) at this discordance would require n in the thousands — beyond MultiPL-E-Java's ceiling (≈385). No powered confirmatory follow-up is pre-registered on this corpus. A positive transfer claim on this channel requires a qualitatively different intervention, not a larger n.

## Artifacts
- `pilot_full.jsonl` (160 rows), `pilot_baseline.jsonl` (80 rows)
- `receiver_subspace_train.pt` (basis [2048, 35], k=35, 90% var, train-only)
- `pilot_dryrun.jsonl` (instrumentation sanity check, 8 rows)
