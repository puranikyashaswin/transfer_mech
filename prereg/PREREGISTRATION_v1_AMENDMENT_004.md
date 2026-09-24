# Amendment 004 — Pre-registered Analysis Plan for the 2×2 Injection-Site Sweep (Positive Arm)

**Date: 24 Sep 2026. Status: FROZEN BEFORE RESULTS.** Written and frozen **before** any inspection of the 2×2 injection-site sweep pass/fail outcomes, to avoid HARKing (hypothesizing after results are known). The sweep data were still being generated at the time of writing; this document fixes the decision rules in advance. Supersedes nothing; it *adds* one positive arm to the exploratory program defined in Amendments 001–003. The exploratory/confirmatory firewall (Amendment 001 §A5, reaffirmed in 003) is intact: **this sweep is exploratory, and its outcome never loosens a frozen confirmatory criterion.**

## 1. What the sweep is

A 2×2 factorial over **injection channel width**, applied to the same-family pair (donor Qwen2.5-Coder-1.5B → receiver Qwen2.5-Coder-3B), with the same ridge map, on the same HumanEval-Java eval slice as the pilot:

| | **1 token** (final prompt token) | **all tokens** |
|---|---|---|
| **1 layer** (receiver layer 17) | `s1_t1` | `s1_tall` |
| **all layers** (all 36 receiver layers) | `sall_t1` | `sall_tall` |

- `s1_t1` is the frozen minimal mechanism of Amendments 001–003 (the null already established).
- `sall_tall` is the richest channel in the sweep — the closest analog to the multi-position/multi-layer mechanisms that succeed in CALM/C2C/LatentMAS.
- Each cell has its **own norm-matched per-site random control** (`arm=random`), so every cell compares content (transfer) to a matched perturbation at that cell's channel width.
- A `baseline / receiver_k1` cell gives the receiver's no-injection pass rate.
- Checkpoint schema: rows keyed by `(bug_id, cell, arm)` with `passed ∈ {true,false}` and `status`.

## 2. Hypotheses, fixed in advance

- **H0 (per cell):** transfer pass rate ≤ matched-random pass rate within the cell (content carries nothing beyond perturbation).
- **H1 (the boundary hypothesis):** there exists a channel width at which transfer **exceeds** its matched random control — i.e., the transfer threshold is crossed between `s1_t1` (null) and richer cells. The graded prediction: **content advantage increases monotonically with channel width** (none in `s1_t1`, possible emergence in `s1_tall`/`sall_t1`, largest in `sall_tall`).

## 3. Success / failure margins, fixed in advance

Let `D_cell = transfer_pass − random_pass` (McNemar discordant difference) within a cell, over the common eval set.

- **A cell is a POSITIVE (channel crosses the threshold) iff:** McNemar exact test on transfer-vs-random within that cell gives **p < 0.05 AND D_cell ≥ 3pp**. (The 3pp floor matches the confirmatory min-effect in Amendment 003 §5 and guards against a significant-but-tiny effect.)
- **Multiple comparisons:** the 4 cells are a pre-specified ordered family, tested in the fixed order `s1_t1 → s1_tall / sall_t1 → sall_tall` with **Holm correction across the 4 transfer-vs-random p-values**. A cell is declared positive only if it survives Holm.
- **The sweep is a POSITIVE result iff at least one cell is positive after Holm.**
- **The sweep is a clean BOUNDARY result iff:** `s1_t1` is null (already shown) AND `sall_tall` is positive after Holm AND the content advantage is non-decreasing across channel width. This is the headline "transfer threshold" finding.
- **ALL-NULL outcome (no cell positive):** we do not claim the boundary exists; we report the negative across all widths and trigger the fallback in §5.

## 4. Confounder rules, fixed in advance

- **Richer channel = bigger perturbation is NOT content transfer.** Each random control is norm-matched to its own cell's transfer delta, so the transfer-vs-random comparison isolates content. A cell where **both** transfer and random beat baseline equally is a **perturbation effect at scale, not a positive** — it does not count toward H1.
- **All-token length mismatch:** donor and receiver prompt token lengths differ; all-token transfer clamps by `min(receiver_len, donor_len)` and pads by repeating the last mapped token (frozen implementation). Disclosed as a limitation of the `*_tall` cells.
- **Discordant-restricted secondary analysis:** the primary metric is the all-bugs population effect; we additionally report each cell restricted to bugs where the donor passes and the receiver baseline fails (the only bugs where a causal flip is possible). Both are reported; the primary claim uses the all-bugs McNemar.

## 5. Pre-registered fallback if all four cells are null

If no cell crosses the threshold, the positive-arm contribution shifts from "width" to "mechanism class":

- **Attention-lite arm:** a single additive injection that instead targets the **attention output / value pathway** (rather than the residual stream) at one layer, with its own matched random control — testing whether the residual-stream *locus* (vs. the channel width) is what blocks transfer.
- **Mechanistic-probe-first criterion:** for any null behavioral cell, report the mechanistic probe (final-layer residual cosine, next-token KL, argmax-changed) for that cell. The ceiling-raising claim becomes: **"a transfer effect is visible in the mechanistic probe at a channel width before (or without) appearing in pass@1"** — i.e., the probe is a more sensitive detector of emerging transfer than execution-verified pass rate. This is pre-registered as the fallback contribution if pass@1 is everywhere null.

## 6. What we may claim after the sweep

- If positive: "single-vector transfer fails, but the same content injected across a wider channel (more tokens / more layers) crosses the transfer threshold — the channel, not the map, is the binding constraint," stated with the Holm-corrected statistics and the norm-matched-control isolation.
- If all-null: "widening the channel within the residual stream up to all-layer/all-token still does not produce content transfer at this scale; the binding constraint is the residual-stream locus or the model scale," plus the probe-first fallback.
- Either way: everything here remains **exploratory** until the Amendment 003 §5 confirmatory run (n≥600, MultiPL-E + Defects4J) is executed.

**Amendment log:** v1 (frozen) → 001 (intervention/receiver/power) → 002 (trained adapter) → 003 (honest re-scoping + controls + confirmatory reaffirmed) → 004 (this: 2×2 sweep analysis plan, frozen before results).
