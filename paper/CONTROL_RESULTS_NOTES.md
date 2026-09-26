# Control Results — FINAL (rigorous_rerun complete, 73/73 bugs, Amendment-003 controls)

**Status: COMPLETE.** Run finished 2026-09-24; results synced to S3 and written into PAPER.md §6.5 + main.tex "Result IIb". Numbers below are recomputed from the final checkpoint (`results/rigorous/rigorous_checkpoint.jsonl`, 803 rows).

## Final all-bugs pass rates (n=73, same-family, α=1.0, 95% Wilson CI)

| arm | pass/n | acc [CI] |
|---|---|---|
| receiver no-op (C1) | 19/73 | 0.260 [0.173, 0.371] |
| donor (Qwen-3B) self | 17/73 | 0.233 [0.151, 0.342] |
| unrelated-task (C5) | 19/73 | 0.260 [0.173, 0.371] |
| self-inject (C4) | 22/73 | 0.301 [0.208, 0.414] |
| random-norm (C2) ×5 draws | 21,21,25,21,23 | 0.304 ± 0.022 |
| **transfer** | 23/73 | 0.315 [0.220, 0.429] |
| **text-hint** | 35/73 | **0.479 [0.369, 0.592]** |

## Paired McNemar (exact, two-sided), n=73

| comparison | discordant (b,c) | p |
|---|---|---|
| transfer vs self-inject | 3 vs 2 | **1.000** |
| transfer vs random (majority of 5) | 7 vs 5 | 0.774 |
| transfer vs unrelated-task | 5 vs 1 | 0.219 |
| **text-hint vs transfer** | **18 vs 6** | **0.0227** (significant) |
| transfer vs receiver baseline | 7 vs 3 | 0.344 |

## Discordant-restricted (donor passes & receiver fails; n=8)

text-hint 5/8 (62.5%) · transfer 3/8 (37.5%) · self-inject 2/8 · unrelated-task 2/8 · random mean 1/8. Discordant bugs: MAX_FILL, NEXT_SMALLEST, PARSE_NESTED_PARENS, REMOVE_VOWELS, SAME_CHARS, SOLVE_STRING, STRING_SEQUENCE, WILL_IT_FLY.

## The dissociation (final)

**text-hint passes & transfer fails: 18 bugs; transfer passes & text-hint fails: 6 bugs.** The information transfers through natural language (48%) but not through the single-vector hidden-state channel (31.5% = random 30.4%). This is the paper's central controlled claim.

## Determinism fix confirmed
Random vectors per-bug-seeded (sha256 of `bug_id|arm|draw`), R=5 independent draws; mean 0.304 ± 0.022 spread. No longer a single un-seeded noise realization.

## Pass rates per arm (partial, n≈46 eval bugs at snapshot 2026-09-24T14:52Z)

| arm | pass/n | rate | reading |
|---|---|---|---|
| **text_hint** | 22/46 | **0.478** | receiver uses donor's fix when shown it as TEXT — highest of all arms |
| transfer | 15/46 | 0.326 | ties random |
| random_norm | 75/230 | 0.326 | (R=5 draws/bug, per-bug seeded) |
| self_inject | 13/46 | 0.283 | receiver's own state |
| receiver_k1 (baseline) | 12/46 | 0.261 | no-op receiver |
| donor_k1 | 11/46 | 0.239 | donor's own pass rate (sanity / discordant numerator) |
| unrelated_task | 11/46 | 0.239 | donor state from a different bug — lowest, no lift |

## The dissociation (the key new finding)

Over bugs with both arms present (n=46):
- **text_hint passes AND transfer fails: 11 bugs** — MAXIMUM_K, MAX_ELEMENT, MAX_FILL, MONOTONIC, MOVE_ONE_BALL, NEXT_SMALLEST, NUMERICAL_LETTER_GRADE, ROUNDED_AVG, SAME_CHARS, SORT_NUMBERS, SORT_THIRD.
- **transfer passes AND text_hint fails: 4 bugs.**

**Interpretation (for the paper).** This is the dissociation that makes the negative airtight. The donor's fix information is *usable* by the receiver — when it arrives as text in the prompt, pass rate jumps from 26% (baseline) to 48% and beats every injection arm. But the *same* information, injected as a single hidden-state vector, produces no lift over a content-free perturbation (transfer 0.326 = random 0.326). And an unrelated-task donor state (0.239) does *worse* than baseline, confirming the transfer arm's (tiny) lift is not generic "any donor state helps." So:
- It is NOT that the receiver can't use the information (text_hint proves it can).
- It is NOT that the information isn't in the donor (donor passes these bugs).
- It IS that **the single-vector residual channel does not deliver the information** — the channel is the binding constraint, not the receiver's competence and not the map's content.

This directly answers the review's "the strongest control never ran" (FATAL 2) and gives the paper a *positive* control result to anchor the negative: information transfers fine through the natural-language channel and not at all through the single-vector hidden-state channel.

## Random control determinism (FATAL/attack 12)

Random vectors are now per-bug seeded (sha256 of `bug_id|arm|draw`) with R=5 independent draws. Mean 0.326 over 230 draws. Per-draw spread and any transfer-vs-random disagreement will be reported (mean ± spread, not one noise realization) once the run completes.

## TODO when run completes
1. Recompute all arms over the FULL eval slice; report exact n per cell.
2. Discordant-restricted table (donor passes & receiver fails) per Amendment 003 §4.
3. Random-draw spread (mean ± std over the 5 draws); transfer-vs-random McNemar with the seeded draws.
4. text_hint vs transfer McNemar (the dissociation statistic) + CI.
5. Feed into make_stats.py, regenerate figures, write the new results subsection.
