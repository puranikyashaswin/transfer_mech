# Amendment 006 — Confirmatory Sizing, Source Mix, Arm Set, and Honest-n Pre-commitment

**Date: 2026-07-31. Appended to frozen PREREGISTRATION_v1.md + Amendments 001–005.**

## Why this amendment exists
Amendment 001 §A5 froze: *"We will size the final test set from the Gate-1 pilot's measured discordance (pilot n≈50) and report that number in a dated amendment before the confirmatory run."* That required sizing amendment was **never written** before the first (now-wiped) confirmatory run, which executed at n=221 without it — a process gap this amendment closes. Separately, the confirmatory corpus and adapter were destroyed when the instance's ephemeral disk was wiped on stop (see METHODOLOGY_NOTES.md "DATA-LOSS ROOT CAUSE"); they must be rebuilt, and the rebuild + run must be governed by an explicit, dated, frozen sizing rule. **This amendment is written BEFORE the corpus rebuild is finalized and BEFORE any GPU/test-split run.** No test-split result has been observed since the data loss.

## 1. Pilot-sized required n (the A5 mandate, now discharged)
Gate-1/rigorous pilot measured:
- **discordance d = 8/73 = 0.110** (transfer vs receiver_k1).
- transfer win-share among discordants = 3/8 = 37.5% (below the 60% claim threshold — consistent with the exploratory null).

McNemar power to claim a paired win-share ≥60% (one-sided α=0.05, power=0.80):
- discordants required D = (z_α + z_β)² / (2w − 1)² = (1.645+0.842)² / (0.2)² ≈ **155 discordant pairs**.
- total n = D / d = 155 / 0.110 ≈ **n = 1,411** (at the measured d=0.11). Even at the more optimistic gate-1-pilot d=0.18, n ≈ **859**.

**Therefore the pilot-sized required n is n ≈ 859–1,411.** (Sensitivity: at d=0.11, n=385 detects only a win-share ≥0.69; n=600 detects ≥0.65; n=1,100 detects ≥0.61.)

## 2. Corpus source mix and achievable n (the A5 escalation path)
Per A5, the confirmatory test = **MultiPL-E-Java + Defects4J single-method subset**, pooled; if they cannot reach 600, expand with **HumanEval-Java held-out + MBPP-Java (MultiPL-E)** until n≥600. Execution-verified only (Gate-0, A6): a problem is usable only if it has a correct translation that PASSES all tests AND a buggy mutant that COMPILES and FAILS ≥1 test.

- MultiPL-E-Java alone has a hard ceiling of **≈385** usable problems (prior verified build: 385 correct + 385 buggy + 385 tests).
- The A5-authorized expansion (MBPP-Java recovery + HumanEval-Java held-out + Defects4J single-method) will be **attempted** to push toward/past 600.
- **Achievable n = the largest execution-verified paired corpus produced by this build, reported exactly.** Pre-registered expectation: **385 ≤ achievable n < 859** (the MultiPL-E ceiling plus whatever the expansion yields is unlikely to reach the pilot-sized 859–1,411). The exact achieved number is reported in the results, with the CI.

## 3. Frozen test split (Amendment 002, unchanged for the rebuild)
The corpus is deterministically ordered by bug_id. Frozen split over the rebuilt corpus:
- **train = bugs 0–79** (adapter training only),
- **dev = 80–119** (adapter α selection only, frozen before test),
- **test = 120–end** (the held-out confirmatory set, one shot).

The trained adapter (Amendment 002 B2) is retrained on the rebuilt train split only, frozen before test. No adapter sees the test split. If the rebuilt corpus has fewer than 120 problems, the whole run aborts (not viable); the MultiPL-E ceiling (385) comfortably exceeds this.

## 4. Arm set (Amendment 003 §2, restated and made complete)
Each test bug is evaluated under ALL of these arms, execution-verified by JUnit:
1. **receiver_k1** — receiver single-shot, no injection (baseline).
2. **adapter** — receiver + frozen trained adapter injection (the intervention).
3. **random_norm** — receiver + same-norm RANDOM vector, **R = 5 per-bug-seeded draws** (seed = BASE_SEED ⊕ hash(bug_id) ⊕ draw_index); magnitude control.
4. **shuffled** — receiver + adapter output for a DIFFERENT bug (specificity).
5. **self_inject** — receiver + its OWN hidden state re-injected (hook sanity).
6. **unrelated_task (C5)** — receiver + donor activation from an unrelated task (task-relevance control).
7. **text_hint** — receiver with a textual hint (positive control / ceiling reference).
8. **donor_k1** — donor single-shot (donor-capability reference).

A run is **complete only when all 8 arms are present for every test bug.** Generation budget = **1024 tokens for every arm** (Amendment 005), temp 0.2, top-p 0.95, seed 42, final-prompt-token additive injection, α=1.0.

## 5. Decision rule and honest-n pre-commitment (A5 + Amendments 003/005, operative form)
- **Claim the headline transfer effect ONLY IF ALL hold:** (i) achieved n ≥ the pilot-sized required n (§1, ≈859–1,411); **AND** (ii) adapter paired win-share ≥ 60% of discordants vs the strongest control; **AND** (iii) effect ≥ +3pp vs the strongest control; **AND** (iv) McNemar exact one-sided p<0.05 vs EVERY control.
- **Because §1 shows the required n is almost certainly unreachable (achievable n < 859), we PRE-COMMIT, in writing, before any run:** we will **report achieved n, the paired effect size, the confidence interval, and an equivalence bound, and we will NOT claim the headline effect** — regardless of how the adapter arm looks. If the adapter underperforms any control (replicating the exploratory adapter < random_norm result), this is reported as a clean **negative/boundary result**.
- This amendment does NOT loosen any frozen criterion; it makes the already-frozen honest-n rule operative and explicit, and discharges the A5 sizing mandate that was previously skipped.

## 6. What is unchanged (still frozen)
Model pair (A1); donor layer 14 / receiver layer 12 / final-token / additive (A2); the five-plus controls (A3) as completed in §4; the exploratory/confirmatory firewall (A4); the effect floor + kill criteria (A5, A6); the trained-adapter map (Amendment 002 B2); the exploratory re-scoping (Amendment 003); the 1024-token budget (Amendment 005). The confirmatory test remains ONE frozen config on held-out data, one shot.
