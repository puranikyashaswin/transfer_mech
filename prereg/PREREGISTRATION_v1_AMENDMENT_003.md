# Amendment 003 — Honest Re-scoping: Exploratory, Under-Powered; Controls Restated; Confirmatory Plan Pinned

**Date: 24 Sep 2026. Status: FROZEN.** Supersedes nothing; *clarifies* Amendment 001 and restates the evidentiary status of all Gate-1 results to date. Written in response to an internal adversarial review (see `paper/METHODOLOGY_NOTES.md`).

## Why this amendment exists

An internal adversarial review found that the write-up's headline claims used n=73 (same-family) and n=43 (adapter) on HumanEval-Java, while Amendment 001 §A5 froze the **confirmatory** test at **n≥600 on MultiPL-E-Java + Defects4J** with the rule *"if we cannot reach the required n, we do not claim the effect — we report the confidence interval and stop."* The write-up claimed an effect at the under-powered n. This amendment corrects that, re-scopes every existing claim to its true evidentiary status, restates the control set, and pins the confirmatory plan. **Exploratory findings never loosen a frozen criterion (firewall intact).**

## 1. Evidentiary status of existing results — DOWNGRADED

All Gate-1 results so far (naive ridge 0/9; norm-matched sweep ≤2/9; trained adapter; same-family pilot; alpha sweep; mechanistic probe) are **exploratory and under-powered**, NOT confirmatory. Specifically:

- The central comparison "transfer vs matched random perturbation" is a **perfect tie** (McNemar exact p=1.0, 5–5 discordant split at n=73) — but this is an **absence of significance, not proof of absence**. At n≈73, even a true 10pp content advantage would not reach p<0.05.
- We therefore claim ONLY: **we detect no transfer-specific advantage over a matched content-free perturbation, with a paired 95% CI of roughly ±9pp at n=73.** We do NOT claim "single-vector transfer does not transfer" as a confirmed fact at this n.
- The one statistically significant paired result is **adapter < random (McNemar p=0.039)**, i.e., the trained adapter is *worse* than a matched perturbation — evidence against content-transfer, but from an exploratory cell.

## 2. Control set — restated honestly

Amendment 001 §A3 froze five controls: (C1) no-op, (C2) norm-matched random, (C3) shuffled donor, (C4) receiver-self, (C5) **unrelated-task donor states**, plus a separately-pinned **text-hints** control ("the key information-matched control"). To date we ran C1–C4 and receiver-sampling, but **did NOT run C5 (unrelated-task) or the text-hints control.** This amendment:

- **Reinstates** C5 (unrelated-task donor states) and **text-hints** as required controls; a run is complete only when all are present.
- **Pins a determinism fix:** all random vectors (C2) must be seeded **per-bug** (seed = stable hash of `bug_id`), and each random control must use **R≥5 independent draws**, reporting mean ± spread. (Found: C2 was seeded globally, so two runs of the same config disagreed on 6/73 bugs — the control was a single noise realization.)

## 3. Corrections to reported numbers

- **Donor-specific flips (same-family):** operationalized as *transfer passes AND random fails AND self fails* = **1 bug (SORT_ARRAY)**, not 2. ("Not flipped by random" = 3: SORT_ARRAY, STRING_SEQUENCE, STRONGEST_EXTENSION.)
- **Denominators:** every cell reports n explicitly. The α-sweep α=1.0 cell ran on 68/73 pilot bugs (5 missing: VOWELS_COUNT, WILL_IT_FLY, WORDS_IN_SENTENCE, WORDS_STRINGS, X_OR_Y); α=0.1/0.5 on 69. The pilot α=1.0 (0.315, n=73) and sweep α=1.0 (0.338, n=68) are the same config on different subsets — both reported, the discrepancy disclosed.
- **α=0.1 inversion:** at α=0.1, self-inject (0.304) *exceeds* transfer (0.275). This is MORE evidence content doesn't matter, and it qualifies the "rises identically with magnitude" framing.

## 4. New analyses required before any "boundary" claim

- **Discordant-restricted same-family:** the same-family eval must be reported restricted to bugs where the **donor passes and the receiver fails** (the only bugs where a causal flip is possible). This requires a donor (Qwen-3B) pass-run on bugs 90–162.
- **Mechanistic probe:** (i) drop the "magnitude-only" claim (random is a smaller norm-increase but a larger direction change — direction, not magnitude, drives the KL); (ii) bound the "argmax never moves" claim (0/30 → ≤11.6% at 95% CI); (iii) use a random (not alphabetical M–S) bug sample; (iv) add a low-α arm to separate "injection too weak" from "content invisible."

## 5. Confirmatory plan — unchanged and reaffirmed

The confirmatory test remains **Amendment 001 §A5, frozen**: discordant selection from a held-out pool, **n≥600** on MultiPL-E-Java + Defects4J, one frozen config, min claim ≥60% win-share of discordants + ≥3pp. **Until that budget is spent, every statement about single-vector transfer is exploratory.** This amendment does not authorize claiming the effect at lower n.

## 6. What we may write now

We may write the exploratory paper **only** if it: (a) labels all results exploratory/under-powered; (b) reports CIs/equivalence bounds instead of "never beats"; (c) runs and reports the full control set (incl. C5 + text-hints) with per-bug-seeded multi-draw random; (d) scopes the boundary claim to the tested cell (HumanEval-Java APR, 1.3–3B, donor layer 14, single-token, temp 0.2); (e) engages single-vector activation-steering literature; (f) presents the registered confirmatory run as the planned test.

**Amendment log:** v1 (frozen) → 001 (intervention/receiver/power) → 002 (trained adapter) → 003 (this: honest re-scoping + controls restated + confirmatory reaffirmed).
