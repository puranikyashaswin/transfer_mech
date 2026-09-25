# Amendment 007 — Instrumented Diagnostic Pilot: Strength × Subspace

**Date:** 2026-07-31
**Status:** Pre-registered BEFORE the diagnostic pilot run. Frozen.
**Amends:** PREREGISTRATION_v1 (+ Amendments 001–006)
**Scope:** EXPLORATORY / DIAGNOSTIC. This amendment governs a pilot, NOT a confirmatory test.

---

## §1. Purpose and evidentiary status (read first)

The confirmatory run (Amendment 006, n=215) established a clean negative result: the
frozen adapter arm (pass 0.516) underperforms both receiver_k1 (0.637) and random_norm
(0.670), with a 10× compile-failure spike (adapter 22.3% vs receiver 2.3%). That is the
confirmatory finding and it stands on its own.

**This pilot exists only to localize WHERE the failure comes from** — injection
*strength* vs injection *subspace* — so that a future, separately-powered confirmatory
run can be aimed correctly. It produces point estimates, wide confidence intervals, and
instrumentation logs (‖Δ‖, receiver-subspace projection) that the confirmatory logs did
not capture. **It makes no confirmatory claim.**

## §2. Why the confirmatory FWER does NOT extend here

Multiple-comparison correction (Holm) applies to a single pre-planned family of
hypotheses tested together against a shared error rate. The confirmatory McNemar family
was already corrected under Amendment 004/006 and that stands.

The diagnostic z-test (compile-failure vs receiver-difficulty split) and this pilot are
**post-hoc, multi-stage exploratory analyses**, not members of the pre-registered
confirmatory family. Folding them retroactively into one Holm correction would (a)
overcorrect tests that were never confirmatory, and (b) lend false confirmatory cover to
an exploratory process.

**Therefore:** the pilot is reported with effect sizes + CIs only, is explicitly labeled
exploratory/diagnostic, and is NOT bundled into the confirmatory family-wise error rate.
Any future confirmatory claim arising from this pilot's readout will be pre-registered as
its OWN amendment with its OWN power analysis and correction, before that run.

## §3. Design

- **Split:** dev (bugs 80–119, n=40) only. The test split (120+, n=215) is untouched and
  remains the clean held-out set.
- **Design:** paired 2×2 — the SAME 40 bugs under all 4 conditions (paired, not
  independent cells; the paired design is what makes n=40 informative at all).
- **Factors:**
  - Injection strength α ∈ {0.5, 1.0}
  - Target subspace ∈ {adapter-as-is, adapter→receiver-PCA-aligned}
- **Subspace provenance (anti-leakage):** the receiver "active subspace" PCA is fit ONLY
  on receiver layer-12 activations from the **train split (bugs 0–79)** — never on dev or
  test. The fitted basis is then *applied* (frozen) to dev bugs for the aligned
  condition. Fitting on the same dev bugs used for readout would leak the evaluation data
  into the alignment path and bias it toward looking better than it is; that is
  prohibited. Train-split-only fitting keeps the aligned condition honest.
- **Instrumentation (new):** per-bug log of ‖Δ‖ (injected-vector norm), the receiver-side
  projection of Δ onto the receiver's layer-12 active subspace, and gate magnitude. This
  is the data the confirmatory logs lacked.
- **Frozen:** receiver, adapter weights, prompt, decoding (temp 0.2, top-p 0.95, seed 42,
  MAX_NEW_TOKENS=1024). Only α and the subspace projection vary.

## §4. Power reality (computed before run — do not relabel later)

Paired McNemar at observed discordance 0.288: n=40 detects only a **≥80% discordant-win**.
A 60% win needs n=194; restoring compile success 78%→90% (two-prop) needs n=146/group.
**n=40 CANNOT deliver a decisive test.** This pilot is therefore scoped to information
gain, not significance. Any claim of a decisive result from this n is pre-committed to be
rejected as overclaiming.

## §5. Pre-specified readout — decision boundaries BEFORE running

The pilot moves us toward a hypothesis only on these pre-committed patterns; anything
else is "cannot tell yet":

- **Evidence for STRENGTH:** compile-failure rate tracks α monotonically (higher α →
  higher compile-failure) in BOTH subspace conditions, AND ‖Δ‖ is the dominant correlate
  of failure. → points to a magnitude problem.
- **Evidence for SUBSPACE:** the aligned condition's compile-failure rate is materially
  lower than the as-is condition's (pre-committed gap ≥ 15pp on the point estimate)
  REGARDLESS of α, AND the receiver-subspace projection of Δ is the dominant correlate of
  failure. → points to an aiming problem.
- **BOTH:** failure drops only when α is low AND subspace is aligned (interaction).
- **CANNOT TELL:** point estimates fall inside the pre-committed bands above but CIs
  overlap, or no clean monotonic/aligned pattern emerges. **Default outcome.** Report
  estimates + CIs and stop; do not spin.

## §6. Stopping rule

Run all 40 bugs × 4 conditions once. Stop. Report the §5 readout against the
pre-committed boundaries. No iterative re-running to chase significance. The output feeds
the design (and power analysis) of a future confirmatory amendment — nothing more.

## §7. What this pilot does NOT do

- Does not touch the test split.
- Does not make or support a headline/confirmatory claim.
- Does not "fix" the adapter — it diagnoses where the fix should aim.

---

*Frozen before the pilot run. Signed into the pre-registration chain as Amendment 007.*
