# Amendment 008 — Corrected-Site Confirmatory Run: Layer 17 + Ridge Baseline Arm

**Date:** 2026-09-26
**Status:** Pre-registered BEFORE the corrected-site run. Frozen on test-split execution.
**Amends:** PREREGISTRATION_v1 (+ Amendments 001–007)
**Scope:** CONFIRMATORY. This amendment governs a single, frozen confirmatory run on the
held-out test split (bugs 120+, n=215).

---

## §1. Purpose — why this run exists

Two gaps remain after the confirmatory run (Amendment 006) and the diagnostic pilot
(Amendment 007):

1. **Injection-site mismatch (disclosed, not fixed).** The confirmatory adapter arm
   injected at receiver **layer 12**. That site was applied by rote from the cross-family
   DeepSeek amendment; the principled midpoint for the Qwen-3B receiver is **layer 17**,
   which is what the exploratory site sweep used. §12.3 of the paper disclosed this
   mismatch rather than resolving it. This run resolves it.

2. **The ridge baseline was never run at confirmatory scale.** The naive ridge map
   (Amendment 007-era baseline) beat the trained adapter 13–1 — but only on dev (n=40),
   and is therefore split-confounded against the adapter's test-split failure. It has never
   been run on the test split. This run puts the ridge map at the corrected site on the
   full test set, turning the baseline comparison into a confirmatory claim.

Both gaps are closed in **one** combined confirmatory run that spends the single remaining
clean test-split touch exactly once.

## §2. What changes vs Amendment 006 (and what does not)

| Aspect | Amendment 006 (prior) | Amendment 008 (this run) |
|--------|-----------------------|--------------------------|
| Receiver injection layer | 12 | **17** (principled midpoint) |
| Adapter | trained for layer-12 site | **retrained for layer-17 site** (train split only, identical architecture/hyperparameters) |
| Ridge map arm | absent | **added** as a confirmatory arm at layer 17 |
| Test split | touched once (Amendment 006) | **touched once here** (second and final confirmatory touch) |
| Decoding config | temp 0.2, top-p 0.95, seed 42, MAX_NEW_TOKENS=1024 | **identical** — any deviation invalidates comparability |

**What does NOT change:** the donor (Qwen2.5-Coder-1.5B, layer 14 donor readout), the
receiver (Qwen2.5-Coder-3B), the corpus, the train/dev/test splits, the α=1.0 injection
strength, and the final-prompt-token additive injection mechanism.

## §3. Anti-leakage / anti-confound discipline

- **Adapter retraining uses ONLY the train split (bugs 0–79).** The test split is not
  touched during retraining. Retraining at the corrected site is required: injecting the
  layer-12-trained adapter at layer 17 would test "an adapter trained for the wrong site,"
  confounding the site correction with a new unintended failure mode. The retrain removes
  that confound so the layer-17 adapter is a fair test of "the adapter at the corrected
  site."
- **Ridge map refit uses ONLY the train split** (same as the adapter). The fitted map is
  then frozen and applied to test bugs.
- **Layer-independent arms are NOT rerun.** receiver_k1 (no injection), text_hint (prompt
  text, no hidden-state injection), and donor_k1 (donor alone, no receiver injection) are
  bit-identical at layer 12 and 17 because the injection site only matters when a hidden
  state is injected. Reusing their Amendment 006 results unchanged is correct and avoids
  both wasted compute and a spurious "which baseline is real" discrepancy. Only arms that
  inject a hidden state at the receiver layer are rerun at layer 17.

## §4. Arm set (this run)

Arms rerun at layer 17 (hidden-state injection at the receiver layer):

1. `adapter` — retrained-at-17 trained adapter, α=1.0
2. `ridge_map` — naive ridge linear map refit at 17, α=1.0 (NEW confirmatory arm)
3. `random_norm` — same-norm control, R=5 per-bug-seeded draws
4. `self_inject` — receiver's own hidden state re-injected (hook sanity)
5. `shuffled` — adapter output for a different bug (content specificity)
6. `unrelated_task` — fixed unrelated-prompt donor activation through the adapter

Arms reused unchanged from Amendment 006 (layer-independent, no receiver injection):

7. `receiver_k1` — receiver single-shot baseline
8. `text_hint` — textual hint (positive control)
9. `donor_k1` — donor single-shot (capability reference)

## §5. Pre-committed gate: ridge/adapter refit sanity threshold

The test-split run proceeds ONLY if the layer-17 refits pass a pre-committed quality floor
on the **train split** (fit can legitimately degrade at a different layer, but must not be
degenerate). The floor is set honestly below the observed layer-12 dev-ridge fit
(R²=0.865, mean cos=0.940) but high enough to guarantee a non-degenerate map:

- **Ridge map:** train-split R² ≥ 0.60 AND mean cosine ≥ 0.75.
- **Retrained adapter:** training converges (final NLL clearly below initial) and the gate
  magnitude is non-degenerate (g_mean > 0, i.e., the adapter did not collapse to the zero
  map).

**If either refit fails its floor: STOP. Do not spend the test-split touch.** Report the
failed refit and re-scope. This threshold is committed in writing here, before any fit
quality is observed at layer 17 — it is not a mid-run judgment call.

**GATE OUTCOME (recorded after refit, before test-split run):** both refits **PASS**.
Ridge map @ L17 train-split R²=0.8763, mean cos=0.9426 (floor: R²≥0.60, cos≥0.75) — at or
slightly above the layer-12 fit. Retrained adapter @ L17 converged, final NLL≈0.0001,
final g_mean=0.1721 (>0, non-degenerate). The test-split run is cleared to proceed.

## §6. Decoding config (identical to Amendment 006 — required for comparability)

temp 0.2, top-p 0.95, seed 42 (BASE_SEED), MAX_NEW_TOKENS=1024, final-prompt-token additive
injection, α=1.0. random_norm draws use the same per-bug seeding scheme as Amendment 006
(BASE_SEED ⊕ stable_hash(bug_id) ⊕ draw_index), with each draw's norm matched to the
layer-17 adapter output norm for that bug.

## §7. Pre-specified confirmatory readout

The headline confirmatory comparisons of this run, decided before execution:

- **Ridge map vs adapter (both at layer 17, both α=1.0, n=215 test):** does the naive map
  still beat the trained adapter at the corrected site? Primary statistic: paired McNemar
  (exact binomial on discordant pairs), pass rate + Wilson CI, Cohen's h. **This is the
  central new confirmatory number.**
- **Adapter at layer 17 vs layer 12 (Amendment 006):** does correcting the site rescue,
  worsen, or leave the negative result unchanged? Report the layer-17 adapter pass rate and
  compile-failure rate against the frozen layer-12 values.
- **Adapter/ridge vs receiver_k1 baseline:** paired McNemar against the reused receiver_k1
  result.

**Multiple-comparison discipline:** the confirmatory family for THIS run is the ridge-vs-
adapter and adapter-vs-baseline comparisons at layer 17, corrected together (Holm) as their
own family. The Amendment 006 confirmatory family is already closed and reported; this
run's family is separate and is labeled as such. The layer-17-vs-layer-12 adapter contrast
is reported descriptively (it reuses a previously-touched test set for the layer-12 side
and is therefore not a fresh confirmatory test).

## §8. Stopping rule

Run each of the 6 injection arms on all 215 test bugs once, at the frozen config. Stop.
Report the §7 readout. No iterative re-running to chase significance. Reused arms are not
re-generated.

## §9. What this run does NOT do

- Does not test a second model family (Workstream 3 — separate, future amendment).
- Does not sweep layers — layer 17 is the single pre-committed corrected site.
- Does not re-open the Amendment 006 confirmatory family (already reported).

---

*Frozen before the corrected-site run. Signed into the pre-registration chain as Amendment 008.*
