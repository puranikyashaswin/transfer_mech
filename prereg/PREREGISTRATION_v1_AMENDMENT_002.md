# Amendment 002 — Trained Adapter Intervention (Gate 1 exploratory finding)
**Date: 23 Sep 2026. Appended to frozen PREREGISTRATION_v1.md + Amendment 001. This amendment RELAXES ONE frozen choice (the fixed ridge map) in response to a pre-registered exploratory finding, and RE-PINS it with a stronger, still-frozen alternative. The exploratory/confirmatory firewall (A4) and all kill criteria are unchanged.**

> **Why this amendment exists.** Amendment A2 froze the intervention as a *fixed ridge-regression map* fit for representational alignment. Gate 1 exploratory testing (23 Sep) found this gives **0/9 causal flips** in its naive form and only **2/9** after norm-matching + strength tuning — matching the published Pythia cross-width negative result (arXiv:2606.03280) almost exactly. The transfer literature (CALM arXiv:2401.02412; Interlat arXiv:2511.09149; StateBridge; MasterKey) is unanimous that robust cross-family transfer requires a **small trained adapter optimized for receiver-side causal use**, not an offline representational fit. Continuing with the frozen ridge map would be testing a mechanism the field already knows fails. This amendment replaces the *map-learning rule*; it does NOT loosen any success criterion, control, or the firewall.

---

## B1. What changes (and what does not)

**Changed:** the alignment map in A2 §3 is no longer a fixed ridge regression. It is replaced by a **trained adapter** (B2). Everything else in A2 is unchanged: same donor layer 14, same receiver layer 12, same final-prompt-token additive injection, same frozen decoding config.

**Unchanged (still frozen):** the model pair (A1), the injection site and token span (A2), the five controls (A3), the exploratory/confirmatory firewall (A4), the power resolution (A5), and every kill criterion. **The confirmatory test still uses ONE frozen config on held-out data, one shot.** The adapter is selected and frozen in the *exploratory* track, exactly as A4 already permits for hyperparameters.

---

## B2. The trained adapter, specified exactly

**Architecture (frozen):** a single linear projection + a gated residual scaling, deliberately minimal:
```
delta = alpha * g * ( W2 · SiLU( W1 · (h_donor − d_mean) ) )   mapped into R^{2048}
```
- `W1 ∈ R^{m×1536}`, `W2 ∈ R^{2048×m}`, bottleneck `m = 512` (frozen).
- `d_mean` = donor training-set mean activation at layer 14 (centering, per Bangachev et al.).
- `g` = a learned per-output-dim gate initialized to a small value (0.1) so training starts near the no-op (identity) regime — the receiver is not disrupted at init.
- `alpha` = a fixed global strength scalar, selected on the dev split and frozen before test.
- Total added parameters ≈ 1536·512 + 512·2048 + 2048 ≈ **1.84M** (≈0.14% of the receiver) — a *small* adapter, per the literature.

**Training objective (the core fix, per the Pythia paper's recommendation #1):** the adapter is trained to **maximize the receiver's log-likelihood of the correct fixed code** when the donor's activation is injected — i.e. *receiver-side causal use*, NOT sender→receiver representational fit. Both base models are **frozen**; only the adapter parameters (W1, W2, g) receive gradients. Gradients flow through the frozen receiver's forward pass to the injection site.

**Train / dev / test split (frozen, no leakage):** the 163 HumanEval-Java bugs are split deterministically by sorted `bug_id`:
- **train:** bugs 0–79 (80 bugs) — adapter gradients + W fit
- **dev:** bugs 80–119 (40 bugs) — alpha/early-stop selection ONLY
- **test:** bugs 120–162 (43 bugs) — the held-out confirmatory set, **never touched during training or selection**

The pilot's 50 bugs (40–89) straddle train/dev; for the confirmatory Gate 2 run the test set is re-derived cleanly from the frozen split above. No test bug's activation, gradient, or pass/fail is used to fit or select anything.

**Optimization (frozen):** AdamW, lr = 1e-3, weight decay 0.01, batch = 8 bugs, ≤ 2000 steps, early stop on dev flip-rate plateau (patience 200 steps). bf16 autocast; adapter params in fp32. Fixed seed 42.

---

## B3. Why this is still a fair test of the *idea*, not overfitting

- The adapter has **1.84M params trained on 80 bugs** — heavily under-determined by design is avoided (bottleneck m=512 keeps it small), and it is selected on a **separate dev split** and frozen before the **held-out test split**. The confirmatory claim rests only on test-split causal flips vs the five controls.
- The controls (A3) are unchanged and now *stronger* in interpretation: a trained adapter that beats no-op / random-norm / shuffled / receiver-self / text-hints on held-out bugs is much harder to explain away than a ridge map.
- If the trained adapter ALSO fails to beat the sampling baseline on held-out data, that is a **strong, clean negative** for cross-family activation transfer at this scale — a publishable result either way.

---

## B4. Kill criteria (unchanged, restated for the adapter)

The Gate 2 confirmatory test kills the transfer hypothesis if the frozen trained adapter does NOT achieve, on the held-out test split: **≥60% win-share of discordant pairs AND ≥3pp gain** over the strongest control (including the receiver-sampling baseline), per A5. A Gate 1 exploratory flip rate that does not clearly beat pass@k sampling is a signal to stop before spending the confirmatory budget.

---

## Amendment log
- **001 (23 Sep):** pinned intervention, receiver, power, controls.
- **002 (23 Sep):** replaced the fixed ridge map with a small trained adapter (B2) after the Gate 1 exploratory finding that the ridge map gives ~0 causal flips; re-pinned train/dev/test split and training config; firewall + kill criteria unchanged.
