# Amendment 001 — Intervention Specification, Receiver Freeze, Power Resolution
**Date: 23 Sep 2026. Appended to frozen PREREGISTRATION_v1.md. This amendment ADDS precision; it does not loosen any frozen criterion. Where it conflicts with v1, the stricter reading wins.**

> **Why this amendment exists.** v1 said "a specified donor→receiver intervention" but never specified it. That is the same vagueness that killed DGLC (fusion code written-but-never-run; patching on a biased metric; Procrustes likely transposed). This amendment pins the mechanism, the receiver, and the statistics *before* Gate 0 runs. Anything not pinned here is a new degree of freedom and requires a new dated amendment.

---

## A1. Frozen model pair (no substitution without a new amendment)

| Role | Model | HF id | hidden_size | layers | attn heads | KV heads | vocab |
|---|---|---|---|---|---|---|---|
| **Donor** | Qwen2.5-Coder-1.5B-Instruct | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1536 | 28 | 12 | 2 (GQA) | 151936 |
| **Receiver (primary, cross-family)** | DeepSeek-Coder-1.3B-Instruct | `deepseek-ai/deepseek-coder-1.3b-instruct` | 2048 | 24 | 16 | 16 (MHA) | 32256 |
| **Receiver (control, same-family)** | Qwen2.5-Coder-3B-Instruct | `Qwen/Qwen2.5-Coder-3B-Instruct` | 2048 | 36 | 16 | 2 (GQA) | 151936 |

- **Why DeepSeek-Coder-1.3B as primary receiver:** Llama-architecture (RMSNorm, SwiGLU, RoPE), 24 layers, hidden 2048, **full MHA (16 KV heads)** — no GQA complications in the residual stream. Cross-family vs Qwen donor is the scientific point (different tokenizer, different pretraining).
- **Why NOT CodeLlama-7B:** hidden 4096, 32 layers — a 4× larger receiver confounds "transfer" with "receiver capacity." 1.3B receiver ≈ 1.5B donor keeps the comparison honest. CodeLlama-7B is recorded here as the *rejected* alternative so it cannot be silently swapped in later.
- **Same-family control** Qwen 1.5B→3B: shares tokenizer + architecture with donor; expected to transfer *more easily* — if cross-family works but same-family doesn't (or vice versa) that is itself a finding.
- All three fit comfortably on the L40S 48GB simultaneously in bf16 (1.5B≈3GB, 1.3B≈2.6GB, 3B≈6GB weights).

---

## A2. The intervention, specified exactly (H1's "specified intervention")

**What is transferred:** the donor's **residual-stream activation** (post-block, pre-next-attention) at a **single donor layer**, linearly mapped into the receiver's residual stream at a **single receiver layer**, **added** to the receiver's stream at a defined token span.

**Concretely, per bug:**
1. Run the donor on the bug prompt; cache the residual stream at **donor layer `L_d` = 14** (of 28 — the midpoint; see A4 for the sweep that is *separate* from the confirmatory claim).
2. Run the receiver on the same prompt; at **receiver layer `L_r` = 12** (of 24 — midpoint), **add** the mapped donor vector to the residual stream at the **final prompt token position only** (the position that conditions the first generated token).
3. **Alignment map:** a single affine map `W ∈ R^{2048×1536}, b ∈ R^{2048}`, `h_recv += W·h_donor + b`. `W, b` are fit by **ridge regression** (λ=1.0) from donor-layer-14 activations to receiver-layer-12 activations on the **TRAIN split only**, over the final-prompt-token positions of train bugs. This is the *only* learned component; it is frozen before dev/test.
4. Generation proceeds from the patched state with the frozen decoding config (temp=0.2, top_p=0.95, seed=42, max_new_tokens=512).

**Token-span choice (frozen):** final-prompt-token only. Rationale: it is the minimal, most-controlled injection (single vector), avoids sequence-length mismatch between the two tokenizers, and is the position with the most direct causal path to the first generated token. Multi-token and per-generated-token variants are **exploratory only** (A4).

**Why "add" not "replace":** additive is the minimal perturbation that preserves the receiver's own computation; replace is a stronger, more destructive intervention reserved for the corrupt/restore mechanism test at Gate 3.

---

## A3. Controls, now mechanically pinned (Gate 2)

Each control is identical to A2 except for the listed change. All run on the same bugs, same decoding, same seed.
- **no-op hook:** patch applied with `W=0, b=0` (receiver unchanged). Must equal always-receiver.
- **shuffled donor:** donor activation taken from a *different* bug's prompt (paired permutation, fixed seed). Tests content-specificity.
- **norm-matched random vectors:** `h_donor` replaced by a Gaussian vector with the same L2 norm as the real donor activation. Tests "is it just adding magnitude."
- **receiver-self:** donor activation replaced by the receiver's *own* layer-12 activation at the same position (identity through W). Tests "is it just perturbation."
- **text hints:** the donor's *generated text* (its attempted fix) appended to the receiver prompt as a comment; no hidden-state transfer. **The key information-matched control** — if hidden-state transfer beats giving the receiver the donor's answer as text, the benefit is sub-symbolic.

**Pass bar (frozen):** transfer must beat **every** control on held-out, paired McNemar exact, one-sided α=0.05, **and** show an effect size ≥ the pre-registered minimum (A5).

---

## A4. Exploratory vs confirmatory (the multiple-comparisons firewall)

- **CONFIRMATORY (the paper's claims):** the single configuration in A2 (L_d=14, L_r=12, final-token, additive, ridge-W), evaluated **once** on the held-out TEST split.
- **EXPLORATORY (hypothesis-generating, NOT claims):** layer sweeps (L_d ∈ {7,14,21}, L_r ∈ {6,12,18}), token-span variants, replace-vs-add, per-token injection. These run on **DEV only**. Any exploratory finding enters the paper as a claim **only if** re-confirmed on TEST with a fresh prereg amendment.

---

## A5. Power resolution (fixes the v1 contradiction)

v1 assumed discordance ~0.2 and cited ~1,700 examples for a 3pp gain — but named HumanEval-Java (163) as the benchmark. 163 bugs can only detect a *very large* effect (win-share ≥67–75% of discordant pairs). Resolved:

- **DEV (tuning + exploratory):** the contaminated 163-bug HumanEval-Java set (already inspected; dev-only per v1).
- **CONFIRMATORY TEST (the claims):** **MultiPL-E Java** (translation of HumanEval-Python's 164 problems) **+ the Defects4J single-method subset**, pooled and split by function family into a held-out test set of **target n ≥ 600 independent problems**. If MultiPL-E-Java + D4J cannot reach 600 usable problems, we expand with **HumanEval-Java's held-out half + MBPP-Java (MultiPL-E)** until n≥600.
- **Minimum detectable effect we commit to claiming:** a paired win-share of **≥60% of discordant pairs** (moderate effect). At discordance 0.15 this needs ~1,314 problems; at 0.20, ~985. **We will size the final test set from the Gate-1 pilot's measured discordance** (pilot n≈50) and report that number in a dated amendment before the confirmatory run. If we cannot reach the required n, **we do not claim the effect** — we report the confidence interval and stop.
- **Effect-size floor:** we claim transfer only if the paired gain is **≥3pp AND** statistically significant. Smaller significant effects are reported as "statistically detectable, practically negligible."

---

## A6. Gate 0 evaluator controls (must all pass before any GPU experiment)

Gate 0 is a *measurement* gate and needs **no GPU** — it validates the JUnit harness. Frozen checks:
1. **Known-valid passes:** the dataset's own `correct/` reference solutions must **all** compile and pass their JUnit tests (163/163 on HumanEval-Java). Any failure = harness bug, fix before proceeding.
2. **Known-invalid fails:** the dataset's `buggy/` solutions must **all fail** their JUnit tests (a buggy solution that passes = bad test or mislabeled bug). Expected: near-0 pass; any pass is investigated and either the bug is dropped or the test fixed, and the decision logged.
3. **No-op identity:** re-running the same solution twice gives identical pass/fail (determinism check; catches flaky timeouts).
4. **Independent recompute:** the headline accuracy table is recomputed from raw per-bug JSONL by a *separate* script, and must match the harness's own summary exactly.

**Implementation:** `transfer_mech/eval_harness/` (see `GATE0_RUNBOOK.md`). Runs on EC2 (or any machine with a JDK). **Not run on the MacBook** per hardware policy.

---

## Amendment log
- **001 (23 Sep 2026):** Added A1 (frozen models), A2 (exact intervention), A3 (pinned controls), A4 (exploratory/confirmatory firewall), A5 (power resolution + n≥600 + effect floor), A6 (Gate 0 evaluator controls). No frozen criterion loosened.
