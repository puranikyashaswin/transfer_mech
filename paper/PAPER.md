# Where Does Activation Transfer Break? A Pre-Registered, Execution-Verified Negative Result for Single-Vector Transfer in Program Repair

**Pre-registered, execution-verified causal study. All claims recomputed from raw logs.**

---

## Abstract

Cross-model activation transfer — steering a *receiver* model with hidden states from a *donor* model — has produced striking results in model composition, KV-cache fusion, and latent multi-agent communication. But these successes share a hidden common factor: they inject information through **multi-token, all-layer, or attention-based** mechanisms, and most *train* a connector. We ask a sharper question: **what is the minimal intervention that actually transfers task-relevant information, and where exactly does the boundary between "works" and "doesn't" lie?** We pre-register and test the simplest possible mechanism — a linearly-mapped residual vector, injected once into the receiver — on a concrete downstream task (program repair, HumanEval-Java → MultiPL-E-Java), where "transfer" has a crisp execution-verified meaning: the receiver passes a test suite it previously failed.

Our headline result is a **pre-registered, one-shot confirmatory negative** on a rebuilt, execution-verified MultiPL-E-Java corpus (n=335; held-out test n=215; Amendment 006). The field's recommended connector — a small bottleneck adapter *trained* for receiver-side causal use — is **significantly worse than doing nothing**: 51.6% vs the receiver's 63.7% (paired effect −12.1pp, 95% CI [−18.3,−4.6]; McNemar p=0.0013), and it fails to compile **10× more often** (22.3% vs 2.3%). Under an explicit honest-n pre-commitment (achievable n below the pilot-powered n=859–1,411), we claim no *positive* transfer effect, but report the adapter arm as a clean significant negative. We then diagnose *why*. A javac-verified error taxonomy (§12.5) shows a distinctive adapter-specific signature (46% extra-close-brace, 27% illegal-package — absent from the receiver and text-hint arms and at most 4% under a random vector; only the shuffled-donor control shows them, at 25% and 22%); a strength×subspace pilot (§12.6, dev, formally CANNOT-TELL per its pre-committed boundaries) shows ~85% of the adapter's injected energy lies *outside* the receiver's active subspace; and a baseline-map diagnostic (§12.7, dev, exploratory) shows the failure is **specific to the trained adapter's architecture, not injection-general**: a zero-task-training linear ridge map through the *identical* channel matches the receiver baseline (65.0%) and beats the trained adapter 13–1 (p=0.0018). A corrected-site confirmatory re-run (Amendment 008) later tests both maps head-to-head on the *same* held-out test split (n=215, receiver layer 17, both maps retrained at that site): there, ridge and adapter are statistically indistinguishable (0.605 vs 0.595, McNemar p=0.890, 95% CI on Δ [−5.7,+7.5]pp), and both sit at or below the unmodified receiver. The dev-split 13–1 advantage was a **split confound** — it does not survive to confirmatory scale (§12.8) — and we report this as the superseding, resolved finding: neither map architecture we tested extracts a usable repair signal from this channel, which is consistent with a channel-limited account rather than a map-training account.

On the exploratory HumanEval-Java split (n=43–73) we additionally map the *boundary*: the canonical single-layer→single-token cell never beats a matched random-perturbation control, and a full control set (Amendment 003) gives a **dissociation** — the donor's fix as natural-language *text* lifts the receiver from 26.0% to 47.9%, while the same information as an injected vector does not. A pre-registered **2×2 site sweep** (Amendment 004) finds exactly one nominally-positive cell — **all donor layers → a single receiver token** (+15.1pp, p=0.043; **not** Holm-surviving across 4 cells, so suggestive only) — while all-token injection collapses to zero. Our pre-registered monotone "more channel is better" prediction was **falsified**: the working channel is wide on the donor side, surgically narrow on the receiver side. We release the full pre-registration, eight amendments, all raw execution-verified logs, and a one-command reproduction.

---

## 1. Introduction

A growing body of work shows that the internal representations of one language model can be *transferred* into another to change its behavior: composing a weak model's context into a strong model via cross-attention (CALM), fusing KV-caches across models (C2C), communicating latent vectors between agents (LatentMAS, Interlat), and steering one model with another's residual stream via linear maps (MasterKey, StateBridge). If representations transfer reliably, this is a powerful primitive: small specialist models could inject skills into large generalists, and mechanistic-interpretability findings made in one model could be *ported* to another.

But read closely, these successes share a structural property that is easy to miss. They inject information through **rich, distributed channels**: cross-attention over *all* donor tokens at *every* layer (CALM), per-layer KV-cache fusion (C2C), or dense per-position projections refreshed at every generation step (LatentMAS). The mechanism that would be most useful for interpretability and cheap composition — **a single residual vector, mapped once, added at one token** — is, to our knowledge, untested as a *causal* intervention on a *downstream* task with *execution-verified* success.

This paper tests exactly that minimal mechanism, and does so adversarially. We pre-register the intervention, the model pair, the controls, and the kill criteria *before* running. We choose a task — program repair — where transfer has an unambiguous causal signature: the receiver fails a bug on its own, and passes it only when the donor's representation is injected. We then ask not "does the map align the representations offline?" (a probe-accuracy question the literature has shown is misleading) but "**does injecting the donor's state cause the receiver to pass a test it otherwise fails, more than a matched control does?**"

The headline configuration fails — a clean, robust **no** — but a pre-registered sweep of the intervention site reveals the failure is *not* uniform. By sweeping the donor source granularity and the injection width, while holding a per-cell matched random-perturbation control constant, we show the minimal mechanism sits at a **sharp, non-monotone boundary**: the only configuration where donor content trends decisively above noise is *all donor layers mapped into a single receiver token*. Single-layer→single-token injection produces only a content-independent perturbation; widening the donor read-out while narrowing the receiver injection is what crosses the threshold.

**Contributions.**

1. **A pre-registered, execution-verified causal test** of single-vector residual transfer for program repair, with five mechanically-pinned controls and a frozen exploratory/confirmatory firewall (§3).
2. **A robust exploratory negative result** for the canonical single-layer→single-token cell across three map-learning methods, two transfer directions, and three injection strengths (§5–§6, §10), plus a *perturbation-effect* decomposition ruling out family boundary, norm mismatch, and underfit map (§6).
3. **A dissociation** (§7): the donor's fix transfers through natural language (47.9%) but not the single-vector channel (31.5% = random), localizing the binding constraint to the channel, not to the receiver's ability to use the information.
4. **A pre-registered 2×2 site sweep revealing a non-monotone transfer boundary** (§10): content transfer trends above noise only for rich donor read-out + single-token receiver injection, falsifying the monotone "more channel is better" prediction (Amendment 004).
5. **A one-shot pre-registered confirmatory negative** (§12): on a rebuilt MultiPL-E-Java corpus (n=215 held-out test), the trained adapter is *significantly worse* than the receiver alone (−12.1pp, paired McNemar p=0.0013) and fails 10× more often at compile (22.3% vs 2.3%).
6. **A mechanistic diagnosis of that failure, resolved at confirmatory scale** (§12.5–§12.8): a javac-verified error taxonomy and a strength×subspace pilot show the trained bottleneck adapter routes ~85% of its energy off the receiver's active subspace (§12.5–§12.6, CANNOT-TELL on strength vs. subspace). A dev-only baseline-map comparison initially suggested a naive zero-training linear map beats the adapter 13–1 (§12.7) — but a corrected-site confirmatory re-run (Amendment 008, §12.8) tests both maps on the same held-out test split and finds them statistically indistinguishable (p=0.890): the dev-split advantage was a split confound. Neither map architecture tested extracts a usable repair signal at confirmatory scale, consistent with a channel-limited account.

**Pre-registration as methodology.** We treat pre-registration not as a compliance checkbox but as the load-bearing structure of the work, because the substantive claim here is a *boundary* — and a boundary claim is only credible if the analysis was not fit to the data. Every design decision with claim-altering force was frozen in a dated document *before* its corresponding run: the intervention, the model pair, the five controls, the power analysis, and four explicit *kill gates* (any of which was pre-committed to stop the project and report the negative). When the data pushed back, we amended openly rather than silently: the amendment trail (§4, in-line) records each pivot — an honest re-scoping when a split proved underpowered, a sweep plan frozen before its results were seen, an honest-n pre-commitment that *removed* the headline claim when the achievable n fell short. The confirmatory firewall was absolute: the held-out MultiPL-E test split was touched exactly once, under a frozen config; all confirmatory-corpus piloting happened on a separate dev split; the subspace basis was fit on a third train split. (Two exploratory-phase exceptions, disclosed: Gate 0 validated the *measurement harness* on all 163 HumanEval-Java bugs including that split's exploratory test set — a measurement-soundness check, not a fit — and one exploratory α substitution, 0.3→0.5, made and disclosed in §6 rather than by amendment. Neither touches the confirmatory test split.) The result is that every number we report carries its epistemic status on its face — confirmatory, exploratory, or suggestive — and the negative result cannot be explained away as post-hoc rationalization, because the criterion that produced it was frozen in advance. We flag this as a contribution in itself: a template for how to *fail rigorously* in empirical interpretability.

---

## 2. Related Work

**Cross-model composition & latent communication.** CALM composes a frozen "augmenting" model into a frozen "anchor" model via a trained cross-attention over the augmenter's *full* output at every layer. C2C fuses KV-caches between models so a long-context model can lend its context to a stronger reasoner. LatentMAS and Interlat let agents exchange *sequences* of latent vectors, refreshed each step. HeteroFusion aligns models by functional module role in weight space. **Common factor:** all inject through *many positions and/or many layers*, and most *train* a connector. Our work isolates the minimal single-vector case these richer mechanisms generalize.

**Linear / training-free steering & transfer.** MasterKey transfers residual "keys" between models with a low-rank SVD map on shared prompts; StateBridge uses orthogonal Procrustes on centered+whitened states with careful norm calibration. Both report positive transfer — but both operate with **per-layer, multi-position** injection and extensive norm/subspace calibration, and MasterKey grid-searches hyperparameters per model pair. Our result is consistent with theirs: we test the *degenerate* single-vector, single-layer, single-token version and find it insufficient.

**Warnings from interpretability.** Several papers caution that representational similarity does not imply causal usability. Gao et al. show CKA-style alignment on *contrastive-difference* vectors transfers, while raw-state transfer is near chance. Usama et al. find high representational convergence without downstream understanding. Mehta et al. show commitment ≠ correctness. Closest to us, a cross-width Pythia study reports high offline alignment (R², cosine) between models but **zero** downstream effect of naive cross-width transfer — our naive-ridge cross-family arm reproduces this exactly (0/9), and we extend it with norm-matching, a trained adapter, same-family controls, and a perturbation decomposition.

**Single-vector activation steering (and why it does not contradict us).** A well-known family of methods *does* steer models with a single additive vector at a single layer: Activation Addition / ActAdd (Turner et al., 2023), Contrastive Activation Addition (Rimsky et al., ACL 2024), Representation Engineering (Zou et al., 2023), latent steering vectors (Subramani et al., Findings of ACL 2022), and Inference-Time Intervention (Li et al., NeurIPS 2023). Two distinctions matter. **(1) Within-model vs cross-model.** Every one of these is a *self-steer*: the vector is extracted from and applied to the *same* model's residual stream. None demonstrates a donor→receiver cross-model transfer — the setting we test. **(2) Behavioral attribute vs execution-verified task.** They steer *behavioral/style* attributes (sentiment, honesty, toxicity, refusal) scored by probes, classifiers, or ratings; the closest to an objective metric is ITI's TruthfulQA accuracy, which is still a *propensity*, not a task whose success is externally verified by execution. None shows a single vector, single layer, injected once, improving an execution-verified downstream task such as program repair. The existence of within-model, behavior-level steering successes therefore does **not** contradict our finding: we test the harder setting — cross-model transfer scored by an execution-verified downstream task — and there the minimal single-vector mechanism fails to beat a matched perturbation.

**Program repair.** Automated program repair (APR) with LLMs is typically single-model. Our transfer framing — a donor model that "knows" the fix lending its representation to a receiver that doesn't — connects APR to model composition and provides a rare *execution-verified* transfer metric.

---

## 3. Pre-Registered Method

All hypotheses, the intervention, the model pair, the controls, the power analysis, and the kill criteria were frozen in `PREREGISTRATION_v1.md` **before** any experiment, then tightened by seven dated amendments (001: intervention/receiver/power; 002: trained adapter; 003: honest re-scoping + controls restated + confirmatory reaffirmed; 004: 2×2 injection-site sweep analysis plan, frozen before results; 005: max_new_tokens=1024; 006: confirmatory sizing + honest-n pre-commitment; 007: instrumented strength×subspace pilot, dev-only exploratory). Exploratory findings never loosen a frozen criterion. The exploratory/confirmatory firewall is strict: any tuning happens on a dev split; the confirmatory claim uses one frozen config on a held-out test split, once. (The full dated amendment trail is summarized in-line in §4.)

### 3.1 Task and metric

**Task.** Program repair on HumanEval-Java (163 bugs). Given a buggy Java method, the model must produce a fixed method that passes the bug's hidden JUnit test suite.

**Metric.** *Execution-verified pass.* A generation passes iff it compiles and passes the full test suite (JUnit4, 3s/test timeout). No string-match, no human judgment. "Transfer" is claimed only on **discordant** bugs — bugs the donor passes but the receiver fails on its own — where a causal flip is possible.

### 3.2 The frozen intervention (Amendment 001)

- **Donor:** Qwen2.5-Coder-1.5B-Instruct (hidden 1536, 28 layers, GQA).
- **Receivers:** DeepSeek-Coder-1.3B-Instruct (**cross-family**; hidden 2048, 24 layers, MHA, Llama arch, vocab 32256) and Qwen2.5-Coder-3B-Instruct (**same-family control**; hidden 2048, 36 layers, shares donor tokenizer/vocab 151936).
- **What is transferred:** the donor's post-block residual activation at **donor layer 14** (midpoint), at the **final prompt token** only.
- **How:** mapped by a linear/affine map into the receiver's residual stream at **receiver layer 12** (DeepSeek) or **17** (Qwen-3B midpoint), and **added** to the receiver's stream at the final prompt token. Additive (not replacement) is the minimal perturbation preserving the receiver's own computation. **Why the midpoint:** we chose the midpoint layer as a principled default — early layers encode surface/lexical features, late layers are committed to the model's own output distribution, and the residual midpoint is where prior work (MasterKey, StateBridge, cross-width Pythia) localizes the most transferable, still-abstract task content. We did **not** sweep receiver layer for the confirmatory claim (a single frozen site is the pre-registered design); a receiver-layer sweep is the most direct follow-up, and we flag it in §13. (On the layer-12-vs-17 site question for the Qwen-3B confirmatory receiver, see §12.3's explicit note.)
- **Decoding (frozen):** temperature 0.2, top-p 0.95, seed 42, max 512 new tokens.

### 3.3 Four map-learning methods (the "how is the map fit?" axis)

1. **Naive ridge (Amendment 001).** Ridge regression (λ=1.0) from donor-layer-14 to receiver-layer activations over final-prompt-token positions of the **train split only**. Offline representational fit.
2. **Norm-matched ridge.** Centered ridge (subtract per-model means) whose output is **renormalized to the receiver's native residual norm** (per StateBridge/Bangachev), with a global strength α.
3. **Trained adapter (Amendment 002).** A small bottlenecked adapter `g·W2·SiLU(W1·(h−d_mean))` (~1.84M params, bottleneck 512) trained for **receiver-side causal use**: it maximizes the frozen receiver's log-likelihood of the correct fixed code when the donor state is injected. Both base models frozen; only the adapter learns. This directly tests the literature's top recommendation for fixing offline-fit transfer — and is the map the confirmatory study (§12) froze. *Training detail:* trained on the **train split only** (HumanEval-Java 0–79 for exploratory; retrained on the rebuilt 80-bug MultiPL-E train split for the confirmatory run, converged at g_mean=0.143), AdamW, gradients flowing through the frozen receiver's forward pass to the injection site, early-stopped on dev log-likelihood; full training curve and hyperparameters in `eval_harness/` and Appendix C.
4. **Baseline linear map (diagnostic, §12.7).** A plain centered ridge (λ=1.0, no nonlinearity, no bottleneck, no task training) emitting the additive delta directly, fit on train and evaluated on dev under the exact confirmatory decoding config. This is the reference against which the adapter's architecture is isolated. *Caveat we state throughout:* because it differs from the adapter in architecture, training, *and* output magnitude at once, a difference between them is **not** cleanly attributable to training alone — it isolates "the trained-adapter package" from "a conservative linear map," and the norm-response follow-up (§13) is needed to separate the magnitude axis.

### 3.4 Five mechanically-pinned controls (Amendment 001)

For every transferred bug we run, under identical decoding: **(C1) no-op** receiver baseline; **(C2) random-norm**: a random vector of the *same norm*; **(C3) shuffled**: the map applied to a *different* bug's donor state; **(C4) self-inject**: the receiver's *own* hidden state re-injected; **(C5) receiver-sampling**: the receiver's pass@k under the frozen sampler (the "killer baseline" that pure sampling must be beaten). C2 is the control that turns out to matter most.

### 3.5 Splits, leakage, and reproducibility

Two corpora, two split schemes — kept distinct throughout. **Confirmatory corpus (MultiPL-E-Java, n=335; the §12 study):** deterministic split by sorted `bug_id` — train 0–79, dev 80–119, test 120+ (215 held-out). No confirmatory test bug's activation, gradient, or pass/fail is used to fit or select anything. **Exploratory corpus (HumanEval-Java, n=163; the §5–§10 studies) uses its own earlier split.** The **cross-family pilot** (§5) ran on 50 exploratory bugs indexed 40–89 of *that* corpus — a window chosen to be disjoint from the exploratory adapter-train range (exploratory bugs 0–39); "disjoint" refers to the exploratory split, not the confirmatory one. Fixed seeds; every run writes per-bug JSONL checkpoints (resumable); results mirrored to S3 same-day; all numbers in this paper recomputed from raw logs.

---

## 4. Experimental Setup

All runs on a single NVIDIA A10G (24GB), bf16. Donor and receiver loaded simultaneously (~9GB weights). Eval harness compiles and runs JUnit in isolated temp dirs. Gate 0 validated the harness on all 163 bugs (163/163 known-valid pass; 0 unexpected known-invalid passes; 0 no-op non-determinism) — so a "pass" is a real pass and the measurement itself is trustworthy.

**The confirmatory arm set (Amendment 006 §4).** Each test bug is run under all 8 arms, identical decoding (temp 0.2, top-p 0.95, seed 42, 1024 new tokens), execution-verified by JUnit in isolated default-package compilations:

| # | Arm | What is injected at receiver layer 12 | What it controls for |
|---|-----|----------------------------------------|----------------------|
| 1 | receiver_k1 | nothing (receiver alone, k=1) | baseline capability |
| 2 | **adapter** | donor L14 state via trained adapter | the intervention |
| 3 | random_norm | random vector, same norm (R=5, per-bug-seeded) | perturbation-specificity (C2) |
| 4 | shuffled | adapter applied to a *different* bug's donor state | content-specificity |
| 5 | self_inject | receiver's own hidden state re-injected | injection-mechanics artifact |
| 6 | unrelated_task | donor state from an unrelated task (C5) | task-relevance |
| 7 | text_hint | donor's fix as natural-language text | positive control (channel-independent) |
| 8 | donor_k1 | donor alone, k=1 | capability reference / ceiling |

**Amendment trail (frozen, dated, in-repo).** The pre-registration evolved through eight amendments, each frozen before its corresponding run: **001** froze the intervention (donor L14 → receiver L12, final token, additive), the model pair, and the power analysis; **002** added the trained bottleneck adapter; **003** re-scoped honestly (the exploratory split is underpowered) and restated the controls; **004** froze the 2×2 site-sweep analysis plan *before* the sweep was run; **005** froze max_new_tokens=1024 for every arm; **006** discharged the deferred sizing mandate with the honest-n pre-commitment (report effect + CI, claim nothing if n < powered n); **007** froze the instrumented strength×subspace pilot as dev-only and exploratory; **008** corrected the receiver injection site (layer 12 → 17), added the naive ridge map as a confirmatory arm, and re-ran the full family on held-out test in a single frozen shot. The full text of each is in `PREREGISTRATION_v1.md` + `amendments/`.

### 4.1 Status of the four pre-registered kill gates

The pre-registration defines four gates, each able to kill the project honestly. We report their status against the runs in this paper.

| Gate | Pre-registered test | Status in this paper |
|---|---|---|
| **0. Measurement** | Known-valid passes, known-invalid fails, no-op identity, independent recompute | **PASS.** 163/163 valid pass, 0 unexpected invalid, 0 no-op non-determinism (§4). Headline table independently recomputed from raw logs (`make_stats.py`, `make_tables.py`). |
| **1. Diversity** | Donor transfer must add beyond extra receiver sampling at matched compute | **FAIL (exploratory).** Pass@5 sampling alone lifts the receiver +20pp (§5); transfer matches but never exceeds a matched random perturbation on the headline cell. Per the frozen rule this stops the confirmatory transfer claim; we report the negative. |
| **2. Transfer** | Transfer must beat ALL controls (no-op, shuffled, random, self, text-hint) on held-out | **FAIL on the headline cell** (transfer = random; text-hint ≫ transfer, §7); **PARTIAL/suggestive on `sall_t1`** (beats its random control nominally, fails the 4-cell Holm, §10). No confirmatory claim entered. |
| **3. Mechanism+gate** | Reproducible localization OR predictive edge | **MET (mechanism).** The 2×2 sweep + mechanistic probe give a reproducible, pre-registered localization: content transfer trends above noise only for rich donor read-out + single-token receiver injection; the injected single-layer content never reaches the output differently from the receiver's own state (§9–§10). Per the frozen rule ("publish Gates 1–2 alone" if this fails but a mechanism is found), we publish the boundary-localization result. |

The honest outcome of the frozen gates is exactly what this paper reports: the minimal single-vector mechanism does **not** pass the transfer gate at the headline site, and the contribution is the localized, control-separated boundary of *where* it begins to.

---

## 5. Result I — Cross-Family Transfer (Qwen → DeepSeek)

**Pilot (50 bugs, k=5).** Receiver pass@1 = 34%, pass@5 = 54% (**+20pp from sampling alone** — a strong killer baseline). Donor pass@1 = 38%. 9 discordant bugs.

**Naive ridge (frozen config): 0/9 discordant flips.** This exactly reproduces the cross-width Pythia negative result.

**Norm-matched ridge + strength sweep:** at most **2/9** discordant flips (FIZZ_BUZZ at α∈{0.1,1.0}, FILE_NAME_CHECK at α=1.0) — weak and config-dependent.

**Trained adapter (held-out test, 43 bugs, 5 arms):**

| Arm | Pass | Acc |
|---|---|---|
| random-norm (C2) | 11 | **25.6%** |
| self-inject (C4) | 11 | **25.6%** |
| receiver no-op (C1) | 9 | 20.9% |
| shuffled (C3) | 6 | 14.0% |
| **trained adapter** | 4 | **9.3%** |

The adapter flips 2 bugs but **breaks 7** the baseline already passed — net negative, and *worse than random noise*. Training converged on the train split (train NLL 0.26→0.0003; note low train NLL indicates convergence, not generalization — the relevant evidence is the held-out result): a map trained end-to-end for causal use still fails to beat the perturbation control.

---

## 6. Result II — Same-Family (Qwen → Qwen-3B) and the Perturbation Effect

If the cross-family failure were a tokenization/pretraining artifact, same-family transfer (shared tokenizer, shared architecture) should work. It does not — and the controls reveal *why* nothing works.

**Pilot (73 held-out bugs, norm-matched ridge, α=1.0):**

| Arm | Pass | Acc [95% Wilson CI] |
|---|---|---|
| transfer | 23 | 31.5% [22.0, 42.9] |
| **random-norm (C2)** | 23 | **31.5% [22.0, 42.9]** |
| self-inject (C4) | 22 | 30.1% [20.8, 41.4] |
| receiver no-op (C1) | 19 | 26.0% [17.3, 37.1] |

Transfer and its matched random control are an exact tie (identical pass counts, identical CIs); the paired McNemar is 5–5 discordant (p=1.0).

Transfer flips 7 bugs vs baseline — but random-norm flips 8, and only **1 of the 7** transfer flips is donor-specific (SORT_ARRAY: passes under transfer, fails under both random and self). A further 3 transfer flips are not flipped by random (SORT_ARRAY, STRING_SEQUENCE, STRONGEST_EXTENSION), but 2 of those *are* flipped by self-inject. The content of the injected vector is, at best, responsible for a single bug.

**Alpha sweep (α ∈ {0.1, 0.5, 1.0})** confirms the mechanism and rules out "wrong strength". *Per-cell n:* α=0.1 and 0.5 ran on 69 bugs, α=1.0 on 68 bugs; 5 pilot bugs are absent from the α=1.0 sweep (VOWELS_COUNT, WILL_IT_FLY, WORDS_IN_SENTENCE, WORDS_STRINGS, X_OR_Y), so the α=1.0 sweep rate (0.338, n=68) is *not* directly comparable to the pilot α=1.0 rate (0.315, n=73) — they are the same config on slightly different subsets, and we report both rather than reconcile them silently. The pre-registered exploratory α was 0.3; α=0.5 was a substitution, disclosed here.

| α | n | transfer | random-norm | self-inject |
|---|---|---|---|---|
| 0.1 | 69 | 0.275 | 0.246 | **0.304** |
| 0.5 | 69 | 0.304 | 0.304 | **0.319** |
| 1.0 | 68 | 0.338 | 0.338 | 0.309 |

At no strength does transfer exceed the same-strength random control; at α=0.5 and 1.0 they tie exactly. At **α=0.1, self-inject (0.304) actually exceeds transfer (0.275)** — an inversion that is itself evidence content doesn't matter. The pass rate broadly rises with injection magnitude, but the clean "identical for content and random" dose-response holds only at α≥0.5; at the weakest strength the self-injection (the receiver's own state) is the most effective perturbation. The overall picture is a **perturbation effect**: a norm-matched vector nudges the receiver off a failing trajectory, and what the vector encodes is largely irrelevant.

---

## 7. Result IIb — The Full Control Set: a Positive Control that Separates "Can't Use" from "Channel Doesn't Deliver"

The pilot above left two pre-registered controls un-run (Amendment 001 C5 unrelated-task, and the text-hint control). We ran the complete control set on all 73 held-out bugs (Amendment 003 §2), with random vectors now per-bug-seeded and drawn R=5 times each. Pass rates (95% Wilson CI, n=73):

| Arm | Pass | Acc [95% CI] | Reading |
|---|---|---|---|
| receiver no-op (C1) | 19 | 0.260 [0.173, 0.371] | baseline |
| donor (Qwen-1.5B) self | 17 | 0.233 [0.151, 0.342] | donor's own pass rate |
| **unrelated-task (C5)** | 19 | 0.260 [0.173, 0.371] | donor state from a *different* bug |
| self-inject (C4) | 22 | 0.301 [0.208, 0.414] | receiver's own state |
| random-norm (C2) ×5 | — | 0.304 ± 0.022 | per-bug-seeded, 5 draws |
| **transfer** | 23 | 0.315 [0.220, 0.429] | mapped donor state |
| **text-hint** | 35 | **0.479 [0.369, 0.592]** | donor's fix shown as **text** in the prompt |

Three findings, each tightening the negative:

**(1) Transfer is statistically indistinguishable from every content-free injection.** Transfer vs self-inject: McNemar exact p=1.000 (3 vs 2 discordant). Transfer vs majority-vote random: p=0.774. Transfer vs unrelated-task: p=0.219. The single significant paired comparison is **text-hint ≫ transfer: McNemar p=0.0227 (18 vs 6 discordant)**.

**(2) The dissociation is the result.** Showing the donor's fix as *text* lifts the receiver from 26% to 48% — the **highest** of any arm — and beats transfer significantly (p=0.023). The *same* information, injected as a single hidden-state vector, produces no lift over random noise. And an unrelated-task donor state (0.260) does *no better than baseline*, ruling out "any donor state helps." So: the receiver **can** use the information (text-hint proves it), the donor **has** the information (donor passes these bugs), but the **single-vector channel does not deliver it**. The binding constraint is the channel, not the receiver's competence and not the map's content.

**(3) On the discordant subset (donor passes, receiver fails; n=8) the pattern holds.** text-hint rescues 5/8 (62.5%); transfer 3/8 (37.5%); self-inject and unrelated-task 2/8 (25%); random mean 1/8 (12.5%). Only **PARSE_NESTED_PARENS** passes under transfer *and* self-inject *and* text-hint (ambiguous attribution), and no bug is rescued by transfer alone that random never touches. The discordant analysis is low-n (8) and reported as exploratory.

This converts the negative from "we couldn't detect transfer" into a controlled claim: **information about the fix transfers readily through the natural-language channel and not at all through the single-vector hidden-state channel, in this cell.** It directly answers the strongest control the pre-registration demanded.

---

## 8. Result III — Ruling Out the Alternative Explanations

The negative is robust to the three most plausible rescue attempts:

- **Not a family/tokenizer boundary.** Same-family (shared tokenizer + architecture) also fails to beat its matched control, just as cross-family does (§6 vs §5).
- **Not a norm mismatch.** Norm-matching to the receiver's native norm (64.17 for Qwen-3B; 21.66 for DeepSeek) was applied; it changed the *rate* of flips but never made content beat noise (§6).
- **Not an underfit map.** An end-to-end-trained causal adapter — the literature's recommended fix — does *worse* than random (§5), and a per-pair strength grid-search (per MasterKey) finds no winning α (§6).

What remains is the parsimonious explanation: **a single additively-injected residual vector at one token does not carry recoverable downstream-task information.** The receiver's subsequent computation does not "read" the injected content as the fix; it only registers that *something* perturbed the state.

---

## 9. Result IV — A Mechanistic Probe: the Injected Content Never Reaches the Output

To close the loop we ask, mechanistically: *does the injected vector actually change what the receiver computes, and does it matter whether the vector carries donor content?* We run a forward-only causal check on 30 same-family bugs: inject the delta at receiver layer 17, then read the receiver's *final-layer* residual stream and *next-token logits* at the final prompt token, and compare to the no-injection baseline.

| Arm | cos(h, h₀) | norm ratio | KL(p₀‖p) | argmax changed |
|---|---|---|---|---|
| transfer | 0.965 | 1.047 | 0.203 | **0.00** |
| self-inject | 0.965 | 1.048 | 0.201 | **0.00** |
| random-norm | 0.947 | 1.022 | 0.329 | 0.53 |

Three findings stand out. **(1) Transfer ≡ self-inject.** On every mechanistic metric, injecting the mapped donor state is statistically identical to re-injecting the receiver's *own* state (cosine 0.965 vs 0.965, KL 0.203 vs 0.201) — the donor content is invisible downstream. **(2) The next-token argmax is essentially unmoved.** Neither transfer nor self-inject changes the argmax next token (0/30; 95% CI ≤ 11.6%). The downstream behavioral flips therefore arise over *many* autoregressive steps, not from a decisive next-token shift. **(3) Random perturbs *more* yet ties downstream.** Random-norm injection moves the representation *more* in direction (lower cosine, higher KL, argmax changed on 53%) yet yields the *same* downstream pass rate. Note this is a *direction*, not magnitude, effect: random's norm-ratio (1.022) is actually *smaller* than transfer's (1.047), so "magnitude predicts flips" is refuted by our own data — what matters is how much the injection reorients the representation. **Direct mechanistic evidence that single-vector injection perturbs but does not inform.** (Probe n=30: bugs at positions 90–119 of the alphabetically-sorted bug list — which happen to spell MAXIMUM_K…SAME_CHARS, i.e. an *arbitrary positional slice*, not a content-based selection. The result is a *per-bug* absence, not an average: transfer and self-inject move the argmax on **0 of 30** individual bugs while random moves it on 16, so it cannot be an artifact of which bugs were sampled. We nonetheless report it as exploratory.)

---

## 10. Result V — The 2×2 Site Sweep: a **Boundary**, Not a Flat Negative (Amendment 004)

The headline cell used one fixed intervention: donor source layer 14, injected at a single token (the final prompt token) at receiver layer 17. If the failure is a *channel-width* limit (our §8 hypothesis), then widening the channel should help. We pre-registered and ran a 2×2 sweep over {source layer 14 **s1** vs all-layers-concatenated **sall**} × {inject at 1 token **t1** vs all tokens **tall**} (Amendment 004, frozen before results; per-cell norm-matched random control; success = transfer beats its random at Holm-corrected p<0.05 **and** ≥3pp effect). Pass rates (n=73, same-family):

| Cell | transfer | random (matched) | effect (transfer − random) | McNemar p |
|---|---|---|---|---|
| **s1_t1** (headline) | 0.315 | 0.301 | +1.4pp | 1.000 |
| **s1_tall** (1 layer → all tokens) | **0.000** | 0.110 | **−11.0pp** | 0.008 |
| **sall_t1** (all layers → 1 token) | **0.329** | 0.178 | **+15.1pp** | 0.043 |
| **sall_tall** (all → all) | 0.000 | 0.000 | 0.0pp | 1.000 |
| receiver no-op baseline | 0.260 | — | — | — |

The result is a **sharp boundary**, not a monotone dose-response:

- **Exactly one cell trends positive: `sall_t1`** (all source layers → a single injection token). Transfer beats its matched random by +15.1pp with McNemar p=0.043 (18 vs 7 discordant) — the **only** cell with a large positive effect in the right direction. **Crucially, under the pre-registered multiple-comparison correction this does *not* survive.** Amendment 004 froze a fixed-order family (`s1_t1 → s1_tall/sall_t1 → sall_tall`); applying the standard Holm step-down over all 4 cells, its p=0.043 exceeds its threshold of 0.017. The conclusion is robust to the order convention: under the frozen fixed-order gatekeeping, `s1_t1` (p=1.0) is tested first and stops the sequence before `sall_t1` is reached. Either way we report it as a **suggestive, not confirmed, positive** — a strong directional signal that the channel asymmetry matters, pending the confirmatory run. Donor content trends above noise **only** when the map can draw on the donor's *full* layer stack and the receiver receives it at *one* clean site.
- **Widening the *token* axis destroys it.** Injecting at *all* tokens kills transfer outright: `s1_tall` transfer = 0/73 (*worse* than its random, p=0.008), and `sall_tall` = 0/73 for *both* arms. Flooding the receiver's residual stream with an injected vector at every position drowns the signal and the model's own computation alike.
- **The pre-registered monotone prediction was falsified.** Amendment 004 predicted content advantage would grow with channel width; instead advantage is **non-monotone** — peaked at the asymmetric cell (wide source, narrow target), zero elsewhere. This is more informative than the prediction: the binding constraint is not simply "more channel," but a *specific asymmetry* (rich donor read-out, surgical receiver injection).

**Reading.** A single donor layer at one token carries no recoverable content (s1_t1 null); the *full* donor layer stack at one token does (sall_t1 positive); but spreading that same content across all tokens is catastrophic (both tall cells null/negative). The channel that works is **wide on the donor side, narrow on the receiver side** — consistent with the literature's positives (which use rich connectors) and refining them: *where* you inject matters as much as *how much*. This is the paper's central constructive finding.

## 11. Discussion: Where Transfer Breaks

The 2×2 sweep (§10) turns our initial flat negative into a **localized boundary**. The single fixed intervention we pre-registered as headline (one donor layer → one token) produces no content-specific effect; but widening the donor read-out to *all* layers while keeping the injection to a *single* token produces a nominally-significant (though not Holm-surviving) transfer effect. Combined with the field's positive results, this supports a sharper **boundary-localization** claim than "single-vector never works":

> **Activation transfer for a downstream task requires (a) a rich donor read-out and (b) a surgically narrow receiver injection.** The `sall_t1` cell — all donor layers mapped into a single receiver token — is exactly this asymmetry, and it is the only cell where content beats noise. Collapsing the donor to one layer (s1) starves the map of content; spreading the injection across all tokens (tall) floods and destroys the receiver's computation. The positives in CALM, C2C, LatentMAS, Interlat, MasterKey, and StateBridge all respect this asymmetry: they read richly from the donor and inject through a constrained, often learned, channel.

**Why might the token axis be so fragile?** A single residual vector at one token is one direction the downstream computation can choose to read or ignore; the receiver can integrate it. The same vector added at *every* token is a constant bias injected at all positions — it interferes with the model's own per-token computation rather than supplying information it can use. The donor side is the opposite: one layer is a thin, possibly unrepresentative slice, but the full layer stack captures how the donor's representation is *built up*, giving the map enough correlated structure to recover content. The asymmetric cell is the sweet spot between these two failure modes.

**A falsified prediction, honestly reported.** We pre-registered (Amendment 004) the prediction that content advantage would increase *monotonically* with channel width. It did not — advantage is peaked at the asymmetric cell and zero or negative everywhere else. We report this as a falsification, not a confirmation: the correct model of the boundary is non-monotone, and the field's "more channel is better" intuition needs the same correction.

**Implications.**

- *For representation porting / mech interp:* an interpretability finding localized to a single direction in one model likely does **not** port to another model via a single-vector map, even within a family. Porting probably requires the richer channels too.
- *For transfer claims:* any reported cross-model transfer should be required to beat a **matched random-perturbation control** — otherwise the effect may be perturbation, not information. We recommend C2 (same-norm random) as a mandatory control for the field.
- *For negative results:* high offline alignment (R²/cosine/CKA) is necessary-but-not-sufficient evidence for causal transfer; execution-verified causal flips against a perturbation control are the right bar.

**Evidentiary status (honest scoping).** We are explicit about what is established. All headline cells are **exploratory and under-powered**: n=73 (same-family), n=43 (adapter), n=30 (probe) on HumanEval-Java. The central comparison (transfer vs matched random) is an exact tie (McNemar p=1.0, 5–5 discordant), but at this n a true ~10pp content advantage would *not* reach significance — so we claim **"no detectable transfer-specific advantage, paired 95% CI ≈ ±9pp,"** not "transfer does not transfer." Per Amendments 003 and 006, the confirmatory test is run on a rebuilt execution-verified MultiPL-E-Java corpus (§12) under an honest-n pre-commitment: the achievable n (215 held-out) is below the pilot-powered n (859–1,411), so the confirmatory result is reported as a boundary (paired effect + CI + equivalence bound), not a headline claim. The exploratory claims here remain exploratory; the confirmatory boundary result is in §12.

**Resolving the exploratory/confirmatory tension.** The exploratory `sall_t1` cell was nominally positive (+15.1pp, p=0.043, not Holm-surviving) while the confirmatory adapter arm is decisively negative (−12.1pp below receiver_k1, paired CI entirely below zero). A skeptical reader should ask: is this a replication failure, a corpus shift, or a methodology artifact? Our reading, supported by §12.5–§12.7: the two results probe **different mechanisms**. The exploratory positive came from an *untrained* all-layers→single-token average map — rich donor read-out, zero learned parameters — while the confirmatory negative came from a *trained bottleneck adapter* at one layer. The trained map learned to route ~85% of its injected energy *outside* the receiver's active subspace (§12.6), and that out-of-subspace component is exactly what produces the syntactic corruption signature (§12.5). The baseline-map diagnostic (§12.7) then isolates the confound: a **naive linear ridge map through the identical channel matches the receiver baseline (and nominally exceeds it on the 3 discordant bugs), while the trained adapter through the same channel is significantly worse than doing nothing (win-share 0.154, p=0.0225)** — ridge beats adapter 13–1 on the paired dev comparison (p=0.0018). The exploratory positive, if real, lives in the rich-or-untrained-map regime, not the trained-adapter regime the field's connector papers use. The tension is therefore not a contradiction but a **map-dependence finding**: what the map learns matters more than that it transfers — and task-training the map is precisely what breaks it. We flag this as the paper's central cautionary claim for the transfer literature: training a connector can be actively harmful when the optimization pressure pushes the map off the receiver's manifold.

**A confound — now resolved at confirmatory scale.** The exploratory positive and the confirmatory negative differ in *three* things at once — corpus (HumanEval-Java vs MultiPL-E-Java), map (untrained all-layers average vs trained bottleneck adapter), and receiver site (Qwen-3B layer 17 vs the frozen layer 12; §12.3). When this section was written, we read the difference as map-dependence based on the dev-only §12.7 baseline-map result. **Amendment 008 (§12.8) has since run the layer-matched confirmatory test:** at layer 17 on held-out test, the naive ridge map and the trained adapter are indistinguishable (0.605 vs 0.595, p=0.890), and both sit below the unmodified receiver. Map-dependence does *not* survive to confirmatory scale; the remaining candidate explanations for the exploratory positive are the corpus difference and the wide donor read-out itself (which §10 already identifies as the active ingredient). We report this as the resolved reading, superseding the map-dependence hypothesis stated above.

**Limitations.** We test one task (Java program repair), one model scale (1.3B–3B), and one donor midpoint layer for the single-layer arm. The 2×2 sweep covers two source granularities (layer 14 vs all-layers) and two injection widths (single-token vs all-tokens), but not intermediate points (a few layers, a few tokens), nor attention/KV-based injection, nor a trained connector on the positive cell. The positive `sall_t1` cell is significant but at n=73 (18 vs 7 discordant); the confirmatory replication (§12) is run under the honest-n pre-commitment, so it tightens the bound but does not reach the pilot-powered n. We do **not** claim generality to non-Instruct models or other benchmarks. Scaling the receiver, richer-but-still-minimal channels, and a trained connector on the positive cell are open.

---

## 12. Confirmatory Study — MultiPL-E-Java, One Frozen Shot (Amendment 006)

The exploratory studies (§5–§10) were run on the 163-bug HumanEval-Java set and, per the exploratory/confirmatory firewall (Amendment 004), can only *generate* hypotheses. Amendment 006 discharges the deferred sizing mandate (Amendment 001 §A5) and runs the confirmatory test on a **rebuilt, larger, execution-verified corpus**. This section reports that confirmatory study. **No exploratory number above was recomputed here, and no confirmatory number below influenced any exploratory choice** — the two are strictly separated by the firewall.

### 12.1 Sizing and the honest-n pre-commitment

Amendment 001 §A5 required the confirmatory test set to be sized from the pilot's measured discordance and reported in a dated amendment *before* the run. Amendment 006 discharges this: the Gate-1 pilot measured discordance d = 8/73 = 0.110, so McNemar power to claim a paired win-share ≥ 60% (one-sided α=0.05, power 0.80) requires **n ≈ 859–1,411**. MultiPL-E-Java has a hard ceiling of ≈385 usable problems, so **the achievable n is almost certainly below the powered n**. We therefore pre-committed, in writing before any test-split run, to an **honest-n rule**: report achieved n, the paired effect size, the confidence interval, and an equivalence bound — and **not** claim the headline *positive* transfer effect, regardless of how the adapter arm looks (Amendment 006 §5). One register clarification we make explicit here and apply throughout: the honest-n rule waives the *positive* claim ("transfer beats baseline"); it does **not** waive a *negative* finding. Because the adapter's paired CI lies entirely below zero in the harm direction (§12.4), we report the adapter arm as a **statistically significant negative** (adapter underperforms baseline), while claiming nothing about the positive transfer effect the study was powered for. This is why the abstract can both decline the headline *positive* claim and still describe the adapter arm as a clean significant *negative* without contradiction.

### 12.2 The rebuilt confirmatory corpus (execution-verified, n = 335)

The confirmatory corpus is **MultiPL-E-Java** (HumanEval + MBPP translated to Java by the same-family 3B receiver, greedy decoding), then **execution-verified end to end**. The attrition funnel, each stage a hard gate:

| Stage | Count | Rule |
|---|---|---|
| Raw MultiPL-E-Java problems | 544 | 158 HumanEval + 386 MBPP |
| Translated completions generated | 543 | 1 drop (extraction failed) |
| **Gate: correct translation passes all tests** | **345** | 128 test_fail + 70 compile_error dropped |
| **Mutate: a mutant that compiles AND fails ≥1 test** | **335** | 10 operator-resistant drops |

A problem enters the corpus only if **both** a correct translation passes the full JUnit suite **and** at least one of 10 deterministic mutation operators (relational flip, method swap, off-by-one/k, literal corruption, sign flip, comparison-chain corruption, negate-boolean-return, return-constant, flip-all-relational) produces a variant that **compiles and fails ≥1 test**. The mutant is the **buggy** input; the gated-correct translation is the **fixed** reference. Tests are real JUnit (raw `assert` converted to `assertEquals`/`assertTrue`/`assertFalse`, then verified to genuinely fail on mutants — a raw-assert/no-`-ea` vacuity bug was caught and fixed pre-run). The full funnel, the drops, and the gate/mutate logic are in `results/confirmatory_corpus/pipeline/` (translate → gate → mutate → build_manifest → fix_tests); the corpus itself is in `results/confirmatory_corpus/`.

**Receiver-circularity caveat (we state this up front).** The corpus translations were generated by the **receiver model itself** (the Qwen-3B receiver translated HumanEval/MBPP to Java). A skeptical reader should worry this inflates results: the receiver may find receiver-authored problems systematically easier, biasing pass rates upward. Three mitigations, and one honest limit. (1) The circularity is **uniform across all 8 arms** — every arm is evaluated on the same receiver-authored corpus, so it inflates (or deflates) all arms together and cancels in the *paired* comparisons that carry every claim (McNemar, win-share, paired effect are all within-bug). (2) The mutations are produced by 10 *deterministic, receiver-independent* operators, so the *bugs* (the thing that makes a problem hard) are not receiver-authored. (3) The execution-verification gate (a correct translation must pass the full JUnit suite, and a mutant must compile AND fail ≥1 test) is applied uniformly regardless of authorship. The honest limit: the *absolute* pass rates (e.g., receiver_k1 63.7%) are measured on receiver-authored problems and should not be compared to a HumanEval-Java number from an independently-authored corpus; the *relative* arm-to-arm conclusions are unaffected. A corpus authored independently of the receiver is a clean robustness follow-up (§13).

**Frozen split and firewall discipline** (deterministic, by sorted `bug_id`): **train = bugs 0–79**, **dev = 80–119**, **test = 120–end = 215 bugs**. The firewall is a methods strength we state explicitly: the **test split was touched exactly once**, under the single frozen Amendment 006 config, and never used for any fitting or selection; the **train split** fit the adapter, the receiver-subspace PCA basis, and the baseline ridge map; the **dev split** hosted the α-selection, the strength×subspace pilot (§12.6), and the baseline-map comparison (§12.7) — all exploratory. No bug serves two roles, and no exploratory finding altered a frozen confirmatory criterion.

### 12.3 Frozen intervention and arms

The intervention is the Amendment 002 B2 **trained adapter** (`g·W2·SiLU(W1·(h−d_mean))`, bottleneck 512, gate init 0.1, d_mean centering, both bases frozen), retrained on the rebuilt 80-bug train split and **frozen before the test run** (final g_mean = 0.143, converged). Donor layer 14 → receiver layer 12, final-prompt-token, additive, α=1.0, temperature 0.2, top-p 0.95, seed 42, **1024 new tokens for every arm** (Amendment 005).

**A note on the receiver site.** The frozen site is **receiver layer 12** — this is the site pre-registered in Amendment 001 and reaffirmed in Amendments 002, 005, and 006. For the cross-family (DeepSeek) receiver, layer 12 is the midpoint (of 24). The confirmatory study uses the **same-family Qwen-3B receiver** (the pre-registered *control* pair, elevated to the confirmatory pair because it shares the donor's tokenizer and showed the only non-null exploratory signal), and for that receiver the *midpoint* is layer 17 (of 36) — the site used in the exploratory 2×2 sweep (§10). The confirmatory run nonetheless used the frozen layer 12 per the letter of the amendments. We flag this explicitly because it is a real deviation between the confirmatory site (12) and the same-family exploratory site (17) that the amendment trail froze but never rationalized for the Qwen-3B receiver. Two readings: (a) it makes the confirmatory negative *conservative* for the boundary claim (the sweep's positive cell used the midpoint, so the confirmatory test at a non-midpoint layer is, if anything, a harder test of single-vector transfer); (b) it is a genuine site mismatch a purist will note, and the layer-sweep follow-up (§13) is the right resolution. We report it rather than paper over it.

Each test bug is run under **all 8 arms** (Amendment 006 §4): receiver_k1 (baseline), adapter (intervention), random_norm (R=5 per-bug-seeded same-norm draws), shuffled, self_inject, unrelated_task (C5), text_hint (positive control), donor_k1 (capability reference). All outcomes are execution-verified by JUnit in isolated default-package compilations.

### 12.4 Confirmatory results

> **`The one-shot confirmatory run is complete. All numbers below are recomputed by `paper/make_stats_confirmatory.py` directly from `results/confirmatory/eval_out.jsonl` (execution-verified). Per the Amendment 006 §5 pre-commitment, achieved n is below the pilot-powered n, so **no headline effect is claimed**: the paired effect and its confidence interval are reported as a boundary result.

**Confirmatory 8-arm comparison (held-out test, n = 215 complete bugs).** Pass rates with 95% Wilson CIs; paired exact McNemar of the adapter against its two pre-registered controls; standard effect size (Cohen's *h*) for the headline contrast.

| Arm | Pass | Rate | 95% Wilson CI | Compile-err |
|-----|------|------|----------------|-------------|
| donor_k1 (donor solves it itself) | 162/215 | 0.753 | [0.692, 0.806] | — |
| **receiver_k1 (α=0 baseline)** | **137/215** | **0.637** | **[0.571, 0.699]** | **2.3%** |
| random_norm (majority-of-5; pre-reg control) | 144/215 | 0.670 | — | — |
| random_norm (per-draw mean) | — | 0.639 | spread 0.056 | — |
| text_hint (positive control) | 135/215 | 0.628 | [0.562, 0.690] | — |
| self_inject | 124/215 | 0.577 | [0.510, 0.641] | — |
| **adapter (the map under test)** | **111/215** | **0.516** | **[0.450, 0.582]** | **22.3%** |
| shuffled | 111/215 | 0.516 | [0.450, 0.582] | — |
| unrelated_task | 79/215 | 0.367 | [0.306, 0.434] | — |

**Pre-registered paired contrasts (adapter vs control), exact McNemar:**

| Contrast | adapter-wins b | control-wins c | discordant | effect (pp) | Cohen's *h* | McNemar p |
|----------|----|----|-----|-------------|-------------|-----------|
| adapter vs **receiver_k1** | 18 | 44 | 62 | **−12.1** [−18.3, −4.6] | −0.245 | **0.0013** |
| adapter vs **random_norm** (majority) | 14 | 47 | 61 | **−15.3** [−20.9, −8.2] | −0.314 | <0.0001 |

Both paired effects are entirely below zero (CI excludes 0 in the harm direction): the adapter **significantly underperforms** both the no-op receiver baseline and a matched-norm random perturbation. Cohen's *h* ≈ −0.25 to −0.31 is a small-to-medium harmful effect (computed from the marginal pass rates 0.516/0.637/0.670 via the arcsine transform). The adapter is also the only arm with an elevated compile-failure rate (22.3% vs 2.3% for the baseline — the §12.5 taxonomy localizes this). The auditable script output behind every number above is reproduced verbatim below.

```
======================================================================
CONFIRMATORY ONE-SHOT EVAL (Amendment 006 §4/§5)
======================================================================
input: results/confirmatory/eval_out.jsonl  (2580 rows)

----------------------------------------------------------------------
(1) COMPLETE BUGS (all 7 single-shot arms >=1 row AND random_norm 5 draws)
----------------------------------------------------------------------
n_complete = 215   n_dropped = 0

----------------------------------------------------------------------
(2) PER-ARM PASS RATES (95% Wilson CI)
----------------------------------------------------------------------
  receiver_k1     137/215 = 0.637  CI[0.571,0.699]
  adapter         111/215 = 0.516  CI[0.450,0.582]
  shuffled        111/215 = 0.516  CI[0.450,0.582]
  self_inject     124/215 = 0.577  CI[0.510,0.641]
  unrelated_task   79/215 = 0.367  CI[0.306,0.434]
  text_hint       135/215 = 0.628  CI[0.562,0.690]
  donor_k1        162/215 = 0.753  CI[0.692,0.806]
  random_norm     mean pass fraction 0.639  per-draw pass counts [137, 144, 132, 137, 137] (of 215)  spread 0.056
                  majority-vote (>=3 of 5 draws) pass rate = 144/215 = 0.670  [the pre-registered
                  random control is the MAJORITY vote; the per-draw mean 0.639 is shown for context]

  COMPILE-FAILURE RATE (the sharpest single contrast):
    adapter   22.3% of generations fail to compile   (48/215)
    receiver_k1  2.3%   self_inject 1.4%   random_norm 2.1%   text_hint 3.3%
    -> the adapter is ~10x more likely than the receiver alone to emit
       uncompilable code; the harm is not a uniform capability loss but a
       specific syntactic-corruption signature (fully dissected in §12.5).

----------------------------------------------------------------------
(3) PRIMARY PRE-REGISTERED TEST (§5): paired exact McNemar
    adapter vs receiver_k1  and  adapter vs random_norm(majority>=3/5)
----------------------------------------------------------------------
  adapter vs receiver_k1:
    b (adapter-pass / control-fail) = 18
    c (adapter-fail / control-pass) = 44
    discordant = 62   exact McNemar p = 0.0013  <-- SIGNIFICANT (p<0.05)
  adapter vs random_norm (majority):
    b (adapter-pass / control-fail) = 14
    c (adapter-fail / control-pass) = 47
    discordant = 61   exact McNemar p = 0.0000  <-- SIGNIFICANT (p<0.05)

----------------------------------------------------------------------
(4) WIN-SHARE on discordant pairs (pre-registered headline metric)
    win-share = b/(b+c), exact 95% binomial CI; claim threshold = 0.60
----------------------------------------------------------------------
  adapter vs receiver_k1: win-share = 18/62 = 0.290  exact 95% CI [0.182,0.419]  -> EXCLUDES-below 0.60; CI entirely below 0.60 -> adapter UNDERPERFORMS control
  adapter vs random_norm (majority): win-share = 14/61 = 0.230  exact 95% CI [0.132,0.355]  -> EXCLUDES-below 0.60; CI entirely below 0.60 -> adapter UNDERPERFORMS control

----------------------------------------------------------------------
(5) HONEST-n / EQUIVALENCE BOUND (§5 PRE-COMMITMENT — no headline claim)
----------------------------------------------------------------------
  achieved n = 215 complete bugs; pre-registered powered n = 859-1411.
  adapter - receiver_k1: paired effect = -12.1pp  95% CI [-18.3pp, -4.6pp]
    -> the data are consistent with an effect in [-18.3pp, -4.6pp];
       the pre-registered powered claim (n>=859) could NOT be tested
       at n=215; this is a boundary result.
    -> adapter pass rate (0.516) is BELOW receiver_k1 (0.637): clean negative/boundary result.
  adapter - random_norm (majority): paired effect = -15.3pp  95% CI [-20.9pp, -8.2pp]
    -> the data are consistent with an effect in [-20.9pp, -8.2pp];
       the pre-registered powered claim (n>=859) could NOT be tested
       at n=215; this is a boundary result.
    -> adapter pass rate (0.516) is BELOW random_norm (0.670): clean negative/boundary result.
  NOTE: adapter (0.516) underperforms receiver_k1 (0.637) — clean negative/boundary result vs this control.
  NOTE: adapter (0.516) underperforms self_inject (0.577) — clean negative/boundary result vs this control.
  NOTE: adapter (0.516) underperforms text_hint (0.628) — clean negative/boundary result vs this control.
  NOTE: adapter (0.516) underperforms donor_k1 (0.753) — clean negative/boundary result vs this control.
  NOTE: adapter (0.516) underperforms random_norm (0.670) — clean negative/boundary result vs this control.

----------------------------------------------------------------------
(6) DISCORDANT-RESTRICTED SUBSET (donor_k1 passes AND receiver_k1 fails)
----------------------------------------------------------------------
n = 45 bugs (of 215 complete)
  adapter                14/45 = 0.311
  text_hint              9/45 = 0.200
  self_inject            4/45 = 0.089
  unrelated_task         12/45 = 0.267
  random_norm (majority) 15/45 = 0.333

  NOTE on the positive control here: on this discordant subset (donor passes, receiver
  fails), text_hint (0.200) is WORSE than random_norm (0.333) and adapter (0.311) — the
  REVERSE of the exploratory HumanEval-Java dissociation (§7, where text-hint was the best
  arm on the discordant 8). This undercuts, on the confirmatory corpus, the §7 reading that
  "the receiver CAN use the donor's information." We flag it rather than reconcile it away:
  on the harder discordant subset of the new corpus, even the natural-language positive
  control does not beat a matched perturbation, so the §7 dissociation does not replicate
  here. This weakens §7's "channel doesn't deliver" framing for this corpus and is discussed
  in §11.
````

The block above is the verbatim output of `make_stats_confirmatory.py`, recomputed directly from `results/confirmatory/eval_out.jsonl` (the one-shot run is complete; these are final, not placeholders). Pre-registered analyses (Amendment 006 §5): per-arm pass rates with 95% Wilson CIs; paired exact McNemar of adapter vs receiver_k1 and adapter vs random_norm (majority-of-5); win-share b/(b+c) with exact 95% Clopper-Pearson CI on the discordant pairs (headline metric, claim threshold 0.60); and the honest-n equivalence bound. Per the pre-commitment (§12.1), the achieved n (215) is below the powered n (859–1,411), so **no headline effect is claimed** — the paired effect and its CI are reported as a boundary result.

---

## 12.5 Compile-Error Taxonomy: What the Adapter Actually Breaks

The adapter's 48 compile errors (22.3% of generations) are not random noise. We recompiled every failed generation with javac and parsed the exact diagnostics into root-cause buckets (Amendment 007, exploratory). The distribution reveals a **distinctive adapter-specific signature**:

| Root Cause | Adapter (n=48) | Shuffled (n=36) | Receiver (n=5) | Text Hint (n=7) | Random Norm (n=23) |
|------------|----------------|-----------------|----------------|-----------------|-------------------|
| **unbalanced_extra_close** | **22 (46%)** | 9 (25%) | 0 | 0 | 1 (4%) |
| **illegal_package_reserved** | **13 (27%)** | 8 (22%) | 0 | 0 | 0 |
| unresolved_symbol | 8 (17%) | 13 (36%) | 2 (40%) | 4 (57%) | 9 (39%) |
| incompatible_types | 4 (8%) | 3 (8%) | 3 (60%) | 3 (43%) | 13 (57%) |
| unclosed_literal | 1 (2%) | 2 (6%) | 0 | 0 | 0 |
| truncation_eof | 0 | 1 (3%) | 0 | 0 | 0 |

**Three findings:**

1. **The adapter's failure mode is syntactic, not semantic.** 73% of adapter compile errors are structural (extra closing brace, illegal package declaration) rather than semantic (unresolved symbol, type mismatch). The adapter corrupts the *syntactic scaffold* of code generation.

2. **This is adapter-specific, not injection-general.** The receiver baseline (0%) and random_norm control (4%) show essentially no extra-close-brace errors. The shuffled control (25%) has elevated but significantly lower rates. The adapter's learned mapping specifically induces this failure mode.

3. **The shuffled control's profile differs.** Shuffled (wrong-direction injection) is dominated by unresolved symbols (36%) — a semantic failure — while the adapter is dominated by structural corruption. This suggests the adapter's learned subspace projection has a qualitatively different effect than random perturbation.

**Mechanistic interpretation.** The adapter's extra-close-brace signature (46%) suggests it corrupts the receiver's tracking of block structure — likely by injecting donor representations that encode a different nesting depth or scope boundary than the receiver expects at that generation point. The illegal-package signature (27%) suggests the adapter induces the receiver to "hallucinate" package declarations (e.g., `package java.util;`) that violate Java's reserved namespace rules. Both are consistent with the adapter mapping donor hidden states that are **out-of-distribution for the receiver's syntactic state machine**.

This taxonomy sharpens the boundary result: the adapter doesn't merely fail to help — it actively **breaks syntactic coherence** in a specific, measurable way that random perturbation does not. The pilot (Amendment 007) tests whether this is a strength effect (‖Δ‖ too large) or a subspace mismatch (Δ outside the receiver's active manifold).

---

## 12.6 Pilot: Strength vs Subspace (Amendment 007)

To diagnose *why* the adapter corrupts syntax, we ran the pre-registered paired 2×2 pilot on the **dev split only** (bugs 80–119, n=40; test split untouched). Factors: **strength** (α ∈ {0.5, 1.0}) × **subspace** (adapter-as-is vs adapter→receiver-PCA-aligned). The receiver active subspace (layer 12, k=35 directions, 90% variance) was fit on the **train split only** (bugs 0–79, anti-leakage per Amendment 007 §3) and applied frozen to dev.

**Instrumentation.** The pilot logs what the confirmatory eval lacked: injected norm ‖Δ‖, subspace projection fraction (`proj_frac`), and adapter gate magnitude. Key measurements:
- `proj_frac` (as_is) = **0.15–0.16**: only ~15% of the adapter's injected vector lies inside the receiver's active subspace.
- `proj_frac` (aligned) = **1.00** (by construction).
- Aligned reduces ‖Δ‖ substantially (e.g., α=1.0: mean ‖Δ‖ 90.2 → 13.8).

**Pass rates (n=40 bugs per cell):**

| Condition | Pass Rate | Compile Errors |
|-----------|-----------|----------------|
| α=0.5 as_is | 19/40 = 47.5% | 2/40 = 5.0% |
| α=0.5 aligned | **24/40 = 60.0%** | 3/40 = 7.5% |
| α=1.0 as_is | 14/40 = 35.0% | **8/40 = 20.0%** |
| α=1.0 aligned | **21/40 = 52.5%** | 3/40 = 7.5% |

**Paired McNemar (discordant pairs, exact binomial):**

| Comparison | b (first wins) | c (second wins) | disc | p (one-sided) |
|------------|----------------|-----------------|------|---------------|
| Subspace @ α=0.5 (aligned > as_is) | 2 | 7 | 9 | 0.180 |
| Subspace @ α=1.0 (aligned > as_is) | 3 | 10 | 13 | 0.092 |
| Strength @ as_is (α=0.5 > α=1.0) | 6 | 1 | 7 | 0.125 |
| Strength @ aligned (α=0.5 > α=1.0) | 4 | 1 | 5 | 0.375 |

**Readout against the Amendment 007 §5 pre-committed boundaries.** We check the observed data against each pre-committed category explicitly (not against a post-hoc narrative):

- **Evidence for STRENGTH** requires compile-failure to track α monotonically *in both* subspace conditions. Observed: as_is rises 5.0%→20.0% across α (monotone ✓), but aligned is flat 7.5%→7.5% (✗). **Not met** — strength is not the dominant factor once geometry is fixed.
- **Evidence for SUBSPACE** requires the aligned-vs-as_is compile-failure gap to exceed the pre-committed ≥15pp *regardless of α*. Observed gaps: α=0.5 → −2.5pp (aligned is *higher*); α=1.0 → +12.5pp. The α=1.0 gap (12.5pp) is **below** the pre-committed 15pp band, and the α=0.5 gap is in the wrong direction. **Not met** against the frozen band.
- **BOTH (interaction)** requires failure to drop *only* when α is low AND aligned. Observed: the lowest-failure cells are the two as_is/aligned α=0.5 cells (5.0%, 7.5%), but α=1.0-aligned (7.5%) also has low failure — the pattern is not a clean α×subspace interaction. **Not met.**
- **→ Pre-committed verdict: CANNOT TELL.** Point estimates fall short of every frozen band. Per §5/§6 we report the estimates + CIs and stop; we do not re-run to chase significance. This is the pre-registered default outcome, reported as such.

**§5 pre-committed readout: "cannot tell" — but with a clear directional signal.** Neither the subspace nor the strength effect reaches significance at n=40 (the pilot was powered only for ≥80% discordant-win; observed discordant-win rates are 65–86%, so underpowered as pre-registered). **However**, the descriptive pattern is consistent and mechanistically interpretable:

1. **Subspace alignment cuts compile errors by 62% at α=1.0** (20.0% → 7.5%) — the adapter's structural-corruption signature is driven by the ~85% of its delta that lies *outside* the receiver's active subspace.
2. **Subspace alignment lifts pass rate at both strengths** (+12.5pp at α=0.5, +17.5pp at α=1.0).
3. **Strength reduction (α=0.5) alone does not fix the problem** — compile errors persist at 5% and pass rate stays below the aligned conditions. The issue is not merely that the injection is too strong; it is that it points in the wrong direction.

**Baseline context (dev receiver_k1, α=0 run through the same harness).** The pilot table above is incomplete without the receiver baseline on the same dev bugs. Running α=0 (no injection) through the identical harness on bugs 80–119 gives **receiver_k1 dev = 23/40 = 57.5%**. Reframed against baseline:

| Condition | Pass Rate | vs Baseline |
|-----------|-----------|-------------|
| receiver_k1 (α=0) | 57.5% | — |
| α=0.5 as_is | 47.5% | −10.0pp |
| α=0.5 aligned | 60.0% | **+2.5pp** |
| α=1.0 as_is | 35.0% | −22.5pp |
| α=1.0 aligned | 52.5% | **−5.0pp** |

Paired McNemar, aligned-α=1.0 vs receiver_k1 on the same 40 bugs: only 4 discordant pairs, aligned win-share = 1/4 = 0.25 (exact p=0.63). **Subspace alignment fixes the corruption but does not recover baseline performance** — the aligned adapter at the confirmatory strength remains 5pp *below* doing nothing at all. Only the α=0.5-aligned cell nominally exceeds baseline, by a single bug (24 vs 23), far below any significance threshold.

**Mechanistic conclusion (exploratory).** The adapter's learned map transfers a donor vector that is ~85% orthogonal to the receiver's active subspace. The orthogonal component corrupts the receiver's syntactic state tracking (producing the extra-close-brace and illegal-package signatures of §12.5) without contributing usable task content. Removing it (alignment) eliminates most of the corruption — evidence *consistent with* a **subspace-mismatch account** over a pure-magnitude account, though the §5 readout is formally CANNOT-TELL on strength-vs-subspace — but the in-subspace remainder still does not add net value over the receiver's own state. In other words: **the adapter's damage appears substantially geometric, but its benefit is absent even after the geometry is fixed.**

**Powered-follow-up decision (pre-registered honest scoping).** We considered powering a confirmatory test of the aligned-α=0.5 cell against receiver_k1 (the only adapter cell trending positive, +2.5pp on dev). The Amendment 006 power analysis for a +2.5pp effect at this observed discordance would require n in the thousands — far beyond MultiPL-E-Java's ceiling (≈385 usable problems) and therefore beyond what this benchmark family can ever deliver. We therefore **decline to pre-register a powered follow-up on this corpus** for the adapter and report the pilot as what it is: an underpowered, honest, mechanistically informative diagnostic that (a) identifies the corruption mechanism (out-of-subspace injection), (b) shows the mechanism is fixable (alignment cuts compile errors 62%), and (c) shows that fixing the mechanism is *not sufficient* for transfer to beat the baseline. A positive adapter-based transfer claim on this channel would require a qualitatively different intervention (richer connector, attention-level channel), not a larger n.

---

## 12.7 The Missing Baseline: a Naive Linear Map Beats the Trained Adapter (exploratory, dev)

The confirmatory negative (§12.4) and the pilot (§12.6) both concern the **trained bottleneck adapter** — the map architecture the field's connector papers recommend. But this leaves a confound: is the failure specific to the trained adapter, or is single-vector cross-model injection at this site broken in general? To answer this we ran the simplest possible alternative map — a **naive centered ridge regression** (λ=1.0, no nonlinearity, no bottleneck, no task training) from donor layer-14 to receiver layer-12 final-prompt-token activations, fit on the **train split only** (bugs 0–79), and evaluated it on the dev split (bugs 80–119, n=40) under the exact confirmatory decoding config (α=1.0, additive delta, final prompt token). The map emits a delta directly (fit target = receiver deviation from its own mean), matching the adapter's injection semantics. Test split untouched; exploratory only.

**Offline fit.** The ridge map achieves R²=0.865, mean cosine=0.940 on train — high offline alignment, the exact "necessary but not sufficient" signature we warn about.

**Dev results (n=40/cell, vs the pilot cells and receiver baseline):**

| Condition | Pass Rate | Compile Err | vs receiver_k1 | ‖Δ‖ median |
|-----------|-----------|-------------|----------------|------------|
| receiver_k1 (α=0) | 23/40 = 57.5% | 10% | — | 0 |
| **baseline_ridge α=1.0** | **26/40 = 65.0%** | 5% | **+7.5pp** | **3.0** |
| adapter α=0.5 aligned | 24/40 = 60.0% | 7.5% | +2.5pp | 6.2 |
| adapter α=1.0 aligned | 21/40 = 52.5% | 7.5% | −5.0pp | 12.4 |
| adapter α=0.5 as_is | 19/40 = 47.5% | 5% | −10.0pp | 40.3 |
| adapter α=1.0 as_is | 14/40 = 35.0% | 20% | −22.5pp | 81.7 |

**Paired McNemar (dev, exact binomial):**
- **ridge vs adapter α=1.0 as_is:** ridge wins 13, adapter wins 1, discordant=14, **win-share 0.929, p=0.0018**. The naive linear map significantly outperforms the trained adapter on this dev split.
- **ridge vs receiver_k1:** ridge wins 3, receiver wins 0, discordant=3, win-share 1.000, p=0.25 (underpowered — only 3 discordant bugs, so no significance, but the point estimate is +7.5pp above baseline).
- adapter α=1.0 as_is vs receiver_k1: win-share 0.154, p=0.0225 (the adapter is significantly *worse* than baseline).

**Three conclusions.**

1. **The failure is adapter-architecture-specific, not injection-general.** The single-vector injection channel at this site is *not* fundamentally broken: a zero-task-training linear probe through the identical channel matches the receiver baseline (and nominally exceeds it on the 3 discordant bugs), while the trained adapter through the same channel collapses to 35% and is significantly worse than doing nothing. The binding constraint is the **map**, not the channel.

2. **Task-trained optimization is the *likely* harm — but architecture and magnitude are confounded here.** The adapter is the only map trained for receiver-side causal use, and it is the only map that catastrophically fails. Its median injected norm (81.7) is **27× larger** than the ridge map's (3.0), and §12.6 shows ~85% of that energy is out-of-subspace corruption. The ridge map's tiny, conservative delta stays on-manifold and preserves the receiver's computation. So "the adapter was task-trained" and "the adapter injects a 27×-larger, mostly off-manifold delta" are *both* true and *not dissociated* by this comparison — the ridge map differs from the adapter in architecture (linear vs bottleneck-nonlinear), training (none vs task), *and* output magnitude simultaneously. Consistent with §12.6's CANNOT-TELL verdict, we therefore cannot yet say whether the harm is the *training objective*, the *magnitude inflation*, or their interaction; the norm-response curve (§13, follow-up b) is the experiment that dissociates them. The conservative claim we stand behind: **task-training the adapter *produced* a large, mostly off-manifold delta, and a map that avoids both matches baseline.**

3. **This reframes the paper's central claim — with the confounds attached.** The negative result is not "single-vector cross-model transfer doesn't work." It is, more precisely: "**on the dev split, a trained bottleneck adapter underperforms a naive linear map through the same channel (13–1), and the adapter — uniquely among the maps tested — produces large off-manifold deltas that corrupt syntax.**" This is consistent with the stronger reading that the field's recommended fix (task-training a connector) actively destroys a channel a naive map leaves intact, and it sits in tension with the connector literature's core assumption that task-trained maps dominate offline-fit maps. But because the comparison confounds architecture with magnitude and crosses splits (above), we offer the strong reading as a hypothesis the data *support*, not one they *establish*.

**Scope — and the split-confound we want to be explicit about.** This is an n=40 **dev-split** exploratory comparison. Every cell in the table above is dev (n=40), so the *within-table* comparisons are internally valid. But the paper's headline adapter number (35%/0.516 and the confirmatory negative of §12.4) is the **confirmatory test-set (n=215)** result, while the ridge map was evaluated only on **dev (n=40)**. The adapter and the ridge map therefore **never meet on the same split at confirmatory scale**: we compare the adapter's confirmatory-test failure against the ridge map's dev success, on different bugs. This is the single most important caveat on the §12.7 claim, and we state it up front: the 13–1 ridge-over-adapter result is a *dev* comparison (the pilot's adapter cells are also dev n=40, so that specific comparison is same-split), but the broader inference "a naive map leaves the channel intact where the adapter fails at confirmatory scale" crosses splits. The clean resolution — running the frozen ridge map once on the held-out test split — would consume the one allowed test touch, which we have already spent on the adapter arm; we therefore decline it and report the ridge result as dev-only. A future confirmatory test of the ridge map needs its own corpus + amendment (per the power analysis, ~860+ bugs — beyond this corpus). We report the finding as exploratory: it (a) isolates the adapter architecture as the *likely* failure locus and (b) identifies the naive linear map as the correct reference point for any future transfer claim on this channel, but it does not constitute confirmatory evidence that the channel works.

---

## 12.8 Confirmatory Resolution at the Corrected Site: the Ridge Map on Test (Amendment 008)

§12.7 left the paper's most important open question unresolved. The naive ridge map beat the trained adapter 13–1, but that result sat on the **dev split (n=40)** while the adapter's failure was measured on **test (n=215)** — so the two maps never met on the same split at confirmatory scale, and §12.7 explicitly declined the one allowed test touch. Amendment 008 closes exactly this gap, and simultaneously corrects a site error: the receiver injection site used throughout §12.4–§12.7 was layer 12, but the Qwen-3B midpoint of the exploratory sweep is layer 17. Amendment 008 therefore re-runs the **entire confirmatory family at layer 17 in a single frozen shot**, adding the ridge map as a confirmatory arm. Both maps were **retrained** at layer 17 (the adapter fully re-converged, g_mean=0.172; the ridge map re-fit at the new site, R²=0.876, mean cosine=0.942 — passing the pre-committed §5 sanity gate of R²≥0.60 and cos≥0.75 *before* any test execution). The test split was touched **once**, decoding config identical to Amendment 006 (temp 0.2, top-p 0.95, seed 42, final-prompt-token additive injection, α=1.0).

Layer-independent arms (receiver_k1, text_hint, donor_k1) pass no `hook_layer` argument, so their code path is the same at layers 12 and 17. Amendment 008 planned to reuse their Amendment 006 results, but they were re-generated in this run, and their counts differ slightly from Amendment 006 (receiver_k1 137→136, text_hint 135→148, donor_k1 162→159; see `prereg/ERRATA.md`). The table reports the re-run counts; these arms appear only for descriptive context, not as members of the layer-17 family.

**Layer-17 confirmatory results (n=215 test per arm; random_norm = 5 draws, majority vote):**

| Arm | Pass | Acc | Wilson 95% CI | Compile Err |
|-----|------|-----|---------------|-------------|
| donor_k1 (context) | 159/215 | 0.740 | [0.677, 0.794] | — |
| text_hint (context) | 148/215 | 0.688 | [0.624, 0.747] | — |
| random_norm (majority) | 140/215 | 0.651 | [0.585, 0.712] | — |
| receiver_k1 (context) | 136/215 | 0.633 | [0.566, 0.694] | — |
| self_inject | 132/215 | 0.614 | [0.547, 0.676] | — |
| **ridge_map α=1.0** | **130/215** | **0.605** | **[0.538, 0.668]** | **6 (2.8%)** |
| **adapter α=1.0** | **128/215** | **0.595** | **[0.529, 0.659]** | **23 (10.7%)** |
| shuffled | 123/215 | 0.572 | [0.505, 0.636] | — |
| unrelated_task | 85/215 | 0.395 | [0.332, 0.462] | — |

**Paired McNemar contrasts (exact binomial, Holm-corrected within the Amendment 008 family of 8):**

| Contrast | Δ acc | a+/b− | a−/b+ | p (exact) | p (Holm) |
|----------|-------|-------|-------|-----------|----------|
| **ridge vs adapter (§7 headline)** | +0.009 | 27 | 25 | 0.890 | 0.890 |
| adapter vs unrelated_task | +0.200 | 66 | 23 | 5.7e-06 | 4.6e-05 |
| adapter vs random_norm | −0.056 | 21 | 33 | 0.134 | 0.936 |
| ridge vs random_norm | −0.047 | 14 | 24 | 0.143 | 0.860 |
| ridge vs receiver_k1 | −0.028 | 7 | 13 | 0.263 | 1.000 |
| adapter vs receiver_k1 | −0.037 | 20 | 28 | 0.312 | 1.000 |
| adapter vs shuffled | +0.023 | 30 | 25 | 0.590 | 1.000 |
| adapter vs self_inject | −0.019 | 23 | 27 | 0.672 | 1.000 |

**Five conclusions.**

1. **The dev-split headline does not survive to test.** At confirmatory scale on the held-out split, the ridge map and the trained adapter are **indistinguishable at this sample size** (0.605 vs 0.595; ridge wins 27, adapter wins 25, discordant=52, McNemar p=0.890). The point estimate is +0.9pp with a 95% CI of [−5.7pp, +7.5pp] — this is a **failure to detect a difference at n=215**, not a proof of equivalence; a true ridge advantage of up to ~7.5pp is consistent with the data. What the data *do* rule out is the §12.7 effect size: the dev-split 13–1 advantage (a ~12pp gap on the adapter cells tested there) sits outside this CI. The §12.7 result was a **split confound**: the adapter's test-set number was depressed relative to its dev behavior, and the ridge map's dev-set number was inflated relative to its test behavior. This is precisely the inferential error the exploratory/confirmatory firewall exists to catch, and it is why §12.7 flagged the cross-split comparison as its single most important caveat.

2. **Neither map beats doing nothing — consistent with a channel-limited account.** Both maps sit *below* receiver_k1 (adapter −3.7pp, 95% CI [−10.0, +2.6]; ridge −2.8pp) and *below* matched random_norm noise (adapter −5.6pp, CI [−12.3, +1.1]; ridge −4.7pp), and none of those contrasts is significant. The §12.7 hypothesis that "the binding constraint is the map, not the channel" is **not supported at test scale** — but we note the inverse claim is equally unsupported: we tested two maps (one task-trained, one naive) at one site on one model pair. The data are *consistent with* a channel-limited account — no map we tried extracts usable content — but they do not establish that no map could. A stronger map class (multi-token, multi-layer, or learned on a larger corpus) remains untested at confirmatory scale.

3. **The compile-failure signature halves at layer 17 but persists.** The adapter's compile-error rate drops from 22.3% (layer 12) to 10.7% (layer 17), and the ridge map — which §12.7 showed keeps its delta ~27× smaller — sits at 2.8%, near the no-injection floor. Magnitude and site both modulate syntactic corruption, consistent with §12.5's out-of-subspace account, but even the cleanest map does not convert the channel into a net positive.

4. **Site correction is a trending, not significant, effect.** Retraining and re-injecting at layer 17 lifts the adapter from 0.516 to 0.595 (+7.9pp, 95% CI [+0.3pp, +15.5pp]; L17 wins 44, L12 wins 27, discordant=71, McNemar exact p=0.057 — reported descriptively per §7 as it crosses amendments). The CI barely excludes zero and the point estimate favors the exploratory sweep's midpoint, but the improvement does not cross significance and does not change the qualitative verdict.

5. **The one thing that works is task content, not the vector.** The only decisive contrast in the family is adapter vs unrelated_task (+20.0pp, p=4.6e-05, Holm-significant): injecting a *task-relevant* delta beats injecting a *task-irrelevant* one. Combined with donor_k1 (0.740) and text_hint (0.688) sitting far above every injection arm, the picture is unchanged from §7 — the information transfers through natural language and through task semantics, not through the single-vector latent channel.

**What this means for the paper's claim.** Amendment 008 removes §12.7's strongest pro-transfer reading and replaces it with a cleaner, more conservative one: **on held-out data at the corrected site, neither of the two map architectures tested — naive or trained — extracts a usable repair signal from a single cross-model vector, and the apparent dev-split advantage of the naive map was a split-comparison artifact.** Whether a richer map class could do better remains open; the boundary of §10 stands.

---

## 13. Conclusion

We pre-registered and tested the simplest cross-model activation-transfer mechanism on an execution-verified program-repair task, adversarially, with matched perturbation controls, and then swept the intervention site under a plan frozen before the results were seen. The headline finding is a **localized boundary, not a flat negative**. Single-layer→single-token injection produces no content-specific effect over a matched random perturbation, and a full control set shows the same information transfers readily through natural language but not through that channel. But a pre-registered 2×2 site sweep reveals exactly one trending-positive cell — **all donor layers mapped into a single receiver token** (+15.1pp over matched random, McNemar p=0.043, though not surviving the 4-cell Holm correction) — while every all-token-injection cell collapses to zero. The transfer boundary is therefore **asymmetric**: rich donor read-out, surgically narrow receiver injection. This falsifies the "more channel is better" prior and hands the field a concrete, pre-registered boundary claim. A mechanistic probe shows the injected single-layer content never reaches the receiver's output differently from the receiver's own state. A powered confirmatory test (Amendment 006) run on a rebuilt, larger MultiPL-E-Java corpus under an explicit honest-n pre-commitment (§12) is decisively negative for the trained adapter, and the follow-up diagnostics (§12.5–§12.7) characterize the failure: the adapter injects ~85% of its energy outside the receiver's active subspace, corrupting syntactic state tracking (§12.5), and subspace alignment alone does not recover baseline (§12.6). Finally, a corrected-site confirmatory re-run (Amendment 008) puts the naive ridge map on the held-out test split head-to-head with the adapter — and the apparent dev-split advantage of the naive map **does not survive to test** (ridge 0.605 vs adapter 0.595, discordant=52, McNemar p=0.890, 95% CI on Δ [−5.7, +7.5]pp). The §12.7 result was a split confound: at confirmatory scale, of the two maps tested — naive and trained — neither extracts a usable repair signal from a single cross-model vector, and both marginally underperform the unmodified receiver. The data are consistent with a channel-limited account, though only two map architectures at one site were tested. For activation transfer, **where** you read and **where** you inject both matter — and at the single-vector boundary we locate, neither of the two map-fitting strategies we tested rescues the channel.

**Limitations and Future Work.** The confirmatory negative and the corrected-site ridge test are both clean, but three scope limits deserve stating plainly. **(1) Same-family pair.** Donor and receiver share the Qwen2.5-Coder architecture and tokenizer, so the boundary we map does not directly bound cross-*family* transfer (though the cross-family pilot, §5, is even more negative). **(2) Single injection site.** We swept donor-source and injection-width but tested only the layer-14→layer-17 final-token site at confirmatory strength; the field's rich-connector positives all use multi-layer, multi-token injection, and we do not claim the boundary holds there. **(3) Confirmatory power.** The confirmatory families (n=215) are honest-n boundaries, not powered headlines (§12.1); a difference smaller than ~9pp between two arms would not reach significance here, so the ridge-vs-adapter equivalence is a *failure to detect a difference*, not a powered equivalence proof. **The natural next experiments, in order:** (a) a powered confirmatory test of the naive linear map against receiver_k1 — per the §12.1 power analysis this needs n≈860+, which exceeds this corpus's ceiling and therefore requires a larger program-repair benchmark (we did not run it; we flag it as the single most informative follow-up); (b) a norm-response curve on the baseline map (α sweep at dev) to test whether the adapter's harm is a pure magnitude effect or a genuine geometry effect independent of scale; (c) the same design on a cross-family pair and a second injection site to probe generality. Each is a clean, pre-registerable follow-up; the artifacts (maps, corpus, harness) are released so any of them can be run directly.

---

## Broader Impact and Ethics

This work studies how task-relevant information transfers between language models through their hidden states, on a synthetic program-repair benchmark. It does not introduce a new capability; it characterizes the *boundary* of an existing one (activation transfer) and shows the minimal mechanism is weaker than the field's framing implies. No human subjects, no personal data, no new data collection. The models used are public, instruction-tuned code models. The main societal consideration is downstream: a clearer map of where cross-model transfer breaks constrains (rather than enables) speculative "capability injection" schemes, which we view as mildly safety-relevant and net-positive for interpretability. We see no dual-use risk beyond that of the underlying public models. All artifacts are released for reproducibility.

---

## Reproducibility

We release the complete artifact at `transfer_mech/` (raw results mirrored to S3):

1. **Pre-registration** + all eight amendments, each frozen before its corresponding run.
2. **Full execution-verified harness** (`eval_harness/`) with a deterministic runner per study.
3. **Raw per-bug JSONL logs** for every study: gate0, cross-family pilot, trained adapter, same-family pilot, α-sweep, rigorous control set, mechanistic probe, site sweep, and the Amendment 008 corrected-site confirmatory re-run (`results/confirmatory_L17/`).
4. **One-command statistical reproduction** — `paper/make_stats.py` recomputes every exploratory p-value and CI from raw logs; `paper/make_stats_confirmatory.py` recomputes the confirmatory (Amendment 006) analyses from `results/confirmatory/eval_out.jsonl`; `results/confirmatory_L17/stats_L17.py` recomputes the Amendment 008 analyses from `results/confirmatory_L17/eval_out.jsonl` + `ridge_out.jsonl`; `paper/make_figures.py` regenerates every figure.
5. **The rebuilt confirmatory corpus** — `results/confirmatory_corpus/pipeline/` (translate → gate → mutate → build_manifest → fix_tests) reproduces the n=335 execution-verified MultiPL-E-Java corpus and its manifest end to end; the corpus, manifest, trained adapter, and pipeline scripts are mirrored to S3.
6. **`paper/REPRODUCE.md`** with exact commands and expected outputs per study.

Models: Qwen/Qwen2.5-Coder-1.5B-Instruct (donor), Qwen/Qwen2.5-Coder-3B-Instruct (same-family receiver), deepseek-ai/deepseek-coder-1.3b-instruct (cross-family receiver). Hardware: single NVIDIA A10G. No test bug is used to fit any map or select any hyperparameter.

---

## Appendix A. Pre-Registration Amendment Log (full trail)

All eight amendments are frozen, dated documents in `prereg/`. Each was committed before its corresponding run; none loosened a frozen criterion.

| Amendment | Frozen before | What changed and why |
|---|---|---|
| v1 + 001 | any experiment | Intervention (donor L14→receiver L12, final-prompt-token, additive), model pair, 5 mechanically-pinned controls (C1–C5), power analysis, 4 kill gates. |
| 002 | adapter training | Added the trained bottleneck adapter (the field's recommended connector) as a pre-registered arm. |
| 003 | control-set run | Honest re-scoping: the exploratory split is underpowered, so controls are restated and the confirmatory reaffirmed; no criterion loosened. |
| 004 | site sweep | Froze the 2×2 injection-site analysis plan (incl. the monotone prediction later falsified) before the sweep was run. |
| 005 | confirmatory run | Froze max_new_tokens=1024 for every arm (removes a length-confound). |
| 006 | confirmatory run | Discharged the deferred sizing mandate with the honest-n pre-commitment: report effect + CI + equivalence bound, claim nothing if n < powered n. |
| 007 | pilot | Froze the strength×subspace pilot as dev-only and exploratory, with the §5 pre-committed readout boundaries (checked in §12.6). |

## Appendix B. Per-Bug Data Availability

Every per-bug outcome for all 8 confirmatory arms (215 bugs × 7 single-shot arms + 5 random_norm draws = 2,580 rows) is in `results/confirmatory/eval_out.jsonl`, schema `{bug_id, arm, status, code, gen_s, seed}`, mirrored to S3. The pilot (160 rows) and baseline-map (40 rows) per-bug outcomes are in `results/pilot/pilot_full.jsonl` and `results/pilot/baseline_ridge_dev.jsonl`. `paper/make_stats_confirmatory.py` recomputes every confirmatory number from these raw logs; no number in the paper is hand-entered.

## Appendix C. Pilot Instrumentation (Amendment 007)

The pilot logs three quantities the confirmatory eval lacked, per generated bug and condition: injected norm **‖Δ‖**, receiver-subspace projection fraction **proj_frac** (‖proj(Δ)‖/‖Δ‖ onto the train-fit k=35 active subspace), and adapter **gate magnitude** (mean |g|). Key values:

| Quantity | as_is | aligned |
|---|---|---|
| proj_frac | 0.15–0.16 (~85% off-subspace) | 1.00 (by construction) |
| ‖Δ‖ @ α=1.0 (mean) | ~90 | ~13.8 |
| adapter gate g_mean | 0.152 (pilot reload; confirmatory 0.143) | 0.152 |

For the baseline-map arm (§12.7), the injected deltas have median norm **3.0** (range 1.6–7.9) — 27.5× smaller than the adapter's as_is median (81.7, range 5.0–172.1). These instrumentation logs are the direct evidence that the adapter's harm is a magnitude-plus-geometry effect (large norm, ~85% off-manifold), not a channel property.

## References

**Cross-model / connector-based activation transfer** (the literature this paper probes):

1. Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Shikhar Vashishth, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, Partha Talukdar. "LLM Augmented LLMs: Expanding Capabilities through Composition." *International Conference on Learning Representations (ICLR)*, 2024. arXiv:2401.02412.

2. Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang. "Cache-to-Cache: Direct Semantic Communication Between Large Language Models." *International Conference on Learning Representations (ICLR)*, 2026.

3. Shilian Chen, Jie Zhou, Qin Chen, Wen Wu, Xin Li, Qi Feng, Liang He. "Can Heterogeneous Language Models Be Fused?" *arXiv preprint*, 2025.

4. Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Yu Cheng, Bo Zheng, Wei Chen, Haochao Ying. "Enabling Agents to Communicate Entirely in Latent Space." *Association for Computational Linguistics (ACL)*, 2026. arXiv:2511.09149.

5. Jiaru Zou, Ruizhong Qiu, Gaotang Li, Xiyuan Yang, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, Ling Yang. "Latent Collaboration in Multi-Agent Systems." *arXiv preprint*, 2025. arXiv:2511.20639.

6. Rishab Balasubramanian, Pin-Jie Lin, Rituraj Sharma, Anjie Fang, Fardin Abdi, Viktor Rozgic, Zheng Du, Mohit Bansal, Tu Vu. "The Master Key Hypothesis: Unlocking Cross-Model Capability Transfer via Linear Subspace Alignment." *arXiv preprint*, 2025.

7. Yanwen Peng, Delvin Ce Zhang, Xi Wang, Nikolaos Aletras. "StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems." *Conference on Language Modeling (COLM)*, 2026. arXiv:2608.13317.

**Internal-geometry / representational-alignment measurement:**

8. Xueping Gao. "Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures." *Proceedings of Machine Learning Research*, 2026. arXiv:2606.16897.

9. Kiril Bangachev, Guy Bresler, Yury Polyanskiy. "Representation Alignment Rests on Linear Structure." *arXiv preprint*, 2026. arXiv:2605.28870.

**Interpretability / negative-transfer / scaling findings we cite:**

10. Peiyan Zhang, Jason Xin. "A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting." *arXiv preprint*, 2026. arXiv:2606.03280.

11. Muhammad Usama, Dong Eui Chang. "Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning." *arXiv preprint*, 2026. arXiv:2605.23315.

12. Aman Mehta. "When Agents Commit Too Soon: Diagnosing Premature Commitment in LLM Agents." *Preprint*, 2026. arXiv:2606.22936.

13. Jason Wei, Najoung Kim, Yi Tay, Quoc V. Le. "Inverse Scaling Can Become U-Shaped." *arXiv preprint*, 2022.


**Benchmarks and program repair:**

14. Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba. "Evaluating Large Language Models Trained on Code." *arXiv preprint*, 2021. arXiv:2107.03374.

15. Chunqiu Steven Xia, Lingming Zhang. "Keep the Conversation Going: Fixing 162 out of 337 bugs for $0.42 each using ChatGPT." *arXiv preprint*, 2023. arXiv:2304.00385.

---

**Reproducibility artifacts (ours).** Pre-registration v1 + Amendments 001–008; `results/confirmatory/eval_out.jsonl` (2,580 rows); `results/confirmatory_L17/` (2,580 rows + 215 ridge rows + stats); `results/confirmatory_corpus/` (335-bug execution-verified MultiPL-E-Java, mirrored to S3); `eval_harness/` (fit_linear_map.py, eval_baseline_map.py, train_adapter.py, eval_adapter.py); `results/pilot/` (BASELINE_MAP_READOUT.md, PILOT_READOUT.md, analyze_baseline.py); `results/adapter_L17/` + `results/ridge_L17.pt`. One-command reproduction in `REPRODUCE.md`.