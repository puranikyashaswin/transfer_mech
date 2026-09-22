# Pre-Registration v1 — "What Transfers?"
**Date: 22 Sep 2026. This document is FROZEN before any experiment. Changes require a dated amendment below.**

## Research question
When does a frozen donor model's internal representation *causally enable* a frozen receiver model to succeed — and can that benefit be predicted from pre-intervention signals?

## Hypotheses (separate — one does not imply another)
- **H1 (transfer):** A specified donor→receiver intervention improves execution-verified outcomes on held-out examples vs. controls.
- **H2 (prediction):** A signal available at decision time predicts H1's benefit beyond output-entropy/disagreement baselines.
- **H3 (mechanism):** A specific donor feature/computation explains a reproducible part of H1's improvement (corrupt→dies, restore→recovers).

## The four gates (each can kill the project honestly)
| Gate | Test | Pass | Fail → |
|---|---|---|---|
| **0. Measurement** | Evaluator controls: known-valid passes, known-invalid fails, no-op identity, independent recompute of headline table | All pass | STOP. Fix eval before anything |
| **1. Diversity** | Cross-family donor candidates vs. extra receiver samples, matched GPU-time | Donor adds beyond sampling | STOP. No transfer story exists |
| **2. Transfer** | Donor transfer beats: no-op hook, shuffled donor, norm-matched random vectors, receiver-self, text hints (matched info) | Beats ALL controls on held-out | STOP. Write negative-mechanism paper |
| **3. Mechanism+gate** | Layer/feature localization with corrupt/restore; predictor beats entropy & always/never-transfer | Reproducible mechanism OR predictive edge | Publish Gates 1–2 alone |

## Models
- Primary pair: **Qwen2.5-Coder-1.5B (donor) → CodeLlama/DeepSeek-Coder ~1.5–3B (receiver)** — cross-family is the point
- Control pair: Qwen 1.5B → Qwen 3B (same-family — expected weak, that's the control)
- Platform: g6e.xlarge (L40S 48GB), already owned

## Benchmarks & splits (frozen NOW)
- HumanEval-Java (163) + Defects4J subset. Train/dev/test split by function family BEFORE any tuning. The old 163-bug set is DEV-ONLY (contaminated by inspection); final claims need the held-out split.
- Ground truth: JUnit execution ONLY. String similarity banned as evidence.

## Leakage policy (the mistake we will not repeat)
- Reference code NEVER enters any decision, selection, hint, or gate input at inference.
- Gate inputs: only data available at decision time (prefix hidden states, output logits so far, input features).
- Every run logs: code hash, checkpoint IDs, tokenizer rev, seeds, decoding config, prompt hash, timing.

## Baselines (all at matched compute)
always-receiver · always-transfer · random gate · receiver entropy · output-disagreement · extra receiver sampling (THE killer baseline) · text hints

## Controls (for H1/H3)
no-op hook · shuffled donor (wrong bug's states) · norm-matched random vectors · receiver-to-itself transfer · unrelated-task donor states

## Statistics
- Primary: paired McNemar (exact) per independent bug. Effect sizes + CIs, not just p.
- Power: pilot (n≈50) for variance → then size final eval. Reference: 3pp paired gain at discordance ~0.2 needs ~1,700 examples — small effects need the big benchmark or we don't claim them.
- Multiple comparisons: exploratory findings need confirmation on held-out before entering the paper as claims.

## Amendment log
_(empty — v1 frozen)_
