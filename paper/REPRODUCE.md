# Reproduction Package — "Where Does Activation Transfer Break?"

Everything needed to reproduce every number in the paper. Requires one GPU (≥24GB, e.g. NVIDIA A10G/L40S), Python 3.10, JDK 11+, and the datasets: HumanEval-Java (163 records, exploratory studies §1–§8) and MultiPL-E-Java (rebuilt to 335 execution-verified bugs, confirmatory study §9 / paper §12).

## 0. Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate scipy
# models (downloaded on first use):
#   Qwen/Qwen2.5-Coder-1.5B-Instruct   (donor)
#   Qwen/Qwen2.5-Coder-3B-Instruct     (same-family receiver)
#   deepseek-ai/deepseek-coder-1.3b-instruct (cross-family receiver)
# JDK 11+ on PATH; dataset needs junit4 + hamcrest + jaxb-api + activation jars in <dataset>/lib/
```

Bugs file: `DGLC_ARCHIVE/05_code/experiments/data/humaneval_java.jsonl` (163 records).
Dataset: HumanEval-Java `temp_hej/` with `lib/` jars.

## 1. Gate 0 — validate the harness (do this first)

```bash
python3 eval_harness/gate0_checks.py --bugs <bugs.jsonl> --dataset <temp_hej>
# Expect: 163/163 known-valid pass, 0 unexpected known-invalid passes, 0 no-op non-determinism.
```

## 2. Cross-family pilot (Qwen → DeepSeek)

```bash
python3 eval_harness/gate1_pilot.py --bugs <bugs.jsonl> --dataset <temp_hej> \
  --out results/gate1/gate1_checkpoint.jsonl --summary results/gate1/gate1_summary.json
# Expect: receiver k=1 = 34%, pass@5 = 54%, donor k=1 = 38%, 9 discordant; naive transfer 0/9.
```

## 3. Cross-family trained adapter (Amendment 002)

```bash
# train (train split 0-79)
python3 eval_harness/train_adapter.py --bugs <bugs.jsonl> --out-dir results/adapter \
  --steps 800 --batch 8 --lr 1e-3 --alpha 1.0 --n-train 80
# eval (held-out test split 120-162, 5 arms)
python3 eval_harness/eval_adapter.py --bugs <bugs.jsonl> --dataset <temp_hej> \
  --adapter results/adapter/adapter.pt --out results/adapter/eval_checkpoint.jsonl \
  --summary results/adapter/eval_summary.json --test-start 120 --test-end 163
# Expect: adapter 4/43 (9.3%), random 11/43 (25.6%), self 11/43, no-op 9/43 (20.9%), shuffled 6/43.
```

## 4. Same-family pilot (Qwen → Qwen-3B)

```bash
python3 eval_harness/same_family_pilot.py --bugs <bugs.jsonl> --dataset <temp_hej> \
  --out results/same_family/sf_checkpoint.jsonl --summary results/same_family/sf_summary.json \
  --alpha 1.0 --eval-start 90 --eval-end 163
# Expect: transfer 23/73, random 23/73, self 22/73, no-op 19/73.
```

## 5. Alpha sweep (specificity across strength)

```bash
python3 eval_harness/sf_alpha_sweep.py --bugs <bugs.jsonl> --dataset <temp_hej> \
  --out results/alpha_sweep/alpha_checkpoint.jsonl --summary results/alpha_sweep/alpha_summary.json \
  --alphas 0.1,0.5,1.0 --eval-start 90 --eval-end 163 \
  --pilot-ckpt results/same_family/sf_checkpoint.jsonl
# Expect: at every alpha, transfer ≈ random (never above); see Table in §6.
```

## 6. Full control set (Amendment 003) — §6.5 / Result IIb

The pilot (§4) left C5 and the text-hint control un-run. The rigorous rerun completes the control set on all 73 held-out bugs, with random vectors **per-bug-seeded** and drawn **R=5** times each.

```bash
python3 eval_harness/rigorous_rerun.py --bugs <bugs.jsonl> --dataset <temp_hej> \
  --out results/rigorous/rigorous_checkpoint.jsonl --summary results/rigorous/rigorous_summary.json \
  --alpha 1.0 --r-draws 5 --eval-start 90 --eval-end 163
# Expect: receiver 19/73, donor 17/73, unrelated-task 19/73, self-inject 22/73,
#         random(mean) 0.304, transfer 23/73, text-hint 35/73 (highest).
#         text-hint >> transfer McNemar p=0.023; transfer vs self p=1.000.
```

## 7. Mechanistic probe — §7.5 / Result IV

```bash
python3 eval_harness/mech_check.py --bugs <bugs.jsonl> --out results/mech/mech_results.json
# 30 bugs (positions 90-119 of the alphabetically-sorted list = MAXIMUM_K..SAME_CHARS;
# an arbitrary positional slice, not content-based). Inject delta at receiver L17, read
# final-layer residual + next-token logits. Expect transfer==self (cos 0.965, KL 0.20),
# argmax moved on 0/30 for transfer+self vs 16/30 for random (per-bug absence, robust).
```

## 8. 2×2 site sweep (Amendment 004) — §7.6 / Result V

```bash
python3 eval_harness/site_sweep.py --bugs <bugs.jsonl> --dataset <temp_hej> \
  --out results/site_sweep/site_checkpoint.jsonl --summary results/site_sweep/site_summary.json \
  --eval-start 90 --eval-end 163
# Cells: s1_t1, s1_tall, sall_t1, sall_tall (each with own matched random control).
# Expect: s1_t1 null; s1_tall transfer 0/73; sall_t1 transfer 0.329 vs random 0.178
#         (+15.1pp, nominal p=0.043, does NOT survive 4-cell Holm); sall_tall 0/73.
```

## 9. Confirmatory corpus + one-shot eval (Amendment 006) — §12

The confirmatory study uses a **rebuilt MultiPL-E-Java corpus** (not the 163-bug HumanEval-Java set), rebuilt end-to-end by `hej_pipeline/` and execution-verified. All commands run on the GPU instance (JDK 11+, GPU).

```bash
# 9a. Rebuild the corpus (translate -> gate -> mutate -> manifest -> fix tests)
cd hej_pipeline
python3 translate.py          # MultiPL-E parquet -> 543 Java completions (3B, greedy)
python3 gate.py               # keep only completions whose translation PASSES all tests -> 345
python3 mutate.py             # 10 operators; keep a mutant that COMPILES and FAILS >=1 test -> 335
python3 build_manifest.py     # -> data/confirmatory_hej/bugs_manifest.jsonl (335 rows)
python3 fix_tests.py          # default-package the TESTs, class-qualify calls (<BUG>.method(...))
# Expect: bugs_manifest.jsonl with 335 rows; frozen split train 0-79 / dev 80-119 / test 120-end=215.

# 9b. Retrain the adapter on the rebuilt train split (Amendment 002 B2)
cd ../eval_harness
python3 train_adapter.py --bugs ../data/confirmatory_hej/bugs_manifest.jsonl \
  --out-dir ../results/confirmatory --steps 1500 --batch 8 --lr 1e-3 --wd 0.01 --alpha 1.0 --n-train 80
# Expect: adapter.pt saved; final g_mean ~= 0.143.

# 9c. ONE-SHOT confirmatory eval (8 arms, frozen test split, exactly once)
python3 eval_adapter.py --bugs ../data/confirmatory_hej/bugs_manifest.jsonl \
  --dataset ../data/confirmatory_hej \
  --adapter ../results/confirmatory/adapter.pt \
  --out ../results/confirmatory/eval_out.jsonl \
  --summary ../results/confirmatory/eval_summary.json \
  --test-start 120 --test-end 335
# Expect: eval_out.jsonl rows {bug_id, arm, status, code, seed, gen_s[, draw]}; 8 arms/bug,
#         random_norm recorded as 5 draws. Execution-verified (default-package JUnit).

# 9d. Confirmatory statistics (Amendment 006 sec.5)
~/.venvs/doctools/bin/python paper/make_stats_confirmatory.py
# Reads results/confirmatory/eval_out.jsonl. Prints per-arm Wilson CIs, paired exact
# McNemar (adapter vs receiver_k1 and vs random_norm-majority), win-share + exact
# Clopper-Pearson CI (claim threshold 0.60), and the honest-n equivalence bound.

# 9e. One-command paper integration (after the run completes)
~/.venvs/doctools/bin/python paper/fill_confirmatory.py
# Runs make_stats_confirmatory.py (single source of truth), requires >=215 complete bugs
# (else refuses), and fills the [[CONFIRMATORY_RESULTS_PLACEHOLDER]] in PAPER.md (§12.4)
# and main.tex. Per Amendment 006 §5 it always states 'no headline effect claimed'.
```

## 10. Statistics + figures

```bash
~/.venvs/doctools/bin/python paper/make_stats.py     # recomputes every exploratory p-value / CI / Holm from raw JSONL
python3 paper/make_tables.py                          # fast stdlib-only raw-count integrity check (no scipy)
~/.venvs/doctools/bin/python paper/make_figures.py   # reads results/*, writes paper/figures/
~/.venvs/doctools/bin/python paper/make_stats_confirmatory.py  # confirmatory (Amendment 006) analyses
```

## Controls (mechanically pinned, Amendment 001)

| id | name | what it is |
|---|---|---|
| C1 | no-op | receiver baseline, no injection |
| C2 | random-norm | random vector of the same norm |
| C3 | shuffled | map applied to a different bug's donor state |
| C4 | self-inject | receiver's own hidden state re-injected |
| C5 | receiver-sampling | receiver pass@k under frozen sampler |

**The load-bearing control is C2 (random-norm).** If a transfer result does not beat C2, the effect is perturbation, not information.

## Fixed config (frozen before experiments)

donor L14 → receiver L12 (DeepSeek) / L17 (Qwen-3B); final-prompt-token additive injection; temperature 0.2, top-p 0.95, seed 42, max 512 new tokens; ridge λ=1.0; adapter bottleneck 512, lr 1e-3, wd 0.01, ≤800 steps.

## Splits (deterministic, by sorted bug_id)

train 0–79, dev 80–119, test 120–162. Cross-family pilot 40–89; same-family eval 90–162. No test bug is used to fit or select anything.
