# transfer_mech

**Does a single injected activation vector carry program-repair information from one code model to another?** A pre-registered, execution-verified study on 215 held-out Java bugs. In this setup, no usable signal transfers: the trained connector the field recommends makes the receiver *worse*, and the one nominally positive exploratory cell (p = 0.043) does not survive Holm correction.

[![verify](https://github.com/puranikyashaswin/transfer_mech/actions/workflows/verify.yml/badge.svg)](https://github.com/puranikyashaswin/transfer_mech/actions/workflows/verify.yml)

Donor: Qwen2.5-Coder-1.5B-Instruct (layer 14). Receiver: Qwen2.5-Coder-3B-Instruct. One mapped residual vector, added once at the final prompt token. Every outcome comes from compiling the patch with `javac` and running its JUnit suite; nothing is scored by string match.

## Results (held-out test split, n = 215, one frozen shot)

| Contrast | Result | Exact McNemar |
|---|---|---|
| Trained adapter vs. no intervention (receiver layer 12, Amendment 006) | **51.6% vs 63.7%**, paired effect −12.1 pp, 95% CI [−18.3, −4.6] | **p = 0.0013** |
| Compile failures, adapter vs. no intervention | **22.3% vs 2.3%** | |
| Ridge map vs. trained adapter (corrected site, receiver layer 17, Amendment 008) | 60.5% vs 59.5%, 95% CI on Δ [−5.7, +7.5] pp | p = 0.890 |

- A zero-training ridge map beat the adapter 13–1 on the dev split (p = 0.0018). That advantage **did not survive** the held-out test split (above), and the paper reports it as a split confound rather than a finding.
- On the exploratory split the donor's fix as plain *text* lifts the receiver from 26.0% to 47.9%, while the same information as an injected vector does not beat a norm-matched random vector. So the information exists; the single-vector channel does not carry it.

Full write-up: [`paper/PAPER.md`](paper/PAPER.md).

## Check the numbers yourself (CPU, no dependencies, ~1 s)

```bash
make verify     # recomputes every number above (28 checks) from the raw per-bug logs
make test       # 38 regression and harness unit tests
```

`scripts/verify_headline.py` reads the per-bug JSONL logs in `results/` (one row per bug × arm, with the generated code and its JUnit status), recomputes pass counts, discordant pairs, exact McNemar p-values and CIs with the standard library only, and exits non-zero if any value drifts from the paper. CI runs both on every push.

```
n_test                         paper       215  logs       215  ok
adapter_vs_receiver_p          paper    0.0013  logs    0.0013  ok
adapter_compile_error_pct      paper      22.3  logs      22.3  ok
l17_ridge_vs_adapter_p         paper      0.89  logs      0.89  ok
...
28/28 headline numbers reproduced from raw logs.
```

## The evaluation harness

`eval_harness/junit_runner.py` is the only source of ground truth. A candidate passes iff it compiles and its JUnit suite passes.

- **Gate 0, harness validation before any model run** (`results/gate0/`): on all 163 HumanEval-Java bugs, 163/163 known-correct references pass, 0/163 known-buggy references pass, re-running 25 no-op candidates gives 0 mismatches, and an independent recompute agrees 163/163. One real harness bug was caught here: a reference needed `javax.xml.bind`, which Java 11 removed.
- **Rebuilt confirmatory corpus** (`results/confirmatory_corpus/`): 335 bugs from MultiPL-E HumanEval-Java and MBPP-Java. A problem is kept only if a correct translation passes its full JUnit suite *and* a deterministic mutation produces a variant that compiles and fails at least one test. Pipeline: `translate → gate → mutate → build_manifest → fix_tests`.
- **Eight arms per bug**: receiver baseline, adapter, norm-matched random vector (5 seeded draws), shuffled donor state, self-injection, unrelated-task vector, text hint (positive control), donor baseline.
- **Failure taxonomy** (`results/taxonomy/`): every compile error recompiled and classified, so "the adapter is worse" comes with *how* it breaks code.

## Pre-registration

[`prereg/`](prereg/) holds the frozen protocol and eight dated amendments. Each amendment was written before the run it governs, and each pivot is recorded rather than silently applied. The held-out split was never used for tuning; all piloting ran on a separate dev split.

The files are published exactly as frozen. A few point to an internal run log and runbook (`METHODOLOGY_NOTES.md`, `GATE0_RUNBOOK.md`) that hold infrastructure details and are not included. One date error and one deviation from Amendment 008 are recorded in [`prereg/ERRATA.md`](prereg/ERRATA.md).

## Layout

```
eval_harness/   JUnit execution harness, gate checks, adapter / ridge-map training and evaluation
prereg/         protocol v1 + amendments 001–008
results/        raw per-bug JSONL logs, summaries, corpus manifest and source, taxonomy
paper/          PAPER.md, REPRODUCE.md, STATS.md, and the scripts that build every table and figure
scripts/        verify_headline.py, the dependency-free headline check
tests/          pytest suite (headline regression + harness helpers)
runner/         EC2 orchestration for GPU runs (all credentials via environment variables)
```

## Reproducing the GPU runs

One ≥24 GB GPU, Python 3.10+, JDK 11+. See [`paper/REPRODUCE.md`](paper/REPRODUCE.md) for every command and its expected output.

```bash
pip install -r requirements-gpu.txt
```

Trained adapter and ridge-map checkpoints (`*.pt`) are not tracked in git; `eval_harness/train_adapter.py` and `eval_harness/fit_linear_map.py` regenerate them (commands in `paper/REPRODUCE.md`). The exploratory HumanEval-Java split uses the HumanEval-Java benchmark ([lin-tan/clm](https://github.com/lin-tan/clm)); the confirmatory corpus is derived from [MultiPL-E](https://github.com/nuprl/MultiPL-E) and is included here.

## Limitations

- One model pair from one family and one injection site at confirmatory strength. The positive results in the literature use multi-token, multi-layer or attention-based connectors, and this study makes no claim about those.
- n = 215 is below the pre-registered powered n (859–1,411). A difference under ~9 pp would not reach significance, so the ridge-vs-adapter tie is a failure to detect a difference, not proof of equivalence.

## License

Code: MIT. Benchmark-derived data keeps its upstream license.
