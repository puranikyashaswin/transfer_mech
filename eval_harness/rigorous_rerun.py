#!/usr/bin/env python3
"""
rigorous_rerun.py — Amendment 003 compliance run. Fixes the experimental gaps
found by the adversarial review:

  1. Per-bug-seeded random control with R independent draws (was: one global-seed
     draw per bug, non-deterministic across runs). Reports the random control as
     mean +/- spread over R draws.
  2. Text-hint control: the donor's CORRECT fix shown to the receiver as a code
     comment in the prompt ("the key information-matched control"). If the
     receiver can't use the answer when SHOWN it, the task ceiling is the binding
     constraint; if it can, activation transfer must beat this to be interesting.
  3. Unrelated-task control (Amendment 001 C5): inject the mapped donor state from
     a DIFFERENT task (a fixed reference bug) — content-bearing but off-task.
  4. Donor pass-run on the eval set, enabling the DISCORDANT-RESTRICTED analysis
     (transfer claimed only where donor passes & receiver fails).

Same-family only (Qwen-1.5B -> Qwen-3B). Resumable per (bug, arm[, draw]).
Runs ON the GPU instance. NOT for the MacBook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from junit_runner import (  # noqa: E402
    evaluate_candidate, extract_class_name, load_dataset, wrap_candidate_method,
)
from gate1_pilot import (  # noqa: E402
    DONOR_ID, PROMPT_TMPL, BASE_SEED, Model, load_bugs,
)
from gate1_sweep import fit_ridge_centered  # noqa: E402

RECEIVER_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DONOR_LAYER = 14
RECEIVER_LAYER = 17
RIDGE_LAMBDA = 1.0
N_TRAIN_FIT = 40
R_DRAWS = 5  # independent random draws per bug for the random control

TEXT_HINT_TMPL = """Fix the bug in this Java code. Return ONLY the fixed code.

## Buggy code
```java
{buggy_code}
```

## A correct reference implementation (for guidance)
```java
{correct_code}
```

Use the reference to produce the fixed code. Return ONLY the fixed code."""


def per_bug_seed(bug_id: str, arm: str, draw: int = 0) -> int:
    h = hashlib.sha256(f"{bug_id}|{arm}|{draw}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eval-start", type=int, default=90)
    ap.add_argument("--eval-end", type=int, default=163)
    ap.add_argument("--r-draws", type=int, default=R_DRAWS)
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--junit-cp", default=None)
    args = ap.parse_args()

    root = Path(args.dataset)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))
    data = load_dataset(root)
    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    buggy_snippet = {b["bug_id"].replace("humaneval_java_", ""): b["buggy_code"] for b in bugs}

    print("loading donor (1.5B) + receiver (3B)...", flush=True)
    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)

    def hidden(model, prompt, layer):
        return model.hidden_at_final_token(prompt, layer).cuda()

    train_bugs = bugs[:N_TRAIN_FIT]
    print(f"fitting centered ridge on {len(train_bugs)} train bugs...", flush=True)
    dvecs, rvecs = [], []
    for b in train_bugs:
        p = PROMPT_TMPL.format(buggy_code=b["buggy_code"])
        dvecs.append(hidden(donor, p, DONOR_LAYER).float().cpu())
        rvecs.append(hidden(receiver, p, RECEIVER_LAYER).float().cpu())
    R = torch.stack(rvecs).float()
    W, d_mean, r_mean = fit_ridge_centered(dvecs, rvecs, RIDGE_LAMBDA)
    W, d_mean, r_mean = W.cuda(), d_mean.cuda(), r_mean.cuda()
    native_norm = float(R.norm(dim=-1).mean())
    print(f"ridge fit done; native norm = {native_norm:.2f}", flush=True)

    def map_norm_match(h_d, alpha):
        mapped = (W @ (h_d.float() - d_mean) + r_mean)
        mapped = mapped / (mapped.norm() + 1e-8) * native_norm
        return (alpha * mapped).to(torch.bfloat16)

    # unrelated-task reference: a fixed off-task bug's donor state (mapped)
    ref_bug = bugs[100]  # any fixed off-task bug, deterministic
    ref_prompt = PROMPT_TMPL.format(buggy_code=ref_bug["buggy_code"])
    delta_unrelated = map_norm_match(hidden(donor, ref_prompt, DONOR_LAYER), args.alpha)

    eval_bugs = bugs[args.eval_start:args.eval_end]
    print(f"held-out eval: {len(eval_bugs)} bugs, R={args.r_draws} random draws/bug", flush=True)

    def eval_code(bug_id, code):
        test_src = data[bug_id]["test"]
        buggy_class = data[bug_id]["buggy"]
        if extract_class_name(code) is None and buggy_class is not None:
            wrapped = wrap_candidate_method(buggy_class, buggy_snippet.get(bug_id, ""), code)
            if wrapped is None:
                return "error"
            code = wrapped
        with tempfile.TemporaryDirectory() as wd:
            r = evaluate_candidate(bug_id, code, test_src, args.javac, args.java, junit_cp, wd)
        return r["status"]

    def gen_with_delta(prompt, delta, seed):
        def hook_fn(hs, input_len, _d=delta):
            return _d
        code, _ = receiver.generate(prompt, seed=seed, hook_layer=RECEIVER_LAYER, hook_fn=hook_fn)
        return code

    # resume
    outp = Path(args.out)
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["bug_id"], r["arm"], r.get("draw", 0)))
        if done:
            print(f"[resume] {len(done)} records done", flush=True)

    outf = open(outp, "a")
    t0 = time.time()
    for bug in eval_bugs:
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        h_d = hidden(donor, prompt, DONOR_LAYER)
        h_r = hidden(receiver, prompt, RECEIVER_LAYER)
        delta_transfer = map_norm_match(h_d, args.alpha)
        n = h_r.norm()

        records = []

        if (bug_id, "donor_k1", 0) not in done:
            code, _ = donor.generate(prompt, seed=BASE_SEED)
            records.append((bug_id, "donor_k1", 0, eval_code(bug_id, code)))
        if (bug_id, "receiver_k1", 0) not in done:
            code, _ = receiver.generate(prompt, seed=BASE_SEED)
            records.append((bug_id, "receiver_k1", 0, eval_code(bug_id, code)))
        if (bug_id, "transfer", 0) not in done:
            records.append((bug_id, "transfer", 0, eval_code(bug_id, gen_with_delta(prompt, delta_transfer, BASE_SEED))))
        if (bug_id, "self_inject", 0) not in done:
            records.append((bug_id, "self_inject", 0, eval_code(bug_id, gen_with_delta(prompt, h_r.to(torch.bfloat16), BASE_SEED))))
        if (bug_id, "unrelated_task", 0) not in done:
            records.append((bug_id, "unrelated_task", 0, eval_code(bug_id, gen_with_delta(prompt, delta_unrelated, BASE_SEED))))
        if (bug_id, "text_hint", 0) not in done:
            th_prompt = TEXT_HINT_TMPL.format(buggy_code=bug["buggy_code"], correct_code=bug.get("correct_code", ""))
            code, _ = receiver.generate(th_prompt, seed=BASE_SEED)
            records.append((bug_id, "text_hint", 0, eval_code(bug_id, code)))
        # multi-draw random control (per-bug seeded)
        for d in range(args.r_draws):
            if (bug_id, "random_norm", d) in done:
                continue
            g = torch.Generator(device="cuda").manual_seed(per_bug_seed(bug_id, "random_norm", d))
            rv = torch.randn(h_r.shape, generator=g, device="cuda", dtype=torch.float32)
            rv = rv / rv.norm() * n
            records.append((bug_id, "random_norm", d, eval_code(bug_id, gen_with_delta(prompt, rv.to(torch.bfloat16), BASE_SEED))))

        for bug_id_, arm, draw, status in records:
            outf.write(json.dumps({"bug_id": bug_id_, "arm": arm, "draw": draw,
                                   "status": status, "passed": status == "pass"}) + "\n")
            outf.flush()
        if records:
            summary_line = "  ".join(f"{a}{'['+str(d)+']' if a=='random_norm' else ''}={s}"
                                      for _, a, d, s in records)
            print(f"{bug_id:<24} {summary_line}", flush=True)

    outf.close()

    # ---- summary ----
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    arms_single = ["donor_k1", "receiver_k1", "transfer", "self_inject", "unrelated_task", "text_hint"]
    table = {}
    for arm in arms_single:
        sub = [r for r in rows if r["arm"] == arm]
        npass = sum(r["passed"] for r in sub)
        table[arm] = {"n": len(sub), "pass": npass, "acc": round(npass / max(1, len(sub)), 4)}
    # random: per-draw acc + mean/spread
    rnd = [r for r in rows if r["arm"] == "random_norm"]
    by_draw = {}
    for r in rnd:
        by_draw.setdefault(r["draw"], []).append(r["passed"])
    draw_accs = [sum(v) / len(v) for _, v in sorted(by_draw.items())]
    import statistics
    table["random_norm"] = {
        "n_draws": len(draw_accs),
        "per_draw_acc": [round(a, 4) for a in draw_accs],
        "mean_acc": round(statistics.mean(draw_accs), 4) if draw_accs else None,
        "spread": round(statistics.pstdev(draw_accs), 4) if len(draw_accs) > 1 else 0.0,
    }
    # discordant-restricted: donor passes & receiver fails
    bybug = {}
    for r in rows:
        bybug.setdefault(r["bug_id"], {})[(r["arm"], r.get("draw", 0))] = r["passed"]
    discordant = [b for b, a in bybug.items()
                  if a.get(("donor_k1", 0)) == 1 and a.get(("receiver_k1", 0)) == 0]
    disc_table = {}
    for arm in ["transfer", "self_inject", "unrelated_task", "text_hint"]:
        npass = sum(bybug[b].get((arm, 0), 0) for b in discordant)
        disc_table[arm] = {"discordant_n": len(discordant), "pass": npass,
                           "acc": round(npass / max(1, len(discordant)), 4)}
    rnd_disc = [sum(bybug[b].get(("random_norm", d), 0) for b in discordant) / max(1, len(discordant))
                for d in range(args.r_draws)]
    disc_table["random_norm"] = {"discordant_n": len(discordant),
                                 "per_draw_acc": [round(a, 4) for a in rnd_disc],
                                 "mean_acc": round(statistics.mean(rnd_disc), 4) if rnd_disc else None}

    summary = {"n_eval": len(eval_bugs), "alpha": args.alpha, "r_draws": args.r_draws,
               "native_norm": round(native_norm, 2),
               "all_bugs": table, "n_discordant": len(discordant),
               "discordant_restricted": disc_table,
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print("\n==== RIGOROUS RERUN SUMMARY ====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
