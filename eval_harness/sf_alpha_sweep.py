#!/usr/bin/env python3
"""
sf_alpha_sweep.py — Same-family alpha sweep, the airtight specificity test.

The pilot showed transfer (31.5%) == random-norm (31.5%) at alpha=1.0. To claim
"single-vector transfer does not exceed a matched random perturbation" rigorously,
we sweep alpha and check whether ANY alpha makes transfer beat the SAME-alpha
random control. If the curves track each other, the effect is pure perturbation.

Per bug per alpha, arms: transfer, random_norm, self_inject.
Baseline receiver_k1 is reused from the pilot checkpoint (no re-run).

Runs ON the GPU instance. Resumable per (bug, alpha, arm). NOT for the MacBook.
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--alphas", default="0.05,0.1,0.3,0.5,1.0,2.0")
    ap.add_argument("--eval-start", type=int, default=90)
    ap.add_argument("--eval-end", type=int, default=163)
    ap.add_argument("--pilot-ckpt", default=None, help="pilot sf_checkpoint.jsonl for receiver_k1 baseline")
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--junit-cp", default=None)
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    root = Path(args.dataset)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))
    data = load_dataset(root)
    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    buggy_snippet = {b["bug_id"].replace("humaneval_java_", ""): b["buggy_code"] for b in bugs}

    # baseline receiver_k1 from pilot checkpoint
    baseline = {}
    if args.pilot_ckpt and Path(args.pilot_ckpt).exists():
        for line in Path(args.pilot_ckpt).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["arm"] == "receiver_k1":
                    baseline[r["bug_id"]] = r["status"]
    print(f"loaded {len(baseline)} receiver_k1 baselines from pilot", flush=True)

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
    W, d_mean, r_mean = fit_ridge_centered(dvecs, rvecs, RIDGE_LAMBDA)
    W, d_mean, r_mean = W.cuda(), d_mean.cuda(), r_mean.cuda()
    native_norm = float(torch.stack(rvecs).norm(dim=-1).mean())
    print(f"ridge fit done; native norm = {native_norm:.2f}", flush=True)

    def map_norm_match(h_d, alpha):
        mapped = (W @ (h_d.float() - d_mean) + r_mean)
        mapped = mapped / (mapped.norm() + 1e-8) * native_norm
        return (alpha * mapped).to(torch.bfloat16)

    eval_bugs = bugs[args.eval_start:args.eval_end]
    print(f"eval bugs: {len(eval_bugs)}, alphas: {alphas}", flush=True)

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
                done.add((r["bug_id"], r["alpha"], r["arm"]))
        if done:
            print(f"[resume] {len(done)} (bug,alpha,arm) done", flush=True)

    outf = open(outp, "a")
    t0 = time.time()
    for bug in eval_bugs:
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        h_d = hidden(donor, prompt, DONOR_LAYER)
        h_r = hidden(receiver, prompt, RECEIVER_LAYER)
        for alpha in alphas:
            delta_transfer = map_norm_match(h_d, alpha)
            for arm, delta in (("transfer", delta_transfer),):
                if (bug_id, alpha, arm) in done:
                    continue
                status = eval_code(bug_id, gen_with_delta(prompt, delta, BASE_SEED))
                outf.write(json.dumps({"bug_id": bug_id, "alpha": alpha, "arm": arm,
                                       "status": status, "passed": status == "pass"}) + "\n")
                outf.flush()
            # random_norm at the same alpha-equivalent norm
            if (bug_id, alpha, "random_norm") not in done:
                g = torch.Generator(device="cuda").manual_seed(BASE_SEED)
                rv = torch.randn(h_r.shape, generator=g, device="cuda", dtype=torch.float32)
                rv = rv / rv.norm() * (native_norm * alpha)
                status = eval_code(bug_id, gen_with_delta(prompt, rv.to(torch.bfloat16), BASE_SEED))
                outf.write(json.dumps({"bug_id": bug_id, "alpha": alpha, "arm": "random_norm",
                                       "status": status, "passed": status == "pass"}) + "\n")
                outf.flush()
            # self_inject at the same alpha
            if (bug_id, alpha, "self_inject") not in done:
                delta_self = (alpha * h_r).to(torch.bfloat16)
                status = eval_code(bug_id, gen_with_delta(prompt, delta_self, BASE_SEED))
                outf.write(json.dumps({"bug_id": bug_id, "alpha": alpha, "arm": "self_inject",
                                       "status": status, "passed": status == "pass"}) + "\n")
                outf.flush()
        print(f"{bug_id:<26} done", flush=True)

    outf.close()
    # summary: pass rate per alpha per arm
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    curve = {}
    for alpha in alphas:
        curve[str(alpha)] = {}
        for arm in ("transfer", "random_norm", "self_inject"):
            sub = [r for r in rows if r["alpha"] == alpha and r["arm"] == arm]
            npass = sum(r["passed"] for r in sub)
            curve[str(alpha)][arm] = {"n": len(sub), "pass": npass, "acc": npass / max(1, len(sub))}
    n_base_pass = sum(1 for s in baseline.values() if s == "pass")
    summary = {"n_eval": len(eval_bugs), "alphas": alphas, "native_norm": round(native_norm, 2),
               "baseline_pass": n_base_pass, "baseline_acc": n_base_pass / max(1, len(eval_bugs)),
               "curve": curve, "elapsed_min": round((time.time() - t0) / 60, 1)}
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print("\n==== ALPHA SWEEP SUMMARY ====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
