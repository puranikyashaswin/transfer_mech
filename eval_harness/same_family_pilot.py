#!/usr/bin/env python3
"""
same_family_pilot.py — Same-family transfer pilot: Qwen2.5-Coder-1.5B (donor)
-> Qwen2.5-Coder-3B (receiver). The positive arm of the transfer paper.

Same tokenizer + architecture as the donor, so this isolates "does hidden-state
transfer work when the family boundary is removed?" — the cross-family failure
(Amendment 002 negative) is the contrast.

Design (exploratory, firewall per Amendment A4):
  - Fit a centered ridge map (donor L14 -> receiver L17, both 1536-dim residual
    streams) on train bugs only; norm-match to the receiver's native norm.
  - Evaluate on held-out bugs (idx 90-162, disjoint from the train fit 0-40 and
    the cross-family pilot 40-89), with controls.
  - Arms per bug: receiver_k1 (baseline), transfer (norm-matched ridge),
    random_norm (same-norm random vector), self_inject (receiver's own state).
  - Record discordant bugs (receiver_k1 fail, transfer pass) = the flips.

Runs ON the GPU instance. Resumable per (bug, arm). NOT for the MacBook.
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
DONOR_LAYER = 14      # donor (1.5B, 28 layers) midpoint
RECEIVER_LAYER = 17   # receiver (3B, 36 layers) midpoint
RIDGE_LAMBDA = 1.0
N_TRAIN_FIT = 40
EVAL_START = 90       # held-out: disjoint from train fit (0-39) and pilot (40-89)
EVAL_END = 163


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eval-start", type=int, default=EVAL_START)
    ap.add_argument("--eval-end", type=int, default=EVAL_END)
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

    # ---- fit the centered ridge map on train bugs ----
    train_bugs = bugs[:N_TRAIN_FIT]
    print(f"fitting centered ridge on {len(train_bugs)} train bugs (L{ DONOR_LAYER}->L{RECEIVER_LAYER})...", flush=True)
    dvecs, rvecs = [], []
    for b in train_bugs:
        p = PROMPT_TMPL.format(buggy_code=b["buggy_code"])
        dvecs.append(hidden(donor, p, DONOR_LAYER).float().cpu())
        rvecs.append(hidden(receiver, p, RECEIVER_LAYER).float().cpu())
    R = torch.stack(rvecs).float()
    W, d_mean, r_mean = fit_ridge_centered(dvecs, rvecs, RIDGE_LAMBDA)
    W, d_mean, r_mean = W.cuda(), d_mean.cuda(), r_mean.cuda()
    native_norm = float(R.norm(dim=-1).mean())
    print(f"ridge fit done; receiver native norm at L{RECEIVER_LAYER} = {native_norm:.2f}", flush=True)

    def map_norm_match(h_d, alpha):
        mapped = (W @ (h_d.float() - d_mean) + r_mean)
        mapped = mapped / (mapped.norm() + 1e-8) * native_norm
        return (alpha * mapped).to(torch.bfloat16)

    # ---- eval on held-out bugs ----
    eval_bugs = bugs[args.eval_start:args.eval_end]
    print(f"held-out eval bugs: {len(eval_bugs)} (idx {args.eval_start}-{args.eval_end-1})", flush=True)

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
                done.add((r["bug_id"], r["arm"]))
        if done:
            print(f"[resume] {len(done)} (bug,arm) done", flush=True)

    outf = open(outp, "a")
    t0 = time.time()
    for bug in eval_bugs:
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        h_d = hidden(donor, prompt, DONOR_LAYER)
        h_r = hidden(receiver, prompt, RECEIVER_LAYER)
        delta_transfer = map_norm_match(h_d, args.alpha)

        arms = {}
        if (bug_id, "receiver_k1") not in done:
            code, _ = receiver.generate(prompt, seed=BASE_SEED)
            arms["receiver_k1"] = eval_code(bug_id, code)
        if (bug_id, "transfer") not in done:
            arms["transfer"] = eval_code(bug_id, gen_with_delta(prompt, delta_transfer, BASE_SEED))
        if (bug_id, "random_norm") not in done:
            g = torch.Generator(device="cuda").manual_seed(BASE_SEED)
            rv = torch.randn(h_r.shape, generator=g, device="cuda", dtype=torch.float32)
            rv = rv / rv.norm() * h_r.norm()
            arms["random_norm"] = eval_code(bug_id, gen_with_delta(prompt, rv.to(torch.bfloat16), BASE_SEED))
        if (bug_id, "self_inject") not in done:
            arms["self_inject"] = eval_code(bug_id, gen_with_delta(prompt, h_r.to(torch.bfloat16), BASE_SEED))

        for arm, status in arms.items():
            outf.write(json.dumps({"bug_id": bug_id, "arm": arm, "status": status,
                                   "passed": status == "pass"}) + "\n")
            outf.flush()
        if arms:
            print(f"{bug_id:<26} " + "  ".join(f"{a}={s}" for a, s in arms.items()), flush=True)

    outf.close()
    # summary
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    arms_list = ["receiver_k1", "transfer", "random_norm", "self_inject"]
    table = {}
    for arm in arms_list:
        sub = [r for r in rows if r["arm"] == arm]
        npass = sum(r["passed"] for r in sub)
        table[arm] = {"n": len(sub), "pass": npass, "acc": npass / max(1, len(sub))}
    # flips: transfer pass, baseline fail
    bybug = {}
    for r in rows:
        bybug.setdefault(r["bug_id"], {})[r["arm"]] = r["status"]
    flips = sorted([b for b, a in bybug.items()
                    if a.get("transfer") == "pass" and a.get("receiver_k1") != "pass"])
    broke = sorted([b for b, a in bybug.items()
                    if a.get("transfer") != "pass" and a.get("receiver_k1") == "pass"])
    summary = {"n_eval": len(eval_bugs), "alpha": args.alpha,
               "donor_layer": DONOR_LAYER, "receiver_layer": RECEIVER_LAYER,
               "native_norm": round(native_norm, 2), "arms": table,
               "flips": flips, "n_flips": len(flips),
               "broke": broke, "n_broke": len(broke),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print("\n==== SAME-FAMILY PILOT SUMMARY ====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
