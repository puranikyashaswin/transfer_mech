#!/usr/bin/env python3
"""
gate1_sweep.py — EXPLORATORY intervention sweep for "What Transfers?" (Gate 1).

Motivated by the transfer literature (see paper/PAPER.md, Related Work). The naive ridge transfer gave 0/9 flips. The literature
(Pythia negative result; StateBridge; MasterKey; Bangachev; Gao) says the fix is:
  1. CENTER activations (subtract per-model dataset mean) before fitting.
  2. NORM-MATCH the mapped vector to the receiver's NATIVE hidden-state norm at
     the injection site (hidden norms can be ~100x off; this is the make-or-break).
  3. SWEEP injection strength alpha (no universal value) and donor/receiver layer.
  4. ADDITIVE injection only (replacement is destructive).

This script sweeps a small grid on the DISCORDANT bugs (donor pass, receiver k=1
fail) from the pilot, and reports the causal flip rate per config. Exploratory
track only — the confirmatory test is a separate, frozen config.

Controls (per Pythia paper) are included:
  - self_inject: receiver's OWN hidden state re-injected (hook sanity)
  - random_norm: same-norm RANDOM vector (magnitude control)
  - shuffled: donor vector from a DIFFERENT bug (signal-specificity control)

Runs ON the GPU instance. Resumable per (bug, config). NOT for the MacBook.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from junit_runner import (  # noqa: E402
    evaluate_candidate, extract_class_name, load_dataset, wrap_candidate_method,
)
from gate1_pilot import (  # noqa: E402
    DONOR_ID, RECEIVER_ID, PROMPT_TMPL, MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
    BASE_SEED, Model, load_bugs,
)


def fit_ridge_centered(donor_vecs, recv_vecs, lam):
    """Centered ridge: subtract per-model means, fit map, return (W, d_mean, r_mean).
    Map is applied as:  y = W @ (x - d_mean) + r_mean  (then norm-matched)."""
    X = torch.stack(donor_vecs).double()
    Y = torch.stack(recv_vecs).double()
    d_mean = X.mean(0, keepdim=True)
    r_mean = Y.mean(0, keepdim=True)
    Xc = X - d_mean
    Yc = Y - r_mean
    d = X.shape[1]
    A = Xc.T @ Xc + lam * torch.eye(d, dtype=torch.double)
    B = Xc.T @ Yc
    W = torch.linalg.solve(A, B).T.float()  # [2048, 1536]
    return W, d_mean.squeeze(0).float(), r_mean.squeeze(0).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pilot-ckpt", required=True, help="gate1 checkpoint (for discordance)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--n-train-fit", type=int, default=40)
    ap.add_argument("--alphas", default="0.1,0.3,1.0")
    ap.add_argument("--donor-layers", default="14")
    ap.add_argument("--receiver-layers", default="12")
    ap.add_argument("--norm-match", type=int, default=1, help="1=rescale to native norm")
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--junit-cp", default=None)
    args = ap.parse_args()

    root = Path(args.dataset)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))
    data = load_dataset(root)
    bugs = load_bugs(Path(args.bugs))
    buggy_snippet = {b["bug_id"].replace("humaneval_java_", ""): b["buggy_code"] for b in bugs}

    # discordant bugs from the pilot
    pilot = [json.loads(l) for l in Path(args.pilot_ckpt).read_text().splitlines() if l.strip()]
    discordant = [r["bug_id"] for r in pilot if r.get("discordant")]
    print(f"discordant bugs to attempt transfer on: {len(discordant)}", flush=True)

    alphas = [float(a) for a in args.alphas.split(",")]
    d_layers = [int(x) for x in args.donor_layers.split(",")]
    r_layers = [int(x) for x in args.receiver_layers.split(",")]

    # resume
    outp = Path(args.out)
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["bug_id"], r["config_id"]))
        if done:
            print(f"[resume] {len(done)} (bug,config) done", flush=True)

    print("Loading models...", flush=True)
    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)

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

    # native receiver norm at injection site (mean over train bugs) per receiver layer
    train_bugs = sorted(bugs, key=lambda b: b["bug_id"])[: args.n_train_fit]
    train_prompts = [PROMPT_TMPL.format(buggy_code=b["buggy_code"]) for b in train_bugs]

    outf = open(outp, "a")
    t0 = time.time()
    for dl in d_layers:
        for rl in r_layers:
            # fit centered ridge for this layer pair
            dvecs = [donor.hidden_at_final_token(p, dl) for p in train_prompts]
            rvecs = [receiver.hidden_at_final_token(p, rl) for p in train_prompts]
            W, d_mean, r_mean = fit_ridge_centered(dvecs, rvecs, lam=1.0)
            W = W.cuda(); d_mean = d_mean.cuda(); r_mean = r_mean.cuda()
            # native receiver norm at this layer
            native_norm = torch.stack(rvecs).norm(dim=1).mean().item()

            for alpha in alphas:
                cfg = f"d{dl}_r{rl}_a{alpha}_nm{args.norm_match}"
                for bug_id in discordant:
                    if (bug_id, cfg) in done:
                        continue
                    b = next(x for x in bugs if x["bug_id"].replace("humaneval_java_", "") == bug_id)
                    prompt = PROMPT_TMPL.format(buggy_code=b["buggy_code"])
                    h_d = donor.hidden_at_final_token(prompt, dl).cuda()
                    mapped = W @ (h_d - d_mean) + r_mean  # centered map
                    if args.norm_match:
                        n = mapped.norm()
                        if n > 0:
                            mapped = mapped * (native_norm / n)
                    delta = (alpha * mapped).to(torch.bfloat16)

                    def hook_fn(hs, input_len, _d=delta):
                        return _d
                    code, _ = receiver.generate(prompt, seed=BASE_SEED,
                                                hook_layer=rl, hook_fn=hook_fn)
                    status = eval_code(bug_id, code)
                    rec = {"bug_id": bug_id, "config_id": cfg, "donor_layer": dl,
                           "receiver_layer": rl, "alpha": alpha,
                           "norm_match": args.norm_match, "status": status,
                           "flipped": status == "pass",
                           "native_norm": round(native_norm, 2),
                           "mapped_norm": round(float(mapped.norm()), 2)}
                    outf.write(json.dumps(rec) + "\n"); outf.flush()
                    print(f"[{cfg}] {bug_id:<26} -> {status} flip={rec['flipped']}", flush=True)

    outf.close()
    # summary
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config_id"], []).append(r)
    table = {c: {"n": len(v), "flips": sum(x["flipped"] for x in v),
                 "flip_rate": sum(x["flipped"] for x in v) / max(1, len(v))}
             for c, v in sorted(by_cfg.items())}
    summary = {"n_discordant": len(discordant), "configs": table,
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print("\n==== SWEEP SUMMARY ====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
