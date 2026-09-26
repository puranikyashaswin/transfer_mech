#!/usr/bin/env python3
"""
eval_baseline_map.py — evaluate the baseline linear (ridge) map on the DEV split.

Answers the critic's P1.3 question: is the confirmatory negative specific to the
trained adapter's architecture, or is single-vector cross-model injection broken
in general at this site?

Firewall: the map was fit on train (0-79) ONLY by fit_linear_map.py. This script
APPLIES the frozen map to dev bugs (80-119, n=40) under the EXACT confirmatory
decoding config (alpha=1.0, temp 0.2, top-p 0.95, seed 42, 1024 new tokens,
final-prompt-token additive injection at receiver layer 12). Test split untouched.

EXPLORATORY/DIAGNOSTIC — makes NO confirmatory claim. Comparison cell: receiver_k1
dev baseline (57.5%, from results/pilot/pilot_baseline.jsonl) and adapter dev cells
(results/pilot/pilot_full.jsonl).

Runs ON the GPU instance. NOT for the MacBook.
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
from junit_runner import evaluate_candidate_default_pkg  # noqa: E402
import gate1_pilot  # noqa: E402
from gate1_pilot import DONOR_ID, RECEIVER_ID, PROMPT_TMPL, BASE_SEED, Model, load_bugs  # noqa: E402
from train_adapter import DONOR_LAYER, RECEIVER_LAYER  # noqa: E402

gate1_pilot.MAX_NEW_TOKENS = 1024  # match confirmatory config (Amendment 005)
CODE_TRUNC = 4000


def retarget_class_name(code: str, bug_id: str) -> str:
    import re
    m = re.compile(r"\bpublic\s+(?:final\s+)?class\s+[A-Za-z_][A-Za-z0-9_]*")
    return m.sub(f"public class {bug_id}", code, count=1) if m.search(code) else code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--dataset", required=True, help="confirmatory_hej root (lib/ jars)")
    ap.add_argument("--map", required=True, help=".pt from fit_linear_map.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-start", type=int, default=80)
    ap.add_argument("--dev-end", type=int, default=120)
    ap.add_argument("--alphas", default="1.0")
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--junit-cp", default=None)
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    root = Path(args.dataset)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))
    data = {}
    for line in open(args.bugs):
        r = json.loads(line)
        data[r["bug_id"]] = {"test": r["tests"]}
    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    dev_bugs = bugs[args.dev_start:args.dev_end]
    print(f"baseline-map dev bugs: {len(dev_bugs)} (idx {args.dev_start}-{args.dev_end-1})", flush=True)

    mp = torch.load(args.map, map_location="cuda")
    W = mp["W"].cuda()            # [d_r, d_d]
    d_mean = mp["d_mean"].cuda()  # [d_d]
    r_mean = mp["r_mean"].cuda()  # [d_r]
    print(f"loaded baseline map: W {list(W.shape)}, train R^2={mp.get('train_r2', float('nan')):.4f}",
          flush=True)

    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)

    def eval_code(bug_id, code):
        ent = data.get(bug_id) or data.get(f"humaneval_java_{bug_id}")
        if ent is None:
            return "error"
        test_src = ent["test"]
        code = retarget_class_name(code, bug_id)
        with tempfile.TemporaryDirectory() as wd:
            r = evaluate_candidate_default_pkg(bug_id, code, test_src, args.javac, args.java, junit_cp, wd)
        return r["status"]

    def gen_with_delta(prompt, delta, seed):
        def hook_fn(hs, input_len, _d=delta):
            return _d
        code, _ = receiver.generate(prompt, seed=seed, hook_layer=RECEIVER_LAYER, hook_fn=hook_fn)
        return code

    outf = open(args.out, "a")
    done = set()
    if Path(args.out).exists():
        for line in Path(args.out).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["bug_id"], r["alpha"]))
        if done:
            print(f"[resume] {len(done)} cells done", flush=True)

    t0 = time.time()
    for bug in dev_bugs:
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        h_d = donor.hidden_at_final_token(prompt, DONOR_LAYER).cuda().float()
        with torch.no_grad():
            raw = W @ (h_d - d_mean)   # delta directly (adapter-parallel semantics:
                                       # fit predicts h_r - r_mean, so W@(h_d-d_mean) IS the delta)
        for alpha in alphas:
            if (bug_id, alpha) in done:
                continue
            delta = alpha * raw
            delta_norm = float(delta.norm())
            d_bf = delta.to(torch.bfloat16)
            t1 = time.time()
            code = gen_with_delta(prompt, d_bf, BASE_SEED)
            status = eval_code(bug_id, code)
            rec = {"bug_id": bug_id, "arm": "baseline_ridge", "alpha": alpha,
                   "status": status, "delta_norm": round(delta_norm, 4),
                   "code": code[:CODE_TRUNC], "gen_s": round(time.time() - t1, 3),
                   "seed": BASE_SEED}
            outf.write(json.dumps(rec) + "\n")
            outf.flush()
            print(f"{bug_id:<24} a={alpha} {status:<13} |d|={delta_norm:.2f}", flush=True)
    outf.close()
    print(f"\nbaseline-map eval done in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
