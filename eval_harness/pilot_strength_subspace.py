#!/usr/bin/env python3
"""
pilot_strength_subspace.py — Amendment 007 instrumented diagnostic PILOT.

Paired 2x2 on the DEV split (bugs 80-119, n=40) ONLY. Test split untouched.

  Factor 1 (strength):  alpha in {0.5, 1.0}
  Factor 2 (subspace):  adapter-as-is  vs  adapter -> receiver-PCA-aligned

Per bug x condition, logs (the data confirmatory eval_out lacked):
  - delta_norm      : ||Delta|| (injected vector L2 norm)
  - proj_frac       : fraction of ||Delta|| lying INSIDE the receiver active subspace
  - gate_mean       : adapter gate |g|.mean() (sanity; constant across conditions)
  - status          : JUnit pass / test_fail / compile_error / error

ANTI-LEAKAGE (Amendment 007 §3): the receiver active-subspace PCA is fit ONLY on
receiver layer-12 activations from the TRAIN split (bugs 0-79) via
--subspace-basis, fit by a SEPARATE script (fit_receiver_subspace.py) that reads
train bugs only. This pilot NEVER touches dev/test activations to FIT anything;
the frozen basis is only APPLIED to dev bugs. Readout is against the §5
pre-committed boundaries. EXPLORATORY/DIAGNOSTIC — makes NO confirmatory claim.

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
from train_adapter import Adapter, DONOR_LAYER, RECEIVER_LAYER  # noqa: E402

gate1_pilot.MAX_NEW_TOKENS = 1024  # Amendment 005
CODE_TRUNC = 4000


def retarget_class_name(code: str, bug_id: str) -> str:
    import re
    m = re.compile(r"\bpublic\s+(?:final\s+)?class\s+[A-Za-z_][A-Za-z0-9_]*")
    return m.sub(f"public class {bug_id}", code, count=1) if m.search(code) else code


def project_out(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the component of delta lying INSIDE the receiver active subspace.
    basis: [d, k] orthonormal columns (top-k receiver PCA directions).
    delta: [d]. Returns delta_perp = delta - P(delta)."""
    coeff = basis.T @ delta.float()          # [k]
    proj = basis @ coeff                      # [d]
    return (delta.float() - proj)


def project_in(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Keep ONLY the component of delta inside the receiver active subspace."""
    coeff = basis.T @ delta.float()
    return basis @ coeff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--dataset", required=True,
                    help="confirmatory_hej root for junit jars (lib/)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--subspace-basis", required=True,
                    help=".pt with {'basis': [d,k] tensor} fit on TRAIN split only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-start", type=int, default=80)
    ap.add_argument("--dev-end", type=int, default=120)
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--junit-cp", default=None)
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    root = Path(args.dataset)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))
    # Load tests from manifest (authoritative source; dataset dir has different key naming)
    data = {}
    for line in open(args.bugs):
        r = json.loads(line)
        data[r["bug_id"]] = {"test": r["tests"]}
    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    dev_bugs = bugs[args.dev_start:args.dev_end]
    print(f"pilot dev bugs: {len(dev_bugs)} (idx {args.dev_start}-{args.dev_end-1})", flush=True)

    basis = torch.load(args.subspace_basis, map_location="cuda")["basis"].float().cuda()  # [d,k]
    d, k = basis.shape
    print(f"receiver active subspace: d={d}, k={k} (fit on TRAIN split only)", flush=True)

    ckpt = torch.load(args.adapter, map_location="cuda")
    adapter = Adapter().cuda()
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    gate_mean = float(adapter.g.abs().mean())
    print(f"adapter gate_mean={gate_mean:.4f} (frozen)", flush=True)

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
                done.add((r["bug_id"], r["alpha"], r["subspace"]))
        if done:
            print(f"[resume] {len(done)} cells done", flush=True)

    t0 = time.time()
    for bug in dev_bugs:
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        h_d = donor.hidden_at_final_token(prompt, DONOR_LAYER).cuda()
        with torch.no_grad():
            raw = adapter(h_d).float()          # unscaled adapter output (alpha applied per-condition)
        for alpha in alphas:
            for subspace in ("as_is", "aligned"):
                if (bug_id, alpha, subspace) in done:
                    continue
                delta = alpha * raw
                if subspace == "aligned":
                    delta = project_in(delta, basis)   # keep only receiver-subspace component
                # instrumentation
                delta_norm = float(delta.norm())
                with torch.no_grad():
                    proj = project_in(delta, basis)
                    proj_frac = float(proj.norm() / (delta.norm() + 1e-9))
                d_bf = delta.to(torch.bfloat16)
                t1 = time.time()
                code = gen_with_delta(prompt, d_bf, BASE_SEED)
                status = eval_code(bug_id, code)
                rec = {"bug_id": bug_id, "alpha": alpha, "subspace": subspace,
                       "status": status, "delta_norm": round(delta_norm, 4),
                       "proj_frac": round(proj_frac, 4), "gate_mean": round(gate_mean, 4),
                       "code": code[:CODE_TRUNC], "gen_s": round(time.time() - t1, 3),
                       "seed": BASE_SEED}
                outf.write(json.dumps(rec) + "\n")
                outf.flush()
                print(f"{bug_id:<24} a={alpha} {subspace:<7} {status:<13} "
                      f"|d|={delta_norm:.2f} proj={proj_frac:.2f}", flush=True)
    outf.close()
    print(f"\npilot done in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
