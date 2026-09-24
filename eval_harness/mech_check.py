#!/usr/bin/env python3
"""
mech_check.py — Mechanistic probe: does the injected delta survive to the output,
and is its effect content-specific or generic?

Forward-only (no JUnit, cheap). For each of N eval bugs we measure, at the final
prompt token, the receiver's LAST-LAYER residual stream and next-token logits:
  - baseline (no injection)
  - transfer delta (norm-matched ridge, alpha=1.0)
  - random_norm delta (same norm)
  - self_inject delta
For each intervention we record, relative to baseline:
  - cosine similarity of last-layer residual direction (does direction change?)
  - relative norm change of last-layer residual (does magnitude change?)
  - KL divergence of next-token distribution (does the OUTPUT distribution move?)
  - whether the argmax next-token changes

If the KL / direction shift for transfer is indistinguishable from random/self,
that is direct mechanistic evidence the injection perturbs but does not inform.

Runs ON the GPU instance. NOT for the MacBook.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate1_pilot import DONOR_ID, PROMPT_TMPL, Model, load_bugs  # noqa: E402
from gate1_sweep import fit_ridge_centered  # noqa: E402

RECEIVER_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DONOR_LAYER = 14
RECEIVER_LAYER = 17
RIDGE_LAMBDA = 1.0
N_TRAIN_FIT = 40


def last_token_state_and_logits(model: Model, prompt: str, inject_layer=None, delta=None):
    """Run a forward pass; optionally inject `delta` at `inject_layer`/final prompt token.
    Returns (last_layer_residual [hidden], next_token_logits [vocab])."""
    tok = model.tok
    from gate1_pilot import apply_chat_template
    text = apply_chat_template(tok, prompt)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    n = enc["input_ids"].shape[1]

    handle = None
    if inject_layer is not None and delta is not None:
        layer = model.net.model.layers[inject_layer]

        def _hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] >= n:
                hs = hs.clone()
                hs[:, n - 1, :] = hs[:, n - 1, :] + delta.to(hs.dtype)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return None
        handle = layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = model.net(**enc, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0, n - 1, :].float()   # final-layer residual at last token
        logits = out.logits[0, n - 1, :].float()                    # next-token logits
    finally:
        if handle is not None:
            handle.remove()
    return last_hidden, logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--eval-start", type=int, default=90)
    args = ap.parse_args()

    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)

    def hidden(model, prompt, layer):
        return model.hidden_at_final_token(prompt, layer).cuda()

    # fit ridge (train split)
    dvecs, rvecs = [], []
    for b in bugs[:N_TRAIN_FIT]:
        p = PROMPT_TMPL.format(buggy_code=b["buggy_code"])
        dvecs.append(hidden(donor, p, DONOR_LAYER).float().cpu())
        rvecs.append(hidden(receiver, p, RECEIVER_LAYER).float().cpu())
    W, d_mean, r_mean = fit_ridge_centered(dvecs, rvecs, RIDGE_LAMBDA)
    W, d_mean, r_mean = W.cuda(), d_mean.cuda(), r_mean.cuda()
    native_norm = float(torch.stack(rvecs).norm(dim=-1).mean())

    def map_norm_match(h_d, alpha=1.0):
        mapped = (W @ (h_d.float() - d_mean) + r_mean)
        mapped = mapped / (mapped.norm() + 1e-8) * native_norm
        return (alpha * mapped).to(torch.bfloat16)

    eval_bugs = bugs[args.eval_start:args.eval_start + args.n_eval]
    print(f"mech check on {len(eval_bugs)} bugs; native norm {native_norm:.2f}", flush=True)

    recs = []
    for bug in eval_bugs:
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        h_d = hidden(donor, prompt, DONOR_LAYER)
        h_r = hidden(receiver, prompt, RECEIVER_LAYER)
        d_transfer = map_norm_match(h_d)
        g = torch.Generator(device="cuda").manual_seed(42)
        rv = torch.randn(h_r.shape, generator=g, device="cuda", dtype=torch.float32)
        d_random = (rv / rv.norm() * native_norm).to(torch.bfloat16)
        d_self = h_r.to(torch.bfloat16)

        base_h, base_logits = last_token_state_and_logits(receiver, prompt)
        base_probs = F.log_softmax(base_logits, dim=-1)
        base_argmax = int(base_logits.argmax())

        for arm, delta in (("transfer", d_transfer), ("random_norm", d_random), ("self_inject", d_self)):
            ih, ilog = last_token_state_and_logits(receiver, prompt, RECEIVER_LAYER, delta)
            cos = float(F.cosine_similarity(base_h, ih, dim=0))
            norm_ratio = float(ih.norm() / (base_h.norm() + 1e-8))
            iprobs = F.log_softmax(ilog, dim=-1)
            kl = float(F.kl_div(base_probs, iprobs, log_target=True, reduction="sum").exp())  # not KL; use proper below
            # proper KL(base || inject)
            kl_div = float(F.kl_div(iprobs, base_probs, log_target=True, reduction="sum"))
            argmax_changed = int(ilog.argmax()) != base_argmax
            recs.append({"bug_id": bug_id, "arm": arm, "cos": round(cos, 4),
                         "norm_ratio": round(norm_ratio, 4), "kl_base_to_inject": round(kl_div, 4),
                         "argmax_changed": bool(argmax_changed)})
        print(f"{bug_id:<24} done", flush=True)

    Path(args.out).write_text(json.dumps(recs, indent=2))
    # aggregate
    print("\n==== MECH CHECK (mean over bugs) ====")
    for arm in ("transfer", "random_norm", "self_inject"):
        sub = [r for r in recs if r["arm"] == arm]
        mc = sum(r["cos"] for r in sub) / len(sub)
        mn = sum(r["norm_ratio"] for r in sub) / len(sub)
        mk = sum(r["kl_base_to_inject"] for r in sub) / len(sub)
        ma = sum(r["argmax_changed"] for r in sub) / len(sub)
        print(f"{arm:<12} cos={mc:.4f}  norm_ratio={mn:.4f}  KL={mk:.4f}  argmax_changed={ma:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
