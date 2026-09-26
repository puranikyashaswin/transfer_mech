#!/usr/bin/env python3
"""
fit_linear_map.py — fit a BASELINE linear (ridge) map donor->receiver, train split only.

This is the critic-roadmap P1.3 "missing baseline": a naive linear probe-style map
(NO learned nonlinearity, no bottleneck) that answers whether the confirmatory
negative is specific to the trained adapter's architecture or is a general property
of cross-model single-vector injection at this site.

ANTI-LEAKAGE: fits ONLY on train-split bugs (0-79) donor-layer-14 / receiver-layer-12
final-prompt-token activations. NEVER reads dev/test. Output is frozen and only
APPLIED by eval_baseline_map.py to dev bugs (the pilot's split pattern).

Map: y = W @ (x - d_mean) + r_mean   (centered ridge, lam=1.0)
Eval-time alpha rescales the mapped delta: alpha=1.0 is the identity-strength map.

Output .pt: {'W': [d_r, d_d], 'd_mean': [d_d], 'r_mean': [d_r],
             'fit_split': 'train', 'n_fit': n, 'lam': lam}

Runs ON the GPU instance. NOT for the MacBook.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate1_pilot  # noqa: E402
from gate1_pilot import DONOR_ID, RECEIVER_ID, PROMPT_TMPL, Model, load_bugs  # noqa: E402
from train_adapter import DONOR_LAYER, RECEIVER_LAYER  # noqa: E402


def fit_ridge_centered(donor_vecs, recv_vecs, lam):
    """Centered ridge: y = W @ (x - d_mean) + r_mean. Same math as gate1_sweep."""
    X = torch.stack(donor_vecs).double()
    Y = torch.stack(recv_vecs).double()
    d_mean = X.mean(0, keepdim=True)
    r_mean = Y.mean(0, keepdim=True)
    Xc = X - d_mean
    Yc = Y - r_mean
    d = X.shape[1]
    A = Xc.T @ Xc + lam * torch.eye(d, dtype=torch.double)
    B = Xc.T @ Yc
    W = torch.linalg.solve(A, B).T.float()  # [d_r, d_d]
    return W, d_mean.squeeze(0).float(), r_mean.squeeze(0).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=80)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    train_bugs = bugs[args.train_start:args.train_end]
    print(f"fitting baseline ridge map on TRAIN split only: {len(train_bugs)} bugs "
          f"(idx {args.train_start}-{args.train_end-1}), donor L{DONOR_LAYER} -> receiver L{RECEIVER_LAYER}",
          flush=True)

    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)
    D, R = [], []
    for bug in train_bugs:
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        D.append(donor.hidden_at_final_token(prompt, DONOR_LAYER).float().cpu())
        R.append(receiver.hidden_at_final_token(prompt, RECEIVER_LAYER).float().cpu())
    print(f"collected paired activations: n={len(D)}, d_donor={D[0].shape[0]}, d_recv={R[0].shape[0]}",
          flush=True)

    # Adapter-parallel delta semantics: the target is the receiver's DEVIATION from
    # its own mean (h_r - r_mean), so that W @ (h_d - d_mean) IS the additive delta,
    # exactly mirroring the adapter's forward (which emits a delta, not a target state).
    r_mean_pre = torch.stack(R).double().mean(0, keepdim=True).float()
    Rdev = [r - r_mean_pre.squeeze(0) for r in R]
    W, d_mean, _ = fit_ridge_centered(D, Rdev, args.lam)
    r_mean = r_mean_pre.squeeze(0)  # stored for provenance (not used in delta form)

    # offline fit quality (reported for the paper: probe R^2 on train, delta space)
    X = torch.stack(D).float()
    Y = torch.stack(Rdev).float()
    Yhat = (X - d_mean) @ W.T
    ss_res = ((Y - Yhat) ** 2).sum().item()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot
    cos = torch.nn.functional.cosine_similarity(Yhat, Y, dim=1).mean().item()
    print(f"offline train fit: R^2={r2:.4f}  mean_cos={cos:.4f}", flush=True)

    torch.save({"W": W, "d_mean": d_mean, "r_mean": r_mean,
                "fit_split": "train", "n_fit": len(D), "lam": args.lam,
                "donor_layer": DONOR_LAYER, "receiver_layer": RECEIVER_LAYER,
                "train_r2": r2, "train_cos": cos}, args.out)
    print(f"saved baseline map -> {args.out}  (W shape {list(W.shape)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
