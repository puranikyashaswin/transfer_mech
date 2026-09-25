#!/usr/bin/env python3
"""
fit_receiver_subspace.py — fit the receiver's layer-12 "active subspace" PCA.

ANTI-LEAKAGE (Amendment 007 §3): fits ONLY on receiver layer-12 activations from
the TRAIN split (bugs 0-79). NEVER reads dev or test bugs. The output basis is
frozen and only APPLIED to dev/test by pilot_strength_subspace.py.

Output: a .pt file {'basis': [d, k] orthonormal tensor, 'var_explained': [...],
'k': k, 'layer': RECEIVER_LAYER, 'fit_split': 'train', 'n_fit': n}.

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
from gate1_pilot import RECEIVER_ID, PROMPT_TMPL, Model, load_bugs  # noqa: E402
from train_adapter import RECEIVER_LAYER  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=80)
    ap.add_argument("--var", type=float, default=0.90,
                    help="cumulative variance threshold to choose k")
    args = ap.parse_args()

    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    train_bugs = bugs[args.train_start:args.train_end]
    print(f"fitting subspace on TRAIN split only: {len(train_bugs)} bugs "
          f"(idx {args.train_start}-{args.train_end-1}), receiver layer {RECEIVER_LAYER}", flush=True)

    receiver = Model(RECEIVER_ID)
    H = []
    for bug in train_bugs:
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        h = receiver.hidden_at_final_token(prompt, RECEIVER_LAYER).cuda().float()
        H.append(h)
    X = torch.stack(H)                        # [n, d]
    n, d = X.shape
    print(f"collected activations: n={n}, d={d}", flush=True)

    # PCA via SVD on centered activations
    Xc = X - X.mean(0, keepdim=True)
    # economy SVD: Xc = U S Vh ; principal directions = rows of Vh
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(1, n - 1)
    total = var.sum()
    cum = torch.cumsum(var / total, dim=0)
    k = int((cum < args.var).sum().item()) + 1
    k = min(k, Vh.shape[0])
    basis = Vh[:k].T.contiguous()             # [d, k], orthonormal columns
    var_explained = (var / total)[:k].tolist()
    print(f"chose k={k} directions for {args.var:.0%} variance "
          f"(cumulative={float(cum[k-1]):.3f})", flush=True)

    torch.save({"basis": basis.cpu(), "var_explained": var_explained, "k": k,
                "layer": RECEIVER_LAYER, "fit_split": "train",
                "n_fit": n, "var_threshold": args.var}, args.out)
    print(f"saved subspace basis -> {args.out}  (basis shape {list(basis.shape)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
