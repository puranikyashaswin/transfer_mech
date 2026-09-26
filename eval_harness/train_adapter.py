#!/usr/bin/env python3
"""
train_adapter.py — Train the cross-family transfer adapter (Amendment 002, B2).

The core fix vs the failed ridge map: the adapter is trained for RECEIVER-SIDE
CAUSAL USE — it maximizes the frozen receiver's log-likelihood of the correct
fixed code when the donor's layer-14 activation is injected at receiver layer 12
(final prompt token). Both base models are FROZEN; only the adapter learns.

Split (frozen, no leakage): train = bugs 0-79, dev = 80-119, test = 120-162.
The adapter is selected on dev (alpha + early stop) and frozen before test.

Runs ON the GPU instance. Resumable: saves adapter checkpoint + a train log.
NOT for the MacBook.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate1_pilot import (  # noqa: E402
    DONOR_ID, RECEIVER_ID, PROMPT_TMPL, TEMPERATURE, TOP_P, MAX_NEW_TOKENS,
    BASE_SEED, Model, load_bugs, apply_chat_template,
)

DONOR_LAYER = int(os.environ.get("DONOR_LAYER", "14"))
# Amendment 008: receiver injection site corrected 12 -> 17 (Qwen-3B midpoint).
# Overridable via env so the layer-12 confirmatory artifacts remain reproducible.
RECEIVER_LAYER = int(os.environ.get("RECEIVER_LAYER", "12"))
BOTTLENECK = 512


class Adapter(nn.Module):
    """delta = alpha * g * W2·SiLU(W1·(h_donor - d_mean))  -> R^{2048}."""

    def __init__(self, d_in=1536, d_out=2048, m=BOTTLENECK):
        super().__init__()
        self.W1 = nn.Linear(d_in, m)
        self.W2 = nn.Linear(m, d_out)
        self.g = nn.Parameter(torch.full((d_out,), 0.1))  # small init ~ near no-op
        self.act = nn.SiLU()
        self.register_buffer("d_mean", torch.zeros(d_in))

    def forward(self, h_donor):  # h_donor: [..., 1536]
        x = self.W2(self.act(self.W1(h_donor - self.d_mean)))
        return self.g * x  # [..., 2048]  (alpha applied at injection time)


def chat_ids(tok, prompt: str):
    text = apply_chat_template(tok, prompt)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    return enc


@torch.no_grad()
def donor_hidden(donor: Model, prompt: str) -> torch.Tensor:
    return donor.hidden_at_final_token(prompt, DONOR_LAYER).cuda()


def receiver_nll_with_injection(receiver: Model, adapter: Adapter, prompt: str,
                                target_code: str, alpha: float) -> torch.Tensor:
    """Teacher-forced NLL of `target_code` under the frozen receiver, with the
    donor activation injected at RECEIVER_LAYER / final prompt token. The
    injection is differentiable w.r.t. the adapter (grad flows through the hook).
    Returns mean NLL over target tokens (scalar, requires_grad)."""
    tok = receiver.tok
    # Build full sequence: prompt (chat-templated) + target code, teacher forced.
    prompt_text = apply_chat_template(tok, prompt)
    full_text = prompt_text + target_code
    enc = tok(full_text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    prompt_ids = tok(prompt_text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"]
    n_prompt = prompt_ids.shape[1]
    input_ids = enc["input_ids"]
    # labels: -100 for prompt tokens, real ids for target tokens
    labels = torch.full_like(input_ids, -100)
    labels[:, n_prompt:] = input_ids[:, n_prompt:]
    if (labels != -100).sum() == 0:
        return torch.tensor(0.0, device="cuda", requires_grad=True)

    h_d = donor_hidden_cache["h"]  # [1536] fp32 on cuda, detached donor
    delta_full = alpha * adapter(h_d)  # [2048], differentiable

    layer = receiver.net.model.layers[RECEIVER_LAYER]

    def _hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] >= n_prompt:
            hs = hs.clone()
            hs[:, n_prompt - 1, :] = hs[:, n_prompt - 1, :] + delta_full.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return None

    handle = layer.register_forward_hook(_hook)
    try:
        out = receiver.net(**enc, labels=labels)
        loss = out.loss
    finally:
        handle.remove()
    return loss


# module-level cache for the donor hidden state of the current training example
donor_hidden_cache: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    train_bugs = bugs[: args.n_train]
    print(f"training adapter on {len(train_bugs)} bugs (train split 0-{args.n_train-1})", flush=True)

    print("loading donor + receiver (frozen)...", flush=True)
    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)
    for p in receiver.net.parameters():
        p.requires_grad_(False)
    for p in donor.net.parameters():
        p.requires_grad_(False)

    adapter = Adapter().cuda()
    # center: donor train mean at layer 14
    print("computing donor train mean (centering)...", flush=True)
    dvecs = [donor_hidden(donor, PROMPT_TMPL.format(buggy_code=b["buggy_code"])) for b in train_bugs]
    adapter.d_mean.copy_(torch.stack(dvecs).mean(0))

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.wd)

    # Pre-extract (prompt, correct_code) pairs. correct code = the fixed method body.
    examples = []
    for b in train_bugs:
        prompt = PROMPT_TMPL.format(buggy_code=b["buggy_code"])
        examples.append((prompt, b["fixed_code"]))

    logf = open(log_path, "a")
    t0 = time.time()
    step = 0
    n = len(examples)
    while step < args.steps:
        # shuffle each epoch
        perm = torch.randperm(n).tolist()
        for i in range(0, n, args.batch):
            if step >= args.steps:
                break
            batch = [examples[j] for j in perm[i:i + args.batch]]
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for prompt, code in batch:
                donor_hidden_cache["h"] = donor_hidden(donor, prompt)
                loss = receiver_nll_with_injection(receiver, adapter, prompt, code, args.alpha)
                loss = loss / len(batch)
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                gnorm = float(adapter.g.abs().mean())
                rec = {"step": step, "nll": round(total, 4),
                       "g_mean": round(gnorm, 4),
                       "elapsed_min": round((time.time() - t0) / 60, 1)}
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                print(f"step {step:4d}/{args.steps}  nll={total:.4f}  g_mean={gnorm:.3f}", flush=True)
            # periodic checkpoint so an interrupt never loses the adapter
            if step % 100 == 0:
                torch.save({"state_dict": adapter.state_dict(), "alpha": args.alpha,
                            "donor_layer": DONOR_LAYER, "receiver_layer": RECEIVER_LAYER,
                            "bottleneck": BOTTLENECK, "n_train": args.n_train,
                            "steps": step}, out_dir / "adapter.pt")

    logf.close()
    ckpt = out_dir / "adapter.pt"
    torch.save({"state_dict": adapter.state_dict(), "alpha": args.alpha,
                "donor_layer": DONOR_LAYER, "receiver_layer": RECEIVER_LAYER,
                "bottleneck": BOTTLENECK, "n_train": args.n_train,
                "steps": args.steps}, ckpt)
    print(f"\nsaved adapter -> {ckpt}", flush=True)
    print(f"final g_mean={float(adapter.g.abs().mean()):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
