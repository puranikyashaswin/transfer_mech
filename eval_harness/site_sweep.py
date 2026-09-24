#!/usr/bin/env python3
"""
site_sweep.py — Amendment 003 positive arm: the 2x2 injection-site sweep that
turns "single-vector transfer fails" into "we localized the minimal bandwidth at
which cross-model residual-stream transfer emerges."

Cells: {single_layer, all_layers} x {final_token, all_tokens}.
  - single_layer/final_token  = the existing null (reference cell).
  - all_tokens                = inject at every prompt-token position (CALM-style
                                bandwidth at one layer).
  - all_layers                = inject the mapped state at every receiver layer
                                (C2C-style depth at one token).
  - all_layers/all_tokens     = full-bandwidth upper bound.

Every cell has a MATCHED random-perturbation control with the SAME per-site and
TOTAL injected norm (defeats "you just injected more perturbation"). Same norm-
matched centered ridge per site, same eval, same seeds. Resumable per
(bug, cell, arm). Runs ON the GPU instance. NOT for the MacBook.
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
    DONOR_ID, PROMPT_TMPL, BASE_SEED, Model, load_bugs, apply_chat_template,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, extract_java_code,
)
from gate1_sweep import fit_ridge_centered  # noqa: E402

RECEIVER_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DONOR_LAYER = 14
RECEIVER_LAYER = 17
RIDGE_LAMBDA = 1.0
N_TRAIN_FIT = 40

CELLS = ["s1_t1", "s1_tall", "sall_t1", "sall_tall"]


def per_bug_seed(bug_id: str, cell: str) -> int:
    h = hashlib.sha256(f"site|{bug_id}|{cell}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


class SiteInjector:
    """Registers forward hooks on one or all receiver layers. For each hooked
    layer, adds a delta to the prompt-token positions. delta_fn(input_len) returns
    the [input_len, hidden] delta (built lazily per forward so it matches the
    generating model's prompt length and dtype/device).
    """

    def __init__(self, receiver: Model, layers: list[int], delta_fns: dict, prompt_len: int):
        self.receiver = receiver
        self.layers = layers
        self.delta_fns = delta_fns  # layer -> fn() -> [prompt_len, hidden]
        self.prompt_len = prompt_len  # receiver's own prompt length
        self.handles = []

    def __enter__(self):
        for L in self.layers:
            layer_mod = self.receiver.net.model.layers[L]
            fn = self.delta_fns[L]

            def _make_hook(_fn):
                def _hook(module, args, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    # Only act on the full prompt pass (seq_len == prompt_len); skip
                    # cached decode steps (seq_len==1) so deltas aren't re-added per token.
                    if hs.shape[1] != self.prompt_len:
                        return None
                    delta = _fn(self.prompt_len).to(hs.dtype)
                    hs = hs.clone()
                    hs[:, :self.prompt_len, :] = hs[:, :self.prompt_len, :] + delta.unsqueeze(0)
                    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
                return _hook
            self.handles.append(layer_mod.register_forward_hook(_make_hook(fn)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def hidden_all_layers(model: Model, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """Capture post-block residual at the final prompt token for each layer."""
    enc, input_len = model._encode(prompt)
    captured = {}
    handles = []

    def _mk(L):
        def _hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[L] = hs[:, input_len - 1, :].detach().float().cpu()
        return _hook
    for L in layers:
        handles.append(model.net.model.layers[L].register_forward_hook(_mk(L)))
    try:
        model.net(**enc)
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def hidden_all_tokens(model: Model, prompt: str, layer: int) -> torch.Tensor:
    """Capture post-block residual at ALL prompt-token positions for one layer.
    Returns [input_len, hidden]."""
    enc, input_len = model._encode(prompt)
    captured = {}

    def _hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["h"] = hs[0, :input_len, :].detach().float().cpu()
    h = model.net.model.layers[layer].register_forward_hook(_hook)
    try:
        model.net(**enc)
    finally:
        h.remove()
    return captured["h"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--eval-start", type=int, default=90)
    ap.add_argument("--eval-end", type=int, default=163)
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--junit-cp", default=None)
    args = ap.parse_args()
    cells = [c for c in args.cells.split(",") if c in CELLS]

    root = Path(args.dataset)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))
    data = load_dataset(root)
    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    buggy_snippet = {b["bug_id"].replace("humaneval_java_", ""): b["buggy_code"] for b in bugs}

    print("loading donor (1.5B) + receiver (3B)...", flush=True)
    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)
    n_layers = len(receiver.net.model.layers)
    print(f"receiver has {n_layers} layers", flush=True)

    # ---- fit one centered ridge per (donor L14 -> each receiver layer) on train ----
    # For all-layers cells we map to every receiver layer. For single-layer cells we
    # use only RECEIVER_LAYER. Train on N_TRAIN_FIT bugs.
    train_bugs = bugs[:N_TRAIN_FIT]
    target_layers = sorted(set([RECEIVER_LAYER] + (list(range(n_layers)) if any("sall" in c for c in cells) else [RECEIVER_LAYER])))
    print(f"fitting ridges for {len(target_layers)} receiver layers on {len(train_bugs)} train bugs...", flush=True)

    # collect donor L14 + receiver target-layer final-token states on train
    donor_train = []
    recv_train = {L: [] for L in target_layers}
    for b in train_bugs:
        p = PROMPT_TMPL.format(buggy_code=b["buggy_code"])
        donor_train.append(donor.hidden_at_final_token(p, DONOR_LAYER).float().cpu().squeeze(0))
        hl = hidden_all_layers(receiver, p, target_layers)
        for L in target_layers:
            recv_train[L].append(hl[L].float().cpu().squeeze(0))
    D = torch.stack(donor_train).float()
    d_mean = D.mean(0)
    maps = {}
    native_norms = {}
    for L in target_layers:
        Rr = torch.stack(recv_train[L]).float()
        W, dm, rm = fit_ridge_centered(donor_train, recv_train[L], RIDGE_LAMBDA)
        maps[L] = (W.cuda(), dm.cuda(), rm.cuda())
        native_norms[L] = float(Rr.norm(dim=-1).mean())
    print("ridges fit.", flush=True)

    def map_final_token(h_d_final: torch.Tensor, L: int) -> torch.Tensor:
        W, dm, rm = maps[L]
        m = (W @ (h_d_final.float().cuda() - dm) + rm)
        m = m / (m.norm() + 1e-8) * native_norms[L]
        return m.to(torch.bfloat16)

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

    def gen_with_sites(prompt: str, layers: list[int], delta_fns: dict, seed: int, prompt_len: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        enc, input_len = receiver._encode(prompt)
        with SiteInjector(receiver, layers, delta_fns, prompt_len):
            out = receiver.net.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                        top_p=TOP_P, do_sample=True, pad_token_id=receiver.tok.eos_token_id)
        gen = receiver.tok.decode(out[0][input_len:], skip_special_tokens=True)
        return extract_java_code(gen)

    eval_bugs = bugs[args.eval_start:args.eval_end]
    print(f"held-out eval: {len(eval_bugs)} bugs, cells={cells}", flush=True)

    # resume
    outp = Path(args.out)
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["bug_id"], r["cell"], r["arm"]))
        if done:
            print(f"[resume] {len(done)} records", flush=True)

    outf = open(outp, "a")
    t0 = time.time()
    for bug in eval_bugs:
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        enc, input_len = receiver._encode(prompt)  # receiver's own prompt length
        plen = input_len

        # donor states needed
        h_d_final = donor.hidden_at_final_token(prompt, DONOR_LAYER)
        need_all_tokens = any("tall" in c for c in cells)
        if need_all_tokens:
            h_d_alltok = hidden_all_tokens(donor, prompt, DONOR_LAYER)  # [donor_len, 1536]
            donor_len = h_d_alltok.shape[0]
        else:
            h_d_alltok, donor_len = None, 0

        # receiver states (final token per target layer; all tokens for norm at RECEIVER_LAYER)
        h_r_final = {L: hidden_all_layers(receiver, prompt, [L])[L] for L in target_layers}

        records = []

        if (bug_id, "baseline", "receiver_k1") not in done:
            torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
            out = receiver.net.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                        top_p=TOP_P, do_sample=True, pad_token_id=receiver.tok.eos_token_id)
            records.append((bug_id, "baseline", "receiver_k1", eval_code(bug_id, extract_java_code(receiver.tok.decode(out[0][input_len:], skip_special_tokens=True)))))

        hidden = receiver.net.config.hidden_size

        def build_delta(cell: str, content: str):
            """content in {'transfer','random'}. Returns (layers, delta_fns: layer->fn(input_len))."""
            sall = "sall" in cell
            tall = "tall" in cell
            layers = list(range(n_layers)) if sall else [RECEIVER_LAYER]
            if content == "random":
                gens = {L: torch.Generator(device="cuda").manual_seed(per_bug_seed(bug_id, cell) + L)
                        for L in layers}
                def make_fn(L):
                    g = gens[L]
                    nn = native_norms[L]
                    def delta_fn(input_len: int) -> torch.Tensor:
                        if tall:
                            rv = torch.randn((input_len, hidden), generator=g, device="cuda", dtype=torch.float32)
                            return (rv / (rv.norm(dim=-1, keepdim=True) + 1e-8) * nn).to(torch.bfloat16)
                        row = torch.zeros((input_len, hidden), device="cuda", dtype=torch.float32)
                        rv = torch.randn((hidden,), generator=g, device="cuda", dtype=torch.float32)
                        row[input_len - 1] = rv / (rv.norm() + 1e-8) * nn
                        return row.to(torch.bfloat16)
                    return delta_fn
                return layers, {L: make_fn(L) for L in layers}
            # transfer
            if tall:
                # Map the donor's all-token states. Donor/receiver prompt lengths differ
                # (different chat templates), so we align by min length and clamp.
                def make_fn(L):
                    W, dm, rm = maps[L]
                    nn = native_norms[L]
                    def delta_fn(pl: int) -> torch.Tensor:
                        D_all = h_d_alltok.float().cuda()  # [donor_len, 1536]
                        m = (D_all - dm) @ W.t() + rm  # [donor_len, hidden]
                        k = min(pl, donor_len)
                        out = m[:k]
                        if k < pl:  # pad by repeating the last mapped token
                            pad = out[-1:].expand(pl - k, -1)
                            out = torch.cat([out, pad], dim=0)
                        return (out / (out.norm(dim=-1, keepdim=True) + 1e-8) * nn).to(torch.bfloat16)
                    return delta_fn
                return layers, {L: make_fn(L) for L in layers}
            vecs = {L: map_final_token(h_d_final, L) for L in layers}  # [hidden] bf16 cuda
            def make_fn(L):
                vec = vecs[L]
                def delta_fn(pl: int) -> torch.Tensor:
                    row = torch.zeros((pl, hidden), dtype=torch.bfloat16, device="cuda")
                    row[pl - 1] = vec
                    return row
                return delta_fn
            return layers, {L: make_fn(L) for L in layers}

        for cell in cells:
            for content in (["transfer", "random"] if cell != "baseline" else []):
                if (bug_id, cell, content) in done:
                    continue
                layers, dfns = build_delta(cell, content)
                code = gen_with_sites(prompt, layers, dfns, BASE_SEED, plen)
                records.append((bug_id, cell, content, eval_code(bug_id, code)))

        for b_, c_, a_, s_ in records:
            outf.write(json.dumps({"bug_id": b_, "cell": c_, "arm": a_, "status": s_, "passed": s_ == "pass"}) + "\n")
            outf.flush()
        if records:
            print(f"{bug_id:<24} " + "  ".join(f"{c}/{a}={s}" for _, c, a, s in records), flush=True)

    outf.close()

    # summary
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    table = {}
    for key in sorted(set((r["cell"], r["arm"]) for r in rows)):
        sub = [r for r in rows if (r["cell"], r["arm"]) == key]
        npass = sum(r["passed"] for r in sub)
        table[f"{key[0]}/{key[1]}"] = {"n": len(sub), "pass": npass, "acc": round(npass / max(1, len(sub)), 4)}
    summary = {"n_eval": len(eval_bugs), "cells": cells, "results": table,
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print("\n==== SITE SWEEP SUMMARY ====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
