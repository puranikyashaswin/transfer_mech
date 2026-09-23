#!/usr/bin/env python3
"""
gate1_pilot.py — Gate 1 (Diversity) pilot for "What Transfers?".

Question (PREREG Gate 1): does a cross-family donor add anything beyond EXTRA
RECEIVER SAMPLING at matched compute? This pilot measures, per bug:
  - receiver single-shot pass (k=1) and receiver sampling oracle (pass@k)
  - donor single-shot pass
  - discordance: bugs where donor passes but receiver k=1 fails (transfer candidates)
  - a TRANSFER SMOKE TEST: inject the donor's residual-stream activation into the
    receiver (frozen config, Amendment A2) on the discordant bugs and see if it
    ever flips a fail to a pass.

Ground truth: JUnit execution ONLY (string similarity banned). The ridge map W
is fit on a TRAIN slice of bugs only, never on the pilot eval bugs (no leakage).

Runs ON the GPU instance. Resumable: appends per-bug results to a checkpoint
JSONL and skips bugs already done. NOT for the MacBook.

Models (frozen, Amendment A1):
  donor    = Qwen/Qwen2.5-Coder-1.5B-Instruct   (hidden 1536, 28 layers)
  receiver = deepseek-ai/deepseek-coder-1.3b-instruct (hidden 2048, 24 layers)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# local harness
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from junit_runner import (  # noqa: E402
    evaluate_candidate,
    extract_class_name,
    find_method_span,
    load_dataset,
    method_names,
    wrap_candidate_method,
)

# ---- Frozen config (Amendment A1/A2) ---------------------------------------
DONOR_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
RECEIVER_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
DONOR_LAYER = 14          # of 28 (midpoint)
RECEIVER_LAYER = 12       # of 24 (midpoint)
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_NEW_TOKENS = 512
BASE_SEED = 42
K_SAMPLES = 5             # receiver sampling budget (killer baseline)
RIDGE_LAMBDA = 1.0
N_TRAIN_FIT = 40          # number of train bugs used to fit W

PROMPT_TMPL = """Fix the bug in this Java code. Return ONLY the fixed code.

```java
{buggy_code}
```"""

# ---- Code extraction (same as verified run) ---------------------------------
def extract_java_code(text: str) -> str:
    if not text:
        return ""
    if "```java" in text:
        s = text.find("```java") + 7
        e = text.find("```", s)
        if e > s:
            return text[s:e].strip()
    if "```" in text:
        s = text.find("```") + 3
        e = text.find("```", s)
        if e > s:
            return text[s:e].strip()
    return text.strip()


def apply_chat_template(tokenizer, prompt: str) -> str:
    try:
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


# ---- Model wrapper -----------------------------------------------------------
class Model:
    def __init__(self, model_id: str):
        self.id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.net = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True,
        ).eval()
        self.n_layers = self.net.config.num_hidden_layers
        self.hidden = self.net.config.hidden_size

    def _encode(self, prompt: str):
        text = apply_chat_template(self.tok, prompt)
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        return enc, enc["input_ids"].shape[1]

    @torch.no_grad()
    def generate(self, prompt: str, seed: int, hook_layer: int | None = None,
                 hook_fn=None) -> tuple[str, int]:
        """Generate; if hook_layer given, register a forward hook calling
        hook_fn(hidden_state_at_final_prompt_token) -> additive delta."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        enc, input_len = self._encode(prompt)
        handle = None
        if hook_layer is not None and hook_fn is not None:
            layer = self.net.model.layers[hook_layer]  # decoder block
            def _hook(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                # Only inject on the PROMPT pass (full sequence present). During
                # cached decoding seq_len==1 and input_len-1 is out of bounds.
                if hs.shape[1] >= input_len:
                    delta = hook_fn(hs, input_len)
                    if delta is not None:
                        hs = hs.clone()
                        hs[:, input_len - 1, :] = hs[:, input_len - 1, :] + delta
                        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
                return None
            handle = layer.register_forward_hook(_hook)
        try:
            out = self.net.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                top_p=TOP_P, do_sample=True, pad_token_id=self.tok.eos_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        gen = self.tok.decode(out[0][input_len:], skip_special_tokens=True)
        return extract_java_code(gen), out.shape[1] - input_len

    @torch.no_grad()
    def hidden_at_final_token(self, prompt: str, layer: int) -> torch.Tensor:
        """Residual-stream activation (post-block) at `layer`, final prompt token."""
        enc, input_len = self._encode(prompt)
        captured = {}
        layer_mod = self.net.model.layers[layer]
        def _hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["h"] = hs[:, input_len - 1, :].detach().float().cpu()
        h = layer_mod.register_forward_hook(_hook)
        try:
            self.net(**enc)
        finally:
            h.remove()
        return captured["h"].squeeze(0)  # [hidden]


# ---- Ridge map W: donor_hidden -> receiver_hidden ---------------------------
def fit_ridge(donor_vecs: list[torch.Tensor], recv_vecs: list[torch.Tensor],
              lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.stack(donor_vecs).double()   # [N, 1536]
    Y = torch.stack(recv_vecs).double()    # [N, 2048]
    N, d = X.shape
    Xb = torch.cat([X, torch.ones(N, 1, dtype=torch.double)], dim=1)  # bias
    A = Xb.T @ Xb + lam * torch.eye(d + 1, dtype=torch.double)
    B = Xb.T @ Y
    Wb = torch.linalg.solve(A, B)          # [d+1, 2048]
    W = Wb[:-1].T.float()                  # [2048, 1536]
    b = Wb[-1].float()                     # [2048]
    return W, b


def load_bugs(path: Path) -> list[dict]:
    bugs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                bugs.append(json.loads(line))
    return bugs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True, help="humaneval_java.jsonl")
    ap.add_argument("--dataset", required=True, help="temp_hej root (for JUnit)")
    ap.add_argument("--out", required=True, help="checkpoint JSONL (resumable)")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--n-pilot", type=int, default=50)
    ap.add_argument("--k", type=int, default=K_SAMPLES)
    ap.add_argument("--junit-cp", default=None)
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    args = ap.parse_args()

    root = Path(args.dataset)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))
    data = load_dataset(root)  # bug_id -> {buggy, correct, test}
    bugs = load_bugs(Path(args.bugs))
    # bug_id (short) -> buggy method snippet (from the jsonl the model was shown)
    buggy_snippet = {b["bug_id"].replace("humaneval_java_", ""): b["buggy_code"]
                     for b in bugs}

    def _buggy_method_snippet(bug_id: str) -> str:
        return buggy_snippet.get(bug_id, "")

    # deterministic order; split train-fit vs pilot-eval (disjoint, no leakage)
    bugs = sorted(bugs, key=lambda b: b["bug_id"])
    train_bugs = bugs[:N_TRAIN_FIT]
    pilot_bugs = bugs[N_TRAIN_FIT:N_TRAIN_FIT + args.n_pilot]
    print(f"train-fit bugs: {len(train_bugs)}  pilot-eval bugs: {len(pilot_bugs)}", flush=True)

    # resume: load done bug_ids
    done = {}
    outp = Path(args.out)
    if outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["bug_id"]] = r
    if done:
        print(f"[resume] {len(done)} pilot bugs already done", flush=True)

    print("Loading donor + receiver...", flush=True)
    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)
    print(f"donor {DONOR_ID}: {donor.n_layers}L h{donor.hidden}", flush=True)
    print(f"receiver {RECEIVER_ID}: {receiver.n_layers}L h{receiver.hidden}", flush=True)

    # ---- Fit W on TRAIN slice (donor L14 -> receiver L12, final prompt token) --
    print(f"Fitting ridge map W on {len(train_bugs)} train bugs...", flush=True)
    dvecs, rvecs = [], []
    for b in train_bugs:
        prompt = PROMPT_TMPL.format(buggy_code=b["buggy_code"])
        dvecs.append(donor.hidden_at_final_token(prompt, DONOR_LAYER))
        rvecs.append(receiver.hidden_at_final_token(prompt, RECEIVER_LAYER))
    W, bias = fit_ridge(dvecs, rvecs, RIDGE_LAMBDA)
    W = W.cuda()
    bias = bias.cuda()
    print(f"W fit: {tuple(W.shape)}, bias {tuple(bias.shape)}", flush=True)

    def eval_code(bug_id: str, code: str) -> str:
        """Evaluate a model generation. If it's a full class, compile directly.
        If it's a method body (no public class), wrap it into the buggy class
        template (replacing the buggy method) so the JUnit test can call it."""
        test_src = data[bug_id]["test"]
        buggy_class = data[bug_id]["buggy"]
        if extract_class_name(code) is None and buggy_class is not None:
            # model produced a method, not a class -> wrap it
            buggy_method = _buggy_method_snippet(bug_id)
            wrapped = wrap_candidate_method(buggy_class, buggy_method, code)
            if wrapped is None:
                return "error"
            code = wrapped
        with tempfile.TemporaryDirectory() as wd:
            r = evaluate_candidate(bug_id, code, test_src, args.javac, args.java,
                                   junit_cp, wd)
        return r["status"]

    outf = open(outp, "a")
    t_start = time.time()
    for i, bug in enumerate(pilot_bugs):
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        if bug_id in done:
            continue
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        rec: dict = {"bug_id": bug_id}

        # 1) receiver k=1 (seed = BASE_SEED)
        r1_code, _ = receiver.generate(prompt, seed=BASE_SEED)
        r1_status = eval_code(bug_id, r1_code)
        rec["receiver_k1"] = {"status": r1_status}

        # 2) receiver sampling oracle pass@k (seeds BASE_SEED+1..+k-1 plus k=1)
        statuses = [r1_status]
        for j in range(1, args.k):
            c, _ = receiver.generate(prompt, seed=BASE_SEED + j)
            statuses.append(eval_code(bug_id, c))
        rec["receiver_samples"] = statuses
        rec["receiver_pass_at_k"] = any(s == "pass" for s in statuses)

        # 3) donor single-shot
        d_code, _ = donor.generate(prompt, seed=BASE_SEED)
        d_status = eval_code(bug_id, d_code)
        rec["donor_k1"] = {"status": d_status}

        # 4) discordance: donor passes, receiver k=1 fails
        rec["discordant"] = (d_status == "pass" and r1_status != "pass")

        # 5) transfer smoke test on discordant bugs: inject donor L14 -> receiver L12
        if rec["discordant"]:
            h_d = donor.hidden_at_final_token(prompt, DONOR_LAYER).cuda()  # [1536]
            delta = (W @ h_d + bias).to(torch.bfloat16)                    # [2048]
            def hook_fn(hs, input_len, _d=delta):
                return _d
            t_code, _ = receiver.generate(prompt, seed=BASE_SEED,
                                          hook_layer=RECEIVER_LAYER, hook_fn=hook_fn)
            t_status = eval_code(bug_id, t_code)
            rec["transfer"] = {"status": t_status, "flipped": t_status == "pass"}
        outf.write(json.dumps(rec) + "\n")
        outf.flush()
        print(f"[{i+1:3d}/{len(pilot_bugs)}] {bug_id:<32} "
              f"r1={r1_status:<12} pass@{args.k}={rec['receiver_pass_at_k']} "
              f"donor={d_status:<12} disc={rec['discordant']}"
              + (f" transfer={rec['transfer']['status']}" if "transfer" in rec else ""),
              flush=True)

    outf.close()

    # ---- Summary (recomputed from checkpoint) --------------------------------
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    n = len(rows)
    r1_pass = sum(1 for r in rows if r["receiver_k1"]["status"] == "pass")
    passk = sum(1 for r in rows if r["receiver_pass_at_k"])
    donor_pass = sum(1 for r in rows if r["donor_k1"]["status"] == "pass")
    disc = sum(1 for r in rows if r["discordant"])
    transfers = [r for r in rows if "transfer" in r]
    t_flip = sum(1 for r in transfers if r["transfer"]["flipped"])
    summary = {
        "n_pilot": n, "k": args.k,
        "receiver_k1_pass": r1_pass, "receiver_k1_acc": r1_pass / max(1, n),
        "receiver_pass_at_k": passk, "receiver_pass_at_k_acc": passk / max(1, n),
        "donor_k1_pass": donor_pass, "donor_k1_acc": donor_pass / max(1, n),
        "sampling_gain_pp": (passk - r1_pass) / max(1, n) * 100,
        "discordant_count": disc, "discordance_rate": disc / max(1, n),
        "transfer_attempts": len(transfers), "transfer_flips": t_flip,
        "transfer_flip_rate_on_discordant": t_flip / max(1, len(transfers)),
        "elapsed_min": round((time.time() - t_start) / 60, 1),
        "config": {"donor": DONOR_ID, "receiver": RECEIVER_ID,
                   "donor_layer": DONOR_LAYER, "receiver_layer": RECEIVER_LAYER,
                   "temp": TEMPERATURE, "top_p": TOP_P, "seed": BASE_SEED,
                   "ridge_lambda": RIDGE_LAMBDA, "n_train_fit": N_TRAIN_FIT},
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print("\n==== GATE 1 PILOT SUMMARY ====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
