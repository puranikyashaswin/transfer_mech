#!/usr/bin/env python3
"""
eval_adapter.py — Evaluate the frozen trained adapter (Amendment 002) on the
HELD-OUT test split (bugs 120-end), with the FULL 8-arm control set required by
pre-registration AMENDMENT 006 §4. Execution-verified (JUnit).

Arms per test bug (Amendment 006 §4, verbatim):
  1. receiver_k1     : receiver single-shot, no injection (baseline)
  2. adapter         : receiver + frozen trained adapter injection (intervention)
  3. random_norm     : receiver + same-norm RANDOM vector, R=5 per-bug-seeded
                       draws; seed = BASE_SEED XOR hash(bug_id) XOR draw_index,
                       draw_index in 0..4 (magnitude control). The arm's pass/fail
                       = the FRACTION of the 5 draws that pass; per-draw statuses
                       and the aggregate are recorded (5 rows in eval_out.jsonl,
                       one per draw).
  4. shuffled        : receiver + adapter output for a DIFFERENT bug (specificity);
                       "different bug" is deterministic = next bug in sorted order.
  5. self_inject     : receiver + its OWN hidden state re-injected (hook sanity)
  6. unrelated_task  : (C5) receiver + donor activation from an UNRELATED task
                       (task-relevance control); the unrelated donor input is a
                       FIXED hardcoded prompt (UNRELATED_PROMPT below), passed
                       through the adapter and injected.
  7. text_hint       : receiver with a TEXTUAL hint appended to the prompt
                       (positive control / ceiling reference); per-bug hint from
                       the bugs manifest when present (keys "hint" / "text_hint"),
                       else the fixed generic HINT_GENERIC string. NO injection.
  8. donor_k1        : DONOR single-shot, no injection (capability reference).

Stable hash used everywhere: hash(bug_id) =
    int.from_bytes(hashlib.sha256(bug_id.encode()).digest()[:8], "little")
(Python's builtin hash() is NOT stable across runs and is NOT used.)

The adapter + alpha are FROZEN (loaded from adapter.pt). One shot per bug per arm
(random_norm: 5 seeded draws). Generation budget = 1024 new tokens for EVERY arm
(Amendment 005); temp 0.2, top-p 0.95; final-prompt-token additive injection.

Runs ON the GPU instance. Resumable per (bug, arm); random_norm resumes per
(bug, draw). NOT for the MacBook.
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
    evaluate_candidate_default_pkg, extract_class_name, load_dataset,
)
import gate1_pilot  # noqa: E402
from gate1_pilot import (  # noqa: E402
    DONOR_ID, RECEIVER_ID, PROMPT_TMPL, BASE_SEED, Model, load_bugs,
)
from train_adapter import Adapter, DONOR_LAYER, RECEIVER_LAYER  # noqa: E402

# Generation budget = 1024 tokens for every arm (Amendment 005). Model.generate
# reads gate1_pilot's module-global; set it at runtime (gate1_pilot.py unmodified).
gate1_pilot.MAX_NEW_TOKENS = 1024
assert gate1_pilot.MAX_NEW_TOKENS == 1024

# ---- Amendment 006 §4 fixed constants ---------------------------------------
R_DRAWS = 5               # random_norm draws per bug
CODE_TRUNC = 4000         # max chars of generated code stored per row

# unrelated_task (C5): FIXED, hardcoded unrelated-task donor prompt. NOT a corpus
# bug; tokenized exactly like any other donor input.
UNRELATED_PROMPT = "The quick brown fox jumps over the lazy dog."

# text_hint: fixed generic hint used when the bugs manifest has no per-bug hint.
HINT_GENERIC = ("Hint: the buggy code has a single-implementation defect; "
                "produce the corrected method.")


_CLASS_DECL_RE = __import__("re").compile(r"\bpublic\s+(?:final\s+)?class\s+[A-Za-z_][A-Za-z0-9_]*")


def retarget_class_name(code: str, bug_id: str) -> str:
    """Force the generated class's public class name to `bug_id`.

    Generation completes the java_prompt scaffold, which declares `class Problem`.
    The corpus TEST invokes `<BUG>.method(...)`, so the candidate's public class
    MUST be named <BUG>. Replace the first `public class <X>` with `public class
    <BUG>` (drop `public` if absent). No-op if already named <BUG>."""
    if _CLASS_DECL_RE.search(code):
        return _CLASS_DECL_RE.sub(f"public class {bug_id}", code, count=1)
    return code


def stable_hash(bug_id: str) -> int:
    """Stable hash(bug_id) = first 8 bytes of sha256, little-endian (A006 §4).
    Python's builtin hash() is NOT stable across runs — never use it here."""
    return int.from_bytes(hashlib.sha256(bug_id.encode()).digest()[:8], "little")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--adapter", required=True, help="adapter.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--test-start", type=int, default=120)
    ap.add_argument("--test-end", type=int, default=335)
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--junit-cp", default=None)
    args = ap.parse_args()

    root = Path(args.dataset)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))
    data = load_dataset(root)
    bugs = sorted(load_bugs(Path(args.bugs)), key=lambda b: b["bug_id"])
    buggy_snippet = {b["bug_id"].replace("humaneval_java_", ""): b["buggy_code"] for b in bugs}
    test_bugs = bugs[args.test_start:args.test_end]
    print(f"held-out test bugs: {len(test_bugs)} (idx {args.test_start}-{args.test_end-1})", flush=True)

    ckpt = torch.load(args.adapter, map_location="cuda")
    alpha = ckpt["alpha"]
    adapter = Adapter().cuda()
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    print(f"loaded adapter (alpha={alpha}, donor L{DONOR_LAYER} -> receiver L{RECEIVER_LAYER})", flush=True)

    donor = Model(DONOR_ID)
    receiver = Model(RECEIVER_ID)

    # unrelated_task (C5): compute the FIXED unrelated donor activation ONCE.
    h_unrel = donor.hidden_at_final_token(UNRELATED_PROMPT, DONOR_LAYER).cuda()
    with torch.no_grad():
        delta_unrel = (alpha * adapter(h_unrel)).to(torch.bfloat16)
    print("unrelated_task (C5) donor activation computed from fixed prompt", flush=True)

    def eval_code(bug_id, code):
        """Evaluate a generated candidate against the corpus TEST (default package,
        class-qualified `<BUG>.method(...)` calls). The candidate must be a
        default-package class named exactly <BUG>. Generation completes the
        java_prompt's `class Problem` scaffold, so we retarget the class name to
        <BUG> before compiling."""
        test_src = data[bug_id]["test"]
        code = retarget_class_name(code, bug_id)
        with tempfile.TemporaryDirectory() as wd:
            r = evaluate_candidate_default_pkg(
                bug_id, code, test_src, args.javac, args.java, junit_cp, wd)
        return r["status"]

    def gen_with_delta(prompt, delta, seed):
        def hook_fn(hs, input_len, _d=delta):
            return _d
        code, _ = receiver.generate(prompt, seed=seed, hook_layer=RECEIVER_LAYER, hook_fn=hook_fn)
        return code

    def run_arm(bug_id, arm, fn, extra=None):
        """Time a generation+eval and append one eval_out.jsonl row."""
        t1 = time.time()
        code = fn()
        status = eval_code(bug_id, code)
        rec = {"bug_id": bug_id, "arm": arm, "status": status,
               "code": code[:CODE_TRUNC], "gen_s": round(time.time() - t1, 3)}
        if extra:
            rec.update(extra)
        outf.write(json.dumps(rec) + "\n")
        outf.flush()
        return status

    # resume: per (bug,arm) for most arms; per (bug,draw) for random_norm
    outp = Path(args.out)
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["arm"] == "random_norm":
                    done.add((r["bug_id"], "random_norm", r["draw"]))
                else:
                    done.add((r["bug_id"], r["arm"]))
        if done:
            print(f"[resume] {len(done)} rows done", flush=True)

    outf = open(outp, "a")
    t0 = time.time()
    for bug in test_bugs:
        bug_id = bug["bug_id"].replace("humaneval_java_", "")
        prompt = PROMPT_TMPL.format(buggy_code=bug["buggy_code"])
        h_d = donor.hidden_at_final_token(prompt, DONOR_LAYER).cuda()
        with torch.no_grad():
            delta_adapter = (alpha * adapter(h_d)).to(torch.bfloat16)
        # receiver's own hidden state (self-inject control)
        h_r = receiver.hidden_at_final_token(prompt, RECEIVER_LAYER).cuda()

        results = []
        # 1. receiver_k1 — baseline, no injection
        if (bug_id, "receiver_k1") not in done:
            s = run_arm(bug_id, "receiver_k1",
                        lambda: receiver.generate(prompt, seed=BASE_SEED)[0],
                        {"seed": BASE_SEED})
            results.append(("receiver_k1", s))
        # 2. adapter — frozen trained adapter injection
        if (bug_id, "adapter") not in done:
            s = run_arm(bug_id, "adapter",
                        lambda: gen_with_delta(prompt, delta_adapter, BASE_SEED),
                        {"seed": BASE_SEED})
            results.append(("adapter", s))
        # 3. random_norm — R=5 per-bug-seeded same-norm draws (magnitude control).
        #    Norm of each draw = the ADAPTER OUTPUT norm for this bug (same-norm
        #    control). Resumes per (bug, draw): completed draws are skipped.
        for draw_index in range(R_DRAWS):
            if (bug_id, "random_norm", draw_index) in done:
                continue
            seed = BASE_SEED ^ stable_hash(bug_id) ^ draw_index
            g = torch.Generator(device="cuda").manual_seed(seed)
            rv = torch.randn(h_r.shape, generator=g, device="cuda", dtype=torch.float32)
            rv = rv / rv.norm() * delta_adapter.float().norm()
            s = run_arm(bug_id, "random_norm",
                        lambda _rv=rv, _seed=seed: gen_with_delta(
                            prompt, _rv.to(torch.bfloat16), _seed),
                        {"draw": draw_index, "seed": seed})
            results.append((f"random_norm[{draw_index}]", s))
        # 4. shuffled — adapter output for a DIFFERENT bug; deterministic:
        #    the NEXT bug in sorted (bug_id) order within the test split.
        if (bug_id, "shuffled") not in done:
            other = test_bugs[(test_bugs.index(bug) + 1) % len(test_bugs)]
            h_other = donor.hidden_at_final_token(
                PROMPT_TMPL.format(buggy_code=other["buggy_code"]), DONOR_LAYER).cuda()
            with torch.no_grad():
                delta_shuf = (alpha * adapter(h_other)).to(torch.bfloat16)
            s = run_arm(bug_id, "shuffled",
                        lambda: gen_with_delta(prompt, delta_shuf, BASE_SEED),
                        {"seed": BASE_SEED})
            results.append(("shuffled", s))
        # 5. self_inject — receiver's OWN hidden state re-injected (hook sanity)
        if (bug_id, "self_inject") not in done:
            s = run_arm(bug_id, "self_inject",
                        lambda: gen_with_delta(prompt, h_r.to(torch.bfloat16), BASE_SEED),
                        {"seed": BASE_SEED})
            results.append(("self_inject", s))
        # 6. unrelated_task (C5) — FIXED unrelated-prompt donor activation
        #    through the adapter, injected (task-relevance control).
        if (bug_id, "unrelated_task") not in done:
            s = run_arm(bug_id, "unrelated_task",
                        lambda: gen_with_delta(prompt, delta_unrel, BASE_SEED),
                        {"seed": BASE_SEED})
            results.append(("unrelated_task", s))
        # 7. text_hint — textual hint appended to the prompt; NO injection
        #    (positive control / ceiling reference). Per-bug hint from the
        #    manifest when available, else the fixed generic hint.
        if (bug_id, "text_hint") not in done:
            hint = bug.get("hint") or bug.get("text_hint") or HINT_GENERIC
            s = run_arm(bug_id, "text_hint",
                        lambda: receiver.generate(prompt + "\n" + hint, seed=BASE_SEED)[0],
                        {"seed": BASE_SEED})
            results.append(("text_hint", s))
        # 8. donor_k1 — DONOR single-shot, no injection (capability reference)
        if (bug_id, "donor_k1") not in done:
            s = run_arm(bug_id, "donor_k1",
                        lambda: donor.generate(prompt, seed=BASE_SEED)[0],
                        {"seed": BASE_SEED})
            results.append(("donor_k1", s))

        if results:
            print(f"{bug_id:<26} " + "  ".join(f"{a}={s}" for a, s in results), flush=True)

    outf.close()
    # summary
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    arms_list = ["receiver_k1", "adapter", "random_norm", "shuffled",
                 "self_inject", "unrelated_task", "text_hint", "donor_k1"]
    table = {}
    for arm in arms_list:
        sub = [r for r in rows if r["arm"] == arm]
        npass = sum(r["status"] == "pass" for r in sub)
        table[arm] = {"n": len(sub), "pass": npass, "acc": npass / max(1, len(sub))}
    # random_norm arm-level pass/fail = FRACTION of the R=5 draws that pass
    rn_bugs = sorted({r["bug_id"] for r in rows if r["arm"] == "random_norm"})
    frac = {bid: sum(r["status"] == "pass" for r in rows
                     if r["arm"] == "random_norm" and r["bug_id"] == bid) / R_DRAWS
            for bid in rn_bugs}
    table["random_norm"]["per_bug_pass_frac"] = frac
    table["random_norm"]["bug_mean_acc"] = sum(frac.values()) / max(1, len(frac))
    summary = {"n_test": len(test_bugs), "alpha": alpha, "arms": table,
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print("\n==== ADAPTER EVAL SUMMARY (held-out test, 8 arms, Amendment 006 §4) ====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
