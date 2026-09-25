#!/usr/bin/env python3
"""
translate.py -- Stage 1 of the confirmatory_hej corpus build.

Translate Python canonical solutions (MultiPL-E humaneval-java + mbpp-java) to
Java using the cached Qwen2.5-Coder-3B-Instruct (deterministic greedy decode,
temperature 0, max_new_tokens 1024). Emits gen/<bug_id>.java candidates under
~/transfer_mech/data/confirmatory_hej/.

bug_id format: HE_<n>_<name> for humaneval, MBPP_<n>_<name> for mbpp.

Resumable: skips bug_ids that already have gen/<bug_id>.java and problems
permanently dropped in drops_translate.jsonl. Writes NOTHING to /opt/dlami/nvme.

Run with:
  PYTHONPATH=/data/home/ubuntu/.local/lib/python3.10/site-packages \
      python3 translate.py
"""

import ast
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME = Path(os.path.expanduser("~"))
OUT_ROOT = HOME / "transfer_mech" / "data" / "confirmatory_hej"
GEN_DIR = OUT_ROOT / "gen"
LOGS_DIR = OUT_ROOT / "logs"
META_PATH = LOGS_DIR / "problems_meta.jsonl"
DROPS_PATH = LOGS_DIR / "drops_translate.jsonl"
PROMPT_LOG_PATH = LOGS_DIR / "prompts_translate.jsonl"
HE_REPO = HOME / "data" / "temp_hej"

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

# nuprl/MultiPL-E java parquets store: prompt = complete Java signature
# (class Problem { ... method stub), tests = Java assert-main, original = a
# local FILE PATH (not the solution). There is NO python solution column.
# The correct reference build is JAVA-COMPLETION: complete the Java method with
# the model, then execution-gate it in gate.py. The parquet's own `prompt` and
# `tests` are the authoritative per-problem Java prompt + tests.

def build_chat_prompt(java_prompt):
    return (
        "Complete this Java method so it satisfies the specification in the "
        "docstring. Output ONLY the completed Java code (imports, class, and "
        "method). No explanation, no markdown fences.\n\n" + java_prompt
    )

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def write_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_jsonl_set(path, key):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)[key])
                except Exception:
                    pass
    return done


def load_meta():
    meta = {}
    if META_PATH.exists():
        with open(META_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    meta[row["bug_id"]] = row
                except Exception:
                    pass
    return meta


def sanitize_name(s):
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()
    return s or "problem"


def first_function_name(source, fallback):
    try:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
    except Exception:
        pass
    return fallback


def humaneval_name(entry_point, task_id):
    m = re.search(r"HumanEval/(\d+)", task_id or "")
    n = m.group(1) if m else (task_id or "0")
    return f"{n}_{sanitize_name(entry_point)}"


# ---------------------------------------------------------------------------
# Data loading (resumable meta log; builds it once from raw sources)
# ---------------------------------------------------------------------------

def build_meta():
    """Enumerate all problems -> rows with bug_id, source, java_prompt, java_tests.
    Authoritative source = nuprl/MultiPL-E java parquets (prompt = Java signature,
    tests = Java assert-main). Appends new rows to problems_meta.jsonl (resumable)."""
    meta = load_meta()
    rows = []

    # ---- MultiPL-E parquet via HF datasets (cached or downloaded) --------
    try:
        from datasets import load_dataset

        for config, source in (
            ("humaneval-java", "humaneval"),
            ("mbpp-java", "mbpp"),
        ):
            try:
                ds = load_dataset("nuprl/MultiPL-E", config)
            except Exception as e:
                print(f"[translate] could not load nuprl/MultiPL-E {config}: {e}")
                continue
            split = list(ds.keys())[0]
            for d in ds[split]:
                name = d.get("name") or ""
                java_prompt = (d.get("prompt") or "").strip()
                java_tests = (d.get("tests") or "").strip()
                if not java_prompt or not java_tests:
                    write_jsonl(DROPS_PATH, {"bug_id": name,
                                             "reason": "empty prompt/tests"})
                    continue
                if source == "humaneval":
                    # name = "HumanEval_<n>_<entry>" -> HE_<n>_<entry>
                    m = re.match(r"HumanEval[_/](\d+)_(.*)", name)
                    bug_id = (f"HE_{m.group(1)}_{sanitize_name(m.group(2))}"
                              if m else f"HE_{sanitize_name(name)}")
                else:
                    # name = "mbpp_<n>_<fname>" -> MBPP_<n>_<fname>
                    m = re.match(r"mbpp_?(\d+)_(.*)", name)
                    bug_id = (f"MBPP_{m.group(1)}_{sanitize_name(m.group(2))}"
                              if m else f"MBPP_{sanitize_name(name)}")
                if bug_id in meta:
                    continue
                row = {
                    "bug_id": bug_id,
                    "source": source,
                    "java_prompt": java_prompt,
                    "java_tests": java_tests,
                }
                meta[bug_id] = row
                write_jsonl(META_PATH, row)
                rows.append(row)
    except ImportError:
        print("[translate] WARNING: datasets not importable; cannot build corpus")

    rows = list(meta.values())
    return rows


# ---------------------------------------------------------------------------
# Java extraction / sanitization
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"```(?:java)?\s*\n(.*?)```", re.DOTALL)


def extract_java(text):
    m = FENCE_RE.search(text)
    code = m.group(1) if m else text
    code = code.strip()
    # strip lines that cannot appear inside a method body context we keep
    kept = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("package "):
            continue
        if s.startswith("public class ") or s.startswith("class "):
            continue
        kept.append(ln)
    code = "\n".join(kept).strip()
    # if removing class wrapper left dangling braces at the very end, trim extras
    # (best-effort: balance check happens later at javac time anyway)
    while code.count("}") > code.count("{"):
        idx = code.rfind("}")
        code = code[:idx] + code[idx + 1:]
    return code.strip()


def is_viable(java_code):
    if "(" not in java_code or "{" not in java_code:
        return False
    if not re.search(r"\b(public|private|protected|static|\w+)\s+[\w<>\[\], ?]+\s+\w+\s*\(", java_code):
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    problems = build_meta()
    problems.sort(key=lambda r: r["bug_id"])
    print(f"[translate] {len(problems)} problems known")

    dropped = load_jsonl_set(DROPS_PATH, "bug_id")
    todo = []
    for row in problems:
        bug_id = row["bug_id"]
        if bug_id in dropped:
            continue
        if (GEN_DIR / f"{bug_id}.java").exists():
            continue
        todo.append(row)
    print(f"[translate] {len(todo)} to translate "
          f"(skipped {len(problems) - len(todo)} existing/dropped)")
    if not todo:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    done = 0
    for row in todo:
        bug_id = row["bug_id"]
        java_prompt = row.get("java_prompt") or ""
        if not java_prompt.strip():
            write_jsonl(DROPS_PATH, {"bug_id": bug_id, "reason": "no java_prompt"})
            continue
        prompt = build_chat_prompt(java_prompt)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt  # fall back to raw prompt if no chat template
        write_jsonl(PROMPT_LOG_PATH, {"bug_id": bug_id, "prompt": text})
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        java_code = extract_java(gen_text)
        if not is_viable(java_code):
            write_jsonl(DROPS_PATH, {"bug_id": bug_id,
                                     "reason": "extraction_failed"})
            print(f"[translate] DROP {bug_id}: extraction failed")
            continue
        with open(GEN_DIR / f"{bug_id}.java", "w") as f:
            f.write(java_code + "\n")
        done += 1
        if done % 20 == 0:
            print(f"[translate] {done}/{len(todo)} translated")
    print(f"[translate] done: {done} new candidates in {GEN_DIR}")


if __name__ == "__main__":
    main()
