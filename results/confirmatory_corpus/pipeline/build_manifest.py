#!/usr/bin/env python3
"""
build_manifest.py -- Stage 4 of the confirmatory_hej corpus build.

For each problem present in all three dirs (correct + buggy + test), emit one
row to bugs_manifest.jsonl:
  {"bug_id", "buggy_code", "prompt", "tests", "test_file", "fixed_code"}
All fields must be non-empty; problems failing validation are logged to
logs/drops.jsonl and excluded.

Idempotent: rewrites bugs_manifest.jsonl from scratch each run (deterministic
row order, sorted by bug_id), so it is safe to re-run after any stage.

Run: python3 build_manifest.py
"""

import json
import os
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
OUT_ROOT = HOME / "transfer_mech" / "data" / "confirmatory_hej"
SRC_CORRECT = OUT_ROOT / "src" / "main" / "java" / "humaneval" / "correct"
SRC_BUGGY = OUT_ROOT / "src" / "main" / "java" / "humaneval" / "buggy"
SRC_TEST = OUT_ROOT / "src" / "test" / "java" / "humaneval"
LOGS_DIR = OUT_ROOT / "logs"
META_PATH = LOGS_DIR / "problems_meta.jsonl"
DROPS_PATH = LOGS_DIR / "drops.jsonl"
MANIFEST_PATH = OUT_ROOT / "bugs_manifest.jsonl"


def write_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


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


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()

    rows = []
    n_missing = n_invalid = 0
    buggy_files = sorted(SRC_BUGGY.glob("*.java"))
    for bf in buggy_files:
        bug_id = bf.stem
        correct_path = SRC_CORRECT / f"{bug_id}.java"
        test_path = SRC_TEST / f"TEST_{bug_id}.java"
        if not correct_path.exists() or not test_path.exists():
            n_missing += 1
            write_jsonl(DROPS_PATH, {
                "bug_id": bug_id, "stage": "manifest",
                "reason": "incomplete_triple",
                "detail": f"correct={correct_path.exists()} test={test_path.exists()}",
            })
            continue

        buggy_code = bf.read_text()
        fixed_code = correct_path.read_text()
        tests = test_path.read_text()
        # new schema stores the Java prompt as java_prompt; fall back to legacy "prompt"
        mrow = meta.get(bug_id) or {}
        prompt = (mrow.get("java_prompt") or mrow.get("prompt") or "")

        row = {
            "bug_id": bug_id,
            "buggy_code": buggy_code,
            "prompt": prompt,
            "tests": tests,
            "test_file": test_path.name,
            "fixed_code": fixed_code,
        }
        bad = [k for k, v in row.items() if not (isinstance(v, str) and v.strip())]
        if bad:
            n_invalid += 1
            write_jsonl(DROPS_PATH, {
                "bug_id": bug_id, "stage": "manifest",
                "reason": "empty_field",
                "detail": ",".join(bad),
            })
            continue
        rows.append(row)

    rows.sort(key=lambda r: r["bug_id"])
    with open(MANIFEST_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"[manifest] wrote {len(rows)} rows to {MANIFEST_PATH} "
          f"(skipped incomplete={n_missing} invalid={n_invalid})")


if __name__ == "__main__":
    main()
