#!/usr/bin/env python3
"""
taxonomy_recompile.py — exact compile-error root-cause taxonomy.

Recompiles the adapter arm's compile_error generations against the real corpus
TESTs (default-package path, identical to eval_adapter.py) and parses javac's
actual diagnostics into pre-registered root-cause buckets. NO model needed — pure
recompilation of stored `code` strings.

Buckets (from javac message patterns):
  unbalanced_extra_close   : "class, interface, or enum expected" after a balanced
                             class body -> spurious trailing '}'
  illegal_package          : package stmt names a java.* / reserved namespace
  unresolved_symbol        : "cannot find symbol" (undeclared var/type/method)
  incompatible_types       : "incompatible types" (bad return / assignment)
  missing_return           : "missing return statement"
  unclosed_string/char     : "unclosed string literal" / char literal
  not_a_statement          : "not a statement"
  truncation_eof           : "reached end of file while parsing"
  other                    : anything else, message preserved

Runs ON the instance (needs javac + junit jars + corpus tests).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from junit_runner import evaluate_candidate_default_pkg  # noqa: E402


def retarget_class_name(code: str, bug_id: str) -> str:
    m = re.compile(r"\bpublic\s+(?:final\s+)?class\s+[A-Za-z_][A-Za-z0-9_]*")
    return m.sub(f"public class {bug_id}", code, count=1) if m.search(code) else code


def bucket(err: str, code: str) -> str:
    e = err.lower()
    if "reached end of file while parsing" in e:
        return "truncation_eof"
    if "unclosed string literal" in e or "unclosed character literal" in e:
        return "unclosed_literal"
    if "class, interface, or enum expected" in e:
        return "unbalanced_extra_close"
    if re.search(r"package (java|javax)\.", code):
        return "illegal_package_reserved"
    if "cannot find symbol" in e:
        return "unresolved_symbol"
    if "incompatible types" in e:
        return "incompatible_types"
    if "missing return statement" in e:
        return "missing_return"
    if "not a statement" in e:
        return "not_a_statement"
    if "illegal start" in e:
        return "illegal_start"
    if "';' expected" in e or "'}' expected" in e:
        return "missing_delimiter"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-out", required=True, help="confirmatory eval_out.jsonl")
    ap.add_argument("--manifest", required=True,
                    help="bugs_manifest.jsonl with 'tests' field (authoritative test source)")
    ap.add_argument("--junit-cp", required=True, help="junit:javatuples:hamcrest classpath")
    ap.add_argument("--arm", default="adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--java", default="java")
    args = ap.parse_args()

    junit_cp = args.junit_cp
    data = {}
    for line in open(args.manifest):
        r = json.loads(line)
        data[r["bug_id"]] = {"test": r["tests"]}

    rows = []
    for line in open(args.eval_out):
        r = json.loads(line)
        if r.get("arm") == args.arm and r.get("status") == "compile_error":
            rows.append(r)
    print(f"recompiling {len(rows)} {args.arm} compile_errors", flush=True)

    import collections
    counts = collections.Counter()
    detail = []
    for r in rows:
        bug_id = r["bug_id"]
        code = retarget_class_name(r.get("code", ""), bug_id)
        ent = data.get(bug_id) or data.get(f"humaneval_java_{bug_id}")
        if ent is None:
            counts["no_test_in_dataset"] += 1
            detail.append({"bug_id": bug_id, "bucket": "no_test_in_dataset",
                           "first_error": ""})
            print(f"{bug_id:<28} no_test_in_dataset", flush=True)
            continue
        test_src = ent["test"]
        with tempfile.TemporaryDirectory() as wd:
            res = evaluate_candidate_default_pkg(bug_id, code, test_src,
                                                 args.javac, args.java, junit_cp, wd)
        err = res.get("detail", "")
        b = bucket(err, code)
        counts[b] += 1
        first_err = next((ln.strip() for ln in err.splitlines() if "error:" in ln), "")
        detail.append({"bug_id": bug_id, "bucket": b, "first_error": first_err[:200]})
        print(f"{bug_id:<28} {b:<24} {first_err[:80]}", flush=True)

    summary = {"arm": args.arm, "n": len(rows), "counts": dict(counts), "detail": detail}
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print("\n=== TAXONOMY ===", flush=True)
    for k, v in counts.most_common():
        print(f"{v:3d}  {k}", flush=True)
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
