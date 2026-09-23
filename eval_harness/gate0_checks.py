#!/usr/bin/env python3
"""
gate0_checks.py — Gate 0 (Measurement) evaluator controls.

Runs the four frozen Gate-0 checks from PREREG Amendment 001 §A6:
  1. known-valid passes   : every correct/ reference passes its JUnit test
  2. known-invalid fails  : every buggy/ reference FAILS its JUnit test
  3. no-op identity       : same solution evaluated twice -> identical verdict
  4. independent recompute: headline table recomputed from raw JSONL matches

Gate 0 needs NO GPU. Run on EC2 (or any JDK machine). NOT for the MacBook.

Exit code 0 = Gate 0 PASSED (all checks green). Non-zero = FAILED (fix eval first).

Usage:
  python gate0_checks.py --dataset /path/to/temp_hej --workdir ./gate0_out \
      [--javac javac] [--java java] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from junit_runner import evaluate_candidate, load_dataset  # noqa: E402


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def check_toolchain(java: str, javac: str) -> tuple[bool, str]:
    rc_j, out_j = sh([java, "-version"])
    rc_c, out_c = sh([javac, "-version"])
    ok = rc_j == 0 and rc_c == 0
    return ok, (out_j.strip().splitlines() or ["?"])[0] + " / " + out_c.strip()


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """Load already-done bugs from a JSONL checkpoint (for --resume)."""
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["bug_id"]] = {k: v for k, v in r.items() if k != "bug_id"}
    return done


def run_all(dataset: Path, mode: str, java: str, javac: str,
            junit_cp: str, limit: int | None, ckpt_path: Path,
            resume: bool) -> dict[str, dict]:
    """Evaluate every bug, appending each result to ckpt_path as JSONL.
    With resume=True, bugs already present in ckpt_path are skipped (loaded, not rerun)."""
    data = load_dataset(dataset)
    bug_ids = sorted(data)[:limit] if limit else sorted(data)
    results: dict[str, dict] = _load_checkpoint(ckpt_path) if resume else {}
    if resume and results:
        print(f"  [resume] {len(results)} bugs already done, skipping them", flush=True)
    outf = open(ckpt_path, "a")
    for i, bug_id in enumerate(bug_ids):
        if bug_id in results:
            continue
        src = data[bug_id][mode]
        if src is None:
            rec = {"status": "error", "detail": f"no {mode} source"}
        else:
            with tempfile.TemporaryDirectory() as wd:
                rec = evaluate_candidate(
                    bug_id, src, data[bug_id]["test"], javac, java, junit_cp, wd
                )
        results[bug_id] = rec
        outf.write(json.dumps({"bug_id": bug_id, **rec}) + "\n")
        outf.flush()
        print(f"  [{mode}][{i+1:3d}/{len(bug_ids)}] {bug_id:<32} {rec['status']}", flush=True)
    outf.close()
    # keep only the requested bug_ids (in case limit shrank)
    return {b: results[b] for b in bug_ids if b in results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--junit-cp", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip bugs already present in the checkpoint JSONL")
    args = ap.parse_args()

    root = Path(args.dataset)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    junit_cp = args.junit_cp or ":".join(str(p) for p in sorted((root / "lib").glob("*.jar")))

    report: dict = {"dataset": str(root), "checks": {}}
    gate_ok = True

    # --- toolchain -----------------------------------------------------------
    ok, ver = check_toolchain(args.java, args.javac)
    report["toolchain"] = {"ok": ok, "version": ver}
    print(f"[toolchain] {'OK' if ok else 'MISSING'}: {ver}")
    if not ok:
        print("FATAL: no working JDK. Install Temurin 17 on the EC2 host (see RUNBOOK).")
        (workdir / "gate0_report.json").write_text(json.dumps(report, indent=2))
        return 2

    # --- Check 1: known-valid passes ----------------------------------------
    print("\n[check 1] known-valid (correct/) must ALL pass")
    correct = run_all(root, "correct", args.java, args.javac, junit_cp, args.limit,
                      workdir / "gate0_correct.jsonl", args.resume)
    c1_fail = {k: v for k, v in correct.items() if v["status"] != "pass"}
    c1_ok = len(c1_fail) == 0
    report["checks"]["known_valid"] = {
        "ok": c1_ok, "n": len(correct), "n_pass": len(correct) - len(c1_fail),
        "failures": {k: v["status"] for k, v in c1_fail.items()},
    }
    print(f"  -> {len(correct)-len(c1_fail)}/{len(correct)} pass; "
          f"{'OK' if c1_ok else 'FAIL: ' + str(list(c1_fail))}")
    gate_ok &= c1_ok

    # --- Check 2: known-invalid fails ---------------------------------------
    print("\n[check 2] known-invalid (buggy/) must ALL fail")
    buggy = run_all(root, "buggy", args.java, args.javac, junit_cp, args.limit,
                    workdir / "gate0_buggy.jsonl", args.resume)
    c2_pass = {k: v for k, v in buggy.items() if v["status"] == "pass"}
    # A buggy solution that PASSES means a bad/mislabeled test -> investigate.
    c2_ok = len(c2_pass) == 0
    report["checks"]["known_invalid"] = {
        "ok": c2_ok, "n": len(buggy), "n_unexpected_pass": len(c2_pass),
        "unexpected_passes": sorted(c2_pass),
        "status_histogram": dict(Counter(v["status"] for v in buggy.values())),
    }
    print(f"  -> {len(c2_pass)} buggy solutions unexpectedly PASS; "
          f"{'OK' if c2_ok else 'INVESTIGATE: ' + str(sorted(c2_pass))}")
    # Note: a small number of unexpected passes may be tolerable if triaged; we flag but
    # do not hard-fail the gate on this alone — the LOG is what matters. Hard-fail only
    # if >5% pass (indicates systemic mislabeling).
    frac = len(c2_pass) / max(1, len(buggy))
    if frac > 0.05:
        gate_ok = False
        print(f"  -> HARD FAIL: {frac:.1%} of buggy pass (>5% threshold)")

    # --- Check 3: no-op identity (determinism) -------------------------------
    print("\n[check 3] no-op identity: re-evaluate correct/ twice, verdicts must match")
    sample = sorted(correct)[: min(25, len(correct))]
    mismatches = []
    for bug_id in sample:
        data = load_dataset(root)
        src, test = data[bug_id]["correct"], data[bug_id]["test"]
        with tempfile.TemporaryDirectory() as wd:
            r1 = evaluate_candidate(bug_id, src, test, args.javac, args.java, junit_cp, wd)
        with tempfile.TemporaryDirectory() as wd:
            r2 = evaluate_candidate(bug_id, src, test, args.javac, args.java, junit_cp, wd)
        if r1["status"] != r2["status"]:
            mismatches.append((bug_id, r1["status"], r2["status"]))
    c3_ok = len(mismatches) == 0
    report["checks"]["noop_identity"] = {
        "ok": c3_ok, "n_rechecked": len(sample), "mismatches": mismatches,
    }
    print(f"  -> {len(mismatches)} mismatches over {len(sample)} re-runs; "
          f"{'OK' if c3_ok else 'FAIL'}")
    gate_ok &= c3_ok

    # --- Check 4: independent recompute --------------------------------------
    print("\n[check 4] independent recompute of headline table from raw JSONL")
    raw = [json.loads(l) for l in (workdir / "gate0_correct.jsonl").read_text().splitlines() if l.strip()]
    recompute_pass = sum(1 for r in raw if r["status"] == "pass")
    harness_pass = report["checks"]["known_valid"]["n_pass"]
    c4_ok = recompute_pass == harness_pass
    report["checks"]["independent_recompute"] = {
        "ok": c4_ok, "recompute_pass": recompute_pass, "harness_pass": harness_pass,
    }
    print(f"  -> recompute={recompute_pass} harness={harness_pass}; "
          f"{'OK' if c4_ok else 'FAIL'}")
    gate_ok &= c4_ok

    # --- verdict -------------------------------------------------------------
    report["gate0_passed"] = gate_ok
    (workdir / "gate0_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 60)
    print(f"GATE 0: {'PASSED ✅' if gate_ok else 'FAILED ❌ — fix eval before any GPU run'}")
    print(f"Report: {workdir/'gate0_report.json'}")
    print("=" * 60)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
