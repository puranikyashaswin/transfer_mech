#!/usr/bin/env python3
"""
junit_runner.py — Execution-verified JUnit harness for HumanEval-Java.

This is the ONLY ground-truth evaluator for transfer_mech. String similarity is
banned as evidence (PREREG v1 §Benchmarks). A solution "passes" iff its JUnit
test suite passes under `javac`/`java` with the dataset's JUnit jar.

Designed to run on EC2 (or any machine with a JDK). NOT for the MacBook.

Dataset layout expected (HumanEval-Java, Defects4J-style):
  <root>/src/main/java/humaneval/buggy/<BUG>.java      # buggy reference
  <root>/src/main/java/humaneval/correct/<BUG>.java    # correct reference
  <root>/src/test/java/humaneval/TEST_<BUG>.java       # JUnit test (targets buggy pkg)
  <root>/lib/junit4-4.12.jar, hamcrest-all-1.3.jar

The test files reference `humaneval.buggy.<BUG>`. To evaluate a *candidate* fix we
write the candidate into the `humaneval.buggy` package (same class name) in a
scratch dir, compile it, compile the test against it, and run JUnit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --- Configuration -----------------------------------------------------------

DEFAULT_TIMEOUT_S = 30          # per-bug wall-clock ceiling for compile+run
JUNIT_TIMEOUT_MS = 3000         # matches @Test(timeout=3000) in the dataset

# --- Java source helpers -----------------------------------------------------

_CLASS_RE = re.compile(r"\bpublic\s+(?:final\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)")
_PACKAGE_RE = re.compile(r"^\s*package\s+([a-zA-Z0-9_.]+)\s*;", re.MULTILINE)


def extract_class_name(java_src: str) -> str | None:
    m = _CLASS_RE.search(java_src)
    return m.group(1) if m else None


_METHOD_SIG_RE = re.compile(
    r"\bpublic\s+static\s+[\w<>\[\], ?\.]+\s+([a-zA-Z_][A-Za-z0-9_]*)\s*\(")


def find_method_span(java_src: str, method_name: str) -> tuple[int, int] | None:
    """Return (start, end) char offsets of the full `public static ... name(...) { ... }`
    block, using brace matching. None if not found."""
    for m in _METHOD_SIG_RE.finditer(java_src):
        if m.group(1) != method_name:
            continue
        # find opening brace of the body
        brace = java_src.find("{", m.end() - 1)
        if brace == -1:
            continue
        depth = 0
        i = brace
        while i < len(java_src):
            c = java_src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return m.start(), i + 1
            i += 1
    return None


def method_names(java_src: str) -> list[str]:
    return _METHOD_SIG_RE.findall(java_src)


def wrap_candidate_method(buggy_class_src: str, buggy_method_src: str,
                          candidate_method_src: str) -> str | None:
    """Wrap a model-generated *method* into the buggy class template by replacing
    the buggy method with the candidate method. Uses only the buggy class (which
    the model is allowed to see), so no reference leakage. Returns None if the
    buggy method cannot be located."""
    names = method_names(buggy_method_src)
    if not names:
        return None
    target = names[-1]  # the buggy method is the last public static in the snippet
    span = find_method_span(buggy_class_src, target)
    if span is None:
        return None
    s, e = span
    # keep the candidate's own indentation minimal; drop a leading package/import
    cand = candidate_method_src.strip()
    return buggy_class_src[:s] + cand + buggy_class_src[e:]


def set_package(java_src: str, package: str) -> str:
    """Force the source into `package` (replacing any existing package line)."""
    if _PACKAGE_RE.search(java_src):
        return _PACKAGE_RE.sub(f"package {package};", java_src, count=1)
    return f"package {package};\n\n" + java_src


# --- Core compile/run --------------------------------------------------------

def _run(cmd: list[str], cwd: str, timeout: int) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr). Never raises."""
    try:
        p = subprocess.run(
            cmd, cwd=cwd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return -2, "", f"LAUNCH_ERROR: {e}"


def evaluate_candidate(
    bug_id: str,
    candidate_src: str,
    test_src: str,
    javac: str,
    java: str,
    junit_cp: str,
    workdir: str,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict:
    """
    Compile `candidate_src` (as humaneval.buggy.<CLASS>) + its JUnit test, run it.
    Returns a result dict with a `status` field:
      "pass" | "test_fail" | "compile_error" | "runtime_error" | "timeout" | "error"
    """
    t0 = time.time()
    pkg = "humaneval.buggy"
    pkg_dir = os.path.join(workdir, *pkg.split("."))
    os.makedirs(pkg_dir, exist_ok=True)

    cls = extract_class_name(candidate_src)
    if cls is None:
        return {"bug_id": bug_id, "status": "error",
                "detail": "no public class found in candidate",
                "elapsed_s": round(time.time() - t0, 3)}

    candidate_src = set_package(candidate_src, pkg)
    cand_path = os.path.join(pkg_dir, f"{cls}.java")
    with open(cand_path, "w") as f:
        f.write(candidate_src)

    # Test file: TEST_<BUG>.java lives in package `humaneval` and imports buggy pkg.
    test_dir = os.path.join(workdir, "humaneval")
    os.makedirs(test_dir, exist_ok=True)
    test_cls = extract_class_name(test_src) or f"TEST_{bug_id}"
    test_path = os.path.join(test_dir, f"{test_cls}.java")
    with open(test_path, "w") as f:
        f.write(test_src)

    classes = os.path.join(workdir, "classes")
    os.makedirs(classes, exist_ok=True)

    # 1) compile candidate + test together
    rc, out, err = _run(
        [javac, "-cp", junit_cp, "-d", classes, cand_path, test_path],
        cwd=workdir, timeout=timeout,
    )
    if rc != 0:
        status = "timeout" if rc == -1 else "compile_error"
        return {"bug_id": bug_id, "status": status,
                "detail": (err or out)[:2000],
                "elapsed_s": round(time.time() - t0, 3)}

    # 2) run JUnit
    run_cp = f"{classes}{os.pathsep}{junit_cp}"
    rc, out, err = _run(
        [java, "-cp", run_cp, "org.junit.runner.JUnitCore", f"humaneval.{test_cls}"],
        cwd=workdir, timeout=timeout,
    )
    elapsed = round(time.time() - t0, 3)
    if rc == -1:
        status = "timeout"
    elif rc == 0:
        status = "pass"
    else:
        # JUnitCore returns 1 on test failure; a crash/NoClassDefFound is runtime_error
        status = "test_fail" if ("FAILURES" in out or "OK" not in out) else "runtime_error"
        if "Could not find or load main class" in err or "NoClassDefFoundError" in err:
            status = "runtime_error"
    return {"bug_id": bug_id, "status": status,
            "detail": (out + "\n" + err)[:2000],
            "elapsed_s": elapsed}


def evaluate_candidate_default_pkg(
    bug_id: str,
    candidate_src: str,
    test_src: str,
    javac: str,
    java: str,
    junit_cp: str,
    workdir: str,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict:
    """
    Default-package evaluator for the REBUILT confirmatory corpus (Amendment 006).
    The corpus TESTs were rewritten (fix_tests.py) to the DEFAULT package with
    class-qualified calls `<BUG>.method(...)` (no `package`, no `import
    humaneval.*`). The candidate must therefore be a DEFAULT-package class named
    exactly <BUG> exposing the method(s) the test invokes. We:
      - strip any package line from the candidate (force default package),
      - write candidate + test FLAT in the workdir,
      - compile together, run JUnit (asserts are real JUnit assert* -> no -ea needed).
    Returns the same status dict as evaluate_candidate.
    """
    t0 = time.time()
    cls = extract_class_name(candidate_src)
    if cls is None:
        return {"bug_id": bug_id, "status": "error",
                "detail": "no public class found in candidate",
                "elapsed_s": round(time.time() - t0, 3)}
    # Force default package (the corpus TESTs are default-package).
    candidate_src = _PACKAGE_RE.sub("", candidate_src, count=1)
    cand_path = os.path.join(workdir, f"{cls}.java")
    with open(cand_path, "w") as f:
        f.write(candidate_src)

    test_cls = extract_class_name(test_src) or f"TEST_{bug_id}"
    test_path = os.path.join(workdir, f"{test_cls}.java")
    with open(test_path, "w") as f:
        f.write(test_src)

    classes = os.path.join(workdir, "classes")
    os.makedirs(classes, exist_ok=True)
    rc, out, err = _run(
        [javac, "-cp", junit_cp, "-d", classes, cand_path, test_path],
        cwd=workdir, timeout=timeout,
    )
    if rc != 0:
        status = "timeout" if rc == -1 else "compile_error"
        return {"bug_id": bug_id, "status": status,
                "detail": (err or out)[:2000],
                "elapsed_s": round(time.time() - t0, 3)}
    run_cp = f"{classes}{os.pathsep}{junit_cp}"
    rc, out, err = _run(
        [java, "-cp", run_cp, "org.junit.runner.JUnitCore", test_cls],
        cwd=workdir, timeout=timeout,
    )
    elapsed = round(time.time() - t0, 3)
    if rc == -1:
        status = "timeout"
    elif rc == 0:
        status = "pass"
    else:
        status = "test_fail" if ("FAILURES" in out or "OK" not in out) else "runtime_error"
        if "Could not find or load main class" in err or "NoClassDefFoundError" in err:
            status = "runtime_error"
    return {"bug_id": bug_id, "status": status,
            "detail": (out + "\n" + err)[:2000],
            "elapsed_s": elapsed}


# --- Batch driver ------------------------------------------------------------

def load_dataset(root: Path) -> dict[str, dict]:
    """Map bug_id -> {buggy, correct, test} source strings."""
    buggy_dir = root / "src/main/java/humaneval/buggy"
    correct_dir = root / "src/main/java/humaneval/correct"
    test_dir = root / "src/test/java/humaneval"
    data: dict[str, dict] = {}
    for tf in sorted(test_dir.glob("TEST_*.java")):
        bug_id = tf.stem[len("TEST_"):]
        entry = {"test": tf.read_text()}
        bf = buggy_dir / f"{bug_id}.java"
        cf = correct_dir / f"{bug_id}.java"
        entry["buggy"] = bf.read_text() if bf.exists() else None
        entry["correct"] = cf.read_text() if cf.exists() else None
        data[bug_id] = entry
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="HumanEval-Java JUnit evaluator")
    ap.add_argument("--dataset", required=True, help="dataset root (temp_hej layout)")
    ap.add_argument("--mode", choices=["correct", "buggy"], required=True,
                    help="which reference to evaluate (Gate 0 uses both)")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--java", default="java")
    ap.add_argument("--junit-cp", default=None,
                    help="path to junit jar (default: <dataset>/lib/*)")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.dataset)
    junit_cp = args.junit_cp
    if junit_cp is None:
        jars = sorted(str(p) for p in (root / "lib").glob("*.jar"))
        junit_cp = os.pathsep.join(jars)

    data = load_dataset(root)
    bug_ids = sorted(data)[: args.limit] if args.limit else sorted(data)

    n_pass = 0
    with open(args.out, "w") as outf:
        for i, bug_id in enumerate(bug_ids):
            src = data[bug_id][args.mode]
            if src is None:
                rec = {"bug_id": bug_id, "status": "error", "detail": f"no {args.mode} source"}
            else:
                with tempfile.TemporaryDirectory() as wd:
                    rec = evaluate_candidate(
                        bug_id, src, data[bug_id]["test"],
                        args.javac, args.java, junit_cp, wd, args.timeout,
                    )
            rec["mode"] = args.mode
            if rec["status"] == "pass":
                n_pass += 1
            outf.write(json.dumps(rec) + "\n")
            outf.flush()
            print(f"[{i+1:3d}/{len(bug_ids)}] {bug_id:<35} {rec['status']}", flush=True)

    total = len(bug_ids)
    print(f"\nMODE={args.mode}  PASS {n_pass}/{total} = {n_pass/total*100:.1f}%")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
