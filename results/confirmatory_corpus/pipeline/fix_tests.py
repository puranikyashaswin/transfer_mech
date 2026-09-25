#!/usr/bin/env python3
"""Fix the 345 TEST_<BUG>.java files so unqualified method calls resolve.

The TEST files gate.py wrote call the entry method UNQUALIFIED (e.g.
`assertTrue((stringXor("1","1").equals("0")));`) but the method lives in the
class `<BUG_ID>`. Inside `public class TEST_<BUG_ID>` that call does not resolve
-> "cannot find symbol" -> compile_error. Fix: qualify each call to a public
static method of the subject class as `<BUG_ID>.<method>(...)`.

Strategy per TEST_<BUG>.java:
  1. Read the matching correct class; collect its `public static <ret> <name>(`
     method names.
  2. In the @Test body, replace a bare `<name>(` that is NOT already preceded by
     `.` or an identifier char (i.e. not already qualified, not a JUnit assertX)
     with `<BUG_ID>.<name>(`.
  3. ALSO strip the `package humaneval;` + `import humaneval.correct.<BUG>;`
     lines and rewrite the test to the DEFAULT package so mutate.py's flat-dir
     compile works. The class reference stays `<BUG_ID>` (same default package).

Idempotent: skips files already rewritten (marker comment). Writes in place.
"""
import re
import sys
from pathlib import Path

BASE = Path("/home/ubuntu/transfer_mech/data/confirmatory_hej")
SRC_CORRECT = BASE / "src/main/java/humaneval/correct"
SRC_TEST = BASE / "src/test/java/humaneval"

MARK = "// QUALIFIED-FIX v1"
METHOD_RE = re.compile(r"\bpublic\s+static\s+[\w<>\[\].,? ]+?\s+(\w+)\s*\(")


def public_static_methods(correct_src):
    return [m.group(1) for m in METHOD_RE.finditer(correct_src)]


def qualify(body, cls, methods):
    for name in sorted(set(methods), key=len, reverse=True):
        # qualify a call not already preceded by '.', word char, or 'assert'
        body = re.sub(rf"(?<![\w.])(?<![\w]){re.escape(name)}\s*\(",
                      f"{cls}.{name}(", body)
    return body


def fix_one(test_path):
    bug_id = test_path.stem[len("TEST_"):]
    correct = SRC_CORRECT / f"{bug_id}.java"
    if not correct.exists():
        return "no_correct"
    src = test_path.read_text()
    if MARK in src:
        return "already"
    methods = public_static_methods(correct.read_text())
    # default package: drop the package decl + the humaneval.correct import
    out = re.sub(r"^\s*package\s+humaneval;\s*\n", "", src, count=1, flags=re.M)
    out = re.sub(r"^\s*import\s+humaneval\.correct\.\w+;\s*\n", "", out,
                 count=1, flags=re.M)
    out = qualify(out, bug_id, methods)
    out = f"{MARK}\n" + out
    test_path.write_text(out)
    return "fixed"


def main():
    files = sorted(SRC_TEST.glob("TEST_*.java"))
    from collections import Counter
    res = Counter(fix_one(f) for f in files)
    print(f"[fix_tests] {len(files)} files -> {dict(res)}")


if __name__ == "__main__":
    sys.exit(main())
