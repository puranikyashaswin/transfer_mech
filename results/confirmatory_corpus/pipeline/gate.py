#!/usr/bin/env python3
"""
gate.py -- Stage 2 of the confirmatory_hej corpus build.

Java-completion + assert-gating architecture:

  translate.py emits gen/<bug_id>.java = the 3B model's raw completion of a
  Java prompt. The authoritative per-problem metadata (MultiPL-E java_prompt +
  java_tests) lives in logs/problems_meta.jsonl.

  * java_prompt  = the FULL MultiPL-E Java prompt: import lines +
                   `class Problem {` + docstring-comment +
                   `public static <ret> <name>(<args>) {` with an OPEN brace,
                   NOT closed.
  * gen/<bug_id>.java = the model's method-body completion (usually ends the
                   method with `}`).
  * java_tests   = the MultiPL-E test tail: starts with `    }` (closing the
                   method), then `public static void main(String[] args) {`
                   + `assert(...)` lines + closing braces -- the continuation
                   that completes `class Problem`.

  A complete compilable Java file = java_prompt + completion + java_tests,
  with EXACTLY ONE leading `}` deduped from java_tests when the completion
  already ends the method (see assemble_full_source).

  GATING = plain-Java assert execution, NOT JUnit:
      javac -cp lib/javatuples-1.2.jar Problem.java
      java  -ea -cp .:lib/javatuples-1.2.jar Problem      (60s timeout)
    rc==0                       -> pass
    rc==1 with AssertionError   -> test_fail
    rc==124 (timeout)           -> test_fail
    javac rc!=0                 -> compile_error

  On PASS, emit the downstream artifacts:
    * src/main/java/humaneval/correct/<BUG_ID>.java
        package humaneval.correct;  +  public class <BUG_ID> with the method
        kept PUBLIC STATIC and the `public static void main` assert block
        REMOVED (it is the test, not the reference).
    * src/test/java/humaneval/TEST_<BUG_ID>.java
        package humaneval; + JUnit4 (@Test(timeout=3000)) whose body is the
        assert-main's assert statements converted to assertEquals/assertTrue,
        calling humaneval.correct.<BUG_ID>.<method>(...).

Resumable: skips bug_ids already kept (src/main/java/humaneval/correct exists)
or already dropped (logs/drops.jsonl). Writes NOTHING to /opt/dlami/nvme --
all scratch work uses tempfile.TemporaryDirectory defaults.

Run: python3 gate.py
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
OUT_ROOT = HOME / "transfer_mech" / "data" / "confirmatory_hej"
GEN_DIR = OUT_ROOT / "gen"
SRC_CORRECT = OUT_ROOT / "src" / "main" / "java" / "humaneval" / "correct"
SRC_BUGGY = OUT_ROOT / "src" / "main" / "java" / "humaneval" / "buggy"
SRC_TEST = OUT_ROOT / "src" / "test" / "java" / "humaneval"
LIB_DIR = OUT_ROOT / "lib"
LOGS_DIR = OUT_ROOT / "logs"
META_PATH = LOGS_DIR / "problems_meta.jsonl"
DROPS_PATH = LOGS_DIR / "drops.jsonl"

JAVATUPLES_JAR = LIB_DIR / "javatuples-1.2.jar"
JUNIT_JAR = LIB_DIR / "junit4-4.12.jar"
HAMCREST_JAR = LIB_DIR / "hamcrest-all-1.3.jar"

GATE_COMPILE_CP = str(JAVATUPLES_JAR)
GATE_RUN_CP = f".:{JAVATUPLES_JAR}"

TEST_TIMEOUT_MS = 3000
JAVA_RUN_TIMEOUT_S = 60
COMPILE_TIMEOUT_S = 60

# ---------------------------------------------------------------------------
# Generic helpers (shared verbatim with mutate.py)
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


def run(cmd, cwd, timeout):
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                           capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"


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


# ---------------------------------------------------------------------------
# Assembly: java_prompt + completion + java_tests -> compilable Problem.java
# ---------------------------------------------------------------------------

def assemble_full_source(java_prompt, completion, java_tests):
    """Assemble one compilable Problem.java from the MultiPL-E parts.

    The model completion (via translate.extract_java) is typically SELF-CONTAINED:
    it re-includes the leading `import ...;` lines AND the full method header+body.
    The prompt is `import...` + `class Problem {` + docstring + `public static ... {`
    (an OPEN method stub). The tests are `    }` (close method) + assert-main + `}`
    (close class). Concatenating naively duplicates imports AND the method header.

    Strategy: imports = the completion's import lines (deduped). Body =
    `class Problem {` + the prompt's docstring + the completion's NON-import lines
    (method header + body). Then append java_tests with exactly ONE leading `}`
    stripped (the completion's method already closes with `}`; java_tests opens with
    the method-closing `}`), leaving the final `}` to close the class.
    """
    comp_lines = [l for l in (completion or "").strip().splitlines()]
    comp_imports = [l.strip() for l in comp_lines if l.strip().startswith("import ")]
    comp_body = [l for l in comp_lines if not l.strip().startswith("import ")]

    # prompt's docstring/comment lines between `class Problem {` and the method stub
    prompt_lines = (java_prompt or "").splitlines()
    docstring = []
    seen_class = False
    for l in prompt_lines:
        if re.search(r"\bclass\s+Problem\b", l):
            seen_class = True
            continue
        if seen_class:
            if re.search(r"\b(public|private|protected)\s+static\b", l):
                break  # reached the method stub; stop collecting docstring
            docstring.append(l)

    out = []
    out.extend(_dedupe(comp_imports))
    out.append("")
    out.append("class Problem {")
    out.extend(docstring)
    out.extend(comp_body)
    base = "\n".join(out).rstrip() + "\n"

    tests = (java_tests or "").strip("\n")
    if (completion or "").strip().endswith("}"):
        lines = tests.splitlines()
        for i, ln in enumerate(lines):
            if ln.strip() == "}":
                del lines[i]
                break
            if ln.strip():  # first non-blank line is not a lone `}`; stop
                break
        tests = "\n".join(lines)
    return base + tests.rstrip() + "\n"


# ---------------------------------------------------------------------------
# assert-main -> pieces (method reference source + JUnit test source)
# ---------------------------------------------------------------------------

def _split_imports_and_rest(src):
    imports, rest = [], []
    for ln in src.splitlines():
        if ln.strip().startswith("import "):
            imports.append(ln.strip())
        else:
            rest.append(ln)
    return imports, rest


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _find_main_span(lines):
    """Return (start_idx, end_idx_exclusive) of the `public static void main`
    block (brace-matched), or (None, None)."""
    for i, ln in enumerate(lines):
        if re.search(r"\bpublic\s+static\s+void\s+main\s*\(", ln):
            depth = 0
            opened = False
            for j in range(i, len(lines)):
                depth += lines[j].count("{")
                if "{" in lines[j]:
                    opened = True
                depth -= lines[j].count("}")
                if opened and depth == 0:
                    return i, j + 1
            return i, len(lines)
    return None, None


ASSERT_EQ_RE = re.compile(
    r"^\s*assert\s+(.+?)\s*==\s*(.+?)\s*;\s*(?://.*)?$"
)
ASSERT_BARE_RE = re.compile(
    r"^\s*assert\s+(.+?)\s*;\s*(?://.*)?$"
)


FLOAT_LIT_RE = re.compile(
    r"^(?:\d+\.\d*(?:[eE][+-]?\d+)?[fFdD]?|\.\d+(?:[eE][+-]?\d+)?[fFdD]?|\d+[fFdD])$"
)


def _convert_assert_line(ln):
    """Java `assert` -> JUnit. Returns converted line (or original if no match).

    Edge cases handled:
      * `assert EXPR == true/false;` -> assertTrue/assertFalse (avoids the
        ambiguous assertEquals overload resolution entirely).
      * `assert EXPR == <float-literal>;` -> 3-arg assertEquals with delta,
        because JUnit4's 2-arg assertEquals(double, double) is deprecated and
        THROWS AssertionError at runtime. The delta form is only used when the
        EXPECTED side is a float literal (overloads resolve via widening).
      * multiple top-level `==` (e.g. `assert a == b && c == d;`) cannot be
        split reliably -- keep the whole boolean expression as assertTrue."""
    indent = ln[:len(ln) - len(ln.lstrip())]
    # MultiPL-E asserts come in TWO forms:
    #   `assert x == 5;`        (whitespace form; handled by the regexes above)
    #   `assert(expr == 5);`    (parenthesized form -- THE ACTUAL DATA FORMAT)
    # The regexes above require whitespace after `assert`, so they MISS the
    # parenthesized form entirely -> raw asserts (no-op without java -ea) leak into
    # the TEST files and every mutant "passes" vacuously. Handle the parenthesized
    # form FIRST: strip `assert( ... )` -> inner expr, then classify.
    pm = re.match(r"^\s*assert\s*\((.*)\)\s*;\s*$", ln)
    if pm:
        expr = pm.group(1).strip()
        if re.search(r"\s==\s*true\s*$", expr):
            return f"{indent}assertTrue(({expr[:-len('== true')].strip()}));"
        if re.search(r"\s==\s*false\s*$", expr):
            return f"{indent}assertFalse(({expr[:-len('== false')].strip()}));"
        m2 = re.match(r"^(.+?)\s*==\s*([^=]+?)\s*$", expr)
        if m2:
            lhs, rhs = m2.group(1).strip(), m2.group(2).strip()
            if FLOAT_LIT_RE.match(rhs):
                return f"{indent}assertEquals(({rhs}), ({lhs}), 1e-4);"
            return f"{indent}assertEquals(({rhs}), ({lhs}));"
        return f"{indent}assertTrue(({expr}));"
    m = ASSERT_EQ_RE.match(ln)
    if m:
        expr, expected = m.group(1).strip(), m.group(2).strip()
        if "==" in expr or "==" in expected:
            bare = ASSERT_BARE_RE.match(ln)
            return f"{indent}assertTrue(({bare.group(1).strip()}));"
        if expected == "true":
            return f"{indent}assertTrue(({expr}));"
        if expected == "false":
            return f"{indent}assertFalse(({expr}));"
        if FLOAT_LIT_RE.match(expected):
            return (f"{indent}assertEquals(({expected}), ({expr}), 1e-4);")
        return f"{indent}assertEquals(({expected}), ({expr}));"
    m = ASSERT_BARE_RE.match(ln)
    if m:
        expr = m.group(1).strip()
        return f"{indent}assertTrue(({expr}));"
    return ln


def make_reference_and_test(java_prompt, completion, java_tests, bug_id):
    """Build (correct_reference_source, junit_test_source)."""
    full = assemble_full_source(java_prompt, completion, java_tests)
    imports, rest = _split_imports_and_rest(full)

    # rename `class Problem` (any modifier form) -> `public class <bug_id>`
    rest_text = "\n".join(rest)
    rest_text = re.sub(r"\b(public\s+)?class\s+Problem\b",
                       f"public class {bug_id}", rest_text, count=1)
    lines = rest_text.splitlines()

    # locate + extract the main block
    mstart, mend = _find_main_span(lines)
    main_lines = lines[mstart:mend] if mstart is not None else []
    ref_body = [ln for i, ln in enumerate(lines)
                if not (mstart is not None and mstart <= i < mend)]

    # strip now-dangling blank lines at the tail of the class body
    while ref_body and not ref_body[-1].strip():
        ref_body.pop()

    ref = "package humaneval.correct;\n\n"
    for ln in _dedupe(imports):
        ref += ln + "\n"
    ref += "\n" + "\n".join(ref_body).rstrip() + "\n"
    if not ref.rstrip().endswith("}"):
        ref = ref.rstrip() + "\n}\n"

    # ---- JUnit test ---------------------------------------------------------
    assert_src_lines = []
    for ln in main_lines:
        if re.match(r"^\s*assert\b", ln):
            assert_src_lines.append(_convert_assert_line(ln))
    body = "\n".join(assert_src_lines) if assert_src_lines else "        // no asserts"

    test = "package humaneval;\n\n"
    test += "import org.junit.Test;\n"
    test += "import static org.junit.Assert.*;\n"
    test += f"import humaneval.correct.{bug_id};\n"
    for ln in _dedupe(imports):
        test += ln + "\n"
    test += (f"\npublic class TEST_{bug_id} {{\n"
             f"    @Test(timeout={TEST_TIMEOUT_MS})\n"
             f"    public void test() {{\n"
             f"{body}\n"
             f"    }}\n"
             f"}}\n")
    return ref, test


# keep the old name around for any external import; not used internally now
def assemble_class(package, class_name, code):
    """Wrap a bare method (or method+imports) into a full Java class source."""
    imports = []
    body = []
    for ln in code.splitlines():
        if ln.strip().startswith("import "):
            imports.append(ln.strip())
        else:
            body.append(ln)
    out = f"package {package};\n\n"
    seen = set()
    for ln in imports:
        if ln not in seen:
            seen.add(ln)
            out += ln + "\n"
    out += f"\npublic class {class_name} {{\n"
    out += "\n".join(body).strip() + "\n"
    out += "}\n"
    return out


# ---------------------------------------------------------------------------
# Gate: compile + run Problem.java with plain asserts
# ---------------------------------------------------------------------------

def gate_candidate(full_source, workdir):
    """Write Problem.java, javac it, run `java -ea Problem`. Classify."""
    prob = Path(workdir) / "Problem.java"
    prob.write_text(full_source)
    rc, out = run(["javac", "-cp", GATE_COMPILE_CP, "Problem.java"],
                  workdir, COMPILE_TIMEOUT_S)
    if rc != 0:
        return "compile_error", f"javac: {out[-1500:]}"
    rc, out = run(["java", "-ea", "-cp", GATE_RUN_CP, "Problem"],
                  workdir, JAVA_RUN_TIMEOUT_S)
    if rc == 0:
        return "pass", out[-500:]
    if rc == 124:
        return "test_fail", "timeout"
    if "AssertionError" in out:
        return "test_fail", out[-1500:]
    return "test_fail", f"rc={rc}: {out[-1500:]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in (SRC_CORRECT, SRC_BUGGY, SRC_TEST, LIB_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for jar in (JAVATUPLES_JAR, JUNIT_JAR, HAMCREST_JAR):
        if not jar.exists():
            raise SystemExit(f"[gate] missing required jar: {jar}")

    meta = load_meta()
    dropped = load_jsonl_set(DROPS_PATH, "bug_id")

    candidates = sorted(GEN_DIR.glob("*.java"))
    kept = sum(1 for f in SRC_CORRECT.glob("*.java"))
    print(f"[gate] {len(candidates)} candidates, {kept} already kept, "
          f"{len(dropped)} dropped")

    n_pass = n_fail = n_skip = 0
    for cand in candidates:
        bug_id = cand.stem
        if bug_id in dropped:
            n_skip += 1
            continue
        if (SRC_CORRECT / f"{bug_id}.java").exists():
            n_skip += 1
            continue

        row = meta.get(bug_id)
        if not row or not (row.get("java_prompt") or "").strip() \
                or not (row.get("java_tests") or "").strip():
            write_jsonl(DROPS_PATH, {"bug_id": bug_id,
                                     "stage": "gate",
                                     "reason": "no_meta"})
            n_fail += 1
            print(f"[gate] DROP {bug_id}: no_meta")
            continue

        java_prompt = row["java_prompt"]
        java_tests = row["java_tests"]
        completion = cand.read_text()

        full_source = assemble_full_source(java_prompt, completion, java_tests)

        with tempfile.TemporaryDirectory(prefix=f"gate_{bug_id}_") as td:
            status, detail = gate_candidate(full_source, Path(td))

        if status == "pass":
            ref, test = make_reference_and_test(java_prompt, completion,
                                                java_tests, bug_id)
            (SRC_CORRECT / f"{bug_id}.java").write_text(ref)
            (SRC_TEST / f"TEST_{bug_id}.java").write_text(test)
            n_pass += 1
            print(f"[gate] KEEP {bug_id}")
        else:
            write_jsonl(DROPS_PATH, {"bug_id": bug_id,
                                     "stage": "gate",
                                     "reason": status,
                                     "detail": detail[-2000:]})
            n_fail += 1
            print(f"[gate] DROP {bug_id}: {status}")

    print(f"[gate] done: kept={n_pass} dropped={n_fail} skipped={n_skip} "
          f"(total kept in tree: {len(list(SRC_CORRECT.glob('*.java')))})")


if __name__ == "__main__":
    main()
