#!/usr/bin/env python3
"""
mutate.py -- Stage 3 of the confirmatory_hej corpus build.

For each KEPT correct file src/main/java/humaneval/correct/<BUG>.java, generate
ONE buggy mutant by applying ONE deterministic mutation (operator flip,
off-by-one, literal corruption, method swap like isUpperCase/isLowerCase,
sort->reverse-order, sign flip, off-by-k, comparison-chain corruption, negate
boolean return, return-constant). The mutant must COMPILE and FAIL >= 1 test
(verified with javac + JUnit against the existing TEST_<BUG>.java).

Operators are tried in a fixed priority order; the first one yielding
compile+test_fail wins and is written to src/main/java/humaneval/buggy/.
If none works, the problem is dropped (logged to logs/drops.jsonl).

Resumable: skips bug_ids that already have a buggy file or are already
dropped. Writes NOTHING to /opt/dlami/nvme.

Run: python3 mutate.py
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
OUT_ROOT = HOME / "transfer_mech" / "data" / "confirmatory_hej"
SRC_CORRECT = OUT_ROOT / "src" / "main" / "java" / "humaneval" / "correct"
SRC_BUGGY = OUT_ROOT / "src" / "main" / "java" / "humaneval" / "buggy"
SRC_TEST = OUT_ROOT / "src" / "test" / "java" / "humaneval"
LIB_DIR = OUT_ROOT / "lib"
LOGS_DIR = OUT_ROOT / "logs"
DROPS_PATH = LOGS_DIR / "drops.jsonl"

JUNIT_JAR = LIB_DIR / "junit4-4.12.jar"
HAMCREST_JAR = LIB_DIR / "hamcrest-all-1.3.jar"
JAVATUPLES_JAR = LIB_DIR / "javatuples-1.2.jar"
JUNIT_CP = f"{JUNIT_JAR}:{HAMCREST_JAR}:{JAVATUPLES_JAR}"
RUN_CP = f".:{JUNIT_CP}"

JAVA_RUN_TIMEOUT_S = 60
COMPILE_TIMEOUT_S = 60

# ---------------------------------------------------------------------------
# Helpers (shared verbatim with gate.py)
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


def javac_cmd(class_name, workdir):
    return ["javac", "-cp", JUNIT_CP, "-d", str(workdir),
            str(workdir / f"{class_name}.java")]


def javac_test(class_name, workdir):
    # The test references the subject class `<BUG_ID>` (compiled to workdir in the
    # prior step). The classpath MUST include the workdir (".") or javac cannot
    # resolve the subject class -> "cannot find symbol". JUNIT_CP is the jars only.
    return ["javac", "-cp", f".:{JUNIT_CP}", "-d", str(workdir),
            str(workdir / f"{class_name}.java")]


def java_junit_cmd(test_class, workdir):
    return ["java", "-cp", RUN_CP, "org.junit.runner.JUnitCore", test_class]


def run(cmd, cwd, timeout):
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                           capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"


def compile_and_run(bug_id, subject_src, test_src, workdir):
    (Path(workdir) / f"{bug_id}.java").write_text(subject_src)
    (Path(workdir) / f"TEST_{bug_id}.java").write_text(test_src)
    rc, out = run(javac_cmd(bug_id, workdir), workdir, COMPILE_TIMEOUT_S)
    if rc != 0:
        return "compile_error", f"subject: {out[-1500:]}"
    rc, out = run(javac_test(f"TEST_{bug_id}", workdir), workdir,
                  COMPILE_TIMEOUT_S)
    if rc != 0:
        return "compile_error", f"test: {out[-1500:]}"
    rc, out = run(java_junit_cmd(f"TEST_{bug_id}", workdir), workdir,
                  JAVA_RUN_TIMEOUT_S)
    if rc == 0 and re.search(r"\bOK \(", out):
        return "pass", out[-500:]
    if rc == 124:
        return "test_fail", "timeout"
    if "FAILURES!!!" in out or "Tests run:" in out:
        return "test_fail", out[-1500:]
    return "test_fail", f"rc={rc}: {out[-1500:]}"


# ---------------------------------------------------------------------------
# Mutation operators. Each takes the correct file text and returns a list of
# candidate mutant strings (may be empty). Deterministic: fixed orders,
# earliest-match-first.
# ---------------------------------------------------------------------------

JAVA_KEYWORDS = {
    "if", "for", "while", "return", "new", "switch", "case", "do", "else",
    "throw", "assert", "synchronized", "instanceof", "catch", "try",
}

STRONG_OPS = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "!=",
              "!=": "=="}


def _mask_strings(code):
    """Return code with string/char literal contents replaced by spaces of
    equal length so regex edits never touch literal contents."""
    def blank(m):
        return m.group(0)[0] + " " * (len(m.group(0)) - 2) + m.group(0)[-1]
    return re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', blank, code)


def flip_relational(code):
    """Flip one strong comparison operator (first occurrence, in masked code).
    Priority order fixed by STRONG_OPS dict insertion order."""
    masked = _mask_strings(code)
    for op in STRONG_OPS:
        idx = masked.find(op)
        while idx != -1:
            # ensure longest-match: '<=' handled before '<' since dict order
            cand = code[:idx] + STRONG_OPS[op] + code[idx + len(op):]
            return [cand]  # exactly one candidate per operator class
        # try next operator
    return []


def flip_all_relational(code):
    """Backup: flip each strong operator occurrence (one mutant per position),
    earliest first, until we have up to 3 candidates."""
    masked = _mask_strings(code)
    out = []
    pattern = sorted(STRONG_OPS, key=len, reverse=True)
    for m in re.finditer("|".join(re.escape(o) for o in pattern), masked):
        op = m.group(0)
        cand = code[:m.start()] + STRONG_OPS[op] + code[m.end():]
        out.append(cand)
        if len(out) >= 3:
            break
    return out


def off_by_one(code):
    """i < length -> i <= length (or length - 1 -> length) style off-by-one."""
    masked = _mask_strings(code)
    # 1) `< X.length` / `< X.size()` -> `<= ...`
    m = re.search(r"<\s*([\w.\[\]()]+)(\.length|\.size\(\))", masked)
    if m:
        cand = code[:m.start()] + "<=" + code[m.start() + 1:]
        return [cand]
    # 2) `i < n` (single identifier) -> `i <= n`
    m = re.search(r"(\b[a-zA-Z_]\w*\b)\s*<\s*(\b[a-zA-Z_]\w*\b)", masked)
    if m and m.group(1) not in JAVA_KEYWORDS and m.group(2) not in JAVA_KEYWORDS:
        cand = code[:m.start()] + f"{m.group(1)} <= {m.group(2)}" + code[m.end():]
        return [cand]
    # 3) numeric literal N -> N+1 (first integer literal not 0)
    m = re.search(r"(?<![\w.])([1-9]\d*)(?![\w.])", masked)
    if m:
        n = int(m.group(1))
        cand = code[:m.start()] + str(n + 1) + code[m.end():]
        return [cand]
    return []


def off_by_k(code):
    """N -> N+2 for the first integer literal >= 1 (stronger corruption)."""
    masked = _mask_strings(code)
    m = re.search(r"(?<![\w.])([1-9]\d*)(?![\w.])", masked)
    if m:
        n = int(m.group(1))
        return [code[:m.start()] + str(n + 2) + code[m.end():]]
    return []


def literal_corruption(code):
    """First integer literal L -> L+1 (0 -> 1)."""
    masked = _mask_strings(code)
    m = re.search(r"(?<![\w.])(\d+)(?![\w.])", masked)
    if m:
        n = int(m.group(1))
        return [code[:m.start()] + str(n + 1) + code[m.end():]]
    return []


def sign_flip(code):
    """`a - b` -> `a + b` or `a + b` -> `a - b` (first arithmetic occurrence)."""
    masked = _mask_strings(code)
    m = re.search(r"(\b[a-zA-Z_][\w.\[\]]*\b)\s*-\s*(\b[a-zA-Z_][\w.\[\]]*\b)",
                  masked)
    if m and m.group(1) not in JAVA_KEYWORDS:
        return [code[:m.start()] + f"{m.group(1)} + {m.group(2)}" + code[m.end():]]
    m = re.search(r"(\b[a-zA-Z_][\w.\[\]]*\b)\s*\+\s*(\b[a-zA-Z_][\w.\[\]]*\b)",
                  masked)
    if m and m.group(1) not in JAVA_KEYWORDS:
        return [code[:m.start()] + f"{m.group(1)} - {m.group(2)}" + code[m.end():]]
    return []


def method_swap(code):
    """Swap paired JDK methods: isUpperCase<->isLowerCase, Math.max<->Math.min,
    sort -> reverse-order sort."""
    swaps = [
        ("isUpperCase", "isLowerCase"),
        ("Character.isUpperCase", "Character.isLowerCase"),
        ("Math.max", "Math.min"),
        ("Collections.sort", "Collections.reverseOrder_sort_placeholder"),
    ]
    masked = _mask_strings(code)
    # plain name swaps
    for a, b in (("isUpperCase", "isLowerCase"),
                 ("Math.max", "Math.min")):
        idx = masked.find(a)
        if idx != -1:
            return [code[:idx] + b + code[idx + len(a):]]
    # sort -> reverse order: Collections.sort(x) -> x.sort(Collections.reverseOrder())
    m = re.search(r"Collections\.sort\s*\(\s*([\w.\[\]()]+?)\s*\)", masked)
    if m:
        arg = m.group(1)
        repl = f"{arg}.sort(Collections.reverseOrder())"
        return [code[:m.start()] + repl + code[m.end():]]
    # Arrays.sort(x) -> Arrays.sort(x, Collections.reverseOrder())
    m = re.search(r"Arrays\.sort\s*\(\s*([\w.\[\]()]+?)\s*\)", masked)
    if m:
        arg = m.group(1)
        repl = f"Arrays.sort({arg}, Collections.reverseOrder())"
        return [code[:m.start()] + repl + code[m.end():]]
    return []


def comparison_chain_corruption(code):
    """a < b && b < c  ->  a < c && b < c style: swap the two sides of the
    first relational expression's right operand with the next one's left."""
    masked = _mask_strings(code)
    m = re.search(
        r"(\b[a-zA-Z_]\w*\b)\s*(<|>|<=|>=)\s*(\b[a-zA-Z_]\w*\b)"
        r"\s*&&\s*"
        r"(\b[a-zA-Z_]\w*\b)\s*(<|>|<=|>=)\s*(\b[a-zA-Z_]\w*\b)",
        masked,
    )
    if m:
        a, op1, b, c, op2, d = m.groups()
        repl = f"{a} {op1} {c} && {b} {op2} {d}"
        return [code[:m.start()] + repl + code[m.end():]]
    return []


def negate_boolean_return(code):
    """`return expr;` inside a boolean method -> `return !(expr);`.
    We detect boolean methods by signature."""
    masked = _mask_strings(code)
    if not re.search(r"\bboolean\s+\w+\s*\(", masked):
        return []
    m = re.search(r"\breturn\s+([^;{}]+);", masked)
    if m:
        expr = code[m.start(1):m.end(1)]
        cand = code[:m.start()] + f"return !({expr});" + code[m.end():]
        return [cand]
    return []


def return_constant(code):
    """Last-resort: replace `return expr;` with a type-appropriate constant.
    Only used if nothing else works; almost surely fails >=1 test."""
    masked = _mask_strings(code)
    ret_type = None
    m = re.search(r"\b(public|private|protected)?\s*(static\s+)?"
                  r"(int|long|double|float|boolean|String|char|void|[A-Z][\w<>\[\]]*)"
                  r"\s+\w+\s*\(", masked)
    if m:
        ret_type = m.group(3)
    m = re.search(r"\breturn\s+[^;{}]+;", masked)
    if not m or ret_type in (None, "void"):
        return []
    const = {
        "int": "0", "long": "0L", "double": "0.0", "float": "0.0f",
        "boolean": "false", "char": "'a'", "String": '""',
    }.get(ret_type, "null")
    return [code[:m.start()] + f"return {const};" + code[m.end():]]


OPERATORS = [
    ("flip_relational", flip_relational),
    ("method_swap", method_swap),
    ("off_by_one", off_by_one),
    ("literal_corruption", literal_corruption),
    ("sign_flip", sign_flip),
    ("off_by_k", off_by_k),
    ("flip_all_relational", flip_all_relational),
    ("comparison_chain_corruption", comparison_chain_corruption),
    ("negate_boolean_return", negate_boolean_return),
    ("return_constant", return_constant),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in (SRC_BUGGY, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    dropped = load_jsonl_set(DROPS_PATH, "bug_id")
    correct_files = sorted(SRC_CORRECT.glob("*.java"))
    print(f"[mutate] {len(correct_files)} kept problems")

    n_mut = n_fail = n_skip = 0
    for cf in correct_files:
        bug_id = cf.stem
        if bug_id in dropped:
            n_skip += 1
            continue
        if (SRC_BUGGY / f"{bug_id}.java").exists():
            n_skip += 1
            continue
        test_path = SRC_TEST / f"TEST_{bug_id}.java"
        if not test_path.exists():
            write_jsonl(DROPS_PATH, {"bug_id": bug_id, "stage": "mutate",
                                     "reason": "missing_test_file"})
            n_fail += 1
            continue

        correct_src = cf.read_text()
        # The mutant goes to the DEFAULT package (strip `package humaneval.correct;`)
        # so it compiles in a flat temp dir. The TEST_ file was rewritten by
        # fix_tests.py to the DEFAULT package with class-qualified method calls
        # (`<BUG>.method(...)`), so it is used as-is and resolves the mutant class.
        buggy_base = re.sub(r"^\s*package\s+humaneval\.correct;\s*\n", "",
                            correct_src, count=1, flags=re.M)
        test_src = test_path.read_text()

        chosen = None
        chosen_op = None
        for op_name, op_fn in OPERATORS:
            try:
                cands = op_fn(buggy_base)
            except Exception as e:
                cands = []
                print(f"[mutate] op {op_name} raised on {bug_id}: {e}")
            for cand in cands:
                if cand.strip() == buggy_base.strip():
                    continue
                with tempfile.TemporaryDirectory(prefix=f"mut_{bug_id}_") as td:
                    status, detail = compile_and_run(bug_id, cand, test_src,
                                                     Path(td))
                if status == "test_fail":
                    chosen = cand
                    chosen_op = op_name
                    break
                # compile_error or pass -> keep trying
            if chosen is not None:
                break

        if chosen is None:
            write_jsonl(DROPS_PATH, {"bug_id": bug_id, "stage": "mutate",
                                     "reason": "no_operator_yielded_test_fail"})
            n_fail += 1
            print(f"[mutate] DROP {bug_id}: no operator worked")
        else:
            (SRC_BUGGY / f"{bug_id}.java").write_text(chosen)
            n_mut += 1
            print(f"[mutate] MUT {bug_id} via {chosen_op}")

    print(f"[mutate] done: mutants={n_mut} dropped={n_fail} skipped={n_skip} "
          f"(total buggy in tree: {len(list(SRC_BUGGY.glob('*.java')))})")


if __name__ == "__main__":
    main()
