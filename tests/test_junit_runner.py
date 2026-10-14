"""Unit tests for the pure Java-source helpers in the execution harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_harness"))
import junit_runner as jr  # noqa: E402

BUGGY_CLASS = """package humaneval.buggy;

import java.util.*;

public class ADD {
    public static int helper(int x) {
        return x;
    }

    public static int add(int a, int b) {
        if (a > 0) {
            return a - b;
        }
        return a - b;
    }
}
"""

BUGGY_METHOD = """    public static int add(int a, int b) {
        if (a > 0) {
            return a - b;
        }
        return a - b;
    }"""


def test_extract_class_name():
    assert jr.extract_class_name(BUGGY_CLASS) == "ADD"
    assert jr.extract_class_name("public final class X {}") == "X"
    assert jr.extract_class_name("class Hidden {}") is None


def test_method_names_finds_public_static_methods():
    assert jr.method_names(BUGGY_CLASS) == ["helper", "add"]


def test_find_method_span_matches_nested_braces():
    s, e = jr.find_method_span(BUGGY_CLASS, "add")
    body = BUGGY_CLASS[s:e]
    assert body.startswith("public static int add")
    assert body.count("{") == body.count("}") == 2
    assert jr.find_method_span(BUGGY_CLASS, "missing") is None


def test_wrap_candidate_replaces_only_the_buggy_method():
    cand = "public static int add(int a, int b) { return a + b; }"
    out = jr.wrap_candidate_method(BUGGY_CLASS, BUGGY_METHOD, cand)
    assert "return a + b;" in out
    assert "return a - b;" not in out
    assert "public static int helper(int x)" in out  # untouched
    assert out.startswith("package humaneval.buggy;")


def test_wrap_candidate_returns_none_when_method_absent():
    assert jr.wrap_candidate_method(BUGGY_CLASS, "int nothing;", "x") is None


def test_set_package_replaces_or_inserts():
    assert jr.set_package(BUGGY_CLASS, "scratch").startswith("package scratch;")
    assert jr.set_package("public class A {}", "p").startswith("package p;\n\npublic class A")
