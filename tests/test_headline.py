"""Regression suite: the paper's headline numbers must reproduce from the raw logs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import verify_headline as vh  # noqa: E402


@pytest.fixture(scope="module")
def got():
    return vh.headline()


@pytest.mark.parametrize("name", list(vh.EXPECTED))
def test_headline_number_matches_paper(got, name):
    want, nd = vh.EXPECTED[name]
    assert round(got[name], nd) == round(want, nd)


def test_mcnemar_exact_known_values():
    assert vh.mcnemar_exact(0, 0) == 1.0
    assert vh.mcnemar_exact(5, 5) == 1.0
    # 13 vs 1 discordants: the dev-split ridge-vs-adapter contrast (PAPER.md §12.7, p=0.0018)
    assert round(vh.mcnemar_exact(13, 1), 4) == 0.0018


def test_clopper_pearson_matches_scipy_reference():
    # scipy.stats.beta.ppf(0.025, 18, 45), beta.ppf(0.975, 19, 44)
    lo, hi = vh.clopper_pearson(18, 62)
    assert lo == pytest.approx(0.181994, abs=1e-6)
    assert hi == pytest.approx(0.419490, abs=1e-6)
    assert vh.clopper_pearson(0, 10)[0] == 0.0
    assert vh.clopper_pearson(10, 10)[1] == 1.0


def test_only_pass_status_counts_as_success():
    rows = [{"bug_id": "B", "arm": a, "status": "pass"} for a in vh.SINGLE_ARMS]
    rows[1]["status"] = "compile_error"  # adapter
    rows += [{"bug_id": "B", "arm": "random_norm", "status": "pass", "draw": d} for d in range(5)]
    bugs, passes, _ = vh.confirmatory_outcomes(rows)
    assert bugs == ["B"]
    assert passes["receiver_k1"]["B"] and not passes["adapter"]["B"]


def test_incomplete_bug_is_dropped():
    rows = [{"bug_id": "B", "arm": a, "status": "pass"} for a in vh.SINGLE_ARMS]
    rows += [{"bug_id": "B", "arm": "random_norm", "status": "pass", "draw": d} for d in range(4)]
    bugs, _, _ = vh.confirmatory_outcomes(rows)
    assert bugs == []
