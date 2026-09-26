#!/usr/bin/env python3
"""
fill_confirmatory.py — populate [[CONFIRMATORY_RESULTS_PLACEHOLDER]] in PAPER.md
and main.tex from the confirmatory eval output, using make_stats_confirmatory.py
as the single source of truth. Run ONLY after the confirmatory eval completes.

Usage:
  ~/.venvs/doctools/bin/python paper/fill_confirmatory.py

It:
  1. Runs make_stats_confirmatory.py, captures its printed report.
  2. Builds the §12.4 results text from the report.
  3. Substitutes the placeholder in PAPER.md (## 12.4) and main.tex.
  4. Refuses to run if the eval_out.jsonl is absent or incomplete (<215 complete bugs).
"""
import json, re, subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # transfer_mech/
EVAL = os.path.join(ROOT, "results", "confirmatory", "eval_out.jsonl")
PY = os.path.expanduser("~/.venvs/doctools/bin/python")
STATS = os.path.join(ROOT, "paper", "make_stats_confirmatory.py")
PAPER_MD = os.path.join(ROOT, "paper", "PAPER.md")
MAIN_TEX = os.path.join(ROOT, "paper", "main.tex")
PLACEHOLDER = "[[CONFIRMATORY_RESULTS_PLACEHOLDER]]"

def completeness():
    if not os.path.exists(EVAL):
        return 0, "eval_out.jsonl absent"
    rows = [json.loads(l) for l in open(EVAL)]
    single = ["receiver_k1","adapter","shuffled","self_inject","unrelated_task","text_hint","donor_k1"]
    by = {}
    for r in rows:
        by.setdefault(r["bug_id"], {}).setdefault(r["arm"], 0)
        by[r["bug_id"]][r["arm"]] += 1
    complete = 0
    for b, arms in by.items():
        ok = all(arms.get(a,0) >= 1 for a in single) and arms.get("random_norm",0) >= 5
        if ok: complete += 1
    return complete, f"{complete} complete bugs"

def run_stats():
    out = subprocess.run([PY, STATS], capture_output=True, text=True)
    return out.stdout + ("\n" + out.stderr if out.stderr.strip() else "")

def build_results_text(report):
    # Keep the report verbatim inside a fenced block for the markdown paper.
    md = []
    md.append("The one-shot confirmatory run is complete. All numbers below are recomputed by `paper/make_stats_confirmatory.py` directly from `results/confirmatory/eval_out.jsonl` (execution-verified). Per the Amendment 006 §5 pre-commitment, achieved n is below the pilot-powered n, so **no headline effect is claimed**: the paired effect and its confidence interval are reported as a boundary result.\n")
    md.append("```")
    md.append(report.rstrip())
    md.append("```")
    return "\n".join(md)

def main():
    n, msg = completeness()
    print(f"[fill_confirmatory] completeness: {msg}")
    if n < 215:
        print(f"[fill_confirmatory] NOT READY: need 215 complete bugs, have {n}. Aborting without changes.")
        sys.exit(2)
    report = run_stats()
    if "NOT RUN YET" in report:
        print("[fill_confirmatory] stats analyzer reports NOT RUN YET. Aborting.")
        sys.exit(3)
    text = build_results_text(report)

    changed = []
    # PAPER.md
    md = open(PAPER_MD).read()
    if PLACEHOLDER in md:
        open(PAPER_MD, "w").write(md.replace(PLACEHOLDER, text))
        changed.append("PAPER.md")
    # main.tex — placeholder sits inside a paragraph; wrap report in a verbatim block.
    # NOTE: main.tex uses the LaTeX-escaped form [[CONFIRMATORY\_RESULTS\_PLACEHOLDER]].
    TEX_PLACEHOLDER = "[[CONFIRMATORY\\_RESULTS\\_PLACEHOLDER]]"
    tex = open(MAIN_TEX).read()
    if TEX_PLACEHOLDER in tex:
        tex_block = ("The one-shot confirmatory run is complete; all numbers are recomputed by "
                     "\\texttt{make\\_stats\\_confirmatory.py} from \\texttt{eval\\_out.jsonl}. "
                     "Per the Amendment 006 honest-n pre-commitment, achieved $n$ is below the "
                     "pilot-powered $n$, so no headline effect is claimed: the paired effect and its "
                     "CI are reported as a boundary result.\n\n\\begin{verbatim}\n" + report.rstrip() + "\n\\end{verbatim}")
        open(MAIN_TEX, "w").write(tex.replace(TEX_PLACEHOLDER, tex_block))
        changed.append("main.tex")
    if changed:
        print(f"[fill_confirmatory] placeholder filled in: {', '.join(changed)}")
    else:
        print("[fill_confirmatory] no placeholder found (already filled?)")
    print("\n--- REPORT ---\n" + report)

if __name__ == "__main__":
    main()
