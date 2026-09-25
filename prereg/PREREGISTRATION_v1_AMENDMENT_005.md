# Amendment 005 — Token Budget for Code Completion on the Confirmatory Corpus

**Date: 2026-07-31. Appended to frozen PREREGISTRATION_v1.md + Amendments 001–004.**

## Why this amendment exists
The confirmatory eval harness was found (during pilot validation, before any test-split results were inspected) to produce **compile_error for every arm** on the new MultiPL-E-Java corpus. Diagnosis: the frozen `MAX_NEW_TOKENS=512` (Amendment 001) truncates the receiver's generations mid-method on this corpus (the receiver emits a short prose preamble plus a code block; 512 tokens is insufficient for the code to close). Verified: at 512 tokens the extracted code has unbalanced braces and fails to compile; at 1024 tokens the code completes and compiles. Continuing at 512 tokens would make the confirmatory run garbage (all arms compile_error), not a test of the transfer hypothesis.

## Change
- **Generation budget for the confirmatory paired eval: `MAX_NEW_TOKENS = 1024`** (up from 512), for ALL arms equally (receiver baseline, adapter, random-norm, shuffled, self-inject, donor, receiver-sampling oracle). Identical budget across arms preserves the paired comparison.
- The extraction helper (`extract_java_code`) is patched to handle a code block with no closing fence (take the code from the opening fence to the end), a robustness fix for truncated/unterminated blocks. This is a harness bugfix, not a method change.

## What is unchanged (still frozen)
- The model pair (A1), donor layer 14 / receiver layer 12, final-token additive injection, α=1.0, temp 0.2, top-p 0.95, seed 42.
- The train/dev/test split over the confirmatory corpus; the adapter trained on the train split only (B2).
- The decision thresholds (Amendment 001 §A5): paired win-share ≥60% of discordants AND ≥3pp vs the strongest control; "if we cannot reach n, we do not claim the effect — report the CI and stop."
- The exploratory/confirmatory firewall (A4): this amendment was written BEFORE inspecting any test-split result; the change is a harness/engineering fix (token budget + extraction), not a response to any observed outcome.

## Honest-n note (Amendment 003 reaffirmed)
The achieved corpus is 385 correct + 301 buggy + 385 tests (MultiPL-E-Java ceiling after Gate-0). The confirmatory test set is the 221 held-out problems (idx 80–300 of the paired manifest). Per A5, we report achieved n + CIs; we do not claim the effect if under-powered.
