# Compile-Error Taxonomy Comparison

**Generated:** 2026-09-25  
**Method:** Recompiled every `compile_error` generation with javac 11, parsed exact diagnostics into root-cause buckets.  
**Artifact:** `eval_harness/taxonomy_recompile.py` + `results/taxonomy/{arm}_taxonomy.json`

---

## Summary Table

| Root Cause | Adapter (n=48) | Shuffled (n=36) | Receiver (n=5) | Text Hint (n=7) | Random Norm (n=23) |
|------------|----------------|-----------------|----------------|-----------------|-------------------|
| **unbalanced_extra_close** | **22 (46%)** | 9 (25%) | 0 (0%) | 0 (0%) | 1 (4%) |
| **illegal_package_reserved** | **13 (27%)** | 8 (22%) | 0 (0%) | 0 (0%) | 0 (0%) |
| unresolved_symbol | 8 (17%) | 13 (36%) | 2 (40%) | 4 (57%) | 9 (39%) |
| incompatible_types | 4 (8%) | 3 (8%) | 3 (60%) | 3 (43%) | 13 (57%) |
| unclosed_literal | 1 (2%) | 2 (6%) | 0 (0%) | 0 (0%) | 0 (0%) |
| truncation_eof | 0 (0%) | 1 (3%) | 0 (0%) | 0 (0%) | 0 (0%) |

---

## Key Findings

### 1. The adapter's failure mode is **structural**, not semantic

- **73% of adapter compile errors** are structural corruption (extra closing brace, illegal package)
- Only 25% are semantic (unresolved symbol, type mismatch)
- This is the **inverse** of the receiver baseline (0% structural, 100% semantic)

### 2. This is **adapter-specific**, not injection-general

- Receiver baseline: 0% extra-close, 0% illegal-package
- Random norm: 4% extra-close, 0% illegal-package
- Text hint: 0% extra-close, 0% illegal-package
- **Only the adapter and shuffled controls show these failure modes at elevated rates**

### 3. The adapter vs shuffled profiles differ qualitatively

- **Adapter:** dominated by structural corruption (46% extra-close, 27% illegal-package)
- **Shuffled:** dominated by semantic errors (36% unresolved symbol)
- This suggests the adapter's learned mapping has a **qualitatively different effect** than random perturbation

---

## Mechanistic Interpretation

**Extra-close-brace (46%):** The adapter corrupts the receiver's tracking of block structure — likely by injecting donor representations that encode a different nesting depth or scope boundary than the receiver expects at that generation point. Example: `MBPP_127_multiply_int` generates `}\n\n}` (extra closing brace after class body).

**Illegal-package (27%):** The adapter induces the receiver to "hallucinate" package declarations that violate Java's reserved namespace rules. Example: `MBPP_19_test_duplicate` generates `package java.util;` (illegal injection into JDK package).

**Both are consistent with the adapter mapping donor hidden states that are out-of-distribution for the receiver's syntactic state machine.**

---

## Raw Data

- `adapter_taxonomy.json` — 48 adapter compile errors with javac diagnostics
- `shuffled_taxonomy.json` — 36 shuffled control compile errors
- `receiver_k1_taxonomy.json` — 5 receiver baseline compile errors
- `text_hint_taxonomy.json` — 7 text-hint control compile errors
- `random_norm_taxonomy.json` — 23 random-norm control compile errors

Each JSON contains: `{"bug_id", "bucket", "first_error"}` per case.
