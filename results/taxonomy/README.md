# Compile-error taxonomy

- `*_taxonomy.json`: javac-verified root causes. Every `compile_error` generation was recompiled with javac 11 and bucketed from the exact diagnostic by `eval_harness/taxonomy_recompile.py`. These are the numbers in `TAXONOMY_COMPARISON.md` and `paper/PAPER.md` §12.5.
- `adapter_compile_errors.json` (48) and `control_compile_errors.json` (71 = shuffled 36, random_norm 23, text_hint 7, receiver_k1 5): the failing generations that were recompiled.
- `adapter_taxonomy_counts.json`: an earlier text-pattern pass over the adapter's 48 failures, made before the javac recompile. It is superseded and kept only for completeness; its buckets differ from the javac ones (for example 18 vs 22 extra-close-brace).
