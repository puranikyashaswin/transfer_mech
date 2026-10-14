# Errata

The pre-registration files in this folder are published exactly as frozen. Corrections are recorded here instead of editing them.

**1. Dates of Amendments 005–007 (recorded 14 Oct 2026).** Amendments 005, 006 and 007 carry the date `2026-07-31`. That date is wrong: they amend PREREGISTRATION_v1 (22 Sep 2026) and were written after Amendment 004 (24 Sep 2026) and before Amendment 008 (26 Sep 2026). Their order relative to the runs they govern is unchanged.

**2. Layer-independent arms in the Amendment 008 run (recorded 14 Oct 2026).** Amendment 008 planned to reuse the Amendment 006 results for receiver_k1, text_hint and donor_k1 rather than re-run them. They were re-generated in the layer-17 run (`results/confirmatory_L17/eval_out.jsonl`), and their counts differ slightly: receiver_k1 137→136, text_hint 135→148, donor_k1 162→159 (of 215). The pre-registered headline contrast of that amendment, ridge map vs trained adapter, does not involve these arms. `paper/PAPER.md` §12.8 reports the re-run counts.
