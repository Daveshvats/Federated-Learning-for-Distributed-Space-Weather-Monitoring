# Response to the external re-review — the v4.14 compute follow-up
(item 2 executed; both compute items of the re-review are now closed)

**Reviewer**: the independent external re-review of the v4.11 state
(five findings, answered at v4.12; its item-3 compute follow-up
reported at v4.13). This revision reports the execution of the
re-review's second and last compute item and the integration of its
verdict. No finding is re-opened; no claim is weakened; one guess
this project's own run card made is corrected on the record below.
**Manuscript**: the v4.14 build of *Federated Solar-Flare Prediction
on SWAN-SF: A Benchmark Audit of Provenance, Leakage, and Evaluation
Protocols* (full edition 68 pp; trimmed submission edition 13 pp).

## Item 2 — the validation story: **EXECUTED; the divergence
disappears — the random validation carve was region-leaky**

The owner executed the committed kit on 2026-10-06
(`python experiments/run_raw_nobn.py --region-disjoint
--raw-dir raw`, both arms, 3,119 s, RTX 4060; the parse-metadata
prerequisite had been shipped in-repo at v4.13.2, so the run was
zero-setup: pull, then run). The artefact of record is the owner
push itself — commit 17af7e4,
`outputs/raw_nobn_region_disjoint.json` (742 lines, the Windows
CRLF form, 25,722 bytes, sha256-frozen by the battery) — the first
full-fidelity owner-side push of the cycle. The carve held out 390
of 2,447 whole NOAA active regions (40,959 validation rows at
1.49% prevalence against the random carve's 2.05%; train and test
untouched; disjointness asserted by the runner; the region map
sourced from the parse metadata `ar` column — the v4.12.1 errata
fix exercised for real).

**The verdict, against the run card's decision rules:**

- **The divergence disappears.** FedAvg's validation ROC-AUC now
  *declines* (0.980 → 0.974, rounds 5–50) in parallel with the test
  decline (0.975 → 0.960) — where the frozen random carve's monitor
  *rose* (0.974 → 0.977, sharpening to 0.978 → 0.989 at seed 43)
  while the test fell. **Mechanism (a), region-sharing between the
  random validation carve and the training shards, is confirmed;
  mechanism (b), partition-5 temporal recency, is rejected** as the
  monitor's rise. The diagnosis now closes exactly as the Section
  6.4 apparatus framed it.
- **The shipped checkpoint is no longer the weakest test round of
  its own trajectory** (round 35 sits 0.0014 above the round-50
  minimum; the random carve's shipped round 50 *was* the minimum),
  and the validation-ROC sensitivity rule now selects round 5 —
  simultaneously the best test round of the entire trajectory
  (0.975/0.450) — where under the random carve it selected round
  40, mid-decay. The honest monitor points at the early checkpoint.
- **The shipped numbers did not improve — and the paper says so.**
  Both selections move −0.001 ROC-AUC (FedAvg 0.9626 → 0.9617,
  FedProx 0.9755 → 0.9743). The v4.12 run card guessed the numbers
  would "likely improve" under the honest monitor; they did not,
  because the leak inflated the monitor's *trend*, not the
  validation-based selections (the thresholded validation-F1 band
  was flat in both carves). This is pinned in the battery as a
  non-improvement so no text can claim otherwise.
- **What survives is disclosed, not buried**: the test decline is
  carve-invariant (0.974 → 0.960 here, 0.974 → 0.963 random) — real
  client drift, so the conservative-lower-bound reading stands; a
  stable ~0.014 validation–test ROC level offset remains, now a
  genuine held-out-region difficulty difference rather than
  leakage; the divergence persists in PR space (validation PR-AUC
  drifting up 0.675 → 0.715 against test 0.450 → 0.357), the
  expected behaviour of a prevalence- and population-sensitive
  statistic — the reason ROC-AUC is the diagnostic axis throughout.
- **FedProx**: no divergence in either carve (validation and test
  both climb), and its region-disjoint monitor recovers to
  validation F1 0.574 against the random carve's 0.092 *at the same
  seed and protocol* — the fifth independent confirmation that the
  fixed-threshold monitor is the fragile layer (its thresholded
  values travel with the validation population; the ranking moves
  0.001).

**Integration (v4.14)**: the Section 6.4 mechanisms paragraph is
resolved in place and the verdict plus residuals paragraphs added
(after Table `tab:rawnobnrep`); the re-run programme's list moves
the region-disjoint run from queued to executed (fifth owner-GPU
batch) and its queue is shortened accordingly; the conclusion's
parity clause gains the closure and its executed-items list goes
from eleven to twelve; the appendix carries the v4.14 version row
(and its caption range, stale since v4.12, is re-synced); the
trimmed edition's divergence paragraph is resolved in place and its
limitations caveat discharged into the resolved audit clause —
zero process language maintained. Every integrated number is
battery-pinned to the artefact (new module
`tests/test_region_disjoint_verdict.py`, 80 checks: the
CRLF/LF-stable sha256 byte-pin, the decision rules re-derived
against the frozen seed-42 comparator, the non-improvement pinned,
the residuals pinned, and the paper text pinned at every
integration site). Page pins 67/12 → 68/13 per the B7 convention —
the full edition reflowed by the verdict paragraphs, the trimmed
edition by five printed lines of resolution text its 12-page build
did not have (twice compressed to hold the budget; the references
tail, not content, spilled).

## Item 3 — unchanged since v4.13/v4.13.1

Parity holds with replication; the seed-43 JSON is landed and
sha256-frozen. Nothing owner-side.

## Verification

Full battery re-run at v4.14: **578/0 torch-less pre-commit, 579/0
post-commit** (this letter's own manifest check activating at
commit, the documented deferral convention) (the new
`tests/test_region_disjoint_verdict.py` contributes 79 checks; the
seed-43 module's run-card pin re-pointed to the card's v4.14 state;
the letters registry carries this letter; page pins 68/13, both
editions recompiled and disclosed). **No owner action remains: both
compute items of the external re-review are executed, integrated,
and frozen.** The re-run programme's remaining queue is unchanged
(vanilla SCAFFOLD re-run; fold-replication breadth).

<!-- LETTER-MANIFEST v1
base: 17af7e420dcb92bf53d01c0f035681328d44b2dc
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status 17af7e4 <this commit>` — the complete list;
nothing else changed:

    M    README.md
    M    RUNLOG.md
    M    docs/ENVIRONMENTS.md
    M    docs/RUN_CARD_v4.12.md
    A    docs/response_letters/RESPONSE_External_v4.14.md
    M    paper/README.md
    M    paper/main.pdf
    M    paper/sections/sec_appendix.tex
    M    paper/sections/sec_conclusion.tex
    M    paper/sections/sec_limitations.tex
    M    paper/sections/sec_results.tex
    M    paper_trimmed/sections/sec_discussion.tex
    M    paper_trimmed/sections/sec_results.tex
    M    submission/README.md
    M    submission/main.pdf
    M    submission/src/sections/sec_discussion.tex
    M    submission/src/sections/sec_results.tex
    M    tests/run_battery.py
    M    tests/test_import_graph.py
    M    tests/test_letters_manifest.py
    A    tests/test_region_disjoint_verdict.py
    M    tests/test_seed43_replication.py
    M    tests/test_submission_apparatus.py
