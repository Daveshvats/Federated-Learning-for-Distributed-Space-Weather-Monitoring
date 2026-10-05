# Record of Revision v4.8 — Post-Closure Content Errata

**Re:** Dossier R-FS9-R8, "The Rendered Page, Guarded" (cycle-closure
audit of v4.7 — verdict **ACCEPT — SUSTAINED — 7.9/10, R-FS9 CYCLE
CLOSED**, residual register C1–C6 + T1/T2 non-blocking, "fold in at
the next natural touch"), **and** the owner-side content review's
three conditions (E1–E3), cross-verified against the repository at
`9d1c193` before this revision was drafted.
**Revision:** v4.8, this commit. No re-review is pending or sought —
this is the v4.4-precedent form: a post-closure errata batch,
executed and recorded.
**Paper of record after this revision:** `paper/main.pdf`, 56 pp
(one page of disclosed reflow: the E1 disclosure paragraph, the E2
qualifiers, and the appendix v4.8 row), zero unresolved references.
**Journal submission manuscript:** regenerated from the record
(`submission/`, source + PDF, 56 pp).

We accept the R8 verdict and its residual register in full, and we
execute the owner-side content review's conditions in the order the
adjudication set: E1 first (the raw-substrate BN diagnostic), then
E2 (the SCAFFOLD qualifiers), E3 (the abstract reorder), with the R8
register folded in alongside. One erratum is owed to the record's
honesty standard and is paid below (C4).

<!-- LETTER-MANIFEST v1
base: 9d1c193
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status 9d1c193 <this commit>` — the complete list;
nothing else changed:

    M    ABSTRACT.md
    M    RUNLOG.md
    A    docs/response_letters/RESPONSE_v4.8_ERRATA.md
    A    experiments/run_raw_bn_diagnostic.py
    M    paper/main.pdf
    M    paper/main.tex
    M    paper/sections/sec_appendix.tex
    M    paper/sections/sec_limitations.tex
    M    paper/sections/sec_results.tex
    M    submission/README.md
    M    submission/main.pdf
    M    submission/src/main.tex
    M    submission/src/sections/sec_appendix.tex
    M    submission/src/sections/sec_limitations.tex
    M    submission/src/sections/sec_results.tex
    M    tests/run_battery.py
    M    tests/test_central_criterion.py
    M    tests/test_import_graph.py
    M    tests/test_letters_manifest.py
    A    tests/test_raw_bn_diagnostic.py
    M    tests/test_submission_apparatus.py
    M    tools/build_submission.py

Twenty-two files: nineteen modified, three added (this letter, the
E1 runner, the E1 guard). No output artefact, log, figure, or
published number changed; the sweep digest and the canonical v4.1
battery log are untouched. The two PDFs differ by the disclosed
recompile (56 pp, one page of reflow) and the submission by the
regeneration that inherits it.

## E1 — the raw-substrate BN-transport diagnostic (the substantive item)

The owner-side content review's point 1, confirmed at the code level
before this revision: the raw-substrate federated MLP arms (FedAvg
0.875, FedProx 0.930, SCAFFOLD 0.768 ROC-AUC) were evaluated through
the same untransported-BatchNorm evaluation path that the paper's own
in-partition diagnostic (Section 6.9, `bn_diagnostic.json`) shows
manufactures apparent federated collapse — `get_weights`/`set_weights`
transport model *parameters* only, so the round-resumable runner's
returned checkpoints carry never-updated running buffers into a
`model.eval()` evaluation. The review's condition is executed in
three parts:

1. **The disclosure (paper, this revision).** The abstract now carries
   the qualifier inline ("federated MLPs degrade — a statement made
   under the same untransported-BN evaluation path the diagnostic
   above indicts, pending the raw-substrate BN diagnostic"); Section 6
   gains a dedicated paragraph after the substrate-fragile block
   stating the raw-substrate MLP degradation is
   *BN-evaluation-conditional* until the checkpoint-level diagnostic
   lands; the limitations section carries the matching qualifier. The
   sequence arms are unaffected (SolarLSTM carries no BatchNorm), so
   the encoder-contrast reading stands either way.
2. **The runner (shipped, execution queued owner-side — RUNLOG ask
   #4).** `experiments/run_raw_bn_diagnostic.py` re-evaluates the
   identical weights of the three raw-substrate MLP checkpoints under
   three BN treatments: **A** init buffers (the published path — and
   a *reproduction gate*: arm-A ROC/PR-AUC must reproduce the
   committed `raw_substrate_eval.json` within 1e-4, else a MISMATCH
   verdict and a non-zero exit, because a silent mismatch would
   poison the same-weights counterfactual framing); **B** pooled
   per-client exact input statistics (law of total variance, TRAINING
   shards only — no leakage); **C** eval-time batch statistics. Each
   checkpoint's protocol fingerprint (algorithm/mu/seed/rounds/agg/
   smote/focal/loss/use_lstm) is validated before evaluation. Arm D
   (the transported round buffers an implementation would actually
   average) is **not reconstructible from the checkpoints** — only
   parameters were ever aggregated — and is documented in the runner
   as queued owner compute (the in-partition diagnostic's phase-C
   method). An event-level layer runs per treatment when the raw
   audit metadata is present and declares its skip otherwise. CPU,
   minutes, no retraining: `python experiments/run_raw_bn_diagnostic.py`.
3. **The guard (battery, this revision).** `tests/test_raw_bn_diagnostic.py`
   — 14 torch-free static contract pins in every environment (the
   arms and treatments pinned, the reproduction gate's markers, the
   fingerprint check, the arm-D disclosure, the pre-hook exact
   statistics, the train-shards-only recalibration, the
   get_model_probs-identical NaN/clip semantics, the declared-skip
   event layer) plus an artefact layer that activates once
   `outputs/raw_bn_diagnostic.json` exists (schema; every arm ×
   treatment's ROC/PR in [0,1]; no MISMATCH gate verdict; the gate's
   reported numbers equal the gated results; the published reference
   equals the committed artefact; event consistency). Until the owner
   runs E1 the artefact layer is a declared, uncounted [SKIP] — the
   B9 convention. Fault-injected both ways with synthetic artefacts:
   a valid one passes, a MISMATCH one fails and exits 1.

The E1 numbers will be integrated into the paper at the next natural
revision once ask #4 runs.

## E2 — the SCAFFOLD implementation-case qualifiers

The five-departure enumeration (sign inversion −1.0000, cold start,
and the rest) has been in `sec_method` since v4.6, and the intro
contribution-5 and conclusion 55/65 mentions already carried the
implementation-case qualifier. This revision extends it to the three
remaining mentions that lacked it: the SCAFFOLD-LSTM event-level
passage in Section 6 ("measuring, like every SCAFFOLD cell, the
sign-inverted, cold-start variant of the five-departure list …, not
the reference algorithm"), the appendix GPU-queue row ("both cells
measuring the disclosed five-departure implementation, not the
reference algorithm"), and the limitations sentence ("both cells
measuring the sign-inverted, cold-start implementation case …, not
the reference algorithm"). Every 55/65 mention in the paper now
carries the qualifier. The sign-corrected owner-GPU re-run remains
queued future work, per the R6 round's scoping.

## E3 — the abstract reorder

The corrected persistence finding now leads the abstract's results
claims: "The deployability verdict, however, is set by the corrected
persistence baselines: no arm beats same-region label inertia (TSS
0.967) at this 24-hour label geometry; …" immediately follows the
leakage-free-fold sentence, and the trailing verification-metrics
sentence no longer duplicates it. `ABSTRACT.md` is resynced and the
submission inherits the reordered abstract through the generator.
The honest framing (same-region label inertia, the 24-hour-lagged
contrast, the Section 6.1 question) is unchanged from the v4.1-era
correction — this is the emphasis reorder the review asked for, not
a retraction.

## The R8 residual register (C1–C6, T1–T2) — folded in

* **C1 (+T1's dead disjunct) — battery crash semantics.** A non-zero
  exit is now a battery failure whatever the module's verdict lines
  say. The two escapes the panel demonstrated are closed: a module
  that prints one `[PASS]` and then dies was previously `[ ERR]`
  with *zero failures counted* and the battery green; a module that
  declared `[SKIP]` and then crashed was previously `SKIP` with the
  traceback suppressed. Both are now `[CRASH]` +1 failure with the
  output tail shown; a module that already reported failures through
  its verdict lines is counted through those lines (status `ERR`, no
  double count); the v4.7 B3 silent-module rule is retained; the dead
  second crash disjunct is removed. Verified with a six-case
  fault-injection harness (PASS-then-die, SKIP-then-die, healthy
  skip, FAIL+exit-1, healthy OK, silent-exit-0).
* **C2 — the letter registry.** Classification is an explicit pinned
  map (`LETTER_REGISTRY`, seven entries with manifest-required
  flags), not the filename-version regex — an unversioned, dotless,
  or below-threshold filename was silently classified as frozen
  legacy and passed. A `.md` file not in the registry now FAILS
  however its filename reads, and a registry entry whose file is
  missing fails (deletion is loud). Fault-injected: a rogue
  unversioned letter and a deleted registered letter both fail,
  exit 1. This letter is itself the first v4.8-convention entry.
* **C3 (+T1's len assertion) — the full bibliography pin.** All 31
  entries of `refs.bib` are pinned in
  `tests/test_submission_apparatus.py` by anchor + title fragment +
  year (20 new pins joining the v4.7 eleven; every candidate was
  validated against the *rendered* text of both PDFs before
  hardcoding — one ligature trap avoided: mcmahan2017's "Efficient"
  contains the `ﬀ` ligature glyph, which the squashed matcher strips,
  so that pin anchors on "learning of deep networks from
  decentralized" instead). The pin count is asserted against the
  refs.bib entry count, so an unpinned entry — including one
  truncated *after* its year field, the panel's exact gap — fails
  the battery. Fault-injected: dropping one pin fails with "30 pins
  but refs.bib has 31 entries."
* **C4 — the erratum owed to the record.** The v4.7 letter claimed
  the v4.6 RUNLOG row's mangled "ade" fragment had been "restored".
  No restoration happened: file-level inspection of the `57d6619`
  blob shows the row already read correctly, with matching brackets;
  the "ade" was a display-pipeline artefact of the assistant-side
  output channel, reproduced in this environment (the round-9
  referee's own retraction, which we verified first-hand, was
  correct). This line is the erratum; the v4.7 letter is otherwise
  frozen per the archive convention.
* **C5 — scan coverage.** `tests/test_central_criterion.py` now
  walks `provenance/` and `data_manifest/` (the import graph's scan
  had covered both since v4.7; this guard's had not). The pinned
  call-site count is verified unchanged at 6.
* **C6 — the external whitelist.** `tests/test_import_graph.py` now
  enforces a pinned `EXTERNAL_MODULES` whitelist (26 stdlib + 10
  third-party names, exactly the repository's actual external
  imports): an import whose head is neither a local module, a local
  module's *stem* (the sys.path-inserted script style), nor
  whitelisted is an error. The v4.6-disclosed design limitation — "a
  plain import of a nonexistent local module is treated as
  external and passes" — is closed. Fault-injected: a probe importing
  a nonexistent module fails, exit 1. The inventory pin is 73 modules
  (the two new files), and the import counts move 297→309 / 38→40 —
  the deliberate bumps: +10 from the new runner's local imports,
  +3 bare-stem imports newly resolved through the stem map, +1
  `import config`.
* **T1 (second half) — PASS-line reconciliation.**
  `check_rendered_bibliography` prints one consolidated `[PASS]`
  line for both PDFs instead of one per PDF, so the module's
  `[PASS]`-line count agrees with its RESULT line.
* **T2 — the display erratum.** The SCAFFOLD-MLP event-table median
  lead cell now reads 23.25 h — the stored median is exactly
  1,395 min = 23.25 h; banker's rounding had displayed 23.2 — with
  the table caption documenting the convention.

## Verification

* Battery (torch-less, environment row 3): **258/0 pre-commit →
  259/0 post-commit** (244 + 14 new static guard checks; the +1 is
  this letter's manifest verifying against the commit's actual git
  diff — the v4.6/v4.7 deferral convention). Per-module:
  75/20/29/34/17/5/1/3/1/5/7/2/skip/13/18/13/14.
* Fault-injections, all first-hand: the six-case battery crash
  harness (C1), the rogue/deleted letter (C2), the dropped pin
  (C3), the unresolved import (C6), the raw-BN synthetic artefacts
  both valid and MISMATCH (E1's guard).
* Paper of record recompiled (tectonic): 56 pp, 0 unresolved
  references; the B7 page pins bumped 55→56/56 deliberately for the
  disclosed reflow. Submission regenerated with v4.8 markers; the
  regeneration is byte-identical to the committed `submission/src`
  (the apparatus check).
* Not changed: `outputs/` (the raw-BN artefact is owner-side until
  ask #4 runs), the sweep digest, the canonical v4.1 battery log,
  every historical letter.

## Deliberately not in this revision

The 12–15 page benchmark-audit split remains a separate owner
decision (it restructures the paper, it is not an errata item). The
sign-corrected SCAFFOLD owner-GPU re-run and arm D of the raw
diagnostic (the faithful re-run with per-round buffer capture)
remain queued owner compute. The 55-page/process language (the
review's point 4) is retained as the honest record of how the work
was executed.
