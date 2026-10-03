# Response to Dossier R-FS9-R7 — v4.7

**Re:** Referee Re-Review R-FS9-R7, "The Submission Inside the Record"
(Regenerated-Submission Audit · Improvements Branch · Commit 57D6619)
**Verdict under response:** ACCEPT — CONDITIONAL — 7.4/10; one blocking
clerical condition (B1) and six non-blocking guard-hardening
recommendations.
**Revision:** v4.7, this commit.
**Paper of record after this revision:** `paper/main.pdf`, 55 pp
(recompiled; zero unresolved references), sha256 b86c44c37d286c8c….
**Journal submission manuscript:** regenerated from the record at
`submission/` (source + PDF, 55 pp), sha256 51c83679cfa8586a….

We accept the verdict, the one condition, and all six recommendations
in full. The panel's framing of this round is the one we adopt: the
regenerated submission worked exactly as designed — it faithfully
inherited the record, *including the record's own rendering defect*,
and the defect's escape route was precisely the property the guards
did not check: the rendered output. The condition (B1) is executed
below together with the rendered-bibliography guard the register
prescribes, the six hardening recommendations, the minor items, and a
full-paper cross-verification and proofread commissioned by the owner
for this revision. The integrity battery reads 244/0 post-commit
(243/0 pre-commit — the manifest deferral documented below), with
honest skip accounting: the two torch-gated checks that previously
printed [SKIP] yet counted as passes no longer do.

<!-- LETTER-MANIFEST v1
base: 57d6619
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status 57d6619 <this commit>` — the complete list;
nothing else changed:

    M    ABSTRACT.md
    M    README.md
    M    RUNLOG.md
    M    docs/ENVIRONMENTS.md
    M    docs/response_letters/README.md
    A    docs/response_letters/RESPONSE_R-FS9-R7_v4.7.md
    M    experiments/run_standard_metrics.py
    D    outputs/fig_clients.png
    M    paper/README.md
    M    paper/main.pdf
    M    paper/main.tex
    M    paper/refs.bib
    M    paper/sections/sec_appendix.tex
    M    requirements.txt
    M    submission/README.md
    M    submission/main.pdf
    M    submission/src/main.tex
    M    submission/src/refs.bib
    M    submission/src/sections/sec_appendix.tex
    M    tests/run_battery.py
    M    tests/test_central_criterion.py
    M    tests/test_fl_smoke.py
    M    tests/test_import_graph.py
    M    tests/test_interpretability_artifact.py
    M    tests/test_letters_manifest.py
    M    tests/test_scaffold_algebra.py
    M    tests/test_submission_apparatus.py
    M    tools/build_submission.py
    M    tools/make_fig_clients.py
    M    tools/make_fig_partition.py

Thirty files: twenty-eight modified, one added (this letter), one
deleted (the stale, unreferenced `outputs/fig_clients.png` — B11
residual hygiene, disclosed here and in the RUNLOG row; the referenced
figures are `paper/figures/` and their byte-identical submission
copies). No output artefact, log, figure, or published number other
than that deletion; the sweep digest and the canonical v4.1 battery
log are untouched; block A's regenerated output is byte-identical to
the committed `standard_metrics.json`.

## Register item 1 (B1) — the two truncated bibliography entries

The dossier's git archaeology is confirmed by our own re-derivation,
and we state the provenance plainly: the v4.5 C1/C2 errata placed
their explanatory BibTeX comments *inside* the two entries, directly
after the author fields (`paper/refs.bib:124` for georgoulis2021,
`:252` for hassani2025). Tectonic's bibtex parser treated each comment
as swallowing the remainder of its entry, so [7] and [22] rendered as
bare author lists — no title, venue, year, or DOI — in the v4.5 PDF,
survived the v4.6 record recompile, and were inherited by the
regenerated submission, which was byte-faithful to the record exactly
as designed. The v4.4 PDF (extracted from the git object store) renders
both entries complete, with the then-current C1/C2 defects.

**The repair, exactly as prescribed:** the two comments were moved
*outside* their entries (now above each, with the full provenance
recorded in the comment text, including why they moved); the record
was recompiled; the submission was regenerated. Verified on the
rendered text of both PDFs: [7] now renders the complete FLARECAST
entry (Journal of Space Weather and Space Climate, 11:39, 2021,
doi:10.1051/swsc/2021023) and [22] the complete Hassani entry
(ApJS 279(1):27, 2025, doi:10.3847/1538-4365/addc73, arXiv:2507.05313).
Both documents remain 55 pp, zero unresolved references.

## The rendered-entries check (register item 1, second half)

`tests/test_submission_apparatus.py` gains a seventh check,
`check_rendered_bibliography()`, converting the B7 blind spot into a
guarded property at the same time:

* `pdftotext` extracts the compiled pages of **both** the paper of
  record and the submission (its absence is a loud battery failure,
  not a skip — poppler-utils is a battery dependency since v4.7);
* the **rendered entry count must equal the refs.bib entry count**
  (31), so entries lost or gained at render time fail;
* **every rendered entry must carry a year** — a truncated entry (the
  B1 species: a bare author list) carries none, so the class is caught
  generically, not just at the two known instances;
* the **page counts are pinned** (55/55) — the content-level pin the
  dossier recommends for tectonic's non-byte-reproducible output;
  a reflow now requires a deliberate, disclosed pin bump;
* **eleven load-bearing entries are pinned by content** ([7], [22],
  the six prior-art entries, and the Fu/Leka/Karimireddy anchors):
  each must match by a unique anchor fragment + its title's first
  words + its year, using hyphenation- and ligature-immune matching
  (LaTeX split "Overcoming" across a line in one entry — the matcher
  squashes to alphanumerics before comparison).

**Fault-injection evidence:** run against the defective v4.6 PDF text
(the extracted rendering, preserved for this verification), the check
flags exactly entries [7] and [22] as rendering no year. The guard
would have caught the defect at either of the two compiles it
survived.

## The six guard-hardening recommendations

**1 (B2) — the manifest registry.** A letter whose filename version is
v4.6 or later *without* a LETTER-MANIFEST block now fails the battery
(the panel's suggested filename-version threshold). Previously such a
letter was silently classified as frozen pre-v4.6 archive — the
convention's entry point was author discipline; it is now the test.
Fault-injected: a fake unregistered v4.7 letter fails with exit 1.
The graceful .git-unavailable degradation now counts zero passes (it
previously counted one vacuous pass — see B9 below).

**2 (B3) — crashed-guard detection.** `run_battery.py` now
distinguishes a crashed guard from a declared skip: a module that
prints neither PASS/FAIL/RESULT lines nor an explicit `[SKIP]`
declaration is a battery **failure** (status CRASH, one failed check),
whatever its exit code. Previously any non-zero-exit module with zero
PASS/FAIL lines was reported as a dataset SKIP and the battery exited
0 — the panel's injection of a `RuntimeError` guard walked past
green. `test_fl_smoke.py` now declares its torch-absence skip
explicitly; modules that ran their checks and print internal gated
[SKIP] lines (one sub-check gated on torch or data) report OK with
the gated check noted, not SKIP.

**3 (B4) — import-graph coverage.** `test_import_graph.py` now walks
`provenance/` and `data_manifest/` (nine previously-unwalked tracked
sources), counts plain `import <local module>` statements (38), and —
closing the module-deletion class the panel demonstrated — **pins the
71-module inventory**: deleting any module (e.g. `communication_cost`)
fails loudly with the missing names, instead of its importers being
reclassified external and skipped.

**4 (B5/B6) — the promised checks and the pins.** The top-15-vs-argmax
check that `test_interpretability_artifact.py`'s docstring promised
since v4.6 is implemented: index `i` maps to
`STATS[i//24]_<FEATURE_COLS[i%24]>` and each stored top-15 list must
equal its vector's top-15-by-value **set** — verified against the
frozen artefact, which is internally consistent (as the panel
verified). In block A, `prev_raw` is now pinned with the same
self-check convention as the other denominators (0.013136071152985096,
tol 1e-12), and the LSTM subsection's silent `.get()` defaults are
replaced with required keys plus per-artefact self-checks (all four
`raw_lstm_*.json` files carry the keys, so no output changes):
`block_a()` re-run after the change is **byte-identical** to the
committed `standard_metrics.json`; the artefact is untouched.

**5 (B8) — count hygiene.** The descriptive counts are asserted so
drift fails loudly: 297 local-name imports + 38 plain local imports
(after the scan extension) in the import-graph guard, and 6
`get_criterion` call sites in the criterion guard. The evidence this
works arrived immediately: the import pin caught the +1 from the new
argmax check's function-local `import config` on its **first** battery
run. The v4.6 RUNLOG row's "294" is corrected in place to 295 (what
the v4.6 guard actually printed), and its mangled fragment ("ade" for
"[made") is restored; both corrections are disclosed in the v4.7 row.

**6 (B11) — residual hygiene.** The dead tqdm pin is removed (nothing
imports it); the stale unreferenced `outputs/fig_clients.png` is
deleted (disclosed above); `paper/README.md`'s dangling
`scripts/fig_analysis.py` references are replaced with the real tools
(`visualize_results.py`, `tools/make_fig_clients.py`,
`tools/make_fig_partition.py`); both figure tools' font registrations
are guarded (matplotlib bundles DejaVu — a missing system path no
longer crashes the tools).

## Minor items (B9, B12, B13)

* **B9** is discharged structurally: the RESULT-counting in the two
  torch-gated modules and in the battery's RESULT override now counts
  skips as skips. The torch-less count therefore *dropped by two*
  while two real checks joined (the argmax check and the
  rendered-bibliography guard) — the honest accounting is itself
  visible in the per-file counts of the RUNLOG row.
* **B12:** the RUNLOG "ade" fragment and the "294" are fixed in place
  (disclosed); `ABSTRACT.md`'s header no longer carries a version
  marker at all (the N2 lesson — version-bearing strings live in
  `BATTERY_VERSION` and the appendix history table only, so there is
  no third place to go stale); the "(v4.0)" preamble comments are
  removed from the record and inherited out of the generated
  submission; the undefined "R6-4" in `submission/README.md` is
  corrected to R8-6; the README's "267 with torch" projection is
  replaced with the honest statement (244 torch-less; the
  dataset/torch-gated checks additionally run when their dependencies
  are present — no untested projection number).
* **B13:** the generator's NOTE-FOR-AUTHORS comment now states the
  workflow instead of inviting hand-editing: the file is generated and
  byte-compared; affiliations are set by editing the title block in
  `tools/build_submission.py` and re-running the generator.
* **B10** (noted, minor): the no-op if/pass block in
  `test_scaffold_algebra.py` that read as a warm-start-caller scan is
  removed rather than left to imply coverage (the 3a/3b call-site
  checks are the real contract), and the queue-log behavioural check
  now reports a declared, uncounted skip when the tracked log is
  absent instead of degrading to a counted pass.

## Claim language (the dossier's Section 6)

We accept the panel's judgement on the letter's rhetoric and adopt its
formulation: the guards verify the properties their authors thought to
verify, and the honest statement is what each guard *checks*, not what
a species "cannot" do. Accordingly, this letter states each guard's
mechanism and its fault-injection evidence, and claims exactly:
rendered truncation of any bibliography entry now fails the battery
(both PDFs, every entry, eleven pinned anchors); an unregistered
v4.6+ letter fails the battery; a crashed or silent guard fails the
battery; a deleted module fails the battery; the interpretability
lists equal the vectors' top-15 sets; block A's denominators are
pinned and self-checked; the descriptive counts are asserted. We do
not claim the machinery ends any species — we claim these specific
properties are checked, in every environment the battery runs in.

## The owner-directed cross-verification and proofread

Beyond the register, the owner commissioned a full-paper
cross-verification and proofread of the v4.7 build. Method: extract
the rendered paper text, tokenise every numeric claim (534 distinct
tokens), and trace each to its owning artefact or to arithmetic
therefrom. Result: **every performance number matches a committed
artefact or verifiable arithmetic from committed values** — including
recomputations performed fresh for this pass: the SolarMLP's 29,377
and the SolarLSTM's 219,265 parameters recomputed from `model.py`'s
architecture; the train-side label agreement (99.71%, 161
disagreements) recomputed from the committed
`provenance/train_meta_slim.csv.gz` over its 56,181 verified rows;
77,865 = 65,406+12,459 (the fold sizes); 476.9M = 331,185×24×60; every
Table 5/7 median lead equal to the event files' stored minutes ÷ 60;
every false-alarm rate from the event files; and the full
communication-cost paragraph as arithmetic from `results.json`'s byte
counts (117.5 KB/direction, 0.235 MB round-trip, 1.41 MB/round,
70.5 MB aggregate, 47.3 MB vs ≈526 MB vs ≈130 MB and the 1.5×/2.4×
ratios). Eleven residual tokens were adjudicated individually: DOI
fragments cited in prose, 24×6 index-range endpoints, one "476.9M"
tokenisation artifact, and counts quoted from frozen appendix-history
rows — none is a performance claim. The proofread found: the 31-entry
bibliography complete (now guarded), zero duplicated sentences, zero
undefined references, zero common typos, one benign math-layout
double-space (a rendering artifact inside a display equation), and
three cosmetic overfull boxes. The verification script and its
adjudicated output are preserved on the owner side; the RUNLOG v4.7
row records the pass.

## What did not change

No number, claim, figure, or experimental code path moved: the paper's
only content changes are the two bibliography entries rendering their
own fields, the appendix v4.7 history row, and the caption range; the
experimental record is untouched (sweep digest and canonical logs
byte-identical; `outputs/` diff confined to the disclosed stale-PNG
deletion; block A regenerated byte-identical; the submission
regeneration remains byte-identical to a fresh generation). The
battery remains 16 modules; six grew checks, none lost any.

## Verification summary

* Battery at v4.7 (torch-less, environment row 3): **243/0
  pre-commit → 244/0 post-commit** — the one-check deferral is this
  letter's own manifest check, which waits for the revision's commit
  to exist (the same deferral the v4.6 letter documented at 242→243);
  per-file 75/20/29/34/17/5/1/3/1/5/7/2/skip/13/18/13, log at the
  gitignored versioned path by design.
* Submission regeneration: byte-identical (20 source files, both
  comparison directions); apparatus test 7/7 including the new
  rendered-bibliography check on both PDFs.
* Fault-injections: the rendered-bibliography check flags the
  defective v4.6 text at [7]+[22]; an unregistered v4.7 letter fails
  the manifest registry; the import-count pin caught a real +1 on its
  first run.
* Fresh-clone first-run contract: verified (exit 0, clean tree, log
  at the gitignored path).
