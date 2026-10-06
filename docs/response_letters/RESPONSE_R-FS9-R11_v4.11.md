# Record of Revision v4.11 — The Pre-Submission Errata Touch
(P1–P8: One Stamp Single-Sourced, Three Pins, and a Wording Sweep)

**Re:** Dossier R-FS9-R11, the Round-14 register-integration
verification (verdict **ACCEPT, SUSTAINED — CONDITION DISCHARGED —
8.4/10 — PRE-SUBMISSION REGISTER ISSUED**), Section 8's programme:
"Nothing mandatory for acceptance… Before any external submission,
one errata touch: the P1 string… with regeneration; the P3/P4/P6
wording fixes; the P5 companion cells at three sites" and "In the
same touch, two SHA pins… and one Sec. 6.4 content pin… The optional
source-comparison guard (P7) rides along for free." This revision is
that touch, executed in full — all eight punch-list items, plus
three same-class sweep finds this side's verification added.

**Revision:** v4.11, this commit. No re-review is pending or sought
(the panel's disposition: the blocking lane is empty; these are
one-touch errata before an external submission).

**Paper of record after this revision:** `paper/main.pdf`, **63 pp**
(was 62 at v4.10; the disclosed reflow — the event-level table's new
FedAvg arm-C companion row, the three companioned "zero of 65"
sites, the sign-normalised gap list, and the appendix v4.11 row; the
page pins are bumped 62/62→63/63 per the B7 convention). The journal
submission is regenerated from the record (`submission/`,
`tools/build_submission.py`; the battery verifies the regeneration is
byte-identical in text).

**No scientific claim, number, figure, or experiment changed.** The
only number-level edits are the two the panel itself priced: the
P3 rounding (−0.005→−0.004, the exact value −0.00447) and the P5
companion cells (numbers read straight off the committed
`raw_bn_diagnostic.json` event layer). The C-block remains owner
decisions, untouched; the B2 SCAFFOLD vanilla re-run remains the one
optional GPU queue entry.

<!-- LETTER-MANIFEST v1
base: 104e6d476887bec4f6c21171d960546dd105f59c
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status 104e6d4 <this commit>` — the complete list;
nothing else changed:

    M    ABSTRACT.md
    M    README.md
    M    RUNLOG.md
    M    docs/ENVIRONMENTS.md
    A    docs/response_letters/RESPONSE_R-FS9-R11_v4.11.md
    M    paper/README.md
    M    paper/main.pdf
    M    paper/main.tex
    M    paper/sections/sec_appendix.tex
    M    paper/sections/sec_intro.tex
    M    paper/sections/sec_results.tex
    M    submission/README.md
    M    submission/main.pdf
    M    submission/src/main.tex
    M    submission/src/sections/sec_appendix.tex
    M    submission/src/sections/sec_intro.tex
    M    submission/src/sections/sec_results.tex
    M    tests/run_battery.py
    M    tests/test_letters_manifest.py
    M    tests/test_raw_bn_diagnostic.py
    M    tests/test_submission_apparatus.py
    M    tools/build_submission.py

(21 modified + this letter = 22 files.)

## P1 — the stale stamp, fixed at the root (MODERATE)

The finding: `tools/build_submission.py` hardcoded "v4.9" in the
title-page stamp and header comment (its lines 204/167), propagated
to `submission/src/main.tex:116` and `:5`, contradicting the
README's "current build: v4.10" — the generator was touched at v4.10
for the T2 re-pin but the stamp was not bumped.

The fix goes one level deeper than the one string the panel priced,
per the register's own standing lesson (N2, the v4.2 mistitled log):
the stamp is now **single-sourced**. `build_submission.py` reads
`BATTERY_VERSION` from `tests/run_battery.py` at generation time
(the repository's version register, bumped with every release) and
refuses to emit an unversioned stamp if the constant is unreadable.
The title page and header comment now read **v4.11** — the current
build — and the stamp cannot trail a release again, because it is no
longer maintained by hand at all. The submission was regenerated;
the apparatus test's byte-compare of the regenerated source against
the committed `submission/src` passes.

## P2 — the two decisive artefacts, pinned (MODERATE)

The finding: an in-range tamper of `raw_nobn_eval.json` (FedAvg ROC
0.9626→0.99) passed the full 373/0 battery at v4.10, and no test
pinned any Section 6.4 / Table 9 paper number — the two newest, most
load-bearing artefacts were the least guarded objects in the
repository.

The fix, in the battery's artefact layer (layer 5 of
`tests/test_raw_bn_diagnostic.py`, 12 new checks):

1. **Two sha256 byte pins** — `outputs/raw_nobn_eval.json`
   (96334bc6ac8922ec…) and `outputs/arm_b_central_sanity.json`
   (ad63c3b0712e1ced…) — the same convention as every other frozen
   record since v4.9 (the interpretability-sha rule).
2. **One content pin for the Table 9 region** — the subsection
   anchor to the table's end is located in
   `paper/sections/sec_results.tex`, and all seven data rows are
   checked against values derived **live** from the (now sha-pinned)
   artefact: the two no-BN selected checkpoints plus the five
   reference columns, each rounded exactly as the paper rounds. Never
   hand-typed — the reference-column standard. The parameter
   arithmetic is pinned with them (28,929 from the artefact's
   protocol block; 29,377−448=28,929). The submission's copy needs no
   separate pin: it is a verbatim record copy, byte-verified by the
   apparatus test's regeneration check.

**Fault-injected, both ways, before this commit** (each mutation
reverted; the module re-run green after each revert):

* the panel's exact v4.10-passing mutation — `raw_nobn_eval.json`
  FedAvg ROC 0.9625921602580426→0.99, in range — now FAILS twice
  (the sha pin and, through it, the content pin's derived row:
  expected `…& 0.963 & 0.366`, the tampered tree says 0.990);
* the panel's coordinated two-sided edit — a Table 9 cell changed
  identically in `paper/sections/sec_results.tex` and
  `submission/src/sections/sec_results.tex` (which passed 373/0 at
  v4.10) — now FAILS the content pin (the artefact still says 0.963;
  the coordinated tree says 0.970).

## P7 — the arm-B machinery's byte identity, enforced (MINOR)

The finding: `pool_client_stats` / `set_bn_stats` / `bn_modules` are
byte-identical source copies of the diagnostic runner's — disclosed
in the runner's header, true since v4.9.2, but enforced by nothing.

The fix: a source-comparison guard in the same layer — both runners
are ast-parsed and the three function segments must match exactly.
Fault-injected: a one-line divergence in
`run_arm_b_central_sanity.py`'s copy fails the guard ("diverged or
missing: pool_client_stats"); the reverted tree passes. This closes
the panel's "true but unenforced" class for the copy-paste
disclosure.

## P3 — the mis-rounded residual (MINOR)

The finding: "−0.005 residual" at two sites (the panel's Section 6.4
and "6.13"; in the paper's own numbering both live in 6.4 and the
gap list of 6.7, "What survives the leakage-free protocol" — the
dossier's section label, not the text, is what is off by six); the
exact value −0.00447 rounds to −0.004, and −0.005 flatters the
past-zero margin by 0.0005.

The fix: `$-0.005$` → `$-0.004$` at both sites — 6.4's "gap closes
past zero ($-0.004$ residual…)" and the 6.7 gap list's no-BN cell.
Two characters, as priced.

## P4 — the "both" antecedent (MINOR)

The finding: 6.3's "0.96864/0.43051 against arm A's own-buffer
0.96870/0.42566, both reproducing the retrain record to better than
10⁻⁶" — true only of arm A's two metrics; arm B is 6.8×10⁻⁵ /
4.9×10⁻³ away, which the same sentence reports as its finding.

The fix: the clause now reads "…against arm A's own-buffer
0.96870/0.42566, **arm A reproducing** the retrain record to better
than 10⁻⁶" — the antecedent names arm A alone, per the panel's
second option. One word.

## P5 — the bare "zero of 65" sites, companioned (MINOR)

The finding: the bare phrase survives at three sites — intro
contribution 5, the event-level table's shipped row (Sec. 6.2), and
6.5's event-level pass — while the B1 companion landed only in the
conclusion, 6.7, and Section 9.

The fix, at all three sites, with the companion cell read straight
off the committed `raw_bn_diagnostic.json` event layer
(`fedavg:C_batch_stats`):

* **Intro contribution 5**: "(FedAvg detects zero of 65 events —
  **38 (58.5%) under the diagnostic's batch-statistics treatment,
  the only evaluation with a usable validation operating point**,
  Section 6.3; SCAFFOLD 0.768/0.057)…"
* **The event-level table (Sec. 6.2)**: a new row directly under the
  shipped zero-detection row — "FedAvg MLP (arm C, batch
  statistics) & 38/65 (58.5%) & 0.28 & 0.35 & 19.8" — with the
  caption naming it "the companion cell to its shipped
  zero-detection row." (38 detected, 58.46% rate, 0.28 FA
  windows/day, 0.354 alerts/day, 1,190-min median lead = 19.8 h —
  all from the artefact.)
* **6.5**: "Where plain FedAvg-MLP detects zero of the 65 events —
  **38 (58.5%) under the diagnostic's batch-statistics treatment,
  the only evaluation with a usable validation operating point
  (Section 6.3)** — FedAvg-LSTM detects 35 (53.8%)…"

## P6 — the abstract's two elisions (MINOR)

The finding: "reconstructs the model's own statistics" drops the
body's "should" (the reconstruction is two to eleven percent off,
disclosed in the body), and "ranking" is not metric-scoped in the
sentence that also prints the PR-AUC values (the no-BN PR ordering
inverts between the arms).

The fix, both single-word class: "where it **should** reconstruct
the model's own statistics and **does preserve the ROC-AUC
ranking**" (the parallel verb keeps the sentence honest on both
metrics — ROC preserved, PR improved), and "restores the federated
MLP to the sequence arms' **ROC-AUC** ranking". The intro's
same-species elision ("at or above the sequence arms' ranking") is
scoped with it, and `ABSTRACT.md` is synced byte-faithfully to the
record abstract.

## P8 — the sweep (TRIVIAL/errata class)

* **ENVIRONMENTS row 3** now carries the v4.10 373/0 count (the one
  the panel's injection matrix ran against) and the v4.11 apparatus
  state.
* **The 6.5 title** re-framed: "The sequence encoder rescues the
  federated arms" → "**The BatchNorm-free sequence arms rescue the
  federated arms**" — the legacy encoder attribution the v4.10
  rewrite withdrew everywhere else. The two sibling phrases ("the
  encoder rescue survives…", "the sequence-encoder rescue is not an
  artefact of one seed") are re-attributed with it.
* **The FedAvg trajectory**: "declines monotonically" → "declines
  from 0.974 (round 5) to 0.963 (round 50) — **every round except a
  +9×10⁻⁶ uptick at the final step**" (r49→r50: 0.962583→0.962592,
  read off the artefact's per-round trajectory). FedProx's "climbs
  monotonically" is verified strictly monotone against the same
  trajectory and is unchanged.
* **The 6.7 gap list's sign convention unified**: the four cells are
  now all signed federated-minus-reference with each cell's
  centralised reference named (FedProx +0.003 vs the centralised MLP
  on the cleaned fold; −0.048 vs the best centralised arm, logistic
  regression, on raw; −0.008 for FedProx-LSTM below the same
  reference; +0.004 past zero vs the centralised MLP for the no-BN
  FedProx). Every magnitude is unchanged; the no-BN cell carries the
  P3 rounding with it.
* **Three same-class sweep finds this side's verification added**
  (found first-hand while re-deriving the dossier, fixed here,
  disclosed in the RUNLOG): the root README's battery counts had
  drifted (375/297/297 against the real 373 at HEAD — now the real
  386); `paper/README.md`'s status block was stuck at v4.7; and the
  appendix evolution-table caption read v2.1–v4.9, one version
  behind its own v4.10 row (now v2.1–v4.11).

## What the panel's Section 8 asked, item by item

1. "Nothing mandatory for acceptance" — agreed; this touch changes
   no claim.
2. "One errata touch: the P1 string… with regeneration; the
   P3/P4/P6 wording fixes; the P5 companion cells at three sites" —
   executed above (P1 at the root, not just the string).
3. "In the same touch, two SHA pins… and one Sec. 6.4 content pin…
   The optional source-comparison guard (P7) rides along for free" —
   executed, with both fault-injections live-verified before commit.
4. "Owner decisions unchanged" — unchanged: B2 SCAFFOLD vanilla
   re-run stays the one optional GPU queue entry; C1 (process purge)
   and C2 (the 15-page carve-out) remain the strategic calls, both
   documented, neither blocking.
5. "What remains is packaging discipline" — the register's
   blocking lane stays empty; the pre-submission register is
   discharged.

## Verification record (this revision's own run)

* `tests/test_raw_bn_diagnostic.py`: 123 → **135** (the 12 new
  layer-5 checks; every layer green).
* Full battery, torch-less environment (row 3): **385/0 pre-commit**
  (this letter's own manifest check defers until its commit exists —
  the documented v4.6/v4.7/v4.8 convention), **386/0 post-commit**.
* Import-graph pins unchanged (layer 5 imports only stdlib `ast` /
  `hashlib`; EXPECTED_FROM_IMPORTS stays 336, plain 43).
* Paper compiled with tectonic: 63 pp, zero unresolved references;
  submission regenerated with the v4.11 stamp; both PDFs'
  rendered-bibliography guard and page pins (63/63) green.
* Fault injections (all reverted, tree clean, module re-run green
  after each): the in-range artefact tamper fails twice; the
  coordinated two-sided Table 9 edit fails the content pin; the
  source divergence fails the P7 guard.
* The canonical v4.1 battery log and the sweep digest are untouched.

The cycle's test is complete in both directions, as the panel
recorded; the paper now says exactly what its artefacts can prove,
and every artefact that says it is pinned to the byte that says it.
