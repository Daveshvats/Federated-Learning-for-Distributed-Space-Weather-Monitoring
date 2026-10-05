# Record of Revision v4.10 — The Register's Paper-Side Integration
(A1's Verdict, A3's Attribution Answer, and the A4/B1–B5 Text)

**Re:** Dossier R-FS9-R10, master change register items A4 and B1–B5
(the Round-13 third-party adjudication, verdict AGREE ~90 PERCENT —
ACCEPT, SUSTAINED — 8.0/10 — E1′ REGISTER ISSUED), executed now that
the register's compute programme is discharged: asks #1–#11 are all
RAN, and the two decisive experiments landed their artefacts at
v4.9.4 (A1: `arm_b_validated`; A3: the
BatchNorm-under-federation attribution answer). This revision is the
ONE paper revision the register sequenced everything toward — the
paper of record changes for the first time since v4.9, and it changes
on the evidence, not ahead of it.
**Revision:** v4.10, this commit. No re-review is pending or sought.
**Paper of record after this revision:** `paper/main.pdf`, **62 pp**
(was 59 at v4.9–v4.9.4; the disclosed recompile reflows three pages —
the new no-BatchNorm attribution-control subsection + Table 9, the
arm-B validation paragraph, the re-attributed readings, and the
appendix v4.10 row). The journal submission is regenerated from the
record (`submission/`, `tools/build_submission.py`, battery-verified
byte-identical regeneration).

Every number below is read off the committed artefacts — none
hand-typed, none re-run. The register items C1–C3 (process-language
purge, the 15-page core-audit carve-out, the delete-the-claim
fallback) remain **owner decisions** and are deliberately NOT
executed here; the C3 fallback is moot (A1–A3 landed in the
current-reading world, which the panel priced at 8.4–8.6 with the
scoped abstract).

<!-- LETTER-MANIFEST v1
base: 83c3848129e0d90d9e19c3b458917205ab61cde7
-->

## Diff manifest (machine-checked by tests/test_letters_manifest.py)

`git diff --name-status 83c3848 <this commit>` — the complete list;
nothing else changed:

    M    ABSTRACT.md
    M    README.md
    M    RUNLOG.md
    M    docs/ENVIRONMENTS.md
    A    docs/response_letters/RESPONSE_R-FS9-R10_v4.10.md
    M    paper/main.pdf
    M    paper/main.tex
    M    paper/sections/sec_appendix.tex
    M    paper/sections/sec_conclusion.tex
    M    paper/sections/sec_discussion.tex
    M    paper/sections/sec_intro.tex
    M    paper/sections/sec_limitations.tex
    M    paper/sections/sec_results.tex
    M    submission/README.md
    M    submission/main.pdf
    M    submission/src/main.tex
    M    submission/src/sections/sec_appendix.tex
    M    submission/src/sections/sec_conclusion.tex
    M    submission/src/sections/sec_discussion.tex
    M    submission/src/sections/sec_intro.tex
    M    submission/src/sections/sec_limitations.tex
    M    submission/src/sections/sec_results.tex
    M    tests/run_battery.py
    M    tests/test_letters_manifest.py
    M    tests/test_raw_bn_diagnostic.py
    M    tests/test_submission_apparatus.py
    M    tools/build_submission.py

## 1. A1 integrated — Table 8's B column is the validated proxy
(register item A2, re-frame branch)

Section 6.3's reading paragraph is extended by the arm-B validation
sentence: applied to the **centralised** raw MLP — where pooled
per-client statistics should reconstruct the model's own running
statistics — the byte-identical pooled-recalibration machinery
preserves the centralised ranking to −6.8×10⁻⁵ ROC-AUC and
*improves* PR-AUC by +4.9×10⁻³ (0.96864/0.43051 against arm A's
0.96870/0.42566, both reproducing the retrain record to better than
10⁻⁶), with the pooled-versus-own buffer deltas (0.021/0.043/0.113
mean relative L2, depth-monotone) recorded in the artefact. The
"pooled recalibration is actively destructive here" sentence STANDS
— now as a validated finding rather than a suspected implementation
defect — and the B column is re-framed as exactly what a
pooled-statistics transport fix would deliver on this substrate.
The arm-D faithful re-run remains the endorsed clean-closure option
for an external submission but is **not required for the reading**
(the register's own fork, resolved by A1's verdict). The abstract's
transport sentence carries the same scoped claim.

## 2. A3 integrated — the attribution inverts (contribution 5
rewritten)

New **Section 6.4** ("No-BatchNorm federation on the raw substrate:
the attribution control") + **Table 9**, reading
`outputs/raw_nobn_eval.json` exactly:

- The single intervention (three `BatchNorm1d` layers → identity,
  28,929 vs 29,377 parameters, the in-partition construction
  verbatim) restores **FedAvg to 0.963/0.366** (+0.088 ROC, +0.310
  PR, ×6.5) and **FedProx to 0.975/0.345** (+0.045, +0.196, ×2.3)
  against their published BN arms.
- Gap arithmetic disclosed: FedAvg's ROC gap to the centralised MLP
  closes 0.096→0.008 (91%) and its PR gap 0.389→0.079 (80%);
  FedProx's ROC gap closes past zero (−0.005) and its PR gap
  0.296→0.100 (66%). Both no-BN arms sit at-or-above their LSTM
  counterparts on ROC-AUC; no-BN FedProx exceeds the **centralised**
  MLP on ROC-AUC (0.975 vs 0.971).
- **Contribution 5 is rewritten**: its title becomes
  "BatchNorm-under-federation-conditional collapse" and the
  "encoder-conditional" attribution is withdrawn; the
  "flattened-feature MLP" framing in the 6.2 pointer, the LSTM
  section's swing attributions ("from the encoder alone" → the
  architecture, with the no-BN swing quantified beside it), "What
  survives", the discussion, the limitations, and the conclusion are
  re-attributed to match. The sequence arms' rescue stands as one
  instance of the BatchNorm-free rescue; the conclusion's parity
  sentence becomes BatchNorm-conditional with the no-BN FedProx row
  quoted beside FedProx-LSTM.
- The honest boundaries the register demands travel with the claim:
  PR-AUC is where work remains (0.345 below FedProx-LSTM's 0.404 and
  the centralised MLP's 0.445 — attribution, not a new
  state-of-the-art row); the in-partition no-BN control is a
  different substrate and ends below the BN-with-correct-statistics
  arm there (the two results are not readable into each other);
  FedAvg's declining test trajectory (0.974→0.963 over rounds 5–50
  while its validation F1 still improves — client drift, 2.05% val
  vs 1.31% test prevalence) makes its row a conservative lower
  bound; SCAFFOLD has no no-BN counterpart in the executed register,
  so its reading stays implementation-case-qualified.

## 3. A4 — the abstract's closure clause, metric label, and tag

The closure clause now states the executed attribution
("BatchNorm-under-federation-conditional, and two controls establish
it: …"), the panel's **F1** metric-label error is fixed ("TSS
0.404" → "PR-AUC 0.404" — the F1 pre-external-submission-mandatory
item, now discharged), and the internal "v4.9" tag is dropped from
the abstract. **B3** executes the Round-12 two-line fix in the same
pass: the persistence sentence now leads with 24-hour-lagged
persistence (0.418) and places same-region label inertia (TSS
0.967) second. **B5** separates the manifest sentence: every number
regenerates from committed *result artefacts*; the underlying
dataset cache is frozen under a *separate* 20-file SHA-256 manifest
verified by a committed script — the conflation the panel's F3
flagged is gone. `ABSTRACT.md` is synced to the record.

## 4. B1, B2, B4 — hygiene, qualifier, and the pin

- **B1**: Section 6.3 discloses which round the degenerate rule
  selects (the strict-inequality rule initialises its best at 0.0
  and never fires on an all-zero history, so the shipped checkpoint
  is the final round's global model, round 50, in both degenerate
  cases); the new arms' val-ROC sensitivity rows and per-round test
  trajectories are integrated in Section 6.4 (born with the arms at
  v4.9.2, landed at v4.9.4, now paper-side); and Section 9's
  "FedAvg detects zero of 65 events" is re-qualified with the arm-C
  38/65 (58.5%) at 0.28 false-alarm windows per day companion cell,
  with the same companion added at "What survives".
- **B2**: Table 8's caption now carries the implementation-case
  qualifier on its SCAFFOLD row (the last unqualified SCAFFOLD site
  in the paper).
- **B4**: `outputs/raw_substrate_rerun.json` is named in Section
  6.3 as the retrain record the artefact reproduces, and its
  SHA-256 (caca966a3b86b3127cfb1367a89ee7836dd98ec70bbf1440ea24c7
  ca6094b0a1) is pinned in `tests/test_raw_bn_diagnostic.py` under
  the artefact-pin convention — the check runs in the battery's
  artefact layer from this revision on.

## 5. Battery and apparatus

- `tests/test_submission_apparatus.py`: page-count pins bumped
  59/59 → **62/62** (the disclosed recompile, B7 convention); the
  submission regeneration is byte-identical (the generator's T2
  assertion re-pinned to the corrected "ROC-AUC 0.970 and PR-AUC
  0.404" pair); rendered bibliography 31/31 entries on both PDFs.
- `tests/test_raw_bn_diagnostic.py`: the B4 pin (one new check) in
  the artefact layer; the A1/A3 layers execute against the landed
  artefacts exactly as at v4.9.4.
- `tests/test_letters_manifest.py`: this letter registered (registry
  11 entries).
- `tests/run_battery.py`: BATTERY_VERSION v4.10.
- Full battery, torch-less (env row 3): **372/0 pre-commit** (this
  letter present but uncommitted, so its manifest verification
  defers — the documented convention), **373/0 post-commit** (this
  letter's manifest verifies against the actual diff). One pin was
  re-pinned during the pre-commit run, disclosed: the apparatus
  test's floor-marker carried the old "TSS 0.404" string — the F1
  metric-label fix propagates to its own pin, exactly as the B7
  convention intends.
- RUNLOG: the v4.9.4 version-history row is backfilled (disclosed
  here — the v4.9.4 commit patched the ask-#10/#11 outcome cells but
  never appended its own version row; the row is restored from the
  v4.9.4 commit message and letter, both committed evidence) and the
  v4.10 row added. ENVIRONMENTS row 3 extended to the v4.10 battery.

## 6. The register disposition, final

| Register item | Status after v4.10 |
|---|---|
| A1 arm-B centralised sanity | **EXECUTED (v4.9.4) + INTEGRATED (v4.10)** — `arm_b_validated`; Table 8's B column re-framed as the validated proxy |
| A2 arm-D transport run / re-frame | **RESOLVED — the re-frame branch executed paper-side**; arm-D stays the endorsed clean-closure option for external submission, not required for the reading |
| A3 no-BN raw federation | **EXECUTED (v4.9.4) + INTEGRATED (v4.10)** — Section 6.4 + Table 9; contribution 5 rewritten; the attribution is BatchNorm-under-federation-conditional |
| A4 abstract scoped rewording + F1 (TSS→PR-AUC) + drop the tag | **EXECUTED** (§3) |
| B1 checkpoint-selection hygiene | **EXECUTED** (§4 — degenerate-round disclosure, sensitivity rows/trajectories, the 38/65 re-qualification) |
| B2 E2 SCAFFOLD Table-8 qualifier | **EXECUTED** (§4; the rerun-or-demote decision stays owner-side, queue unchanged) |
| B3 persistence sentence swap | **EXECUTED** (§3) |
| B4 name + SHA-pin `raw_substrate_rerun.json` | **EXECUTED** (§4) |
| B5 abstract manifest-sentence rewording | **EXECUTED** (§3) |
| B6 battery minors | Executed at v4.9.2, unchanged |
| C1–C3 | **Owner decisions, untouched** — C3 is moot (A1–A3 landed in the current-reading world) |

## 7. What the owner runs next

1. `git pull origin improvements` (this commit).
2. **Nothing mandatory.** The register's compute programme is
   discharged and the paper-side integration is landed. Optional:
   re-run the battery (`python tests/run_battery.py`) to watch the
   v4.10 log title and the B4 pin activate; the B2 SCAFFOLD
   rerun remains the one optional owner-GPU queue entry, an owner
   decision as before; the C1/C2 packaging decisions (process
   language purge; the 15-page core-audit carve-out) are the
   remaining strategic calls, both documented in the register and
   both owner-side.

The cycle the Round-13 panel priced at 8.4–8.6 "if A1–A3 land with
results that hold the current reading" has landed them, integrated
them, and closed the register's blocking lane. The paper's story is
now the one the evidence supports: the collapse was never the
encoder's, and it was never the evaluation path's — it was
BatchNorm's, under federation, and removing it restores the federated
MLP to the centralised ranking.
