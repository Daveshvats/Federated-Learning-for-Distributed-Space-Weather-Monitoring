# Response to Dossier R-FS9-R3 — Re-Review of the v4.2 One-Clause Fix

**Manuscript:** "Federated Solar-Flare Prediction on SWAN-SF: A Benchmark Audit of Provenance, Leakage, and Evaluation Protocols"
**Revision:** v4.3, improvements branch (commit in the RUNLOG), PDF 53 pp
**Referee dossier:** R-FS9-R3, "The One-Clause Fix, Audited" — Accept, conditional on four clerical items (N1–N4)

We thank both chairs for a fourth-round dossier that verified our revision at the level of byte-level re-execution — the sweep artefact reproduced SHA-256-identically in the panel's own torch-less environment — and for a residue register that is, as the panel states, conditions on the record rather than on the science. The four items are fixed below exactly as prescribed; nothing else in the paper was touched. No experiment was run, no artefact regenerated, and no number, claim, or figure changed in this revision — the appendix version-history row and its caption range are the only additions to the compiled paper. *[Erratum, v4.4: this minimality claim is refuted by the revision's own diff — the two battery-count sentences of Section 4 were also rewritten (the components sentence now names the sweep-coverage module; the components-list sentence reports the canonical 219/219 record and the 224/0 torch-less v4.3 count with its twenty-four skips), exactly as the RUNLOG v4.3 entry disclosed at the time. Caught by Dossier R-FS9-R4 (its P1); see the erratum at the end.]*

## N1 — the corrected response letter was unverifiable

**Conceded in full; the letter is now in the repository.** The panel is right that a claim about a document the record cannot show is the exact class of unsupported provenance assertion this cycle was conducted to eliminate — and that we made it in the commit message, the RUNLOG, and the appendix row simultaneously. The corrected R-FS9-R1 response letter (with its inline erratum markers and closing erratum section) is now committed at `docs/response_letters/RESPONSE_R-FS9-R1_v4.1_ERRATA.md`, alongside this cycle's R-FS9-R2 and R-FS9-R3 letters, so every letter-level claim in the ledger is verifiable in-repo from this revision forward. The directory carries a README stating its purpose. We took the panel's first option (commit the letter) rather than striking the fourth location, because the erratum is genuine history and the record is stronger with it in.

## N2 — the mistitled battery log

**Fixed at both the log and the root cause.** `logs/test_battery_v4.2.log` now opens "integrity battery — single authoritative count (v4.2)" — the one word the panel prescribed. The root cause was a hardcoded version string in `tests/run_battery.py`; the title is now emitted from a single `BATTERY_VERSION` constant with a comment recording this finding, so the log title cannot silently trail the release again.

## N3 — the sweep artefact had no battery coverage

**Fixed: `tests/test_lag_sweep_artifact.py`**, in the existing `test_audit_artifact.py` pattern, adds 29 checks to the battery. It validates, with everything read at run time from the two committed artefacts and no number hand-copied into the test: strict-JSON parsing of `lag_definition_sweep.json` (NaN/Infinity abort); the 9-definition structure (8 candidates + 1 shipped, each with pooled and P5 blocks and clears-the-floor lists); the shipped-rule row against block C of `standard_metrics.json` to 1e-9 (the generation script's self-check, now battery-guarded); the clears-the-floor lists for **every** definition recomputed from block A's leakage-free fold at each definition's own P5 floor; the invariance of the clears list across the point-lag family; and the lookback variant's separately-stored `n_evaluated` — the semantic footnote the panel recorded for completeness. A future edit that silently broke the sweep's agreement with the metrics artefact now fails the battery.

## N4 — the missing ledger hash

**Fixed.** The RUNLOG's v4.2 entry now records the sweep artefact's digest — `sha256[:16] ceb7189826af15af` (full: `ceb7189826af15af84be4efb012ab939bab30d6512197ddff5900e5b6f682e32`), matching the ledger's existing 16-hex convention.

## The environment observation

The panel recorded, without a finding number, that the environment count in the record is drifting upward and that a one-line environment manifest would end the bookkeeping. `docs/ENVIRONMENTS.md` now exists: one table, every environment in the record, what each ran, and which committed log or manifest records it. We add nothing to the count ourselves: the v4.3 battery log is the same python 3.12.14 / numpy 2.1.3 / torch-absent environment as the v4.2 log, and the canonical v4.1 log (219/219, python 3.13.5 / numpy 2.2.4 / torch 2.14.1+cpu) is retained untouched, as in v4.2.

## Verification note

The battery re-run in this revision's verification environment: **224 passed, 0 failed** — the v4.2 count of 195 plus the 29 new sweep-coverage checks — with the same 24 torch-dependent skips as the panel's own fresh run (`logs/test_battery_v4.3.log`; per-file 75/20/29/34/17/5/skip/13/18/13 — the v4.2 order with the new module inserted after the audit-artefact suite). The sweep script was re-executed once more before committing: byte-identical again (SHA-256 match), consistency self-check PASS. The compiled paper is 53 pages with zero unresolved references (the appendix's new v4.3 row and its caption range, now v2.1–v4.3, add the one page); no other page, claim, or number changed. *[Erratum, v4.4: "no other page, claim, or number changed" is wrong in the same way — the two rewritten Section 4 battery sentences carry exactly such a change; see the erratum at the end.]*

With the four clerical items closed and nothing else changed, we take the panel's disposition — acceptance conditional on N1–N4 — as satisfied, and post this as the final version of the cycle. *[Erratum, v4.4: "nothing else changed" is likewise understated — see the erratum at the end.]* The closing sentence of the dossier deserves a reply in kind: the cycle's standard, converged on by both sides across four rounds, was that every claim regenerates from a committed artefact and the artefact wins. The four items fixed today were the last places where the record itself failed that standard. They are fixed; the ledger is closed.

---

## Erratum (v4.4, against Dossier R-FS9-R4)

Dossier R-FS9-R4 — the cycle's closing round — found this letter's
minimality claim refuted by the revision's own diff, and the finding is
correct. The v4.3 commit rewrote the two battery sentences of
`paper/sections/sec_experiments.tex`: the components sentence now names
`tests/test_lag_sweep_artifact.py`, and the components-list sentence
reports both the canonical torch-equipped 219/219 record and the 224/0
torch-less v4.3 count with its twenty-four skips. The opening
paragraph's "the appendix version-history row and its caption range are
the only additions to the compiled paper," the verification note's "no
other page, claim, or number changed," and the same phrase in the
closing paragraph were therefore each understated; all three are
corrected above in place with inline erratum markers. The RUNLOG v4.3
entry disclosed these edits accurately at the time — the underlying
record (paper, logs, ledger) was honest; the letter, the one
hand-written artefact in the pipeline, drifted from the record it
summarises, for the third time in the cycle (R-FS9-R2 F1, R-FS9-R3 N1,
and now R-FS9-R4 P1). The rewritten sentences themselves are accurate
and stand unchanged; the defect was exclusively in this letter's
description of what was done. Dossier R-FS9-R4's other two non-blocking
items are fixed in the same v4.4 revision: the battery's default log
path is now versioned and refuses to overwrite an existing log (P2),
and the archived R-FS9-R2 letter's page-count slip carries its own
erratum (P3) — see the RUNLOG v4.4 entry.
