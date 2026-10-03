# Response to Dossier R-FS9-R2 — Re-Review of the v4.1 Corrected Apparatus

**Manuscript:** "Federated Solar-Flare Prediction on SWAN-SF: A Benchmark Audit of Provenance, Leakage, and Evaluation Protocols"
**Revision:** v4.2, improvements branch (commit in the RUNLOG), PDF 51 pp
**Referee dossier:** R-FS9-R2, "The Corrected Apparatus, Audited" — Minor Revision (conditional acceptance)

We thank both chairs for a round that verified our work at the level of re-implementation, retracted the panel's own expired-prior error where we were right, and caught the one false clause we shipped under the sentence announcing our correction. The verdict's framing — that the remaining distance to acceptance is one clause long — is accepted, and that clause (and everything behind it) is fixed in this revision. No new experiment, retraining, or artefact regeneration was required for items 1–4 of the panel's Chapter 7; the optional item 5 (F7) was executed rather than dropped, because dropping a sensitivity claim is weaker than committing the artefact that backs it.

## F1 — the corrected headline's false exclusivity (moderate)

**Conceded in full, and fixed in all four places.** The panel is right, and the reproach lands exactly where it should: the v4.0 sentence we replaced carried no exclusivity; the v4.1 sentence introduced one, in the very clause that announced the correction — the class of defect this paper exists to audit, committed by us, one round after we documented the same failure mode in the v4.0 bibliography. Logistic regression, at its frozen F-beta threshold on the leakage-free fold, stores TSS 0.489 (confusion 990 / 37,998 / 0 / 36,377) against the 0.418 floor, and at HSS no arm clears (XGBoost 0.201 < 0.434). The corrected statement now reads, in the abstract:

> "…under the corrected persistence baselines, no arm beats same-region label inertia (TSS 0.967) at this 24-hour label geometry; on TSS, XGBoost (0.848) and logistic regression (0.489) alone clear 24-hour-lagged persistence (0.418), and on HSS no arm clears it (0.434)."

Section 6.1 now carries the two-floor statement with the same enumeration (both occurrences: "XGBoost ($0.848$) and logistic regression ($0.489$) are the only arms above it… while on HSS no arm clears the floor at all"; and the closing "…and on this fold, at TSS, two arms do, none at HSS"); the appendix's v4.1 revision row is corrected in place and a v4.2 row records the round; the archived R-FS9-R1 response letter is corrected with inline erratum markers and a closing erratum section. In the same pass, logistic regression's TSS/HSS (0.489 / 0.025) was added to Section 6.1's first finding, which previously enumerated only XGBoost and the neural arms — the same omission pattern, fixed at the source.

We also accept the panel's framing of its own share in this (R-FS9-R1 certified "XGBoost above the floor survives" without enumerating the arms) as given, not as mitigation. The lesson we encode in the artefact, per the fix list: a floor claim is a statement about every arm, and the committed sweep now answers "which arms clear" per definition, read from `standard_metrics.json` at run time — never hand-copied.

## F2 — the event-level false-alarm range (minor)

**Fixed.** Section 6.1's event-level narration now reads "1.7 false-alarm windows per day versus **11.6–12.7** for the neural arms" (tab:eventlevel: centralised MLP 12.7, FedAvg 11.6, FedProx 11.9). The 8.6 endpoint belonged to logistic regression, as the dossier states, and remains where it belongs — in the table row it came from.

## F3 — the conflated environments (minor)

**Fixed; both environments named.** The sentence now reads: "the protocol run of record used Python 3.12; the committed battery log header records the v4.1 verification battery's environment --- python 3.13.5, numpy 2.2.4, torch 2.14.1+cpu --- the library versions pinned in `requirements.txt`". We took the panel's second option in substance — the header is now cited only for what it records (the battery), and the protocol run's interpreter is stated separately.

## F4 — "bit-exact" (minor)

**Fixed; both halves of the panel's item 4 executed.** Section 6.1's apparatus sentence now reads: "strict JSON; the artefact-derived blocks (A, C, D) are content-exact on re-execution, block B's reliability recomputation additionally requires the uncommitted phase-cache models, and the artefact embeds a wall-clock `elapsed_s` field, so file-level equality is not claimed." The appendix row is softened identically. The archived letter's "bit-identical across two consecutive re-executions" claim is corrected to content-identical in its erratum.

## F5 — the letter erratum (trivial)

**Fixed.** The archived letter's A3 now reads "five leakage-free arms", and its erratum section lists the miscount explicitly.

## F6 — the docstring cadence (trivial)

**Fixed.** `persistence_predictions_24h`'s docstring now reads "a tolerance near half the measured 60-min median within-AR cadence" — the stored artefact value is 60.0, and the stale ~64-min figure is gone.

## F7 — the definition sweep (note)

**Committed, not dropped.** `experiments/run_lag_definition_sweep.py` → `outputs/lag_definition_sweep.json` now exist in the repository. The script computes eight candidate definitions of the lagged floor on the committed provenance table (both P5 and pooled test), each with its definition string, n, confusion counts, TSS and HSS, and cross-references the leakage-free arms' frozen F-beta TSS/HSS (read at run time from `standard_metrics.json`) so the "which arms clear this floor" question is answered per definition by the artefact itself. It is self-contained (pandas/numpy only; a torch-less fresh clone can run it) and the shipped-rule reference row reproduces block C's stored values exactly (consistency check: PASS).

The committed numbers, which the paper now cites in Section 6.1: the **seven point-lag variants** (nearest at ±30/±60/±120; most-recent-observation-≥24h; exact-1440; index offsets 23/24) land at **TSS 0.392–0.418** on the P5 test, with exactly two arms clearing each (XGBoost, logistic regression) and none at HSS; the **lookback-flare variant** scores **TSS 0.964 / HSS 0.790** — label inertia under another name, which no arm clears. We note two honest disclosures this artefact forces, both now stated in the letter erratum and the paper: (i) our reconstruction of the panel's most-recent-observation rule scores 0.416, just above the panel's own 0.374 — a residual definitional difference disclosed rather than hidden, and immaterial to any conclusion; and (ii) the v4.1 letter's "every one of them leaves the conclusions unchanged" was over-scoped — the invariance holds across the point-lag family only, which is why the paper reports inertia and lagged persistence as two distinct floors.

## Residuals carried forward (Chapter 6)

All three are acknowledged and stand as stated by the panel: block B's phase-cache dependency (now stated in the paper under F4), the committed run manifest as the v3.x record (unchanged from v4.1), and the leakage-gate test's replication of main.py's decision branch. On the last: the panel's suggested five-line refactor (a shared imported function) is noted as owner-side follow-up; we did not execute it in this round because the panel itself carries it "without weight," and a v4.2 scope that touches the gate's call path would require re-running the full battery in a torch-equipped environment — which this revision's verification environment, having lost its torch installation, cannot do.

## Verification note

The battery was re-run in this revision's verification environment (python 3.12.14, pandas/numpy/scipy/matplotlib/sklearn, torch not importable — the sandbox has changed since v4.1, and the log header says so): **195 passed, 0 failed, 24 torch-dependent checks skipped** — the same delta the panel's own torch-less fresh run showed, and a consistent skip accounting with the committed v4.1 log (219/219, `logs/test_battery.log`, retained untouched; the fresh run is committed as `logs/test_battery_v4.2.log`). The paper's appendix v4.2 row states exactly this. The new sweep artefact is strict JSON, deterministic (no wall-clock fields), and reproduces block C's shipped-rule values to 1e-9.

One closing observation, offered in the register the dossier itself set: the panel's retraction of the torch sub-claim and its disclosure of the R1 omission are the first time in this cycle that either side has corrected the other's *facts* rather than framing, and both corrections went to the live record rather than to memory. We have adopted the same standard on our side — every number in this letter regenerates from a committed artefact, including the ones that refute our own previous letter.
