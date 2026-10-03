# Response to Dossier R-FS9-R1 — Re-Review of the v4.0 Evidence Edition

**Manuscript:** "Federated Solar-Flare Prediction on SWAN-SF: A Benchmark Audit of Provenance, Leakage, and Evaluation Protocols"
**Revision:** v4.1, commit `58443e8` (improvements branch), PDF 51 pp
**Referee dossier:** R-FS9-R1, "The Evidence Edition, Audited" — Major Revision (narrow scope)

We thank both chairs for a delta audit that is, once again, correct in essentially every particular. The verdict's central finding — that the evidence edition's flagship additions shipped broken in exactly the layer that distinguishes evidence from assertion — was verified by us line-by-line before this revision was planned, and every fix below was executed against the dossier's Chapter 7 list in order. One sub-claim is respectfully rebutted with evidence at the end of this letter.

The honest headline of the corrected apparatus, which both chairs asked to hear the way an operations room hears it: **at this window geometry, no arm in this paper beats same-region previous-window label persistence (TSS 0.967) — XGBoost included; on TSS, XGBoost (0.848) and logistic regression (0.489) alone clear the 24-hour-lagged floor (0.418), and on HSS no arm clears it (0.434).** Section 6.1 and the abstract now state this plainly, and we agree with the mission chair that it is not an embarrassing result but the honest one — and, for a benchmark-audit paper, the best exhibit the re-review could have handed us: even the auditors fell into the evaluation-protocol trap the paper exists to audit. The paper now says so. *[Erratum, v4.2: this letter's original second clause claimed XGBoost was the only arm above the lagged floor — a false exclusivity introduced by the correction itself and caught by Dossier R-FS9-R2 (its F1); see the erratum at the end.]*

---

## A. The verification apparatus (Chapter 3)

**A1 — persistence baseline ordering.** Fixed. `run_standard_metrics.py` now sorts each active region's windows by `ts_start_min` (with `pool_row` as a deterministic tie-break). We independently reproduced the dossier's scramble diagnostics before fixing (3,125/3,156 AR groups non-monotone in time; adjacent pairs 163,856 increasing vs 164,109 decreasing; median adjacent-pair gap 2,880 min = 2.0 days). The stored value 0.348 is retired.

Both variants are computed and reported, on the pooled test and the partition-5 test, each with its definition string, n, and confusion stored in the artefact:

| Definition (P5 test) | TSS | HSS | n |
|---|---|---|---|
| Previous-window label inertia | **0.967** | 0.970 | 74,612 |
| 24-hour-lagged persistence | **0.418** | 0.434 | 56,648 |

The label-inertia floor matches the dossier's 0.964/0.969 to tie-breaking on first-window inclusion. The lagged variant differs from the dossier's 0.374/0.407 because the matching rule differs: we implement the point-lag definition — the label of the same-region window *nearest to t − 1440 min*, matched within ±35 min (a tolerance near half the measured 60-min median cadence) — and the candidate definitions are committed since v4.2 as `experiments/run_lag_definition_sweep.py` → `outputs/lag_definition_sweep.json`: nearest-match at tolerances 30/60/120 min; a most-recent-observation-≥24h rule; exact-1440; window-index offsets 23/24; and a lookback-flare rule. The seven point-lag variants land at TSS 0.392–0.418 on the P5 test (the panel's own most-recent-observation figure, 0.374, sits just below our reading of that rule) and under every one of them exactly two arms clear at TSS (XGBoost 0.848, logistic regression 0.489) and none at HSS; the lookback-flare variant scores 0.964 — label inertia under another name, which no arm clears — which is why the paper reports inertia and lagged persistence as two distinct floors. The definition is stated in the artefact and in the paper, and the number regenerates from the committed code — which is the standard this paper asks of others. Section 6.1's persistence paragraph is rewritten around the two floors; the "XGBoost sits above that floor" sentence is deleted and replaced with the two-floor statement above; the abstract now carries the corrected headline; and the paragraph discloses, in the paper's own self-audit register, that the earlier pool-row implementation is precisely the class of defect the paper exists to audit. *[Erratum, v4.2: the original letter claimed "every one of them leaves the conclusions unchanged (XGBoost above, neural arms below…)" — wrong twice over: the correct arm enumeration is two-above-on-TSS/none-on-HSS, and the lookback-flare variant does not preserve it; the committed sweep now carries the exact numbers.]*

**A2 — LSTM F2 aggregation.** Fixed. A merged view (`metrics_view`) now resolves the F2 operating point from the nested `test` block (where the artefacts store threshold/precision/recall). All ten LSTM arms carry TSS/HSS at their frozen thresholds, and our regenerated cells match the dossier's expected values exactly: central 0.456, FedAvg 0.305, FedProx 0.634, SCAFFOLD 0.707 (seed-42); seed-43 and SMOTE cells likewise (0.479/0.333/0.560/0.677 and 0.383/0.330).

**A3 — raw-2D frozen-FPR cells.** Fixed. The same merged view resolves `frozen_operating_points` from the top level of each entry, where the raw-2D artefacts (and the leakage-free fold artefacts, which had the same latent gap) actually store them. All six raw-2D arms × four frozen-FPR points now carry metrics, as do the five leakage-free arms (previously fbeta-only).

**A4 — NaN literals.** Fixed at the source: the precision/recall confusion inversion now implements zero-alert semantics (stored precision = recall = 0 encodes no alerts → fp = 0, TSS = HSS = 0.0). The artefact is additionally passed through a strict-JSON sanitiser and written with `allow_nan=False`, so the file is parseable by JavaScript, R and Go. The previously-NaN cell (FedAvg-MLP raw, fbeta point) now reads TSS = HSS = 0.0.

**A5 — Poisson intervals.** Fixed. The by-hand formula is replaced with exact chi-square (Garwood) quantiles: `[χ²_{α/2}(2k)/2, χ²_{1−α/2}(2k+2)/2]`. For k = 20 the interval is now [12.2, 30.9] (was [15.2, 23.9]); for k = 0, [0, 3.69] (was 1.92). The Wilson summary sentence is recomputed from the regenerated artefact and now reads: half-widths of ±2.8–4.0 points on the leakage-free fold, up to ±11.8 points on the raw substrate.

**A6 — regeneration and coverage.** `outputs/standard_metrics.json` regenerated: strict JSON, zero NaN/Infinity literals, and content-identical across re-executions (the artefact embeds a wall-clock `elapsed_s` field, so file bytes differ between runs; the artefact-derived blocks A/C/D are deterministic). *[Erratum, v4.2: the original letter said "bit-identical across two consecutive re-executions" — an overstatement of file-level equality; Dossier R-FS9-R2 F4.]* The abstract's coverage sentence is narrowed to what the apparatus actually covers — every stored operating point of every substrate (in-partition, leakage-free, raw-2D fbeta + frozen points; ten LSTM cells; event-level tables), with reliability explicitly scoped to the six in-partition arms. The residual minor items are also fixed: the persistence entry's mislabelled `auc` field is removed (it duplicated TSS); the definitions block states the measured 60-min within-AR cadence and names 12 min as the MVTS record cadence (the two are now also disentangled in the communication paragraph); the Block D docstring's unfulfilled seed-band promise is deleted (the seed-43 cells remain as separate committed cells).

**What the apparatus already got right is unchanged:** the TSS/HSS/BSS/Wilson formulas, the per-split test climatology BSS reference, no threshold shopping, and the uncosmetic inclusion of the embarrassing results.

## B. The bibliography (Chapter 4)

All six new entries now carry metadata **read from the record** (arXiv listing pages, DBLP, IEEE Xplore, IOP/Semantic Scholar), verified during this revision with the source record named in each entry's note field:

- `wang2023bn` → Wang, Yanmeng; Shi, Qingjiang; Chang, Tsung-Hui. IEEE TNNLS 36(1):1692–1706, 2025, DOI 10.1109/TNNLS.2023.3323302 (DBLP record confirmed; arXiv:2301.02982 retained in the note).
- `guerraoui2024bn` → Guerraoui, Rachid; Pinot, Rafael; Rizk, Geovani; Stephan, John; Taiani, François. Title without "Heterogeneous" (arXiv:2405.14670).
- `bnscaffold2024` → Quintana, Gonzalo Inaki; Vancamberg, Laurence; Jugnon, Vincent; Mougeot, Mathilde; Desolneux, Agnès. Title ends "…Statistics in Federated Learning" (arXiv:2410.03281).
- `angryk2019` → Ahmadzadeh, Azim; Hostetter, Maxwell; Aydin, Berkay; Georgoulis, Manolis K.; Kempton, Dustin; Mahajan, Sushant; Angryk, Rafal A. "Challenges with Extreme Class-Imbalance and Temporal Coherence: A Study on Solar Flare Data", IEEE Big Data 2019, pp. 1423–1431 (the 2020 Sci-Data roster is no longer copied onto the 2019 paper).
- `ahmadzadeh2021` → Ahmadzadeh, Azim; Aydin, Berkay; Georgoulis, Manolis K.; Kempton, Dustin; Mahajan, Sushant; Angryk, Rafal A. ApJS 254(2):23, 2021, DOI 10.3847/1538-4365/abec88.
- `li2021fedbn` → Li, Xiaoxiao (one letter).

`fu2023`: the DOI is moved into the `doi` field; the "no formal DOI" self-contradiction and the duplicated posting date are removed. The three orphaned entries: `bolduc2002` and `baker2013` are deleted; `boteler2019` — the canonical review of the literature we explicitly do not evaluate — is now cited once, in the introduction's boundary disclaimer ("see Boteler 2019 for that literature"). The file carries 31 entries with **zero orphans**. The header now asserts only what was actually checked, and names the specific entries and records checked in this revision. We accept the dossier's characterisation — metadata written from memory under a re-verification header — without reservation, and the lesson is now encoded in the header itself.

## C. The paper text (Chapter 5)

- **C1:** `\ref{sec:limits}` → `\ref{sec:limitations}`; the compiled PDF has **zero unresolved references**.
- **C2:** `CLIENT_NAMES` are now plain `Client A`–`Client F` in `config.py` (with a comment that no longer contradicts its own contents), in `partition_clients.py`, and in `tools/make_fig_partition.py`; both `fig_clients.png` and `fig_partition.png` are regenerated from the committed scripts — no geographic labels remain anywhere in the rendered figures, and the captions now match what the figures show.
- **C3:** the conclusion's 23.4 h echo carries the label-boundary caveat ("a label-boundary quantity at this 24-hour label horizon, not a usable warning time"); Section 6.3's "4× the 1-hour minimum warning horizon" spin is deleted and replaced by the same caveat.
- **C4:** `sec_problem.tex` is swept. §3.2's custodian sentence now states the truth this revision's framing demands — that no data-locality constraint binds this benchmark, and the protocol is a controlled simulation of one. §3.3's "different custodians monitor different populations" is replaced with the single-instrument reality and the Dirichlet partition as a simulation instrument. §3.4's grid-protection cost structure is replaced by the rare-event cost structure (an all-clear forecast is 98–99% accurate and useless). §3.1's "agencies" → "simulated shards". §4.4 is retitled "Non-IID partitioning into simulated clients"; the partition figure caption and the "hostile federation" paragraph are de-regionalised; "geographic heterogeneity" → "label heterogeneity".
- **C5:** the "recorded per-run in the run manifest" sentence is rewritten to match the artefacts: the torch (2.14.1) and numpy (2.2.4) versions of the verification environment are pinned in `requirements.txt` **and recorded in the committed battery log header** (see the rebuttal below on the version's existence); the manifest writer records them per-run from v4.0 onward, and the committed manifest is identified as the v3.x run record that predates the field. The `requirements.txt` comment is made honest in the same terms.
- **C6:** the tab:main caption now states the true comparison: the climatology row is the validation-fitted base rate (the prior-shift convention), and the honest Brier reference is the **test-set climatology (0.0185 at 1.88% prevalence), which no arm's calibrated Brier beats**.

## D. Repository hygiene (Chapter 7)

- **D1:** `tests/test_leakage_gate.py` (5 checks) exercises the gate's failure path end-to-end on synthetic indices: shared regions across the train/test boundary → the region check fails, `overall_pass` is False, and the replicated main.py decision rule evaluates to REFUSE; the `--allow-in-partition` override is honoured; the clean path passes; the region-table-absent path stays open (v3.x semantics, now pinned by a test); and the window-overlap detector is exercised in both its flagging and clean modes. The battery stands at **219/219** (214 + 5), one count, log committed.
- **D2:** the battery log header records the executing environment (`python 3.13.5, numpy 2.2.4, torch 2.14.1+cpu`), so the environment claim is now artefact-backed; the manifest writer (main.py) records torch/numpy on every pipeline run from v4.0 onward.

## One respectful rebuttal, with evidence

The dossier states that `torch==2.14.1` "has never existed" (repeated from Dossier R-FS9, where we noted the same sub-claim in the v4.0 response). We ask the panel to re-verify: on the PyPI package index this revision's environment queries, `torch 2.14.1` is the **current release** (`pip index versions torch` → "Available versions: 2.14.1, 2.14.0, 2.13.0, …; INSTALLED: 2.14.1+cpu"), and it is the interpreter environment (`/usr/bin/python3.13`) in which this revision's battery, verification apparatus, and figure regeneration executed — as the committed `logs/test_battery.log` header now records. The pin therefore stands. We emphasise that the *adjacent* and load-bearing part of the dossier's finding — that the committed `run_manifest.json` contained no torch/numpy fields and therefore did not support the paper's "recorded per-run" sentence — was **correct**, and is fixed as described under C5/D2; nothing in this rebuttal changes any conclusion of the re-review.

---

## Summary for the editor

All five blocking items of the re-review's Chapter 7 are executed: the persistence floors are corrected, defined, and stated plainly in the abstract, Section 6.1, and the conclusions the corrected numbers force; the coverage holes are closed and the artefact is strict JSON, content-exact on re-execution across its artefact-derived blocks; the bibliography is rebuilt from records with a header that claims only what was checked; the page-one cross-reference and the figure/caption contradiction are fixed; and the environment claims are artefact-backed. The one new experiment the dossier speculated might be needed — regenerating the metrics artefact — was performed; no retraining was required, and no number outside the verification apparatus changed. The battery is 219/219 with the leakage gate's failure path now under test.

We end where the dossier ends: v3.9's numbers were real and its story was not; v4.0 made the story real; v4.1 makes the evidence layer match it. The paper that now stands is the one both chairs described wanting — a careful, mostly honest benchmark audit of a widely used dataset whose random-split leakage the community has quietly tolerated since 2019, with the persistence floor finally measured by an instrument whose time axis points the right way.

---

## Erratum (v4.2, against Dossier R-FS9-R2)

Dossier R-FS9-R2 verified every claim of this letter against the committed artefacts and found three statements of ours to be wrong. They are corrected above in place, with inline erratum markers, and are listed here so the record is complete:

1. **The headline's second clause** — "XGBoost (0.848) is the only arm above the 24-hour-lagged floor (0.418)" — was a false exclusivity: logistic regression clears the floor at TSS 0.489 on the same fold and threshold, and at HSS no arm clears (XGBoost 0.201 < 0.434). The correct clause now reads as corrected above and appears in the paper's abstract, Section 6.1, and appendix (v4.2 revision row).
2. **The A1 sensitivity sentence** — "every one of [the eight candidate definitions] leaves the conclusions unchanged (XGBoost above, neural arms below, inertia above all)" — was wrong in its arm enumeration (two arms clear, not one) and in its scope (the lookback-flare variant, at TSS 0.964, does not preserve the lagged-floor ordering; the invariance holds across the seven point-lag variants only). The sweep is now a committed artefact: `experiments/run_lag_definition_sweep.py` → `outputs/lag_definition_sweep.json`.
3. **The A6 regeneration claim** — "bit-identical across two consecutive re-executions" — overstated file-level equality; the artefact embeds a wall-clock `elapsed_s` field. Corrected to content-exact, with block B's phase-cache dependency stated.

Two further letter-level errata, both trivial and both caught by the dossier: A3 said "six leakage-free arms" where the fold has five, and the tolerance parenthetical described ±35 min as "half the measured 60-min median cadence" (35 is near, not equal to, half of 60; the paper now states the tolerance as ±35 minutes plainly).

The panel's verdict on the manuscript itself — that the v4.1 revision is verified correct in essentially every particular this letter claimed — stands, and we accept the one substantive catch with the same standard this cycle has converged on: the claim was checked against the committed artefact, on both sides of the table, and the artefact won.
