# SF-9 — Response to External Review #2 (v3.0.1 manuscript → v3.1 code / v3.2 paper)

> This document maps every finding of the second external review to the
> concrete fix shipped on the `improvements` branch. Status legend:
>
> - **DONE** — fix is complete in code and/or manuscript, committed.
> - **SHIPPED + QUEUED** — protocol/feature is implemented, unit-tested, and
>   smoke-validated, but its *full-quantitative* execution needs the complete
>   SWAN-SF corpus and training re-run (owner-side compute); the manuscript
>   reports this status explicitly and labels pre-existing numbers accordingly.
> - **PAPER-ONLY** — wording/scope fix in the manuscript.
>
> Test suite: **78/78 checks pass** (58 original + 20 new for the v3.1
> protocol extensions). No headline number was changed: the seed-42
> frozen-protocol artefacts are the single source of numeric truth; all
> revisions are protocol hardening, claim correction, and scope honesty.

---

## A. The five mandatory pre-submission items

### M1. Region-disjoint train/val/test split (active-region-based)

**Status: SHIPPED + QUEUED.**

- `data_preparation.region_disjoint_val_split()` — greedy region-to-fold
  assignment minimising class-rate drift; unit-tested (test #11b block):
  disjointness, coverage, rate-drift bounds.
- The cleaned SWAN-SF export carries **no HARP/AR identifiers**, so the
  generator cannot execute on it. The manuscript (Sec. 8, "Region-identity
  and validation inflation") states this, attributes the val≈0.99 vs
  test≈0.95 gap to probable region overlap across the stratified
  carve-out, and queues the raw-metadata split as re-run item (1).
- Leakage audit already flags the limitation at runtime and refuses to
  train on audit failure.

### M2. Natural-prevalence validation + threshold-at-FPR protocol

**Status: SHIPPED + QUEUED.**

- `evaluation.select_fpr_thresholds_on_validation()` — deployment
  thresholds for FPR budgets {0.5, 1, 2, 5}% frozen as the
  (1−f)-quantile of **validation negative scores** (no test
  information); `frozen_operating_point_metrics()` reports realised
  test FPR / recall / precision / alert rate in the single test pass.
- Numerically validated under a 25× prevalence mismatch (realised FPR
  within 0.3% of target) — see tests/test_pipeline_integrity.py.
- Wired into `main.py` (run point 5c); `results.json` schema extended
  with `fpr_thresholds_selected_on: "validation (negative quantile)"`.
- The natural-prevalence validation *split* mode is implemented; using
  it as the default substrate requires the full-data re-run, which is
  queued (re-run item 2). Manuscript Sec. 4.7 describes the protocol
  exactly as implemented, including its queued status.

### M3. Resolve α=1.0 vs α=5.0 contradiction; freeze the final experiment

**Status: DONE (documentation + config provenance) / QUEUED (promotion re-run).**

- `config.py` now records both values with provenance:
  `DIRICHLET_ALPHA = 1.0` (preregistered headline + all statistical
  studies: multiseed, ablation, client-level) and
  `DIRICHLET_ALPHA_SWEEP_WINNER = 5.0` (validation-selected optimum,
  15-round sweep; single test pass PR-AUC 0.297 ≈ headline 0.307).
- Manuscript Sec. 5.3 ("Federated configuration and preregistration")
  states the distinction explicitly: α=1.0 is the adversarial *study
  regime*, α=5.0 the *tuning optimum*; promoting the sweep winner to
  headline requires re-running multiseed + ablations under it — queued
  (re-run item 7). Both values are in the run manifest.

### M4. Rewrite FedProx/federation claims to match ablation evidence

**Status: DONE (paper-only).**

- "Load-bearing" language removed everywhere. The defensible claim —
  stated identically in the abstract, intro contributions, results
  (Sec. 6.5), discussion (Sec. 7.1), and conclusion — is: **FedProx was
  the most consistent stabiliser among the components studied, under
  the preregistered configuration, on this benchmark.**
- The ablation contradiction is now stated in the paper itself
  (Sec. 6.6): the best FedAvg+DA-FL cell (ROC-AUC 0.963) *exceeds* the
  FedProx headline (0.954), so stability is not attributable to the
  proximal term alone.
- "Federation at zero cost" → "did not underperform the pooled
  centralised MLP of identical architecture on this benchmark and
  configuration", always paired with the budget caveat (M6/#15).

### M5. Reframe scope: data-local (not privacy), precursor alert (not GIC warning), SCADA as future interface

**Status: DONE (paper-only, title included).**

- Title: "Federated Learning for **Data-Locality-Preserving**
  Distributed Space Weather Monitoring: **A Solar-Flare Prediction
  Component** for Grid-Resilience Pipelines" (was "Privacy-Preserving
  … Smart Grid Protection").
- "Privacy-preserving" purged; abstract, Sec. 3.2, and conclusion say
  plainly: *the verified property is data locality, not cryptographic
  privacy*.
- "GIC early warning" → "M/X flare-precursor alerts: an upstream
  component that a future geomagnetic-hazard pipeline could consume;
  not evaluated here". Discussion Sec. 7.2 ("Operational positioning:
  an upstream precursor alert") does not claim GIC prediction.
- SCADA appears only as a future integration interface with an explicit
  "not evaluated" tag (Sec. 4.1, Fig. 1 caption, Sec. 7.2, limitations,
  conclusion).
- `docs/GIC_BOUNDARY.md` (pre-existing) remains the long-form boundary
  statement and is consistent with the new text.

---

## B. Point-by-point response (findings 6–25)

### #6. Prior shift as a named methodological problem + calibration comparison
**SHIPPED + QUEUED.**
- Manuscript Sec. 6.4 now opens by naming prior shift "a first-class
  methodological problem … the central obstacle between the current
  results and deployment-grade probabilities".
- The honest numbers are in the paper: FedProx calibrated Brier 0.264 /
  ECE 0.496 — **worse than climatology** (0.239 / 0.470). The previous
  "posterior is usable" sentence is gone.
- `evaluation.py` now ships `PriorShiftCorrection`, `PlattScaling`,
  `IsotonicCalibration`, `TemperatureScaling`, `IdentityCalibration`,
  `select_calibration()`; `experiments/run_calibration_comparison.py`
  evaluates all arms + frozen-FPR thresholding on balanced val, natural
  val, and the single test pass. Smoke-mode artefact
  `outputs/calibration_comparison.json` (synthetic probabilistic model)
  numerically validates the harness end-to-end. Full FL-model execution
  is queued (re-run item 3).

### #8. Figure 1 client labels + caption disclaimer
**DONE.**
- `fig_architecture.png` regenerated: clients are now "Client A …
  Client F" with regions only; no agency names, no instrument names;
  aggregation formula shows the plain size-weighted default with DA-FL
  marked as ablation; output layer renamed "FLARE-PRECURSOR ALERT
  LAYER — upstream component", final stage "Future: GIC/SCADA
  pipeline — not evaluated in this work".
- Caption in Sec. 4.1 states "neutral labels A–F; no real institution
  participates".
- `config.CLIENT_NAMES` neutralised (source of truth for all future
  figures); `fig_partition.png` and `fig_clients.png` regenerated with
  neutral labels.

### #9. Data-sovereignty premise rewording
**DONE (paper-only).** Intro paragraph 3 rewritten: institutions
"restrict redistribution … whether legally binding or institutional
policy"; the study "does not claim that any specific agency's flare
data cannot be pooled; it takes data locality as a plausible deployment
constraint and studies its consequences under a controlled simulation".
"Structurally identical" privacy analogy softened to "structurally
similar".

### #10. Title change
**DONE.** See M5. pdftitle/hyperref metadata, custom title block, and
repo `ABSTRACT.md` all updated to match.

### #13. recall@FPR is a window-level curve statistic (correlated windows)
**DONE (paper-only).** Table 3 caption and Sec. 7.2 now say exactly
this: the values are points on the *test ROC curve* at target FPR
(no threshold transfer), and because the 60-minute sliding windows of
one active region are correlated, "a 2% window-level FPR does not
translate directly into an operational false-alarm rate per event or
per day". The "recovers 51% of eventual M/X events" sentence was
corrected to "51% of positive *windows* (recall 0.506)" with the
event-level caveat attached.

### #14. Client-level evaluation on untouched holdouts
**SHIPPED + QUEUED.**
- `evaluate_clients.py` gains an untouched-holdout mode: each client
  reserves a fraction that *no* federated model trains on, so
  Local-vs-Global become clean generalisation estimates.
- Unit-tested; execution needs retraining — queued (re-run item 5).
- Manuscript Sec. 6.7 + Sec. 8 keep the within-distribution caveat
  ("optimistic for the federated side") and state the queued fix.

### #15. Centralized-MLP training-budget table
**DONE.**
- `centralized_baseline.training_budget_report()` emits a
  machine-readable audit; wired into `main.py` and `results.json`
  (`training_budget` key).
- Manuscript Table (Sec. 5.5): ≤30 epochs vs 50×10 local epochs
  (500 effective passes per client shard), batch 256 vs 512, early
  stopping patience-5-on-val-BCE vs fixed rounds, checkpoint rules,
  and the statement "budgets are **not** matched — the comparison
  confounds federation with budget; budget-matched re-run queued".
- Every headline FedProx-vs-centralised-MLP sentence in the paper now
  carries this caveat.

### #16. n=5 statistical wording
**DONE (paper-only).** Sec. 6.5 rewritten: what five seeds support
(sign-consistent superiority under *this* configuration, p=0.031) vs
what they do not (generalisation; CIs quantify seed-to-seed
variability of a fixed configuration, not future-observation
uncertainty); ablation cross-reference that the gap narrows under
DA-FL.

### #17. SMOTE not a contribution
**DONE (paper-only).** Intro contribution list demotes SMOTE to an
ablation arm with "(inert on this benchmark variant)"; a dedicated
post-contribution paragraph states what is *not* claimed (zero
synthetic samples, null result, not a contribution); Sec. 4.6
documents the disabled-by-default configuration; Sec. 6.6 keeps the
null-result reporting.

### #18. Raw SWAN-SF experiment
**QUEUED (largest extension, documented).** Sec. 8 "Benchmark-
construction effects and the raw-benchmark experiment" reframes the
current results as answering "can a federation learn the *cleaned*
benchmark?" and specifies the raw-partition programme (separating
benchmark-construction effects from federation effects, engaging the
SMOTE arm, enabling natural-prevalence validation). Listed as re-run
item 9. No code can produce this without the raw corpus.

### #19. Event-level evaluation
**SHIPPED + QUEUED.**
- `experiments/run_event_level.py`: event detection rate, missed
  events, alerts per detected event, duplicate-alert rate (with
  cooldown de-duplication), warning lead time (median [min, max]),
  alert rate per day. CLI + programmatic; unit-tested (two real bugs
  caught by tests: cooldown dedup previously missed total-alert/false-
  alarm accounting; budget report crashed on lr=None — both fixed).
- Execution needs event grouping keys + timestamps from raw metadata —
  queued (re-run item 4). Manuscript Sec. 8 "Event-level evaluation"
  paragraph + Table 3 caveat state the window-level limitation.

### #20. Communication-cost extension
**DONE (paper-only, measurement honesty).** Sec. 6.8 rewritten: the
numbers are *parameter-exchange arithmetic* (bytes counted, not
measured on a live deployment); training cadence vs observation
cadence distinguished; wall-clock latency, aggregation compute,
straggler recovery, and secure-aggregation overhead explicitly *not
measured* (single CPU container, in-process, no real network); TLS/
authentication pointers to the threat model.

### #21. Threat-model table
**DONE.** New Sec. 4.8 "Threat model and security posture" with
Table (tab:threat): curious server (secure aggregation — implemented
as verified module, not wired into training, overhead unmeasured);
gradient inversion (SecAgg + DP hooks — documented, not evaluated);
membership inference (DP hooks — documented, not evaluated); malicious
client / poisoning / backdoor (**not addressed**); client
impersonation (TLS + auth — deployment guidance); network interception
(TLS — deployment guidance). Consistent with
`privacy_analysis/THREAT_MODEL.md`, which remains the long-form
version. The benchmark study runs undefended — stated in the paper.

### #22. Literature-search protocol
**DONE.** New Sec. 2.4 "Positioning of this work and search protocol":
databases (web, arXiv, NASA ADS, ScienceDirect/IEEE via search engine),
date (2026-09-29), four verbatim query strings, inclusion/exclusion
criteria, screening counts (30 records → 0 in scope), raw listings
committed at `docs/lit_search/`. Limitations stated (web-engine
coverage, single-reviewer screening; PRISMA-style dual log queued).
Re-verified: adjacent works found are FL for *solar power* /
*terrestrial weather* / *spacecraft computing* — none on flare or
space-weather forecasting.

### #23. Manuscript corrections
**DONE (all four).**
- **(a)** Table 2 gains an ECE column (XGB 0.115, FedAvg 0.342,
  CentMLP 0.439, FedProx 0.496, LR 0.732, Clim 0.470 — from
  `results.json`, 15 bins) and the caption notes climatology beats all
  neural models on Brier.
- **(b)** The "Table ??" reference (undefined `tab:shap`) is fixed →
  `Fig.~\ref{fig:shap}` in Sec. 7.3.
- **(c)** Appendix version-history table extended with v3.0, v3.0.1,
  v3.1 rows (diagnosed failure mode → fix), caption "v2.1–v3.1".
- **(d)** "Calibratore" → "calibrator" (Sec. 5.1).

### #24. Conclusion claims taxonomy
**DONE.** Conclusion rewritten around a three-tier classification:
**Demonstrated** (stability/metrics/null-result/SHAP-grouping, with
protocol qualifiers) / **Supported with caveats** (budget confound,
DA-FL-best-cell, within-distribution client holdouts, window-level
FPR semantics, seed-only CIs) / **Not demonstrated** (cryptographic
privacy, region-disjoint generalisation, deployment-grade calibration,
event-level performance, GIC/SCADA pipeline, LSTM/SCAFFOLD, raw
benchmark) — each tied to the queued re-run item that would upgrade it.

### #25 / overall verdict alignment
The reviewer's summary judgements are now literally reflected in the
manuscript: FL is a strong *prototype* on a *simulated* federation
(Sec. 8 "Partition realism"); statistical evaluation is solid but
needs natural-prevalence validation (M2); calibration is unresolved
and stated as open (Sec. 6.4); leakage/generalisation is the headline
limitation (Sec. 8, first paragraph); novelty rests on a documented
search protocol (Sec. 2.4); operational readiness is not claimed
(conclusion tier 3).

---

## C. Queued re-run programme (owner-side compute required)

Implemented, unit-tested, blocked only on compute/raw metadata:

1. Region-disjoint splits from raw SWAN-SF metadata (needs HARP/AR IDs)
2. Natural-prevalence validation + full frozen-FPR operating-point report
3. Calibration-comparison harness (raw / prior-shift / Platt / isotonic / temperature)
4. Event-level evaluation (needs event keys + timestamps)
5. Untouched-holdout client evaluation
6. Budget-matched centralized-vs-federated training
7. Sweep-winner promotion (α=5.0) through multiseed + ablation
8. Federated LSTM and SCAFFOLD evaluation
9. Raw unbalanced SWAN-SF experiment programme

Items 1–5 are preconditions for quoting any number as operational
performance rather than benchmark result.
