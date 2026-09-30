# SF-9 — Response to External Review #2 (v3.0.1 manuscript → v3.1 code / v3.2 paper → v3.3 executed edition)

> This document maps every finding of the second external review to the
> concrete fix shipped on the `improvements` branch. Status legend:
>
> - **DONE** — fix is complete in code and/or manuscript, committed.
> - **SHIPPED + QUEUED** — protocol/feature is implemented, unit-tested, and
>   smoke-validated, but its *full-quantitative* execution needs the complete
>   SWAN-SF corpus and training re-run (owner-side compute); the manuscript
>   reports this status explicitly and labels pre-existing numbers accordingly.
> - **EXECUTED** — previously queued item has since been run end-to-end on
>   the real Cleaned SWAN-SF corpus in an independent re-execution
>   (2026-09-29, CPU-only, public dataset artefacts SHA-256-verified against
>   the frozen data manifest); results are committed as artefacts and
>   reported in the manuscript (v3.3).
> - **PAPER-ONLY** — wording/scope fix in the manuscript.
>
> Test suite: **78/78 checks pass** (58 original + 20 new for the v3.1
> protocol extensions) plus 16/16 FL smoke checks. Independent
> re-execution (v3.3): every headline test metric of all six models
> reproduced **exactly** (bit-stable CPU protocol, seed 42) — see
> `outputs/results.json` (re-executed), `outputs/calibration_comparison.json`
> (real mode), `outputs/client_holdout_eval.json`.

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

**Status: EXECUTED (threshold protocol) / SHIPPED + QUEUED (natural-prevalence validation).**

- `evaluation.select_fpr_thresholds_on_validation()` — deployment
  thresholds for FPR budgets {0.5, 1, 2, 5}% frozen as the
  (1−f)-quantile of **validation negative scores** (no test
  information); `frozen_operating_point_metrics()` reports realised
  test FPR / recall / precision / alert rate in the single test pass.
- Numerically validated under a 25× prevalence mismatch (realised FPR
  within 0.3% of target) — see tests/test_pipeline_integrity.py.
- Wired into `main.py` (run point 5c); `results.json` schema extended
  with `fpr_thresholds_selected_on: "validation (negative quantile)"`.
- **Executed in the independent re-execution on the real corpus:** the
  frozen thresholds realise 28.2 / 37.0 / 50.4 / 72.1% test FPR for
  FedProx against the 0.5 / 1 / 2 / 5% targets (XGBoost best at 5.4 /
  12.1 / 20.1 / 43.1%). The negative-score distribution shifts with the
  prior, so the (1−f)-quantile transfer fails jointly with probability
  calibration — now reported quantitatively in the manuscript
  (Sec. 6.4) and consistent with the paper's threshold-transfer-failure
  narrative. The synthetic-test 0.3% result held because that test's
  val/test score distributions were matched; the real corpus's 26×
  prevalence shift moves the negative quantile as well.
- The natural-prevalence validation *split* mode is implemented; using
  it as the default substrate requires the raw benchmark's untouched
  partition structure, which is queued (re-run item 2). Manuscript
  Sec. 4.7 describes the protocol exactly as implemented.

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
**SHIPPED → EXECUTED (real-data arm comparison, v3.3).**
- Manuscript Sec. 6.4 now opens by naming prior shift "a first-class
  methodological problem … the central obstacle between the current
  results and deployment-grade probabilities".
- The honest numbers are in the paper: FedProx calibrated Brier 0.264 /
  ECE 0.496 — **worse than climatology** (0.239 / 0.470). The previous
  "posterior is usable" sentence is gone.
- `evaluation.py` ships `PriorShiftCorrection`, `PlattScaling`,
  `IsotonicCalibration`, `TemperatureScaling`, `IdentityCalibration`,
  `select_calibration()`; `experiments/run_calibration_comparison.py`
  evaluates all arms on validation and the single test pass.
- **Executed in the independent re-execution against the trained
  frozen-protocol models** (v3.3, committed as
  `outputs/calibration_comparison.json`, mode: real): selection by
  minimum validation Brier picks **isotonic for all three models**, and
  for both neural models that is the **worst arm on test** (FedProx
  test Brier 0.541 vs 0.264 raw). No arm rescues the neural posteriors;
  only XGBoost benefits (Platt 0.063→0.053 Brier). Isotonic's ties
  destroy ranking resolution (PR-AUC 0.307→0.182 for FedProx). The
  manuscript (v3.3) reports this as a closed negative finding
  (Table 6) — the calibration question is no longer open, and the
  answer is that no validation-fit decision layer survives the 26×
  prior shift. Natural-prevalence validation (raw benchmark) remains
  the queued remedy.

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
**SHIPPED → EXECUTED (v3.3).**
- `evaluate_clients.py` has an untouched-holdout mode: each client
  reserves a fraction that *no* federated model trains on, so
  Local-vs-Global become clean generalisation estimates.
- **Executed** as a dedicated re-run
  (`experiments/run_client_holdout.py`, committed as
  `outputs/client_holdout_eval.json`): every shard split 80/20 before
  training, FedAvg + FedProx retrained on the 80% portions (frozen
  config, 50 rounds), local models trained on the same portions, all
  evaluated once on the untouched holdouts. Results: mean PR-AUC
  **local 0.983 / FedAvg 0.866 / FedProx 0.988**; FedProx within
  0.001–0.010 of local on the four largest clients and clearly better
  on the two smallest (+0.027, +0.019); FedAvg below local on every
  client. Figure 6 is regenerated from this artefact and Sec. 6.7
  (v3.3) reports the clean-protocol numbers; the legacy
  within-shard-slice values are kept in the text only as the
  superseded, optimistically biased variant.

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

## C. Re-run programme — status after the v3.3 executions

Implemented, unit-tested. **Items 2, 3, 5, 6, and 7 were executed in
the v3.3 independent re-executions** (see M2, #6, #14, #15, M3
above); the remaining items are:

1. Region-disjoint splits from raw SWAN-SF metadata (needs HARP/AR IDs)
2. Natural-prevalence validation as default substrate (needs the raw
   benchmark's untouched partition structure; frozen-FPR *report*
   executed — see M2)
3. ~~Calibration-comparison harness~~ **EXECUTED** (real mode, v3.3)
4. Event-level evaluation (needs event keys + timestamps)
5. ~~Untouched-holdout client evaluation~~ **EXECUTED** (v3.3)
6. ~~Budget-matched centralized-vs-federated training~~ **EXECUTED**
   (v3.3 second execution; `outputs/budget_matched.json` + runner
   `experiments/run_budget_matched.py`: 500-epoch cap, two early-stop
   variants — (a) fires at epoch 13 and reproduces the headline
   0.888/0.153 exactly, (b) FL-style fixed budget reaches 0.917/0.219
   vs FedProx 0.954/0.307; the gap survives budget parity and the
   shipped reference was undertrained, so the confound ran *against*
   the federated arm)
7. ~~Sweep-winner promotion (α=5.0)~~ **EXECUTED** (v3.3 second
   execution; `outputs/alpha5_promotion.json` + runner
   `experiments/run_alpha_promotion.py`: full 50-round frozen protocol
   — FedProx 0.955/0.296 ≈ headline 0.954/0.307, promotion REFUTED,
   α=1.0 retained; FedAvg completes without collapse at α=5.0
   (0.915/0.201), isolating the headline divergence as a
   heterogeneity-severity effect)
8. Federated LSTM and SCAFFOLD evaluation
9. Raw unbalanced SWAN-SF experiment programme

Items 1, 2, and 4 are preconditions for quoting any number as
operational performance rather than benchmark result.

---

## D. v3.3 addendum — independent re-execution (2026-09-29)

The complete headline pipeline was re-executed from scratch in a clean
CPU-only environment using only public artefacts: the Cleaned SWAN-SF
release downloaded from the dataset repository's official distribution,
all 20 partition files SHA-256-verified byte-identical to
`data_manifest/manifest.json`, then the frozen protocol end-to-end
(data load, immutable splits, Dirichlet partition, pooled baselines,
FedAvg + FedProx to 50 rounds, validation-fit calibration/thresholds,
single-pass test evaluation, figures, SHAP). Every test-set metric of
all six models reproduced **exactly** (all 8 metrics × 6 models match
the frozen artefacts at three decimals; the FedAvg divergence
signature reproduces round-for-round). This upgrades the paper's
reproducibility claim from “artefacts are available” to “numbers are
re-derivable from public inputs”, reported in Sec. 5.6 and the v3.3
abstract. The same session executed the previously queued calibration
comparison (#6), untouched-holdout evaluation (#14), and frozen-FPR
operating-point report (M2); the multiseed, sweep, and ablation studies
were not re-executed and remain the owner-side committed artefacts.

---

## E. Second-execution addendum (2026-09-30)

A second independent CPU-only session re-verified the pipeline
reproduction (headline metrics again identical to three decimals,
figures byte-identical) and completed the two re-run items the first
execution left queued:

- **Budget-matched training (#15 / item 6) — EXECUTED.**
  `outputs/budget_matched.json`, runner
  `experiments/run_budget_matched.py` (epoch-level resumable, phase
  -cache reusing). Variant (a): 500-epoch cap with the shipped early
  stopping — fires at **epoch 13**, reproducing the headline numbers
  exactly (0.888/0.153): the shipped centralised reference was
  *undertrained*, i.e. the budget confound ran against the federated
  arm. Variant (b): FL-style fixed 500-epoch budget (no early stop,
  best-validation checkpoint) — reaches **0.917/0.219**, still below
  FedProx's 0.954/0.307 on threshold-free metrics: **the headline gap
  survives budget parity** and is a genuine federation effect, not a
  budget artifact.
- **Sweep-winner promotion (M3 / item 7) — EXECUTED, REFUTED.**
  `outputs/alpha5_promotion.json`, runner
  `experiments/run_alpha_promotion.py` (round-resumable). Full
  50-round frozen protocol at α=5.0: FedProx **0.955/0.296**,
  indistinguishable from the α=1.0 headline (0.954/0.307) — the
  15-round sweep optimum does not transfer to the study regime, and
  **α=1.0 is retained**. Incidental finding: plain FedAvg completes
  all 50 rounds without collapsing at α=5.0 (0.915/0.201), isolating
  the headline FedAvg divergence as a heterogeneity-severity effect
  rather than an intrinsic failure of plain averaging.

Both runners are committed and deterministic; the manuscript (Secs.
5.3, 5.5, 8) folds these results in, the conclusion's claims taxonomy
upgrades the budget confound from "supported with caveats" to
resolved, and the appendix v3.3 row is extended accordingly.

---

## F. Third-execution addendum: raw-metadata provenance audit and the leakage-free fold (2026-09-30, v3.4)

The dependency that blocked queued items 1, 2, and 4 (raw SWAN-SF
HARP/event metadata) was resolved by downloading the raw benchmark
itself (Harvard Dataverse, doi:10.7910/DVN/EBCFKM, 6.5 GB + 34 MB
addenda with the GOES flare lists) and aligning every cleaned window
to its raw instance (provenance/ scripts; matcher: argmax/argmin
position invariance, 48 normalization-invariant keys per window; test
verification 98.7-100.0% with 100.00% label agreement). The alignment
produced an audit finding that supersedes the framing of several
review items:

**Finding 1 — the cleaned export's train/test pairing shares instances.**
The test export of partition p contains ALL raw instances of p; the
training export of p is a RUS-Tomek-TimeGAN rebalanced subset of the
same instances. 56,005/56,006 verified training windows are also test
windows, and **100% of the flaring test instances (6,234) are in the
training pool**. The cleaned release's own paper
(10.3847/1538-4365/ad7c4a) prescribes temporally-preceding
train/test partition combinations; the shipped pipeline paired all
five training exports with the same five test exports. The published
in-partition headline numbers therefore partially measure memorised
data — most severely the minority class. This is the answer the
review's insistence on raw metadata was asking for, and it was not
detectable by the pipeline-level runtime leakage audit.

**Finding 2 — TimeGAN synthetic share.** 85.7-90.0% of training
positive windows have no raw counterpart (TimeGAN-generated); ~0% of
negatives. The balanced substrate the field trains on is majority
synthetic in its positive class.

**Item-by-item status after this execution:**

1. **Region-disjoint splits (R7) — RESOLVED BY A STRONGER RESULT.**
   No HARP/region id spans two partitions, so the
   temporally-preceding fold executed below is automatically
   region-disjoint at the train/test boundary. The
   region-disjoint *validation* generator remains implemented and
   unit-tested for within-fold studies; the metadata to drive it
   (per-window region ids) is now committed
   (provenance/train_meta_slim.csv.gz).
2. **Natural-prevalence validation (R4/M2) — EXECUTED at the test
   boundary.** The leakage-free fold evaluates at the natural 1.31%
   prevalence of partition 5 with the frozen-FPR operating points
   (realised FPRs 5.5-49.4% at the 2% target — the threshold-transfer
   failure persists leakage-free). Natural-prevalence *training*
   substrate (item 9) remains queued.
3. ~~Calibration comparison~~ (executed v3.3) — unchanged; the fold
   confirms no calibrator rescues the neural arms.
4. **Event-level evaluation (R13/R19) — EXECUTED.** 65 physical M/X
   events on the fold's test partition, event keys from the raw
   filenames + addenda GOES peak list (65/65 matched); detection
   98.5-100%; median lead-to-peak 23.4 h; XGBoost's false-alarm
   burden 1.7 windows/day vs 8.6-12.7 for the neural arms; ~70%
   duplicate alerts after 1-h cooldown. `outputs/event_level_p5.json`.
5. ~~Untouched holdouts~~ (executed v3.3) — unchanged.
6. ~~Budget-matched training~~ (executed, second addendum) — note the
   in-partition framing: the parity result stands as protocol-stability
   evidence; the gap it preserved is an in-partition artifact (see the
   fold).
7. ~~α=5.0 promotion~~ (executed, refuted) — unchanged.
8. Federated LSTM/SCAFFOLD — **SCAFFOLD EXECUTED (v3.5)** on the
   raw substrate (identical frozen budget as FedProx/FedAvg): fails
   outright at ROC 0.768 / PR 0.057, Brier 0.137, event detection
   54/65 at 6.4 false-alarm windows per day. The LSTM arm alone
   remains queued (owner GPU; ~8–15 h on CPU).
9. Raw unbalanced programme — **EXECUTED (v3.5)**
   (`experiments/raw_substrate.py` builds the substrate,
   `experiments/run_raw_substrate.py` retrains the frozen protocol,
   `experiments/run_event_level_raw.py` provides event metrics;
   artefacts `outputs/raw_substrate_eval.json`,
   `outputs/event_level_raw_p5.json`,
   `outputs/raw_substrate_verification.json`). FPCKNN imputation and
   LSBZM normalisation are reproduced in-pipeline from the release
   paper's description with train-only parameters; the reproduction
   is verified against the released export on the audit-matched P5
   windows (100% match; observed-nonzero median Spearman ρ=0.898,
   min 0.842; the release is itself an exact monotone image of the
   raw values, ρ=1.000; imputed positions are method-dependent,
   ρ=0.313; the release re-imputes the R_VALUE zero mass (60.9% of
   that column) as if missing, this reproduction preserves zeros).
   Headline: train raw P1–4 at natural 2.05% prevalence -> test raw
   P5 (1.31%): LR 0.978/0.448, XGB 0.974/0.329, central MLP
   0.971/0.445, FedAvg 0.875/0.056 (0/65 events detected), FedProx
   0.930/0.149, SCAFFOLD 0.768/0.057. Centralised arms are
   substrate-robust; every federated arm degrades sharply; XGB's
   minority precision on the cleaned fold was inflated by the
   TimeGAN synthetic positives (PR 0.457 -> 0.329). Remaining:
   fold/seed replication and the natural-prevalence SMOTE arm.

**Fourth-execution addendum: the raw-substrate retraining (v3.5).**
Same frozen protocol, substrate = raw SWAN-SF with in-pipeline
FPCKNN/LSBZM reproduction (details in item 9 above and paper Sec 6.2):

| Model | fold (cleaned) ROC/PR | raw substrate ROC/PR | events | FA/day |
|---|---|---|---|---|
| Logistic regression | 0.978 / 0.455 | 0.978 / 0.448 | 51/65 | 0.3 |
| XGBoost | 0.971 / 0.457 | 0.974 / 0.329 | 40/65 | 0.2 |
| Centralised MLP | 0.973 / 0.382 | 0.971 / 0.445 | 52/65 | 0.3 |
| FedAvg | 0.906 / 0.370 | 0.875 / 0.056 | 0/65 | 0.0 |
| FedProx | 0.976 / 0.308 | 0.930 / 0.149 | 48/65 | 1.3 |
| SCAFFOLD | — | 0.768 / 0.057 | 54/65 | 6.4 |

Reading: the federated arms' leakage-free standing was still a
property of the balanced/synthetic training substrate; at natural
prevalence the focal-loss clients cannot learn the 2% minority
through 6-client Dirichlet heterogeneity, and the best federated arm
is dominated by every centralised baseline. The paper (v3.5) re-tiers
all claims across the three substrates.

**The leakage-free fold (runner
`experiments/run_partition_disjoint.py`, artefacts
`outputs/partition_disjoint_eval.json` + `event_level_p5.json`):**
train P1-4 (65,406/12,459 train/val) -> single-pass test P5 (75,365
windows, 1.31% prevalence), everything else frozen (α=1.0, 6 clients,
50 rounds, seed 42, Fed-Focal, prior-shift calibration, val-frozen
thresholds).

| Model | in-partition ROC/PR | leakage-free ROC/PR |
|---|---|---|
| Logistic regression | 0.818 / 0.114 | 0.978 / 0.455 |
| XGBoost | 0.977 / 0.493 | 0.971 / 0.457 |
| Centralised MLP | 0.888 / 0.153 | 0.973 / 0.382 |
| FedAvg | 0.575 / 0.199 | 0.906 / 0.370 |
| FedProx | 0.954 / 0.307 | 0.976 / 0.308 |

Reading: all arms generalise (the task is easier than the
in-partition protocol suggested); the central-vs-federated gap
disappears; FedProx retains checkpoint-independent ROC stabilisation
over FedAvg (0.976 vs 0.906 — FedAvg survives only via its round-35
validation checkpoint); the PR ordering between the FL arms reverses;
the calibration/threshold-transfer failures persist; only XGBoost's
posteriors are deployment-usable. The manuscript (v3.4) re-tiers all
claims accordingly and discloses the in-partition results as
protocol-stability evidence.
