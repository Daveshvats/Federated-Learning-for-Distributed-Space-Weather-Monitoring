# SF-9 — Claim Status Table (Scientific Scope Freeze)

> **Rule:** every manuscript claim must map to evidence produced by code in this
> repository. Anything without evidence is *Future Work*, *Proposed Architecture*,
> or *Hypothesis* — never a Result.
>
> Status levels: `DEMONSTRATED` (reproducible evidence exists),
> `PARTIAL` (evidence exists but with a known caveat),
> `NOT DEMONSTRATED` (no evidence yet).

| # | Claim | Status | Evidence / Gap | Action required |
|---|-------|--------|----------------|-----------------|
| 1 | FL can train a solar-flare classifier across simulated cross-silo, non-IID shards without pooling raw data | DEMONSTRATED (with caveats) | FedAvg/FedProx runs on SWAN-SF, 6 clients, Dirichlet α=1.0 | Re-run on `improvements` code after partition fix; multi-seed |
| 2 | FedProx outperforms FedAvg under simulated non-IID | DEMONSTRATED (scoped) | 5/5 seeds, one-sided Wilcoxon p=0.031, 2.7x narrower AUC CI; ablation shows DA-FL also stabilises FedAvg (best cell 0.963 > FedProx 0.954), so claim is "most consistent stabiliser", not "load-bearing" | Scope wording frozen in v3.2 paper; alpha=5.0 promotion queued |
| 3 | Solar flare prediction on SWAN-SF | DEMONSTRATED | ROC/PR metrics on held-out test set | Validation-based thresholding, PR-AUC + recall@FPR |
| 4 | Operational GIC prediction | NOT DEMONSTRATED | No CME propagation / IMF / dB/dt / geoelectric model | Scope: upstream flare component only. See `docs/GIC_BOUNDARY.md` |
| 5 | SCADA integration | NOT DEMONSTRATED (architecture only) | Text description in manuscript; no SCADA experiment | Label as proposed architecture |
| 6 | Privacy-preserving / cryptographic privacy | NOT DEMONSTRATED | FL ≠ private; gradient inversion risk acknowledged, no secure aggregation or DP | Terminology: "data-locality-preserving". See `privacy_analysis/THREAT_MODEL.md` |
| 7 | Real inter-agency federation | NOT DEMONSTRATED | 6 clients simulated on one machine, public benchmark | Label as simulation. Sovereignty premise must be qualified |
| 8 | Fed-Focal loss helps under FL imbalance | PARTIAL | FedFocalLoss used, but no ablation vs BCE/weighted BCE/focal | Ablation matrix (`experiments/run_ablations.py`) |
| 9 | SHAP identifies physically meaningful predictors | PARTIAL (improved) | Real statistic-parameter names preserved end-to-end; FL model attributed via DeepSHAP; feature-level agreement moderate (rho=0.334, top-15 overlap 53%), physical-group agreement strong | No per-feature claims; group-level only |
| 10 | SMOTE per-client balancing | NOT DEMONSTRATED | `apply_smote` is **dead code** — defined in `data_preparation.py` but never called by `main.py` or `federated_learning.py` | Wire into ablation matrix; remove unsupported claims |
| 11 | SCAFFOLD third baseline | PARTIAL | Implemented with NaN fixes, but disabled in config; last run excluded it | Enable and validate in future runs |
| 12 | "Converges within 50 rounds" | DEMONSTRATED | Validation-monitored convergence (v3.0.1 frozen protocol); FedProx val AUC 0.99+/-0.01 for all 50 rounds | "Operationally viable" not claimed |

## Frozen scope statement (use in manuscript)

> This work demonstrates **federated learning as a feasible training paradigm for
> solar-flare prediction on the SWAN-SF benchmark under simulated cross-silo,
> non-IID partitioning**, and quantifies the accuracy cost of federation relative
> to centralized baselines. It does **not** demonstrate GIC prediction, SCADA
> deployment, cryptographic privacy, or real inter-agency operation.

## Terminology corrections

| Before (overclaim) | After (accurate) |
|---|---|
| "privacy-preserving" | "data-locality-preserving" (until secure aggregation / DP implemented) |
| "centralized upper bound" (XGBoost) | "centralized XGBoost reference baseline" |
| "six international observatories" | "six simulated regional clients" |
| "operational warning system" | "flare-probability research prototype" |
| "NASA/ESA/JAXA cannot pool data" | "agencies may face governance, locality, and institutional constraints on data sharing" (NASA data is substantially public) |

---

## v3.1 review-2 claim updates (2026-09-29)

New claim rows introduced by the second external review
(see `docs/REVIEW2_RESPONSE.md` for the full mapping):

| # | Claim | Status | Evidence / Gap |
|---|-------|--------|----------------|
| 13 | Deployment thresholds at fixed FPR budgets selected without test information | IMPLEMENTED + QUEUED | `select_fpr_thresholds_on_validation` (negative-quantile), numerically validated under 25x prevalence mismatch; full-data report queued |
| 14 | Region-disjoint train/val/test splits | IMPLEMENTED + QUEUED | `region_disjoint_val_split` unit-tested; cleaned export lacks HARP/AR IDs, so not executable |
| 15 | Centralized-MLP vs FL comparison is budget-controlled | PARTIAL -> DOCUMENTED CONFOUND | `training_budget_report` shipped; budgets NOT matched (30 vs 500 effective passes); paper states confound; matched re-run queued |
| 16 | Calibration resolved at natural prevalence | NOT DEMONSTRATED | FedProx Brier 0.264 / ECE 0.496 worse than climatology (0.239 / 0.470); comparison harness shipped, full run queued |
| 17 | Event-level operational metrics | IMPLEMENTED + QUEUED | `run_event_level.py` (detection rate, lead time, duplicate alerts); needs event keys from raw metadata |
| 18 | Client-level Local-vs-Global on untouched holdouts | IMPLEMENTED + QUEUED | untouched-holdout mode in `evaluate_clients.py`; needs retraining |
| 19 | Documented literature-search protocol for the "first FL flare paper" claim | DEMONSTRATED | Sec. 2.4 + `docs/lit_search/` (4 queries, 2026-09-29, 30 records, 0 in scope) |
| 20 | Threat model stated in manuscript | DEMONSTRATED (as documentation) | Paper Table `tab:threat`; poisoning/backdoor explicitly "not addressed" |

Terminology freeze (v3.2 paper): "data-locality-preserving" (never
"privacy-preserving"), "simulated regional clients A-F" (never agency
names), "flare-precursor alert component" (never "GIC early warning"),
"M/X windows" for window-level statistics (never "events").

---

## v3.3 executed-claim updates (2026-09-29, independent re-execution)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 21 | Headline numbers are reproducible from public inputs | DEMONSTRATED | Independent re-execution: dataset SHA-256-verified against frozen manifest, frozen protocol re-run on CPU (seed 42); all 8 metrics x 6 models match exactly at 3 decimals; FedAvg divergence signature reproduces round-for-round |
| 22 | Calibration arms resolve the prior-shift problem | REFUTED (closed negative) | `outputs/calibration_comparison.json` (real mode): validation-Brier selection picks isotonic for all 3 models — the worst arm on test for both neural models (FedProx 0.541 vs 0.264 raw); no arm rescues neural posteriors; isotonic ties destroy ranking (PR-AUC 0.307 -> 0.182); only XGBoost benefits (Platt) |
| 23 | Client-level Local-vs-Global on untouched holdouts | DEMONSTRATED | `outputs/client_holdout_eval.json`: mean PR-AUC local 0.983 / FedAvg 0.866 / FedProx 0.988; FedProx within 0.001-0.010 of local on 4 largest clients, +0.027/+0.019 on the two smallest; FedAvg below local on every client |
| 24 | Validation-frozen FPR thresholds transfer to deployment | REFUTED (quantified) | Frozen at 0.5/1/2/5% targets -> realised test FPR 28.2/37.0/50.4/72.1% (FedProx), 20.1% at 2% target (XGBoost); no calibration arm materially improves; reported in paper Sec. 6.4 |

Updates to earlier rows: row 13 -> EXECUTED (report; negative result
above); row 16 -> REFUTED-at-balanced-validation (harness executed,
natural-prevalence validation still queued); row 18 -> DEMONSTRATED.

---

## v3.3 second-execution claim updates (2026-09-30)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 25 | FedProx-vs-centralised gap is not a budget artifact | DEMONSTRATED | `outputs/budget_matched.json` (runner `experiments/run_budget_matched.py`): shipped reference early-stops at epoch 13 (undertrained — confound ran against the federated arm); FL-style fixed 500-epoch budget reaches 0.917/0.219 vs FedProx 0.954/0.307 — gap survives budget parity |
| 26 | Sweep winner (alpha=5.0) should be promoted to headline | REFUTED (negative result) | `outputs/alpha5_promotion.json` (runner `experiments/run_alpha_promotion.py`): full 50-round protocol at alpha=5.0 gives 0.955/0.296 vs headline 0.954/0.307 — no transfer; alpha=1.0 retained. FedAvg non-collapse at alpha=5.0 (0.915) shows headline divergence is a heterogeneity-severity effect |

---

## v3.4 provenance-audit claim updates (2026-09-30)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 27 | Cleaned-SWANSF train/test exports are instance-disjoint | **REFUTED (headline audit finding)** | `outputs/dataset_structure_audit.json` + `provenance/`: every cleaned window aligned to its raw instance (argmax/argmin-invariant matching, test verification 98.7-100.0%, label agreement 100.00%); test export of partition p = ALL raw instances of p; train export = RUS-Tomek-TimeGAN subset of the same instances; 56,005/56,006 verified train windows are also test windows; 100% of flaring test instances (6,234) are training windows |
| 28 | Training positives are real observations | REFUTED (85.7-90.0% synthetic) | TimeGAN-generated windows have no raw counterpart; synthetic share of train positives: P1-P4 85.7-85.8%, P5 90.0%; negatives ~0% |
| 29 | Federated-vs-centralised gap survives a leakage-free protocol | **REFUTED** | `outputs/partition_disjoint_eval.json` (train P1-4 -> test P5, frozen protocol): FedProx 0.976/0.308 vs architecture-matched centralised MLP 0.973/0.382 — ROC within 0.003, PR reversed. The in-partition gap (0.954 vs 0.888) was a protocol-pairing artifact, as was its budget-parity survival (row 25) |
| 30 | FedProx stabilises the federation vs plain FedAvg | DEMONSTRATED (leakage-free) | Fold: FedProx 0.976 vs FedAvg 0.906 ROC-AUC; FedAvg survives only via its round-35 validation checkpoint (val F1 collapsed to 0 by round 40) — checkpoint-lottery dynamics; FedProx checkpoint-independent. PR reverses (FedAvg 0.370 > FedProx 0.308), so the v3.3 "5/5 seeds on PR-AUC" claim is in-partition only |
| 31 | Event-level detection is the deployability boundary | REFUTED (calibration is) | `outputs/event_level_p5.json`: 65 M/X events, detection 98.5-100%, median lead-to-peak 23.4h; XGBoost false-alarm burden 1.7 windows/day vs 8.6-12.7 neural arms (~7x) — mirrors the window-level calibration gap |
| 32 | Threshold-transfer failure is an overlap artifact | REFUTED (persists leakage-free) | Fold realised FPR at frozen 2% target: XGBoost 15.3%, FedProx 49.4%, central MLP 88.2% — the balanced-training/natural-test interface, not instance overlap, is the cause |

Row updates: rows 25/26 remain valid as **in-partition
protocol-stability evidence** (disclosed); row 13 (recall@FPR
window-level caveat) now carries event-level corroboration; row 16's
natural-prevalence evaluation is executed at the fold's test boundary.

---

## v3.5 raw-substrate claim updates (2026-09-30)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 33 | Centralised baselines generalise to the raw unbalanced benchmark | **DEMONSTRATED** | `outputs/raw_substrate_eval.json` (frozen protocol retrained on raw P1-4, natural 2.05% prevalence, FPCKNN/LSBZM reproduced in-pipeline, single-pass test raw P5): LR 0.978/0.448 (unchanged to 3 decimals vs the cleaned fold), central MLP 0.971/0.445 (PR up from 0.382), XGB 0.974/0.329 |
| 34 | Federated arms' leakage-free standing transfers to the raw benchmark | **REFUTED** | Same run: FedProx 0.930/0.149 (was 0.976/0.308), FedAvg 0.875/0.056 (was 0.906/0.370) with 0/65 events detected, SCAFFOLD 0.768/0.057 fails outright (Brier 0.137). Best federated arm dominated by every centralised baseline; the standing was a property of the balanced synthetic substrate |
| 35 | XGBoost's cleaned-fold PR-AUC reflects real minority-class learning | REFUTED (partially synthetic) | Raw-substrate XGB PR-AUC 0.329 vs 0.457 on the cleaned fold: the TimeGAN synthetic positives inflated minority precision by ~0.13 PR-AUC |
| 36 | SCAFFOLD is a viable stabiliser for this problem | REFUTED (first real-substrate execution) | `outputs/raw_substrate_eval.json` + `outputs/event_level_raw_p5.json`: 0.768/0.057, Brier 0.137, event detection 54/65 at 6.4 false-alarm windows/day (~20x the centralised arms' burden) |
| 37 | The FPCKNN/LSBZM preprocessing reproduction is rank-faithful | DEMONSTRATED (bounded) | `outputs/raw_substrate_verification.json`: on the audit-matched P5 windows (100% match), observed-nonzero Spearman vs the released export: median 0.898, min 0.842; the release is itself an exact monotone image of raw (rho=1.000), residual gap = float16-storage tie dilution; imputed positions method-dependent (median rho 0.313); the release re-imputes the R_VALUE zero mass (60.9% of the column) as if missing, this reproduction preserves zeros |

Row updates: rows 29-32 (leakage-free fold) now carry the raw-substrate
corroboration in `sec:res-raw`; the deployability boundary (row 31)
extends: on the raw substrate the centralised MLP's Brier 0.010 is
best of all arms and no federated arm is calibration-usable on any
substrate.

---

## v3.6 federated-LSTM claim updates (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 38 | The raw-substrate federated collapse is a property of federation | **REFUTED (architecture-conditional)** | `outputs/raw_lstm_eval.json` (owner GPU, RTX 4060, 2.6 h; frozen protocol, substrate identity asserted): FedAvg-LSTM 0.958/0.305 vs FedAvg-MLP 0.875/0.056 (+0.083/+0.249 from the encoder alone, identical shards/budgets/aggregation); FedProx-LSTM 0.970/0.404, above its own pooled comparator (0.962/0.314, budget-caveated: central capped at 30 epochs) and XGBoost's raw PR-AUC 0.329. Centrally the LSTM is the weaker architecture (0.962/0.314 vs 0.971/0.445) — the collapse is a property of the flattened-feature MLP arms, not of federation per se |
| 39 | FedProx-over-FedAvg stabilisation is a cleaned-substrate artifact | REFUTED (most replicated finding) | Stabilisation appears on every executed substrate-encoder combination: in-partition MLP 0.954 vs 0.575; cleaned fold 0.976 vs 0.906; raw MLP 0.930 vs 0.875; raw LSTM 0.970 vs 0.958 (both rank metrics, identical shards/budgets) |
| 40 | No federated arm is calibration-usable on any substrate | SOFTENED | The three federated sequence arms' Brier beat the climatology floor (0.0130) on the raw substrate (FedAvg 0.0112, FedProx 0.0106, SCAFFOLD 0.0120; every federated MLP arm at or above it) — the only federated results to do so — but FedProx-LSTM's ECE 0.020 is 4x the central LSTM's 0.005, SCAFFOLD's 0.035 is the worst of the sequence arms, and the centralised MLP's Brier 0.0104 remains best of all ten arms; the deployability boundary (calibration, not detection) stands |

Row updates: rows 34/35 (raw-substrate collapse) now carry the
architecture qualifier — the MLP arms collapse, the sequence arms do
not; row 30 (FedProx stabilisation) gains the raw-LSTM replication;
the conclusion's smart-grid message is re-worded (parity is
encoder-dependent; no federated arm exceeds the best centralised
baseline on any substrate).

## v3.7 event-level LSTM claim updates (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 41 | The sequence-encoder rescue is a window-level (ranking) phenomenon only | **REFUTED** | `outputs/event_level_raw_lstm_p5.json` (standalone CPU pass over the GPU run's stored test probs + audit metadata; 65/65 GOES peaks matched, labels asserted): FedAvg-LSTM detects 35/65 M/X events (53.8%, 0.2 FA windows/day) where its MLP counterpart detects 0/65; the rescue survives validation-frozen thresholds and 1-h cooldown, not just ranking metrics |
| 42 | FedProx-LSTM dominates FedProx-MLP at event level | **CONFIRMED** | 49/65 (75.4%) vs 48/65 (73.8%) detection at 0.4 vs 1.3 FA windows/day (3.5x lower burden); median lead 23.2 h (p10 7.7 h, p90 24.0 h) — the joint-longest on the substrate |
| 43 | No federated arm exceeds the best centralised baseline on any substrate (now including event level) | **CONFIRMED (extended)** | Raw P5 event-level frontier: LR 51/65 (78.5%) and central MLP 52/65 (80.0%) at 0.2-0.3 FA/day vs FedProx-LSTM 49/65 (75.4%) at 0.4; data-locality framing stands at window AND event level |

Row updates: the v3.6 architecture-conditional collapse (rows 34/35,
38) now carries the event-level quantifier — FedAvg-MLP 0/65 events
vs FedAvg-LSTM 35/65 on identical shards/protocol.

## v3.8 owner-GPU queue claim updates (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 44 | SCAFFOLD-LSTM, seed-43 replication, and the natural-prevalence SMOTE ablation on the raw substrate | STAGED (code-verified, not yet executed) | `experiments/run_gpu_queue.py` one-command queue + `run_federated_lstm.py --scaffold/--smote/--seed 43` (34 new checks, 176/176 total; smoke-verified on the real 3D caches on CPU incl. the fixed 3D SMOTE path and the `_aux` fallback). NO numeric claim about these arms may enter the paper until the owner-side run lands in `outputs/raw_lstm_{scaffold,seed43,smote}.json` |

## v3.8 executed claim updates (2026-10-01, second owner-GPU batch)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 44 | SCAFFOLD-LSTM, seed-43 replication, and the natural-prevalence SMOTE ablation on the raw substrate | **EXECUTED** (was STAGED) | One resumable owner-GPU queue (`experiments/run_gpu_queue.py`), 8.7 h (RTX 4060 Laptop; 1.5/3.9/3.2 h per step); all 6 artefacts committed (`outputs/raw_lstm_{scaffold,seed43,smote}.json` + event-level JSONs); validated on receipt by a 64-check audit (protocol fingerprints, substrate identity 214,888/40,932/75,365 @ 1.3136%, embedded frozen-counterpart bit-matches, 65/65 GOES peak matches, log cross-checks) |
| 45 | SCAFFOLD-LSTM recovers the hardest raw-substrate MLP failure | **CONFIRMED** | 0.970/0.294 vs MLP counterpart 0.768/0.057 (+0.202 ROC, +0.238 PR — the largest single-encoder swing on this substrate); largest-encoder-swing status also guards the four-arm federation matrix: central 0.962/0.314, FedAvg 0.958/0.305, FedProx 0.970/0.404, SCAFFOLD 0.970/0.294 |
| 46 | The raw-substrate sequence-arm rankings are seed-robust | **CONFIRMED** | seed 43 re-execution (reseeded init + Dirichlet shards, frozen val carve): ROC-AUC moves ≤ 0.005 on every arm (central 0.962→0.962, FedAvg 0.958→0.961, FedProx 0.970→0.965, SCAFFOLD 0.968); no collapse, FedProx ≥ FedAvg, event detection within a 7-point band (49→45, 55→50 of 65) |
| 47 | Per-client SMOTE at natural prevalence is a clean negative | **CONFIRMED** | ratio 0.25 minority:majority, 3D-aware: FedAvg 0.961/0.307 (natural 0.958/0.305 — unchanged; events 35→40); FedProx 0.946/0.288 vs 0.970/0.404 (−0.116 PR-AUC; events 49→37); completes the cleaned-substrate SMOTE null — the rescue is architectural, not class-balance-driven |
| 48 | Row 43's "no federated arm exceeds the centralised baseline" now requires an event-level qualifier | **QUALIFIED** | SCAFFOLD-LSTM posts the substrate's highest event detection rate, 55/65 (84.6%) at 0.41 FA windows/day — above LR 78.5% and central MLP 80.0% — but at ~2x their false-alarm burden, with window-level ranking still 0.008/0.154 below LR and the worst sequence-encoder calibration (ECE 0.035); the claim holds at window-level ranking and at matched operating budgets |

Row updates: rows 34/35/38/43 (v3.6/v3.7 collapse/rescue/fresh-hold
claims) now carry the v3.8 four-arm + two-seed quantifiers; the
paper's Sec 6.5/6.6, abstract, and conclusion are the narrative
counterparts of rows 44–48.
