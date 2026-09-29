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
