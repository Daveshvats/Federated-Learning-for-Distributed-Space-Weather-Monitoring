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
| 2 | FedProx outperforms FedAvg under simulated non-IID | PARTIAL | Single seed, FedProx AUC 0.930 vs FedAvg 0.664; but DA-FL aggregation was used in BOTH arms and shards were not disjoint | Fix confounds (plain FedAvg arm, disjoint shards), 5 seeds + 95% CI |
| 3 | Solar flare prediction on SWAN-SF | DEMONSTRATED | ROC/PR metrics on held-out test set | Validation-based thresholding, PR-AUC + recall@FPR |
| 4 | Operational GIC prediction | NOT DEMONSTRATED | No CME propagation / IMF / dB/dt / geoelectric model | Scope: upstream flare component only. See `docs/GIC_BOUNDARY.md` |
| 5 | SCADA integration | NOT DEMONSTRATED (architecture only) | Text description in manuscript; no SCADA experiment | Label as proposed architecture |
| 6 | Privacy-preserving / cryptographic privacy | NOT DEMONSTRATED | FL ≠ private; gradient inversion risk acknowledged, no secure aggregation or DP | Terminology: "data-locality-preserving". See `privacy_analysis/THREAT_MODEL.md` |
| 7 | Real inter-agency federation | NOT DEMONSTRATED | 6 clients simulated on one machine, public benchmark | Label as simulation. Sovereignty premise must be qualified |
| 8 | Fed-Focal loss helps under FL imbalance | PARTIAL | FedFocalLoss used, but no ablation vs BCE/weighted BCE/focal | Ablation matrix (`experiments/run_ablations.py`) |
| 9 | SHAP identifies physically meaningful predictors | PARTIAL | SHAP computed on centralized XGBoost surrogate only; feature names degraded to `feature_N` in figures | Fix name propagation; extend SHAP to FL global model |
| 10 | SMOTE per-client balancing | NOT DEMONSTRATED | `apply_smote` is **dead code** — defined in `data_preparation.py` but never called by `main.py` or `federated_learning.py` | Wire into ablation matrix; remove unsupported claims |
| 11 | SCAFFOLD third baseline | PARTIAL | Implemented with NaN fixes, but disabled in config; last run excluded it | Enable and validate in future runs |
| 12 | "Converges within 50 rounds, operationally viable" | PARTIAL | Convergence curves exist but were plotted on the TEST set | Re-plot from validation monitoring |

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
