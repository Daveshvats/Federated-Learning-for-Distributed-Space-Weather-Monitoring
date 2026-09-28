# SF-9 — Dependency-Ordered Revision Plan

> Order matters: a data-validity bug invalidates every downstream experiment.
> Fix in this sequence; do not polish prose or figures until Gate 2 passes.

```
Gate 1  DATA VALIDITY          schema, disjoint partitions, leakage audit, provenance
   ↓
Gate 2  STATISTICAL VALIDITY   train/val/test, threshold isolation, calibration,
   ↓                         multi-seed + CI
Gate 3  ML VALIDITY            centralized MLP, plain FedAvg, FedProx, ablations,
   ↓                         client-level evaluation
Gate 4  PHYSICAL VALIDITY      physical/geographic partition, FL SHAP, flare
   ↓                         definition, CME/GIC boundary
Gate 5  OPERATIONAL/SECURITY   threat model, secure aggregation, communication,
                             failure modes, SCADA simulation
   ↓
NASA-facing resubmission
```

## Stage map (16 stages → concrete deliverables on this branch)

| Stage | Deliverable on `improvements` | Status |
|-------|------------------------------|--------|
| 0. Freeze claims | `audit/CLAIMS.md`, `audit/BUG_REGISTER.md` | done |
| 1. Dataset/provenance audit | `data_manifest/generate_manifest.py` + `data_manifest/README.md` | done (code) |
| 2. Partitioning + leakage | fixed `partition_clients.py`; `leakage_audit/audit_leakage.py` + smoke test | done (code) |
| 3. Train/val/test protocol | `data_preparation.preprocess` returns immutable splits; threshold/calibration/checkpoint/monitoring moved to validation | done (code) |
| 4. Config/pipeline freeze | `config.py` consolidated (seed, aggregation strategy, SMOTE flag, calibration method, RUN_ID); `results.json` + `run_manifest.json` per run | done (code) |
| 5. Centralized baselines | climatology + LR + centralized MLP + XGBoost (spw from train) in `centralized_baseline.py` | done (code) |
| 6. FL baselines + ablations | plain FedAvg default; `experiments/run_ablations.py` (loss × SMOTE × aggregation × algorithm) | done (code) |
| 7. Hyperparameter tuning | `experiments/run_sweep.py` (μ × α grid, validation-selected) | done (code) |
| 8. Calibration + thresholding | `calibration.py` (prior-shift/Platt/isotonic/temperature; ECE, Brier, recall@FPR) | done (code) |
| 9. Multi-seed statistics | `experiments/run_multiseed.py` (5 seeds, mean/std/95% CI, paired tests) | done (code) |
| 10. Client-level evaluation | `evaluate_clients.py` (local-only vs FedAvg vs FedProx per client) | done (code) |
| 11. Physical realism | `partition_clients.partition_data_geographic` (observatory-informed option) | done (code) |
| 12. SMOTE physical plausibility | `experiments/analyze_smote_validity.py` (distribution + correlation comparison) | done (code) |
| 13. Interpretability validation | `interpretability.py` (SHAP on XGBoost AND FL global model) | done (code) |
| 14. Privacy/security | `privacy_analysis/THREAT_MODEL.md`; `secure_aggregation.py` scaffold (masked-secagg protocol) | done (docs/scaffold) |
| 15. Communication/deployment | `communication_cost.py` (bytes/round measurement + dropout simulation) | done (code) |
| 16. Claims, figures, audit | `README.md`, `limitations/LIMITATIONS.md`, revised `ABSTRACT.md`, honest paper claims | done (docs) |

**Stages 5-16 produce code + docs ready to execute; the experiments themselves
require the SWAN-SF data and GPU and are the owner's next action** (commands in
`README.md`). No numbers are fabricated on this branch: every metric table in
the revised manuscript is generated from `results.json` artifacts.

## The five hard gates — pass criteria

1. **Data validity**: `python tests/test_pipeline_integrity.py` → all green
   (disjointness = 0 overlap, coverage = 100%, leakage detector catches seeded
   leakage, split contract immutable).
2. **Statistical validity**: `run_multiseed.py` output contains mean ± std ± CI
   for every headline metric; threshold & calibration chosen on validation only.
3. **ML validity**: ablation matrix isolates {FedAvg, DA-FL} × {BCE, focal,
   Fed-Focal} × {no-SMOTE, SMOTE} × {μ=0, μ>0}; centralized MLP exists.
4. **Physical validity**: geographic partition option runs; SHAP names show
   physical features; GIC chain explicitly bounded (see `docs/GIC_BOUNDARY.md`).
5. **Operational/security**: threat model documented; secagg scaffold compiles;
   communication cost measured per round.
