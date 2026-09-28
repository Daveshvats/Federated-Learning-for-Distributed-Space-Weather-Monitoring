# paper/ — manuscript & audit artifacts

## Status (v3.1, 2026-09): REAL RESULTS EDITION ✅

| File | Status |
|---|---|
| `SF9_Federated_Space_Weather_Paper.pdf` + `main.tex` + `sections/` + `refs.bib` | **24 pp, regenerated with real frozen-protocol numbers** from the completed experiment programme (headline run, 24-cell ablation, 9-config sweep, 5-seed study, interpretability, communication). Every value is machine-generated from `outputs/*.json`. |
| `SF9_Research_Review_and_Code_Audit.pdf` + `review.tex` | Companion audit (rating 7/10, findings B1-B25) — the bug register that drove the `improvements` branch. Historical document: describes the v2.x state it audited. |
| `figures/` | All result figures regenerated from `outputs/results.json` et al. by the pipeline + `scripts/fig_analysis.py` (ablation/multiseed/client figures). `fig_architecture.png`, `fig_partition.png` are protocol-independent illustrations. |

## Key real results in this edition

- Test (331,185 windows, 1.88% prevalence, single pass): XGBoost AUC
  0.977 / PR 0.493; **FedProx 0.954 / 0.307**; Centralised MLP 0.888 /
  0.153; FedAvg 0.575 (diverged); LR 0.824; climatology 0.500
- FedProx **exceeds the same-architecture pooled MLP** — federation
  costs no ranking skill within the hypothesis class
- FedAvg training divergence (val AUC 0.99 → 0.05 after round ~20);
  FedProx stable on all 5 seeds (Wilcoxon p = 0.031)
- Threshold-transfer failure under 26× prevalence shift documented;
  recall-at-FPR operating points reported
- SMOTE arm inert (pre-balanced benchmark) — reported null
- Physical-group SHAP consistency (helicity, R-value, Lorentz force)

## Regenerating after future runs

1. `python main.py --no-lstm` → `outputs/results.json`, fresh figures
2. `python experiments/run_ablations.py` → `outputs/ablation_results.json`
3. `python experiments/run_sweep.py` → `outputs/sweep_results.json`
4. `python experiments/run_multiseed.py --seeds 5` → `outputs/multiseed_results.json`
5. `python experiments/run_interpretability.py` → `outputs/interpretability.json`
6. `python /path/to/scripts/fig_analysis.py` (ablation/multiseed/client figures)
7. `cd paper && tectonic main.tex`
8. Update `sections/*.tex` **from the JSON files only**; keep claim-status
   language aligned with `../audit/CLAIMS.md`
