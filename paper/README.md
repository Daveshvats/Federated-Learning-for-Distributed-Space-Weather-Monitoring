# paper/ — manuscript of record & audit artifacts

## Status (v4.5, 2026-10): R-FS9 CYCLE CLOSED — ACCEPT SUSTAINED

| File | Status |
|---|---|
| `main.pdf` + `main.tex` + `sections/` + `refs.bib` | **The manuscript of record** (v4.5), compiled with `tectonic`. Every printed number regenerates from a committed artefact in `outputs/` — the standard the R-FS9 referee cycle (six rounds, v3.9 reject-as-framed -> v4.5 accept-sustained, rating 7.9/10) converged on. |
| `review.tex` | Source of the historical v2.x-era companion audit (findings B1-B25) that drove the `improvements` branch — kept as a historical document. Its compiled PDF and the superseded manuscript PDF were removed at v4.5 (R-FS9-R5 front-door finding; preserved in git history). |
| `figures/` | The live figure set referenced by `main.tex` (8 figures). Root-level duplicate figures and the two unreferenced `figures/` leftovers were removed at v4.5; the live set regenerates from `outputs/*.json` by the pipeline + `scripts/fig_analysis.py`. |

## The honest headline (as corrected at v4.2)

- Under the corrected persistence baselines on the leakage-free fold:
  **no arm beats same-region label inertia (TSS 0.967) at this 24-hour
  label geometry; on TSS, XGBoost (0.848) and logistic regression
  (0.489) alone clear 24-hour-lagged persistence (0.418); on HSS, no
  arm clears it (0.434)**
- Random-split leakage: 100% test-positive overlap in the same-partition
  pairing — the community-tolerated defect this benchmark audit exists
  to surface
- Calibration, not detection, is the deployability boundary under the
  benchmark's prevalence shift
- Six simulated clients, one public benchmark: a scope boundary the
  paper states plainly

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
