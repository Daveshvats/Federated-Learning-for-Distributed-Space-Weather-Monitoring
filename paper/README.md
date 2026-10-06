# paper/ — manuscript of record & audit artifacts

## Status (v4.11, 2026-10): R-FS9-R11 ACCEPT — SUSTAINED, 8.4/10 (the
Round-13 condition discharged and verified at v4.10 — A1
`arm_b_validated`, A3 the BatchNorm-under-federation attribution;
the v4.11 pre-submission errata executed the panel's punch list:
the stale journal-edition stamp, the two unpinned decisive artefacts,
and six one-word-class wording fixes — nothing blocking)

| File | Status |
|---|---|
| `main.pdf` + `main.tex` + `sections/` + `refs.bib` | **The manuscript of record** (v4.11), compiled with `tectonic`. Every printed number regenerates from a committed artefact in `outputs/` — the standard the R-FS9 referee cycle (eleven dossiers, v3.9 reject-as-framed -> v4.5 accept-sustained 7.9/10 -> v4.7 accept-conditional 7.4/10 -> v4.10 register integration -> v4.11 accept-sustained 8.4/10) converged on. |
| `review.tex` | Source of the historical v2.x-era companion audit (findings B1-B25) that drove the `improvements` branch — kept as a historical document. Its compiled PDF and the superseded manuscript PDF were removed at v4.5 (R-FS9-R5 front-door finding; preserved in git history). |
| `figures/` | The live figure set referenced by `main.tex` (8 figures). Root-level duplicate figures and the two unreferenced `figures/` leftovers were removed at v4.5; the results figures regenerate via `visualize_results.py` (writes `outputs/*.png`), the client/partition pair via `tools/make_fig_clients.py` / `tools/make_fig_partition.py` (the `scripts/fig_analysis.py` reference that stood here pointed at a path that does not exist in this repository — Dossier R-FS9-R7 B11). |

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
6. `python visualize_results.py` (writes `outputs/*.png`; copy the
   needed figures into `figures/`), plus `python tools/make_fig_clients.py`
   and `python tools/make_fig_partition.py` (write into `figures/`
   directly)
7. `cd paper && tectonic main.tex`
8. Update `sections/*.tex` **from the JSON files only**; keep claim-status
   language aligned with `../audit/CLAIMS.md`
