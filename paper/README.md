# paper/ — the extended edition of record

## Status (v4.12, 2026-10): the full/extended edition

This directory is the **full edition** — the extended record of the
study: the complete experiment apparatus, the LSTM/SMOTE/client
studies, the SCAFFOLD consolidation appendix, and the configuration
version-history table. The publishable edition is the trimmed
12-page paper at `paper_trimmed/` (compiled to
`submission/main.pdf`); both editions are kept by design. The v4.12
revision responds to an independent external re-review (five
findings, all accepted): the FedProx parity claim re-worded to
"matching within recorded retrain nondeterminism", the
validation-story prevalence explanation withdrawn in favour of the
two testable candidate mechanisms (with the region-disjoint re-run
kit committed at `docs/RUN_CARD_v4.12.md`), and all SCAFFOLD cells
consolidated out of the main tables and conclusion into the
appendix with their implementation status disclosed.

| File | Status |
|---|---|
| `main.pdf` + `main.tex` + `sections/` + `refs.bib` | **The extended edition of record** (v4.12, 65 pp), compiled with `tectonic`. Every printed number regenerates from a committed artefact in `outputs/`. |
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
