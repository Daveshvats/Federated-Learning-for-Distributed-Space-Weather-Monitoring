# SF-9: Federated Learning for Distributed Space Weather Monitoring

Federated solar-flare prediction on the SWAN-SF benchmark under simulated
cross-silo, non-IID partitioning — with a **leakage-audited, validation-only
selection protocol**.

> **Branch**: `improvements` (v3.1). The `main` branch preserves the
> original v2.x pipeline and its committed figures. Every headline number
> produced before v3.0 used a flawed protocol (see `audit/BUG_REGISTER.md`)
> and must be regenerated with the code on this branch.
>
> **v3.1 (review-2 response)**: protocol hardening for the second external
> review — see `docs/REVIEW2_RESPONSE.md` for the complete 25-finding
> mapping. Headline numbers are unchanged (frozen seed-42 artefacts); the
> revision ships new protocol code (all unit-tested, 78/78 checks), a
> revised manuscript (v3.2, `paper/main.pdf`), neutral client labels, a
> threat-model table, and a documented literature-search protocol.

## What this repo demonstrates

- Feederated training (FedAvg / FedProx / SCAFFOLD) of a solar-flare
  classifier across **disjoint, audited** Dirichlet client shards
- The **accuracy cost of federation** vs centralized baselines
  (climatology, logistic regression, centralized MLP, XGBoost)
- Calibration + operational thresholding under extreme prior shift
  (49% training prevalence -> 1.9% test prevalence)
- Client-level evaluation (does federation help each participant?)
- SHAP interpretability on both the XGBoost reference and the FL model

What it does **not** demonstrate: GIC prediction, SCADA deployment,
cryptographic privacy, real inter-agency operation — see
`audit/CLAIMS.md` and `limitations/LIMITATIONS.md`.

## Quick start

```bash
pip install -r requirements.txt

# 1. data provenance manifest (before anything else)
python data_manifest/generate_manifest.py

# 2. integrity tests (no dataset required — synthetic fixtures)
python tests/test_pipeline_integrity.py
python tests/test_fl_smoke.py

# 3. place the Cleaned SWAN-SF pkl files under data/cleaned/{train,test}/
#    https://github.com/samresume/Cleaned-SWANSF-Dataset
#    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM

# 4. full pipeline (MLP mode recommended first)
python main.py --no-lstm          # resumable: phase + round-level caches
#    python main.py --fresh       # force full recompute
#    LSTM mode: python main.py

# 5. experiment battery (all resumable across interruptions)
python experiments/run_ablations.py --rounds 20      # component isolation
python experiments/run_sweep.py --rounds 15           # mu x alpha grid
python experiments/run_multiseed.py --seeds 5         # mean/std/95% CI
python experiments/analyze_smote_validity.py          # SMOTE plausibility
python experiments/run_interpretability.py            # XGB + FL SHAP
python experiments/run_calibration_comparison.py      # calibration arms (v3.1)
python experiments/run_event_level.py --input outputs/event_scores.json  # v3.1
python secure_aggregation.py                          # secagg self-test
```

Every run writes `outputs/results.json` + `outputs/run_manifest.json`
(machine-readable, timestamped) — **figures and paper tables must be
generated from these files, never typed by hand**.

## Completed experiment programme (this branch, v3.0.1 — numbers unchanged in v3.1)

All experiments below have been executed on the real Cleaned SWAN-SF
data (train 97,764 @ 48.87% / test 331,185 @ 1.88%) with artefacts
committed under `outputs/`:

| Experiment | Artefact | Key outcome |
|---|---|---|
| Headline run (50 rounds, seed 42) | `results.json` | FedProx AUC 0.954 vs XGBoost 0.977 vs pooled MLP 0.888 |
| 24-cell ablation grid | `ablation_results.json` | FedProx stable everywhere; DA-FL rescues FedAvg; SMOTE inert |
| mu/alpha sweep (9 configs) | `sweep_results.json` | val-selected winner alpha=5.0, mu=0.01 |
| 5-seed study | `multiseed_results.json` | FedProx > FedAvg on 5/5 seeds (Wilcoxon p=0.031) |
| SMOTE validity | `smote_validity.json` | not applicable (pre-balanced data) — reported null |
| Interpretability | `interpretability.json` | physical-group consistency (helicity/R-value/Lorentz) |
| Communication | in `results.json` | 0.24 MB/client/round, 70.5 MB total |

## Repository layout

```
SF9/
├── main.py                     # v3.0 orchestrator (frozen protocol)
├── config.py                   # single configuration source
├── data_preparation.py         # immutable train/val/test contract
├── partition_clients.py        # disjoint Dirichlet + geographic partition
├── federated_learning.py       # FedAvg/FedProx/SCAFFOLD (val-monitored)
├── centralized_baseline.py     # climatology/LR/central-MLP/XGBoost
├── evaluation.py               # metrics, thresholds, calibration
├── evaluate_clients.py         # local-vs-federated per client
├── interpretability.py         # XGBoost + FL SHAP, consistency check
├── communication_cost.py       # bytes/round, dropout simulation
├── secure_aggregation.py       # Bonawitz masking scaffold
├── visualize_results.py        # figures from results dict
├── audit/                      # claims table, bug register, revision plan
├── leakage_audit/              # runtime leakage verification
├── data_manifest/              # dataset provenance manifest
├── experiments/                # ablations, sweep, multiseed, SMOTE
├── privacy_analysis/           # threat model
├── limitations/                # honest limitations
├── docs/                       # GIC boundary, lit-search, review-2 response
├── tests/                      # integrity + smoke tests (78 checks)
├── logs/                       # execution logs
├── paper/                      # manuscript (LaTeX + PDF) + review
└── outputs/                    # results.json, figures (regenerated)
```

## Evaluation contract (enforced by code + tests)

| Split | Used for | Never used for |
|---|---|---|
| TRAIN | client training | — |
| VAL (16% of train) | monitoring, calibration fitting, threshold search, checkpoint selection, hyperparameters | reporting |
| TEST (dataset-defined) | **single final evaluation** | selection of any kind |

Additional guarantees, each with a test in `tests/`:

- client shards are **disjoint with 100% coverage** (permutation-based
  Dirichlet allocation)
- the leakage detector must **catch seeded leakage** before it is trusted
- threshold selection on validation gives **no oracle upside** on test
- all metrics (incl. accuracy) are **recomputed at the frozen threshold**
- secure-aggregation masks cancel exactly (server sees sums only)

## Reproducibility statement

- **Dataset**: Cleaned SWAN-SF (RUS-Tomek-TimeGAN, LSBZM-Norm,
  FPCKNN-impute, WithoutC), 5 partitions, SHA-256 inventory in
  `data_manifest/manifest.json`; label = M/X flare within the export's
  forecast window.
- **Code**: this repository, `improvements` branch; version string in
  `config.py` snapshotted into every `run_manifest.json`.
- **Environment**: Python ≥ 3.10; `pip freeze` output should be archived
  with each experiment; torch CPU/GPU both supported (CUDA auto-detected).
- **Seeds**: root seed `config.SEED` (default 42); multi-seed protocol
  uses `SEED + k`, k = 0..4; partitioning, model init, shuffling, and
  SMOTE all derive from the run seed.
- **Training**: N_ROUNDS communication rounds, LOCAL_EPOCHS local epochs,
  AdamW (FedAvg/FedProx) or SGD (SCAFFOLD), gradient clipping, Fed-Focal
  loss (alpha clamped to [0.05, 0.25]).
- **Evaluation**: threshold + calibration selected on validation; single
  test pass; metrics defined in `evaluation.compute_all_metrics`.
- **Hardware**: CPU-runnable for MLP smoke tests; the full 5-partition
  dataset benefits from a ≥12 GB GPU (EVAL_BATCH_SIZE=2048 tuned for
  RTX 3060 12GB).

## Citation / attribution

- Dataset: Angryk et al., *SWAN-SF*, Scientific Data 7, 2020
  (doi:10.1038/s41597-020-0548-x); cleaned variant per
  samresume/Cleaned-SWANSF-Dataset.
- Algorithms: FedAvg (McMahan et al., 2017), FedProx (Li et al., 2020),
  SCAFFOLD (Karimireddy et al., 2020), Focal loss (Lin et al., 2017),
  SMOTE (Chawla et al., 2002), SHAP (Lundberg & Lee, 2017).
- Manuscript and audit: see `paper/`.

## Next actions (v3.1 queued re-run programme)

Implemented, unit-tested, awaiting owner-side compute / raw metadata
(full list with paper cross-references in `docs/REVIEW2_RESPONSE.md` §C):

1. Region-disjoint splits from raw SWAN-SF metadata (needs HARP/AR IDs)
2. Natural-prevalence validation + full frozen-FPR operating-point report
3. Calibration comparison on the trained FL models (arms shipped)
4. Event-level evaluation (needs event keys + timestamps)
5. Untouched-holdout client evaluation
6. Budget-matched centralized-vs-federated training
7. Sweep-winner promotion (alpha=5.0) through multiseed + ablation
8. Federated LSTM and SCAFFOLD evaluation
9. Raw unbalanced SWAN-SF experiment programme

Items 1-5 are preconditions for quoting any number in the paper as
operational performance rather than benchmark result.
