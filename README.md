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
python experiments/run_calibration_comparison.py      # calibration arms (v3.3: executed, real mode)
python experiments/run_client_holdout.py              # untouched-holdout client eval (v3.3: executed)
python experiments/run_budget_matched.py              # 500-epoch budget-matched centralised (v3.3: executed)
python experiments/run_alpha_promotion.py             # alpha=5.0 promotion run (v3.3: executed, negative)
python experiments/run_partition_disjoint.py           # leakage-free fold P1-4 -> P5 (v3.4: executed)
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

## Next actions (v3.1 queued re-run programme — updated v3.5)

Implemented, unit-tested; items 2 (frozen-FPR report), 3 (calibration
comparison), 5 (untouched holdouts), 6 (budget-matched training), and
7 (alpha=5.0 promotion — negative) were **executed** in the v3.3
independent re-executions (see `docs/REVIEW2_RESPONSE.md` §D-E).
Items 1, 2, and 4 were executed on the v3.4 leakage-free fold
(region-disjointness holds by construction; natural-prevalence test
boundary; event-level evaluation, 65/65 GOES peak matches). Item 9 and
the SCAFFOLD half of item 8 were **executed in v3.5**:

8. ~~Federated LSTM and~~ SCAFFOLD evaluation — **SCAFFOLD EXECUTED
   (v3.5)**: fails on the raw substrate (0.768/0.057, Brier 0.137,
   54/65 events at 6.4 FA windows/day). The LSTM arm alone remains
   queued (owner GPU).
9. ~~Raw unbalanced SWAN-SF experiment programme~~ **EXECUTED (v3.5)**
   (`experiments/raw_substrate.py` + `run_raw_substrate.py` +
   `run_event_level_raw.py`; artefacts `outputs/raw_substrate_eval.json`,
   `outputs/event_level_raw_p5.json`,
   `outputs/raw_substrate_verification.json`): frozen protocol retrained
   on raw P1-4 at natural 2.05% prevalence with FPCKNN/LSBZM reproduced
   in-pipeline (train-only parameters, verified vs the release):
   centralised arms substrate-robust (LR 0.978/0.448, XGB 0.974/0.329,
   MLP 0.971/0.445), federated arms collapse (FedAvg 0.875/0.056 with
   0/65 events, FedProx 0.930/0.149, SCAFFOLD 0.768/0.057).

Remaining: fold/seed replication of the leakage-free and raw-substrate
runs, the federated LSTM arm (GPU), and the natural-prevalence SMOTE
ablation. The event-level figures (`outputs/event_level_p5.json`,
`outputs/event_level_raw_p5.json`) are the operational performance
figures, with the single-fold caveat.

## v3.3 independent re-execution (2026-09-29)

The headline pipeline was re-executed from the public Cleaned SWAN-SF
release (all 20 pkl partitions SHA-256-verified against
`data_manifest/manifest.json`) on CPU only. **Every test-set metric of
all six models reproduced exactly** (three-decimal match on all 48
model-metric pairs; FedAvg divergence reproduces round-for-round) —
the frozen protocol is bit-stable, so the published numbers are
re-derivable from public inputs. The same session executed the
calibration-arm comparison (negative result: no validation-fit arm
survives the 26x prior shift), the untouched-holdout client
evaluation (local 0.983 / FedAvg 0.866 / FedProx 0.988 mean PR-AUC),
and the frozen-FPR operating-point report (realised test FPR
28.2-72.1% against 0.5-5% targets for FedProx). Artefacts:
`outputs/results.json`, `outputs/calibration_comparison.json`,
`outputs/client_holdout_eval.json`; manuscript v3.3 (Sec. 5.6).

Dataset acquisition: the Cleaned SWAN-SF partitions are distributed
via the link in `download.txt` of the
`samresume/Cleaned-SWANSF-Dataset` repository; place the 10 train and
10 test `.pkl` files under `data/cleaned/train/` and
`data/cleaned/test/` respectively, then verify with
`python data_manifest/generate_manifest.py`.


## v3.4 provenance audit + leakage-free fold (2026-09-30)

Raw SWAN-SF benchmark obtained (Harvard Dataverse
doi:10.7910/DVN/EBCFKM) and every cleaned window aligned to its raw
instance (provenance/; argmax/argmin-invariant matching, test
verification 98.7-100.0%, 100.00% label agreement). Audit findings:
the cleaned export's same-partition train/test pairing shares
instances (test export = all raw instances; train export = rebalanced
subset of the same; 100% of flaring test windows are training
windows; 85.7-90.0% of training positives are TimeGAN-synthetic).
Leakage-free fold executed under the benchmark's intended
temporally-preceding protocol (train P1-4 -> test P5,
experiments/run_partition_disjoint.py): all arms generalise
(ROC 0.906-0.978), central-vs-FL gap disappears (FedProx 0.976/0.308
vs centralised MLP 0.973/0.382), FedProx ROC stabilisation over
FedAvg survives (0.976 vs 0.906, checkpoint-independent), PR ordering
reverses, calibration/threshold-transfer failures persist, event-level
evaluation on 65 M/X events (detection 98.5-100%, XGBoost false-alarm
burden ~7x lower than neural arms, 23.4h median lead). Artefacts:
outputs/dataset_structure_audit.json, outputs/partition_disjoint_eval.json,
outputs/event_level_p5.json, provenance/ (scripts + slim metadata).
