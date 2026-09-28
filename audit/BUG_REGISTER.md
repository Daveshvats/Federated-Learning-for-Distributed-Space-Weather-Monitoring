# SF-9 — Bug Register (Code Audit Findings)

> Each entry: location, severity, effect on results, fix status on the
> `improvements` branch. Severities: **P0** invalidates reported numbers,
> **P1** confounds scientific conclusions, **P2** reproducibility/quality,
> **P3** cosmetic.

## P0 — Invalidates reported numbers

| ID | Location | Finding | Effect | Fix |
|----|----------|---------|--------|-----|
| B1 | `main.py:79-121,355-406` | **Test-set threshold selection** — F-beta-optimal thresholds are searched directly on the held-out test labels | All thresholded metrics (precision/recall/F1/F2/accuracy) are optimistically biased; test set is no longer untouched | Thresholds selected on a NEW validation split; test touched exactly once |
| B2 | `partition_clients.py:230-245` | **Non-disjoint Dirichlet shards** — each (client,class) independently samples indices with replacement-free `np.random.choice`; nothing prevents the same underlying sample going to 2+ clients; `int()` truncation drops samples | Duplicate training samples across clients (data duplication across "sovereign" clients — also a simulation-validity problem); total shard size ≠ train size | Permutation-based disjoint allocation with exact coverage; returns index assignment for audit |
| B3 | `federated_learning.py:566-581` | **SCAFFOLD checkpoint selection on test** — best-F1 checkpoint tracked and restored using test-set F1 | Model selection leakage | Selection on validation |
| B4 | `main.py:363-389` | **Stale FL accuracy** — after threshold optimization, `accuracy` is never recomputed for FedAvg/FedProx/SCAFFOLD (only centralized models get `accuracy_score` recomputed). Comparison table shows accuracy at the OLD 0.35 threshold | Reported FedAvg/FedProx accuracy (0.019) is wrong; confusion matrix implies ~0.97 | Recompute all metrics at frozen threshold |
| B5 | `federated_learning.py:413-421,469-477` (and scaffold) | **FL monitoring on test set** — convergence history evaluated every 5 rounds on test data | Convergence figures are test-set curves; mild selection leakage | Monitor on validation |

## P1 — Confounds scientific conclusions

| ID | Location | Finding | Effect | Fix |
|----|----------|---------|--------|-----|
| B6 | `federated_learning.py:407-411,463-467` | **DA-FL aggregation used for both FedAvg and FedProx arms** — "FedAvg" is actually distribution-aware weighted aggregation (size × √φ, cap 2/N) | Cannot attribute FedProx vs FedAvg differences to the proximal term; the "FedAvg" label is wrong | Plain size-weighted FedAvg default; DA-FL as explicit ablation arm |
| B7 | `data_preparation.py:473-499` + grep | **SMOTE is dead code** — `apply_smote` never invoked anywhere | Any manuscript statement about SMOTE-based balancing is unsupported | Wire into ablation matrix; correct manuscript |
| B8 | `federated_learning.py:109, 409, 465`; `losses.py:70,193` | **Hardcoded `global_pos_rate=0.4887`** — training prevalence baked into aggregation weights and loss alpha | Breaks silently on any other dataset/split; embeds stale data knowledge | Compute from training shards and pass explicitly |
| B9 | `centralized_baseline.py:53` | **XGBoost `scale_pos_weight=10.0` "to handle test imbalance"** — test-set knowledge baked into training | Confounded centralized reference | Compute `n_neg/n_pos` from training labels |
| B10 | `main.py` (whole flow) | **No validation split exists** — only train/test | No protocol place for threshold/calibration/model selection | Immutable train/val/test contract in `data_preparation.preprocess` |
| B11 | Single run everywhere | **Single seed (42), no variance estimates** | No statistical claims possible | Multi-seed runner + mean/std/95% CI |
| B12 | `data_preparation.py:95-111` | **Feature-name loss** — stat-prefixed names (`mean_TOTUSJH`) replaced by generic `feature_N` | SHAP figure shows `feature_N` labels; interpretation weakened | Propagate real names through pipeline |
| B13 | `data_preparation.py:137-210` | **Fabricated `_HARPNUM_MOD`/`HARPNUM_MOD`** — random client assignment presented as HARP-based partitioning | Misleading provenance; conflates partition identity | Renamed `_SIM_CLIENT`; documented as synthetic; not used as feature |
| B14 | `centralized_baseline.py` + `main.py` | **No centralized MLP baseline** — comparison mixes architecture (MLP vs XGBoost) with federation | Federation cost not isolated | Centralized MLP trained on pooled 2D features |

## P2 — Reproducibility / engineering

| ID | Location | Finding | Fix |
|----|----------|---------|-----|
| B15 | `federated_learning.py:388,444,517` | client selection uses `N_CLIENTS` instead of `len(shards)` (breaks `--clients N`) | use `len(shards)` |
| B16 | `main.py:5` vs `config.py:7` | version string drift ("v2.1" vs "v2.6") | single `VERSION` in config, snapshot into results |
| B17 | `config.py:115` vs `losses.py:215` | `FOCAL_ALPHA=0.25` then clamped to (0.05, 0.25) — config implies tunable, code caps it | document + propagate as parameter |
| B18 | `outputs/` only contains PNGs | **No machine-readable results** — table values live only inside a PNG | `results.json` + `run_manifest.json` written every run; figures regenerated from them |
| B19 | `evaluate_model` threshold default 0.35 | implicit default threshold in monitoring | explicit config + report |
| B20 | `visualize_results.py:4` "ICE2CT-2026" vs `ABSTRACT.md` Track 3 | venue naming inconsistency | neutral naming |
| B21 | `losses.py:17-22` | unverifiable citations ("arxiv 2602.01633, Feb 2026", "cited 29/445") | keep verifiable refs only (Lin 2017, Sarkar 2020) |
| B22 | `fix_encoding.py` | Windows-only utility runs unconditionally in docs flow | keep, documented |
| B23 | `main.py:235` | 3D/2D sample-count assert can pass while partition membership differs (2D flattened from different source than 3D) | assert feature-count consistency too + manifest records both |

## P3 — Cosmetic

| ID | Finding | Fix |
|----|---------|-----|
| B24 | Emoji/checkmark glyphs in logs break some Windows consoles | `fix_encoding.py` retained; logs ASCII-safe in new modules |
| B25 | Dead `MixupBCELoss`, `PersonalizedSolarMLP` unused | kept as library code, marked experimental |

---

### Cross-reference to revision gates

- **Gate 1 (data validity):** B2, B12, B13, B23
- **Gate 2 (statistical validity):** B1, B3, B4, B5, B10, B11, B17, B18, B19
- **Gate 3 (ML validity):** B6, B7, B8, B9, B14, B15
- **Gate 4 (physical validity):** B12 (SHAP), claims 4/9 in `CLAIMS.md`
- **Gate 5 (operational/security):** B16, B18, claims 5/6/7 in `CLAIMS.md`
