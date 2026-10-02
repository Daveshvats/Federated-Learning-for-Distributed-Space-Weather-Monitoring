# RUNLOG — owner-side execution ledger

Every command this project asks the **owner** to run, whether it was
delivered, and whether it ran.  Maintained by the assistant; the bottom
section is appended automatically by `experiments/run_gpu_queue.py`
(machine-written — do not edit by hand).

Legend: GPU? = does it need the owner's GPU.  Status: GIVEN = delivered,
awaiting execution · RAN = executed by owner · DONE = executed by
assistant (no owner action).

| # | Given | Ask | Command | GPU? | Delivered | Status | Outcome |
|---|-------|-----|---------|------|-----------|--------|---------|
| 1 | 2026-09-30 | Item-8 LSTM arms on the raw substrate (frozen protocol, seed 42) | `python experiments/run_federated_lstm.py` | YES (~2.6 h) | chat + git e185e4a | **RAN** (9,290 s on RTX 4060 Laptop) | `outputs/raw_lstm_eval.json` validated 44/45 checks → paper v3.6; event-level pass executed on CPU by assistant → v3.7 |
| 2 | 2026-10-01 | Sync local repo with the queue package on `origin/improvements` | `git stash && git pull origin improvements && git stash drop` | **no** (10 s, any machine) | chat | **RAN** (implicit: queue code exists only post-pull) | CRLF-noise stashed + dropped; untracked files + data/ untouched, as predicted |
| 3 | 2026-10-01 | The remaining GPU programme in ONE batch: SCAFFOLD-LSTM (seed 42), seed-43 replication (4 arms), per-client SMOTE ablation | `python experiments/run_gpu_queue.py` | YES (~7-8 h, overnight; resumable) | git (this commit) + chat | **RAN** (31,250 s = 8.7 h on RTX 4060 Laptop) | all 3 steps COMPLETED and wrote all 6 JSONs; each child exited 0xC0000409 at Windows teardown AFTER the atomic writes (benign — logs + queue summary confirm full execution); JSONs uploaded 2026-10-01, validation + v3.8 integration pending |

## Why ask #2 is safe on the owner's local repo (verified from the
uploaded split-zip, Task 7)

- Local HEAD is `e185e4a`, one commit behind `origin/improvements`
  (`0c37ed8` + the queue commit) — a clean fast-forward.
- The ~107 "modified" tracked files are **CRLF line-ending noise only**
  (Windows checkout); `git stash` parks them and `git stash drop`
  discards pure noise — nothing of value is lost.
- The two genuinely-new local files (`README.txt`,
  `outputs/raw_lstm_eval.json`) are **untracked**: plain `git stash`
  never touches untracked files, so they survive the whole sequence.
- `data/` (998 MB of 3D caches + checkpoints) is gitignored → untouched.
- Full backup of the un-synced state exists on the assistant side
  (`repo_upload/`, verified bit-identical) — worst case is recoverable.

## Assistant-side work that needed NO owner GPU (for the record)

| Date | Work | Result |
|------|------|--------|
| 2026-09-29 → 10-01 | frozen-protocol reproduction, calibration, holdout, budget-matched, alpha-promotion, event-level LSTM pass, paper v3.2–v3.7, all tests | tests green at the time (the 142/142 figure of the era's test suite; the v4.0 battery is the single authoritative count) ; all pushed to `origin/improvements` |
| 2026-10-01 | queue package build: 3D-aware SMOTE (`data_preparation.apply_smote`), SCAFFOLD-LSTM arm + `--smote`/`--scaffold-only`/`--force` flags (`run_federated_lstm.py`), `_aux` fallback in the in-runner event-level path, `run_gpu_queue.py` driver, this ledger | smoke-verified on the real 3D caches on CPU before handover |
| 2026-10-03 | v4.0 evidence edition (third-review Tier-1 programme): `data_manifest/verify_manifest.py` (20/20 SHA-256, hard gate at pipeline start), SHIPPED/FOLD reference dicts replaced by run-time reads + both eval JSONs corrected, provenance totals recomputed on dual bases (`rebuild_audit_totals.py`, `test_audit_artifact.py`), GPU queue logs committed + hash-bearing ledger lines, `tests/run_battery.py` single count (214/214), phase-cache key over all 20 manifest files, loud-failure loader, region-ID leakage gate (`--allow-in-partition`), Leka-standard verification apparatus (`run_standard_metrics.py`), bibliography repaired + FedBN-lineage/prior-warning citations, GIC/grid framing stripped, title narrowed to benchmark audit, paper recompiled | battery 214/214 (`logs/test_battery.log`); manifest verification PASS (`logs/verify_manifest.log`); paper v4.0 PDF built |
| 2026-10-03 | v4.1 (Dossier R-FS9-R1, items A1-A6/B1-B4/C1-C6/D1-D2): persistence baseline recomputed on true time order (`ts_start_min`, not the temporally-scrambled `pool_row`) with BOTH floors (label inertia TSS 0.967/HSS 0.970 P5; 24h-lag 0.418/0.434), LSTM F2 aggregation via merged nested-`test` view (10 arms now carry TSS/HSS: central 0.456, fedavg 0.305, fedprox 0.634, scaffold 0.707), raw-2D frozen-FPR cells via top-level `frozen_operating_points` (24 cells), zero-alert NaN → 0.0 semantics + strict JSON (`allow_nan=False`), exact chi-square (Garwood) Poisson intervals, Wilson/Poisson spans restated from the regenerated artefact, cadence 60 min recorded; six bib entries replaced with records read from arXiv/DBLP/IEEE/IOP (`wang2023bn` → Wang/Shi/Chang TNNLS 36(1); `guerraoui2024bn` → 5 authors, no "Heterogeneous"; `bnscaffold2024` → Quintana/Vancamberg/Jugnon/Mougeot/Desolneux; `angryk2019` → Ahmadzadeh/Hostetter BigData 2019 pp.1423-1431; `ahmadzadeh2021` → ApJS 254(2):23; `li2021fedbn` → Xiaoxiao), fu2023 DOI in field + self-contradiction removed, bolduc2002/baker2013 deleted, boteler2019 cited in the intro boundary disclaimer; `sec:limits`→`sec:limitations` fixed; `CLIENT_NAMES` → plain A–F (config, partition_clients, make_fig_partition) + both figures regenerated; sec_problem custodian/grid passages replaced with public-data simulation framing; §4.4 retitled "simulated clients"; conclusion 23.4h label-boundary caveat + §6.3 "4×" spin removed; tab:main caption corrected (test climatology 0.0185); abstract narrowed + persistence headline; manifest environment claim matched to artefacts (battery log header records python/numpy/torch); `tests/test_leakage_gate.py` (5 checks, failure path exercised); v4.1 appendix row; paper recompiled | battery 219/219 (`logs/test_battery.log`, env header python 3.13.5/numpy 2.2.4/torch 2.14.1+cpu); `standard_metrics.json` strict-JSON, bit-exact on re-execution; paper v4.1 PDF built (51 pp) |

## Machine-written (auto-appended by run_gpu_queue.py — do not edit)

<!-- v4.0 ledger reconstruction: lines below are derived from the
     committed queue logs and artefacts (each line carries the
     sha256[:16] of its artefacts, so post-hoc edits are
     detectable). The original queue process appended its DONE
     lines in a different working copy; these lines are the
     committed-record equivalent. -->
2026-10-03 | queue_scaffold.log: queue step completed on owner GPU (cuda:0); artefact hashes raw_lstm_scaffold.json#a63f9333e56743e6 event_level_raw_scaffold.json#84536fee63b8b1ee
2026-10-03 | queue_seed43.log: queue step completed on owner GPU (cuda:0); artefact hashes raw_lstm_seed43.json#d0b314b8d52b3cbc event_level_raw_seed43.json#0f3d44d28dd7593c
2026-10-03 | queue_smote.log: queue step completed on owner GPU (cuda:0); artefact hashes raw_lstm_smote.json#462387c8e5845bc2 event_level_raw_smote.json#dd7af0251a5d6291
2026-10-03 | bn_diagnostic (phase-cache CPU): artefact hashes bn_diagnostic.json#51b9bcc256eb7aa1 repro_run.log#a53b5511c3bc5182
2026-10-03 | no-BN control (CPU): artefact hashes nobn_control.json#7e2b1ae8a07a2770
| 2026-10-03 | v4.2 (Dossier R-FS9-R2, items 1-5 / F1-F7): F1 false exclusivity corrected in all four places — abstract, §6.1 (×2), appendix v4.1 row, archived R-FS9-R1 letter (with inline erratum markers + closing erratum section) — to the artefact-backed statement: at TSS XGBoost (0.848) and logistic regression (0.489) alone clear the 24h-lag floor (0.418), at HSS none clears (0.434); LR TSS/HSS 0.489/0.025 added to §6.1's first finding; F2 FA range re-attributed (8.6→11.6–12.7 for the neural arms); F3 environment parenthetical split (protocol Python 3.12; battery header records the v4.1 verification battery's python 3.13.5/numpy 2.2.4/torch 2.14.1+cpu, cited only for what it records); F4 "bit-exact"→content-exact for artefact-derived blocks A/C/D + block B phase-cache dependency + elapsed_s disclosure (§6.1 + appendix row); F5 letter erratum (six→five leakage-free arms); F6 docstring cadence ~64→60 min; F7 executed not dropped: `experiments/run_lag_definition_sweep.py` → `outputs/lag_definition_sweep.json` (8 candidate definitions + shipped reference, strict JSON, deterministic, shipped-rule consistency vs block C PASS; seven point-lag variants TSS 0.392–0.418 with two arms clearing each; lookback-flare variant 0.964 = inertia under another name); §6.1 sweep pointer added; appendix v4.2 row + caption range v2.1–v4.2; ABSTRACT.md resynced from v3.9 → v4.2 abstract | paper recompiled (52 pp, 0 unresolved refs); battery re-verified in current torch-less env: 195 passed / 0 failed / 24 skipped (`logs/test_battery_v4.2.log`, header records python 3.12.14; canonical v4.1 log 219/219 retained untouched) |
