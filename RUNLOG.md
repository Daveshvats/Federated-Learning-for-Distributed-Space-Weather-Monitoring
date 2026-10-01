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
| 2 | 2026-10-01 | Sync local repo with the queue package on `origin/improvements` | `git stash && git pull origin improvements && git stash drop` | **no** (10 s, any machine) | chat | **GIVEN — PENDING** | see safety note below |
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
| 2026-09-29 → 10-01 | frozen-protocol reproduction, calibration, holdout, budget-matched, alpha-promotion, event-level LSTM pass, paper v3.2–v3.7, all tests | 142/142 checks green; all pushed to `origin/improvements` |
| 2026-10-01 | queue package build: 3D-aware SMOTE (`data_preparation.apply_smote`), SCAFFOLD-LSTM arm + `--smote`/`--scaffold-only`/`--force` flags (`run_federated_lstm.py`), `_aux` fallback in the in-runner event-level path, `run_gpu_queue.py` driver, this ledger | smoke-verified on the real 3D caches on CPU before handover |

## Machine-written (auto-appended by run_gpu_queue.py — do not edit)
