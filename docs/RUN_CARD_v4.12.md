# RUN CARD v4.12 (rev. v4.12.1) — the external-review compute items (2) and (3)

The independent re-review of v4.11 asked for two small compute jobs
that this sandbox cannot run (no GPU, no dataset cache). Both are
prepared, guarded, and namespaced so they cannot disturb the frozen
v4.9.4 artefacts. This card is the exact recipe.

**v4.12.1 errata (2026-10-06, after the first owner-side execution):**
two latent defects in the item-2 kit, both fixed and now
battery-guarded (tests/test_region_disjoint.py, 16 checks):

- the region map was read from the SAMPLED audit meta
  (`provenance/train_meta_slim.csv.gz` — 97,764 rows over partitions
  1..5, its `pool_row` indexing the slim file itself), so a real run
  would have aborted at its own coverage gate (97,764 of 255,820 pool
  rows) even with the raw dir present. Regions now come from the
  parse metadata's `ar` column — the SAME `p{p}_meta.csv` files that
  supply the labels, full pool coverage by construction.
- the raw-dir default is the POSIX path `/tmp/swansf_raw`; on Windows
  it resolves against the current drive and the first execution died
  with a bare `FileNotFoundError`. The split now pre-flights all four
  `p{1..4}_meta.csv` files and exits with a guided message naming
  `--raw-dir` and the regeneration command.

The item-3 kit (seed 43) needed no code change: the first owner
execution reached round-1 training on the RTX 4060 and was
interrupted with Ctrl+C (the traceback ends in `KeyboardInterrupt`,
not a defect). Just re-run it.

## Context

- **Item 2 (validation story).** In `outputs/raw_nobn_eval.json`,
  FedAvg no-BN validation ROC-AUC *rises* (0.9736 → 0.9773) while
  test ROC-AUC *falls* (0.974 → 0.963). ROC-AUC is prevalence-
  invariant, so the 2.05%/1.31% prevalence difference cannot explain
  it. Two candidates remain: (a) the random validation carve shares
  active regions with the training shards (leakage), or (b) partition
  5 is the temporally latest fold (drift). The region-disjoint
  re-run separates them.
- **Item 3 (one seed).** The no-BN control is seed-42 only. The big
  swings (+0.088/+0.310) are safe, but the level claims (sequence-arm
  parity, centralised-MLP parity) sit inside the recorded retrain
  noise band (0.002–0.023 ROC-AUC), so they need seed 43.

## Prerequisites

1. **The repo at v4.12.1** (branch `improvements`), working tree
   clean — `git pull` to pick up the errata commit.
2. **The substrate cache** `data/cache/rawsubstrate/data.npz` —
   present on the owner box; both runs resume it (the seed-43 run
   refuses to start without it, by design).
3. **The raw parse metadata** `p1..p4_meta.csv`, in any directory,
   passed as `--raw-dir`. These are NOT in the repo (the first
   execution failed on exactly this). Two cases:
   - you still have the parsed files somewhere (e.g.
     `C:\tmp\swansf_raw`) — point `--raw-dir` there;
   - they are gone — regenerate ONLY the metadata (no npz; the
     substrate cache provides X, it is never read from raw_dir in
     these runs) from the public benchmark (Harvard Dataverse,
     doi:10.7910/DVN/EBCFKM; `partition1..5_instances.tar.gz`):

     ```powershell
     # once per training partition (minutes each, writes p{p}_meta.csv only)
     python provenance/swansf_parse_partition.py <extracted>\partition1 <rawdir>\p1 --meta-only
     python provenance/swansf_parse_partition.py <extracted>\partition2 <rawdir>\p2 --meta-only
     python provenance/swansf_parse_partition.py <extracted>\partition3 <rawdir>\p3 --meta-only
     python provenance/swansf_parse_partition.py <extracted>\partition4 <rawdir>\p4 --meta-only
     ```
4. **GPU preferred** (the seed-42 control took 1,698.78 s for both
   arms on the owner GPU; CPU would work but is much slower).

## The two runs

```powershell
# Item 2 — region-disjoint validation carve, both arms, seed 42.
# ~1,700 s on the same GPU class as the original control.
python experiments/run_raw_nobn.py --region-disjoint --raw-dir <rawdir>

# Item 3 — seed-43 replication, both arms, frozen random carve.
# ~1,700 s likewise. (No --raw-dir needed — the frozen carve rides
# the substrate cache.)
python experiments/run_raw_nobn.py --seed 43
```

Both are round-resumable (state files `nobnrd_<algo>_state.pt` and
`nobns43_<algo>_state.pt` — separate namespaces; the original
`nobn_<algo>_state.pt` files are never touched, and the runner
refuses to resume a checkpoint from a foreign namespace or seed).
If a run is interrupted, just re-invoke the same command: it resumes
at the next round.

## What each run writes

- `outputs/raw_nobn_region_disjoint.json` — same schema as
  `raw_nobn_eval.json`, plus a `validation_split` block (region
  counts, val prevalence, disjointness statement, the v4.12.1
  `region_source` disclosure) and a `protocol.validation` marker.
- `outputs/raw_nobn_eval_seed43.json` — same schema, with
  `protocol.seed = 43` and the validation marker noting the reseed.

## Built-in guards (the run is self-checking)

- The raw parse metadata files are pre-flighted; a missing file is a
  guided exit naming `--raw-dir` and the regeneration command.
- The region carve re-derives the frozen random split and verifies it
  by a label round-trip against the cache; any mismatch aborts
  loudly rather than mis-splitting.
- Region disjointness is asserted, not assumed; `ar = -1` sentinel
  rows abort.
- The whole-region assignment is deterministic (seed-42 region
  permutation); reruns are identical.
- All of the above are exercised on synthetic fixtures by the battery
  (tests/test_region_disjoint.py — happy path + determinism, source
  contract, guided exit, round-trip tamper, ar sentinel; battery
  403/0 at v4.12.1).

## How to read the outcomes (decision rules)

- **Item 2, divergence disappears** (val ROC no longer rises while
  test falls; the shipped checkpoint is no longer the weakest test
  round): the random carve was region-leaky — the paper's validation
  story updates at v4.13, and the shipped numbers likely *improve*
  under the honest monitor.
- **Item 2, divergence persists**: partition-5 recency (temporal
  drift) becomes the leading explanation; the FedAvg declining
  trajectory is real client drift and the row stays a conservative
  lower bound.
- **Item 3, parity holds within ±0.005 ROC-AUC**: the level claims
  stand with replication.
- **Item 3, parity breaks**: the parity wording softens to
  "single-seed parity" and the divergence is reported.

## Reporting back

Paste (or commit and push, if running on the owner box directly):
1. both JSON files, verbatim;
2. the stdout tail of each run (the `[nobn]` summary lines).

Integration at v4.13 then follows the established convention: the
artefacts land byte-faithfully, get sha256-pinned in
`tests/test_raw_bn_diagnostic.py`, and the paper integrates the
verdicts (Sections 6.4/9 and the limitations re-run programme).
