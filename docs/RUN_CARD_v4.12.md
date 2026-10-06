# RUN CARD v4.12 — the external-review compute items (2) and (3)

The independent re-review of v4.11 asked for two small compute jobs
that this sandbox cannot run (no GPU, no dataset cache). Both are
prepared, guarded, and namespaced so they cannot disturb the frozen
v4.9.4 artefacts. This card is the exact recipe.

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

- The repo at v4.12 (branch `improvements`), working tree clean.
- The raw benchmark at `/tmp/swansf_raw` (p1..p5_raw.npz + the
  `p{p}_meta.csv` parse metadata) — the same directory the substrate
  was built from.
- The substrate cache `data/cache/rawsubstrate/data.npz` present
  (it is; the seed-43 run refuses to start without it, by design).
- GPU preferred (the seed-42 control took 1,698.78 s for both arms
  on the owner GPU; CPU would work but is much slower).

## The two runs

```bash
# Item 2 — region-disjoint validation carve, both arms, seed 42.
# ~1,700 s on the same GPU class as the original control.
python experiments/run_raw_nobn.py --region-disjoint

# Item 3 — seed-43 replication, both arms, frozen random carve.
# ~1,700 s likewise.
python experiments/run_raw_nobn.py --seed 43
```

Both are round-resumable (state files `nobnrd_<algo>_state.pt` and
`nobn_s43_<algo>_state.pt` — separate namespaces; the original
`nobn_<algo>_state.pt` files are never touched, and the runner
refuses to resume a checkpoint from a foreign namespace or seed).

## What each run writes

- `outputs/raw_nobn_region_disjoint.json` — same schema as
  `raw_nobn_eval.json`, plus a `validation_split` block (region
  counts, val prevalence, disjointness statement) and a
  `protocol.validation` marker.
- `outputs/raw_nobn_eval_seed43.json` — same schema, with
  `protocol.seed = 43` and the validation marker noting the reseed.

## Built-in guards (the run is self-checking)

- The region carve re-derives the frozen random split and verifies it
  by a label round-trip against the cache; any mismatch aborts
  loudly rather than mis-splitting.
- Region disjointness is asserted, not assumed.
- The whole-region assignment is deterministic (seed-42 region
  permutation); reruns are identical.
- The provenance meta must cover every pool row, or the run aborts.

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
