# Raw-SWAN-SF provenance audit (v3.4)

Scripts and compact artefacts for aligning the Cleaned SWAN-SF export
to the raw benchmark and quantifying the train/test instance overlap.
Full narrative: paper Sec. 5.7 (data-provenance audit) and Sec. 6.1
(leakage-free evaluation); numbers:
`outputs/dataset_structure_audit.json`.

## Pipeline

1. `swansf_parse_partition.py <partition_dir> <out_prefix>` —
   extract the 24 SHARP parameters (in the cleaned attribute order)
   from every raw instance CSV + filename metadata (HARP id, window
   start/end, flare tag, FL/NF label folder). Raw source:
   Harvard Dataverse `doi:10.7910/DVN/EBCFKM`
   (`partition1..5_instances.tar.gz`, 6.5 GB; `addenda.tar.gz`, 34 MB).
2. `swansf_match_v4.py <part> <test|train> <out.csv>` — align every
   cleaned pkl window to a raw instance. Matcher: argmax/argmin
   position along the 60 timesteps per feature (48 invariants under
   any per-feature monotone normalization). Real match: >= 28/48, or
   >= 16/48 with mean per-feature Spearman >= 0.70.
   Verification: test pools 98.7-100.0% verified, 100.00% label
   agreement, median tie margin 31-32.
3. `swansf_verify_leakage.py` — instance-level overlap between the
   train and test exports (the audit's decisive check).
4. `swansf_aggregate_meta.py` — pooled per-window metadata (event
   keys, timestamps, region ids) in pipeline row order.
5. `swansf_audit_artifact.py` — compact audit JSON for the repo.
6. `swansf_event_level.py` — event-level evaluation on the
   leakage-free fold (uses `experiments/run_event_level.py`).

## Committed artefacts

- `test_meta_slim.csv.gz` / `train_meta_slim.csv.gz` — per-window
  alignment in pooled pipeline order (partition 1..5 vstack):
  region (HARP) id, verified flag, event key, window start/end
  (minutes since epoch).
- `region_overlap.json` — train/test region overlap audit.

## Reproduce

The raw partition archives are public (link above). Download, then:

    python3 provenance/swansf_parse_partition.py <dir>/partition1 p1
    python3 provenance/swansf_match_v4.py 1 test  match_test_p1.csv
    python3 provenance/swansf_match_v4.py 1 train match_train_p1.csv
    ... (partitions 2-5)
    python3 provenance/swansf_aggregate_meta.py
    python3 provenance/swansf_verify_leakage.py

The leakage-free fold itself is
`experiments/run_partition_disjoint.py` (round-resumable; artefacts
`outputs/partition_disjoint_eval.json`, `outputs/event_level_p5.json`).
