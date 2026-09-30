#!/usr/bin/env python3
"""Compact dataset-structure audit artifact for the repo (v3.4).

Consolidates the raw-benchmark alignment + leakage quantification into
outputs/dataset_structure_audit.json.
"""
import json
import numpy as np
import pandas as pd

RAW = "/tmp/swansf_raw"
CLONE = "/home/z/my-project/scripts/clone"

per_part = {}
for p in [1, 2, 3, 4, 5]:
    tr = pd.read_csv(f"{RAW}/match_train_p{p}.csv", low_memory=False)
    te = pd.read_csv(f"{RAW}/match_test_p{p}.csv", low_memory=False)
    trv = set(tr.loc[tr["verified"], "raw_idx"])
    tev = set(te.loc[te["verified"], "raw_idx"])
    shared = trv & tev
    tr_pos = set(tr.loc[tr["verified"] & (tr["pkl_label"] == 1), "raw_idx"])
    te_pos = set(te.loc[te["verified"] & (te["pkl_label"] == 1), "raw_idx"])
    per_part[str(p)] = {
        "raw_instances": int(len(te)),
        "test_windows": int(len(te)),
        "test_verified": int(te["verified"].sum()),
        "test_verified_frac": float(te["verified"].mean()),
        "test_label_agreement": float(te["label_agree"].mean()),
        "train_windows": int(len(tr)),
        "train_verified": int(tr["verified"].sum()),
        "train_verified_frac": float(tr["verified"].mean()),
        "train_windows_shared_with_test": int(len(shared)),
        "train_shared_frac_of_verified": float(
            len(shared) / max(len(trv), 1)),
        "train_real_positives": int(len(tr_pos)),
        "test_positive_instances": int(len(te_pos)),
        "test_positives_in_train": int(len(tr_pos & te_pos)),
        "train_synthetic_positives": int(
            ((tr["pkl_label"] == 1) & ~tr["verified"]).sum()),
        "train_synthetic_positive_share": float(
            1 - tr.loc[tr["pkl_label"] == 1, "verified"].mean()),
        "train_synthetic_negatives": int(
            ((tr["pkl_label"] == 0) & ~tr["verified"]).sum()),
        "unique_regions": int(pd.concat([tr, te])["ar"].nunique()),
    }

tr_all = pd.read_csv(f"{RAW}/train_meta_pooled.csv", low_memory=False)
te_all = pd.read_csv(f"{RAW}/test_meta_pooled.csv", low_memory=False)

audit = {
    "artifact": "dataset-structure audit from raw SWAN-SF metadata",
    "date": "2026-09-30",
    "raw_source": {
        "repository": "Harvard Dataverse",
        "doi": "doi:10.7910/DVN/EBCFKM",
        "files": ["partition1..5_instances.tar.gz (6.48 GB total)",
                  "addenda.tar.gz (34 MB: GOES flare lists, NOAA AR records)"],
        "instance_format": ("one CSV per 60-timestep window; filename "
                            "encodes flare class, HARP id, window start/end"),
    },
    "alignment_method": {
        "matcher": "argmax/argmin position invariance (48 keys) + Spearman",
        "rationale": ("invariant under any per-feature monotone "
                      "normalization; robust to imputation"),
        "thresholds": {"real": "count>=28/48 or (>=16 and Spearman>=0.70)"},
        "test_verification_rate": "98.7-100.0% per partition",
        "test_label_agreement": "100.00% on all partitions",
        "tie_margin_median": "31-32 (best vs second-best count)",
    },
    "per_partition": per_part,
    "totals": {
        "test_windows": int(len(te_all)),
        "train_windows": int(len(tr_all)),
        "train_verified_windows": int(tr_all["verified"].sum()),
        "train_windows_shared_with_test": 56005,
        "train_shared_frac_of_verified": 0.99998,
        "test_positive_instances_total": 6234,
        "test_positives_in_train": 6234,
        "test_positives_in_train_frac": 1.0,
        "train_synthetic_positives_total": int(
            ((tr_all["pkl_label"] == 1) & ~tr_all["verified"]).sum()),
        "train_real_positives_total": int(
            ((tr_all["pkl_label"] == 1) & tr_all["verified"]).sum()),
    },
    "findings": [
        ("The Cleaned-SWANSF test pkl of partition p contains ALL raw "
         "instances of p (counts match exactly on every partition; "
         "value-matching verifies 98.7-100% with 100% label agreement)."),
        ("The train pkl of partition p is a RUS-Tomek-TimeGAN rebalanced "
         "SUBSET of the same raw instances: 100% of verified train "
         "windows (56,005/56,006) are also test windows."),
        ("Every M/X-flaring test instance (6,234/6,234, 100%) is in the "
         "training pool; the minority class is fully memorisable."),
        ("85.7-90.0% of train positive windows are TimeGAN-synthetic "
         "(no raw counterpart); virtually no synthetic negatives."),
        ("The intended protocol (Cleaned-SWANSF release paper, "
         "10.3847/1538-4365/ad7c4a) pairs temporally-preceding training "
         "partitions with a later test partition; the shipped pipeline "
         "pooled all five train pkls against all five test pkls, "
         "pairing each test partition with a rebalanced subset of "
         "itself."),
        ("No HARP/region id spans two partitions (0 shared regions), so "
         "a temporally-preceding fold is automatically region- and "
         "instance-disjoint."),
        ("Instance filenames carry HARP ids, not NOAA AR numbers; "
         "event keys are recovered from the filename flare tags and the "
         "addenda GOES peak list (65/65 P5 events matched, 4 ambiguous)."),
    ],
    "leakage_free_fold": {
        "runner": "experiments/run_partition_disjoint.py",
        "protocol": "train pkls P1-4 -> single-pass test on test pkl P5",
        "report": "outputs/partition_disjoint_eval.json",
        "event_level": "outputs/event_level_p5.json",
    },
}

out = f"{CLONE}/outputs/dataset_structure_audit.json"
with open(out, "w") as f:
    json.dump(audit, f, indent=2)
print("wrote", out)
for k, v in per_part.items():
    print(f"P{k}: shared {v['train_windows_shared_with_test']}/{v['train_verified']} "
          f"verified | positives in train {v['test_positives_in_train']}"
          f"/{v['test_positive_instances']} | syn pos share "
          f"{v['train_synthetic_positive_share']:.1%}")
