#!/usr/bin/env python3
"""Aggregate per-partition match tables into pooled metadata.

The pipeline pools partitions in order 1..5 (np.vstack), so
pool_row = offset[partition] + pkl_row.

Outputs (/tmp/swansf_raw):
  test_meta_pooled.csv   pool_row, partition, pkl_row, ar, label,
                         flare tag, event_id, ts_start/end (min epoch),
                         verified
  train_meta_pooled.csv  same + region_id (-1 = synthetic/unmatched)
  region_overlap.json    train/test AR overlap audit (leakage quant)
"""
import numpy as np
import pandas as pd
import json
import os

RAW = os.environ.get("SWANSF_RAW", "/tmp/swansf_raw")  # v4.0: overridable
TEST_SIZES = {1: 73492, 2: 88557, 3: 42510, 4: 51261, 5: 75365}
TRAIN_SIZES = {1: 18773, 2: 19807, 3: 19965, 4: 19320, 5: 19899}


def build(pool, sizes):
    offs, off = {}, 0
    for p in sorted(sizes):
        offs[p] = off
        off += sizes[p]
    frames = []
    for p in sorted(sizes):
        d = pd.read_csv(f"{RAW}/match_{pool}_p{p}.csv", low_memory=False)
        assert len(d) == sizes[p], (p, len(d), sizes[p])
        d["partition"] = p
        d["pool_row"] = d["pkl_row"] + offs[p]
        frames.append(d)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled


def add_fields(d, pool):
    d["start_dt"] = pd.to_datetime(d["start"], errors="coerce")
    d["end_dt"] = pd.to_datetime(d["end"], errors="coerce")
    d["ts_end_min"] = (d["end_dt"].astype("int64") // 60_000_000_000)
    d["ts_start_min"] = (d["start_dt"].astype("int64") // 60_000_000_000)
    # event id: FL windows -> flare tag (class@peak:arN); background ->
    # (partition, ar, day)
    fl = d["label"] == 1
    d["event_id"] = np.where(
        fl,
        (d["flare_class"].astype(str) + "@" + d["peak"].fillna("").astype(str)
         + ":ar" + d["ar"].astype(str)),
        ("BG:p" + d["partition"].astype(str) + ":ar" + d["ar"].astype(str)
         + ":" + d["start_dt"].dt.strftime("%Y-%m-%d").fillna("na")))
    d["region_id"] = np.where(d["verified"], d["ar"], -1)
    return d


def main():
    te = build("test", TEST_SIZES)
    te = add_fields(te, "test")
    cols = ["pool_row", "partition", "pkl_row", "region_id", "ar", "label",
            "pkl_label", "label_agree", "verified", "cnt", "flare_class",
            "peak", "role", "event_id", "ts_start_min", "ts_end_min",
            "ts_first", "ts_last", "file"]
    te[cols].to_csv(f"{RAW}/test_meta_pooled.csv", index=False)
    print(f"[test] pooled {len(te)} rows; unique event groups "
          f"{te['event_id'].nunique()}; true-event groups "
          f"{te.loc[te['label']==1,'event_id'].nunique()}; "
          f"verified {te['verified'].mean():.2%}; "
          f"label agree {te['label_agree'].mean():.2%}")

    tr = build("train", TRAIN_SIZES)
    tr = add_fields(tr, "train")
    tr[cols].to_csv(f"{RAW}/train_meta_pooled.csv", index=False)
    print(f"[train] pooled {len(tr)} rows; verified {tr['verified'].mean():.2%}; "
          f"real positives {int(((tr['pkl_label']==1)&tr['verified']).sum())}; "
          f"synthetic positives "
          f"{int(((tr['pkl_label']==1)&~tr['verified']).sum())}")

    # AR overlap audit (train pool REAL windows vs test windows)
    tr_real = set(tr.loc[tr["verified"], "ar"].astype(int))
    te_ar = set(te.loc[te["verified"], "ar"].astype(int))
    ov = sorted(tr_real & te_ar)
    tr_windows_in_ov = int(tr.loc[tr["verified"] & tr["ar"].isin(ov)].shape[0])
    te_windows_in_ov = int(te.loc[te["verified"] & te["ar"].isin(ov)].shape[0])
    # also: ARs appearing in >1 test partition (test-internal overlap)
    per_part = te.loc[te["verified"]].groupby("ar")["partition"].nunique()
    multi = per_part[per_part > 1]
    report = {
        "n_train_ars_real": len(tr_real),
        "n_test_ars": len(te_ar),
        "n_shared_ars": len(ov),
        "shared_ar_list": ov,
        "train_windows_in_shared_ars": tr_windows_in_ov,
        "train_windows_frac": tr_windows_in_ov / len(tr),
        "test_windows_in_shared_ars": te_windows_in_ov,
        "test_windows_frac": te_windows_in_ov / len(te),
        "test_ars_in_multiple_partitions": int(len(multi)),
    }
    json.dump(report, open(f"{RAW}/region_overlap.json", "w"), indent=2)
    print(f"[overlap] train ARs {len(tr_real)}, test ARs {len(te_ar)}, "
          f"shared {len(ov)}; train windows in shared ARs "
          f"{tr_windows_in_ov} ({report['train_windows_frac']:.2%}); "
          f"test {te_windows_in_ov} ({report['test_windows_frac']:.2%}); "
          f"test ARs spanning >1 partition: {len(multi)}")


if __name__ == "__main__":
    main()
