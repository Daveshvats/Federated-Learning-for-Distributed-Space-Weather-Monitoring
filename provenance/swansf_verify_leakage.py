#!/usr/bin/env python3
"""Rigorous verification: do the Cleaned-SWANSF train and test pkls share
the SAME instances?

Test 1: instance-level overlap — raw_idx sets of verified train matches vs
        test matches (per partition, positives/negatives separately).
Test 2: value identity — for shared instances, compare the train pkl window
        and the test pkl window directly (same normalization -> identical
        values? different stats -> monotone relation).
Test 3: label identity on shared instances.
"""
import pickle
import numpy as np
import pandas as pd

RAW = "/tmp/swansf_raw"
CLONE = "/home/z/my-project/scripts/clone/data/cleaned"

print("=== Test 1: instance-level overlap (matched raw_idx) ===")
tot = {}
for p in [1, 2, 3, 4, 5]:
    tr = pd.read_csv(f"{RAW}/match_train_p{p}.csv", low_memory=False)
    te = pd.read_csv(f"{RAW}/match_test_p{p}.csv", low_memory=False)
    trv = set(tr.loc[tr["verified"], "raw_idx"])
    tev = set(te.loc[te["verified"], "raw_idx"])
    shared = trv & tev
    tr_pos = tr.loc[tr["verified"] & (tr["pkl_label"] == 1), "raw_idx"]
    te_pos = te.loc[te["verified"] & (te["pkl_label"] == 1), "raw_idx"]
    pos_shared = set(tr_pos) & set(te_pos)
    tot[p] = (len(trv), len(shared), len(pos_shared))
    print(f"P{p}: train-verified {len(trv):6d} | shared-with-test "
          f"{len(shared):6d} ({len(shared)/max(len(trv),1)*100:5.1f}%) | "
          f"positive instances: train {len(set(tr_pos)):4d}, "
          f"test {len(set(te_pos)):4d}, shared {len(pos_shared):4d}")
print("TOTAL train-verified:", sum(t[0] for t in tot.values()),
      "| shared:", sum(t[1] for t in tot.values()),
      "| shared positives:", sum(t[2] for t in tot.values()))

print("\n=== Test 2: value identity on shared instances (P3) ===")
Xtr = np.asarray(pickle.load(open(
    f"{CLONE}/train/Partition3_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl", "rb")), np.float32)
Xte = np.asarray(pickle.load(open(
    f"{CLONE}/test/Partition3_LSBZM-Norm_FPCKNN-impute.pkl", "rb")), np.float32)
tr = pd.read_csv(f"{RAW}/match_train_p3.csv", low_memory=False)
te = pd.read_csv(f"{RAW}/match_test_p3.csv", low_memory=False)
tr_v = tr.loc[tr["verified"]].set_index("raw_idx")
te_v = te.loc[te["verified"]].set_index("raw_idx")
shared = sorted(set(tr_v.index) & set(te_v.index))[:200]
maxd, nsame = 0.0, 0
for j in shared:
    a = Xtr[tr_v.loc[j, "pkl_row"]]     # train window of raw instance j
    b = Xte[te_v.loc[j, "pkl_row"]]     # test window of raw instance j
    d = float(np.abs(a - b).max())
    maxd = max(maxd, d)
    nsame += int(np.allclose(a, b, atol=1e-5))
print(f"200 shared instances: identical (atol 1e-5): {nsame}/200; "
      f"max |train - test| value diff: {maxd:.6f}")

print("\n=== Test 3: duplicates WITHIN pools (same instance, two rows) ===")
for pool in ["train", "test"]:
    d = pd.read_csv(f"{RAW}/match_{pool}_p3.csv", low_memory=False)
    v = d.loc[d["verified"], "raw_idx"]
    print(f"P3 {pool}: {len(v)} verified, {v.duplicated().sum()} duplicate "
          f"raw_idx claims")
