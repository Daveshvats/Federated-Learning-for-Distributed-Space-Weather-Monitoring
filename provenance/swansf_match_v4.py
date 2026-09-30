#!/usr/bin/env python3
"""Align Cleaned-SWANSF pkl windows to raw instances (v4 — invariant space).

Retrieval AND matching in argmax/argmin-position space (48 binary keys,
each invariant under any per-feature monotone normalization):
  count(i, j) = #features where argmax positions agree + same for argmin
  -> best match = argmax_j count(i, j)   (chunked, vectorised)

Verification:
  real:        count >= 28/48
  borderline:  16 <= count < 28 AND mean per-feature Spearman >= 0.70
  else:        synthetic (train) / unmatched (test)

Usage: swansf_match_v4.py <part> <pool:test|train> <out_csv>
"""
import sys, time, pickle
import numpy as np
import pandas as pd

RAW = "/tmp/swansf_raw"
CLONE = "/home/z/my-project/scripts/clone/data/cleaned"
REAL_CNT = 28
BORDER_CNT = 16
SPEAR_THR = 0.70


def load_cleaned(part, split):
    tag = ("RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute" if split == "train"
           else "LSBZM-Norm_FPCKNN-impute")
    X = pickle.load(open(f"{CLONE}/{split}/Partition{part}_{tag}.pkl", "rb"))
    y = pickle.load(open(f"{CLONE}/{split}/Partition{part}_Labels_{tag}.pkl", "rb"))
    return np.asarray(X, dtype=np.float32), np.asarray(y).astype(int)


def argext(X, block=6):
    """argmax/argmin along time axis per feature; chunked over features to
    bound temporaries (RAM is only 4 GB)."""
    n = X.shape[0]
    amax = np.empty((n, 24), np.int16)
    amin = np.empty((n, 24), np.int16)
    for s in range(0, 24, block):
        e = min(s + block, 24)
        B = X[:, :, s:e]
        fm = np.where(np.isnan(B), -np.inf, B)
        fn = np.where(np.isnan(B), np.inf, B)
        allnan = np.isnan(B).all(axis=1)
        amax[:, s:e] = np.where(allnan, -1, fm.argmax(axis=1))
        amin[:, s:e] = np.where(allnan, 999, fn.argmin(axis=1))
    return amax, amin


def hamming_best(cmax, cmin, rmax, rmin, chunk_m=512):
    n, m = len(cmax), len(rmax)
    best_cnt = np.zeros(n, np.int16)
    best_j = np.zeros(n, np.int32)
    second_cnt = np.zeros(n, np.int16)
    rows = np.arange(n)
    for s in range(0, m, chunk_m):
        e = min(s + chunk_m, m)
        cnt = np.zeros((n, e - s), np.int16)
        for f in range(24):
            cnt += (rmax[s:e, f][None, :] == cmax[:, f][:, None])
            cnt += (rmin[s:e, f][None, :] == cmin[:, f][:, None])
        j1 = cnt.argmax(axis=1)
        c1 = cnt.max(axis=1)
        prev_best = best_cnt.copy()
        displ = c1 > prev_best
        best_cnt = np.where(displ, c1, prev_best)
        best_j = np.where(displ, s + j1, best_j)
        # second-best: max of (displaced prev best | non-displacing chunk
        # best) and the within-chunk runner-up
        second_cnt = np.maximum(
            second_cnt, np.where(displ, prev_best, c1))
        cnt[rows, j1] = -1
        second_cnt = np.maximum(second_cnt, cnt.max(axis=1))
    return best_cnt, best_j, second_cnt


def spearman_mean(a_win, b_win):
    cors = []
    for f in range(24):
        a, b = a_win[:, f], b_win[:, f]
        m = ~np.isnan(b)
        a, b = a[m], b[m]
        if len(a) < 12:
            continue
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        if ra.std() == 0 or rb.std() == 0:
            continue
        cors.append(np.corrcoef(ra, rb)[0, 1])
    return float(np.mean(cors)) if cors else np.nan


def main():
    part, pool, out_csv = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    t0 = time.time()
    Xr = np.load(f"{RAW}/p{part}_raw.npz")["X"]  # float32, (m,60,24)
    meta = pd.read_csv(f"{RAW}/p{part}_meta.csv", low_memory=False).reset_index(names="raw_idx")
    Xc, y = load_cleaned(part, pool)
    n, m = len(Xc), len(Xr)
    print(f"[load] p{part} {pool}: cleaned {n}, raw {m}")

    cmax, cmin = argext(Xc)
    rmax, rmin = argext(Xr)
    bc, bj, sc = hamming_best(cmax, cmin, rmax, rmin)
    print(f"[match] best-cnt: median {np.median(bc):.0f} p10 "
          f"{np.percentile(bc,10):.0f} p90 {np.percentile(bc,90):.0f} "
          f"({time.time()-t0:.0f}s)")

    ver = np.zeros(n, bool)
    spear = np.full(n, np.nan)
    for i in np.where((bc >= BORDER_CNT) & (bc < REAL_CNT))[0]:
        spear[i] = spearman_mean(Xc[i], Xr[bj[i]])
        ver[i] = bool(spear[i] >= SPEAR_THR) if not np.isnan(spear[i]) else False
    ver[bc >= REAL_CNT] = True

    out = pd.DataFrame({"pkl_row": np.arange(n), "raw_idx": bj, "cnt": bc,
                        "cnt_second": sc, "spearman": spear, "verified": ver})
    out = out.merge(meta, on="raw_idx", how="left")
    out["pkl_label"] = y
    out["label_agree"] = (out["label"] == out["pkl_label"])
    out.to_csv(out_csv, index=False)
    v = out["verified"]
    print(f"[verify] {v.sum()}/{n} ({v.mean()*100:.1f}%) verified; "
          f"label agree (verified) {out.loc[v,'label_agree'].mean()*100:.2f}%; "
          f"label agree (all) {out['label_agree'].mean()*100:.2f}%; "
          f"tie margin (cnt - 2nd) median "
          f"{np.median(out['cnt'] - out['cnt_second']):.0f}; "
          f"{time.time()-t0:.0f}s total")
    if pool == "train":
        pos = out["pkl_label"] == 1
        print(f"[train] synthetic share: all {1-v.mean():.1%} | "
              f"positives {1-v[pos].mean():.1%} | negatives {1-v[~pos].mean():.1%}")
        vr = out[v]
        print(f"[train] verified cover {vr['ar'].nunique()} ARs; "
              f"verified label-1 windows {int((vr['pkl_label']==1).sum())} "
              f"of {int(pos.sum())} positives")


if __name__ == "__main__":
    main()
