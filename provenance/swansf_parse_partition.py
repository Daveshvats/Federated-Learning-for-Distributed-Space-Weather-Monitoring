#!/usr/bin/env python3
"""Parse raw SWAN-SF partition instances -> compact npz + metadata csv.

Each instance file: 55 tab-separated columns, 1 header + 60 data rows.
Extracts the 24 SHARP parameters (in the Cleaned-SWANSF attribute order)
plus metadata from the filename (label folder, flare class, NOAA AR,
window start/end).

Usage: python3 swansf_parse_partition.py <partition_dir> <out_prefix>
  e.g. swansf_parse_partition.py /tmp/swansf_raw/p3/partition3 /tmp/swansf_raw/p3
"""
import os, sys, re, csv, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Cleaned-SWANSF attribute order (24 features)
CLEANED_ORDER = ['R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ',
                 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT',
                 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM',
                 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY',
                 'MEANJZD', 'MEANALP', 'TOTFX']

# FL:    M1.0@6861:Primary_ar3311_s..._e...csv   (M/X-class)
# NF:    B3.9@8351:Primary_ar4186_s..._e...csv   (B/C-class only)
# NF:    FQ_ar3293_s..._e...csv                  (flare-quiet)
FNAME_RE_FL = re.compile(
    r"^(?P<class>[A-Z][0-9.]+)@(?P<peak>\d+):(?P<role>\w+)_ar(?P<ar>\d+)"
    r"_s(?P<s>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"_e(?P<e>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.csv$")
FNAME_RE_FQ = re.compile(
    r"^FQ_ar(?P<ar>\d+)"
    r"_s(?P<s>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"_e(?P<e>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.csv$")

# sentinels in SWAN-SF raw: -9999 and 1e30 family plus empty
def _to_float(s):
    if not s:
        return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    if v <= -9000 or v >= 1e29:
        return np.nan
    return v


def parse_one(args):
    path, label, ncol = args
    base = os.path.basename(path)
    m = FNAME_RE_FL.match(base)
    if m:
        meta = {"file": base, "label": label,
                "flare_class": m.group("class"), "peak": m.group("peak"),
                "role": m.group("role"), "ar": int(m.group("ar")),
                "start": m.group("s"), "end": m.group("e")}
    else:
        m = FNAME_RE_FQ.match(base)
        meta = {"file": base, "label": label, "flare_class": "FQ",
                "peak": "", "role": "",
                "ar": int(m.group("ar")) if m else -1,
                "start": m.group("s") if m else "",
                "end": m.group("e") if m else ""}
    rows = []
    with open(path, "r", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_idx = [header.index(c) for c in CLEANED_ORDER]
        ts_idx = header.index("Timestamp")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < ncol:
                continue
            ts = parts[ts_idx]
            vals = [_to_float(parts[ci]) for ci in col_idx]
            rows.append((ts, vals))
    meta["nrows"] = len(rows)
    meta["ts_first"] = rows[0][0] if rows else ""
    meta["ts_last"] = rows[-1][0] if rows else ""
    arr = np.full((60, 24), np.nan, np.float64)
    for i, (_, vals) in enumerate(rows[:60]):
        arr[i] = vals
    return meta, arr


def main():
    pdir = sys.argv[1].rstrip("/")          # .../partition3
    out = sys.argv[2]                        # output prefix
    jobs = []
    for folder, label in (("FL", 1), ("NF", 0)):
        d = os.path.join(pdir, folder)
        for fn in sorted(os.listdir(d)):
            jobs.append((os.path.join(d, fn), label, 55))
    print(f"[parse] {len(jobs)} instance files from {pdir}")

    t0 = time.time()
    metas, arrs = [], []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for meta, arr in ex.map(parse_one, jobs, chunksize=200):
            metas.append(meta)
            arrs.append(arr)
    X = np.stack(arrs).astype(np.float32)    # (n, 60, 24)
    print(f"[parse] done in {time.time()-t0:.0f}s -> {X.shape}; "
          f"NaN frac {np.isnan(X.astype(np.float64)).mean():.4f}")

    # row-count audit
    nrows = np.array([m["nrows"] for m in metas])
    print(f"[parse] rows/file: min={nrows.min()} max={nrows.max()} "
          f"!=60: {(nrows != 60).sum()}")

    np.savez_compressed(out + "_raw.npz", X=X)
    import pandas as pd
    pd.DataFrame(metas).to_csv(out + "_meta.csv", index=False)
    print(f"[parse] saved {out}_raw.npz + {out}_meta.csv")


if __name__ == "__main__":
    main()
