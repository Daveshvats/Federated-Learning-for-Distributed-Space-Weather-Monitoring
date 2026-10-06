#!/usr/bin/env python3
"""tests/test_region_meta.py — the repo-shipped raw parse metadata
(the v4.12 external-review item-2 prerequisite, landed v4.13.2).

The region-disjoint re-run needs the parse metadata p1..p4_meta.csv
(labels + the NOAA-AR `ar` column) that the owner's original
substrate build consumed — the owner-box parse files were lost with
the WSL /tmp wipe (the committed box inventory records their
absence). v4.13.2 ships the REGENERATED metadata at raw/p{1..4}_meta.csv,
produced by streaming the public benchmark archives (Harvard
Dataverse, doi:10.7910/DVN/EBCFKM; partition1..4_instances.tar.gz —
4,407,825,217 bytes total, each archive's exact byte size verified
at download) through the parse semantics of
provenance/swansf_parse_partition.py (identical schema, filename
regexes, and FL-then-NF sorted job order), then verified three
independent ways at generation time:
  1. per-partition counts vs the committed audit
     (outputs/dataset_structure_audit.json): instances, FL
     positives, unique regions (the RAW-parse region counts; the
     audit's MATCHED-view counts are 3 lower on P2 and 1 lower on
     P4 — unmatched windows hide small regions from the matched
     view — P1/P3 agree exactly);
  2. pool totals: 255,820 rows / 5,244 positives / 2.0499%
     prevalence — the substrate's own recorded 2.05%;
  3. the slim-meta join: every verified P1-4 window of the ORIGINAL
     parse (provenance/train_meta_slim.csv.gz — 45,194 verified
     rows) joins the regenerated metadata on (ar, window start/end)
     with 100% label agreement — the original parse's own record,
     re-encountered row by row.

With this module the item-2 run is zero-setup on the owner box:
git pull, then `python experiments/run_raw_nobn.py
--region-disjoint --raw-dir raw` (the run card's own command). The
runner's own guards still apply on top (pool-size, carve
round-trip, region coverage, disjointness assert).

Run:  python tests/test_region_meta.py   (or via the battery)
"""
import csv
import gzip
import hashlib
import os
import sys
from calendar import timegm
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
SLIM = os.path.join(ROOT, "provenance", "train_meta_slim.csv.gz")

FILES = {p: os.path.join(RAW, f"p{p}_meta.csv") for p in (1, 2, 3, 4)}

# byte-pins of the regenerated parse metadata (LF form, as committed)
SHA256 = {
    1: "6d67dfcf2b56384e9d91e72e0831688b05062a53d6de8f168c578e9fc6ae29d9",
    2: "55bc90b7012db5024e4c70a3f814e071e2f16971178e916a3b1828eec782ef36",
    3: "3cdecda29a52d375aae77b47ff07a971c7f8816e41200edb089a3c4e02d95517",
    4: "9509dfbb28ed6bd8963d17525484bc9d140e9461daddcfdb5e94a41140bf5036",
}

# the committed audit's per-partition truth (RAW-parse region counts)
AUDIT = {
    1: dict(raw_instances=73492, positives=1254, unique_regions=700),
    2: dict(raw_instances=88557, positives=1401, unique_regions=885),
    3: dict(raw_instances=42510, positives=1424, unique_regions=390),
    4: dict(raw_instances=51261, positives=1165, unique_regions=472),
}
POOL_ROWS, POOL_POS = 255820, 5244
SLIM_VERIFIED_P12 = 45194   # verified P1-4 rows of the original parse

N_CHECKS = 0
FAILED = []


def check(name, cond, note=""):
    global N_CHECKS
    N_CHECKS += 1
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" {note}" if note else ""))
    if not cond:
        FAILED.append(name)


def _minutes(iso):
    """ISO window edge -> epoch minutes (pandas-naive semantics)."""
    return timegm(datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S")
                  .timetuple()) // 60


def main():
    print("layer 1 — the shipped parse metadata (sha256 + counts + order):")
    keys = {}
    tot_rows = tot_pos = 0
    for p in (1, 2, 3, 4):
        path = FILES[p]
        a = AUDIT[p]
        if not os.path.exists(path):
            check(f"p{p}: raw/p{p}_meta.csv is committed", False)
            continue
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read().replace(b"\r\n",
                                                     b"\n")).hexdigest()
        check(f"p{p}: sha256 byte-pin", digest == SHA256[p],
              f"(got {digest[:16]}…)")
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
            cols = rdr.fieldnames
            rows = list(rdr)
        check(f"p{p}: schema exact (the parse script's 11 columns)",
              cols == ["file", "label", "flare_class", "peak", "role",
                       "ar", "start", "end", "nrows", "ts_first",
                       "ts_last"])
        check(f"p{p}: instance count == audit ({a['raw_instances']:,})",
              len(rows) == a["raw_instances"], f"({len(rows):,})")
        pos = sum(1 for r in rows if r["label"] == "1")
        check(f"p{p}: FL positives == audit ({a['positives']:,})",
              pos == a["positives"], f"({pos:,})")
        ars = {int(r["ar"]) for r in rows}
        check(f"p{p}: unique ar == the raw-parse count "
              f"({a['unique_regions']:,})", len(ars) == a["unique_regions"],
              f"({len(ars):,})")
        check(f"p{p}: no ar = -1 (every filename parsed)",
              -1 not in ars)
        labels = [r["label"] for r in rows]
        files = [r["file"] for r in rows]
        n_fl = labels.count("1")
        check(f"p{p}: row order = FL block then NF block, each "
              f"filename-sorted (the parse job order)",
              labels == ["1"] * n_fl + ["0"] * (len(labels) - n_fl) and
              files[:n_fl] == sorted(files[:n_fl]) and
              files[n_fl:] == sorted(files[n_fl:]))
        check(f"p{p}: label column values are the runner's 1/0 strings",
              set(labels) <= {"0", "1"})
        for r in rows:
            keys[(p, int(r["ar"]), _minutes(r["start"]),
                  _minutes(r["end"]))] = int(r["label"])
        tot_rows += len(rows)
        tot_pos += pos

    print("layer 2 — the pool contract (the substrate's own record):")
    check("pool rows == 255,820 (P1-4, the substrate pool)",
          tot_rows == POOL_ROWS, f"({tot_rows:,})")
    check("pool positives == 5,244 (the audit total)",
          tot_pos == POOL_POS, f"({tot_pos:,})")
    check("pool prevalence == 2.0499% (the substrate's 2.05%)",
          abs(tot_pos / tot_rows - POOL_POS / POOL_ROWS) < 1e-12)

    print("layer 3 — the slim-meta join (the ORIGINAL parse, "
          "re-encountered):")
    n_join = n_ok = n_v = 0
    with gzip.open(SLIM, "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["region_id"] == "-1" or int(r["partition"]) not in (1, 2, 3, 4):
                continue
            n_v += 1
            k = (int(r["partition"]), int(r["ar"]),
                 int(r["ts_start_min"]), int(r["ts_end_min"]))
            if k in keys:
                n_join += 1
                if keys[k] == int(r["label"]):
                    n_ok += 1
    check(f"every verified P1-4 slim row joins ({SLIM_VERIFIED_P12:,})",
          n_join == n_v == SLIM_VERIFIED_P12,
          f"({n_join:,} / {n_v:,})")
    check("joined labels agree 100% (the audit's own standard)",
          n_ok == n_join, f"({n_ok:,} / {n_join:,})")

    print("layer 4 — the runner contract (what --region-disjoint reads):")
    check("the four files sit exactly where the run card's "
          "--raw-dir raw points (raw/p{1..4}_meta.csv)",
          all(os.path.exists(FILES[p]) for p in (1, 2, 3, 4)))
    sys.path.insert(0, ROOT)
    import experiments.run_raw_nobn as runner  # noqa: E402
    preflight_missing = [fp for fp in
                         (os.path.join(RAW, f"p{p}_meta.csv")
                          for p in runner.TRAIN_PARTS)
                         if not os.path.exists(fp)]
    check("the runner's own pre-flight finds no missing file",
          preflight_missing == [])
    rid = runner._load_region_ids(RAW, 3)
    check("_load_region_ids reads the shipped metadata (dtype/length)",
          rid.dtype == __import__("numpy").int64 and len(rid) == 42510)
    y3 = runner.load_labels(RAW, 3)
    check("load_labels round-trips the shipped labels (1,424 "
          "positives)", int(y3.sum()) == 1424)

    print(f"RESULT: {N_CHECKS - len(FAILED)} passed, "
          f"{len(FAILED)} failed")
    return not FAILED


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
