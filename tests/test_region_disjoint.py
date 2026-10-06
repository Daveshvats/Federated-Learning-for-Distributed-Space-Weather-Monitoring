#!/usr/bin/env python3
"""
tests/test_region_disjoint.py — v4.12.1 kit-errata guard: the
region-disjoint split's data sources and failure paths.

The v4.12 kit shipped region_disjoint_split() reading its region map
from the SAMPLED audit meta (provenance/train_meta_slim.csv.gz —
97,764 rows spanning partitions 1..5, its pool_row indexing the slim
file's own order), so a real owner-side run would have aborted at the
coverage gate (97,764 of 255,820 pool rows) even with the raw dir
present — and the first owner execution died earlier still, on the
POSIX raw-dir default under Windows (a bare FileNotFoundError). This
module exercises the REAL split function on synthetic fixtures
(torch-free) so both defect classes stay closed:

  1. happy path — parse metas (label + ar) drive the split; the
     frozen random carve round-trips; no region spans the train/val
     boundary; the whole-region assignment is deterministic on rerun;
     the split stats carry the v4.12.1 region_source disclosure;
  2. source contract — the split reads `ar` from the parse metas via
     _load_region_ids and no longer accepts a meta_path (the
     slim-audit regression cannot silently return);
  3. guided exit — a missing p{p}_meta.csv is a SystemExit whose
     message names --raw-dir and the regeneration command, never a
     bare FileNotFoundError;
  4. round-trip tamper — cached labels disagreeing with the parse
     metas abort loudly;
  5. ar sentinel — ar = -1 rows abort (regions unassignable).

scikit-learn gates the module (the split re-derives the frozen carve
with sklearn.model_selection.train_test_split); a missing sklearn is
a declared [SKIP] with a clean exit, per the battery convention.

Run:  python tests/test_region_disjoint.py   (or via run_battery.py)
"""
import csv
import os
import re
import sys
import tempfile

import numpy as np

try:
    from sklearn.model_selection import train_test_split  # noqa: F401
except ImportError:
    print("[SKIP] tests/test_region_disjoint.py: scikit-learn not "
          "importable in this environment (the frozen-carve "
          "re-derivation needs it)")
    sys.exit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_raw_nobn import (        # noqa: E402
    TRAIN_PARTS, region_disjoint_split)
import config as cfg                          # noqa: E402

FAILED = []
N_CHECKS = 0


def check(name, cond):
    global N_CHECKS
    N_CHECKS += 1
    print(("[PASS] " if cond else "[FAIL] ") + name)
    if not cond:
        FAILED.append(name)


# ── fixtures: the synthetic parse metadata + the frozen carve ──────────

SIZES = {1: 500, 2: 400, 3: 300, 4: 200}      # pool rows per partition
N_REGIONS = 40                                 # synthetic region ids


def _synthetic_ar(n, offset=0):
    return ((np.arange(n) + offset) % N_REGIONS + 100).astype(np.int64)


def _write_parse_metas(tmp, ar_override=None):
    """p1..p4_meta.csv with label + ar, in pool order P1..P4."""
    rng = np.random.default_rng(0)
    y_parts, ar_parts = [], []
    for p in TRAIN_PARTS:
        n = SIZES[p]
        y = (rng.random(n) < 0.02).astype(np.float32)
        ar = _synthetic_ar(n)
        if ar_override is not None:
            ar = ar_override(p, ar)
        y_parts.append(y)
        ar_parts.append(ar)
        with open(os.path.join(tmp, f"p{p}_meta.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label", "ar"])
            w.writeheader()
            for i in range(n):
                w.writerow({"file": f"inst_p{p}_{i}.csv",
                            "label": "1" if y[i] else "0",
                            "ar": int(ar[i])})
    return np.concatenate(y_parts), np.concatenate(ar_parts)


def _carve(y_pool, seed=cfg.SEED):
    n = len(y_pool)
    val_idx = train_test_split(
        np.arange(n), test_size=cfg.VAL_SPLIT,
        random_state=seed, stratify=y_pool)[1]
    val_set = set(val_idx.tolist())
    train_idx = np.array([i for i in range(n) if i not in val_set])
    return train_idx, val_idx


def _cache_arrays(y_pool):
    """The cache as raw_substrate.build would have written it: X/y in
    carved order (the pool order is recoverable only via the split)."""
    rng = np.random.default_rng(1)
    X_pool = rng.random((len(y_pool), 10)).astype(np.float32)
    train_idx, val_idx = _carve(y_pool)
    return (X_pool[train_idx], y_pool[train_idx],
            X_pool[val_idx], y_pool[val_idx],
            X_pool, train_idx, val_idx)


def _exit_message(fn):
    """Run fn; return the SystemExit message, or a marker if the
    failure leaked as a different exception / no exit at all."""
    try:
        fn()
    except SystemExit as e:
        return str(e.code) if e.code is not None else ""
    except FileNotFoundError as e:
        return f"__BARE_FILE_NOT_FOUND__ {e}"
    except Exception as e:                     # noqa: BLE001
        return f"__WRONG_EXCEPTION__ {type(e).__name__}: {e}"
    return "__NO_EXIT__"


def main():
    tmp = tempfile.mkdtemp()

    # ── 1. happy path ─────────────────────────────────────────────────
    y_pool, region_of = _write_parse_metas(tmp)
    X_tr, y_tr, X_va, y_va, X_pool, train_idx, val_idx = \
        _cache_arrays(y_pool)

    out1 = region_disjoint_split(X_tr, y_tr, X_va, y_va, raw_dir=tmp,
                                 seed=cfg.SEED)
    X_tr2, y_tr2, X_va2, y_va2, stats = out1

    # regions never span the boundary (rows are unique -> byte lookup)
    pool_lookup = {X_pool[i].tobytes(): i for i in range(len(X_pool))}
    tr_regions = {int(region_of[pool_lookup[r.tobytes()]])
                  for r in X_tr2}
    va_regions = {int(region_of[pool_lookup[r.tobytes()]])
                  for r in X_va2}
    check("1a whole-region disjointness: no region id on both sides",
          tr_regions.isdisjoint(va_regions))
    check("1b label round-trip: every output label matches its pool "
          "row (train side)",
          all(y_pool[pool_lookup[r.tobytes()]] == lab
              for r, lab in zip(X_tr2, y_tr2)))
    check("1c label round-trip: every output label matches its pool "
          "row (val side)",
          all(y_pool[pool_lookup[r.tobytes()]] == lab
              for r, lab in zip(X_va2, y_va2)))
    check("1d stats row counts agree with the arrays",
          stats["train_rows"] == len(y_tr2)
          and stats["val_rows"] == len(y_va2))
    check("1e stats region counts agree with the derived sets",
          stats["n_regions_val"] == len(va_regions)
          and stats["n_regions_total"] == len(np.unique(region_of)))
    check("1f val size honours the random carve target (>=, whole "
          "regions only)",
          stats["val_rows"] >= stats["val_target_rows_random_carve"])
    check("1g stats disclose the v4.12.1 region source",
          "p1..p4_meta.csv" in stats.get("region_source", "")
          and "v4.12.1" in stats.get("region_source", ""))

    # determinism: identical outputs on rerun
    out2 = region_disjoint_split(X_tr, y_tr, X_va, y_va, raw_dir=tmp,
                                 seed=cfg.SEED)
    check("1h deterministic on rerun (arrays + stats identical)",
          np.array_equal(out1[0], out2[0])
          and np.array_equal(out1[3], out2[3])
          and out1[4] == out2[4])

    # ── 2. source contract: the region map is the parse metadata ────
    with open(os.path.join(ROOT, "experiments", "run_raw_nobn.py")) as f:
        src = f.read()
    m = re.search(r"def region_disjoint_split\(.*?(?=\ndef |\Z)",
                  src, re.S)
    func_src = m.group(0) if m else ""
    check("2a split source reads regions via _load_region_ids",
          "_load_region_ids(raw_dir, p)" in func_src)
    check("2b split accepts no meta_path (the slim-audit regression "
          "cannot return)",
          "meta_path" not in func_src
          and "train_meta_slim" not in func_src)
    check("2c split pre-flights the parse metas before any work",
          "_guided_raw_exit(raw_dir, missing)" in func_src)

    # ── 3. guided exit on missing raw metas ──────────────────────────
    empty = tempfile.mkdtemp()
    msg = _exit_message(lambda: region_disjoint_split(
        X_tr, y_tr, X_va, y_va, raw_dir=empty, seed=cfg.SEED))
    check("3a missing raw dir: SystemExit (not FileNotFoundError)",
          not msg.startswith("__"))
    check("3b guided message names --raw-dir and the regeneration "
          "command",
          "--raw-dir" in msg and "--meta-only" in msg
          and "swansf_parse_partition.py" in msg)

    # partial: only p1..p3 present -> exit names p4_meta.csv
    partial = tempfile.mkdtemp()
    for p in (1, 2, 3):
        with open(os.path.join(partial, f"p{p}_meta.csv"), "w",
                  newline="") as f:
            csv.DictWriter(
                f, fieldnames=["file", "label", "ar"]).writeheader()
    msg2 = _exit_message(lambda: region_disjoint_split(
        X_tr, y_tr, X_va, y_va, raw_dir=partial, seed=cfg.SEED))
    check("3c partial raw dir: exit names the missing p4_meta.csv",
          "p4_meta.csv" in msg2 and not msg2.startswith("__"))

    # ── 4. round-trip tamper: cache labels disagree with the metas ──
    y_tampered = y_tr.copy()
    y_tampered[0] = 1.0 - y_tampered[0]
    msg3 = _exit_message(lambda: region_disjoint_split(
        X_tr, y_tampered, X_va, y_va, raw_dir=tmp, seed=cfg.SEED))
    check("4 tampered cache labels abort loudly (round-trip gate)",
          "round-trip FAILED" in msg3)

    # ── 5. ar = -1 sentinel rows abort ───────────────────────────────
    tmp5 = tempfile.mkdtemp()
    _write_parse_metas(
        tmp5, ar_override=lambda p, ar:
        (np.where(np.arange(len(ar)) == 0, -1, ar).astype(np.int64)
         if p == 1 else ar))
    msg4 = _exit_message(lambda: region_disjoint_split(
        X_tr, y_tr, X_va, y_va, raw_dir=tmp5, seed=cfg.SEED))
    check("5 ar = -1 sentinel rows abort (regions unassignable)",
          "ar = -1" in msg4)

    print(f"RESULT: {N_CHECKS - len(FAILED)} passed, "
          f"{len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
