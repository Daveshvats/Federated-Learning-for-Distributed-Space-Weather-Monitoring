"""
leakage_audit/audit_leakage.py
──────────────────────────────
Runtime leakage audit (Gate 1 pass criterion, audit findings B2/B10/B13).

Checks, in order of severity:
  1. SHARD DISJOINTNESS  — no sample index may appear in two client shards
  2. SHARD COVERAGE      — every training sample assigned exactly once
  3. SPLIT EXCLUSIVITY   — train / validation / test index sets must not
                           intersect (by construction, but verified here so
                           future refactors cannot silently break the contract)
  4. REGION LEAKAGE      — if active-region IDs are available, no region may
                           span an evaluation boundary (train/val vs test)
  5. WINDOW OVERLAP      — if window sample IDs / timestamps are available,
                           overlapping sliding windows must not straddle
                           evaluation boundaries
  6. SELF-CHECK          — the detector must CATCH seeded leakage (a
                           deliberately contaminated split), otherwise the
                           detector itself is broken

Usage:
    python leakage_audit/audit_leakage.py            # self-check mode
    python leakage_audit/audit_leakage.py --json out # write machine-readable report

Programmatic:
    from leakage_audit.audit_leakage import run_audit
    report = run_audit(shards, train_idx, val_idx, test_idx, region_ids=...)
"""

import argparse
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_disjointness(shard_indices):
    """shard_indices: list of index arrays. Returns overlap statistics."""
    seen = {}
    overlaps = []
    for cid, idx in enumerate(shard_indices):
        for i in np.asarray(idx):
            i = int(i)
            if i in seen:
                overlaps.append((i, seen[i], cid))
            else:
                seen[i] = cid
    total_assigned = sum(len(np.asarray(s)) for s in shard_indices)
    unique_assigned = len(seen)
    return {
        "duplicate_count": len(overlaps),
        "duplicates": overlaps[:20],
        "total_assigned": int(total_assigned),
        "unique_assigned": int(unique_assigned),
        "pass": len(overlaps) == 0,
    }


def check_coverage(shard_indices, n_train):
    seen = set()
    for idx in shard_indices:
        seen.update(np.asarray(idx).tolist())
    missing = sorted(set(range(n_train)) - seen)
    return {
        "n_train": int(n_train),
        "assigned": len(seen),
        "missing_count": len(missing),
        "pass": len(missing) == 0 and len(seen) == n_train,
    }


def check_split_exclusivity(train_idx, val_idx, test_idx):
    t, v, te = set(map(int, np.asarray(train_idx))), \
               set(map(int, np.asarray(val_idx))), \
               set(map(int, np.asarray(test_idx)))
    return {
        "train_val_overlap": len(t & v),
        "train_test_overlap": len(t & te),
        "val_test_overlap": len(v & te),
        "pass": not (t & v) and not (t & te) and not (v & te),
    }


def check_region_leakage(region_ids, train_idx, val_idx, test_idx):
    """
    No active region may span an evaluation boundary. If region_ids is
    None, the check is skipped with status 'skipped' (documented
    limitation: cleaned SWAN-SF export has no per-sample region IDs).
    """
    if region_ids is None:
        return {"status": "skipped",
                "reason": "no region IDs available (cleaned export)",
                "pass": None}
    region_ids = np.asarray(region_ids)
    r_train = set(region_ids[np.asarray(train_idx)].tolist())
    r_val = set(region_ids[np.asarray(val_idx)].tolist())
    r_test = set(region_ids[np.asarray(test_idx)].tolist())
    tv = r_train & r_val
    tt = r_train & r_test
    vt = r_val & r_test
    return {
        "train_val_region_overlap": len(tv),
        "train_test_region_overlap": len(tt),
        "val_test_region_overlap": len(vt),
        "example_overlap": sorted(list(tt))[:10],
        "pass": not (tv or tt or vt),
    }


def check_window_overlap(sample_keys, train_idx, test_idx, stride, window):
    """
    If sample_keys are the *window origin* positions (e.g. minute indices)
    within the same time series, two windows overlap when their origins
    differ by less than `window` steps. Overlapping windows must not be
    split across evaluation boundaries.
    """
    if sample_keys is None:
        return {"status": "skipped",
                "reason": "no window origin keys provided", "pass": None}
    keys = np.asarray(sample_keys)
    k_train = set(keys[np.asarray(train_idx)].tolist())
    k_test = set(keys[np.asarray(test_idx)].tolist())
    boundary_violations = 0
    for kt in k_test:
        for d in range(1, window - stride):
            if (kt - d) in k_train or (kt + d) in k_train:
                boundary_violations += 1
                break
    return {
        "boundary_violations": boundary_violations,
        "pass": boundary_violations == 0,
    }


def _detector_self_check():
    """
    The detector must catch deliberately seeded leakage, otherwise the
    audit itself cannot be trusted.
    """
    results = {}

    # seeded shard overlap
    a = np.arange(100)
    shards = [a[:40], a[35:70], a[70:]]  # client 0 and 1 share 35..39
    r = check_disjointness(shards)
    results["detects_shard_overlap"] = (not r["pass"]) and r["duplicate_count"] == 5

    # seeded split overlap
    r = check_split_exclusivity(np.arange(50), np.arange(40, 60), np.arange(55, 80))
    results["detects_split_overlap"] = not r["pass"]

    # seeded region leakage
    regions = np.repeat(np.arange(10), 10)  # region r spans indices [10r,10r+10)
    r = check_region_leakage(regions, np.arange(0, 55), np.arange(50, 70), np.arange(60, 100))
    results["detects_region_leakage"] = not r["pass"]

    # clean case must pass
    clean = [a[:33], a[33:66], a[66:]]
    results["clean_shards_pass"] = check_disjointness(clean)["pass"]
    results["clean_split_pass"] = check_split_exclusivity(
        np.arange(0, 60), np.arange(60, 80), np.arange(80, 100))["pass"]

    return results


def run_audit(shards=None, train_idx=None, val_idx=None, test_idx=None,
              n_train=None, region_ids=None, sample_keys=None,
              stride=1, window=60):
    """
    Full audit. shards may be a list of (X, y) tuples OR index arrays.
    Returns a report dict with an overall pass flag.
    """
    report = {"detector_self_check": _detector_self_check()}

    detector_ok = all(report["detector_self_check"].values())
    report["detector_ok"] = detector_ok

    if shards is not None:
        # normalise to index arrays if (X,y) tuples passed
        if n_train is None:
            raise ValueError("n_train required when auditing shards")
        idx_arrays = []
        offset = 0
        for s in shards:
            if isinstance(s, tuple):
                idx_arrays.append(np.arange(offset, offset + len(s[1])))
                offset += len(s[1])
            else:
                idx_arrays.append(np.asarray(s))
        report["shard_disjointness"] = check_disjointness(idx_arrays)
        report["shard_coverage"] = check_coverage(idx_arrays, n_train)

    if train_idx is not None and test_idx is not None:
        report["split_exclusivity"] = check_split_exclusivity(
            train_idx, val_idx if val_idx is not None else np.array([], int),
            test_idx)
    if region_ids is not None and train_idx is not None:
        report["region_leakage"] = check_region_leakage(
            region_ids, train_idx,
            val_idx if val_idx is not None else np.array([], int), test_idx)
    if sample_keys is not None and train_idx is not None:
        report["window_overlap"] = check_window_overlap(
            sample_keys, train_idx, test_idx, stride, window)

    checks = [v.get("pass") for k, v in report.items()
              if isinstance(v, dict) and "pass" in v and v.get("pass") is not None]
    report["overall_pass"] = all(checks) if checks else None
    return report


def _print_report(report):
    print("\n" + "=" * 64)
    print("  SF-9 LEAKAGE AUDIT REPORT")
    print("=" * 64)
    sc = report.get("detector_self_check", {})
    print(f"\n  Detector self-check:")
    for name, ok in sc.items():
        print(f"    {'PASS' if ok else 'FAIL':>6}  {name}")
    for section in ("shard_disjointness", "shard_coverage",
                    "split_exclusivity", "region_leakage", "window_overlap"):
        if section in report:
            r = report[section]
            status = "SKIP" if r.get("pass") is None else \
                     ("PASS" if r["pass"] else "FAIL")
            print(f"\n  [{status}] {section}")
            for k, v in r.items():
                if k in ("duplicates", "example_overlap"):
                    continue
                print(f"    {k}: {v}")
    print(f"\n  OVERALL: {'PASS' if report.get('overall_pass') else 'FAIL/INCOMPLETE'}")
    print("=" * 64 + "\n")
    return report.get("overall_pass", False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="write report to this path")
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()

    if args.self_check_only:
        report = {"detector_self_check": _detector_self_check(),
                  "detector_ok": all(_detector_self_check().values())}
        ok = report["detector_ok"]
    else:
        # demo audit on a synthetic disjoint partition
        from partition_clients import allocate_indices_dirichlet
        rng = np.random.RandomState(0)
        y = rng.binomial(1, 0.05, 5000)
        assignment = allocate_indices_dirichlet(y, alpha=1.0, seed=42)
        shards = [np.where(assignment == k)[0] for k in range(6)]
        # proper three-way exclusive split of ALL indices
        all_idx = np.arange(len(y))
        perm = rng.permutation(len(y))
        n_tr, n_v = int(0.64 * len(y)), int(0.16 * len(y))
        train_idx, val_idx, test_idx = all_idx[perm[:n_tr]], \
            all_idx[perm[n_tr:n_tr + n_v]], all_idx[perm[n_tr + n_v:]]
        report = run_audit(shards=shards, train_idx=train_idx,
                           val_idx=val_idx, test_idx=test_idx, n_train=len(y))
        ok = _print_report(report)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[Audit] Report written → {args.json}")

    sys.exit(0 if ok else 1)
