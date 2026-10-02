"""
tests/test_audit_artifact.py  (v4.0 — review M3/M4/m5)
─────────────────────────────────────────────────────
Recomputes every headline leakage statistic of
outputs/dataset_structure_audit.json from data that is committed:

  * per-partition sums        -> unique-instance-basis totals
  * train/test_meta_slim      -> row-basis totals + label agreement

If any typed literal or inconsistent total re-enters the artefact, this
test fails. Run standalone or via tests/run_battery.py.
"""

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def main():
    audit_path = os.path.join(ROOT, "outputs", "dataset_structure_audit.json")
    with open(audit_path) as f:
        audit = json.load(f)
    t = audit["totals"]
    pp = audit["per_partition"]

    tr = pd.read_csv(os.path.join(ROOT, "provenance", "train_meta_slim.csv.gz"))
    te = pd.read_csv(os.path.join(ROOT, "provenance", "test_meta_slim.csv.gz"))

    print("\n[A1] unique-instance totals == sums of per-partition entries")
    for tot_key, per_key in [
            ("train_verified_unique_instances", "train_verified_unique"),
            ("train_unique_instances_shared_with_test",
             "train_windows_shared_with_test"),
            ("train_real_positives_unique", "train_real_positives"),
            ("test_positive_instances_total", "test_positive_instances"),
            ("test_positives_in_train", "test_positives_in_train"),
            ("train_synthetic_positives_unique",
             "train_synthetic_positives")]:
        s = sum(v[per_key] for v in pp.values())
        check(f"totals.{tot_key} == sum(per-partition {per_key}) "
              f"({s})", t[tot_key] == s, f"totals say {t[tot_key]}")

    print("\n[A2] row-basis totals == recomputed from slim tables")
    v = tr[tr["verified"]]
    check(f"train_verified_windows_rows == {int(len(v))}",
          t["train_verified_windows_rows"] == len(v))
    check(f"train_windows == {len(tr)}", t["train_windows"] == len(tr))
    check(f"test_windows == {len(te)}", t["test_windows"] == len(te))
    vp = ((tr["pkl_label"] == 1) & tr["verified"]).sum()
    check(f"train_verified_positive_windows_rows == {int(vp)}",
          t["train_verified_positive_windows_rows"] == vp)
    syn = ((tr["pkl_label"] == 1) & ~tr["verified"]).sum()
    check(f"train_synthetic_positives_rows == {int(syn)}",
          t["train_synthetic_positives_rows"] == syn)

    print("\n[A3] headline leakage statistics (paper-cited)")
    check("100% of unique verified train instances shared with test",
          t["train_shared_frac_of_verified_unique"] == 1.0)
    check("every flaring test instance in train: 6234/6234 (100%)",
          t["test_positives_in_train"] == 6234 and
          t["test_positives_in_train_frac"] == 1.0)
    lo, hi = t["train_synthetic_positive_share_range"]
    check("synthetic-positive share range is 85.7-90.0%",
          abs(lo - 0.85714) < 1e-4 and abs(hi - 0.90) < 1e-4,
          f"got {lo:.4f}-{hi:.4f}")

    print("\n[A4] duplicate-instance reconciliation")
    d = t["duplicate_instance_windows"]
    check("176 duplicate rows = 56,181 rows - 56,005 unique",
          d["total"] == t["train_verified_windows_rows"] -
          t["train_verified_unique_instances"] == 176)
    check("160 duplicate positives = 6,395 rows - 6,235 unique",
          d["positive"] == t["train_verified_positive_windows_rows"] -
          t["train_real_positives_unique"] == 160)

    print("\n[A5] label agreement (m5)")
    agree_tr = (v["label"] == v["pkl_label"]).mean()
    vt = te[te["verified"]]
    agree_te = (vt["label"] == vt["pkl_label"]).mean()
    check(f"train label agreement {agree_tr:.4%} (161 disagreements)",
          abs(t["train_label_agreement"] - agree_tr) < 1e-9 and
          int((v["label"] != v["pkl_label"]).sum()) == 161)
    check(f"test label agreement {agree_te:.4%}",
          abs(t["test_label_agreement"] - agree_te) < 1e-9)

    print("\n[A6] no stale v3.4 keys resurface")
    for stale in ["train_windows_shared_with_test", "0.99998"]:
        check(f"totals block has no stale '{stale}'",
              stale not in t or stale not in
              [k for k in t if isinstance(t.get(k), float)])

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
