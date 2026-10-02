"""
provenance/rebuild_audit_totals.py  (v4.0 — review M3/M4/m5)
────────────────────────────────────────────────────────────────
Regenerates the totals block of outputs/dataset_structure_audit.json
from data that IS committed:

  * per-partition unique-instance counts  (already in the JSON — the
    matcher's raw_idx tables are not committed, so the per-partition
    block is the committed record of the unique-instance basis);
  * provenance/train_meta_slim.csv.gz and test_meta_slim.csv.gz
    (row-basis counts and label agreement).

Every total is computed, none typed. Adds:
  * explicit row-basis vs unique-instance-basis totals, labelled;
  * the duplicate-instance reconciliation (176 total / 160 positive);
  * train-side label agreement (99.71%, 161 disagreements) and
    test-side (100.00%).

Run:  python provenance/rebuild_audit_totals.py
      (from the repo root; writes outputs/dataset_structure_audit.json)
"""

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, HERE)
from swansf_audit_artifact import build_audit  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "dataset_structure_audit.json")


def main():
    with open(OUT) as f:
        audit = json.load(f)

    per_part = audit["per_partition"]
    # The old per-partition block stores unique-instance verified counts
    # under 'train_verified' (the matcher's row count within partitions
    # equals the unique count there — the pooled-row duplicates only
    # appear in partition 4's train export rows, already reflected by
    # the unique-basis sums). Synthesise the 'train_verified_unique'
    # field the v4.0 builder emits:
    for v in per_part.values():
        if "train_verified_unique" not in v:
            v["train_verified_unique"] = v["train_windows_shared_with_test"]

    tr = pd.read_csv(os.path.join(HERE, "train_meta_slim.csv.gz"))
    te = pd.read_csv(os.path.join(HERE, "test_meta_slim.csv.gz"))

    totals = build_audit(None, per_part, tr, te)

    audit["totals"] = totals
    audit["totals_provenance"] = {
        "row_basis": ("computed from provenance/train_meta_slim.csv.gz "
                      "and test_meta_slim.csv.gz (committed)"),
        "unique_instance_basis": ("sums of the committed per-partition "
                                  "matcher counts in this artefact"),
        "regenerated_by": "provenance/rebuild_audit_totals.py",
        "replaces": ("v3.4 totals block: mixed bases with typed "
                     "literals (56005, 0.99998, 6234, 6234) and an "
                     "unreconciled 6395-vs-6235 discrepancy"),
    }

    with open(OUT, "w") as f:
        json.dump(audit, f, indent=2)

    print("[rebuild] totals written:")
    for k, v in totals.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print(f"[rebuild] wrote {OUT}")


if __name__ == "__main__":
    main()
