"""
scripts_clone/patch_reference_fields.py  (v4.0 — review M1)
────────────────────────────────────────────────────────────────
Surgically regenerates the 'shipped_in_partition' (and
'fold_leakage_free_cleaned') reference fields inside
outputs/partition_disjoint_eval.json and outputs/raw_substrate_eval.json
from the machine-readable sources (outputs/results.json and
outputs/partition_disjoint_eval.json respectively), replacing the
stale hand-typed logistic-regression entry (0.818/0.114) with the true
artefact value (0.8236/0.2348). Nothing else in the JSONs is touched.

Run:  python3 patch_reference_fields.py   (from the repo root)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_shipped(results_path):
    with open(results_path) as f:
        res = json.load(f)
    out = {}
    for name, m in res.get("test_metrics", {}).items():
        if isinstance(m, dict) and "roc_auc" in m and "pr_auc" in m:
            out[name] = {"roc_auc": float(m["roc_auc"]),
                         "pr_auc": float(m["pr_auc"]),
                         "source": "outputs/results.json"}
    return out


def main():
    results_path = os.path.join(HERE, "outputs", "results.json")

    shipped = load_shipped(results_path)
    print("[patch] shipped reference (from results.json):")
    for k, v in shipped.items():
        print(f"  {k:<22s} {v['roc_auc']:.4f} / {v['pr_auc']:.4f}")

    # ── 1. partition_disjoint_eval.json ─────────────────────────────────
    p1 = os.path.join(HERE, "outputs", "partition_disjoint_eval.json")
    with open(p1) as f:
        d = json.load(f)
    changed = 0
    for name, entry in d.get("results", {}).items():
        old = entry.get("shipped_in_partition")
        new = shipped.get(name)
        if old != new:
            changed += 1
            print(f"[patch] {p1}: {name} shipped_in_partition "
                  f"{old} -> {new}")
        entry["shipped_in_partition"] = new
    with open(p1, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[patch] wrote {p1} ({changed} fields corrected)")

    # ── 2. raw_substrate_eval.json ──────────────────────────────────────
    p2 = os.path.join(HERE, "outputs", "raw_substrate_eval.json")
    with open(p2) as f:
        d = json.load(f)
    # fold reference: read fresh from the (now corrected) partition JSON
    fold = {}
    with open(p1) as f:
        pdj = json.load(f)
    for name, entry in pdj.get("results", {}).items():
        m = entry.get("test") or {}
        if isinstance(m, dict) and "roc_auc" in m:
            fold[name] = {"roc_auc": float(m["roc_auc"]),
                          "pr_auc": float(m["pr_auc"]),
                          "source": "outputs/partition_disjoint_eval.json"}
    changed = 0
    for name, entry in d.get("results", {}).items():
        old = entry.get("shipped_in_partition")
        new = shipped.get(name)
        if old != new:
            changed += 1
            print(f"[patch] {p2}: {name} shipped_in_partition "
                  f"{old} -> {new}")
        entry["shipped_in_partition"] = new
        oldf = entry.get("fold_leakage_free_cleaned")
        newf = fold.get(name)
        if oldf != newf:
            print(f"[patch] {p2}: {name} fold_leakage_free_cleaned "
                  f"{oldf} -> {newf}")
        entry["fold_leakage_free_cleaned"] = newf
    with open(p2, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[patch] wrote {p2} ({changed} fields corrected)")

    # consistency assertions — the whole point of the exercise
    d1 = json.load(open(p1))["results"]
    d2 = json.load(open(p2))["results"]
    for name in shipped:
        if name in d1 and name in d2:
            a = d1[name]["shipped_in_partition"]
            b = d2[name]["shipped_in_partition"]
            assert a == b == shipped[name], name
    lr = d1["logistic_regression"]
    print("[patch] tab:leakfree LR in-partition cell now "
          f"{lr['shipped_in_partition']['roc_auc']:.4f}/"
          f"{lr['shipped_in_partition']['pr_auc']:.4f} "
          "(matches tab:main)")
    print("[patch] DONE — both JSONs consistent with results.json")


if __name__ == "__main__":
    main()
