"""
experiments/run_lag_definition_sweep.py  (v4.2)
────────────────────────────────
Sensitivity sweep over candidate definitions of the 24-hour-lagged
persistence baseline (Dossier R-FS9-R2 item F7).

The v4.1 response letter stated that eight candidate definitions of
the lagged floor were tested before the shipped rule (the same-region
window whose start is nearest t - 1440 min, matched within +/-35 min)
was chosen, and that all point-lag variants land in 0.37-0.43 with
unchanged conclusions. That claim had no committed artefact. This
script is the artefact: it computes the shipped rule plus eight
candidate definitions on the committed provenance table, on both the
partition-5 test and the pooled test, stores the full confusion
counts / TSS / HSS of each, and cross-references the leakage-free
arms' frozen F-beta TSS/HSS (read at runtime from
outputs/standard_metrics.json, never hand-copied) so that "which arms
clear this floor" is answered by the artefact itself.

Candidate definitions (each predicts the label of a window at t from
same-region windows near t - 1440 min):
  1. nearest_30            nearest start to t-1440, tolerance 30 min
  2. nearest_60            nearest start to t-1440, tolerance 60 min
  3. nearest_120           nearest start to t-1440, tolerance 120 min
  4. most_recent_ge_24h    latest window starting at or before t-1440
                           (the R-FS9-R1 panel rule; no tolerance cap)
  5. exact_1440            start exactly t-1440 (tolerance 0)
  6. index_offset_23       window 23 positions earlier in time order
  7. index_offset_24       window 24 positions earlier in time order
  8. any_flare_in_lookback any positive label among same-AR windows
                           starting in [t-1440, t)
  + shipped (reference)    nearest start to t-1440, tolerance 35 min

Usage:   python experiments/run_lag_definition_sweep.py
Output:  outputs/lag_definition_sweep.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAG_MIN = 1440


# ── helpers (kept self-contained so a torch-less fresh clone can run this) ──

def tss_hss_from_counts(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tss = rec - fpr
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    hss = num / den if den else 0.0
    return float(tss), float(hss)


def counts(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    return {"tp": int(((pred == 1) & (y == 1)).sum()),
            "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()),
            "tn": int(((pred == 0) & (y == 0)).sum())}


def evaluate(y, pred):
    c = counts(y, pred)
    tss, hss = tss_hss_from_counts(c)
    return {"n_evaluated": int(len(y)), "confusion": c,
            "tss": tss, "hss": hss}


# ── per-group evaluation engines ────────────────────────────────────────────

def _groups(sub):
    df = sub.sort_values(["ar", "ts_start_min", "pool_row"])
    out = []
    for _, g in df.groupby("ar", sort=False):
        out.append((g["ts_start_min"].values.astype(np.int64),
                    g["label"].values.astype(int)))
    return out


def run_nearest(grps, tol_min):
    """Label of the window whose start is nearest t-1440, kept only
    when |start - (t-1440)| <= tol_min. tol_min=0 -> exact matches."""
    y_out, p_out = [], []
    for ts, lbl in grps:
        target = ts - LAG_MIN
        pos = np.searchsorted(ts, target)
        for i in range(len(ts)):
            best, best_d = None, np.inf
            for j in (pos[i] - 1, pos[i]):
                if 0 <= j < len(ts):
                    d = abs(int(ts[j]) - int(target[i]))
                    if d < best_d:
                        best, best_d = j, d
            if best is not None and best_d <= tol_min:
                y_out.append(int(lbl[i]))
                p_out.append(int(lbl[best]))
    return np.array(y_out), np.array(p_out)


def run_most_recent_ge_24h(grps):
    """Label of the latest window starting at or before t-1440
    (the R-FS9-R1 dossier's rule; no tolerance cap)."""
    y_out, p_out = [], []
    for ts, lbl in grps:
        target = ts - LAG_MIN
        pos = np.searchsorted(ts, target, side="right")
        for i in range(len(ts)):
            j = pos[i] - 1
            if j >= 0:
                y_out.append(int(lbl[i]))
                p_out.append(int(lbl[j]))
    return np.array(y_out), np.array(p_out)


def run_index_offset(grps, k):
    """Label of the window k positions earlier in ts_start_min order."""
    y_out, p_out = [], []
    for ts, lbl in grps:
        for i in range(k, len(ts)):
            y_out.append(int(lbl[i]))
            p_out.append(int(lbl[i - k]))
    return np.array(y_out), np.array(p_out)


def run_any_flare_in_lookback(grps):
    """1 if any same-AR window starting in [t-1440, t) has a positive
    label, else 0 (lookback-flare rule; evaluates every window)."""
    y_out, p_out = [], []
    for ts, lbl in grps:
        csum = np.concatenate([[0], np.cumsum(lbl)])
        for i in range(len(ts)):
            lo = np.searchsorted(ts, ts[i] - LAG_MIN, side="left")
            p = 1 if csum[i] - csum[lo] > 0 else 0
            y_out.append(int(lbl[i]))
            p_out.append(p)
    return np.array(y_out), np.array(p_out)


# ── main ────────────────────────────────────────────────────────────────────

DEFS = [
    ("nearest_30", "candidate",
     "nearest same-AR start to t-1440 min, tolerance 30 min",
     lambda g: run_nearest(g, 30)),
    ("nearest_60", "candidate",
     "nearest same-AR start to t-1440 min, tolerance 60 min",
     lambda g: run_nearest(g, 60)),
    ("nearest_120", "candidate",
     "nearest same-AR start to t-1440 min, tolerance 120 min",
     lambda g: run_nearest(g, 120)),
    ("most_recent_ge_24h", "candidate",
     "latest same-AR window starting at or before t-1440 min "
     "(a most-recent-observation reading of the R-FS9-R1 rule; the "
     "panel's own variant scored 0.374)",
     run_most_recent_ge_24h),
    ("exact_1440", "candidate",
     "same-AR start exactly t-1440 min (equivalent to any tolerance "
     "below the 60-min grid: within-AR start gaps are multiples of "
     "the cadence)",
     lambda g: run_nearest(g, 0)),
    ("index_offset_23", "candidate",
     "same-AR window 23 positions earlier in ts_start_min order",
     lambda g: run_index_offset(g, 23)),
    ("index_offset_24", "candidate",
     "same-AR window 24 positions earlier in ts_start_min order",
     lambda g: run_index_offset(g, 24)),
    ("any_flare_in_lookback", "candidate",
     "any positive label among same-AR windows starting in "
     "[t-1440, t)",
     run_any_flare_in_lookback),
    ("shipped_nearest_35", "shipped",
     "nearest same-AR start to t-1440 min, tolerance 35 min "
     "(the definition stored in standard_metrics.json block C)",
     lambda g: run_nearest(g, 35)),
]


def main():
    te = pd.read_csv(os.path.join(ROOT, "provenance",
                                  "test_meta_slim.csv.gz"))
    scopes = {"pooled_test": te,
              "p5_test": te[te["partition"] == 5]}
    grps = {k: _groups(v) for k, v in scopes.items()}

    # leakage-free arm reference: read at runtime, never hand-copied
    sm = json.load(open(os.path.join(ROOT, "outputs",
                                     "standard_metrics.json")))
    lf = sm["block_a_derived"]["leakage_free_fold"]
    arms = {name: {"tss": float(v["fbeta_threshold"]["tss"]),
                   "hss": float(v["fbeta_threshold"]["hss"])}
            for name, v in lf.items()}

    rows = []
    for did, role, desc, fn in DEFS:
        row = {"id": did, "role": role, "definition": desc}
        for scope in ("pooled_test", "p5_test"):
            y, p = fn(grps[scope])
            row[scope] = evaluate(y, p)
        f = row["p5_test"]
        row["p5_clears_floor"] = {
            "tss": sorted(a for a, v in arms.items() if v["tss"] > f["tss"]),
            "hss": sorted(a for a, v in arms.items() if v["hss"] > f["hss"]),
        }
        rows.append(row)
        print(f"{did:24s} P5 TSS {f['tss']:.4f}  HSS {f['hss']:.4f}  "
              f"n={f['n_evaluated']:6d}  clears(TSS): "
              f"{row['p5_clears_floor']['tss']}")

    # consistency check: the shipped reference row must match block C
    stored = sm["block_c_baselines"]["p5_test"]["persistence_24h_lag"]
    shipped = next(r for r in rows if r["role"] == "shipped")
    ok = (abs(shipped["p5_test"]["tss"] - stored["tss"]) < 1e-9
          and abs(shipped["p5_test"]["hss"] - stored["hss"]) < 1e-9)
    print(f"\nshipped-rule consistency vs block C: "
          f"{'PASS' if ok else 'FAIL'}")

    report = {
        "purpose": ("sensitivity sweep over candidate definitions of the "
                    "24-hour-lagged persistence floor — commits the "
                    "R-FS9-R1 response letter's 'eight candidate "
                    "definitions' claim as a regenerable artefact "
                    "(Dossier R-FS9-R2 item F7)"),
        "input": "provenance/test_meta_slim.csv.gz",
        "arm_reference": ("outputs/standard_metrics.json "
                          "block_a_derived.leakage_free_fold "
                          "(fbeta_threshold), read at runtime"),
        "leakfree_arms_fbeta": arms,
        "lag_min": LAG_MIN,
        "definitions": rows,
        "shipped_rule_consistency_vs_block_c": "PASS" if ok else "FAIL",
    }
    out = os.path.join(ROOT, "outputs", "lag_definition_sweep.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, allow_nan=False, sort_keys=False)
        f.write("\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
