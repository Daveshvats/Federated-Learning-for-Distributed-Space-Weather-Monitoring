"""
experiments/run_standard_metrics.py  (v4.0 — review Tier-1 item 8)
────────────────────────────────────────────────────────────────
Standard flare-forecast verification apparatus (the toolkit of
Leka et al. 2019, the comparison paper this manuscript cites):

  Block A (derived, no re-computation of probabilities):
    TSS = recall - FPR and HSS at EVERY stored operating point of
    every committed evaluation artefact (fbeta thresholds and frozen
    FPR points; window-level and event-level), derived from the
    full-precision confusion components the artefacts already store.
    Brier skill score vs the test-set climatology.

  Block B (recomputed from the phase-cache models):
    10-bin reliability diagrams, ECE, raw Brier and BSS for the
    in-partition arms on the frozen pooled test split.

  Block C (baselines):
    Climatology (base-rate) and same-AR previous-window persistence
    baselines, pooled test and the P5 (leakage-free-fold) test.

Output: outputs/standard_metrics.json (+ console summary).

Run:  python experiments/run_standard_metrics.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "standard_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def confusion_from_pr(precision, recall, n_pos, n_neg):
    """Full-precision confusion counts from stored precision/recall."""
    tp = recall * n_pos
    fp = (tp / precision - tp) if precision > 0 else float("nan")
    fn = n_pos - tp
    tn = n_neg - fp
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def tss_from_counts(c):
    pod = c["tp"] / max(c["tp"] + c["fn"], 1e-9)
    pofd = c["fp"] / max(c["fp"] + c["tn"], 1e-9)
    return pod - pofd


def hss_from_counts(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    denom = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return (2 * tp * tn - 2 * fp * fn) / denom if denom else float("nan")


def derived_operating_points(m, n_windows, n_pos):
    """Standard metrics at one model's stored operating points."""
    n_neg = n_windows - n_pos
    out = {}
    # main (fbeta) threshold operating point
    if "threshold" in m and "recall" in m:
        c = confusion_from_pr(m.get("precision", 0.0), m["recall"],
                              n_pos, n_neg)
        out["fbeta_threshold"] = {
            "threshold": m["threshold"], "confusion": c,
            "tss": tss_from_counts(c), "hss": hss_from_counts(c),
        }
    # frozen FPR operating points: TSS = recall - realised_fpr exactly
    fops = m.get("frozen_operating_points") or {}
    items = fops.items() if isinstance(fops, dict) else \
        ([(k, v) for k, v in fops] if isinstance(fops, list) else [])
    for name, op in items:
        rec, fpr = op.get("recall"), op.get("realised_fpr")
        if rec is None or fpr is None:
            continue
        tp = rec * n_pos
        fp = fpr * n_neg
        c = dict(tp=tp, fp=fp, fn=n_pos - tp, tn=n_neg - fp)
        out[str(name)] = {
            "frozen_threshold": op.get("frozen_threshold"),
            "realised_fpr": fpr, "recall": rec,
            "tss": rec - fpr, "hss": hss_from_counts(c),
        }
    return out


def reliability_bins(y, p, n_bins=10):
    """10-bin reliability table (equal-width bins on [0,1])."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1],
                         "n": 0, "mean_pred": None, "emp_freq": None})
            continue
        rows.append({"bin": b, "lo": float(edges[b]),
                     "hi": float(edges[b + 1]), "n": n,
                     "mean_pred": float(p[sel].mean()),
                     "emp_freq": float(y[sel].mean())})
    ece = sum(r["n"] / max(len(y), 1) * abs(r["emp_freq"] - r["mean_pred"])
              for r in rows if r["n"] > 0)
    return rows, float(ece)


def persistence_predictions(slim_df):
    """Same-AR previous-window persistence: for each AR (time-ordered by
    pool_row), predict the previous window's label."""
    y = slim_df["label"].values.astype(int)
    pred = np.zeros_like(y)
    order = np.arange(len(slim_df))
    for _, g in slim_df.assign(row=order).groupby("ar"):
        idx = g["row"].values
        lbl = g["label"].values.astype(int)
        pred[idx[1:]] = lbl[:-1]
    return y, pred


def counts_tss_hss(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    c = dict(tp=tp, fp=fp, fn=fn, tn=tn)
    return c, tss_from_counts(c), hss_from_counts(c)


# ─────────────────────────────────────────────────────────────────────────────
# Block A — derived from stored artefacts
# ─────────────────────────────────────────────────────────────────────────────

def block_a():
    derived = {}

    # 1. in-partition protocol (results.json)
    r = json.load(open(os.path.join(ROOT, "outputs", "results.json")))
    n_win = 331185
    prev = r["test_metrics"]["fedprox_mlp"]["prevalence"]
    n_pos = round(prev * n_win)
    derived["in_partition"] = {}
    for name, m in r["test_metrics"].items():
        entry = {"n_windows": n_win, "n_pos": n_pos,
                 "prevalence": prev, "roc_auc": m.get("roc_auc"),
                 "brier": m.get("brier"),
                 "bss_vs_climatology": (
                     1 - m["brier"] / (prev * (1 - prev)))
                 if m.get("brier") is not None else None}
        entry.update(derived_operating_points(m, n_win, n_pos))
        derived["in_partition"][name] = entry

    # 2. leakage-free fold (partition_disjoint_eval.json)
    d = json.load(open(os.path.join(ROOT, "outputs",
                                    "partition_disjoint_eval.json")))
    n_win_p5, prev_p5 = 75365, 0.013136071152985096
    n_pos_p5 = round(prev_p5 * n_win_p5)
    derived["leakage_free_fold"] = {}
    for name, entry in d["results"].items():
        m = entry.get("test", {})
        e = {"n_windows": n_win_p5, "n_pos": n_pos_p5,
             "prevalence": prev_p5,
             "roc_auc": m.get("roc_auc"), "brier": m.get("brier"),
             "bss_vs_climatology": (
                 1 - m["brier"] / (prev_p5 * (1 - prev_p5)))
             if m.get("brier") is not None else None}
        e.update(derived_operating_points(m, n_win_p5, n_pos_p5))
        derived["leakage_free_fold"][name] = e

    # 3. raw substrate 2D arms (raw_substrate_eval.json)
    d = json.load(open(os.path.join(ROOT, "outputs",
                                    "raw_substrate_eval.json")))
    sub = d.get("substrate", {})
    n_win_raw = sub.get("test", 75365)
    prev_raw = sub.get("test_pos", prev_p5)
    n_pos_raw = round(prev_raw * n_win_raw)
    derived["raw_substrate_2d"] = {}
    for name, entry in d["results"].items():
        m = entry.get("test", {}) or {}
        e = {"n_windows": n_win_raw, "n_pos": n_pos_raw,
             "prevalence": prev_raw,
             "roc_auc": m.get("roc_auc"), "brier": m.get("brier"),
             "bss_vs_climatology": (
                 1 - m["brier"] / (prev_raw * (1 - prev_raw)))
             if m.get("brier") is not None else None}
        e.update(derived_operating_points(m, n_win_raw, n_pos_raw))
        derived["raw_substrate_2d"][name] = e

    # 4. LSTM arms on the raw substrate
    derived["raw_substrate_lstm"] = {}
    for fname in ["raw_lstm_eval.json", "raw_lstm_scaffold.json",
                  "raw_lstm_seed43.json", "raw_lstm_smote.json"]:
        path = os.path.join(ROOT, "outputs", fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        sub = d.get("substrate", {})
        nw = sub.get("test", 75365)
        pv = sub.get("test_pos", prev_p5)
        npos = round(pv * nw)
        for name, m in d.get("results", {}).items():
            e = {"artefact": fname, "n_windows": nw, "n_pos": npos,
                 "prevalence": pv, "roc_auc": m.get("roc_auc"),
                 "brier": m.get("brier"),
                 "bss_vs_climatology": (
                     1 - m["brier"] / (pv * (1 - pv)))
                 if m.get("brier") is not None else None}
            e.update(derived_operating_points(m, nw, npos))
            derived["raw_substrate_lstm"][f"{fname[:-5]}::{name}"] = e

    # 5. event-level (window-level contingency at the event thresholds)
    derived["event_level"] = {}
    for fname in ["event_level_p5.json", "event_level_raw_lstm_p5.json",
                  "event_level_raw_p5.json", "event_level_raw_scaffold.json",
                  "event_level_raw_seed43.json", "event_level_raw_smote.json"]:
        path = os.path.join(ROOT, "outputs", fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        n_bg = d.get("n_event_groups", 0) - d.get("n_true_events", 0)
        for name, m in d.get("models", {}).items():
            det = m.get("n_detected_events", 0)
            miss = m.get("n_missed_events", 0)
            fa_groups = m.get("background_groups_with_alerts", 0)
            tn_groups = n_bg - fa_groups
            c = dict(tp=det, fn=miss, fp=fa_groups, tn=tn_groups)
            derived["event_level"][f"{fname[:-5]}::{name}"] = {
                "n_true_events": d.get("n_true_events"),
                "n_background_groups": n_bg,
                "confusion_events": c,
                "tss": tss_from_counts(c),
                "hss": hss_from_counts(c),
                "window_fpr": (
                    m["false_alarm_windows"] / m["n_windows"]
                    if m.get("false_alarm_windows") is not None else None),
            }
    return derived


# ─────────────────────────────────────────────────────────────────────────────
# Block B — recomputed from phase-cache models (in-partition protocol)
# ─────────────────────────────────────────────────────────────────────────────

def block_b():
    import torch
    z = np.load(os.path.join(ROOT, "data", "cache", "phase_data.npz"))
    X_test = z["X_test"].astype(np.float32)
    y_test = z["y_test"].astype(int)

    baselines = torch.load(os.path.join(ROOT, "data", "cache",
                                        "phase_baselines.pt"),
                           map_location="cpu", weights_only=False)
    probs = {}

    # climatology
    pbar = y_test.mean()
    probs["climatology"] = np.full(len(y_test), pbar, dtype=np.float64)

    # logistic regression
    lr = baselines["logistic_regression"]
    probs["logistic_regression"] = lr.predict_proba(X_test)[:, 1]

    # xgboost ( Booster or sklearn wrapper )
    xgb = baselines["xgboost"]
    try:
        probs["xgboost"] = xgb.predict_proba(X_test)[:, 1]
    except AttributeError:
        import xgboost as xgblib
        D = xgblib.DMatrix(X_test)
        probs["xgboost"] = xgb.predict(D)

    # centralized MLP + federated arms (pickled SolarMLP modules)
    for key, label in [("centralized_mlp", "centralized_mlp")]:
        model = baselines[key]
        with torch.no_grad():
            logits = model(torch.from_numpy(X_test))
        probs[label] = torch.sigmoid(logits).numpy().astype(np.float64)
    for fname, label in [("phase_fedavg.pt", "fedavg_mlp"),
                         ("phase_fedprox.pt", "fedprox_mlp")]:
        d = torch.load(os.path.join(ROOT, "data", "cache", fname),
                       map_location="cpu", weights_only=False)
        model = d["model"]
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_test))
        probs[label] = torch.sigmoid(logits).numpy().astype(np.float64)

    out = {"protocol": "in-partition (shipped), pooled 5-partition test",
           "n_test_windows": int(len(y_test)),
           "prevalence": float(pbar),
           "climatology_brier": float(pbar * (1 - pbar)),
           "models": {}}
    from sklearn.metrics import roc_auc_score
    for name, p in probs.items():
        rows, ece = reliability_bins(y_test, p)
        brier = float(np.mean((p - y_test) ** 2))
        out["models"][name] = {
            "brier_raw": brier,
            "bss_vs_climatology_raw": 1 - brier / (pbar * (1 - pbar)),
            "ece_raw": ece,
            "auc_raw": float(roc_auc_score(y_test, p))
            if np.std(p) > 0 else 0.5,
            "reliability_10bin": rows,
            "note": ("raw (pre-calibration) probabilities from the "
                     "phase-cache models on the frozen test split"),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Block D — event-level uncertainty (review Tier-2 item 1)
# ─────────────────────────────────────────────────────────────────────────────

def block_d():
    """Analytic CIs on the event-level rates (Wilson for detection,
    Poisson for alert rates) plus the seed-42 vs seed-43 swing as the
    empirical seed band."""
    def wilson(k, n, z=1.96):
        if n == 0:
            return (None, None)
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))

    def poisson(k, z=1.96):
        lo = 0.5 * (2 * k - 1 - z * np.sqrt(max(k - 0.5, 0))) if k > 0 else 0.0
        hi = 0.5 * (2 * k - 1 + z * np.sqrt(k + 0.5)) if k > 0 else z * z / 2
        return (max(0.0, lo), hi)

    out = {}
    for fname in ["event_level_p5.json", "event_level_raw_lstm_p5.json",
                  "event_level_raw_p5.json", "event_level_raw_scaffold.json",
                  "event_level_raw_seed43.json", "event_level_raw_smote.json"]:
        path = os.path.join(ROOT, "outputs", fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        n_ev = d.get("n_true_events", 0)
        days = None
        for name, m in d.get("models", {}).items():
            det = m.get("n_detected_events", 0)
            lo, hi = wilson(det, n_ev)
            apd = m.get("alerts_per_day")
            fad = m.get("false_alarm_windows_per_day")
            faw = m.get("false_alarm_windows")
            days = m.get("observation_days")
            plo, phi = poisson(faw) if faw is not None else (None, None)
            out[f"{fname[:-5]}::{name}"] = {
                "detection_rate": det / n_ev if n_ev else None,
                "detection_ci95_wilson": [lo, hi],
                "alerts_per_day": apd,
                "false_alarm_windows_per_day": fad,
                "false_alarm_windows_per_day_ci95_poisson": (
                    [plo / days, phi / days]
                    if (plo is not None and days) else None),
            }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Block C — baselines
# ─────────────────────────────────────────────────────────────────────────────

def block_c():
    te = pd.read_csv(os.path.join(ROOT, "provenance",
                                  "test_meta_slim.csv.gz")).sort_values("pool_row")
    out = {}

    # pooled test (in-partition protocol)
    y, pred = persistence_predictions(te)
    c, tss, hss = counts_tss_hss(y, pred)
    pbar = y.mean()
    out["pooled_test"] = {
        "n_windows": int(len(y)),
        "climatology": {"brier": float(pbar * (1 - pbar)), "tss": 0.0,
                        "auc": 0.5,
                        "note": "base-rate forecast, no discrimination"},
        "persistence_same_ar_prev_window": {
            "confusion": c, "tss": tss, "hss": hss,
            "auc": float(np.mean(pred[y == 1]) -
                         np.mean(pred[y == 0])) if pred.sum() else 0.5,
        },
    }

    # P5 only (the leakage-free fold's test partition)
    p5 = te[te["partition"] == 5]
    y5, pred5 = persistence_predictions(p5)
    c5, tss5, hss5 = counts_tss_hss(y5, pred5)
    pbar5 = y5.mean()
    out["p5_test"] = {
        "n_windows": int(len(y5)),
        "climatology": {"brier": float(pbar5 * (1 - pbar5)), "tss": 0.0,
                        "auc": 0.5},
        "persistence_same_ar_prev_window": {
            "confusion": c5, "tss": tss5, "hss": hss5,
            "auc": float(np.mean(pred5[y5 == 1]) -
                         np.mean(pred5[y5 == 0])) if pred5.sum() else 0.5,
        },
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    report = {
        "purpose": ("standard flare-forecast verification apparatus "
                    "(Leka et al. 2019 toolkit): TSS, HSS, Brier skill "
                    "score, reliability, persistence and climatology "
                    "baselines — review Tier-1 item 8"),
        "definitions": {
            "tss": "TSS = POD - POFD = recall - FPR",
            "hss": "HSS = 2(TP*TN - FP*FN)/[(TP+FN)(FN+TN)+(TP+FP)(FP+TN)]",
            "bss": "BSS = 1 - Brier/Brier_climatology (test-set base rate)",
            "persistence": ("same active region, previous window "
                            "(~12 min cadence): the label-inertia floor"),
        },
        "block_a_derived": block_a(),
        "block_b_recomputed": block_b(),
        "block_c_baselines": block_c(),
        "block_d_event_uncertainty": block_d(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[std] wrote {OUT}")

    # console digest
    print("\n[pooled-test baselines]")
    print(f"  climatology Brier {report['block_c_baselines']['pooled_test']['climatology']['brier']:.4f}")
    p = report["block_c_baselines"]["pooled_test"]["persistence_same_ar_prev_window"]
    print(f"  persistence: TSS {p['tss']:.3f}  HSS {p['hss']:.3f}")
    print("\n[P5 leakage-free-fold baselines]")
    p5 = report["block_c_baselines"]["p5_test"]["persistence_same_ar_prev_window"]
    print(f"  persistence: TSS {p5['tss']:.3f}  HSS {p5['hss']:.3f}")
    print("\n[TSS at fbeta operating points — leakfree fold]")
    for name, e in report["block_a_derived"]["leakage_free_fold"].items():
        fb = e.get("fbeta_threshold", {})
        print(f"  {name:<22s} TSS {fb.get('tss', float('nan')):+.3f} "
              f"HSS {fb.get('hss', float('nan')):+.3f}")
    print("\n[TSS at fbeta operating points — in-partition]")
    for name, e in report["block_a_derived"]["in_partition"].items():
        fb = e.get("fbeta_threshold", {})
        print(f"  {name:<22s} TSS {fb.get('tss', float('nan')):+.3f} "
              f"HSS {fb.get('hss', float('nan')):+.3f}")


if __name__ == "__main__":
    main()
