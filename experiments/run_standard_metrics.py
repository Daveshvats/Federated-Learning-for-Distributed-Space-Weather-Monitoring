"""
experiments/run_standard_metrics.py  (v4.1 — Dossier R-FS9-R1, items A1-A6)
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
    Climatology (base-rate) and TWO same-AR persistence baselines
    (v4.1 fix, R-FS9-R1 A1): previous-window label inertia (windows
    ordered by ts_start_min, NOT pool_row — pool_row order is
    temporally scrambled within ARs) and 24-hour-lagged persistence
    (Leka-style), pooled test and the P5 (leakage-free-fold) test.

  Block D — event-level uncertainty: Wilson detection CIs and exact
    chi-square (Garwood) Poisson intervals for alert rates.

Output: outputs/standard_metrics.json (strict JSON, no NaN/Inf
literals; JavaScript/R/Go parsers accept it) + console summary.

Run:  python experiments/run_standard_metrics.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "standard_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def confusion_from_pr(precision, recall, n_pos, n_neg):
    """Full-precision confusion counts from stored precision/recall.

    Zero-alert semantics (R-FS9-R1 A4): an operating point that stores
    precision = 0 and recall = 0 encodes a forecast with no alerts;
    the correct counts are fp = 0, fn = n_pos, TSS = HSS = 0.0 (the
    v4.0 code produced NaN here, emitting invalid JSON literals).
    """
    tp = recall * n_pos
    if tp <= 0:
        # tp = 0 implies precision = 0 in the stored artefacts;
        # a zero-alert forecast fires no positive predictions.
        fp = 0.0
    else:
        fp = tp / precision - tp
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
    if not denom:
        return 0.0
    val = (2 * tp * tn - 2 * fp * fn) / denom
    return float(val) if np.isfinite(val) else None


def metrics_view(entry):
    """Merge a model entry with its nested test block (R-FS9-R1 A2/A3).

    The artefacts store the F2-selected operating point inside the
    nested 'test' sub-dictionary while the frozen FPR operating
    points live at the top level of the entry; the v4.0 code read
    only one of the two locations per family, leaving 10 LSTM arms
    and 24 raw-2D frozen cells with no metrics. The merged view
    exposes both.
    """
    view = dict(entry)
    test = entry.get("test")
    if isinstance(test, dict):
        view.update(test)
    return view


def derived_operating_points(m, n_windows, n_pos):
    """Standard metrics at one model's stored operating points."""
    n_neg = n_windows - n_pos
    out = {}
    # main (fbeta) threshold operating point (metrics_view guarantees
    # threshold/recall are visible whether stored flat or nested)
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
    """Same-AR previous-window persistence (label inertia).

    Windows are ordered by ts_start_min within each active region.
    The v4.0 implementation ordered by pool_row, whose within-AR
    order is temporally scrambled (recomputed on the committed
    provenance tables: 99% of AR groups non-monotone in time,
    adjacent pairs a coin flip, median adjacent-pair gap 2,880 min
    = 2 days), so the stored "previous window" was a random
    same-AR window typically two days away (R-FS9-R1 A1).
    """
    df = slim_df.sort_values(["ar", "ts_start_min", "pool_row"])
    prev = df.groupby("ar", sort=False)["label"].shift(1)
    mask = prev.notna().values
    y = df["label"].values.astype(int)
    return y[mask], prev.values[mask].astype(int)


def persistence_predictions_24h(slim_df, lag_min=1440, tol_min=35):
    """Leka-style 24-hour-lagged persistence.

    Forecast for a window at time t = the label of the same-AR window
    whose start time is nearest to t - 1440 min, matched only when
    within tol_min (a tolerance near half the measured 60-min
    median within-AR cadence);
    windows without a match inside the tolerance are excluded.
    """
    df = slim_df.sort_values(["ar", "ts_start_min", "pool_row"])
    y_out, p_out = [], []
    for _, g in df.groupby("ar", sort=False):
        ts = g["ts_start_min"].values.astype(np.int64)
        lbl = g["label"].values.astype(int)
        target = ts - lag_min
        j = np.searchsorted(ts, target)
        for i in range(len(ts)):
            best, best_d = None, np.inf
            for jj in (j[i] - 1, j[i]):
                if 0 <= jj < len(ts):
                    d = abs(int(ts[jj]) - int(target[i]))
                    if d < best_d:
                        best, best_d = jj, d
            if best is not None and best_d <= tol_min:
                y_out.append(int(lbl[i]))
                p_out.append(int(lbl[best]))
    return np.array(y_out), np.array(p_out)


def _cadence(slim_df):
    """Median within-AR consecutive-window spacing (minutes)."""
    gaps = []
    df = slim_df.sort_values(["ar", "ts_start_min", "pool_row"])
    for _, g in df.groupby("ar", sort=False):
        t = g["ts_start_min"].values
        gaps.extend(np.diff(t).tolist())
    return float(np.median(gaps)) if gaps else None


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
        m = metrics_view(entry)
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
        m = metrics_view(entry)   # frozen points live at top level (A3)
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
            m = metrics_view(m)   # F2 point lives in the nested 'test' (A2)
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
    """Analytic CIs on the event-level rates: Wilson intervals for
    detection rates and exact chi-square (Garwood) Poisson intervals
    for false-alarm alert rates (v4.1 fix, R-FS9-R1 A5: the v4.0
    by-hand Poisson formula was ~2x too narrow against the exact
    quantiles)."""
    def wilson(k, n, z=1.96):
        if n == 0:
            return (None, None)
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))

    def poisson(k, z=1.96):
        """Exact Poisson CI (Garwood 1932): chi-square quantiles.
        For k=20 this yields [12.2, 30.9] where the v4.0 by-hand
        formula gave [15.2, 23.9]; for k=0, [0, 1.92] -> [0, 3.69].
        """
        if k is None:
            return (None, None)
        k = int(k)
        alpha = 0.05
        lo = 0.0 if k == 0 else float(chi2.ppf(alpha / 2, 2 * k)) / 2
        hi = float(chi2.ppf(1 - alpha / 2, 2 * k + 2)) / 2
        return (lo, hi)

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
                                  "test_meta_slim.csv.gz"))
    cadence = _cadence(te)
    out = {"within_ar_window_cadence_min": cadence}

    def baselines(sub, key):
        """Climatology + both persistence variants on one split."""
        blk = {}
        y, pred = persistence_predictions(sub)
        c, tss, hss = counts_tss_hss(y, pred)
        pbar = y.mean()
        blk["climatology"] = {"brier": float(pbar * (1 - pbar)),
                              "tss": 0.0, "auc": 0.5,
                              "note": "base-rate forecast, no discrimination"}
        blk["persistence_prev_window"] = {
            "confusion": c, "tss": tss, "hss": hss,
            "n_evaluated": int(len(y)),
            "definition": ("same active region, immediately preceding "
                           "window in ts_start_min order (label inertia; "
                           f"~{cadence:.0f}-min within-AR cadence)"),
        }
        y24, p24 = persistence_predictions_24h(sub)
        c24, tss24, hss24 = counts_tss_hss(y24, p24)
        blk["persistence_24h_lag"] = {
            "confusion": c24, "tss": tss24, "hss": hss24,
            "n_evaluated": int(len(y24)),
            "definition": ("same active region, window nearest to "
                           "t - 1440 min within +/-35 min (Leka-style "
                           "24-hour-lagged persistence)"),
        }
        return blk

    # pooled test (in-partition protocol)
    out["pooled_test"] = baselines(te, "pooled_test")

    # P5 only (the leakage-free fold's test partition)
    out["p5_test"] = baselines(te[te["partition"] == 5], "p5_test")
    return out


# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(obj):
    """Strict-JSON sanitiser: NaN/Inf -> null (R-FS9-R1 A4 — the v4.0
    artefact carried four literal NaN tokens, which strict JSON
    parsers in JavaScript, R and Go reject outright)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main():
    t0 = time.time()
    report = {
        "purpose": ("standard flare-forecast verification apparatus "
                    "(Leka et al. 2019 toolkit): TSS, HSS, Brier skill "
                    "score, reliability, persistence and climatology "
                    "baselines — v4.1 (Dossier R-FS9-R1 items A1-A6)"),
        "definitions": {
            "tss": "TSS = POD - POFD = recall - FPR",
            "hss": "HSS = 2(TP*TN - FP*FN)/[(TP+FN)(FN+TN)+(TP+FP)(FP+TN)]",
            "bss": "BSS = 1 - Brier/Brier_climatology (test-set base rate)",
            "persistence_prev_window": ("same active region, immediately "
                                        "preceding window in ts_start_min "
                                        "order (hourly within-AR cadence; "
                                        "the 12-min figure is the MVTS "
                                        "record cadence, a different "
                                        "quantity): the label-inertia floor"),
            "persistence_24h_lag": ("same active region, window nearest "
                                    "to t - 1440 min within +/-35 min: "
                                    "the Leka-style 24-hour-lagged floor"),
            "zero_alert_semantics": ("stored precision = recall = 0 "
                                     "encodes a zero-alert forecast: "
                                     "fp = 0 and TSS = HSS = 0.0"),
        },
        "block_a_derived": block_a(),
        "block_b_recomputed": block_b(),
        "block_c_baselines": block_c(),
        "block_d_event_uncertainty": block_d(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(_sanitize(report), f, indent=2, allow_nan=False)
    print(f"[std] wrote {OUT} (strict JSON, allow_nan=False)")

    # console digest
    bc = report["block_c_baselines"]
    print(f"\n[within-AR window cadence] {bc['within_ar_window_cadence_min']:.0f} min")
    for split in ["pooled_test", "p5_test"]:
        print(f"\n[{split} baselines]")
        print(f"  climatology Brier {bc[split]['climatology']['brier']:.4f}")
        p = bc[split]["persistence_prev_window"]
        print(f"  persistence (prev window):  TSS {p['tss']:.3f}  HSS {p['hss']:.3f}  (n={p['n_evaluated']})")
        p24 = bc[split]["persistence_24h_lag"]
        print(f"  persistence (24h lag):      TSS {p24['tss']:.3f}  HSS {p24['hss']:.3f}  (n={p24['n_evaluated']})")
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
    print("\n[TSS at fbeta operating points — raw-substrate LSTM arms]")
    for name, e in report["block_a_derived"]["raw_substrate_lstm"].items():
        fb = e.get("fbeta_threshold", {})
        print(f"  {name:<38s} TSS {fb.get('tss', float('nan')):+.3f} "
              f"HSS {fb.get('hss', float('nan')):+.3f}")


if __name__ == "__main__":
    main()
