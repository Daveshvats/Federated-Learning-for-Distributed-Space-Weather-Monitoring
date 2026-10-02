#!/usr/bin/env python3
"""Event-level evaluation (review items 4/R13/R19) on the leakage-free
fold (test partition 5), using raw-benchmark event keys.

Inputs:
  clone/data/cache/pdisjoint/test_probs.npz  (y_test, probs per model)
  /tmp/swansf_raw/match_test_p5.csv          (pkl_row, event_id, ts, ar, ...)
  /tmp/swansf_raw/integrated_flare_data/goes_flares_integrated.csv (peaks)
  clone/outputs/partition_disjoint_eval.json (thresholds)

Outputs: clone/outputs/event_level_p5.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("SF9_CLONE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # v4.0: overridable
from experiments.run_event_level import event_level_metrics

CLONE = os.environ.get("SF9_CLONE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # v4.0: overridable
RAW = os.environ.get("SWANSF_RAW", "/tmp/swansf_raw")  # v4.0: overridable
COOLDOWN = 5   # 5 windows = 1 h at 12-min cadence


def main():
    probs = np.load(f"{CLONE}/data/cache/pdisjoint/test_probs.npz")
    y = np.asarray(probs["y_test"]).astype(int)
    meta = pd.read_csv(f"{RAW}/test_meta_pooled.csv", low_memory=False)
    meta = meta.loc[meta["partition"] == 5].sort_values("pkl_row").reset_index(drop=True)
    assert len(meta) == len(y), (len(meta), len(y))
    assert (meta["pkl_label"].values.astype(int) == y).all()

    ev = meta["event_id"].values
    ts = meta["ts_end_min"].values.astype(float)
    true_groups = meta.loc[meta["label"] == 1, "event_id"].unique()
    print(f"[data] {len(y)} windows | {meta['event_id'].nunique()} groups | "
          f"{len(true_groups)} true flare events | positives "
          f"{int(meta['label'].sum())}")

    # flare peak times from the addenda integrated list.
    # NOTE: raw instance filenames carry HARP numbers, not NOAA ARs, so
    # matching is by (GOES class, peak time in [window end, +48h]).
    fl = pd.read_csv(
        f"{RAW}/integrated_flare_data/goes_flares_integrated.csv",
        low_memory=False)
    fl["peak_dt"] = pd.to_datetime(fl["peak_time"])
    fl["cls"] = fl["goes_class"].astype(str)

    peak_map = {}
    matched = ambiguous = 0
    for g in true_groups:
        sub = meta[meta["event_id"] == g]
        last_end = sub["ts_end_min"].max()
        cls = str(sub["flare_class"].iloc[0])
        lo = pd.Timestamp(last_end, unit="m")
        cand = fl[(fl["cls"] == cls) & (fl["peak_dt"] >= lo)
                  & (fl["peak_dt"] <= lo + pd.Timedelta(hours=48))]
        if len(cand):
            peak_map[g] = (cand["peak_dt"].iloc[0]
                           - pd.Timestamp(0)).total_seconds() / 60.0
            matched += 1
            ambiguous += int(len(cand) > 1)
    print(f"[peaks] matched {matched}/{len(true_groups)} true events by "
          f"(class, time); ambiguous (multiple candidates): {ambiguous}")

    report = json.load(open(f"{CLONE}/outputs/partition_disjoint_eval.json"))
    out = {"cooldown_windows": COOLDOWN,
           "n_windows": int(len(y)),
           "n_event_groups": int(meta['event_id'].nunique()),
           "n_true_events": int(len(true_groups)),
           "flare_peak_match_rate": matched / max(len(true_groups), 1),
           "models": {}}

    for name in ["logistic_regression", "xgboost", "centralized_mlp",
                 "fedavg_mlp", "fedprox_mlp"]:
        p = probs[name]
        thr = report["results"][name]["threshold"]
        r = event_level_metrics(y, p, ev, threshold=thr,
                                timestamps=ts, cooldown_windows=COOLDOWN)
        # lead-to-flare-peak for detected events
        alerts = (p >= thr).astype(int)
        leads = []
        for g, peak_min in peak_map.items():
            idx = np.where(ev == g)[0]
            g_alerts = alerts[idx]
            if g_alerts.sum() > 0:
                first_alert = ts[idx][g_alerts == 1].min()
                leads.append(peak_min - first_alert)
        r["lead_to_flare_peak_minutes"] = {
            "median": float(np.median(leads)) if leads else None,
            "p10": float(np.percentile(leads, 10)) if leads else None,
            "p90": float(np.percentile(leads, 90)) if leads else None,
            "min": float(np.min(leads)) if leads else None,
            "max": float(np.max(leads)) if leads else None,
            "n": len(leads),
            "negative_means_alert_after_peak": True,
        }
        out["models"][name] = r
        lt = r["lead_to_flare_peak_minutes"]
        print(f"[{name:20s}] thr {thr:.3f} | events "
              f"{r['n_detected_events']}/{r['n_true_events']} "
              f"({r['event_detection_rate']*100:.1f}%) | "
              f"false-alarm windows/day {r['false_alarm_windows_per_day']:.1f}"
              f" | alerts/day {r['alerts_per_day']:.1f} | "
              f"dup rate {r['duplicate_alert_rate']*100:.0f}% | "
              f"lead-to-peak median "
              f"{(lt['median']/60 if lt['median'] is not None else float('nan')):.1f}h")

    with open(f"{CLONE}/outputs/event_level_p5.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[out] -> {CLONE}/outputs/event_level_p5.json")


if __name__ == "__main__":
    main()
