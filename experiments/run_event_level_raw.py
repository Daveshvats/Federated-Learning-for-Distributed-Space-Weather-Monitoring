"""
experiments/run_event_level_raw.py  (review items 4 + 9)
────────────────────────────────────────────────────────
Event-level evaluation of the RAW-SUBSTRATE models (see
experiments/run_raw_substrate.py) on raw partition 5, using the
raw-benchmark event keys of the provenance audit.

Window order here is RAW parse order; the audit's match CSV provides
raw_idx -> pkl_row, and the pooled test meta provides event_id /
ts_end_min per pkl_row.  29 of 75,365 windows are unmatched by the
audit matcher and are excluded from the event-level accounting
(disclosed in the output).

Inputs:
  data/cache/rawsubstrate/test_probs.npz   (y_test + calibrated probs)
  /tmp/swansf_raw/match_test_p5.csv        (raw_idx <-> pkl_row)
  /tmp/swansf_raw/test_meta_pooled.csv     (event_id, ts_end_min, ...)
  /tmp/swansf_raw/integrated_flare_data/goes_flares_integrated.csv
  outputs/raw_substrate_eval.json          (thresholds)

Output: outputs/event_level_raw_p5.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_event_level import event_level_metrics

RAW = "/tmp/swansf_raw"
CLONE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOLDOWN = 5   # 5 windows = 1 h at 12-min cadence
MODELS = ["logistic_regression", "xgboost", "centralized_mlp",
          "fedavg_mlp", "fedprox_mlp", "scaffold_mlp"]


def main():
    probs = np.load(os.path.join(CLONE, "data/cache/rawsubstrate",
                                 "test_probs.npz"))
    y = np.asarray(probs["y_test"]).astype(int)

    match = pd.read_csv(f"{RAW}/match_test_p5.csv", low_memory=False)
    match = match[match["verified"] == True]                 # noqa: E712
    match = match.drop_duplicates("raw_idx")
    r2p = dict(zip(match["raw_idx"], match["pkl_row"]))

    pooled = pd.read_csv(f"{RAW}/test_meta_pooled.csv", low_memory=False)
    pooled = pooled[pooled["partition"] == 5]
    p2meta = {r.pkl_row: r for r in pooled.itertuples()}

    rows = []
    for i in range(len(y)):
        p = r2p.get(i)
        if p is None or p not in p2meta:
            continue
        m = p2meta[p]
        rows.append((i, p, m.event_id, m.ts_end_min, str(m.flare_class),
                     int(m.label)))
    meta = pd.DataFrame(rows, columns=["raw_idx", "pkl_row", "event_id",
                                       "ts_end_min", "flare_class",
                                       "label"])
    assert (meta["label"].values == y[meta["raw_idx"].values]).all()
    print(f"[data] {len(y)} raw windows | {len(meta)} matched "
          f"({len(meta)/len(y):.2%}) | {meta['event_id'].nunique()} groups "
          f"| {meta.loc[meta.label == 1, 'event_id'].nunique()} true events")

    idx = meta["raw_idx"].values
    ev = meta["event_id"].values
    ts = meta["ts_end_min"].values.astype(float)
    yy = y[idx]
    true_groups = meta.loc[meta["label"] == 1, "event_id"].unique()

    fl = pd.read_csv(
        f"{RAW}/integrated_flare_data/goes_flares_integrated.csv",
        low_memory=False)
    fl["peak_dt"] = pd.to_datetime(fl["peak_time"])
    fl["cls"] = fl["goes_class"].astype(str)

    peak_map, matched, ambiguous = {}, 0, 0
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
    print(f"[peaks] matched {matched}/{len(true_groups)} true events; "
          f"ambiguous: {ambiguous}")

    report = json.load(open(os.path.join(CLONE, "outputs",
                                         "raw_substrate_eval.json")))
    out = {"purpose": ("event-level evaluation of the raw-substrate "
                       "(item 9) models on raw P5, natural prevalence"),
           "cooldown_windows": COOLDOWN,
           "n_windows": int(len(yy)),
           "n_matched_by_audit": int(len(meta)),
           "n_event_groups": int(meta["event_id"].nunique()),
           "n_true_events": int(len(true_groups)),
           "flare_peak_match_rate": matched / max(len(true_groups), 1),
           "models": {}}

    for name in MODELS:
        p = probs[name][idx]
        thr = report["results"][name]["threshold"]
        r = event_level_metrics(yy, p, ev, threshold=thr,
                                timestamps=ts, cooldown_windows=COOLDOWN)
        alerts = (p >= thr).astype(int)
        leads = []
        for g, peak_min in peak_map.items():
            gi = np.where(ev == g)[0]
            g_alerts = alerts[gi]
            if g_alerts.sum() > 0:
                first_alert = ts[gi][g_alerts == 1].min()
                leads.append(peak_min - first_alert)
        r["lead_to_flare_peak_minutes"] = {
            "median": float(np.median(leads)) if leads else None,
            "p10": float(np.percentile(leads, 10)) if leads else None,
            "p90": float(np.percentile(leads, 90)) if leads else None,
            "n": len(leads),
            "negative_means_alert_after_peak": True,
        }
        out["models"][name] = r
        lt = r["lead_to_flare_peak_minutes"]
        print(f"[{name:20s}] thr {thr:.3f} | events "
              f"{r['n_detected_events']}/{r['n_true_events']} "
              f"({r['event_detection_rate']*100:.1f}%) | FA/day "
              f"{r['false_alarm_windows_per_day']:.1f} | alerts/day "
              f"{r['alerts_per_day']:.1f} | dup "
              f"{r['duplicate_alert_rate']*100:.0f}% | lead "
              f"{(lt['median']/60 if lt['median'] is not None else float('nan')):.1f}h")

    path = os.path.join(CLONE, "outputs", "event_level_raw_p5.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[out] -> {path}")


if __name__ == "__main__":
    main()
