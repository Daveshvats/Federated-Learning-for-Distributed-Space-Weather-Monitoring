"""
experiments/run_event_level_lstm.py  (review item 8, event-level half)
────────────────────────────────────────────────────────────────
Standalone, torch-free event-level evaluation of the raw-substrate
LSTM arms (see experiments/run_federated_lstm.py) on raw partition 5.

It exists so the event-level pass can run on ANY machine — including a
CPU-only box — from the two small artefacts the GPU run leaves behind:

  data/cache/rawsubstrate/test_probs_lstm.npz   (y_test + calibrated
                                                 test probs per arm)
  outputs/raw_lstm_eval.json                    (validation-frozen
                                                 thresholds per arm)

plus the aux metadata (audit match table, pooled test meta, GOES flare
list) under data/cache/rawsubstrate/aux/ — the GPU-bundle layout, also
where run_federated_lstm.py looks for it.

Usage:
    python experiments/run_event_level_lstm.py            # tag 'lstm'
    python experiments/run_event_level_lstm.py --tag seed43

Output: outputs/event_level_raw_lstm_p5.json (or _<tag>.json)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_event_level import event_level_metrics

CLONE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(CLONE, "data", "cache", "rawsubstrate")
COOLDOWN = 5   # 5 windows = 1 h at 12-min cadence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lstm",
                    help="probs/report tag (test_probs_<tag>.npz)")
    ap.add_argument("--cache", default=CACHE,
                    help="substrate cache dir (probs npz + aux/)")
    ap.add_argument("--aux", default=None,
                    help="aux dir override (default: <cache>/aux)")
    ap.add_argument("--eval", default=None,
                    help="eval json override (default: "
                         "outputs/raw_lstm_eval.json)")
    ap.add_argument("--out", default=None,
                    help="output json override")
    ap.add_argument("--cooldown", type=int, default=COOLDOWN)
    args = ap.parse_args()

    aux = args.aux or os.path.join(args.cache, "aux")
    probs_path = os.path.join(args.cache, f"test_probs_{args.tag}.npz")
    eval_path = args.eval or os.path.join(
        CLONE, "outputs",
        "raw_lstm_eval.json" if args.tag == "lstm"
        else f"raw_lstm_{args.tag}.json")
    out_json = args.out or os.path.join(
        CLONE, "outputs",
        "event_level_raw_lstm_p5.json" if args.tag == "lstm"
        else f"event_level_raw_{args.tag}.json")

    need = [os.path.join(aux, "match_test_p5.csv"),
            os.path.join(aux, "test_meta_pooled.csv"),
            os.path.join(aux, "integrated_flare_data",
                         "goes_flares_integrated.csv")]
    missing = [p for p in need + [probs_path, eval_path] if
               not os.path.exists(p)]
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))

    report = json.load(open(eval_path))
    thresholds = {k: v["threshold"] for k, v in report["results"].items()}
    order = [k for k in ("central_lstm", "fedavg_lstm", "fedprox_lstm")
             if k in thresholds]

    probs = np.load(probs_path)
    y = np.asarray(probs["y_test"]).astype(int)

    match = pd.read_csv(os.path.join(aux, "match_test_p5.csv"),
                        low_memory=False)
    match = match[match["verified"] == True]                # noqa: E712
    match = match.drop_duplicates("raw_idx")
    r2p = dict(zip(match["raw_idx"], match["pkl_row"]))

    pooled = pd.read_csv(os.path.join(aux, "test_meta_pooled.csv"),
                         low_memory=False)
    pooled = pooled[pooled["partition"] == 5]
    p2meta = {r.pkl_row: r for r in pooled.itertuples()}

    rows = []
    for i in range(len(y)):
        p = r2p.get(i)
        if p is None or p not in p2meta:
            continue
        m = p2meta[p]
        rows.append((i, m.event_id, m.ts_end_min, str(m.flare_class),
                     int(m.label)))
    meta = pd.DataFrame(rows, columns=["raw_idx", "event_id", "ts_end_min",
                                       "flare_class", "label"])
    assert (meta["label"].values == y[meta["raw_idx"].values]).all(), \
        "label mismatch between probs npz and audit metadata"

    idx = meta["raw_idx"].values
    ev = meta["event_id"].values
    ts = meta["ts_end_min"].values.astype(float)
    yy = y[idx]
    true_groups = meta.loc[meta["label"] == 1, "event_id"].unique()
    print(f"[data] {len(y):,} raw windows | {len(meta):,} matched "
          f"({len(meta)/len(y):.2%}) | {meta['event_id'].nunique()} "
          f"groups | {len(true_groups)} true events", flush=True)

    fl = pd.read_csv(os.path.join(aux, "integrated_flare_data",
                                  "goes_flares_integrated.csv"),
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
          f"ambiguous: {ambiguous}", flush=True)

    out = {"purpose": ("event-level evaluation of the raw-substrate LSTM "
                       "arms (central / FedAvg / FedProx SolarLSTM) on "
                       "raw P5, natural prevalence — standalone CPU pass "
                       "over the GPU run's test_probs_<tag>.npz"),
           "cooldown_windows": args.cooldown,
           "n_windows": int(len(yy)),
           "n_matched_by_audit": int(len(meta)),
           "n_event_groups": int(meta["event_id"].nunique()),
           "n_true_events": int(len(true_groups)),
           "flare_peak_match_rate": matched / max(len(true_groups), 1),
           "models": {}}
    for name in order:
        p = np.asarray(probs[name], dtype=float)[idx]
        thr = thresholds[name]
        r = event_level_metrics(yy, p, ev, threshold=thr,
                                timestamps=ts, cooldown_windows=args.cooldown)
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
        print(f"[{name:<14s}] thr {thr:.3f} | events "
              f"{r['n_detected_events']}/{r['n_true_events']} "
              f"({r['event_detection_rate']*100:.1f}%) | FA/day "
              f"{r['false_alarm_windows_per_day']:.2f} | alerts/day "
              f"{r['alerts_per_day']:.2f} | lead "
              f"{(lt['median']/60 if lt['median'] is not None else float('nan')):.1f}h",
              flush=True)

    tmp = out_json + ".tmp"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(tmp, out_json)
    print(f"-> {out_json}", flush=True)


if __name__ == "__main__":
    main()
