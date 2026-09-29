"""
experiments/run_event_level.py  (review R13/R19)
────────────────────────────────────────────────
Event-level evaluation for solar-flare alerting.

MOTIVATION
    Window-level metrics (ROC-AUC, PR-AUC, recall@FPR) treat every
    60-timestep sliding window as an independent decision. Operationally
    they are not: a single active region can emit dozens of overlapping
    windows, and a single physical flare event triggers many correlated
    model decisions. A 2% window-level FPR therefore does NOT mean "2%
    of operational alerts are false", and one detected event can
    generate a burst of duplicate alerts.

    This module regroups window decisions into physical events and
    reports the metrics an operations desk actually needs:
      * event detection rate  (fraction of true events with >=1 alert)
      * missed events         (list)
      * false-alarm windows per event-free day (needs timestamps)
      * alerts per detected event / duplicate-alert rate
      * warning lead time     (first alert minus event start; needs
                               timestamps), median + minimum
      * alert rate per day    (total alerts / observation days)

USAGE
    Programmatic:
        from experiments.run_event_level import event_level_metrics
        report = event_level_metrics(y_true, probs, event_ids,
                                     timestamps=ts, threshold=0.51)

    CLI (owner re-run after the next training pass):
        python experiments/run_event_level.py --input outputs/event_scores.json
    where the JSON holds {y_true, probs, event_ids, timestamps?, threshold}.

    The event_ids input should be a physical-event grouping key —
    ideally (active-region id, flare peak time) from raw SWAN-SF
    metadata; a synthetic grouping (region id alone) is accepted with
    an explicit caveat.
"""
from __future__ import annotations

import argparse
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def event_level_metrics(y_true, y_probs, event_ids, threshold=0.5,
                        timestamps=None, cooldown_windows=0,
                        alert_label="alert"):
    """
    Parameters
    ----------
    y_true : array of window labels (1 = window precedes/tracks an event)
    y_probs : array of window probabilities
    event_ids : array; every window sharing a value belongs to one
        physical event-group (e.g. region id + event time cluster)
    threshold : decision threshold applied to y_probs
    timestamps : optional array (numeric, e.g. minutes); enables lead
        time and per-day alert rates
    cooldown_windows : optional alert de-duplication: after a window
        raises an alert, suppress subsequent alerts for this many
        windows within the same event group (models persistence logic)

    Returns a report dict; all fields are JSON-serialisable.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)
    event_ids = np.asarray(event_ids)
    n = len(y_true)
    if not (len(y_probs) == len(event_ids) == n):
        raise ValueError("y_true, y_probs, event_ids must be aligned")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=float)
        if len(timestamps) != n:
            raise ValueError("timestamps must align with windows")

    alerts = (y_probs >= threshold).astype(int)

    # ── group windows by event id ─────────────────────────────────────
    groups = {}
    for i in range(n):
        groups.setdefault(str(event_ids[i]), []).append(i)

    event_groups = {g: idx for g, idx in groups.items()}

    # ── alert de-duplication (cooldown / persistence logic, R13) ──────
    # Applied ONCE, within each event group, before any statistic is
    # computed, so total alerts, false alarms, detection and duplicate
    # rates are all consistent with the de-duplicated alert stream an
    # operator would actually see.
    if cooldown_windows > 0:
        alerts = alerts.copy()
        for g, idx in event_groups.items():
            g_idx = np.sort(np.asarray(idx))
            last_alert_rel = -10**9
            for rel, w in enumerate(g_idx):
                if alerts[w] == 1:
                    if rel - last_alert_rel >= cooldown_windows:
                        last_alert_rel = rel
                    else:
                        alerts[w] = 0

    true_events = {g: idx for g, idx in event_groups.items()
                   if y_true[idx].max() == 1}
    background_groups = {g: idx for g, idx in event_groups.items()
                         if y_true[idx].max() == 0}

    detected, missed, alerts_per_event, lead_times = [], [], [], []
    for g, idx in true_events.items():
        idx = np.asarray(sorted(idx))
        g_alerts = alerts[idx]
        n_alerts = int(g_alerts.sum())
        alerts_per_event.append(n_alerts)
        if n_alerts > 0:
            detected.append(g)
            if timestamps is not None:
                event_start = timestamps[idx][y_true[idx] == 1].min()
                first_alert = timestamps[idx][g_alerts == 1].min()
                lead_times.append(float(first_alert - event_start))
        else:
            missed.append(g)

    # ── background (event-free) false alarms ──────────────────────────
    false_alarm_windows = int(sum(int(alerts[np.asarray(idx)].sum())
                                  for idx in background_groups.values()))
    false_alarm_groups = int(sum(1 for idx in background_groups.values()
                                 if alerts[np.asarray(idx)].sum() > 0))

    total_alerts = int(alerts.sum())
    report = {
        "threshold": float(threshold),
        "cooldown_windows": int(cooldown_windows),
        "n_windows": int(n),
        "n_event_groups": len(event_groups),
        "n_true_events": len(true_events),
        "n_detected_events": len(detected),
        "event_detection_rate": (len(detected) / len(true_events)
                                 if true_events else None),
        "n_missed_events": len(missed),
        "missed_events": missed[:50],
        "alerts_per_detected_event_mean": (
            float(np.mean(alerts_per_event)) if alerts_per_event else None),
        "duplicate_alert_rate": (
            float(1.0 - 1.0 / np.mean([a for a in alerts_per_event if a > 0]))
            if any(a > 1 for a in alerts_per_event) else 0.0),
        "false_alarm_windows": false_alarm_windows,
        "background_groups_with_alerts": false_alarm_groups,
        "total_alerts": total_alerts,
        "alert_rate_per_window": float(alerts.mean()),
        "lead_time_minutes": {
            "median": float(np.median(lead_times)) if lead_times else None,
            "min": float(np.min(lead_times)) if lead_times else None,
            "max": float(np.max(lead_times)) if lead_times else None,
            "n_events_with_lead_time": len(lead_times),
        } if timestamps is not None else None,
    }
    if timestamps is not None:
        span = float(timestamps.max() - timestamps.min())
        days = max(span / 1440.0, 1e-9)  # minutes -> days
        report["observation_days"] = days
        report["alerts_per_day"] = float(total_alerts / days)
        report["false_alarm_windows_per_day"] = float(false_alarm_windows / days)
    report["caveats"] = [
        "window-level FPR != operational false-alarm rate: overlapping "
        "windows of one active region are correlated decisions",
        "event grouping quality bounds all event-level statistics; "
        "region-id-only grouping is an approximation",
        "lead time requires the event start time; without raw SWAN-SF "
        "metadata it is reported as null",
    ]
    return report


def _synthetic_self_check():
    """Deterministic fixture: 2 events detected, 1 missed, known leads."""
    # event A: windows 0..9 (labels 1), probs high at window 2 (alert)
    # event B: windows 10..19 (labels 1), no alert -> missed
    # event C: background windows 20..29, one false alert at 25
    y = np.array([1]*10 + [1]*10 + [0]*10)
    p = np.full(30, 0.1)
    p[2] = 0.9; p[25] = 0.9
    ev = np.array(["A"]*10 + ["B"]*10 + ["C"]*10)
    ts = np.arange(30) * 60.0  # 60-minute spacing
    r = event_level_metrics(y, p, ev, threshold=0.5, timestamps=ts)
    ok = (r["n_true_events"] == 2 and r["n_detected_events"] == 1
          and r["n_missed_events"] == 1
          and r["event_detection_rate"] == 0.5
          and r["false_alarm_windows"] == 1
          and r["lead_time_minutes"]["median"] == 2*60.0)
    return ok, r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", default=None,
                    help="JSON with {y_true, probs, event_ids, "
                         "timestamps?, threshold?}")
    ap.add_argument("--self-check", action="store_true",
                    help="run the deterministic synthetic fixture")
    ap.add_argument("--output", default="outputs/event_level_report.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cooldown", type=int, default=0)
    args = ap.parse_args()

    if args.self_check or args.input is None:
        ok, r = _synthetic_self_check()
        print(json.dumps(r, indent=2))
        print(f"[Self-check] {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    with open(args.input) as f:
        data = json.load(f)
    r = event_level_metrics(
        data["y_true"], data["probs"], data["event_ids"],
        threshold=data.get("threshold", args.threshold),
        timestamps=data.get("timestamps"),
        cooldown_windows=args.cooldown)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(r, f, indent=2)
    print(f"[EventLevel] report -> {args.output}")
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("missed_events", "caveats")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
