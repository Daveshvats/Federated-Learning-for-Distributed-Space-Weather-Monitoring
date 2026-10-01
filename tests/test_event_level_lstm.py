"""
tests/test_event_level_lstm.py  (v3.6)
──────────────────────────────────────
Unit tests for experiments/run_event_level_lstm.py — the standalone,
torch-free event-level pass over a GPU run's test_probs_<tag>.npz:

  1. missing-input guard: SystemExit listing every absent artefact
  2. end-to-end wiring on synthetic aux metadata + probs + eval json:
     correct match/meta join, label assert, GOES peak join, all three
     arms reported, deterministic detection/lead-time arithmetic
  3. label-mismatch guard: corrupted y_test trips the assert
  4. Windows _aux fallback: an 'aux' directory renamed '_aux' (AUX
     is a reserved Windows device name) resolves without --aux

Runs without torch.  Run:  python tests/test_event_level_lstm.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments", "run_event_level_lstm.py")


def build_fixture(tmp, corrupt_labels=False):
    """Synthetic mini-audit: 40 test windows, 3 event groups (2 true
    flaring events + background), a GOES list with one peak per true
    event, and a probs npz whose three arms differ only in noise."""
    aux = os.path.join(tmp, "aux")
    os.makedirs(os.path.join(aux, "integrated_flare_data"))

    rng = np.random.default_rng(0)
    n = 40
    ev = np.array(["BG"] * 20 + ["EV1"] * 10 + ["EV2"] * 10)
    lab = (ev != "BG").astype(int)
    # BG: 0-228 | EV1: 240-348 | EV2: 4000-4108 — the two events are
    # >48h apart so each event's GOES search window [last_end,
    # last_end+48h] contains exactly its own peak (no ambiguous match)
    ts = np.concatenate([np.arange(20) * 12.0,
                         240.0 + np.arange(10) * 12.0,
                         4000.0 + np.arange(10) * 12.0])

    pooled = pd.DataFrame({
        "partition": 5,
        "pkl_row": np.arange(n),
        "event_id": ev,
        "ts_end_min": ts,
        "flare_class": np.where(lab == 1, "M1", "FQ"),
        "label": lab,
    })
    pooled.to_csv(os.path.join(aux, "test_meta_pooled.csv"), index=False)

    match = pd.DataFrame({"raw_idx": np.arange(n),
                          "pkl_row": np.arange(n),
                          "verified": True})
    match.to_csv(os.path.join(aux, "match_test_p5.csv"), index=False)

    # one GOES peak 10h (600 min) after each true event's last window
    goes = []
    for g in ("EV1", "EV2"):
        last_end = ts[ev == g].max()
        peak = pd.Timestamp(last_end, unit="m") + pd.Timedelta(hours=10)
        goes.append({"goes_class": "M1", "peak_time": str(peak)})
    pd.DataFrame(goes).to_csv(
        os.path.join(aux, "integrated_flare_data",
                     "goes_flares_integrated.csv"), index=False)

    # probs: positives ~0.9 (above thr), negatives ~0.02 (below);
    # arm 2 (fedavg) misses EV2 (its windows never cross the threshold)
    y = lab.copy()
    if corrupt_labels:
        y[1] = 1 - y[1]
    probs = {}
    for arm, miss in (("central_lstm", None),
                      ("fedavg_lstm", "EV2"),
                      ("fedprox_lstm", None)):
        p = np.where(lab == 1, 0.9, 0.02).astype(np.float32)
        p = p + rng.normal(0, 0.005, n).astype(np.float32)
        if miss:
            p[ev == miss] = 0.05
        probs[arm] = np.clip(p, 1e-4, 0.999)
    np.savez_compressed(os.path.join(tmp, "test_probs_lstm.npz"),
                        y_test=y.astype(np.int8), **probs)

    evj = {"results": {"central_lstm": {"threshold": 0.5},
                       "fedavg_lstm": {"threshold": 0.5},
                       "fedprox_lstm": {"threshold": 0.5}}}
    eval_path = os.path.join(tmp, "raw_lstm_eval.json")
    json.dump(evj, open(eval_path, "w"))
    return aux, eval_path, os.path.join(tmp, "test_probs_lstm.npz")


def run_script(aux, eval_path, out_json, cache, extra=()):
    cmd = [sys.executable, SCRIPT, "--cache", cache]
    if aux is not None:
        cmd += ["--aux", aux]
    cmd += ["--eval", eval_path, "--out", out_json, *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_missing_inputs():
    tmp = tempfile.mkdtemp(prefix="evlstm_miss_")
    try:
        r = run_script(os.path.join(tmp, "nope"),
                       os.path.join(tmp, "nope.json"),
                       os.path.join(tmp, "out.json"), tmp)
        check("missing inputs exit nonzero with a list",
              r.returncode != 0 and "missing inputs" in r.stderr
              and "match_test_p5.csv" in r.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end():
    tmp = tempfile.mkdtemp(prefix="evlstm_e2e_")
    try:
        aux, eval_path, probs_path = build_fixture(tmp)
        out_json = os.path.join(tmp, "event_level_raw_lstm_p5.json")
        r = run_script(aux, eval_path, out_json, tmp)
        check("script exits 0 on a complete fixture",
              r.returncode == 0, r.stderr[-300:])
        out = json.load(open(out_json))

        check("all three arms reported in canonical order",
              list(out["models"]) ==
              ["central_lstm", "fedavg_lstm", "fedprox_lstm"])
        check("audit accounting: 40 windows, 40 matched, 2 true events",
              out["n_windows"] == 40 and out["n_matched_by_audit"] == 40
              and out["n_true_events"] == 2)
        check("GOES peaks matched 2/2 (100%)",
              abs(out["flare_peak_match_rate"] - 1.0) < 1e-9)

        m = out["models"]["central_lstm"]
        check("central/fedprox detect 2/2 events",
              m["n_detected_events"] == 2 and
              out["models"]["fedprox_lstm"]["n_detected_events"] == 2)
        check("fedavg (blinded to EV2) detects only 1/2",
              out["models"]["fedavg_lstm"]["n_detected_events"] == 1)

        # lead time: first alert of each event = its first window
        # (t=240 for EV1, t=360+... for EV2); peak = last window +10h
        # EV1: first alert ts=240, last end=348, peak=948 -> 708 min
        leads = m["lead_to_flare_peak_minutes"]
        check("median lead-to-peak arithmetic (11.8h = 708 min)",
              leads["median"] is not None and
              abs(leads["median"] - 708.0) < 1e-6 and
              leads["n"] == 2, str(leads))
        check("model record carries the event_level_metrics schema",
              all(k in m for k in
                  ("n_detected_events", "event_detection_rate",
                   "false_alarm_windows_per_day", "alerts_per_day")))
        check("cooldown recorded from the protocol default",
              out["cooldown_windows"] == 5)
        return tmp, aux, eval_path
    except Exception as e:                     # noqa: BLE001
        check("end-to-end fixture run", False, str(e))
        shutil.rmtree(tmp, ignore_errors=True)
        return None, None, None


def test_label_mismatch_guard():
    tmp = tempfile.mkdtemp(prefix="evlstm_bad_")
    try:
        aux, eval_path, _ = build_fixture(tmp, corrupt_labels=True)
        r = run_script(aux, eval_path, os.path.join(tmp, "out.json"), tmp)
        check("corrupted y_test trips the label assert",
              r.returncode != 0 and "label mismatch" in r.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_windows_aux_fallback():
    """Windows extractors rename bundle 'aux' -> '_aux' (AUX is a
    reserved device name); the runner must resolve it without --aux
    exactly as with an explicit --aux path."""
    tmp = tempfile.mkdtemp(prefix="evlstm_waux_")
    try:
        aux, eval_path, _ = build_fixture(tmp)
        os.rename(aux, os.path.join(tmp, "_aux"))
        out_json = os.path.join(tmp, "event_level_raw_lstm_p5.json")
        r = run_script(None, eval_path, out_json, tmp)
        check("script resolves _aux without --aux (Windows case)",
              r.returncode == 0, r.stderr[-300:])
        if r.returncode == 0:
            out = json.load(open(out_json))
            check("fallback run reports the same 2 true events",
                  out["n_true_events"] == 2 and
                  out["models"]["central_lstm"]["n_detected_events"] == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("── tests/test_event_level_lstm.py ──")
    test_missing_inputs()
    tmp, aux, eval_path = test_end_to_end()
    test_label_mismatch_guard()
    test_windows_aux_fallback()
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
