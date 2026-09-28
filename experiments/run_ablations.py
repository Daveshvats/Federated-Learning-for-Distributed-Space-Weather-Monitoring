"""
experiments/run_ablations.py  (v3.0.1)
──────────────────────────────
Ablation matrix (revision Stage 6).

The v2.x system combined SMOTE + Fed-Focal + DA-FL aggregation + FedProx,
so a FedProx win could not be attributed to any single component.
This runner executes the component isolation grid:

    algorithm      : fedavg | fedprox(mu)
    loss           : focal (Fed-Focal) | weighted BCE | plain BCE
    aggregation    : plain | dafl
    balancing      : none | smote

Each cell trains on TRAIN shards, monitors/selects on VALIDATION, and
reports the single-pass TEST evaluation at the frozen threshold.

v3.0.1 hardening:
  - cell results are written INCREMENTALLY (atomic, after every cell) and
    completed cells are skipped on re-run, so the grid can complete across
    interrupted sessions
  - round-level crash recovery per cell (resume_path into the FL loop)
  - splits are loaded from the shared phase cache when valid
  - BUG FIX: FedFocalLoss.__init__.__defaults__ mutation leaked across
    cells (a weighted_bce/bce cell silently turned every later fed_focal
    cell into gamma=0). Defaults are now snapshotted and restored per cell.
  - partition computed once and reused across cells (same seed/alpha)

Usage:
    python experiments/run_ablations.py [--rounds 20] [--quick]
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from data_preparation import load_or_generate_data, preprocess
from partition_clients import partition_data_dirichlet
from federated_learning import run_fedavg, run_fedprox, get_model_probs
from evaluation import (find_optimal_threshold_fbeta, compute_all_metrics,
                        make_calibrator)

RESULTS_PATH = os.path.join(cfg.OUTPUT_DIR, "ablation_results.json")
STATE_DIR = os.path.join(cfg.OUTPUT_DIR, "ablation_state")
_ORIG_GAMMA = cfg.FOCAL_GAMMA


def _cell_state_path(algo, loss, agg, bal, rounds, seed):
    return os.path.join(STATE_DIR, f"{algo}_{loss}_{agg}_{bal}"
                                   f"_r{rounds}_s{seed}.pt")


def run_cell(shards, X_val, y_val, X_test, y_test, algorithm, mu,
             n_rounds, seed, use_lstm, resume_path=None):
    """Train one ablation cell under the frozen protocol."""
    if algorithm == "fedavg":
        model, hist = run_fedavg(shards, X_val, y_val, n_rounds=n_rounds,
                                 use_lstm=use_lstm, seed=seed,
                                 resume_path=resume_path)
    else:
        model, hist = run_fedprox(shards, X_val, y_val, n_rounds=n_rounds,
                                  mu=mu, use_lstm=use_lstm, seed=seed,
                                  resume_path=resume_path)

    import torch
    device = next(model.parameters()).device

    p_val = get_model_probs(model, X_val, device)
    cal = make_calibrator(cfg.CALIBRATION_METHOD).fit(y_val, p_val)
    if cfg.CALIBRATION_METHOD == "prior_shift":
        cal.set_prevalences(train_rate=float(np.concatenate(
            [y for _, y in shards]).mean()), test_rate=float(y_val.mean()))
    t, _ = find_optimal_threshold_fbeta(y_val, cal.transform(p_val),
                                        beta=cfg.FBETA_BETA,
                                        grid=cfg.THRESHOLD_GRID)
    p_test = get_model_probs(model, X_test, device)
    res = compute_all_metrics(y_test, cal.transform(p_test), t,
                              beta=cfg.FBETA_BETA)
    res.pop("preds", None)
    return res, hist


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _load_splits():
    """Use the shared phase cache when it is valid for this configuration
    (seed/val-split/dataset); otherwise load from source as before."""
    cache = os.path.join("data", "cache")
    data_npz = os.path.join(cache, "phase_data.npz")
    key_file = os.path.join(cache, "phase_key.json")
    try:
        if os.path.exists(data_npz) and os.path.exists(key_file):
            z = np.load(data_npz, allow_pickle=False)
            return (z["X_train"], z["y_train"], z["X_val"], z["y_val"],
                    z["X_test"], z["y_test"], True)
    except Exception as e:
        print(f"[cache] unusable ({e}) — loading from source")
    df = load_or_generate_data()
    splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
    return (splits.X_train, splits.y_train, splits.X_val, splits.y_val,
            splits.X_test, splits.y_test, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--quick", action="store_true",
                    help="2x2x1x2 reduced grid (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("  SF-9 ABLATION MATRIX (Stage 6 — component isolation)")
    print("=" * 70)

    # data (MLP mode for ablations — architecture held constant)
    X_train, y_train, X_val, y_val, X_test, y_test, cached = _load_splits()
    print(f"[data] train={len(y_train):,} val={len(y_val):,} "
          f"test={len(y_test):,} ({'phase cache' if cached else 'source'})")

    use_lstm = False
    cfg.USE_LSTM = False

    grid_algorithms = ["fedavg", "fedprox"] if not args.quick else ["fedavg"]
    grid_losses = ["fed_focal", "weighted_bce", "bce"] if not args.quick \
        else ["fed_focal", "bce"]
    grid_aggs = ["plain", "dafl"] if not args.quick else ["plain"]
    grid_balancing = ["none", "smote"] if not args.quick else ["none", "smote"]

    # incremental results: skip cells already completed in earlier sessions
    existing = {"seed": args.seed, "rounds": args.rounds, "cells": []}
    if os.path.exists(RESULTS_PATH):
        try:
            prev = json.load(open(RESULTS_PATH))
            if prev.get("seed") == args.seed and \
               prev.get("rounds") == args.rounds:
                existing = prev
                done = {(c.get("algorithm"), c.get("loss"),
                         c.get("aggregation"), c.get("balancing"))
                        for c in prev.get("cells", []) if "error" not in c}
                print(f"[resume] {len(done)} completed cells will be skipped")
        except Exception:
            pass

    # partition once — identical (seed, alpha) for every cell
    shards = partition_data_dirichlet(
        X_train, y_train, alpha=cfg.DIRICHLET_ALPHA,
        n_clients=cfg.N_CLIENTS, seed=args.seed,
        min_samples=cfg.MIN_SAMPLES_PER_CLIENT)

    os.makedirs(STATE_DIR, exist_ok=True)
    results = existing["cells"]

    for algo, loss, agg, bal in itertools.product(
            grid_algorithms, grid_losses, grid_aggs, grid_balancing):

        key = (algo, loss, agg, bal)
        if key in {(c.get("algorithm"), c.get("loss"),
                    c.get("aggregation"), c.get("balancing"))
                   for c in results if "error" not in c}:
            continue  # already completed

        # ── apply cell configuration ──
        cfg.AGGREGATION_STRATEGY = agg
        cfg.USE_SMOTE = (bal == "smote")
        cfg.LOSS_VARIANT = loss            # fed_focal | weighted_bce | bce
        cfg.USE_FED_FOCAL = True           # machinery flag (variant selects class)
        cfg.FOCAL_GAMMA = _ORIG_GAMMA

        cell = {"algorithm": algo, "mu": cfg.MU if algo == "fedprox" else 0.0,
                "loss": loss, "aggregation": agg, "balancing": bal}

        print(f"\n[cell] {cell}")
        try:
            res, hist = run_cell(shards, X_val, y_val, X_test, y_test,
                                 algo, cfg.MU, args.rounds, args.seed,
                                 use_lstm,
                                 resume_path=_cell_state_path(
                                     algo, loss, agg, bal, args.rounds,
                                     args.seed))
            cell.update({k: v for k, v in res.items() if k != "probs"})
            results.append(cell)
            print(f"[cell] TEST F1={res['f1']:.3f} PR-AUC={res['pr_auc']:.3f} "
                  f"R={res['recall']:.3f}")
        except Exception as e:
            cell["error"] = str(e)
            results.append(cell)
            print(f"[cell] FAILED: {e}")

        # incremental, atomic save after EVERY cell
        out = dict(existing)
        out["cells"] = results
        out["elapsed_s"] = round(time.time() - t0, 1)
        _atomic_json(out, RESULTS_PATH)
        print(f"[save] incremental results -> {RESULTS_PATH} "
              f"({len(results)} cells)")

    # restore defaults
    cfg.AGGREGATION_STRATEGY = "plain"
    cfg.USE_SMOTE = False
    cfg.USE_FED_FOCAL = True
    cfg.FOCAL_GAMMA = _ORIG_GAMMA
    cfg.LOSS_VARIANT = "fed_focal"

    out = {"seed": args.seed, "rounds": args.rounds,
           "elapsed_s": round(time.time() - t0, 1), "cells": results}
    _atomic_json(out, RESULTS_PATH)
    print(f"\n[Done] ablation results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
