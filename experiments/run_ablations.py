"""
experiments/run_ablations.py  (v3.0)
────────────────────────────────────
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

Usage:
    python experiments/run_ablations.py [--rounds 20] [--quick]
"""

import argparse
import copy
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from data_preparation import load_or_generate_data, preprocess, \
    load_and_scale_3d_data
from partition_clients import partition_data_dirichlet
from federated_learning import run_fedavg, run_fedprox, get_model_probs
from evaluation import (find_optimal_threshold_fbeta, compute_all_metrics,
                        make_calibrator)


def run_cell(shards, X_val, y_val, X_test, y_test, algorithm, mu,
             n_rounds, seed, use_lstm):
    """Train one ablation cell under the frozen protocol."""
    if algorithm == "fedavg":
        model, hist = run_fedavg(shards, X_val, y_val, n_rounds=n_rounds,
                                 use_lstm=use_lstm, seed=seed)
    else:
        model, hist = run_fedprox(shards, X_val, y_val, n_rounds=n_rounds,
                                  mu=mu, use_lstm=use_lstm, seed=seed)

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
    df = load_or_generate_data()
    splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test

    use_lstm = False
    cfg.USE_LSTM = False

    grid_algorithms = ["fedavg", "fedprox"] if not args.quick else ["fedavg"]
    grid_losses = ["fed_focal", "weighted_bce", "bce"] if not args.quick \
        else ["fed_focal", "bce"]
    grid_aggs = ["plain", "dafl"] if not args.quick else ["plain"]
    grid_balancing = ["none", "smote"] if not args.quick else ["none", "smote"]

    results = []
    for algo, loss, agg, bal in itertools.product(
            grid_algorithms, grid_losses, grid_aggs, grid_balancing):

        # ── apply cell configuration ──
        cfg.AGGREGATION_STRATEGY = agg
        cfg.USE_SMOTE = (bal == "smote")
        if loss == "fed_focal":
            cfg.USE_FED_FOCAL = True
        else:
            cfg.USE_FED_FOCAL = False
            # weighted_bce vs plain bce: set DynamicFocalLoss base_alpha
            import losses as losses_mod
            if loss == "weighted_bce":
                # gamma=0 reduces focal loss to alpha-weighted BCE
                losses_mod.FedFocalLoss.__init__.__defaults__ = (0.0, 0.25, 'mean')
                cfg.USE_FED_FOCAL = True
                cfg.FOCAL_GAMMA = 0.0
            else:
                cfg.FOCAL_GAMMA = 0.0
                losses_mod.FedFocalLoss.__init__.__defaults__ = (0.0, 0.5, 'mean')
                cfg.USE_FED_FOCAL = True

        cell = {"algorithm": algo, "mu": cfg.MU if algo == "fedprox" else 0.0,
                "loss": loss, "aggregation": agg, "balancing": bal}

        print(f"\n[cell] {cell}")
        shards = partition_data_dirichlet(
            X_train, y_train, alpha=cfg.DIRICHLET_ALPHA,
            n_clients=cfg.N_CLIENTS, seed=args.seed,
            min_samples=cfg.MIN_SAMPLES_PER_CLIENT)

        try:
            res, hist = run_cell(shards, X_val, y_val, X_test, y_test,
                                 algo, cfg.MU, args.rounds, args.seed,
                                 use_lstm)
            cell.update({k: v for k, v in res.items() if k != "probs"})
            results.append(cell)
            print(f"[cell] TEST F1={res['f1']:.3f} PR-AUC={res['pr_auc']:.3f} "
                  f"R={res['recall']:.3f}")
        except Exception as e:
            cell["error"] = str(e)
            results.append(cell)
            print(f"[cell] FAILED: {e}")

    # restore defaults
    cfg.AGGREGATION_STRATEGY = "plain"
    cfg.USE_SMOTE = False
    cfg.USE_FED_FOCAL = True
    cfg.FOCAL_GAMMA = 2.0

    out = {"seed": args.seed, "rounds": args.rounds,
           "elapsed_s": round(time.time() - t0, 1), "cells": results}
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUTPUT_DIR, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[Done] ablation results -> {path}")


if __name__ == "__main__":
    main()
