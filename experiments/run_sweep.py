"""
experiments/run_sweep.py  (v3.0)
────────────────────────────────
Hyperparameter sweep (revision Stage 7).

Grid from the original paper's future-work section:
    FedProx mu     : 0.001, 0.01, 0.1
    Dirichlet alpha: 0.5, 1.0, 5.0

Every configuration is selected on VALIDATION (PR-AUC); the test set is
evaluated once per configuration ONLY to report the final generalisation
of the validation-chosen winner (documented in the output JSON).

Usage:
    python experiments/run_sweep.py [--rounds 15] [--quick]
"""

import argparse
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


def evaluate_config(shards, X_val, y_val, mu, n_rounds, seed):
    model, hist = run_fedprox(shards, X_val, y_val, n_rounds=n_rounds,
                              mu=mu, use_lstm=False, seed=seed)
    import torch
    device = next(model.parameters()).device
    p_val = get_model_probs(model, X_val, device)
    cal = make_calibrator(cfg.CALIBRATION_METHOD).fit(y_val, p_val)
    if cfg.CALIBRATION_METHOD == "prior_shift":
        cal.set_prevalences(
            train_rate=float(np.concatenate([y for _, y in shards]).mean()),
            test_rate=float(y_val.mean()))
    t, _ = find_optimal_threshold_fbeta(y_val, cal.transform(p_val),
                                        beta=cfg.FBETA_BETA,
                                        grid=cfg.THRESHOLD_GRID)
    p_cal = cal.transform(p_val)
    m = compute_all_metrics(y_val, p_cal, t, beta=cfg.FBETA_BETA)
    m.pop("preds", None)
    return m, model, cal, t, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("  SF-7 HYPERPARAMETER SWEEP (validation-selected)")
    print("=" * 70)

    df = load_or_generate_data()
    splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test
    cfg.USE_LSTM = False

    mus = [0.001, 0.01, 0.1] if not args.quick else [0.01]
    alphas = [0.5, 1.0, 5.0] if not args.quick else [1.0]

    rows = []
    for alpha in alphas:
        shards = partition_data_dirichlet(
            X_train, y_train, alpha=alpha, n_clients=cfg.N_CLIENTS,
            seed=args.seed, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)
        for mu in mus:
            print(f"\n[sweep] alpha={alpha} mu={mu}")
            try:
                m, model, cal, t, hist = evaluate_config(
                    shards, X_val, y_val, mu, args.rounds, args.seed)
                row = {"alpha": alpha, "mu": mu,
                       "val_pr_auc": m["pr_auc"], "val_f1": m["f1"],
                       "val_roc_auc": m["roc_auc"], "threshold": t}
                rows.append(row)
                print(f"[sweep] VAL PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f}")
            except Exception as e:
                rows.append({"alpha": alpha, "mu": mu, "error": str(e)})
                print(f"[sweep] FAILED: {e}")

    # winner by VALIDATION PR-AUC
    valid_rows = [r for r in rows if "val_pr_auc" in r]
    if valid_rows:
        winner = max(valid_rows, key=lambda r: r["val_pr_auc"])
        print(f"\n[sweep] VALIDATION winner: alpha={winner['alpha']} "
              f"mu={winner['mu']} (PR-AUC {winner['val_pr_auc']:.3f})")

        # single test evaluation of the validation-chosen configuration
        shards = partition_data_dirichlet(
            X_train, y_train, alpha=winner["alpha"], n_clients=cfg.N_CLIENTS,
            seed=args.seed, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)
        m, model, cal, t, _ = evaluate_config(
            shards, X_val, y_val, winner["mu"], args.rounds, args.seed)
        import torch
        device = next(model.parameters()).device
        p_test = get_model_probs(model, X_test, device)
        test_res = compute_all_metrics(y_test, cal.transform(p_test), t,
                                       beta=cfg.FBETA_BETA)
        test_res.pop("preds", None)
        winner["test_metrics"] = test_res
        print(f"[sweep] TEST (winner config): PR-AUC="
              f"{test_res['pr_auc']:.3f} F1={test_res['f1']:.3f}")
    else:
        winner = None

    out = {"seed": args.seed, "rounds": args.rounds, "grid": rows,
           "winner": winner, "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(cfg.OUTPUT_DIR, "sweep_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[Done] sweep results -> {path}")


if __name__ == "__main__":
    main()
