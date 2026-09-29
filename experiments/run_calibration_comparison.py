"""
experiments/run_calibration_comparison.py  (review R6)
──────────────────────────────────────────────────────
Prior-shift robustness: a systematic comparison of calibration arms
under the benchmark's 26x training-to-deployment prevalence shift.

Arms (all fit on VALIDATION only, applied once to TEST):
    raw             : identity (uncalibrated model probabilities)
    prior_shift     : logit-space prior correction toward the
                      validation-estimated deployment prevalence
    platt           : logistic scaling on log-odds
    isotonic        : monotone regression
    temperature     : single-parameter logit temperature

For each model x arm we report test Brier, ECE, plus the
validation-frozen FPR operating points (R4 protocol):
    select_fpr_thresholds_on_validation  -> frozen thresholds
    frozen_operating_point_metrics       -> realised test behaviour

Model selection across arms happens on VALIDATION Brier (proper
scoring rule, threshold-independent) — never on test.

Usage (owner, real data):
    python experiments/run_calibration_comparison.py
    python experiments/run_calibration_comparison.py --rounds 20

Smoke mode (no data / no torch required):
    python experiments/run_calibration_comparison.py --synthetic
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from evaluation import (make_calibrator, compute_all_metrics,
                        expected_calibration_error, select_calibration,
                        select_fpr_thresholds_on_validation,
                        frozen_operating_point_metrics)
from sklearn.metrics import brier_score_loss

FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)
ARMS = ("none", "prior_shift", "platt", "isotonic", "temperature")


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def compare_arms(y_val, p_val, y_test, p_test, model_name):
    """
    Fit every calibration arm on validation, evaluate each exactly once
    on test. Returns per-arm metrics + the validation-selected winner.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    rows = {}
    for arm in ARMS:
        cal = make_calibrator(arm).fit(y_val, p_val)
        p_val_c = cal.transform(p_val)
        p_test_c = cal.transform(p_test)
        val_brier = float(brier_score_loss(np.asarray(y_val).astype(int), p_val_c))

        # R4 deployment thresholds selected on VALIDATION (negative
        # quantile at target FPR), then applied once to test
        frozen = select_fpr_thresholds_on_validation(y_val, p_val_c,
                                                     FPR_TARGETS)
        ops = frozen_operating_point_metrics(y_test, p_test_c, frozen)

        rows[arm] = {
            "validation_brier": val_brier,
            "test_brier": float(brier_score_loss(
                np.asarray(y_test).astype(int), p_test_c)),
            "test_ece": expected_calibration_error(y_test, p_test_c),
            "test_roc_auc": float(roc_auc_score(y_test, p_test_c)),
            "test_pr_auc": float(average_precision_score(y_test, p_test_c)),
            "frozen_operating_points": ops,
            "calibrator_class": type(cal).__name__,
        }

    winner = min(rows, key=lambda a: rows[a]["validation_brier"])
    return {
        "model": model_name,
        "arms": rows,
        "validation_selected_arm": winner,
        "selection_rule": "min validation Brier (proper scoring rule)",
    }


def _synthetic_mode():
    """Plumbing check with synthetic validation/test score arrays."""
    rng = np.random.RandomState(7)
    n_val, n_test = 12000, 60000
    y_val = rng.randint(0, 2, n_val)                    # 50% prevalence
    y_test = (rng.rand(n_test) < 0.02).astype(int)      # 2% prevalence
    # badly calibrated scores: positives high, negatives low
    p_val = np.where(y_val == 1, rng.beta(6, 1.4, n_val),
                     rng.beta(2, 5, n_val))
    p_test = np.where(y_test == 1, rng.beta(6, 1.4, n_test),
                      rng.beta(2, 5, n_test))
    report = compare_arms(y_val, p_val, y_test, p_test, "synthetic-model")
    report["mode"] = "synthetic"
    report["note"] = ("plumbing check only — no scientific claim; real "
                      "run requires the Cleaned SWAN-SF artefacts")
    path = os.path.join(cfg.OUTPUT_DIR, "calibration_comparison.json")
    _atomic_json(report, path)
    print(f"[CalibComp] synthetic report -> {path}")
    print(f"[CalibComp] validation-selected arm: {report['validation_selected_arm']}")
    for arm, r in report["arms"].items():
        print(f"  {arm:<12} val Brier {r['validation_brier']:.4f} | "
              f"test Brier {r['test_brier']:.4f} | test ECE {r['test_ece']:.4f}")
    ok = all("test_brier" in r and np.isfinite(r["test_brier"])
             for r in report["arms"].values()) and len(report["arms"]) == 5
    print(f"[Self-check] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rounds", type=int, default=cfg.N_ROUNDS)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.synthetic:
        return _synthetic_mode()

    # ── real-data path (owner; requires dataset + torch) ──────────────
    t0 = time.time()
    from data_preparation import load_or_generate_data, preprocess
    from partition_clients import partition_data_dirichlet
    from federated_learning import run_fedprox, run_fedavg, get_model_probs
    from centralized_baseline import (train_centralized, mlp_probs,
                                      training_budget_report)

    df = load_or_generate_data()
    splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)
    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test
    cfg.USE_LSTM = False

    shards = partition_data_dirichlet(
        X_train, y_train, alpha=cfg.DIRICHLET_ALPHA, n_clients=cfg.N_CLIENTS,
        seed=cfg.SEED, min_samples=cfg.MIN_SAMPLES_PER_CLIENT)

    report = {"mode": "real", "rounds": args.rounds, "seed": cfg.SEED,
              "fpr_targets": list(FPR_TARGETS), "models": {}}

    # federated FedProx arm
    fedprox, _ = run_fedprox(shards, X_val, y_val, rounds=args.rounds,
                             mu=cfg.FEDPROX_MU)
    report["models"]["fedprox_mlp"] = compare_arms(
        y_val, get_model_probs(fedprox, X_val), y_test,
        get_model_probs(fedprox, X_test), "fedprox_mlp")

    # centralized references
    centr = train_centralized(X_train, y_train, X_val, y_val, seed=cfg.SEED)
    if centr.get("mlp") is not None:
        report["models"]["centralized_mlp"] = compare_arms(
            y_val, mlp_probs(centr["mlp"], X_val), y_test,
            mlp_probs(centr["mlp"], X_test), "centralized_mlp")
    if centr.get("xgboost") is not None:
        xgb_p_val = centr["xgboost"].predict_proba(X_val)[:, 1]
        xgb_p_test = centr["xgboost"].predict_proba(X_test)[:, 1]
        report["models"]["xgboost"] = compare_arms(
            y_val, xgb_p_val, y_test, xgb_p_test, "xgboost")

    report["training_budget"] = training_budget_report(
        len(y_train), n_val=len(y_val), fl_rounds=args.rounds,
        fl_local_epochs=cfg.LOCAL_EPOCHS)
    report["elapsed_s"] = time.time() - t0

    path = args.output or os.path.join(cfg.OUTPUT_DIR,
                                       "calibration_comparison.json")
    _atomic_json(report, path)
    print(f"[CalibComp] report -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
