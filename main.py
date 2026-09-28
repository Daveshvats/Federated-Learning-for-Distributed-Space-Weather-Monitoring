"""
main.py  (v3.0 — improvements branch)
────────────────────────────────────────
SF-9: Federated Space Weather Monitoring — full pipeline orchestrator.

v3.0 PROTOCOL (fixes audit findings B1, B3, B4, B5, B10):

    TRAIN  ──> FL client training / centralized training
    VAL    ──> monitoring, calibration fitting, F-beta threshold search,
                checkpoint selection (everything that "selects")
    TEST   ──> touched EXACTLY ONCE, with the frozen pipeline

    1.  Load 2D data (and 3D data if LSTM mode)
    2.  Immutable train/val/test split (+ runtime leakage audit)
    3.  Partition TRAIN into disjoint Dirichlet client shards
    4.  Centralized baselines: climatology, LR, centralized MLP, XGBoost
    5.  FedAvg + FedProx (+SCAFFOLD if enabled), monitored on VAL
    6.  Calibration selected on VAL (Brier score)
    7.  F-beta threshold searched on VAL — never on test
    8.  ONE final evaluation on TEST at the frozen threshold;
        ALL metrics recomputed there (stale-accuracy bug B4 fixed)
    9.  results.json + run_manifest.json written; figures regenerated
        from machine-readable results (B18)
    10. Client-level evaluation + communication-cost instrumentation

Usage:
  python main.py
  python main.py --rounds 30 --clients 4
  python main.py --no-lstm            # MLP mode
  python main.py --calibration platt  # override calibration method
"""

import argparse
import json
import os
import time
import numpy as np
import sys

import config as cfg
from data_preparation import load_or_generate_data, preprocess, \
    load_and_scale_3d_data
from partition_clients import partition_data_dirichlet
from centralized_baseline import (train_centralized, evaluate_centralized,
                                  model_probs, compute_shap)
from federated_learning import (run_fedavg, run_fedprox, run_scaffold,
                                get_model_probs)
from evaluation import (make_calibrator, select_calibration,
                        find_optimal_threshold_fbeta, compute_all_metrics)
from leakage_audit.audit_leakage import run_audit, _print_report
from visualize_results import (plot_confusion_matrices, plot_roc_curves,
                               plot_fl_convergence, plot_shap_importance,
                               plot_comparison_table, print_results_table)

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                  errors='replace')


def parse_args():
    p = argparse.ArgumentParser(description="SF-9 Federated Solar Flare Prediction")
    p.add_argument("--rounds", type=int, default=cfg.N_ROUNDS)
    p.add_argument("--clients", type=int, default=cfg.N_CLIENTS)
    p.add_argument("--mu", type=float, default=cfg.MU)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--alpha", type=float, default=cfg.DIRICHLET_ALPHA)
    p.add_argument("--calibration", default=cfg.CALIBRATION_METHOD,
                   choices=["none", "prior_shift", "platt", "isotonic",
                            "temperature"])
    p.add_argument("--dirichlet", action="store_true")
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--no-scaffold", action="store_true")
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--eval-batch-size", type=int, default=cfg.EVAL_BATCH_SIZE)
    return p.parse_args()


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[Results] Written -> {path}")


def main():
    args = parse_args()

    if args.no_lstm:
        cfg.USE_LSTM = False
    if args.no_scaffold:
        cfg.USE_SCAFFOLD = False
    if args.no_mixup:
        cfg.USE_MIXUP = False

    run_id = cfg.RUN_ID or time.strftime("run_%Y%m%d_%H%M%S")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    t0 = time.time()

    use_lstm = cfg.USE_LSTM
    model_name = 'LSTM' if use_lstm else 'MLP'

    print("\n" + "=" * 70)
    print(f"  SF-9 v{cfg.VERSION} | run: {run_id}")
    print(f"  Clients: {args.clients} | Rounds: {args.rounds} | "
          f"mu: {args.mu} | seed: {args.seed}")
    print(f"  Model: {model_name} | agg: {cfg.AGGREGATION_STRATEGY} | "
          f"calib: {args.calibration} | SMOTE: {cfg.USE_SMOTE}")
    print("  Protocol: threshold+calibration on VALIDATION; "
          "test touched ONCE")
    print("=" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. LOAD DATA + IMMUTABLE SPLITS
    # ═══════════════════════════════════════════════════════════════════════
    df = load_or_generate_data()
    splits = preprocess(df, val_fraction=cfg.VAL_SPLIT)

    X_train_2d, y_train = splits.X_train, splits.y_train
    X_val_2d, y_val = splits.X_val, splits.y_val
    X_test_2d, y_test = splits.X_test, splits.y_test
    feature_names = splits.features

    # ── runtime leakage audit (Gate 1 pass criterion) ──
    audit = run_audit(train_idx=splits.train_idx, val_idx=splits.val_idx,
                      test_idx=splits.test_idx)
    _print_report(audit)
    if not audit.get("overall_pass"):
        raise RuntimeError("LEAKAGE AUDIT FAILED — refusing to train. "
                           "Fix the split before running experiments.")

    # ── 3D data for LSTM (same contract: val carved from train) ──
    X_train_fl, X_val_fl, X_test_fl = X_train_2d, X_val_2d, X_test_2d
    y_train_fl, y_val_fl, y_test_fl = y_train, y_val, y_test
    if use_lstm:
        print("=" * 70)
        print("  LOADING 3D DATA FOR LSTM MODELS")
        print("=" * 70 + "\n")
        try:
            (X_train_3d, y_train_3d, X_val_3d, y_val_3d,
             X_test_3d, y_test_3d, _) = load_and_scale_3d_data()
            assert len(y_train_3d) == len(y_train) and \
                len(y_val_3d) == len(y_val), \
                "3D/2D sample mismatch — check data sources"
            X_train_fl, X_val_fl, X_test_fl = X_train_3d, X_val_3d, X_test_3d
            y_train_fl, y_val_fl, y_test_fl = y_train_3d, y_val_3d, y_test_3d
            print(f"[Data Path] LSTM mode: 3D {X_train_3d.shape[1:]} for FL | "
                  f"2D ({X_train_2d.shape[1]} feats) for baselines\n")
        except Exception as e:
            print(f"\n[ERROR] 3D data unavailable ({e})\n"
                  f"[FALLBACK] Switching to MLP mode.\n")
            cfg.USE_LSTM = use_lstm = False
            model_name = 'MLP'

    # ═══════════════════════════════════════════════════════════════════════
    # 2. DISJOINT DIRICHLET PARTITION (TRAIN only)
    # ═══════════════════════════════════════════════════════════════════════
    print("[Partition] Disjoint Dirichlet non-IID partitioning "
          f"(alpha={args.alpha}, seed={args.seed}) ...\n")
    shards, assignment = partition_data_dirichlet(
        X_train_fl, y_train_fl, alpha=args.alpha,
        n_clients=args.clients, seed=args.seed,
        min_samples=cfg.MIN_SAMPLES_PER_CLIENT, return_indices=True)

    # partition audit: verify disjointness against TRAIN indices
    part_audit = run_audit(
        shards=[np.where(assignment == k)[0] for k in range(args.clients)],
        n_train=len(y_train_fl))
    if not part_audit.get("shard_disjointness", {}).get("pass") or \
       not part_audit.get("shard_coverage", {}).get("pass"):
        raise RuntimeError("PARTITION AUDIT FAILED (overlap/coverage).")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. CENTRALIZED BASELINES (pooled 2D data)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Centralised] Training pooled baselines ...")
    c_models = train_centralized(X_train_2d, y_train,
                                 X_val_2d, y_val, seed=args.seed)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. FEDERATED TRAINING (monitoring on VALIDATION)
    # ═══════════════════════════════════════════════════════════════════════
    print()
    fedavg_model, fedavg_history = run_fedavg(
        shards, X_val_fl, y_val_fl, n_rounds=args.rounds,
        use_lstm=use_lstm, eval_batch_size=args.eval_batch_size,
        seed=args.seed)

    print()
    fedprox_model, fedprox_history = run_fedprox(
        shards, X_val_fl, y_val_fl, n_rounds=args.rounds, mu=args.mu,
        use_lstm=use_lstm, eval_batch_size=args.eval_batch_size,
        seed=args.seed)

    scaffold_model, scaffold_history = None, []
    if cfg.USE_SCAFFOLD:
        print()
        scaffold_model, scaffold_history = run_scaffold(
            shards, X_val_fl, y_val_fl, n_rounds=args.rounds,
            use_lstm=use_lstm, eval_batch_size=args.eval_batch_size,
            warm_start_model=fedavg_model, seed=args.seed)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. CALIBRATION + THRESHOLD — selected on VALIDATION ONLY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EVALUATION PROTOCOL (validation -> freeze -> test once)")
    print("=" * 70)

    import torch
    device = next(fedavg_model.parameters()).device

    def fl_probs(model, X):
        return get_model_probs(model, X, device,
                               batch_size=args.eval_batch_size)

    # gather VAL probabilities for every model
    val_probs = {
        "climatology": model_probs(c_models["climatology"], X_val_2d),
        "logistic_regression": model_probs(c_models["logistic_regression"], X_val_2d),
        "xgboost": model_probs(c_models["xgboost"], X_val_2d),
        f"fedavg_{model_name.lower()}": fl_probs(fedavg_model, X_val_fl),
        f"fedprox_{model_name.lower()}": fl_probs(fedprox_model, X_val_fl),
    }
    if "centralized_mlp" in c_models:
        val_probs["centralized_mlp"] = model_probs(
            c_models["centralized_mlp"], X_val_2d)
    if scaffold_model is not None:
        val_probs[f"scaffold_{model_name.lower()}"] = \
            fl_probs(scaffold_model, X_val_fl)

    # ── 5a. calibration selection on VAL (Brier) ──
    method = args.calibration
    if method == "auto":
        method, cal_table = select_calibration(y_val, val_probs["xgboost"])
        print(f"[Calibration] auto-selected: {method} (val Brier per method: "
              f"{cal_table})")

    calibrators = {}
    for name, p in val_probs.items():
        cal = make_calibrator(method).fit(y_val, p)
        if method == "prior_shift":
            cal.set_prevalences(train_rate=float(y_train.mean()),
                                test_rate=float(y_val.mean()))
        calibrators[name] = cal

    # ── 5b. threshold search on (calibrated) VAL probabilities ──
    thresholds = {}
    for name, p in val_probs.items():
        p_cal = calibrators[name].transform(p)
        t, fb = find_optimal_threshold_fbeta(
            y_val, p_cal, beta=cfg.FBETA_BETA, grid=cfg.THRESHOLD_GRID)
        thresholds[name] = t
        print(f"  [VAL] {name:<28} threshold={t:.3f}  F{cfg.FBETA_BETA:.0f}={fb:.3f}")

    print(f"\n  Frozen: calibration={method} | thresholds={thresholds}")
    print("  -> applying ONCE to the held-out TEST set ...\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 6. FINAL TEST EVALUATION (single pass, frozen pipeline)
    # ═══════════════════════════════════════════════════════════════════════
    test_probs = {
        "climatology": model_probs(c_models["climatology"], X_test_2d),
        "logistic_regression": model_probs(c_models["logistic_regression"], X_test_2d),
        "xgboost": model_probs(c_models["xgboost"], X_test_2d),
        f"fedavg_{model_name.lower()}": fl_probs(fedavg_model, X_test_fl),
        f"fedprox_{model_name.lower()}": fl_probs(fedprox_model, X_test_fl),
    }
    if "centralized_mlp" in c_models:
        test_probs["centralized_mlp"] = model_probs(
            c_models["centralized_mlp"], X_test_2d)
    if scaffold_model is not None:
        test_probs[f"scaffold_{model_name.lower()}"] = \
            fl_probs(scaffold_model, X_test_fl)

    all_results = {}
    for name, p in test_probs.items():
        p_cal = calibrators[name].transform(p)
        res = compute_all_metrics(y_test_fl if "fed" in name or "scaffold" in name
                                  else y_test, p_cal, thresholds[name],
                                  beta=cfg.FBETA_BETA)
        res["calibration"] = method
        res["probs"] = p_cal
        res["preds"] = res.pop("preds", None)
        all_results[name] = res

    # display names
    display = {
        "climatology": "Climatology",
        "logistic_regression": "Logistic Regression",
        "centralized_mlp": "Centralized MLP",
        "xgboost": "XGBoost (Centralized ref.)",
        f"fedavg_{model_name.lower()}": f"FedAvg {model_name}",
        f"fedprox_{model_name.lower()}": f"FedProx {model_name}",
        f"scaffold_{model_name.lower()}": f"SCAFFOLD {model_name}",
    }
    pretty_results = {display.get(k, k): v for k, v in all_results.items()}
    print_results_table(pretty_results)

    # ═══════════════════════════════════════════════════════════════════════
    # 7. CLIENT-LEVEL EVALUATION (Stage 10)
    # ═══════════════════════════════════════════════════════════════════════
    try:
        from evaluate_clients import evaluate_client_level
        client_results = evaluate_client_level(
            shards, (fedavg_model, fedprox_model),
            X_val_fl, y_val_fl, model_name,
            batch_size=args.eval_batch_size)
    except Exception as e:
        print(f"[Client Eval] Skipped: {e}")
        client_results = None

    # ═══════════════════════════════════════════════════════════════════════
    # 8. COMMUNICATION COST (Stage 15)
    # ═══════════════════════════════════════════════════════════════════════
    try:
        from communication_cost import measure_communication
        comm = measure_communication(fedavg_model, args.clients,
                                     n_rounds=args.rounds)
    except Exception as e:
        print(f"[Comm Cost] Skipped: {e}")
        comm = None

    # ═══════════════════════════════════════════════════════════════════════
    # 9. FIGURES (from machine-readable results)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Figures] Generating plots ...")
    plot_confusion_matrices(pretty_results, y_test_fl)
    plot_roc_curves(pretty_results, y_test_fl)
    plot_fl_convergence(fedavg_history, fedprox_history, scaffold_history)
    plot_comparison_table(pretty_results)

    try:
        _, mean_shap = compute_shap(c_models["xgboost"], X_test_2d,
                                    feature_names)
        if mean_shap is not None:
            plot_shap_importance(mean_shap, feature_names)
    except Exception as e:
        print(f"[SHAP] Skipped: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # 10. MACHINE-READABLE RESULTS (B18)
    # ═══════════════════════════════════════════════════════════════════════
    serialisable = {}
    for name, res in all_results.items():
        serialisable[name] = {k: v for k, v in res.items()
                              if k not in ("probs", "preds")}

    results = {
        "run_id": run_id,
        "version": cfg.VERSION,
        "protocol": {
            "split": {"train": len(y_train), "val": len(y_val),
                      "test": len(y_test)},
            "calibration": method,
            "thresholds": thresholds,
            "threshold_selected_on": "validation",
            "test_touched_once": True,
        },
        "test_metrics": serialisable,
        "fl_convergence_val": {
            "fedavg": fedavg_history,
            "fedprox": fedprox_history,
            "scaffold": scaffold_history,
        },
        "client_level": client_results,
        "communication": comm,
        "leakage_audit": {k: v for k, v in audit.items()
                          if isinstance(v, (str, int, float, bool))},
    }
    _save_json(results, cfg.RESULTS_JSON)

    manifest = {
        "run_id": run_id, "version": cfg.VERSION,
        "seed": args.seed, "clients": args.clients, "rounds": args.rounds,
        "mu": args.mu, "alpha": args.alpha, "model": model_name,
        "aggregation": cfg.AGGREGATION_STRATEGY,
        "calibration": method, "use_smote": cfg.USE_SMOTE,
        "use_fed_focal": cfg.USE_FED_FOCAL,
        "dirichlet_alpha": args.alpha,
        "feature_names": feature_names,
        "python": sys.version.split()[0],
        "elapsed_s": round(time.time() - t0, 1),
    }
    _save_json(manifest, cfg.RUN_MANIFEST)

    # ═══════════════════════════════════════════════════════════════════════
    # 11. SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n[Done] {run_id} finished in {elapsed:.1f}s")
    print(f"[Done] results -> {cfg.RESULTS_JSON}")
    if "centralized_mlp" in all_results and f"fedprox_{model_name.lower()}" in all_results:
        cost = all_results["centralized_mlp"]["pr_auc"] - \
            all_results[f"fedprox_{model_name.lower()}"]["pr_auc"]
        print(f"\n  Federation cost (PR-AUC): centralized MLP "
              f"{all_results['centralized_mlp']['pr_auc']:.3f} vs FedProx "
              f"{all_results[f'fedprox_{model_name.lower()}']['pr_auc']:.3f} "
              f"-> delta {cost:+.3f}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
