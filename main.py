"""
main.py
────────
SF-9: Federated Learning for Distributed Space Weather Monitoring
Full pipeline orchestrator.

Run order:
  1. Load / generate SWAN-SF data
  2. Preprocess (scale, train/test split)
  3. Partition into N_CLIENTS regional shards (non-IID)
  4. Train centralised baselines (LR + XGBoost) — upper bound
  5. Run FedAvg for N_ROUNDS
  6. Run FedProx for N_ROUNDS
  7. Evaluate all models on the global test set
  8. Generate all paper figures

Usage:
  python main.py
  python main.py --rounds 30          # faster debug run
  python main.py --clients 4          # fewer clients
"""

import argparse
import os
import time
import numpy as np

import config as cfg
from data_preparation     import load_or_generate_data, preprocess
from partition_clients    import partition_data, partition_data_dirichlet
from centralized_baseline import train_centralized, evaluate_centralized, compute_shap
from federated_learning   import run_fedavg, run_fedprox, evaluate_model
from model                import make_fresh_model
from visualize_results    import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_fl_convergence,
    plot_shap_importance,
    plot_comparison_table,
    print_results_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SF-9 Federated Solar Flare Prediction")
    parser.add_argument("--rounds",  type=int, default=cfg.N_ROUNDS,   help="FL communication rounds")
    parser.add_argument("--clients", type=int, default=cfg.N_CLIENTS,  help="Number of regional clients")
    parser.add_argument("--mu",      type=float, default=cfg.MU,       help="FedProx proximal coefficient")
    parser.add_argument("--dirichlet", action="store_true",            help="Use Dirichlet partitioning instead of HARPNUM_MOD")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    t0 = time.time()

    print("\n" + "=" * 60)
    print("  SF-9: Federated Space Weather Monitoring")
    print(f"  Clients: {args.clients} | Rounds: {args.rounds} | μ: {args.mu}")
    print("=" * 60 + "\n")

    # ── 1. DATA ────────────────────────────────────────────────────────────
    df = load_or_generate_data()

    # ── 2. PREPROCESS ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # ── 3. PARTITION ──────────────────────────────────────────────────────
    print("[Partition] Splitting training data into regional client shards ...\n")
    if args.dirichlet or "HARPNUM_MOD" not in df.columns:
        shards = partition_data_dirichlet(X_train, y_train, alpha=0.5)
    else:
        # Map train indices back to HARPNUM_MOD
        train_harp = df["HARPNUM_MOD"].values[:len(X_train)]
        shards = partition_data(X_train, y_train, train_harp)

    # ── 4. CENTRALISED BASELINES ──────────────────────────────────────────
    print("\n[Centralised] Training pooled baselines ...\n")
    c_models  = train_centralized(X_train, y_train)
    c_results = evaluate_centralized(c_models, X_test, y_test)

    # ── 5. FEDAVG ─────────────────────────────────────────────────────────
    print()
    fedavg_model, fedavg_history = run_fedavg(shards, X_test, y_test, n_rounds=args.rounds)
    fedavg_metrics = evaluate_model(fedavg_model, X_test, y_test)

    # ── 6. FEDPROX ────────────────────────────────────────────────────────
    print()
    fedprox_model, fedprox_history = run_fedprox(shards, X_test, y_test,
                                                 n_rounds=args.rounds, mu=args.mu)
    fedprox_metrics = evaluate_model(fedprox_model, X_test, y_test)

    # ── 7. COMBINED RESULTS ───────────────────────────────────────────────
    all_results = {
        "Logistic Regression":   c_results["logistic_regression"],
        "XGBoost (Centralised)": c_results["xgboost"],
        "FedAvg MLP":            fedavg_metrics,
        "FedProx MLP":           fedprox_metrics,
    }
    print_results_table(all_results)

    # ── 8. FIGURES ────────────────────────────────────────────────────────
    print("[Figures] Generating paper plots ...\n")

    plot_confusion_matrices(all_results, y_test)
    plot_roc_curves(all_results, y_test)
    plot_fl_convergence(fedavg_history, fedprox_history)
    plot_comparison_table(all_results)

    # SHAP for XGBoost
    try:
        _, mean_shap = compute_shap(c_models["xgboost"], X_test, feature_names)
        plot_shap_importance(mean_shap, feature_names)
    except Exception as e:
        print(f"[SHAP] Skipped: {e}")

    elapsed = time.time() - t0
    print(f"\n[Done] Total runtime: {elapsed:.1f}s")
    print(f"[Done] All outputs saved to '{cfg.OUTPUT_DIR}/'")
    print("\nKey takeaway for your paper:")
    print(f"  FedProx F1  = {fedprox_metrics['f1']:.3f}  vs  "
          f"XGBoost F1 = {c_results['xgboost']['f1']:.3f}  "
          f"(privacy cost = {c_results['xgboost']['f1'] - fedprox_metrics['f1']:.3f})")
    print(f"  FedProx Recall = {fedprox_metrics['recall']:.3f}  vs  "
          f"FedAvg Recall = {fedavg_metrics['recall']:.3f}  "
          f"(FedProx advantage = {fedprox_metrics['recall'] - fedavg_metrics['recall']:.3f})\n")


if __name__ == "__main__":
    main()
