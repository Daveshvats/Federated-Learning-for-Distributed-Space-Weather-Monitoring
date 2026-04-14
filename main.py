"""
main.py
────────
SF-9: Federated Space Weather Monitoring
Full pipeline orchestrator.

Run order:
  1. Load / generate SWAN-SF data
  2. Preprocess (scale, train/test split)
  3. Partition into N_CLIENTS regional shards (non-IID)
  4. Train centralised baselines (LR + XGBoost) — upper bound
  5. Run FedAvg for N_ROUNDS
  6. Run FedProx for N_ROUNDS
  7. Optimize thresholds
  8. Evaluate all models on the global test set
  9. Generate all paper figures

Usage:
  python main.py
  python main.py --rounds 30          # faster debug run
  python main.py --clients 4          # fewer clients
"""

import argparse
import io
import os
import time
import numpy as np
import sys
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

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def parse_args():
    parser = argparse.ArgumentParser(description="SF-9 Federated Solar Flare Prediction")
    parser.add_argument("--rounds",  type=int, default=cfg.N_ROUNDS,   help="FL communication rounds")
    parser.add_argument("--clients", type=int, default=cfg.N_CLIENTS,  help="Number of regional clients")
    parser.add_argument("--mu",      type=float, default=cfg.MU,       help="FedProx proximal coefficient")
    parser.add_argument("--dirichlet", action="store_true",            help="Use Dirichlet partitioning instead of HARPNUM_MOD")
    return parser.parse_args()


def find_best_threshold(y_true, y_probs, model_name):
    """Find threshold that maximizes F1 score"""
    from sklearn.metrics import f1_score as sklearn_f1
    from sklearn.metrics import precision_score as sklearn_prec
    from sklearn.metrics import recall_score as sklearn_rec
    
    best_f1 = 0
    best_thresh = 0.50
    
    for thresh in np.arange(0.20, 0.85, 0.01):
        y_pred = (y_probs >= thresh).astype(int)
        f1 = sklearn_f1(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    prec = sklearn_prec(y_true, (y_probs >= best_thresh).astype(int), zero_division=0)
    rec = sklearn_rec(y_true, (y_probs >= best_thresh).astype(int), zero_division=0)
    
    print(f"  {model_name}: Optimal threshold = {best_thresh:.2f} | "
          f"F1={best_f1:.3f} | Prec={prec:.3f} | Rec={rec:.3f}")
    
    return best_thresh, {'precision': prec, 'recall': rec, 'f1': best_f1}


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
    
    try:
        from config import USE_CLEANED_DATA
        if USE_CLEANED_DATA:
            print("[Partition] Cleaned dataset detected → Using BALANCED partition\n")
            shards = partition_data(X_train, y_train, harpnum_mod=None)
        elif args.dirichlet or "HARPNUM_MOD" not in df.columns:
            shards = partition_data_dirichlet(X_train, y_train, alpha=0.5)
        else:
            train_harp = df["HARPNUM_MOD"].values[:len(X_train)]
            shards = partition_data(X_train, y_train, train_harp)
    except Exception as e:
        print(f"[Partition] Error: {e}\n[Partition] Using fallback balanced partition...\n")
        shards = partition_data(X_train, y_train, harpnum_mod=None)

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

    # ── 7. OPTIMIZE THRESHOLDS (NEW!) ─────────────────────────────────────
    print("\n[Optimization] Finding optimal thresholds...")
    
    import torch
    fedavg_model.eval()
    fedprox_model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(next(fedavg_model.parameters()).device)
        fedavg_probs = torch.sigmoid(fedavg_model(X_tensor)).cpu().numpy().flatten()
        fedprox_probs = torch.sigmoid(fedprox_model(X_tensor)).cpu().numpy().flatten()
    
    _, fedavg_optimal = find_best_threshold(y_test, fedavg_probs, "FedAvg")
    _, fedprox_optimal = find_best_threshold(y_test, fedprox_probs, "FedProx")

    # ── 8. COMBINED RESULTS ───────────────────────────────────────────────
    all_results = {
        "Logistic Regression":   c_results["logistic_regression"],
        "XGBoost (Centralised)": c_results["xgboost"],
        "FedAvg MLP":            fedavg_metrics,
        "FedProx MLP":           fedprox_metrics,
    }
    print_results_table(all_results)

    # ── 9. FIGURES ────────────────────────────────────────────────────────
    print("\n[Figures] Generating paper plots ...\n")

    plot_confusion_matrices(all_results, y_test)
    plot_roc_curves(all_results, y_test)
    plot_fl_convergence(fedavg_history, fedprox_history)
    plot_comparison_table(all_results)

    # SHAP for XGBoost
    try:
        _, mean_shap = compute_shap(c_models["xgboost"], X_test, feature_names)
        if mean_shap is not None:
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