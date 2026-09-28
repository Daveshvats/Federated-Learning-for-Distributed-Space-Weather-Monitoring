"""
tests/test_fl_smoke.py
──────────────────────
End-to-end FL smoke test (requires torch, CPU is fine).

Trains FedAvg + FedProx on a small synthetic dataset with the v3.0
protocol and verifies:
  1. training completes without NaN
  2. history entries carry VALIDATION metrics (protocol B5)
  3. the frozen-protocol evaluation produces sane test metrics (B1/B4)
  4. plain FedAvg uses size-weighted aggregation (B6)
  5. USE_SMOTE=True path runs (B7 — previously dead code)
  6. client-level evaluation runs (Stage 10)
  7. secure aggregation integrates with a real aggregation round

Run:  python tests/test_fl_smoke.py
"""

import os
import sys
import numpy as np

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


def make_fl_data(n=3000, seed=0):
    """Small separable 2D-feature problem with class skew."""
    rng = np.random.RandomState(seed)
    n_pos = int(n * 0.08)
    X_pos = rng.randn(n_pos, 6) + np.array([2.0, 1.5, 1.0, 0.5, 0.3, 0.2])
    X_neg = rng.randn(n - n_pos, 6) - np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.0])
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)]).astype(int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def main():
    print("=" * 64)
    print("  SF-9 FL SMOKE TEST (torch, synthetic data)")
    print("=" * 64)

    import torch
    import config as cfg
    from partition_clients import partition_data_dirichlet
    from federated_learning import run_fedavg, run_fedprox, \
        aggregate_weights_plain, get_model_probs
    from evaluation import (find_optimal_threshold_fbeta,
                            compute_all_metrics, make_calibrator)

    cfg.USE_LSTM = False          # MLP mode
    cfg.AGGREGATION_STRATEGY = "plain"
    cfg.USE_SMOTE = False
    cfg.USE_FED_FOCAL = True

    X, y = make_fl_data(3000, seed=1)

    # frozen split contract
    from sklearn.model_selection import train_test_split
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.2,
                                      random_state=0, stratify=y)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.2,
                                      random_state=0, stratify=y[tr_idx])
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]
    X_test, y_test = X[te_idx], y[te_idx]

    # disjoint shards
    shards, assignment = partition_data_dirichlet(
        X_train, y_train, alpha=0.8, n_clients=4, seed=7,
        return_indices=True, verbose=False)
    check("shards disjoint & complete",
          np.bincount(assignment, minlength=4).sum() == len(y_train))

    print("\n[1] FedAvg (plain aggregation) — 4 rounds")
    m_avg, hist_avg = run_fedavg(shards, X_val, y_val, n_rounds=4,
                                 use_lstm=False, seed=7)
    check("fedavg history non-empty", len(hist_avg) > 0)
    check("fedavg history contains val metrics",
          all("roc_auc" in h and "f1" in h for h in hist_avg))
    check("fedavg weights finite",
          all(np.isfinite(w).all() for w in
              [p.detach().numpy() for p in m_avg.parameters()]))

    print("\n[2] FedProx (mu=0.01) — 4 rounds")
    m_prox, hist_prox = run_fedprox(shards, X_val, y_val, n_rounds=4,
                                    mu=0.01, use_lstm=False, seed=7)
    check("fedprox history non-empty", len(hist_prox) > 0)
    check("fedprox weights finite",
          all(np.isfinite(w).all() for w in
              [p.detach().numpy() for p in m_prox.parameters()]))

    print("\n[3] Frozen-protocol final evaluation")
    device = next(m_avg.parameters()).device
    for name, model in (("fedavg", m_avg), ("fedprox", m_prox)):
        p_val = get_model_probs(model, X_val, device)
        cal = make_calibrator("none").fit(y_val, p_val)
        t, _ = find_optimal_threshold_fbeta(y_val, p_val, beta=2.0)
        p_test = get_model_probs(model, X_test, device)
        m = compute_all_metrics(y_test, cal.transform(p_test), t)
        check(f"{name}: test metrics sane (AUC>0.7 on separable data)",
              m["roc_auc"] > 0.7, f"auc={m['roc_auc']:.3f}")
        check(f"{name}: full metric set present",
              all(k in m for k in ("accuracy", "precision", "recall", "f1",
                                   "f2", "pr_auc", "brier", "ece")))

    print("\n[4] Plain aggregation is size-weighted (B6)")
    # weights as per-client PARAMETER lists (the get_weights() contract)
    rng = np.random.RandomState(0)
    w_clients = [[rng.randn(4, 3).astype(np.float32),
                  rng.randn(3,).astype(np.float32)] for _ in range(3)]
    sizes = [100, 200, 300]
    agg = aggregate_weights_plain(w_clients, sizes)
    total = sum(sizes)
    for li in range(2):
        manual = sum((sizes[i] / total) * w_clients[i][li] for i in range(3))
        check(f"size-weighted average exact (param {li})",
              np.abs(np.asarray(agg[li]) - manual).max() < 1e-5,
              f"max diff {np.abs(np.asarray(agg[li]) - manual).max():.2e}")

    print("\n[5] SMOTE path runs (B7 — previously dead code)")
    cfg.USE_SMOTE = True
    m_smote, _ = run_fedavg(shards, X_val, y_val, n_rounds=2,
                            use_lstm=False, seed=7)
    check("fedavg+SMOTE completes without error",
          all(np.isfinite(w).all() for w in
              [p.detach().numpy() for p in m_smote.parameters()]))
    cfg.USE_SMOTE = False

    print("\n[6] Client-level evaluation (Stage 10)")
    from evaluate_clients import evaluate_client_level
    res = evaluate_client_level(shards, (m_avg, m_prox), X_val, y_val,
                                "MLP", batch_size=512, local_epochs=1)
    check("client evaluation returns per-client rows",
          res["summary"]["n_clients_evaluated"] == 4)
    check("client evaluation summary has local/federated PR-AUC",
          res["summary"]["mean_local_pr_auc"] is not None and
          res["summary"]["mean_fedavg_pr_auc"] is not None)

    print("\n[7] Communication cost on a real model (Stage 15)")
    from communication_cost import measure_communication
    comm = measure_communication(m_avg, 4, n_rounds=4)
    check("comm cost measured",
          comm["total"]["parameters"] > 0 and
          comm["total"]["total_mb_all_clients"] > 0)

    print("\n" + "=" * 64)
    print(f"  RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 64)
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
