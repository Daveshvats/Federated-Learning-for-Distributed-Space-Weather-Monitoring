"""
federated_learning.py
──────────────────────
Implements both FedAvg and FedProx from scratch.

FedAvg  (McMahan et al., 2017)
  Server sends global weights → clients train locally → server averages.

FedProx (Li et al., 2020)
  Same as FedAvg but each client adds a PROXIMAL TERM to its local loss:
      L_prox = L_CE + (μ/2) * ||w_local - w_global||²
  This penalises local drift and handles non-IID data heterogeneity,
  which is the exact scenario in distributed space weather monitoring.

WHY THIS MATTERS FOR THE PAPER:
  Each national observatory sees a different subset of active regions
  (non-IID). FedProx is designed for exactly this heterogeneous setting.
  Comparing FedAvg vs FedProx quantifies the cost of data heterogeneity
  and the benefit of the proximal correction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from config import (N_CLIENTS, N_ROUNDS, LOCAL_EPOCHS,
                    FRACTION_FIT, MU, LR, CLIENT_NAMES, THRESHOLD)
from model import SolarMLP, make_loader, get_weights, set_weights, clone_model, make_fresh_model


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL CLIENT TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def local_train_fedavg(
    model: SolarMLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = LOCAL_EPOCHS
) -> SolarMLP:
    """Standard local SGD training (no proximal term)."""
    model.train()
    loader = make_loader(X, y)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(epochs):
        for Xb, yb in loader:
            optimiser.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()

    return model


def local_train_fedprox(
    model: SolarMLP,
    global_model: SolarMLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = LOCAL_EPOCHS,
    mu: float = MU
) -> SolarMLP:
    """
    Local training with FedProx proximal term.
    L_total = BCE(pred, y) + (μ/2) * ||w_local - w_global||²
    """
    model.train()
    loader = make_loader(X, y)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    # Freeze a snapshot of global weights (do not update)
    global_params = [p.data.detach().clone() for p in global_model.parameters()]

    for _ in range(epochs):
        for Xb, yb in loader:
            optimiser.zero_grad()
            pred = model(Xb)
            bce  = criterion(pred, yb)

            # Proximal term: penalise drift from global model
            prox = torch.tensor(0.0)
            for local_p, global_p in zip(model.parameters(), global_params):
                prox = prox + ((local_p - global_p) ** 2).sum()
            prox = (mu / 2.0) * prox

            loss = bce + prox
            loss.backward()
            optimiser.step()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SERVER AGGREGATION (FEDAVG RULE)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_weights(
    client_weights: List[list],
    client_sizes:   List[int]
) -> list:
    """
    Weighted average of client model weights (proportional to dataset size).
    This is the FedAvg aggregation rule — the only difference between
    FedAvg and FedProx is in the local training step, not here.
    """
    total = sum(client_sizes)
    averaged = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx])
        for w, n in zip(client_weights, client_sizes):
            layer_avg += (n / total) * w[layer_idx]
        averaged.append(layer_avg)
    return averaged


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: SolarMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = THRESHOLD
) -> Dict[str, float]:
    """Return accuracy, recall, precision, f1, roc_auc for a trained model."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, probs),
        "probs":     probs,
        "preds":     preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def run_fedavg(
    shards:  List[Tuple[np.ndarray, np.ndarray]],
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_rounds: int = N_ROUNDS
) -> Tuple[SolarMLP, List[Dict]]:
    """
    Full FedAvg training loop.
    Returns the final global model and per-round metric history.
    """
    input_dim = shards[0][0].shape[1]
    global_model = make_fresh_model(input_dim)
    history = []

    print("=" * 60)
    print(" FedAvg Training")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        # Select participating clients
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected   = np.random.choice(N_CLIENTS, n_selected, replace=False)

        client_weights = []
        client_sizes   = []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            local = clone_model(global_model)
            local = local_train_fedavg(local, X_c, y_c)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))

        # Aggregate
        new_weights = aggregate_weights(client_weights, client_sizes)
        set_weights(global_model, new_weights)

        # Evaluate every 5 rounds (and last round)
        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test)
            history.append({"round": rnd, **metrics})
            print(f"  Round {rnd:>3} | "
                  f"F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}")

    return global_model, history


def run_fedprox(
    shards:   List[Tuple[np.ndarray, np.ndarray]],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    n_rounds: int = N_ROUNDS,
    mu:       float = MU
) -> Tuple[SolarMLP, List[Dict]]:
    """
    Full FedProx training loop.
    Returns the final global model and per-round metric history.
    """
    input_dim = shards[0][0].shape[1]
    global_model = make_fresh_model(input_dim)
    history = []

    print("=" * 60)
    print(f" FedProx Training (μ = {mu})")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected   = np.random.choice(N_CLIENTS, n_selected, replace=False)

        client_weights = []
        client_sizes   = []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            local = clone_model(global_model)
            local = local_train_fedprox(local, global_model, X_c, y_c, mu=mu)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))

        new_weights = aggregate_weights(client_weights, client_sizes)
        set_weights(global_model, new_weights)

        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test)
            history.append({"round": rnd, **metrics})
            print(f"  Round {rnd:>3} | "
                  f"F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}")

    return global_model, history
