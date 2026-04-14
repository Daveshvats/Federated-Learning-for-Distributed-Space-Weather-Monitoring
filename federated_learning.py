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
from model import SolarMLP, make_loader, get_weights, set_weights, clone_model, make_fresh_model, get_device


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL CLIENT TRAINING — FedAvg (plain BCE, no proximal term)
# ─────────────────────────────────────────────────────────────────────────────

def local_train_fedavg(
    model:  SolarMLP,
    X:      np.ndarray,
    y:      np.ndarray,
    epochs: int = LOCAL_EPOCHS
) -> SolarMLP:
    """Standard local SGD training (no proximal term)."""
    device    = next(model.parameters()).device
    model.train()
    loader    = make_loader(X, y)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(epochs):
        for Xb, yb in loader:
            # Move batch to same device as model
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimiser.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL CLIENT TRAINING — FedProx (BCE + proximal regularisation)
# ─────────────────────────────────────────────────────────────────────────────

def local_train_fedprox(
    model:        SolarMLP,
    global_model: SolarMLP,
    X:            np.ndarray,
    y:            np.ndarray,
    epochs:       int   = LOCAL_EPOCHS,
    mu:           float = MU
) -> SolarMLP:
    """
    Local training with FedProx proximal term.
    L_total = BCE(pred, y) + (μ/2) * ||w_local - w_global||²
    """
    device    = next(model.parameters()).device
    model.train()
    loader    = make_loader(X, y)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    # Snapshot of global weights — frozen, used only for distance calculation
    global_params = [p.data.detach().clone().to(device)
                     for p in global_model.parameters()]

    for _ in range(epochs):
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimiser.zero_grad()
            pred = model(Xb)
            bce  = criterion(pred, yb)

            # Proximal term: penalise local drift from global model
            prox = torch.tensor(0.0, device=device)
            for local_p, global_p in zip(model.parameters(), global_params):
                prox = prox + ((local_p - global_p) ** 2).sum()
            prox = (mu / 2.0) * prox

            loss = bce + prox
            loss.backward()
            optimiser.step()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SERVER AGGREGATION — weighted FedAvg rule
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_weights(
    client_weights: List[list],
    client_sizes:   List[int]
) -> list:
    """
    Weighted average of client model weights proportional to dataset size.
    This aggregation rule is identical for both FedAvg and FedProx —
    the difference between the two algorithms is purely in local training.
    """
    total    = sum(client_sizes)
    averaged = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for w, n in zip(client_weights, client_sizes):
            layer_avg += (n / total) * w[layer_idx].astype(np.float64)
        averaged.append(layer_avg.astype(np.float32))
    return averaged


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:     SolarMLP,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    threshold: float = THRESHOLD
) -> Dict:
    """Evaluate a trained model; returns accuracy, recall, precision, F1, ROC-AUC."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs    = model(X_tensor).cpu().numpy()

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
# FEDAVG TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_fedavg(
    shards:   List[Tuple[np.ndarray, np.ndarray]],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    n_rounds: int = N_ROUNDS
) -> Tuple[SolarMLP, List[Dict]]:
    """
    Complete FedAvg training loop.
    Returns the final global model and per-evaluation-round metric history.
    """
    input_dim    = shards[0][0].shape[1]
    # BUG FIX: make_fresh_model returns (model, device) — unpack both
    global_model, device = make_fresh_model(input_dim)
    history      = []

    print("=" * 60)
    print(f" FedAvg Training  [{device}]")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected   = np.random.choice(len(shards), n_selected, replace=False)

        client_weights = []
        client_sizes   = []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            # Clone global model (stays on same device)
            local = clone_model(global_model)
            local = local_train_fedavg(local, X_c, y_c)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))

            # Free local model from GPU memory after weight extraction
            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue

        # Aggregate and push back to global model
        new_weights = aggregate_weights(client_weights, client_sizes)
        set_weights(global_model, new_weights)

        # Evaluate every 5 rounds and on the final round
        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test)
            history.append({"round": rnd, **metrics})
            print(f"  Round {rnd:>3} | "
                  f"F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}")

    return global_model, history


# ─────────────────────────────────────────────────────────────────────────────
# FEDPROX TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_fedprox(
    shards:   List[Tuple[np.ndarray, np.ndarray]],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    n_rounds: int   = N_ROUNDS,
    mu:       float = MU
) -> Tuple[SolarMLP, List[Dict]]:
    """
    Complete FedProx training loop.
    Returns the final global model and per-evaluation-round metric history.
    """
    input_dim    = shards[0][0].shape[1]
    # BUG FIX: make_fresh_model returns (model, device) — unpack both
    global_model, device = make_fresh_model(input_dim)
    history      = []

    print("=" * 60)
    print(f" FedProx Training (μ = {mu})  [{device}]")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected   = np.random.choice(len(shards), n_selected, replace=False)

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

            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue

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