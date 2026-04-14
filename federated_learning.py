"""
federated_learning.py
───────────────────
Implements FedAvg and FedProx from scratch with improvements.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from config import (N_CLIENTS, N_ROUNDS, LOCAL_EPOCHS,
                    FRACTION_FIT, MU, LR, CLIENT_NAMES, THRESHOLD)
from model import SolarMLP, make_loader, get_weights, set_weights, clone_model, make_fresh_model, get_device


# ════════════════════════════════════════════════════════════════════════
# LOCAL TRAINING — FedAvg (IMPROVED WITH CLASS WEIGHTS + GRADIENT CLIPPING)
# ════════════════════════════════════════════════════════════════════════

def local_train_fedavg(
    model:  SolarMLP,
    X:      np.ndarray,
    y:      np.ndarray,
    epochs: int = LOCAL_EPOCHS
) -> SolarMLP:
    """
    Improved local training with:
    - Class-weighted loss for imbalance
    - Gradient clipping for stability
    - AdamW optimizer with weight decay
    - Learning rate scheduling
    """
    device = next(model.parameters()).device
    model.train()
    loader = make_loader(X, y)
    
    # ✅ FIX #1: Calculate class weights for imbalanced data
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    pos_weight_val = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    
    # ✅ FIX #2: Use weighted BCE loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # ✅ FIX #3: AdamW with L2 regularization
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # ✅ FIX #4: Cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, 
        T_max=epochs,
        eta_min=LR * 0.01
    )

    for epoch in range(epochs):
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            
            optimiser.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            
            # ✅ FIX #5: Gradient clipping (prevents explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimiser.step()
        
        scheduler.step()

    return model


# ════════════════════════════════════════════════════════════════════════
# LOCAL TRAINING — FedProx (IMPROVED WITH SAME FIXES + PROXIMAL TERM)
# ════════════════════════════════════════════════════════════════════════

def local_train_fedprox(
    model:        SolarMLP,
    global_model: SolarMLP,
    X:            np.ndarray,
    y:            np.ndarray,
    epochs:       int   = LOCAL_EPOCHS,
    mu:           float = MU
) -> SolarMLP:
    """
    Local training with FedProx proximal term + all improvements.
    L_total = BCE(pred, y) + (μ/2) * ||w_local - w_global||²
    """
    device = next(model.parameters()).device
    model.train()
    loader = make_loader(X, y)
    
    # ✅ FIX #1: Class weights
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    
    # ✅ FIX #2: Weighted loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # ✅ FIX #3: Better optimizer
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Global weights snapshot (for proximal term)
    global_params = [p.data.detach().clone().to(device)
                     for p in global_model.parameters()]

    for epoch in range(epochs):
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            
            optimiser.zero_grad()
            pred = model(Xb)
            bce = criterion(pred, yb)

            # Proximal term: penalize drift from global model
            prox = torch.tensor(0.0, device=device)
            for local_p, global_p in zip(model.parameters(), global_params):
                prox = prox + ((local_p - global_p) ** 2).sum()
            prox = (mu / 2.0) * prox

            loss = bce + prox
            loss.backward()
            
            # ✅ FIX #4: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimiser.step()

    return model


# ════════════════════════════════════════════════════════════════════════
# SERVER AGGREGATION
# ════════════════════════════════════════════════════════════════════════

def aggregate_weights(
    client_weights: List[list],
    client_sizes:   List[int]
) -> list:
    """
    Weighted average of client model weights proportional to dataset size.
    """
    total = sum(client_sizes)
    averaged = []
    
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for w, n in zip(client_weights, client_sizes):
            layer_avg += (n / total) * w[layer_idx].astype(np.float64)
        averaged.append(layer_avg.astype(np.float32))
    
    return averaged


# ════════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model:     SolarMLP,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    threshold: float = THRESHOLD
) -> Dict:
    """Evaluate trained model; returns metrics dict."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Ensure probs is 1D
    probs = probs.flatten()
    
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


# ════════════════════════════════════════════════════════════════════════
# FEDAVG TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════

def run_fedavg(
    shards:   List[Tuple[np.ndarray, np.ndarray]],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    n_rounds: int = N_ROUNDS
) -> Tuple[SolarMLP, List[Dict]]:
    """
    Complete FedAvg training loop.
    Returns final global model and per-round metric history.
    """
    input_dim = shards[0][0].shape[1]
    global_model, device = make_fresh_model(input_dim)
    history = []

    print("=" * 60)
    print(f" FedAvg Training  [{device}]")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected = np.random.choice(len(shards), n_selected, replace=False)

        client_weights = []
        client_sizes = []

        for cid in selected:
            X_c, y_c = shards[cid]
            if len(X_c) == 0:
                continue
            
            local = clone_model(global_model)
            local = local_train_fedavg(local, X_c, y_c)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_c))

            del local
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not client_weights:
            continue

        new_weights = aggregate_weights(client_weights, client_sizes)
        set_weights(global_model, new_weights)

        # Evaluate every 5 rounds and on final round
        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test)
            history.append({"round": rnd, **metrics})
            print(f"  Round {rnd:>3} | "
                  f"F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}")

    return global_model, history


# ════════════════════════════════════════════════════════════════════════
# FEDPROX TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════

def run_fedprox(
    shards:   List[Tuple[np.ndarray, np.ndarray]],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    n_rounds: int = N_ROUNDS,
    mu:       float = MU
) -> Tuple[SolarMLP, List[Dict]]:
    """
    Complete FedProx training loop.
    Returns final global model and per-round metric history.
    """
    input_dim = shards[0][0].shape[1]
    global_model, device = make_fresh_model(input_dim)
    history = []

    print("=" * 60)
    print(f" FedProx Training (μ = {mu})  [{device}]")
    print("=" * 60)

    for rnd in range(1, n_rounds + 1):
        n_selected = max(1, int(FRACTION_FIT * N_CLIENTS))
        selected = np.random.choice(len(shards), n_selected, replace=False)

        client_weights = []
        client_sizes = []

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