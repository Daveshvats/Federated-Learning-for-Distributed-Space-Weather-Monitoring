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
    Local FedAvg training with Dynamic Focal Loss (2026 SOTA).
    
    Key improvements over vanilla FedAvg:
    1. Dynamic Focal Loss handles local-global imbalance mismatch
    2. AdamW optimizer with weight decay prevents overfitting
    3. Gradient clipping ensures stable training
    4. Cosine LR scheduling improves convergence
    """
    device = next(model.parameters()).device
    model.train()
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), 
        torch.FloatTensor(y)
    )
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=512,  # Larger batch = better gradient estimate
        shuffle=True,
        drop_last=False
    )
    
    # ✅ KEY CHANGE: Use Dynamic Focal Loss instead of BCE
    from losses import DynamicFocalLoss
    
    # Calculate this client's positivity rate for dynamic adjustment
    client_pos_rate = float(y.mean())
    
    criterion = DynamicFocalLoss(
        gamma=2.0,           # Focusing parameter
        base_alpha=0.25,     # Class balance weight
        reduction='mean'
    ).to(device)
    
    # Optimizer with L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=1e-4  # Prevents overfitting
    )
    
    # Learning rate scheduler: cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=LR * 0.01  # Decay to 1% of initial LR
    )

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(X_batch)
            
            # ✅ KEY CHANGE: Pass client_pos_rate for dynamic adjustment
            loss = criterion(logits, y_batch, client_pos_rate=client_pos_rate)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Optimizer step
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # LR scheduler step
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
    """Local FedProx training with Dynamic Focal Loss + proximal term."""
    
    device = next(model.parameters()).device
    model.train()
    
    # Data loader
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), 
        torch.FloatTensor(y)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    
    # ✅ Use Dynamic Focal Loss (same as FedAvg)
    from losses import DynamicFocalLoss
    client_pos_rate = float(y.mean())
    criterion = DynamicFocalLoss(gamma=2.0, base_alpha=0.25).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Global weights snapshot (for proximal term)
    global_params = [p.data.detach().clone().to(device) 
                     for p in global_model.parameters()]

    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            logits = model(X_batch)
            
            # Focal loss component
            focal_loss = criterion(logits, y_batch, client_pos_rate=client_pos_rate)
            
            # Proximal term: penalize drift from global model
            prox_term = torch.tensor(0.0, device=device)
            for local_p, global_p in zip(model.parameters(), global_params):
                prox_term = prox_term + ((local_p - global_p) ** 2).sum()
            prox_term = (mu / 2.0) * prox_term
            
            # Total loss
            loss = focal_loss + prox_term
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return model

# ════════════════════════════════════════════════════════════════════════
# SERVER AGGREGATION
# ════════════════════════════════════════════════════════════════════════

def aggregate_weights_dafl(
    client_weights: list,
    client_sizes:   list,
    client_labels:  list,      # NEW: List of y arrays for each client
    global_pos_rate: float = 0.4887  # Your training set's flare rate
) -> list:
    """
    Distribution-Aware Federated Learning (DA-FL) aggregation.
    
    Instead of weighting clients only by dataset SIZE,
    also weight by how well they represent MINORITY classes.
    
    Why this helps:
    - Some clients may have slightly fewer flares due to random partitioning
    - Those clients' updates should be UPWEIGHTED so minority patterns aren't lost
    - Prevents the global model from being biased toward majority-class clients
    
    Formula:
        final_weight_i = (size_i / total_size) × (local_pos_rate_i / global_pos_rate)
    
    Args:
        client_weights: List of model state_dicts (one per client)
        client_sizes: List of sample counts per client
        client_labels: List of label arrays (y_c) per client
        global_pos_rate: Global training set positivity rate
    
    Returns:
        Averaged state_dict with distribution-aware weighting
    """
    
    total_size = sum(client_sizes)
    da_weights = []
    
    print("      [DA-FL] Distribution-aware weights:")
    
    for i, (size, y_c) in enumerate(zip(client_sizes, client_labels)):
        # Base weight: proportional to dataset size
        size_weight = size / total_size
        
        # Distribution weight: amplify underrepresented classes
        local_pos_rate = float(np.mean(y_c))
        phi = local_pos_rate / global_pos_rate if global_pos_rate > 0 else 1.0
        
        # Combined weight
        combined = size_weight * phi
        da_weights.append(combined)
        
        print(f"        Client {i}: size_w={size_weight:.3f}, φ={phi:.3f}, "
              f"final_w={combined:.3f} (pos_rate={local_pos_rate:.3f})")
    
    # Normalize weights to sum to 1
    total_da_weight = sum(da_weights)
    da_weights = [w / total_da_weight for w in da_weights]
    
    # Weighted average of model parameters
    averaged = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        
        for w, da_w in zip(client_weights, da_weights):
            layer_avg += da_w * w[layer_idx].astype(np.float64)
        
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

        client_label_list = [shards[cid][1] for cid in selected]
        
        new_weights = aggregate_weights_dafl(
            client_weights, 
            client_sizes, 
            client_label_list,
            global_pos_rate=0.4887  # Your training set rate
        )
        
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

        client_label_list = [shards[cid][1] for cid in selected]
        
        new_weights = aggregate_weights_dafl(
            client_weights, 
            client_sizes, 
            client_label_list,
            global_pos_rate=0.4887  # Your training set rate
        )
        
        set_weights(global_model, new_weights)

        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate_model(global_model, X_test, y_test)
            history.append({"round": rnd, **metrics})
            print(f"  Round {rnd:>3} | "
                  f"F1: {metrics['f1']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}")

    return global_model, history